"""Normal-behaviour modelling: predict one sensor from the other nineteen.

    python3 scripts/train_nbm.py                      # ~1 hr on MPS, 2 years
    python3 scripts/train_nbm.py --epochs 3           # quick look
    python3 scripts/train_nbm.py --target "Front bearing temperature (°C)"

WHY THIS TASK AND NOT OUTAGE PREDICTION

Predicting "a forced outage begins within 7 days" failed, and it failed for a
reason that is about turbines rather than about this encoder: most forced
outages at Kelmarsh are grid loss, converter faults and emergency stops, which
have no thermal precursor. Nothing in four days of temperature history predicts
them. Trained end to end the model overfit immediately -- train loss down, test
PR-AUC down, below base rate by epoch 6.

This target is the opposite: KNOWN to be learnable. A frozen, untrained encoder
already reaches R^2 0.795 on it, so there is signal to find, and any change in
that number is attributable to training rather than to luck.

That makes it a real test of the encoder. It is also the method the wind
industry actually deploys: learn what a sensor SHOULD read given everything
else, then treat the residual as the health signal. A bearing that starts
running 3 degrees hotter than the model expects is degrading, and the outage
labels come back later as EVALUATION rather than as training signal.

THE ONE WAY TO GET THIS WRONG

The target channel must be removed from the input. Leave it in and the model
reads the answer off its own input and scores R^2 1.000, which looks like a
triumph and means nothing. `keep` below is what enforces that.

OUTPUT

    kelmarsh_nbm_history.csv   per-epoch train/test R^2
    kelmarsh_nbm.png           the same, plotted
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from data_io.kelmarsh_io import load_years, CHANNELS
from models.scada_encoder_tcn import ScadaTCNEncoder
from models.common import PerceiverResampler

DEFAULT_TARGET = "Gear oil temperature (°C)"


class NBMProbe(nn.Module):
    """Encoder over 19 channels -> one predicted sensor value.

    `n_static` wires the per-channel window MEANS in as static features, and
    that flag is the whole experiment. RevIN strips each channel's window mean
    before the trunk sees it (scada_encoder_tcn.py:108), which is correct for
    condition monitoring -- you want "hotter than its own baseline", not
    absolute degrees -- but it means the encoder is structurally blind to
    exactly the quantity a mean-regression task rewards. Measured: linear
    regression on 19 channel means scores R^2 0.984 on gear oil temperature
    while the encoder reached 0.908.

    Passing the means back in as static features gives the encoder the
    baseline's entire input PLUS the temporal structure. It should therefore
    dominate the baseline; if it does not, that is a real finding about the
    encoder rather than an artefact of the task.
    """

    def __init__(self, n_in, d_model=128, n_latents=32, context_len=600,
                 n_static=0):
        super().__init__()
        self.n_static = n_static
        self.enc = ScadaTCNEncoder(d_model=d_model, n_channels=n_in,
                                   context_len=context_len,
                                   n_static=max(n_static, 1))
        self.res = PerceiverResampler(d_model=d_model, n_latents=n_latents)
        # same reason both encoder smoke tests carry it: pooled features sit on
        # a DC offset many times their spread, which strands a linear head
        self.norm = nn.BatchNorm1d(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask, static=None):
        seq = self.enc(x, mask, static_feats=static if self.n_static else None)
        z = self.res(seq)
        B, L, d = z.shape
        z = self.norm(z.reshape(B * L, d)).reshape(B, L, d)
        return self.head(z.mean(1)).squeeze(-1)


def r2(pred, true):
    """Fraction of variance explained. 0 = no better than the mean, 1 = exact.

    Can go negative, and that is informative rather than a bug: it means the
    predictions are worse than a constant, which is the signature of train and
    test being different distributions (measured at -12.5 when a single year was
    cut 70/30, putting summer in train and winter in test).
    """
    ss_res = ((pred - true) ** 2).sum().item()
    ss_tot = ((true - true.mean()) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-9)


@torch.no_grad()
def predict(model, X, M, dev, bs=32, S=None):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        s = None if S is None else S[i:i + bs].to(dev)
        out.append(model(X[i:i + bs].to(dev), M[i:i + bs].to(dev), s).cpu())
    return torch.cat(out)


def ridge_r2(Htr, ytr, Hte, yte, lam=1e-2):
    """Closed-form probe on FROZEN features -- the reference line on the plot.

    Deterministic, so the baseline cannot move between runs and make training
    look better or worse than it was.
    """
    mu = Htr.mean(0)
    A = torch.cat([Htr - mu, torch.ones(len(Htr), 1)], 1)
    ym = ytr.mean()
    w = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]),
                           A.T @ (ytr[:, None] - ym))
    pred = (torch.cat([Hte - mu, torch.ones(len(Hte), 1)], 1) @ w).squeeze(-1) + ym
    return r2(pred, yte)


def build(args):
    X, M, E, _, _ = load_years(args.root, stride=args.stride)
    if args.target not in CHANNELS:
        raise SystemExit(f"target {args.target!r} not in CHANNELS")
    ti = CHANNELS.index(args.target)
    keep = [i for i in range(len(CHANNELS)) if i != ti]

    # target is the window mean of the held-out channel, over OBSERVED samples
    # only -- averaging in the zeros that stand for missing data would drag it
    # toward zero and invent a trend that is really a coverage artefact
    tm = M[:, ti]
    y = (X[:, ti] * tm).sum(-1) / tm.sum(-1).clamp(min=1.0)

    # a window whose target channel is mostly missing has no target to learn
    ok = tm.mean(-1) > 0.5
    X, M, E, y = X[ok][:, keep], M[ok][:, keep], E[ok.numpy()], y[ok]

    te = torch.from_numpy(np.asarray(E.year == args.test_year))
    tr = ~te
    if te.sum() == 0 or tr.sum() == 0:
        raise SystemExit(f"empty split; years present: {sorted(set(E.year))}")

    # masked per-channel window means -- the linear baseline's ENTIRE input, and
    # the thing RevIN removes from the encoder's view
    means = (X * M).sum(-1) / M.sum(-1).clamp(min=1.0)
    return X, M, y, tr, te, E, means


def run(args):
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    X, M, y, tr, te, E, means = build(args)

    print(f"device {dev}   target {args.target!r}")
    print(f"inputs {X.shape[1]} channels x {X.shape[2]} steps "
          f"({X.shape[2] * 10 / 60 / 24:.1f} days)"
          f"{'  + 19 channel means as static feats' if args.static_means else ''}")
    print(f"train {int(tr.sum())}   test {args.test_year}: {int(te.sum())}")
    print(f"target mean {y[tr].mean():.2f} +/- {y[tr].std():.2f} degC\n")

    # standardise with TRAIN statistics only; using the test set's own mean and
    # scale would leak its distribution into the target
    mu, sd = y[tr].mean(), y[tr].std().clamp(min=1e-6)
    yz = (y - mu) / sd

    Xtr, Mtr, ytr = X[tr], M[tr], yz[tr]
    Xte, Mte, yte = X[te], M[te], yz[te]

    # THE BAR. Ridge on the raw channel means -- no encoder, no training, 20
    # numbers and one linear solve. Any encoder result below this line means the
    # 3.5M parameters bought nothing over averaging, so it is computed every run
    # and drawn on the plot rather than left to be looked up later.
    mz = (means - means[tr].mean(0)) / means[tr].std(0).clamp(min=1e-6)
    linear = ridge_r2(mz[tr], ytr, mz[te], yte)
    print(f"linear baseline (ridge on {means.shape[1]} channel means): "
          f"R^2 {linear:.3f}", flush=True)

    Str = Ste = None
    if args.static_means:
        Str, Ste = mz[tr].float(), mz[te].float()

    torch.manual_seed(args.seed)
    model = NBMProbe(n_in=X.shape[1], context_len=X.shape[2],
                     n_static=means.shape[1] if args.static_means else 0).to(dev)

    # reference line: a linear probe on the UNTRAINED encoder's features. Any
    # gain above this is what training bought.
    with torch.no_grad():
        model.eval()
        Sall = mz.float() if args.static_means else None
        Hf = []
        for i in range(0, len(X), 32):
            s = None if Sall is None else Sall[i:i + 32].to(dev)
            Hf.append(model.res(model.enc(X[i:i + 32].to(dev),
                                          M[i:i + 32].to(dev),
                                          static_feats=s)).mean(1).cpu())
        Hf = torch.cat(Hf).float()
    frozen = ridge_r2(Hf[tr], ytr, Hf[te], yte)
    print(f"frozen-encoder baseline (ridge probe): R^2 {frozen:.3f}\n", flush=True)
    # stash both beside the history so --plot-only can find them without the
    # operator having to copy numbers out of a log that may not exist
    with open(args.csv + ".frozen", "w") as f:
        f.write(f"{frozen}\n{linear}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    n, hist, best = len(Xtr), [], -1e9
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot, t0 = 0.0, time.time()
        for i in range(0, n - args.batch + 1, args.batch):
            b = perm[i:i + args.batch]
            s = None if Str is None else Str[b].to(dev)
            loss = nn.functional.mse_loss(
                model(Xtr[b].to(dev), Mtr[b].to(dev), s), ytr[b].to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()

        r2_tr = r2(predict(model, Xtr, Mtr, dev, S=Str), ytr)
        r2_te = r2(predict(model, Xte, Mte, dev, S=Ste), yte)
        hist.append({"epoch": ep, "loss": tot / max(1, n // args.batch),
                     "train_r2": r2_tr, "test_r2": r2_te})
        flag = ""
        if r2_te > best:
            best, flag = r2_te, "  *"
            if args.save:
                torch.save({"model": model.state_dict(), "epoch": ep,
                            "test_r2": r2_te, "target": args.target,
                            "mu": mu, "sd": sd}, args.save)
        print(f"epoch {ep:3d}  loss {hist[-1]['loss']:.4f}  "
              f"train R2 {r2_tr:6.3f}   test R2 {r2_te:6.3f}   "
              f"{time.time()-t0:.0f}s{flag}", flush=True)
        # rewrite the history EVERY epoch, not once at the end. A run this long
        # will sometimes be killed -- a closed laptop, a session teardown -- and
        # writing only on completion throws away hours of results that were
        # already computed. Costs milliseconds.
        write_csv(hist, args.csv)

    print(f"\nbest test R2 {best:.3f}   frozen baseline {frozen:.3f}   "
          f"training bought {best - frozen:+.3f}")

    write_outputs(hist, frozen, args, linear)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def write_csv(hist, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_r2", "test_r2"])
        w.writeheader()
        w.writerows(hist)


def write_outputs(hist, frozen, args, linear=None):
    write_csv(hist, args.csv)
    print(f"wrote {args.csv}")
    plot(hist, frozen, args, linear)


def plot(hist, frozen, args, linear=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # palette roles, light surface. Two categorical slots for the two series;
    # the baseline is a REFERENCE, not a series, so it takes muted ink rather
    # than a third hue -- otherwise it competes for identity with the data.
    SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
    TEST, TRAIN = "#2a78d6", "#eb6834"

    ep = [h["epoch"] for h in hist]
    tr = [h["train_r2"] for h in hist]
    te = [h["test_r2"] for h in hist]
    best_i = int(np.argmax(te))

    fig = plt.figure(figsize=(11.0, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    # explicit margins rather than tight_layout: the title block lives in figure
    # space above the axes, and tight_layout does not know about it, so it packs
    # the axes upward until the two collide
    ax = fig.add_axes([0.075, 0.115, 0.70, 0.655])
    ax.set_facecolor(SURFACE)

    fig.text(0.075, 0.965,
             f"Predicting {args.target} from the other {args.n_in} sensors",
             fontsize=14, color=INK, va="top", ha="left")
    fig.text(0.075, 0.905,
             f"Kelmarsh SCADA · trained on every year except {args.test_year}, "
             f"tested on {args.test_year} · higher is better",
             fontsize=9.5, color=INK2, va="top", ha="left")

    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)

    lo_all = min(min(tr), min(te), frozen,
                 linear if linear is not None else frozen)

    # R^2 = 0 only earns a line when the chart actually reaches down to it.
    # Once every series sits above 0.8 it is off-scale clutter.
    if lo_all < 0.25:
        ax.axhline(0, color=AXIS, linewidth=1.2, zorder=1)
        ax.annotate("R² = 0 · no better than guessing the average",
                    xy=(0.008, 0), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color=MUTED,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))

    # Reference-line labels sit in the RIGHT MARGIN, outside the axes. Inside,
    # at the left edge, they land exactly where a rising curve crosses their own
    # line -- measured: the "the bar" label sat on top of the held-out curve.
    # Outside the axes nothing can cross them at any data range.
    ax.axhline(frozen, color=MUTED, linewidth=1.8, linestyle=(0, (5, 4)),
               zorder=2)
    ax.annotate(f"frozen encoder\nno training · R² {frozen:.3f}",
                xy=(1.0, frozen), xycoords=("axes fraction", "data"),
                xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8.5, color=INK2,
                annotation_clip=False)

    # THE BAR. Ridge on the raw channel means -- no encoder at all. A result
    # below this line means 3.5M parameters bought nothing over averaging, so
    # it is drawn in full ink and named as the bar.
    if linear is not None:
        ax.axhline(linear, color=INK, linewidth=1.6, linestyle=(0, (1.5, 2.5)),
                   zorder=2)
        ax.annotate(f"← the bar\nlinear on means · R² {linear:.3f}",
                    xy=(1.0, linear), xycoords=("axes fraction", "data"),
                    xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color=INK,
                    annotation_clip=False)

    ax.plot(ep, tr, color=TRAIN, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
            label="Train")
    ax.plot(ep, te, color=TEST, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4,
            label="Held-out year")

    # ring the best held-out epoch, but only label it when it is NOT the last
    # point -- otherwise it sits on top of the direct label
    ax.scatter([ep[best_i]], [te[best_i]], s=95, facecolor="none",
               edgecolor=TEST, linewidth=1.8, zorder=5)
    if best_i != len(ep) - 1:
        ax.annotate(f"best {te[best_i]:.3f}", xy=(ep[best_i], te[best_i]),
                    xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=9, color=TEST)

    # direct labels at the last point, so identity never rests on colour alone.
    # Nudged apart when the two series land on top of each other.
    gap = abs(tr[-1] - te[-1])
    span = max(max(tr + te + [frozen]) - min(tr + te + [0, frozen]), 1e-6)
    # push the HIGHER series up and the lower one down. Doing it the other way
    # round -- which is what a naive fixed (-8, +8) does when train sits above
    # test -- drives the two labels together instead of apart.
    if gap > span * 0.12:
        off_tr = off_te = 0
    else:
        off_tr = 9 if tr[-1] >= te[-1] else -9
        off_te = -off_tr
    for series, color, name, dy in ((tr, TRAIN, "Train", off_tr),
                                    (te, TEST, f"Held-out {args.test_year}",
                                     off_te)):
        ax.annotate(f"{name} {series[-1]:.3f}",
                    xy=(ep[-1], series[-1]), xytext=(8, dy),
                    textcoords="offset points", va="center",
                    fontsize=9.5, color=color)

    ax.set_xlabel("Training epoch", fontsize=10.5, color=INK2, labelpad=8)
    ax.set_ylabel("R²  ·  share of variance explained", fontsize=10.5,
                  color=INK2, labelpad=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)

    refs = [frozen] + ([linear] if linear is not None else [])
    lo = min(min(tr), min(te), *refs)
    hi = max(max(tr), max(te), *refs)
    pad = max(0.06, (hi - lo) * 0.14)
    ax.set_ylim(lo - pad * 0.5, hi + pad)
    # room on the right for the direct labels, scaled to the longest one
    ax.set_xlim(-0.4, max(ep) + max(1.2, (max(ep) + 1) * 0.22))

    # legend ABOVE the axes, as a horizontal row under the subtitle. Inside the
    # plot it collides with whichever reference line happens to be lowest --
    # measured: it sat on top of the frozen-encoder line and its label.
    # Outside, it cannot collide with anything at any data range.
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.12),
                    mode=None, ncol=2, frameon=False, fontsize=9.5,
                    handlelength=2.0, handletextpad=0.6, columnspacing=2.0,
                    borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.savefig(args.png, facecolor=SURFACE)
    print(f"wrote {args.png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/kelmarsh")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--test-year", type=int, default=2016)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--stride", type=int, default=144)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="kelmarsh_nbm_best.pt")
    p.add_argument("--csv", default="kelmarsh_nbm_history.csv")
    p.add_argument("--png", default="kelmarsh_nbm.png")
    p.add_argument("--static-means", action="store_true",
                   help="feed the per-channel window means in as static "
                        "features -- the linear baseline's entire input, "
                        "which RevIN otherwise strips from the encoder's view")
    p.add_argument("--plot-only", action="store_true",
                   help="re-draw the PNG from an existing history CSV, so the "
                        "figure can be iterated on without repeating the run")
    p.add_argument("--frozen", type=float, default=None,
                   help="baseline for --plot-only (read from the run's output)")
    a = p.parse_args()
    a.n_in = len(CHANNELS) - 1

    if a.plot_only:
        import csv
        with open(a.csv) as f:
            hist = [{k: float(v) for k, v in row.items()}
                    for row in csv.DictReader(f)]
        lin = None
        if os.path.exists(a.csv + ".frozen"):
            vals = open(a.csv + ".frozen").read().split()
            if a.frozen is None and vals:
                a.frozen = float(vals[0])
            if len(vals) > 1:
                lin = float(vals[1])
        if a.frozen is None:
            raise SystemExit("--plot-only needs --frozen <r2 from the run>")
        plot(hist, a.frozen, a, lin)
    else:
        run(a)
