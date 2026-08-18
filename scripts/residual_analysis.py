"""Do NBM residuals rise before a generator fault?

    python3 scripts/residual_analysis.py                    # uses the last run
    python3 scripts/residual_analysis.py --ckpt runs/x/run.pt --test-year 2016

WHAT THIS ASKS, AND WHY IT IS NOT THE R^2 QUESTION

train_nbm.py already showed the encoder predicts generator bearing temperature
better than ridge on 19 channel means -- 0.941 against 0.856 on a held-out year.
That is a STATISTICAL win, averaged over ~11,000 windows that are almost all
ordinary operation.

Faults are the rare tail. A model can win on the average and tie on the tail,
so "more accurate" does not imply "warns earlier". This measures the tail
directly:

    residual = actual bearing temperature - predicted

A healthy machine sits near zero. A generator whose cooling is failing runs
hotter than its operating point justifies, so the residual should drift up in
the days before the fault is logged.

THREE THINGS MAKE IT HONEST

  1. BOTH MODELS. The encoder's residuals AND the 20-parameter linear
     baseline's, on the same incidents. If the linear model warns just as
     early, the encoder is statistically better and operationally irrelevant --
     which is a real possible outcome and the one worth being able to detect.
  2. A CONTROL. The same curve around random dates with no fault nearby. It
     must stay flat; if it rises too, the "warning" is seasonal drift.
  3. HELD-OUT YEAR ONLY. Residuals from the year the model never trained on.
     On training years the model has already fitted the fault period and the
     residual is suppressed by construction.

WHICH FAULTS

Generator fan overload is the most common forced outage at Kelmarsh (336 of
1032 across 2016-2021) and it is the right physics for this target: a failing
fan means the generator runs hot, so a generator-bearing residual should see
it. Codes 2550/2650/2655 fire together, so incidents are deduplicated to one
per turbine per day.

OUTPUT

    runs/<name>/residuals.png    superposed-epoch curve, both models + control
    runs/<name>/residuals.csv    the underlying per-day means
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from data_io.kelmarsh_io import load_years, load_turbine, make_windows, CHANNELS
from models.scada_encoder_tcn import ScadaTCNEncoder
from models.common import PerceiverResampler
import torch.nn as nn

FAULT_RE = r"generator|bearing|temperat|overload|fan|cool"


class NBMProbe(nn.Module):
    """Must match train_nbm.py's architecture exactly to load its weights."""

    def __init__(self, n_in, d_model=128, n_latents=32, context_len=600,
                 n_static=0):
        super().__init__()
        self.n_static = n_static
        self.enc = ScadaTCNEncoder(d_model=d_model, n_channels=n_in,
                                   context_len=context_len,
                                   n_static=max(n_static, 1))
        self.res = PerceiverResampler(d_model=d_model, n_latents=n_latents)
        self.norm = nn.BatchNorm1d(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask, static=None):
        z = self.res(self.enc(x, mask,
                              static_feats=static if self.n_static else None))
        B, L, d = z.shape
        z = self.norm(z.reshape(B * L, d)).reshape(B, L, d)
        return self.head(z.mean(1)).squeeze(-1)


def load_faults(root, pattern=FAULT_RE):
    """Forced outages whose message looks thermal -> one row per incident.

    The three generator-fan codes share a timestamp, so without the dedup a
    single cooling failure would count three times and dominate the average.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "scada_*",
                                           "Status_Kelmarsh_*.csv"))):
        d = pd.read_csv(f, skiprows=9)
        d["turbine"] = int(re.search(r"Kelmarsh_(\d)_", f).group(1))
        rows.append(d)
    S = pd.concat(rows, ignore_index=True)
    S = S[S["IEC category"] == "Forced outage"]
    S = S[S.Message.str.contains(pattern, case=False, na=False)].copy()
    S["start"] = pd.to_datetime(S["Timestamp start"])
    S["day"] = S.start.dt.floor("D")
    return (S.groupby(["turbine", "day"], as_index=False)
              .agg(message=("Message", "first")))


def ridge_fit(A, b, lam=1e-2):
    mu = A.mean(0)
    X = torch.cat([A - mu, torch.ones(len(A), 1)], 1)
    bm = b.mean()
    w = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]),
                           X.T @ (b[:, None] - bm))
    return lambda Z: (torch.cat([Z - mu, torch.ones(len(Z), 1)], 1)
                      @ w).squeeze(-1) + bm


def superpose(resid, days, turbines, incidents, lo, hi):
    """Mean residual at each day offset around every incident.

    Returns (offsets, mean, sem, n_incidents_contributing).
    """
    offs = np.arange(lo, hi + 1)
    buckets = {o: [] for o in offs}
    used = 0
    for t, d0 in incidents:
        sel = turbines == t
        if not sel.any():
            continue
        dt = (days[sel] - d0).astype("timedelta64[D]").astype(int)
        r = resid[sel]
        hit = False
        for o in offs:
            m = dt == o
            if m.any():
                buckets[o].append(float(r[m].mean()))
                hit = True
        used += hit
    mean = np.array([np.mean(buckets[o]) if buckets[o] else np.nan for o in offs])
    sem = np.array([
        (np.std(buckets[o]) / max(np.sqrt(len(buckets[o])), 1)) if buckets[o]
        else np.nan for o in offs])
    return offs, mean, sem, used


def run(args):
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    target = ck["target"]
    mu, sd = float(ck["mu"]), float(ck["sd"])
    print(f"checkpoint  {args.ckpt}")
    print(f"target      {target}   (epoch {ck['epoch']}, R2 {ck['test_r2']:.3f})")

    ti = CHANNELS.index(target)
    keep = [i for i in range(len(CHANNELS)) if i != ti]

    # rebuild windows WITH turbine ids, which load_years gives us
    X, M, E, T, _ = load_years(args.root, stride=args.stride)
    tm = M[:, ti]
    y = (X[:, ti] * tm).sum(-1) / tm.sum(-1).clamp(min=1.0)
    ok = tm.mean(-1) > 0.5
    X, M, E, T, y = X[ok][:, keep], M[ok][:, keep], E[ok.numpy()], T[ok], y[ok]

    te = torch.from_numpy(np.asarray(E.year == args.test_year))
    tr = ~te
    means = (X * M).sum(-1) / M.sum(-1).clamp(min=1.0)
    mz = (means - means[tr].mean(0)) / means[tr].std(0).clamp(min=1e-6)
    print(f"windows     {len(y)}   held-out {args.test_year}: {int(te.sum())}")

    # -- encoder residuals ---------------------------------------------------
    model = NBMProbe(n_in=X.shape[1], context_len=X.shape[2],
                     n_static=mz.shape[1]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            preds.append(model(X[i:i+32].to(dev), M[i:i+32].to(dev),
                               mz[i:i+32].float().to(dev)).cpu())
    pred_enc = torch.cat(preds) * sd + mu               # back to degrees C
    res_enc = (y - pred_enc).numpy()

    # -- linear baseline residuals, fitted on the SAME training years --------
    f = ridge_fit(mz[tr], (y[tr] - mu) / sd)
    pred_lin = f(mz) * sd + mu
    res_lin = (y - pred_lin).numpy()

    tev = te.numpy()
    print(f"\nheld-out {args.test_year} residual spread (degC):")
    print(f"  encoder  mean {res_enc[tev].mean():+.3f}  sd {res_enc[tev].std():.3f}")
    print(f"  linear   mean {res_lin[tev].mean():+.3f}  sd {res_lin[tev].std():.3f}")

    # -- incidents in the held-out year --------------------------------------
    faults = load_faults(args.root)
    faults = faults[faults.day.dt.year == args.test_year]
    incidents = [(int(r.turbine), np.datetime64(r.day, "D"))
                 for r in faults.itertuples()]
    print(f"\n{len(incidents)} thermal incidents in {args.test_year} "
          f"across {faults.turbine.nunique()} turbines")
    if len(incidents) < 5:
        raise SystemExit("too few incidents in the held-out year to average")

    days = E.floor("D").values.astype("datetime64[D]")
    turb = T.numpy()
    keep_te = tev
    d_te, t_te = days[keep_te], turb[keep_te]

    # control: random dates at least `hi` days from any real incident
    rng = np.random.default_rng(0)
    real = {(t, d) for t, d in incidents}
    pool = sorted({(int(t), d) for t, d in zip(t_te, d_te)})
    ctrl = [p for p in pool
            if all(abs((p[1] - d).astype(int)) > args.hi + 10
                   for t, d in real if t == p[0])]
    ctrl = [ctrl[i] for i in rng.choice(len(ctrl),
                                        size=min(len(incidents) * 3, len(ctrl)),
                                        replace=False)] if ctrl else []
    print(f"{len(ctrl)} control dates (>{args.hi + 10} days from any incident)")

    out = {}
    for name, r in (("encoder", res_enc[keep_te]), ("linear", res_lin[keep_te])):
        offs, m, s, n = superpose(r, d_te, t_te, incidents, args.lo, args.hi)
        out[name] = (offs, m, s, n)
        print(f"  {name:<8} contributed by {n} incidents")
    offs, cm, cs, cn = superpose(res_enc[keep_te], d_te, t_te, ctrl,
                                 args.lo, args.hi)
    out["control"] = (offs, cm, cs, cn)

    write(out, target, args)


def write(out, target, args):
    offs = out["encoder"][0]
    df = pd.DataFrame({"day_offset": offs})
    for k in ("encoder", "linear", "control"):
        df[f"{k}_mean"] = out[k][1]
        df[f"{k}_sem"] = out[k][2]
    df.to_csv(args.csv, index=False)
    print(f"\nwrote {args.csv}")
    plot(out, target, args)


def plot(out, target, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
    ENC, LIN = "#2a78d6", "#eb6834"

    fig = plt.figure(figsize=(10.5, 5.6), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax = fig.add_axes([0.075, 0.115, 0.71, 0.655])
    ax.set_facecolor(SURF)

    fig.text(0.075, 0.965, "Does the model see a generator fault coming?",
             fontsize=14, color=INK, va="top")
    fig.text(0.075, 0.905,
             f"Residual in {target.split(' (')[0]} · {args.test_year} held-out · "
             f"{out['encoder'][3]} incidents · above 0 = hotter than predicted",
             fontsize=9.5, color=INK2, va="top")

    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)

    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=1)
    # day 0 is when the fault was LOGGED; everything left of it is warning time
    ax.axvline(0, color=INK, linewidth=1.4, linestyle=(0, (3, 3)), zorder=2)
    ax.annotate("fault logged", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(5, -12), textcoords="offset points",
                fontsize=9, color=INK, ha="left")

    for name, color, label in (("control", MUTED, "No fault (control)"),
                               ("linear", LIN, "Linear baseline"),
                               ("encoder", ENC, "Encoder")):
        o, m, s, n = out[name]
        ok = ~np.isnan(m)
        ax.plot(o[ok], m[ok], color=color, linewidth=2.2 if name != "control" else 1.8,
                linestyle="-" if name != "control" else (0, (5, 4)),
                marker="o" if name != "control" else None, markersize=3.5,
                markeredgecolor=SURF, markeredgewidth=0.8,
                zorder=4 if name == "encoder" else 3, label=label)
        ax.fill_between(o[ok], (m - s)[ok], (m + s)[ok], color=color,
                        alpha=0.13, linewidth=0, zorder=2)
        if ok.any():
            ax.annotate(f"{label}", xy=(o[ok][-1], m[ok][-1]),
                        xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=color,
                        annotation_clip=False)

    ax.set_xlabel("Days relative to the logged fault", fontsize=10.5,
                  color=INK2, labelpad=8)
    ax.set_ylabel("Residual  ·  actual − predicted  (°C)", fontsize=10.5,
                  color=INK2, labelpad=8)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.12),
                    ncol=3, frameon=False, fontsize=9.5, handlelength=2.0,
                    borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.savefig(args.png, facecolor=SURF)
    print(f"wrote {args.png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/kelmarsh")
    p.add_argument("--ckpt", default="runs/genbrg_static/run.pt")
    p.add_argument("--test-year", type=int, default=2016)
    p.add_argument("--stride", type=int, default=144)
    p.add_argument("--lo", type=int, default=-30, help="days before the fault")
    p.add_argument("--hi", type=int, default=5, help="days after")
    p.add_argument("--csv", default="runs/genbrg_static/residuals.csv")
    p.add_argument("--png", default="runs/genbrg_static/residuals.png")
    run(p.parse_args())
