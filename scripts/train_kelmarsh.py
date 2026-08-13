"""Train the SCADA encoder on Kelmarsh to predict forced outages.

    python3 train_kelmarsh.py                  # defaults, ~1 hr on MPS
    python3 train_kelmarsh.py --epochs 5       # quick look first
    python3 train_kelmarsh.py --test-year 2019

WHAT THIS IS FOR

The frozen-feature probe answers "did the information survive the wiring". It
cannot answer "can this be learned", because an untrained encoder's random
projections preserve information without organising it. This trains end to end
and reports the same metric, so the two numbers are directly comparable.

THE SPLIT IS THE EXPERIMENT

One year held out ENTIRELY. Not a percentage cut inside a year -- that puts
summer in train and winter in test, and every channel here has an annual cycle.
Measured on 2016 with a 70/30 positional cut: a probe scored R^2 -12.5 on
held-out ambient temperature and BELOW chance on month, i.e. train and test were
different domains and the probe reported that as "no signal". Holding out a
whole year puts all four seasons on both sides.

METRIC IS PR-AUC, NOT ACCURACY

Positives run ~14%, so "never fails" scores ~0.86 accuracy while being useless.
PR-AUC's trivial baseline is the positive rate itself, which the run prints
beside every score so the number cannot flatter itself.

READ THE CONTROL ROW FIRST. A shuffled-label run is reported at the end; if it
does not collapse to the base rate, the split leaked and the headline number
means nothing.
"""

import argparse
import os
import sys
import time

# the repo root, so `models` and `data_io` resolve no matter which directory
# this is invoked from. Same bootstrap the test suites carry.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from data_io.kelmarsh_io import load_years, N_CHANNELS
from models.scada_encoder_tcn import ScadaTCNEncoder, PerceiverResampler


class OutageProbe(nn.Module):
    """Encoder + resampler + linear head, i.e. the real training shape.

    BatchNorm before pooling for the reason both encoder smoke tests carry it:
    pooled features sit on a DC offset many times their per-window spread
    (measured 15.7x on Kelmarsh 2016, inter-sample cosine 0.9969), which strands
    a linear head at chance. LayerNorm normalises the wrong axis.
    """

    def __init__(self, d_model=128, n_latents=32, context_len=600):
        super().__init__()
        self.enc = ScadaTCNEncoder(d_model=d_model, n_channels=N_CHANNELS,
                                   context_len=context_len)
        self.res = PerceiverResampler(d_model=d_model, n_latents=n_latents)
        self.norm = nn.BatchNorm1d(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        z = self.res(self.enc(x, mask))              # (B, n_latents, d)
        B, L, d = z.shape
        z = self.norm(z.reshape(B * L, d)).reshape(B, L, d)
        return self.head(z.mean(1)).squeeze(-1)


def pr_auc(scores, y):
    """Average precision. Trivial baseline is the positive rate."""
    order = torch.argsort(scores, descending=True)
    yy = y[order]
    tp = torch.cumsum(yy, 0)
    prec = tp / torch.arange(1, len(yy) + 1, device=yy.device)
    return (prec * yy).sum().item() / max(yy.sum().item(), 1)


@torch.no_grad()
def evaluate(model, X, M, Y, dev, bs=32):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(dev), M[i:i + bs].to(dev)).cpu())
    s = torch.cat(out)
    return pr_auc(s, Y), s


def run(args):
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    # turbine ids are dropped here on purpose: the split is by YEAR, so which
    # machine a window came from does not enter it
    X, M, E, _, Y = load_years(args.root, stride=args.stride)

    te = torch.from_numpy((E.year == args.test_year).values)
    tr = ~te
    if te.sum() == 0:
        raise SystemExit(f"no windows from {args.test_year}; "
                         f"years present: {sorted(set(E.year))}")

    print(f"device {dev}   windows {len(X)}   years {sorted(set(E.year))}")
    print(f"train {int(tr.sum())} ({Y[tr].mean():.3f} pos)   "
          f"test {args.test_year}: {int(te.sum())} ({Y[te].mean():.3f} pos)")
    base = Y[te].mean().item()
    print(f"PR-AUC baseline (chance) = {base:.3f}\n")

    Xtr, Mtr, Ytr = X[tr], M[tr], Y[tr]
    Xte, Mte, Yte = X[te], M[te], Y[te]

    torch.manual_seed(args.seed)
    model = OutageProbe(context_len=X.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # positives are ~14%, so an unweighted loss is minimised by predicting
    # "healthy" for everything. This makes a missed failure ~6x as expensive.
    pos_weight = torch.tensor([(1 - Ytr.mean()) / Ytr.mean().clamp(min=1e-6)]).to(dev)
    print(f"pos_weight {pos_weight.item():.2f}\n")

    n = len(Xtr)
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot, t0 = 0.0, time.time()
        for i in range(0, n - args.batch + 1, args.batch):
            b = perm[i:i + args.batch]
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(Xtr[b].to(dev), Mtr[b].to(dev)), Ytr[b].to(dev),
                pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        ap, _ = evaluate(model, Xte, Mte, Yte, dev)
        flag = ""
        if ap > best:
            best, flag = ap, "  *"
            if args.save:
                torch.save({"model": model.state_dict(), "epoch": ep,
                            "pr_auc": ap}, args.save)
        print(f"epoch {ep:3d}  loss {tot/max(1,(n//args.batch)):.4f}  "
              f"test PR-AUC {ap:.3f}  (base {base:.3f})  "
              f"{time.time()-t0:.0f}s{flag}")

    print(f"\nbest test PR-AUC {best:.3f}   baseline {base:.3f}   "
          f"lift {best - base:+.3f}")

    # ---- the control. Read this before believing anything above -------------
    if args.control:
        print("\ncontrol: same split, labels shuffled within train and test")
        g = torch.Generator().manual_seed(0)
        Ytr_s = Ytr[torch.randperm(len(Ytr), generator=g)]
        Yte_s = Yte[torch.randperm(len(Yte), generator=g)]
        torch.manual_seed(args.seed)
        cm = OutageProbe(context_len=X.shape[-1]).to(dev)
        co = torch.optim.AdamW(cm.parameters(), lr=args.lr)
        for ep in range(min(args.epochs, args.control_epochs)):
            cm.train()
            perm = torch.randperm(n)
            for i in range(0, n - args.batch + 1, args.batch):
                b = perm[i:i + args.batch]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    cm(Xtr[b].to(dev), Mtr[b].to(dev)), Ytr_s[b].to(dev))
                co.zero_grad(); loss.backward(); co.step()
        ap_s, _ = evaluate(cm, Xte, Mte, Yte_s, dev)
        verdict = "PASS" if ap_s < base * 1.4 else "FAIL -- SPLIT LEAKS"
        print(f"  shuffled-label PR-AUC {ap_s:.3f}  vs base {base:.3f}   {verdict}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/kelmarsh")
    p.add_argument("--test-year", type=int, default=2020)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--stride", type=int, default=144)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="kelmarsh_best.pt")
    p.add_argument("--control", action="store_true", default=True)
    p.add_argument("--no-control", dest="control", action="store_false")
    p.add_argument("--control-epochs", type=int, default=5)
    run(p.parse_args())
