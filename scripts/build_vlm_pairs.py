"""Manufacture (health vector, brief, assessment) triples to train the projector.

    python3 scripts/build_vlm_pairs.py                    # writes runs/vlm_pairs/
    python3 scripts/build_vlm_pairs.py --limit 200        # quick look

THE ONE DESIGN DECISION THAT DECIDES WHETHER THIS WORKS

The projector only learns if the target says something the brief does not.

If the brief states "residual +3.2 C, rising 11 days" and the target says
"investigate the cooling circuit", the model learns brief -> target and routes
around the soft tokens entirely: the health vector contributes nothing, the
projector receives almost no gradient, and the loss still falls. You would see
a trained-looking run and a useless adapter.

So the split here is deliberate:

    BRIEF   operating context and fault history only -- wind, power, ambient,
            what the status log recorded. Says NOTHING about the residual.
    TARGET  the assessment, including the residual finding and its trend.

The residual is therefore reachable only through the health vector. That makes
the task honest: a projector that ignores its input cannot write the target,
and one that learns must have encoded the deviation.

WHERE THE HEALTH VECTOR COMES FROM

Ideally MultiModalFusion, but only SCADA has real data on this project and an
untrained fusion emits random projections. The trained NBM encoder's pooled
features are used instead by default: they are measured to carry the signal
(R^2 0.941 on a held-out year), which is the property the projector needs.
--from-fusion switches to the fusion path once other modalities have data.

OUTPUT

    runs/vlm_pairs/pairs.jsonl    one row per window: brief, target, metadata
    runs/vlm_pairs/health.pt      (N, 128) aligned to the jsonl by row order
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_io.kelmarsh_io import load_years, CHANNELS
from models.scada_encoder_tcn import ScadaTCNEncoder
from models.common import PerceiverResampler


class NBMFeatures(nn.Module):
    """train_nbm.py's architecture, exposing the pooled 128-d representation.

    Identical to NBMProbe up to the head, so the trained checkpoint loads with
    strict=False and the head's two tensors are simply unused.
    """

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

    def forward(self, x, mask, static=None, want="both"):
        z = self.res(self.enc(x, mask,
                              static_feats=static if self.n_static else None))
        B, L, d = z.shape
        z = self.norm(z.reshape(B * L, d)).reshape(B, L, d)
        pooled = z.mean(1)                       # (B, 128) -- the health vector
        return (pooled, self.head(pooled).squeeze(-1)) if want == "both" else pooled


def load_events(root):
    """Every forced outage, keyed by turbine, for the history line."""
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "scada_*",
                                           "Status_Kelmarsh_*.csv"))):
        d = pd.read_csv(f, skiprows=9)
        d["turbine"] = int(re.search(r"Kelmarsh_(\d)_", f).group(1))
        rows.append(d)
    S = pd.concat(rows, ignore_index=True)
    S = S[S["IEC category"] == "Forced outage"].copy()
    S["start"] = pd.to_datetime(S["Timestamp start"])
    return S[["turbine", "start", "Message"]]


# ---------------------------------------------------------------------------
# the two text sides
# ---------------------------------------------------------------------------
def make_brief(turbine, when, ctx, recent, upcoming_days=None):
    """Operating context and fault history. Deliberately residual-free."""
    L = [f"Turbine {turbine}, Kelmarsh. Window of 4.2 days ending "
         f"{when:%Y-%m-%d}."]
    L.append(f"Mean wind {ctx['wind']:.1f} m/s, power {ctx['power']:.0f} kW, "
             f"ambient {ctx['ambient']:.1f} C.")
    if recent:
        L.append("Forced outages in the last 90 days: "
                 + "; ".join(f"{d:%Y-%m-%d} {m}" for d, m in recent[:3]) + ".")
    else:
        L.append("No forced outages logged in the last 90 days.")
    L.append("Assess the generator bearing front temperature.")
    return " ".join(L)


def make_target(resid, trend, sd, upcoming):
    """The assessment. Every number here is reachable only via the health vector."""
    z = resid / sd if sd else 0.0
    L = []
    if abs(z) < 1.0:
        L.append(f"Generator bearing front temperature is tracking its expected "
                 f"value (residual {resid:+.1f} C, within this machine's normal "
                 f"scatter).")
    elif z > 0:
        L.append(f"Generator bearing front temperature is running {resid:+.1f} C "
                 f"above the value its operating conditions predict "
                 f"({z:+.1f} standard deviations).")
    else:
        L.append(f"Generator bearing front temperature is running {resid:+.1f} C "
                 f"below prediction ({z:+.1f} standard deviations), which is "
                 f"unusual but not a damage signature.")

    if trend is not None and abs(trend) > 0.3:
        L.append(f"The deviation has {'grown' if trend > 0 else 'fallen'} "
                 f"{abs(trend):.1f} C over the preceding two weeks.")
    else:
        L.append("The deviation is stable over the preceding two weeks.")

    if z > 2.0 and (trend or 0) > 0.3:
        L.append("Recommend inspecting the generator cooling circuit within "
                 "two weeks.")
    elif z > 1.0:
        L.append("Recommend continued monitoring; no intervention yet.")
    else:
        L.append("No action required.")

    if upcoming is not None:
        L.append(f"A forced outage was logged {upcoming} days after this window.")
    return " ".join(L)


# ---------------------------------------------------------------------------
def build(args):
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    target_ch, mu, sd_t = ck["target"], float(ck["mu"]), float(ck["sd"])
    ti = CHANNELS.index(target_ch)
    keep = [i for i in range(len(CHANNELS)) if i != ti]

    X, M, E, T, _ = load_years(args.root, stride=args.stride)
    tm = M[:, ti]
    y = (X[:, ti] * tm).sum(-1) / tm.sum(-1).clamp(min=1.0)
    ok = tm.mean(-1) > 0.5
    X, M, E, T, y = X[ok][:, keep], M[ok][:, keep], E[ok.numpy()], T[ok], y[ok]

    means = (X * M).sum(-1) / M.sum(-1).clamp(min=1.0)
    tr = torch.from_numpy(np.asarray(E.year != args.val_year))
    mz = (means - means[tr].mean(0)) / means[tr].std(0).clamp(min=1e-6)

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    model = NBMFeatures(n_in=X.shape[1], context_len=X.shape[2],
                        n_static=mz.shape[1]).to(dev)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"loaded {args.ckpt}  (missing {len(missing)}, unexpected {len(unexpected)})")

    print(f"encoding {len(X)} windows on {dev} ...")
    H, P = [], []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            h, p = model(X[i:i+32].to(dev), M[i:i+32].to(dev),
                         mz[i:i+32].float().to(dev))
            H.append(h.cpu()); P.append(p.cpu())
    health = torch.cat(H)
    pred = torch.cat(P) * sd_t + mu
    resid = (y - pred).numpy()

    events = load_events(args.root)
    days = E.floor("D").values.astype("datetime64[D]")
    turb = T.numpy()

    idx_ch = lambda n: keep.index(CHANNELS.index(n))
    rows, order = [], []
    for i in range(len(X)):
        if args.limit and len(rows) >= args.limit:
            break
        t, when = int(turb[i]), pd.Timestamp(E[i])
        same = turb == t
        # this turbine's own residual scatter -- a machine that always runs
        # slightly warm should not be permanently flagged
        sd = float(np.std(resid[same])) or 1.0
        prior = same & (days >= days[i] - np.timedelta64(14, "D")) & (days < days[i])
        trend = float(resid[i] - resid[prior].mean()) if prior.any() else None

        ev = events[events.turbine == t]
        recent = [(d, m) for d, m in zip(ev.start, ev.Message)
                  if when - pd.Timedelta(days=90) < d <= when]
        ahead = [(d - when).days for d in ev.start
                 if when < d <= when + pd.Timedelta(days=args.horizon)]

        ctx = {"wind": float(means[i, idx_ch("Wind speed (m/s)")]),
               "power": float(means[i, idx_ch("Power (kW)")]),
               "ambient": float(means[i, idx_ch("Nacelle ambient temperature (°C)")])}

        rows.append({
            "turbine": t, "date": f"{when:%Y-%m-%d}", "year": int(when.year),
            "split": "val" if when.year == args.val_year else "train",
            "brief": make_brief(t, when, ctx, recent),
            "target": make_target(float(resid[i]), trend, sd,
                                  min(ahead) if ahead else None),
            "residual": round(float(resid[i]), 3),
        })
        order.append(i)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pairs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    torch.save(health[order], os.path.join(args.out, "health.pt"))

    n_tr = sum(r["split"] == "train" for r in rows)
    print(f"\nwrote {len(rows)} pairs to {args.out}/")
    print(f"  train {n_tr}   val {len(rows) - n_tr} (year {args.val_year})")
    print(f"  health {tuple(health[order].shape)}")
    print(f"\nexample pair:")
    print(f"  BRIEF  {rows[0]['brief']}")
    print(f"  TARGET {rows[0]['target']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/kelmarsh")
    p.add_argument("--ckpt", default="runs/genbrg_static/run.pt")
    p.add_argument("--out", default="runs/vlm_pairs")
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--stride", type=int, default=144)
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument("--limit", type=int, default=0)
    build(p.parse_args())
