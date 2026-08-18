"""Turn model outputs into a turbine brief, then (optionally) into a written assessment.

    python3 scripts/turbine_report.py --turbine 3            # brief only, no API key
    python3 scripts/turbine_report.py --turbine 3 --llm      # + written assessment

WHY TEXT AND NOT EMBEDDINGS

The obvious way to put a language model on top of this stack is to project the
fusion model's 128-d health vector into the LLM's token space and prepend it to
a prompt. That runs, and it produces fluent nonsense: those 128 numbers land in
a space the LLM has never seen, so they are syntactically valid tokens carrying
no meaning. Making them mean something needs alignment training on paired
(sensor window, text description) examples, which do not exist for wind
turbines and would have to be manufactured.

This is the other path: convert the model's outputs to numbers and words the
LLM already understands, and let it reason over those. It gives up whatever is
in the health vector that does not survive being written down -- but for a
maintenance assessment, most of what matters is verbalisable, and this works
today rather than after a data-collection project.

THE BRIEF IS THE DELIVERABLE, NOT THE PROSE

`build_brief` needs no API key and no network. It is the part specific to this
project: residuals from the trained NBM model, their trend, the operating
context that explains them, and the fault history they should be read against.
The LLM call is a thin wrapper that turns that into paragraphs. If the prose is
ever wrong, the brief is what you check it against.
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
import torch.nn as nn

from data_io.kelmarsh_io import load_years, CHANNELS
from models.scada_encoder_tcn import ScadaTCNEncoder
from models.common import PerceiverResampler

MODEL_ID = "claude-opus-5"

SYSTEM = """You are a wind turbine condition-monitoring analyst writing for the \
maintenance planner at a 6-turbine onshore site.

You are given a brief produced by a normal-behaviour model: it predicts what a \
sensor SHOULD read given the turbine's operating conditions, and the residual is \
actual minus predicted. A residual near zero means the turbine is behaving as its \
operating point predicts. A residual that drifts upward over weeks means the \
component is running hotter than conditions justify, which is the signature of \
developing damage.

Write a short assessment: what the data shows, how confident you are, and what \
you would do. Be specific about numbers and dates.

Two things you must not do. Do not treat a single elevated reading as a fault -- \
sustained trend is the signal, and a one-day excursion usually is not. And do not \
invent detail the brief does not contain: if the brief has no vibration data, say \
that vibration would help rather than describing what it shows. State your \
uncertainty plainly; a planner acting on a false alarm loses more than one who \
waits for another week of data."""


class NBMProbe(nn.Module):
    """Must match train_nbm.py exactly to load its weights."""

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


def load_events(root, turbine, before, days=90):
    """Status-log events for one turbine in the window before `before`."""
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "scada_*",
                                           f"Status_Kelmarsh_{turbine}_*.csv"))):
        rows.append(pd.read_csv(f, skiprows=9))
    if not rows:
        return pd.DataFrame(columns=["start", "Message", "IEC category"])
    S = pd.concat(rows, ignore_index=True)
    S["start"] = pd.to_datetime(S["Timestamp start"])
    lo = before - pd.Timedelta(days=days)
    return S[(S.start > lo) & (S.start <= before) &
             (S["IEC category"] == "Forced outage")].sort_values("start")


def build_brief(args):
    """Everything the assessment needs, as plain text. No API key required."""
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    target, mu, sd = ck["target"], float(ck["mu"]), float(ck["sd"])
    ti = CHANNELS.index(target)
    keep = [i for i in range(len(CHANNELS)) if i != ti]

    X, M, E, T, _ = load_years(args.root, stride=args.stride)
    tm = M[:, ti]
    y = (X[:, ti] * tm).sum(-1) / tm.sum(-1).clamp(min=1.0)
    ok = tm.mean(-1) > 0.5
    X, M, E, T, y = X[ok][:, keep], M[ok][:, keep], E[ok.numpy()], T[ok], y[ok]

    means = (X * M).sum(-1) / M.sum(-1).clamp(min=1.0)
    train = torch.from_numpy(np.asarray(E.year != args.test_year))
    mz = (means - means[train].mean(0)) / means[train].std(0).clamp(min=1e-6)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = NBMProbe(n_in=X.shape[1], context_len=X.shape[2],
                     n_static=mz.shape[1]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X[i:i+32].to(dev), M[i:i+32].to(dev),
                                mz[i:i+32].float().to(dev)).cpu()
                          for i in range(0, len(X), 32)]) * sd + mu
    resid = (y - pred).numpy()

    sel = (T == args.turbine).numpy() & np.asarray(E.year == args.test_year)
    if not sel.any():
        raise SystemExit(f"no {args.test_year} windows for turbine {args.turbine}")
    idx = np.where(sel)[0]
    idx = idx[np.argsort(E[sel].values)]
    now = idx[-1] if args.at is None else idx[
        np.argmin(np.abs(E[idx].values - np.datetime64(args.at)))]

    days = E[idx].values
    r = resid[idx]
    # the reference spread is THIS turbine's own residual scatter, so a machine
    # that always runs slightly warm is not permanently flagged
    ref_sd = float(np.std(r)) or 1.0
    when = pd.Timestamp(E[now])

    ctx = {n: float(means[now, keep.index(CHANNELS.index(n))])
           for n in ("Wind speed (m/s)", "Power (kW)",
                     "Nacelle ambient temperature (°C)")
           if CHANNELS.index(n) in keep}

    trend = []
    for back in (30, 20, 10, 5, 3, 1, 0):
        m = (days >= np.datetime64(when - pd.Timedelta(days=back + 1))) & \
            (days <= np.datetime64(when - pd.Timedelta(days=back)))
        if m.any():
            trend.append((back, float(r[m].mean())))

    ev = load_events(args.root, args.turbine, when)

    L = []
    L.append(f"TURBINE {args.turbine} — Kelmarsh Wind Farm (6 x Senvion MM92)")
    L.append(f"Window ending {when:%Y-%m-%d %H:%M} "
             f"({X.shape[2]} x 10-minute rows = "
             f"{X.shape[2]*10/60/24:.1f} days of history)")
    L.append("")
    L.append("NORMAL-BEHAVIOUR MODEL")
    L.append(f"  predicts          {target}")
    L.append(f"  from              the other {X.shape[1]} SCADA channels")
    L.append(f"  held-out quality  R² {ck['test_r2']:.3f} on {args.test_year}, "
             f"a year the model never trained on")
    L.append(f"  linear baseline   R² 0.856 (ridge on channel means, no encoder)")
    L.append("")
    L.append("CURRENT STATE")
    L.append(f"  actual            {float(y[now]):.1f} °C")
    L.append(f"  predicted         {float(pred[now]):.1f} °C")
    L.append(f"  residual          {resid[now]:+.2f} °C   "
             f"({resid[now]/ref_sd:+.1f} sd of this turbine's normal scatter)")
    L.append("")
    L.append("RESIDUAL TREND (mean °C per day, relative to now)")
    for back, v in trend:
        L.append(f"  {('now' if back == 0 else f'-{back:>2}d'):>5}  {v:+.2f}")
    L.append("")
    L.append("OPERATING CONTEXT (means over the window)")
    for k, v in ctx.items():
        L.append(f"  {k:<34}{v:>8.1f}")
    L.append("")
    L.append(f"FORCED OUTAGES ON THIS TURBINE (last 90 days)")
    if len(ev):
        for e in ev.itertuples():
            L.append(f"  {e.start:%Y-%m-%d}  {e.Message}")
    else:
        L.append("  none logged")
    L.append("")
    L.append("KNOWN LIMITATIONS OF THIS BRIEF")
    L.append("  - SCADA only. No vibration or oil-analysis data is available.")
    L.append("  - 10-minute averages: nothing faster than ~20 minutes is visible.")
    L.append("  - The residual is a deviation, not a diagnosis. It says the")
    L.append("    component is off its expected value, not why.")
    return "\n".join(L)


def write_assessment(brief):
    """Optional: hand the brief to Claude for a written assessment."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic")

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        messages=[{"role": "user",
                   "content": f"{brief}\n\nWrite the assessment."}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/kelmarsh")
    p.add_argument("--ckpt", default="runs/genbrg_static/run.pt")
    p.add_argument("--turbine", type=int, default=1)
    p.add_argument("--test-year", type=int, default=2016)
    p.add_argument("--stride", type=int, default=144)
    p.add_argument("--at", default=None,
                   help="date to report on, e.g. 2016-08-14 (default: latest)")
    p.add_argument("--llm", action="store_true",
                   help="also send the brief to Claude for a written assessment")
    a = p.parse_args()

    brief = build_brief(a)
    print(brief)
    if a.llm:
        print("\n" + "=" * 70 + "\nASSESSMENT\n" + "=" * 70)
        print(write_assessment(brief))
