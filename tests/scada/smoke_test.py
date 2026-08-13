"""End-to-end smoke test on synthetic data.

The stage tests verify the FORWARD pass. This verifies the model can actually
LEARN -- gradients reach every parameter, and the thing can fit a signal. A
model can pass every shape and causality check and still be untrainable.

Run this before spending any cluster time:

    python3 tests/scada/smoke_test.py     (from anywhere; no PYTHONPATH needed)
"""

import os
import sys
import time

# tests/scada, tests, and the repo root, so `helpers` and the encoder modules
# resolve no matter which directory this is invoked from
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, _TESTS, os.path.dirname(_TESTS)]

import torch
import torch.nn as nn

from scada_encoder_tcn import ScadaTCNEncoder, PerceiverResampler
from helpers import banner, report


# ---------------------------------------------------------------------------
# synthetic SCADA with a LEARNABLE signal in it
# ---------------------------------------------------------------------------
def make_batch(B=4, C=20, T=600, n_static=7, degraded=None, seed=None):
    """Fake SCADA where channel 11 (gearbox temp) drifts upward when degraded.

    A pure-noise batch would let a broken model 'pass' by predicting the mean,
    so the signal has to be something a working model can find and a broken
    one cannot.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    t = torch.linspace(0, 1, T)

    x = torch.randn(B, C, T, generator=g) * 0.3
    # slow diurnal-ish base on every channel
    x += torch.sin(2 * torch.pi * t * 3)[None, None, :]

    if degraded is None:
        degraded = torch.randint(0, 2, (B,), generator=g).float()
    # gearbox temp drifts up over the window when degraded
    x[:, 11] += degraded[:, None] * t[None, :] * 4.0
    # vibration channel gets noisier when degraded
    x[:, 19] += degraded[:, None] * torch.randn(B, T, generator=g) * 1.5

    mask = (torch.rand(B, C, T, generator=g) > 0.03).float()
    x = x * mask                       # missing entries stored as 0

    cat = torch.stack([
        torch.randint(0, 12, (B,), generator=g),
        torch.randint(0, 256, (B,), generator=g),
        torch.randint(0, 2, (B,), generator=g),
    ], dim=1)
    static = torch.randn(B, n_static, generator=g)
    return x, mask, cat, static, degraded


class Probe(nn.Module):
    """Encoder + resampler + a linear head, i.e. the real training shape."""

    def __init__(self, d=128, C=20, T=600, n_latents=32):
        super().__init__()
        self.enc = ScadaTCNEncoder(d_model=d, n_channels=C, context_len=T)
        self.resampler = PerceiverResampler(d_model=d, n_latents=n_latents)
        # Pooling leaves every dim a DC offset ~49x its per-window spread, which
        # strands the head at chance. Needs per-dimension normalization from
        # CURRENT statistics: LayerNorm normalizes the wrong axis, running stats
        # lag the co-training encoder and diverge. Before pooling so B=1 works.
        self.norm = nn.BatchNorm1d(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x, mask, cat, static):
        seq = self.enc(x, mask, cat, static)
        z = self.resampler(seq)             # (B, n_latents, d)
        B, L, d = z.shape
        z = self.norm(z.reshape(B * L, d)).reshape(B, L, d)
        return self.head(z.mean(dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
def test_gradient_flow():
    banner("smoke 1: gradient reaches every parameter")
    torch.manual_seed(0)
    model = Probe()
    x, m, cat, st, y = make_batch(B=4, seed=0)

    loss = nn.functional.binary_cross_entropy_with_logits(model(x, m, cat, st), y)
    loss.backward()

    dead, none_grad = [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            none_grad.append(name)
        elif p.grad.abs().max().item() == 0.0:
            dead.append(name)

    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    if none_grad:
        print(f"  no grad at all: {none_grad}")
    if dead:
        print(f"  zero grad: {dead}")
    report("every parameter has a gradient", len(none_grad), len(none_grad) == 0,
           fmt="{:.0f}")
    report("no parameter is dead", len(dead), len(dead) == 0, fmt="{:.0f}")

    # gradient magnitude should be sane -- not exploded, not vanished
    gn = torch.cat([p.grad.flatten() for p in model.parameters()
                    if p.grad is not None]).norm().item()
    print(f"  global grad norm: {gn:.3f}")
    report("grad norm is finite and non-trivial", gn, 1e-6 < gn < 1e5, fmt="{:.3f}")


def test_overfit_one_batch():
    banner("smoke 2: overfit a single batch")
    torch.manual_seed(0)
    model = Probe()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    x, m, cat, st, y = make_batch(B=4, seed=1)

    model.train()
    first = None
    for step in range(60):
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(x, m, cat, st), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first is None:
            first = loss.item()
        if step % 15 == 0 or step == 59:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = (model(x, m, cat, st) > 0).float()
        acc = (pred == y).float().mean().item()

    report("loss decreased", first - loss.item(), loss.item() < first * 0.5,
           fmt="{:.4f}")
    report("fits the batch (acc)", acc, acc >= 0.99, fmt="{:.2f}")


def test_generalizes_a_little():
    banner("smoke 3: learns the signal, not just the batch")
    torch.manual_seed(0)
    model = Probe()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step in range(80):
        x, m, cat, st, y = make_batch(B=4, seed=100 + step)
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(x, m, cat, st), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 30 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    model.eval()
    accs = []
    with torch.no_grad():
        for s in range(9000, 9005):          # unseen seeds
            x, m, cat, st, y = make_batch(B=4, seed=s)
            accs.append(((model(x, m, cat, st) > 0).float() == y).float().mean().item())
    acc = sum(accs) / len(accs)
    print(f"  held-out accuracy over 5 fresh batches: {acc:.3f}")
    report("beats chance on unseen data", acc, acc > 0.75, fmt="{:.3f}")


def test_timing():
    banner("smoke 4: speed and memory")
    torch.manual_seed(0)
    model = Probe()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for B in (1, 2, 4):
        x, m, cat, st, y = make_batch(B=B, seed=7)
        for _ in range(2):                    # warmup
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(x, m, cat, st), y)
            opt.zero_grad(); loss.backward(); opt.step()

        t0 = time.time()
        for _ in range(3):
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(x, m, cat, st), y)
            opt.zero_grad(); loss.backward(); opt.step()
        per_step = (time.time() - t0) / 3

        model.eval()
        with torch.no_grad():
            t0 = time.time()
            for _ in range(3):
                model(x, m, cat, st)
            inf = (time.time() - t0) / 3
        model.train()

        print(f"  B={B:3d}  train {per_step*1000:8.1f} ms/step "
              f"({B/per_step:6.1f} win/s)   inference {inf*1000:7.1f} ms")

    n = sum(p.numel() for p in model.parameters())
    print(f"  total parameters: {n:,}  ({n*4/1e6:.1f} MB in fp32)")


def test_missing_data_robustness():
    banner("smoke 5: survives heavy missing data")
    torch.manual_seed(0)
    model = Probe().eval()
    x, m, cat, st, y = make_batch(B=4, seed=3)

    for frac in (0.0, 0.3, 0.7, 0.95):
        mm = (torch.rand_like(m) > frac).float()
        with torch.no_grad():
            out = model(x * mm, mm, cat, st)
        finite = bool(torch.isfinite(out).all())
        print(f"  {frac*100:4.0f}% missing -> finite={finite}  "
              f"range [{out.min():.2f}, {out.max():.2f}]")
        assert finite, f"non-finite output at {frac} missing"

    # a fully dead channel must not produce NaN
    mm = m.clone()
    mm[:, 7] = 0.0
    with torch.no_grad():
        out = model(x * mm, mm, cat, st)
    report("dead channel stays finite", float(torch.isfinite(out).all()),
           bool(torch.isfinite(out).all()), fmt="{:.0f}")


def test():
    """Full smoke test at production size.

    MEMORY WARNING -- measured at d_model=128, C=20, T=600, 8 blocks:
        B=2  -> ~1.9 GB      B=4  -> ~3.4 GB
    i.e. roughly 0.8 GB per window. The cost is the trunk: folding gives B*C
    sequences, and all 8 residual blocks retain activations for backprop.
    B=32 would need ~25 GB, more than most single GPUs. Plan cluster runs
    around gradient accumulation with a small per-step batch, or reduce
    d_model / block count.
    """
    test_gradient_flow()
    test_overfit_one_batch()
    test_generalizes_a_little()
    test_timing()
    test_missing_data_robustness()
    print("\nsmoke test complete\n")


if __name__ == "__main__":
    test()