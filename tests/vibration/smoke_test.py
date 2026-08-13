"""End-to-end smoke test on synthetic vibration data.

The stage tests verify the FORWARD pass. This verifies the model can actually
LEARN -- gradients reach every parameter, and a head can recover a fault
signature from the tokens. An encoder can pass every shape, causality and DSP
check and still be untrainable.

WHAT THE SIGNAL IS, AND WHY

Both classes carry a 5 kHz resonance on the gearbox channels with IDENTICAL
energy. The only difference is that the degraded windows have theirs amplitude-
modulated at a bearing defect rate near 160 Hz, with the modulation rescaled so
broadband RMS matches to ~4e-4.

That matching is the point. It means:
  - the level token cannot separate the classes (same RMS)
  - the raw spectrogram plane cannot either (same energy at 5 kHz)
  - only the ENVELOPE plane can, because modulation is the sole difference

So this does not just check that something trains. It checks that the envelope
demodulation branch carries information a head can use -- closing the loop with
vibration stage 1, which proves only that the frontend REPRESENTS the fault.

Run this before spending any cluster time:

    python3 tests/vibration/smoke_test.py  (from anywhere; no PYTHONPATH needed)
"""

import math
import os
import sys
import time

# tests/vibration, tests, and the repo root, so `helpers` and the encoder
# modules resolve no matter which directory this is invoked from
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, _TESTS, os.path.dirname(_TESTS)]

import torch
import torch.nn as nn

from models.vibration_encoder_2dconv import VibrationConv2dEncoder
from models.scada_encoder_tcn import PerceiverResampler
from helpers import banner, report

FS = 25600.0
L = 25600
DEPTH = 0.8
NORM = math.sqrt(1 + DEPTH ** 2 / 2)      # RMS of (1 + DEPTH*cos), for matching


# ---------------------------------------------------------------------------
# synthetic vibration with a LEARNABLE fault signature in it
# ---------------------------------------------------------------------------
def make_batch(B=4, C=8, degraded=None, seed=None):
    """Fake CMS snapshots where degraded windows carry a modulated resonance.

    A pure-noise batch would let a broken model 'pass' by predicting the mean,
    and an amplitude difference would let it pass by reading the level token.
    Neither is available here.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    t = torch.arange(L) / FS

    x = torch.randn(B, C, L, generator=g) * 0.5
    # shaft harmonics and gear mesh, present in BOTH classes as background
    for f in (25.0, 50.0, 480.0):
        x += 0.3 * torch.cos(2 * math.pi * f * t)[None, None, :]

    if degraded is None:
        degraded = torch.randint(0, 2, (B,), generator=g).float()

    # defect rate jitters +/-5% because shaft speed is never exactly constant;
    # without it the model can memorise one filterbank band
    f_def = 160.0 * (1 + 0.1 * (torch.rand(B, generator=g) - 0.5))
    carrier = torch.cos(2 * math.pi * 5000.0 * t)[None, :]
    env = 1 + DEPTH * torch.cos(2 * math.pi * f_def[:, None] * t[None, :])
    # healthy gets a bare carrier; degraded gets the modulated one, rescaled to
    # the same RMS so total energy is identical
    env = torch.where(degraded[:, None] > 0, env / NORM, torch.ones_like(env))

    x[:, 2] += 3.0 * env * carrier            # nearest accelerometer
    x[:, 3] += 1.5 * env * carrier            # weaker, one bearing away

    return x, torch.ones(B, C), degraded


class Probe(nn.Module):
    """Encoder + resampler + a linear head, i.e. the real training shape."""

    def __init__(self, d=128, C=8, n_latents=32):
        super().__init__()
        self.enc = VibrationConv2dEncoder(d_model=d, n_channels=C)
        self.resampler = PerceiverResampler(d_model=d, n_latents=n_latents)
        # Pooling leaves every dim a DC offset ~40x its per-window spread --
        # the resampler's shared latents dominate the residual -- which strands
        # the head at chance. Needs per-dimension normalization from CURRENT
        # statistics: LayerNorm normalizes the wrong axis (across features
        # within a token, not across tokens within a feature), and running stats
        # lag the co-training encoder. Before pooling, so B=1 still works.
        # The SCADA probe carries the identical fix for the identical reason.
        self.norm = nn.BatchNorm1d(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x, mask):
        z = self.resampler(self.enc(x, mask))       # (B, n_latents, d)
        B, S, d = z.shape
        z = self.norm(z.reshape(B * S, d)).reshape(B, S, d)
        return self.head(z.mean(dim=1)).squeeze(-1)


def _train(model, steps, lr=1e-3, seed0=100, B=4, log_every=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    loss = None
    for step in range(steps):
        x, m, y = make_batch(B=B, seed=seed0 + step)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x, m), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:3d}  loss {loss.item():.4f}")
    return loss.item()


# ---------------------------------------------------------------------------
def test_signal_is_honest():
    banner("smoke 0: the task cannot be cheated")
    torch.manual_seed(0)
    x, m, y = make_batch(B=64, seed=0)

    # if the classes differed in amplitude, the level token alone would solve
    # the task and the envelope branch would never have to work
    for ch in (2, 3):
        rms = x[:, ch].pow(2).mean(-1).sqrt()
        gap = (rms[y == 1].mean() - rms[y == 0].mean()).abs().item()
        report(f"ch{ch} RMS matched across classes", gap, gap < 0.01,
               fmt="{:.5f}")

    # both classes must be present, or accuracy is meaningless
    frac = y.mean().item()
    report("classes roughly balanced", frac, 0.3 < frac < 0.7, fmt="{:.2f}")


def test_gradient_flow():
    banner("smoke 1: gradient reaches every parameter")
    torch.manual_seed(0)
    model = Probe()
    x, m, y = make_batch(B=4, seed=0)

    nn.functional.binary_cross_entropy_with_logits(model(x, m), y).backward()

    none_grad, dead = [], []
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
    report("every parameter has a gradient", len(none_grad), not none_grad,
           fmt="{:.0f}")
    report("no parameter is dead", len(dead), not dead, fmt="{:.0f}")

    gn = torch.cat([p.grad.flatten() for p in model.parameters()
                    if p.grad is not None]).norm().item()
    print(f"  global grad norm: {gn:.3f}")
    report("grad norm is finite and non-trivial", gn, 1e-6 < gn < 1e5,
           fmt="{:.3f}")


def test_overfit_one_batch():
    banner("smoke 2: overfit a single batch")
    torch.manual_seed(0)
    model = Probe()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    x, m, y = make_batch(B=4, seed=1)

    model.train()
    first = None
    for step in range(60):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x, m), y)
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
        acc = ((model(x, m) > 0).float() == y).float().mean().item()

    report("loss decreased", first - loss.item(), loss.item() < first * 0.5,
           fmt="{:.4f}")
    report("fits the batch (acc)", acc, acc >= 0.99, fmt="{:.2f}")


def test_generalizes_a_little():
    banner("smoke 3: learns the signature, not just the batch")
    torch.manual_seed(0)
    model = Probe()
    # loss spikes mid-run are normal here (small batch + BatchNorm) and the run
    # recovers, so the assertions are on the END state, not on monotonicity
    _train(model, steps=80, log_every=30)

    model.eval()
    accs = []
    with torch.no_grad():
        for s in range(9000, 9005):          # unseen seeds
            x, m, y = make_batch(B=4, seed=s)
            accs.append((((model(x, m) > 0).float() == y).float().mean().item()))
    acc = sum(accs) / len(accs)
    print(f"  held-out accuracy over 5 fresh batches: {acc:.3f}")
    # NOTE: this sits at 1.000 across seeds, so it has no headroom -- it catches
    # a BROKEN encoder, not a degraded one. To make it a sharper regression
    # guard, drop DEPTH toward 0.3 and move the threshold up near the new score.
    report("beats chance on unseen data", acc, acc > 0.75, fmt="{:.3f}")


def test_timing():
    banner("smoke 4: speed and memory")
    torch.manual_seed(0)
    model = Probe()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for B in (1, 2, 4):
        x, m, y = make_batch(B=B, seed=7)
        for _ in range(2):                    # warmup
            loss = nn.functional.binary_cross_entropy_with_logits(model(x, m), y)
            opt.zero_grad(); loss.backward(); opt.step()

        t0 = time.time()
        for _ in range(3):
            loss = nn.functional.binary_cross_entropy_with_logits(model(x, m), y)
            opt.zero_grad(); loss.backward(); opt.step()
        per_step = (time.time() - t0) / 3

        model.eval()
        with torch.no_grad():
            t0 = time.time()
            for _ in range(3):
                model(x, m)
            inf = (time.time() - t0) / 3
        model.train()

        print(f"  B={B:3d}  train {per_step*1000:8.1f} ms/step "
              f"({B/per_step:6.1f} snap/s)   inference {inf*1000:7.1f} ms")

    n = sum(p.numel() for p in model.parameters())
    print(f"  total parameters: {n:,}  ({n*4/1e6:.1f} MB in fp32)")


def test_missing_data_robustness():
    banner("smoke 5: survives dead accelerometers")
    torch.manual_seed(0)
    model = Probe()
    # a TRAINED model, deliberately. Vibration stage 3 already checks that a
    # fresh encoder stays finite with channels masked out -- but a fresh encoder
    # emits near-constant tokens, so that check is close to trivial. What
    # matters in deployment is whether a model whose BatchNorm running stats
    # were estimated on mostly-live sensors still behaves when one dies.
    _train(model, steps=30, seed0=200)

    model.eval()
    x, m, y = make_batch(B=4, seed=3)
    for n_dead in (0, 1, 4, 7, 8):
        mm = m.clone()
        if n_dead:
            mm[:, :n_dead] = 0.0              # encoder zeroes the signal itself
        with torch.no_grad():
            out = model(x, mm)
        finite = bool(torch.isfinite(out).all())
        print(f"  {n_dead}/8 dead -> finite={finite}  "
              f"range [{out.min():.2f}, {out.max():.2f}]")
        assert finite, f"non-finite output with {n_dead} dead channels"

    # losing the two channels that CARRY the signature must not crash, and the
    # model should stop being confident rather than assert the wrong class
    mm = m.clone()
    mm[:, 2:4] = 0.0
    with torch.no_grad():
        blind = model(x, mm)
    report("blind to the signal channels, still finite",
           float(torch.isfinite(blind).all()),
           bool(torch.isfinite(blind).all()), fmt="{:.0f}")

    # an entirely silent snapshot must not produce NaN anywhere in the frontend
    with torch.no_grad():
        dead = model(torch.zeros(2, 8, L), torch.zeros(2, 8))
    report("all-silent snapshot is finite", float(torch.isfinite(dead).all()),
           bool(torch.isfinite(dead).all()), fmt="{:.0f}")


def test():
    """Full smoke test at production size.

    COST -- measured on an M3 Pro CPU at d_model=128, C=8, L=25600:
        B=1  36 ms/step      B=2  89 ms/step      B=4  154 ms/step
        peak RSS 0.39 GB          0.42 GB              0.56 GB
    Add roughly 25 ms/step for make_batch, which synthesises B*C*25600 samples
    every call and is not part of the model.

    That is ~27x faster and ~6x lighter than the SCADA trunk (4.2 s/step,
    ~3.4 GB at B=4), because folding gives only B*C=32 sequences and the conv
    stack runs on a 64x17 grid rather than 600 full-length timesteps. The whole
    suite finishes in about 31 seconds, so unlike the SCADA smoke test it is
    cheap enough to run on every change.
    """
    test_signal_is_honest()
    test_gradient_flow()
    test_overfit_one_batch()
    test_generalizes_a_little()
    test_timing()
    test_missing_data_robustness()
    print("\nsmoke test complete\n")


if __name__ == "__main__":
    test()
