"""End-to-end smoke test for the multimodal fusion stack.

The three stage tests verify the FORWARD pass: routing (stage 1), presence
masking (stage 2), modality dropout (stage 3). All of them are mechanical --
they would pass on a model that fuses nothing, as long as it fused nothing
with the right tensor shapes.

This file asks the question those cannot: does putting two modalities in one
health vector actually BUY anything?

WHAT THE SIGNAL IS, AND WHY

A turbine is degraded if EITHER of two independent faults is present:

    degraded  <=>  power curve sags  OR  bearing defect        each w.p. 0.3

SCADA sees only the sag. Vibration sees only the defect. So:

    majority class      0.510      guess the common label
    one modality        0.790      see one fault, guess on the other
    both modalities     1.000

That ladder is the whole point. A single modality is capped at 0.790 by
information theory, not by how good its encoder is -- no amount of training
lets vibration see a power sag. So a fused model scoring above 0.790 has
provably combined both, and it cannot have got there by one branch getting
lucky. That is a much stronger claim than "accuracy went up".

It is also the honest argument for building this at all: two different faults,
each visible to a different sensor, neither sensor sufficient alone.

WHY THE PAYOFF TEST PROBES FROZEN FEATURES

Training this end to end to convergence takes ~3000 steps (measured: a linear
head on CACHED features needs that many, and joint training is strictly
harder). At ~500 ms/step that is 25 minutes, which is not a smoke test.

So the ablation fits a closed-form ridge probe on features from a FROZEN,
untrained stack. That is not a weaker test -- it is a sharper one. It asks
whether each modality's information SURVIVES the encoder, resampler, fusion
blocks and pooling, which is exactly the wiring question, with no optimizer in
the way to blame. It is deterministic: no learning rate, no step count, nothing
that can make the test flaky on a bad seed. Trainability is covered separately
by smoke 1 and smoke 2.

Run this before spending any cluster time.
"""

import math
import time

import torch
import torch.nn as nn

from multimodal_fusion import MultiModalFusion, StubEncoder
from scada_encoder_tcn import ScadaTCNEncoder
from vibration_encoder_2dconv import VibrationConv2dEncoder
from helpers import banner, report

FS = 25600.0
L = 25600
DEPTH = 0.8
NORM = math.sqrt(1 + DEPTH ** 2 / 2)      # RMS of (1 + DEPTH*cos), for matching

T_SCADA, C_SCADA = 120, 6
C_VIB = 8
D, K = 128, 32

P_FAULT = 0.3                              # each fault fires independently
SAG = 0.8                                  # power drop, as a fraction
TEMP = 4.0                                 # nacelle temperature ramp, in sigma

P_DEGRADED = 1 - (1 - P_FAULT) ** 2                 # 0.510
CEIL_MAJORITY = max(P_DEGRADED, 1 - P_DEGRADED)     # 0.510
CEIL_SINGLE = P_FAULT + (1 - P_FAULT) ** 2          # 0.790
NAMES = ["scada", "vibration", "acoustic", "rgbd"]


# ---------------------------------------------------------------------------
# synthetic data: one fault per modality, independent
# ---------------------------------------------------------------------------
def make_scada(B, sag, g):
    """Power curve with a mid-window sag and a trailing temperature ramp.

    The sag has to be a STEP, not a uniform scaling. RevIN normalizes by
    whole-window statistics, so a fault that scales the entire window is
    erased before the trunk ever sees it -- measured at 1.39e-05 for a global
    sag versus 1.81 for this step.

    The temperature ramp is a second, CORRELATED view of the same fault. With
    the sag alone the untrained stack read it at 0.604 against the 0.790
    ceiling, which left fusion nothing to add over vibration alone (+0.026).
    """
    t = torch.linspace(0, 1, T_SCADA)
    phase = torch.rand(B, 1, generator=g) * 6.283
    # wind speed: a slow gust cycle, phase-randomised so no two windows align
    u = 8.0 + 1.2 * torch.sin(2 * math.pi * 2 * t[None] + phase) \
        + 0.3 * torch.randn(B, T_SCADA, generator=g)
    power = (u / 12.0) ** 3                              # cube law
    step = (t > 0.5).float()[None]
    power = power * (1.0 - SAG * sag[:, None] * step)

    rest = torch.randn(B, C_SCADA - 2, T_SCADA, generator=g) * 0.5
    ramp = torch.clamp((t - 0.5) * 2, min=0.0)[None]
    rest[:, 0] = rest[:, 0] + TEMP * sag[:, None] * ramp

    x = torch.cat([u[:, None], power[:, None], rest], dim=1)
    return x, torch.ones(B, C_SCADA, T_SCADA)


def make_vib(B, defect, g):
    """Gearbox resonance, amplitude-modulated at a bearing defect rate.

    Lifted from the vibration smoke test, including the RMS matching: both
    classes carry identical energy at 5 kHz, so only the envelope plane can
    separate them. The level token cannot.
    """
    t = torch.arange(L) / FS
    x = torch.randn(B, C_VIB, L, generator=g) * 0.5
    for f in (25.0, 50.0, 480.0):                        # shaft, gear mesh
        x += 0.3 * torch.cos(2 * math.pi * f * t)[None, None, :]

    f_def = 160.0 * (1 + 0.1 * (torch.rand(B, generator=g) - 0.5))
    carrier = torch.cos(2 * math.pi * 5000.0 * t)[None, :]
    env = 1 + DEPTH * torch.cos(2 * math.pi * f_def[:, None] * t[None, :])
    env = torch.where(defect[:, None] > 0, env / NORM, torch.ones_like(env))

    x[:, 2] += 3.0 * env * carrier                       # nearest accelerometer
    x[:, 3] += 1.5 * env * carrier                       # one bearing away
    return x, torch.ones(B, C_VIB)


def make_batch(B=4, seed=None):
    """Returns (inputs, y, sag, defect). Acoustic and rgbd are pure noise.

    They are here because the fusion registry holds four modalities and the
    test should exercise all of them -- and because a modality carrying NO
    information is a useful control: if the probe scores above chance on
    those two alone, something is leaking.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    sag = (torch.rand(B, generator=g) < P_FAULT).float()
    defect = (torch.rand(B, generator=g) < P_FAULT).float()

    xs, ms = make_scada(B, sag, g)
    xv, mv = make_vib(B, defect, g)
    inputs = {
        "scada": {"x": xs, "mask": ms},
        "vibration": {"x": xv, "mask": mv},
        "acoustic": {"x": torch.randn(B, 4, 4096, generator=g),
                     "mask": torch.ones(B, 4)},
        "rgbd": {"x": torch.randn(B, 3, 2048, generator=g),
                 "mask": torch.ones(B, 3)},
    }
    return inputs, torch.maximum(sag, defect), sag, defect


def build_fusion(modality_dropout=0.0, seed=0):
    torch.manual_seed(seed)
    return MultiModalFusion(
        {
            "scada": ScadaTCNEncoder(d_model=D, n_channels=C_SCADA,
                                     context_len=T_SCADA,
                                     dilations=(1, 2, 4, 8, 16)),
            "vibration": VibrationConv2dEncoder(d_model=D, n_channels=C_VIB),
            "acoustic": StubEncoder(D, n_tokens=32, n_channels=4),
            "rgbd": StubEncoder(D, n_tokens=16, n_channels=3),
        },
        d_model=D, n_latents=K, modality_dropout=modality_dropout,
    )


class Probe(nn.Module):
    """Fusion + a linear head, i.e. the real training shape.

    BatchNorm for the same reason both encoder probes carry it: pooling leaves
    every dimension with a DC offset many times its per-window spread (measured
    21.5x here, and inter-sample cosine similarity 0.9987), which strands a
    linear head at chance. LayerNorm normalizes the wrong axis.
    """

    def __init__(self, d=D, modality_dropout=0.0, seed=0):
        super().__init__()
        self.fusion = build_fusion(modality_dropout, seed)
        self.norm = nn.BatchNorm1d(d)
        self.head = nn.Linear(d, 1)

    def forward(self, inputs, present=None):
        h = self.fusion(inputs, present=present)
        return self.head(self.norm(h)).squeeze(-1)


def _present(pattern, B):
    return torch.tensor(pattern, dtype=torch.float32).expand(B, -1).clone()


PATTERNS = {
    "both":       [1., 1., 0., 0.],
    "scada only": [1., 0., 0., 0.],
    "vib only":   [0., 1., 0., 0.],
    "noise only": [0., 0., 1., 1.],     # control: must land at chance
}


def ridge(Htr, ytr, Hte, yte, lam=1e-2):
    """Closed-form probe. Deterministic -- no lr, no step count, no seed."""
    mu = Htr.mean(0)
    X = torch.cat([Htr - mu, torch.ones(len(Htr), 1)], dim=1)
    w = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]),
                           X.T @ (ytr[:, None] - ytr.mean()))
    Xte = torch.cat([Hte - mu, torch.ones(len(Hte), 1)], dim=1)
    return (((Xte @ w).squeeze(-1) > 0).float() == yte).float().mean().item()


# ---------------------------------------------------------------------------
def test_signal_is_honest():
    banner("smoke 0: the task cannot be cheated")
    torch.manual_seed(0)
    _, y, sag, defect = make_batch(B=2048, seed=0)

    frac = y.mean().item()
    report("classes roughly balanced", frac, 0.35 < frac < 0.65, fmt="{:.3f}")
    report("matches P(degraded) theory", abs(frac - P_DEGRADED),
           abs(frac - P_DEGRADED) < 0.05, fmt="{:.3f}")

    # each fault ALONE is capped, and the cap is a property of the task rather
    # than of any encoder. This is what makes the fusion claim in smoke 3 hard.
    a_sag = (sag == y).float().mean().item()
    a_def = (defect == y).float().mean().item()
    a_or = (torch.maximum(sag, defect) == y).float().mean().item()
    report("sag alone hits the single ceiling", a_sag,
           abs(a_sag - CEIL_SINGLE) < 0.05, fmt="{:.3f}")
    report("defect alone hits it too", a_def,
           abs(a_def - CEIL_SINGLE) < 0.05, fmt="{:.3f}")
    report("both together are exact", a_or, a_or == 1.0, fmt="{:.3f}")

    # the two faults must be INDEPENDENT, or one modality could infer the other
    # and the single-modality ceiling would be a fiction
    corr = ((sag - sag.mean()) * (defect - defect.mean())).mean() / \
           (sag.std() * defect.std() + 1e-9)
    report("faults are independent", corr.abs().item(),
           corr.abs().item() < 0.08, fmt="{:.3f}")

    print(f"  ceilings: majority {CEIL_MAJORITY:.3f}   "
          f"one modality {CEIL_SINGLE:.3f}   both 1.000")


def test_gradient_flow():
    banner("smoke 1: gradient reaches every parameter")
    torch.manual_seed(0)
    model = Probe()
    inputs, y, _, _ = make_batch(B=4, seed=0)

    nn.functional.binary_cross_entropy_with_logits(model(inputs), y).backward()

    # Two parameter groups are SUPPOSED to be idle on this batch, and asserting
    # on them would be asserting that optional inputs are mandatory:
    # ScadaTCNEncoder.forward takes `categorical` and `static_feats` as optional
    # arguments and make_batch supplies neither. The SCADA suite, which does
    # supply them, is where those paths are covered.
    IDLE = ("encoders.scada.cat_embeds", "encoders.scada.static_proj")

    none_grad, dead, idle = [], [], []
    for name, p in model.named_parameters():
        if any(k in name for k in IDLE):
            idle.append(name)
        elif p.grad is None:
            none_grad.append(name)
        elif p.grad.abs().max().item() == 0.0:
            dead.append(name)

    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}  "
          f"({len(idle)} idle by design, see IDLE)")
    if none_grad:
        print(f"  no grad at all: {none_grad[:6]}")
    if dead:
        print(f"  zero grad: {dead[:6]}")
    report("every parameter has a gradient", len(none_grad), not none_grad,
           fmt="{:.0f}")
    report("no parameter is dead", len(dead), not dead, fmt="{:.0f}")

    # both REAL encoders must be pulling, not just one. If a modality's branch
    # were silently detached, the model would still train -- on one modality --
    # and every other check in this file except smoke 3 would still pass.
    for name in ("scada", "vibration"):
        gn = torch.cat([p.grad.flatten()
                        for p in model.fusion.encoders[name].parameters()
                        if p.grad is not None]).norm().item()
        report(f"{name} encoder receives gradient", gn, gn > 1e-8, fmt="{:.2e}")

    gn = torch.cat([p.grad.flatten() for p in model.parameters()
                    if p.grad is not None]).norm().item()
    print(f"  global grad norm: {gn:.3f}")
    report("grad norm is finite and non-trivial", gn, 1e-6 < gn < 1e5,
           fmt="{:.3f}")


def test_absent_modality_gets_no_gradient():
    banner("smoke 1b: an absent modality is not trained on")
    torch.manual_seed(0)
    model = Probe()
    inputs, y, _, _ = make_batch(B=4, seed=0)

    # acoustic off for the WHOLE batch: its encoder is skipped, so its
    # parameters must come back with no gradient at all. Stage 2 proves the
    # forward pass ignores it; this proves the backward pass does too. A leak
    # here would train an encoder on data the turbine does not have.
    present = _present([1., 1., 0., 1.], 4)
    nn.functional.binary_cross_entropy_with_logits(
        model(inputs, present=present), y).backward()

    off = [p.grad for p in model.fusion.encoders["acoustic"].parameters()]
    touched = sum(1 for g in off if g is not None and g.abs().max().item() > 0)
    report("absent encoder untouched by backward", float(touched),
           touched == 0, fmt="{:.0f}")

    # `missing` fills the absent block so the concatenation is well defined,
    # and `pad` then masks that block out of every attention. Those two facts
    # together mean no gradient can EVER reach it, under any presence pattern
    # -- which is why it is a buffer. If someone promotes it back to a
    # Parameter, they have added 512 weights the optimizer will carry forever
    # and never train, and this catches that.
    names = dict(model.fusion.named_parameters())
    report("missing is not a trainable parameter", float("missing" in names),
           "missing" not in names, fmt="{:.0f}")
    report("missing is a registered buffer",
           float("missing" in dict(model.fusion.named_buffers())),
           "missing" in dict(model.fusion.named_buffers()), fmt="{:.0f}")

    # the present ones still learn, or the check above would be trivially true
    on = torch.cat([p.grad.flatten()
                    for p in model.fusion.encoders["rgbd"].parameters()
                    if p.grad is not None]).norm().item()
    report("a present stub still gets gradient", on, on > 1e-8, fmt="{:.2e}")


def test_overfit_one_batch():
    banner("smoke 2: overfit a single batch")
    torch.manual_seed(0)
    model = Probe()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    inputs, y, _, _ = make_batch(B=8, seed=1)

    model.train()
    first = None
    for step in range(60):
        loss = nn.functional.binary_cross_entropy_with_logits(model(inputs), y)
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
        acc = ((model(inputs) > 0).float() == y).float().mean().item()

    report("loss decreased", first - loss.item(), loss.item() < first * 0.5,
           fmt="{:.4f}")
    report("fits the batch (acc)", acc, acc >= 0.99, fmt="{:.2f}")


def test_fusion_actually_pays(n_train=24, n_test=12, B=16):
    banner("smoke 3: fusion beats either modality alone")
    fusion = build_fusion(modality_dropout=0.0).eval()

    # generate each batch ONCE and push it through all four presence patterns.
    # Synthesising B*8*25600 vibration samples costs more than the forward pass
    # does, so regenerating per pattern would nearly double the runtime.
    feats = {name: [] for name in PATTERNS}
    labels = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_train + n_test):
            seed = i if i < n_train else 5000 + i
            inputs, y, _, _ = make_batch(B=B, seed=seed)
            labels.append(y)
            for name, pat in PATTERNS.items():
                feats[name].append(fusion(inputs, present=_present(pat, B)))
    Y = torch.cat(labels)
    n = n_train * B
    print(f"  cached {n} train / {n_test*B} test per pattern "
          f"in {time.time()-t0:.0f} s")

    acc = {}
    for name in PATTERNS:
        H = torch.cat(feats[name])
        acc[name] = ridge(H[:n], Y[:n], H[n:], Y[n:])
        print(f"  {name:<11} held-out {acc[name]:.3f}")

    # the control first: two modalities carrying nothing but noise must land at
    # chance. If this passes above the majority rate, the probe is reading
    # something it should not and every number below is suspect.
    report("noise-only modalities stay at chance", acc["noise only"],
           acc["noise only"] < CEIL_MAJORITY + 0.08, fmt="{:.3f}")

    # each modality on its own must carry real information
    report("scada alone beats majority", acc["scada only"],
           acc["scada only"] > CEIL_MAJORITY + 0.08, fmt="{:.3f}")
    report("vibration alone beats majority", acc["vib only"],
           acc["vib only"] > CEIL_MAJORITY + 0.08, fmt="{:.3f}")

    # THE POINT OF THIS FILE. 0.790 is the information-theoretic cap on any
    # single modality, so clearing it is proof the health vector carries both.
    report("both beat the single-modality CEILING", acc["both"],
           acc["both"] > CEIL_SINGLE, fmt="{:.3f}")

    gain = acc["both"] - max(acc["scada only"], acc["vib only"])
    report("gain over best single modality", gain, gain > 0.05, fmt="{:+.3f}")


def test_timing():
    banner("smoke 4: speed and memory")
    torch.manual_seed(0)
    model = Probe()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # B=1 trains n/a, infers fine, and the asymmetry is worth being precise
    # about: MultiModalFusion itself has no batch-coupled layer anywhere -- the
    # encoders use GroupNorm and LayerNorm exactly so a single turbine is a
    # valid forward pass. The BatchNorm that needs B>1 lives in this PROBE's
    # head, and only in training mode. Inference at B=1 is the deployment case
    # (one turbine, one window) and it works.
    for B in (1, 2, 4):
        inputs, y, _, _ = make_batch(B=B, seed=7)

        if B > 1:
            for _ in range(2):                    # warmup
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(inputs), y)
                opt.zero_grad(); loss.backward(); opt.step()

            t0 = time.time()
            for _ in range(3):
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(inputs), y)
                opt.zero_grad(); loss.backward(); opt.step()
            per_step = (time.time() - t0) / 3
            train = (f"train {per_step*1000:8.1f} ms/step "
                     f"({B/per_step:6.1f} turbine/s)")
        else:
            train = "train        -- (probe BatchNorm needs B>1)"

        model.eval()
        with torch.no_grad():
            t0 = time.time()
            for _ in range(3):
                model(inputs)
            inf = (time.time() - t0) / 3
        model.train()

        print(f"  B={B:3d}  {train}   inference {inf*1000:7.1f} ms")

    # the model proper, with no probe head, must be finite for a lone turbine
    fusion = build_fusion().eval()
    inputs, _, _, _ = make_batch(B=1, seed=7)
    with torch.no_grad():
        one = fusion(inputs)
    report("B=1 inference is finite", float(torch.isfinite(one).all()),
           bool(torch.isfinite(one).all()), fmt="{:.0f}")

    n = sum(p.numel() for p in model.parameters())
    print(f"  total parameters: {n:,}  ({n*4/1e6:.1f} MB in fp32)")


def test_degrades_gracefully():
    banner("smoke 5: a trained model survives losing a modality")
    torch.manual_seed(0)
    # trained WITH modality dropout, which is the deployment story: the model
    # has to have seen degraded inputs to behave on them. A fresh model would
    # pass this trivially because it emits near-constant output anyway.
    model = Probe(modality_dropout=0.15)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(20):
        inputs, y, _, _ = make_batch(B=4, seed=200 + step)
        loss = nn.functional.binary_cross_entropy_with_logits(model(inputs), y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    print(f"  trained 20 steps, final loss {loss.item():.4f}")

    model.eval()
    inputs, y, _, _ = make_batch(B=4, seed=3)
    for name, pat in PATTERNS.items():
        with torch.no_grad():
            out = model(inputs, present=_present(pat, 4))
        finite = bool(torch.isfinite(out).all())
        # spread ACROSS samples, printed rather than asserted. 20 steps is far
        # too few to escape the DC dominance documented in smoke 3, so this is
        # near zero here and that is expected -- it is a finiteness check, not
        # an accuracy one. If it were ever asserted it would be a flaky test.
        spread = (out.max() - out.min()).item()
        print(f"  {name:<11} finite={finite}  mean {out.mean():+.3f}  "
              f"spread across samples {spread:.2e}")
        assert finite, f"non-finite output for {name}"

    # the extreme case: one modality left, and it is a stub carrying noise
    with torch.no_grad():
        alone = model(inputs, present=_present([0., 0., 0., 1.], 4))
    report("one noise modality left, still finite",
           float(torch.isfinite(alone).all()),
           bool(torch.isfinite(alone).all()), fmt="{:.0f}")

    # and dropping a modality has to CHANGE the answer -- if it did not, the
    # model would be ignoring that modality entirely
    with torch.no_grad():
        full = model(inputs, present=_present([1., 1., 1., 1.], 4))
        no_vib = model(inputs, present=_present([1., 0., 1., 1.], 4))
    shift = (full - no_vib).abs().max().item()
    report("losing vibration moves the output", shift, shift > 1e-3,
           fmt="{:.4f}")


def test():
    """Full smoke test at production size.

    COST -- measured on an M3 Pro CPU at d_model=128, K=32, 4 modalities:
        B=1  ~340 ms/step     B=2  ~530 ms/step     B=4  ~900 ms/step
    dominated by the vibration encoder on 25600-sample windows and by
    make_batch, which synthesises B*8*25600 samples per call.

    smoke 3 is the expensive one at ~40 s: it caches 576 windows through four
    presence patterns. The whole file runs in roughly two minutes.

    NOT COVERED HERE: end-to-end generalization. A linear head on cached
    features needs ~3000 steps to converge, so joint training to convergence
    is a ~25 minute job. smoke 3 answers the same question -- is the
    information there -- in 40 deterministic seconds, and smoke 1 and 2 cover
    trainability. If the fusion stack is ever changed in a way that could hurt
    OPTIMIZATION rather than representation, that longer run is the check to
    write.
    """
    test_signal_is_honest()
    test_gradient_flow()
    test_absent_modality_gets_no_gradient()
    test_overfit_one_batch()
    test_fusion_actually_pays()
    test_timing()
    test_degrades_gracefully()
    print("\nsmoke test complete\n")


if __name__ == "__main__":
    test()
