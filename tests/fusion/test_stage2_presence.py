"""Stage 2 -- presence: an absent modality must not reach the output.

This is the correctness property the whole design rests on. If a masked block
leaks, then "runs without a couple of them" is a lie that no shape test would
ever catch: the tensors would all be the right size and the numbers would be
quietly wrong, contaminated by a sensor the turbine does not have.

Two code paths produce an absent block and they must agree:
  whole batch absent -> the encoder is skipped entirely
  some samples absent -> the encoder runs, those rows are overwritten
"""

import itertools

import torch

from multimodal_fusion import MultiModalFusion, StubEncoder
from helpers import banner, report

D, K = 128, 32
NAMES = ["scada", "vibration", "acoustic", "rgbd"]


def _fusion(seed=0):
    torch.manual_seed(seed)
    return MultiModalFusion(
        {n: StubEncoder(D, n_tokens=16 + 8 * i, n_channels=3)
         for i, n in enumerate(NAMES)},
        d_model=D, n_latents=K, modality_dropout=0.0,
    ).eval()


def _inputs(B=2, L=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {n: {"x": torch.randn(B, 3, L, generator=g),
                "mask": torch.ones(B, 3)} for n in NAMES}


def test():
    banner("fusion stage 2: presence masking")
    f = _fusion()
    B = 2

    # --- an absent modality's DATA cannot move the output -------------------
    # sample 0 has acoustic, sample 1 does not. Replacing sample 1's acoustic
    # input with garbage must change nothing at all, for either sample.
    present = torch.tensor([[1., 1., 1., 1.],
                            [1., 1., 0., 1.]])
    base = _inputs(B, seed=1)
    with torch.no_grad():
        a = f(base, present=present)

    poisoned = {n: {k: v.clone() for k, v in d.items()} for n, d in base.items()}
    poisoned["acoustic"]["x"][1] += 1000.0          # only the ABSENT sample
    with torch.no_grad():
        b = f(poisoned, present=present)

    drift = (a - b).abs().max().item()
    report("absent sample ignores its own data", drift, drift == 0.0)

    # and it must not disturb the sample that DOES have acoustic
    report("present sample unaffected", (a[0] - b[0]).abs().max().item(),
           (a[0] - b[0]).abs().max().item() == 0.0)

    # --- the two absence code paths must agree -----------------------------
    # path A: acoustic off for the whole batch -> encoder never runs
    # path B: acoustic off for sample 1 only    -> encoder runs, row overwritten
    off_all = torch.tensor([[1., 1., 0., 1.], [1., 1., 0., 1.]])
    with torch.no_grad():
        pa = f(base, present=off_all)[1]
        pb = f(base, present=present)[1]
    gap = (pa - pb).abs().max().item()
    report("skipped encoder == masked rows", gap, gap < 1e-5)

    # dropping the key entirely is a third way to say absent, same answer
    no_key = {n: d for n, d in base.items() if n != "acoustic"}
    with torch.no_grad():
        pc = f(no_key, present=present)[1]
    gap2 = (pa - pc).abs().max().item()
    report("missing input key == absent", gap2, gap2 < 1e-5)

    # a modality with no input tensors stays absent even if present says 1
    with torch.no_grad():
        pd_ = f(no_key, present=torch.ones(B, 4))[1]
    gap3 = (pa - pd_).abs().max().item()
    report("present=1 cannot conjure missing input", gap3, gap3 < 1e-5)

    # --- no cross-sample leakage (this is what catches a stray BatchNorm) ---
    full = torch.ones(B, 4)
    other = {n: {k: v.clone() for k, v in d.items()} for n, d in base.items()}
    other["vibration"]["x"][0] += 500.0             # perturb sample 0 only
    with torch.no_grad():
        c = f(base, present=full)
        d_ = f(other, present=full)
    leak = (c[1] - d_[1]).abs().max().item()
    report("sample 1 unaffected by sample 0", leak, leak < 1e-5)

    # --- every non-empty subset stays finite -------------------------------
    worst, n_sub = 0.0, 0
    outs = {}
    for r in range(1, 5):
        for combo in itertools.combinations(range(4), r):
            p = torch.zeros(B, 4)
            p[:, list(combo)] = 1.0
            with torch.no_grad():
                out = f(base, present=p)
            assert bool(torch.isfinite(out).all()), f"non-finite for {combo}"
            worst = max(worst, out.abs().max().item())
            outs[combo] = out
            n_sub += 1
    report("all 15 subsets finite", float(n_sub), n_sub == 15, fmt="{:.0f}")
    report("no magnitude blow-up", worst, worst < 1e3, fmt="{:.2f}")

    # the model must actually RESPOND to which modalities are present --
    # if every subset gave the same answer, masking would be doing nothing
    solo = [outs[(i,)] for i in range(4)]
    spread = min((solo[i] - solo[j]).abs().max().item()
                 for i in range(4) for j in range(i + 1, 4))
    report("different subsets -> different states", spread, spread > 1e-3,
           fmt="{:.4f}")

    # --- guards ------------------------------------------------------------
    try:
        f(base, present=torch.zeros(B, 4))
        report("rejects a sample with no modalities", 0.0, False)
    except AssertionError:
        report("rejects a sample with no modalities", 1.0, True, fmt="{:.0f}")

    try:
        f(base, present=torch.ones(B, 3))
        report("rejects wrong present shape", 0.0, False)
    except AssertionError:
        report("rejects wrong present shape", 1.0, True, fmt="{:.0f}")


if __name__ == "__main__":
    test()
