"""Stage 3 -- modality dropout: force the model to build backup paths.

Trained only on complete turbines, a fusion head collapses onto whichever
stream is most informative. It has no reason not to: the backup path is never
exercised, so nothing penalises letting it rot. Then the first turbine without
vibration monitoring arrives and the model is worse than the SCADA-only
baseline it replaced.

Randomly hiding modalities during training is what prevents that. The rules it
has to obey are narrow, and all three failure modes are silent:
  drop too much  -> some sample has nothing left and attention softmaxes to NaN
  drop the wrong -> an ABSENT modality gets switched on, feeding it garbage
  drop at eval   -> validation numbers become noise and are not reproducible
"""

import torch

from models.multimodal_fusion import MultiModalFusion, StubEncoder
from helpers import banner, report

D, K = 128, 32
NAMES = ["scada", "vibration", "acoustic", "rgbd"]


def _fusion(p=0.3, seed=0):
    torch.manual_seed(seed)
    return MultiModalFusion(
        {n: StubEncoder(D, n_tokens=16, n_channels=3) for n in NAMES},
        d_model=D, n_latents=K, modality_dropout=p,
    )


def _inputs(B, L=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {n: {"x": torch.randn(B, 3, L, generator=g),
                "mask": torch.ones(B, 3)} for n in NAMES}


def test():
    banner("fusion stage 3: modality dropout")
    torch.manual_seed(0)
    f = _fusion(p=0.3)
    B = 1024
    full = torch.ones(B, 4)

    # --- eval: a no-op, and the forward pass is reproducible ---------------
    f.eval()
    kept = f._drop_modalities(full)
    report("eval leaves presence untouched", (kept - full).abs().max().item(),
           torch.equal(kept, full))

    ins = _inputs(4, seed=1)
    with torch.no_grad():
        r1, r2 = f(ins, present=torch.ones(4, 4)), f(ins, present=torch.ones(4, 4))
    report("eval forward is deterministic", (r1 - r2).abs().max().item(),
           torch.equal(r1, r2))

    # --- train: modalities actually get hidden -----------------------------
    f.train()
    kept = f._drop_modalities(full)
    frac = kept.mean().item()
    print(f"  kept fraction {frac:.3f}  (expected ~0.702 at p=0.3, M=4)")
    report("something is dropped", frac, frac < 0.99, fmt="{:.3f}")
    report("rate matches configuration", frac, 0.65 < frac < 0.76, fmt="{:.3f}")

    # --- the hard guarantee: never strand a sample -------------------------
    worst = min(f._drop_modalities(full).sum(1).min().item() for _ in range(50))
    report("every sample keeps >= 1 modality", worst, worst >= 1, fmt="{:.0f}")

    # at a brutal rate this is the only thing standing between the model and a
    # fully-masked attention row, so check it where it actually gets stressed
    hard = _fusion(p=0.95)
    hard.train()
    worst = min(hard._drop_modalities(full).sum(1).min().item() for _ in range(50))
    report("holds at p=0.95 too", worst, worst >= 1, fmt="{:.0f}")

    # --- dropout removes, never adds ---------------------------------------
    partial = torch.zeros(B, 4)
    partial[:, 0] = 1.0
    partial[:, 1] = 1.0                       # only scada + vibration exist
    f.train()
    added = 0.0
    for _ in range(20):
        out = f._drop_modalities(partial)
        added = max(added, (out * (1.0 - partial)).max().item())
        assert bool((out.sum(1) >= 1).all()), "stranded a sample"
    report("never resurrects an absent modality", added, added == 0.0)

    # --- p=0 disables it entirely, even in train mode ----------------------
    off = _fusion(p=0.0)
    off.train()
    kept = off._drop_modalities(full)
    report("p=0 is a no-op in train mode", (kept - full).abs().max().item(),
           torch.equal(kept, full))

    # --- a dropped modality is genuinely gone from the forward pass --------
    # with p=0 and an explicit mask we already know masking works (stage 2);
    # here the mask comes from DROPOUT, so this checks the plumbing between
    # _drop_modalities and the attention pad mask
    f.train()
    torch.manual_seed(3)
    ins = _inputs(8, seed=2)
    _, parts = f(ins, present=torch.ones(8, 4), return_parts=True)
    eff, pad = parts["present"], parts["pad_mask"]
    expect = (eff.repeat_interleave(K, dim=1) == 0)
    report("pad mask follows the dropout draw",
           float((pad != expect).sum()), bool((pad == expect).all()), fmt="{:.0f}")
    report("some blocks really are masked", float(pad.sum()), pad.any(),
           fmt="{:.0f}")

    # --- construction guard ------------------------------------------------
    try:
        _fusion(p=1.0)
        report("rejects modality_dropout >= 1", 0.0, False)
    except AssertionError:
        report("rejects modality_dropout >= 1", 1.0, True, fmt="{:.0f}")


if __name__ == "__main__":
    test()
