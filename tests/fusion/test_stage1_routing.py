"""Stage 1 -- routing: N modalities in, one fixed-shape health state out.

The point of the per-modality resampler is that the fusion body stops caring
how many tokens a modality natively produces. This stage checks that the
interface really is uniform, using the two REAL encoders whose token counts
differ by 17x.
"""

import torch

from models.multimodal_fusion import MultiModalFusion, StubEncoder
from models.scada_encoder_tcn import ScadaTCNEncoder
from models.vibration_encoder_2dconv import VibrationConv2dEncoder
from helpers import banner, check_shape, report

D, K = 128, 32


def _stub_fusion(n_mod=4, **kw):
    enc = {f"m{i}": StubEncoder(D, n_tokens=16 + 8 * i, n_channels=3)
           for i in range(n_mod)}
    return MultiModalFusion(enc, d_model=D, n_latents=K, **kw)


def _stub_inputs(fusion, B=4, L=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {n: {"x": torch.randn(B, 3, L, generator=g), "mask": torch.ones(B, 3)}
            for n in fusion.names}


def test():
    banner("fusion stage 1: routing and shape contract")
    torch.manual_seed(0)
    B = 2

    # --- the real thing: two encoders with very different token counts -------
    # production defaults on purpose. A reduced SCADA config can land on 48
    # tokens, exactly matching vibration, and then this stage silently stops
    # testing the thing it exists to test.
    scada = ScadaTCNEncoder(d_model=D)
    vib = VibrationConv2dEncoder(d_model=D, n_channels=8)
    fusion = MultiModalFusion(
        {"scada": scada, "vibration": vib,
         "acoustic": StubEncoder(D, n_tokens=64, n_channels=4),
         "rgbd": StubEncoder(D, n_tokens=32, n_channels=3)},
        d_model=D, n_latents=K, modality_dropout=0.0,
    ).eval()

    xs, ms = torch.randn(B, 20, 600), torch.ones(B, 20, 600)
    xv, mv = torch.randn(B, 8, 25600), torch.ones(B, 8)
    inputs = {
        "scada": {"x": xs, "mask": ms},
        "vibration": {"x": xv, "mask": mv},
        "acoustic": {"x": torch.randn(B, 4, 16000), "mask": torch.ones(B, 4)},
        "rgbd": {"x": torch.randn(B, 3, 4096), "mask": torch.ones(B, 3)},
    }

    with torch.no_grad():
        n_scada = scada(xs, ms).shape[1]
        n_vib = vib(xv, mv).shape[1]
        health, parts = fusion(inputs, return_parts=True)

    print(f"  native tokens: scada {n_scada}, vibration {n_vib} "
          f"-> {K} latents each")
    report("native counts really do differ", n_scada / n_vib,
           n_scada != n_vib, fmt="{:.1f}x")
    check_shape("health state", health, (B, D))
    check_shape("fused sequence", parts["tokens"], (B, 4 * K, D))

    # every modality owns exactly K contiguous slots, in registration order
    report("block count == modalities", parts["tokens"].shape[1] / K,
           parts["tokens"].shape[1] == len(fusion.names) * K, fmt="{:.0f}")

    # --- shape is independent of how many modalities are registered ----------
    for n_mod in (1, 2, 3, 4, 6):
        f = _stub_fusion(n_mod, modality_dropout=0.0).eval()
        with torch.no_grad():
            h, p = f(_stub_inputs(f, B=3), return_parts=True)
        ok = h.shape == (3, D) and p["tokens"].shape == (3, n_mod * K, D)
        report(f"{n_mod} modalities -> health (3, {D})", float(n_mod), ok,
               fmt="{:.0f}")

    # --- modality identity: identical DATA on two modalities must still -----
    # --- produce distinguishable blocks, via the modality embedding ---------
    # equal-sized stubs here, unlike _stub_fusion: the two branches have to be
    # weight-for-weight identical or a difference proves nothing
    f = MultiModalFusion(
        {"m0": StubEncoder(D, n_tokens=16, n_channels=3),
         "m1": StubEncoder(D, n_tokens=16, n_channels=3)},
        d_model=D, n_latents=K, modality_dropout=0.0,
    ).eval()
    g = torch.Generator().manual_seed(7)
    same = torch.randn(2, 3, 512, generator=g)
    f.encoders["m1"].load_state_dict(f.encoders["m0"].state_dict())
    f.resamplers["m1"].load_state_dict(f.resamplers["m0"].state_dict())
    f.latent_norms["m1"].load_state_dict(f.latent_norms["m0"].state_dict())
    with torch.no_grad():
        _, p = f({"m0": {"x": same, "mask": torch.ones(2, 3)},
                  "m1": {"x": same, "mask": torch.ones(2, 3)}},
                 return_parts=True)
    blk0, blk1 = p["tokens"][:, :K], p["tokens"][:, K:2 * K]
    diff = (blk0 - blk1).abs().max().item()
    report("identical data, modalities differ", diff, diff > 1e-3, fmt="{:.4f}")

    emb = f.modality_embed.weight
    report("embedding rows == modalities", emb.shape[0],
           emb.shape[0] == 2, fmt="{:.0f}")
    report("embedding is trainable", float(emb.requires_grad),
           emb.requires_grad, fmt="{:.0f}")


if __name__ == "__main__":
    test()
