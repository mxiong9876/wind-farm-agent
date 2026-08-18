"""End-to-end inference: four modalities -> encoders -> fusion -> VLM -> text.

    python3 scripts/pipeline_demo.py                    # shapes only, no VLM
    python3 scripts/pipeline_demo.py --vlm              # + the language model
    python3 scripts/pipeline_demo.py --vlm --generate   # + actually write text

WHAT THIS PROVES AND WHAT IT DOES NOT

It proves the WIRING: that four encoders with wildly different input shapes and
token counts land on a common 32-latent block each, that the fusion collapses
them to one 128-d health vector, that the projector lifts that into the
language model's embedding space at the right width and scale, and that the
whole chain runs in one forward pass inside a laptop's memory.

It proves NOTHING about the text. The projector is untrained, so the soft
tokens carry no meaning and anything generated is fluent noise. That is the
expected result here, not a bug -- see models/vlm_bridge.py. Training needs
paired (sensor window, text) examples that do not exist yet.

MEMORY

Forward-only, so no optimizer state and no retained activations. Weights
dominate: ~4.0GB for Qwen3-VL-2B, ~350MB for DINOv2, ~10MB for everything you
wrote. Backward through the 2B would also fit; backward through the 8B would
not, which is the line where a server starts being necessary.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.multimodal_fusion import MultiModalFusion, StubEncoder
from models.scada_encoder_tcn import ScadaTCNEncoder
from models.vibration_encoder_2dconv import VibrationConv2dEncoder
from models.rgb_ViT import RGBEncoder

D = 128          # every encoder emits this width; the fusion assumes it
K = 32           # latents per modality after each PerceiverResampler


def build_fusion(rgb_pretrained=False):
    """The four registered modalities, at Kelmarsh's real SCADA geometry."""
    torch.manual_seed(0)
    return MultiModalFusion({
        "scada": ScadaTCNEncoder(d_model=D, n_channels=20, context_len=600),
        "vibration": VibrationConv2dEncoder(d_model=D, n_channels=8),
        "rgb": RGBEncoder(d_model=D, pretrained=rgb_pretrained),
        "acoustic": StubEncoder(D, n_tokens=64, n_channels=4),
    }, d_model=D, n_latents=K, modality_dropout=0.0).eval()


def make_inputs(B=1, real_scada=True):
    """Real Kelmarsh SCADA where available; synthetic for the rest.

    Only SCADA has real data on this project. The other three are shaped
    correctly and carry noise -- enough to exercise the routing, which is what
    this demo is for.
    """
    src = "synthetic"
    x = m = None
    if real_scada:
        try:
            from data_io.kelmarsh_io import load_years
            X, M, _, _, _ = load_years("data/kelmarsh", years=range(2016, 2017))
            x, m, src = X[:B], M[:B], "real Kelmarsh 2016"
        except Exception:
            pass
    if x is None:
        x, m = torch.randn(B, 20, 600), torch.ones(B, 20, 600)

    return {
        "scada": {"x": x, "mask": m},
        "vibration": {"x": torch.randn(B, 8, 25600), "mask": torch.ones(B, 8)},
        "rgb": {"x": torch.rand(B, 3, 518, 518), "mask": torch.ones(B)},
        "acoustic": {"x": torch.randn(B, 4, 16000), "mask": torch.ones(B, 4)},
    }, src


def main(a):
    print("building encoders + fusion ...")
    t0 = time.time()
    fusion = build_fusion(rgb_pretrained=a.pretrained_rgb)
    inputs, src = make_inputs(a.batch)
    print(f"  ready in {time.time()-t0:.1f}s   scada source: {src}\n")

    print("INPUTS")
    for name, kw in inputs.items():
        print(f"  {name:<10} {tuple(kw['x'].shape)}")

    # per-modality token counts before the resampler equalises them -- this is
    # the imbalance the per-modality resamplers exist to remove
    print("\nENCODER TOKENS (before resampling)")
    with torch.no_grad():
        for name in fusion.names:
            t = fusion.encoders[name](**inputs[name])
            print(f"  {name:<10} {tuple(t.shape)}")

    print("\nFUSION")
    t0 = time.time()
    with torch.no_grad():
        health, parts = fusion(inputs, return_parts=True)
    print(f"  tokens     {tuple(parts['tokens'].shape)}  "
          f"= 4 modalities x {K} latents")
    print(f"  health     {tuple(health.shape)}   finite "
          f"{bool(torch.isfinite(health).all())}   {time.time()-t0:.1f}s")

    # graceful degradation: the same call with modalities switched off
    print("\n  presence masking (same inputs, modalities withheld):")
    for label, p in (("all four", [1., 1., 1., 1.]),
                     ("scada only", [1., 0., 0., 0.]),
                     ("no camera", [1., 1., 0., 1.])):
        with torch.no_grad():
            h = fusion(inputs, present=torch.tensor([p]).expand(a.batch, -1))
        print(f"    {label:<12} {tuple(h.shape)}  "
              f"first 3: {[round(v, 3) for v in h[0, :3].tolist()]}")

    if not a.vlm:
        print("\n(skipping the VLM; pass --vlm to include it)")
        return

    print("\nVLM BRIDGE")
    from models.vlm_bridge import FusionToVLM
    t0 = time.time()
    bridge = FusionToVLM(a.model, d_fusion=D, n_soft=8)
    print(f"  loaded {a.model} in {time.time()-t0:.0f}s   d_lm {bridge.d_lm}")

    brief = ["Turbine 3, window ending 2016-08-14. Generator bearing residual "
             "+3.2 C, rising over 11 days. Wind 7.2 m/s, power 980 kW."] * a.batch
    t0 = time.time()
    with torch.no_grad():
        emb, att, _ = bridge.build_inputs(health, brief)
        out = bridge.vlm(inputs_embeds=emb, attention_mask=att)
    print(f"  inputs_embeds {tuple(emb.shape)} = {bridge.n_soft} soft "
          f"+ {emb.shape[1]-bridge.n_soft} text")
    print(f"  logits        {tuple(out.logits.shape)}   finite "
          f"{bool(torch.isfinite(out.logits).all())}   {time.time()-t0:.1f}s")

    n = sum(p.numel() for p in fusion.parameters())
    nb = sum(p.numel() for p in bridge.parameters())
    print(f"\n  fusion stack  {n:,} params")
    print(f"  bridge + VLM  {nb:,} params "
          f"({sum(p.numel() for p in bridge.trainable_parameters()):,} trainable)")

    if a.generate:
        print("\nGENERATION (projector UNTRAINED -- expect fluent nonsense)")
        t0 = time.time()
        for txt in bridge.generate(health, brief, max_new_tokens=48,
                                   do_sample=False):
            print(f"  {txt.strip()[:300]!r}")
        print(f"  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--vlm", action="store_true")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--pretrained-rgb", action="store_true",
                   help="real DINOv2 weights (default: random, for speed)")
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    main(p.parse_args())
