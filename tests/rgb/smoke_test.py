"""Smoke test for the frozen DINOv2 RGB encoder.

Almost all of this file checks that the backbone stays FROZEN, because that is
the property the whole design rests on and the only one that fails silently.
An encoder whose ViT drifts into train mode still returns the right shapes and
still trains -- it just returns different "frozen" features on every forward
pass, and the head downstream fits noise.

Runs with pretrained=False where it can, so shape and freezing checks need no
download. The signal test needs real weights and skips if they are not cached:

    python3 tests/rgb/smoke_test.py    (from anywhere; no PYTHONPATH needed)

Cache the weights with:
    python3 -c "import timm; timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)"
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TESTS)
sys.path[:0] = [_HERE, _TESTS, _ROOT]

import torch

from models.rgb_ViT import RGBEncoder
from helpers import banner, report

S = 224          # 16x16 patches; small enough to keep the suite quick
WEIGHTS = "models--timm--vit_base_patch14_dinov2.lvd142m"


def _weights_cached():
    return os.path.isdir(os.path.expanduser(
        f"~/.cache/huggingface/hub/{WEIGHTS}"))


# ---------------------------------------------------------------------------
def test_shapes_and_freezing():
    banner("rgb 1: shapes, token layout, and what is frozen")
    torch.manual_seed(0)
    enc = RGBEncoder(d_model=128, pretrained=False).eval()

    with torch.no_grad():
        out = enc(torch.rand(2, 3, S, S))
    n_patch = (S // enc.patch) ** 2
    print(f"  {tuple(out.shape)} = 1 CLS + {n_patch} patches, width 128")
    report("output is (B, 1+P, d)", 1.0,
           out.shape == (2, 1 + n_patch, 128), fmt="{:.0f}")

    n_tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    n_fz = sum(p.numel() for p in enc.parameters() if not p.requires_grad)
    print(f"  trainable {n_tr:,}   frozen {n_fz:,}")
    # the ratio IS the design: blade imagery is scarce, so the trainable
    # surface is a LayerNorm and one Linear against an 86M-parameter backbone
    report("trainable surface stays tiny", n_tr / (n_tr + n_fz),
           n_tr / (n_tr + n_fz) < 0.01, fmt="{:.4f}")
    report("backbone is frozen", float(n_fz), n_fz > 8e7, fmt="{:.0f}")


def test_backbone_stays_eval():
    banner("rgb 2: the backbone stays in eval() even in train mode")
    enc = RGBEncoder(d_model=128, pretrained=False)
    enc.train()
    # THE failure this file exists for. Without RGBEncoder.train()'s override,
    # model.train() re-enables the ViT's dropout and stochastic depth, so the
    # same image yields different "frozen" features every step -- noise the
    # head cannot learn through, and invisible unless you diff two passes.
    report("parent is training", float(enc.training), enc.training, fmt="{:.0f}")
    report("vit is NOT training", float(not enc.vit.training),
           not enc.vit.training, fmt="{:.0f}")

    x = torch.rand(1, 3, S, S)
    with torch.no_grad():
        a, b = enc(x), enc(x)
    d = (a - b).abs().max().item()
    print(f"  max diff across two passes in train mode: {d:.2e}")
    report("frozen path is deterministic", d, d == 0.0, fmt="{:.2e}")


def test_gradient_routing():
    banner("rgb 3: gradient reaches the projection and not the backbone")
    torch.manual_seed(0)
    enc = RGBEncoder(d_model=128, pretrained=False).train()
    enc(torch.rand(1, 3, S, S)).sum().backward()

    g = enc.proj[1].weight.grad
    report("projection gets gradient", 0.0 if g is None else g.abs().max().item(),
           g is not None and g.abs().max().item() > 0, fmt="{:.2e}")
    # a leak here would train 86M parameters on a handful of blade photos
    vit_grads = [p.grad for p in enc.vit.parameters() if p.grad is not None]
    report("backbone gets NO gradient", float(len(vit_grads)),
           not vit_grads, fmt="{:.0f}")


def test_variable_image_size():
    banner("rgb 4: accepts any multiple of the patch size")
    enc = RGBEncoder(d_model=128, pretrained=False).eval()
    for s in (224, 322, 518):
        with torch.no_grad():
            o = enc(torch.rand(1, 3, s, s))
        report(f"{s}x{s} -> {(s // enc.patch) ** 2} patches", float(o.shape[1]),
               o.shape[1] == (s // enc.patch) ** 2 + 1, fmt="{:.0f}")

    # a patch-14 backbone mis-tiles a non-multiple size rather than raising,
    # and the token grid then does not correspond to the image
    try:
        enc(torch.rand(1, 3, 500, 500))
        ok = False
    except AssertionError:
        ok = True
    report("rejects a non-multiple size", float(ok), ok, fmt="{:.0f}")

    # with dynamic_img_size off, timm accepts only the checkpoint's native size
    # and rejects the rest deep inside patch_embed, naming the model rather
    # than the caller
    st = RGBEncoder(d_model=128, pretrained=False, dynamic_img_size=False).eval()
    try:
        st(torch.rand(1, 3, 224, 224))
        ok = False
    except AssertionError:
        ok = True
    report("static mode rejects 224 clearly", float(ok), ok, fmt="{:.0f}")


def test_mask_and_preprocess():
    banner("rgb 5: dead-camera mask and ImageNet normalisation")
    enc = RGBEncoder(d_model=128, pretrained=False).eval()
    x = torch.rand(2, 3, S, S)
    with torch.no_grad():
        out = enc(x, mask=torch.tensor([1.0, 0.0]))
    report("masked frame is zeroed", float(bool((out[1] == 0).all())),
           bool((out[1] == 0).all()), fmt="{:.0f}")
    report("unmasked frame survives", float(bool((out[0] != 0).any())),
           bool((out[0] != 0).any()), fmt="{:.0f}")

    # DINOv2 was trained on ImageNet-normalised input and its features degrade
    # on raw [0,1] pixels -- a silent quality loss, not an error
    p = enc.preprocess(x)
    print(f"  raw mean {x.mean():.3f} -> normalised mean {p.mean():+.3f}")
    report("preprocess shifts the distribution", abs(float(p.mean())),
           abs(float(p.mean() - x.mean())) > 0.1, fmt="{:.3f}")


def test_carries_defect_signal():
    banner("rgb 6: frozen features separate a synthetic defect")
    if not _weights_cached():
        print("  SKIP: pretrained DINOv2 not in ~/.cache/huggingface/hub")
        return

    torch.manual_seed(0)

    def make(n, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(n, 3, S, S, generator=g) * 0.15 + 0.55
        yy = torch.linspace(0, 1, S).view(1, 1, S, 1)
        x = x + 0.12 * torch.sin(2 * math.pi * yy * 1.5
                                 + torch.rand(n, 1, 1, 1, generator=g) * 6)
        y = (torch.rand(n, generator=g) < 0.5).float()
        for i in range(n):
            if y[i] > 0:                       # thin dark hairline "crack"
                r0 = int(torch.randint(30, S - 30, (1,), generator=g))
                c0 = int(torch.randint(20, S - 90, (1,), generator=g))
                for k in range(int(torch.randint(50, 85, (1,), generator=g))):
                    x[i, :, min(S - 1, r0 + k // 6), min(S - 1, c0 + k)] -= 0.40
        # match per-image brightness so mean pixel value cannot solve it
        x = x - x.mean(dim=(1, 2, 3), keepdim=True) + 0.60
        return x.clamp(0, 1), y

    def ridge(Htr, ytr, Hte, yte, lam=1e-2):
        mu = Htr.mean(0)
        A = torch.cat([Htr - mu, torch.ones(len(Htr), 1)], 1)
        w = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]),
                               A.T @ (ytr[:, None] - ytr.mean()))
        s = (torch.cat([Hte - mu, torch.ones(len(Hte), 1)], 1) @ w).squeeze(-1)
        return ((s > 0).float() == yte).float().mean().item()

    enc = RGBEncoder(d_model=128, pretrained=True).eval()
    xtr, ytr = make(96, 0)
    xte, yte = make(64, 999)
    gap = abs(float(xtr[ytr == 1].mean() - xtr[ytr == 0].mean()))
    report("brightness matched across classes", gap, gap < 1e-4, fmt="{:.6f}")

    with torch.no_grad():
        f = lambda X: torch.cat([enc(enc.preprocess(X[i:i + 8])).mean(1)
                                 for i in range(0, len(X), 8)])
        Htr, Hte = f(xtr), f(xte)

    acc = ridge(Htr, ytr, Hte, yte)
    pix = ridge(xtr.flatten(1)[:, ::43], ytr, xte.flatten(1)[:, ::43], yte)
    torch.manual_seed(1)
    sh = ridge(Htr, ytr[torch.randperm(len(ytr))],
               Hte, yte[torch.randperm(len(yte))])
    print(f"  crack covers ~{50 * 3 / (S * S) * 100:.2f}% of pixels")
    # raw pixels at chance and shuffled labels at chance together license
    # believing the DINOv2 number -- either one alone would not
    report("raw pixels stay at chance", pix, pix < 0.65, fmt="{:.3f}")
    report("shuffled labels stay at chance", sh, sh < 0.70, fmt="{:.3f}")
    report("DINOv2 features separate the defect", acc, acc > 0.85, fmt="{:.3f}")


def test():
    """Frozen-backbone checks plus one signal probe.

    COST: about 20 s without pretrained weights. Test 6 adds ~40 s and needs
    the DINOv2 checkpoint cached; it skips cleanly when absent, so a fresh
    clone still gets a green suite.
    """
    test_shapes_and_freezing()
    test_backbone_stays_eval()
    test_gradient_routing()
    test_variable_image_size()
    test_mask_and_preprocess()
    test_carries_defect_signal()
    print("\nrgb smoke test complete\n")


if __name__ == "__main__":
    test()
