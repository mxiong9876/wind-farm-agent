"""Smoke test for the fusion -> Qwen3-VL soft-token bridge.

Everything here checks the SPLICE, not the language model. A bridge can return
a plausible loss while supervising the wrong positions, leaking gradient into a
frozen 8B backbone, or emitting soft tokens at a norm the model's attention
ignores -- none of which raises, and none of which shows up as anything except
a projector that never learns.

Needs the Qwen3-VL checkpoint cached and SKIPS cleanly when it is absent, so a
fresh clone still gets a green suite:

    python3 tests/vlm/smoke_test.py    (from anywhere; no PYTHONPATH needed)

Cache the 2B (~4GB) with:
    python3 -c "from transformers import Qwen3VLForConditionalGeneration as M; \\
                M.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')"
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TESTS)
sys.path[:0] = [_HERE, _TESTS, _ROOT]

import torch

from helpers import banner, report

# the 2B is the smoke-test size; production uses 8B+. Both exercise the same
# splice, and the splice is all this file tests.
MODEL_ID = os.environ.get("VLM_SMOKE_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
CACHE = os.path.expanduser("~/.cache/huggingface/hub")

BRIEF = ["Turbine 3: generator bearing residual +3.2 C, rising for 11 days.",
         "Turbine 1: all residuals within this machine's normal scatter."]
TARGET = ["Investigate the generator cooling circuit within two weeks.",
          "No action required."]


def _cached():
    """True only when the checkpoint is COMPLETE.

    Directory existence is not enough. An interrupted download leaves the
    directory in place with a `.incomplete` blob, so a presence check reports
    cached and the suite then fails loading a truncated checkpoint -- which
    reads as a broken bridge rather than a broken download. Requiring both a
    real weight file and no partial blobs makes an interrupted download SKIP,
    which is the honest outcome.
    """
    d = os.path.join(CACHE, "models--" + MODEL_ID.replace("/", "--"))
    if not os.path.isdir(d):
        return False
    blobs, partial = [], False
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".incomplete"):
                partial = True
            elif f.endswith((".safetensors", ".bin")):
                blobs.append(os.path.join(root, f))
    return bool(blobs) and not partial


_BRIDGE = None


def _bridge():
    """Load once — an 8B checkpoint per test would dominate the runtime."""
    global _BRIDGE
    if _BRIDGE is None:
        from models.vlm_bridge import FusionToVLM
        torch.manual_seed(0)
        _BRIDGE = FusionToVLM(MODEL_ID, d_fusion=128, n_soft=8)
    return _BRIDGE


# ---------------------------------------------------------------------------
def test_projector_shape_and_scale():
    banner("vlm 1: the projector matches the model's embedding space")
    b = _bridge()
    print(f"  model {MODEL_ID}   d_lm {b.d_lm}   soft tokens {b.n_soft}")

    z = b.projector(torch.randn(3, 128).to(
        next(b.vlm.parameters()).device))
    report("(B, K, d_lm)", 1.0, z.shape == (3, b.n_soft, b.d_lm), fmt="{:.0f}")

    # THE quiet failure. A fresh Linear emits vectors whose norm is unrelated
    # to the embedding distribution: too small and attention ignores them, so
    # the projector receives almost no gradient and never learns; too large and
    # they swamp the prompt. Neither raises.
    emb_norm = b.vlm.get_input_embeddings().weight.detach().float() \
        .norm(dim=-1).median().item()
    soft_norm = z.detach().float().norm(dim=-1).mean().item()
    print(f"  embedding median norm {emb_norm:.3f}   soft token norm {soft_norm:.3f}")
    ratio = soft_norm / max(emb_norm, 1e-6)
    report("soft tokens are scale-matched", ratio,
           0.5 < ratio < 2.0, fmt="{:.3f}")

    n_tr = sum(p.numel() for p in b.trainable_parameters())
    n_fz = sum(p.numel() for p in b.parameters()) - n_tr
    print(f"  trainable {n_tr:,}   frozen {n_fz:,}")
    report("only the projector trains", n_tr / (n_tr + n_fz),
           n_tr / (n_tr + n_fz) < 0.05, fmt="{:.4f}")


def test_splice_layout():
    banner("vlm 2: soft tokens occupy real positions and are never supervised")
    b = _bridge()
    emb, att, lab = b.build_inputs(torch.randn(2, 128), BRIEF, TARGET)
    print(f"  inputs_embeds {tuple(emb.shape)} = {b.n_soft} soft "
          f"+ {emb.shape[1] - b.n_soft} text")
    report("embeds/mask/labels agree on length", 1.0,
           emb.shape[1] == att.shape[1] == lab.shape[1], fmt="{:.0f}")
    report("soft tokens are attended", float(att[:, :b.n_soft].min()),
           bool((att[:, :b.n_soft] == 1).all()), fmt="{:.0f}")

    # a soft token has no token id to predict, and the prompt is given; scoring
    # either trains the model to reproduce its own input
    report("soft positions excluded from loss", 1.0,
           bool((lab[:, :b.n_soft] == -100).all()), fmt="{:.0f}")
    n_sup = int((lab != -100).sum())
    print(f"  supervised positions: {n_sup} of {lab.numel()}")
    report("only the target is supervised", float(n_sup),
           0 < n_sup < lab.numel() * 0.5, fmt="{:.0f}")

    # without labels_text there is nothing to score at all -- the inference path
    _, _, lab2 = b.build_inputs(torch.randn(2, 128), BRIEF)
    report("no target -> nothing supervised", float((lab2 != -100).sum()),
           int((lab2 != -100).sum()) == 0, fmt="{:.0f}")


def test_health_reaches_the_model():
    banner("vlm 3: the health vector actually changes the output")
    b = _bridge()
    torch.manual_seed(0)
    h1, h2 = torch.randn(1, 128), torch.randn(1, 128)
    with torch.no_grad():
        l1 = b(health=h1, text=BRIEF[:1]).logits
        l2 = b(health=h2, text=BRIEF[:1]).logits
    d = (l1 - l2).abs().max().item()
    print(f"  max logit difference between two health vectors: {d:.3e}")
    # if this were zero the splice would be decorative: the model would be
    # reading the prompt only, and no amount of projector training would help
    report("different health -> different logits", d, d > 1e-4, fmt="{:.3e}")


def test_gradient_routing():
    banner("vlm 4: gradient reaches the projector and not the backbone")
    b = _bridge()
    b.train()
    report("vlm stays in eval", float(not b.vlm.training),
           not b.vlm.training, fmt="{:.0f}")

    b.zero_grad(set_to_none=True)
    out = b(health=torch.randn(2, 128), text=BRIEF, labels_text=TARGET)
    out.loss.backward()
    print(f"  loss {out.loss.item():.4f}")
    report("loss is finite", float(torch.isfinite(out.loss)),
           bool(torch.isfinite(out.loss)), fmt="{:.0f}")

    g = b.projector.net[1].weight.grad
    report("projector gets gradient", 0.0 if g is None else g.abs().max().item(),
           g is not None and g.abs().max().item() > 0, fmt="{:.2e}")
    # gradient must flow BACK THROUGH the frozen LM to reach the projector --
    # that it arrives is the proof the splice is differentiable end to end
    leaked = [n for n, p in b.vlm.named_parameters() if p.grad is not None]
    if leaked:
        print(f"  leaked into: {leaked[:3]}")
    report("no gradient leaks into the VLM", float(len(leaked)),
           not leaked, fmt="{:.0f}")


def test():
    """Splice checks against a real checkpoint.

    COST: ~90 s on the 2B once cached, dominated by loading. Set
    VLM_SMOKE_MODEL to test another size; the splice is identical.

    NOT COVERED: whether the projector learns anything. It cannot until it is
    trained on paired (sensor window, text) examples, which do not exist yet --
    see models/vlm_bridge.py. These checks establish that the wiring is sound,
    so that when pairs do exist a failure is attributable to the data or the
    training, not the plumbing.
    """
    if not _cached():
        print(f"\nSKIP: {MODEL_ID} not in {CACHE}")
        print("      see this file's docstring for the one-line download\n")
        return

    test_projector_shape_and_scale()
    test_splice_layout()
    test_health_reaches_the_model()
    test_gradient_routing()
    print("\nvlm smoke test complete\n")


if __name__ == "__main__":
    test()
