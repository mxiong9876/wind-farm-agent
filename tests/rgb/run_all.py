"""Run the rgb checks.

    python3 tests/rgb/run_all.py       (from anywhere; no PYTHONPATH needed)

Covers models/rgb_ViT.py. Like tests/kelmarsh, this suite is a smoke test rather than a set
of stage tests, and is included in tests/run_all.py anyway: it builds no
synthetic task and trains nothing, so it costs seconds rather than minutes.

Needs no download for most checks: shape, freezing and gradient-routing
tests build the ViT with pretrained=False. Only the defect-signal probe needs
real DINOv2 weights, and it skips when they are not cached.
"""

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, _TESTS, os.path.dirname(_TESTS)]

import smoke_test

STAGES = [("rgb", smoke_test)]

if __name__ == "__main__":
    for name, mod in STAGES:
        try:
            mod.test()
        except AssertionError:
            print(f"\nFAILED at {name}\n")
            traceback.print_exc()
            sys.exit(1)
    print("\nall rgb checks passed\n")
