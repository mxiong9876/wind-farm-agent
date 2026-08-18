"""Run the vlm checks.

    python3 tests/vlm/run_all.py       (from anywhere; no PYTHONPATH needed)

Covers models/vlm_bridge.py. Like tests/kelmarsh, this suite is a smoke test rather than a set
of stage tests, and is included in tests/run_all.py anyway: it builds no
synthetic task and trains nothing, so it costs seconds rather than minutes.

SKIPS ENTIRELY without a cached Qwen3-VL checkpoint -- there is no meaningful
way to test a splice into a language model without the language model. A fresh
clone still gets a green suite.
"""

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, _TESTS, os.path.dirname(_TESTS)]

import smoke_test

STAGES = [("vlm", smoke_test)]

if __name__ == "__main__":
    for name, mod in STAGES:
        try:
            mod.test()
        except AssertionError:
            print(f"\nFAILED at {name}\n")
            traceback.print_exc()
            sys.exit(1)
    print("\nall vlm checks passed\n")
