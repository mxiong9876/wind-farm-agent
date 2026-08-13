"""Run every fusion stage test in order. Stops at the first failure.

    python3 tests/fusion/run_all.py       (from anywhere; no PYTHONPATH needed)

Covers multimodal_fusion.py. Unlike the encoder suites this one does import
both real encoders (stage 1, to prove two very different token counts land on
the same 32 latents), but it does not depend on their suites passing -- a
broken SCADA encoder would fail here for the same reason it fails there, not
for a fusion reason.
"""

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, _TESTS, os.path.dirname(_TESTS)]

import test_stage1_routing
import test_stage2_presence
import test_stage3_modality_dropout

STAGES = [
    ("1 routing", test_stage1_routing),
    ("2 presence masking", test_stage2_presence),
    ("3 modality dropout", test_stage3_modality_dropout),
]

if __name__ == "__main__":
    for name, mod in STAGES:
        try:
            mod.test()
        except AssertionError:
            print(f"\nFAILED at fusion stage {name}\n")
            traceback.print_exc()
            sys.exit(1)
    print("\nall fusion stages passed\n")
