"""Run every vibration stage test in order. Stops at the first failure.

Covers vibration_encoder_2dconv.py only. The SCADA suite is independent and
lives in ../scada -- neither needs the other to run.

    python3 tests/vibration/run_all.py    (from anywhere; no PYTHONPATH needed)

Stage 3 imports PerceiverResampler from scada_encoder_tcn, but only to confirm
both modalities reach fusion through the same component. Nothing here depends
on the SCADA suite passing.
"""

import os
import sys
import traceback

# tests/vibration, tests, and the repo root, so `helpers` and the encoder
# modules resolve no matter which directory this is invoked from
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, _TESTS, os.path.dirname(_TESTS)]

import test_stage1_dsp
import test_stage2_conv_trunk
import test_stage3_tokens

STAGES = [
    ("1 dsp frontend", test_stage1_dsp),
    ("2 conv trunk", test_stage2_conv_trunk),
    ("3 token contract", test_stage3_tokens),
]

if __name__ == "__main__":
    for name, mod in STAGES:
        try:
            mod.test()
        except AssertionError:
            print(f"\nFAILED at vibration stage {name}\n")
            traceback.print_exc()
            sys.exit(1)
    print("\nall vibration stages passed\n")
