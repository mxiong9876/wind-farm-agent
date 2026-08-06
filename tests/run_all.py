"""Run every stage test in order. Stops at the first failure."""

import sys
import traceback

import test_stage1_stem
import test_stage2_causalconv
import test_stage3_channelnorm
import test_stage4_causalblock
import test_stage5_trunk
import test_stage6_revin
import test_stage7_fold
import test_stage8_tokens
import test_stage9_channel_identity
import test_stage10_resampler

STAGES = [
    ("1 stem", test_stage1_stem),
    ("2 causal conv", test_stage2_causalconv),
    ("3 channel norm", test_stage3_channelnorm),
    ("4 causal block", test_stage4_causalblock),
    ("5 trunk", test_stage5_trunk),
    ("6 revin", test_stage6_revin),
    ("7 fold", test_stage7_fold),
    ("8 tokens", test_stage8_tokens),
    ("9 channel identity", test_stage9_channel_identity),
    ("10 resampler", test_stage10_resampler),
]

if __name__ == "__main__":
    for name, mod in STAGES:
        try:
            mod.test()
        except AssertionError:
            print(f"\nFAILED at stage {name}\n")
            traceback.print_exc()
            sys.exit(1)
    print("\nall stages passed\n")
