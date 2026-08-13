"""Run the Kelmarsh loader checks.

    python3 tests/kelmarsh/run_all.py     (from anywhere; no PYTHONPATH needed)

Covers data_io/kelmarsh_io.py. UNLIKE THE OTHER SUITES, this one is the smoke
test rather than a set of stage tests, and it is included in tests/run_all.py
anyway. The distinction elsewhere is cost: smoke tests train, so they are
minutes. This one builds no model and trains nothing -- it parses CSVs and
checks shapes, masks and label boundaries in about 20 seconds.

It SKIPS cleanly when the archives are absent, since data/ is gitignored, so a
fresh clone still gets a green suite.
"""

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, _TESTS, os.path.dirname(_TESTS)]

import smoke_test

STAGES = [
    ("loader", smoke_test),
]

if __name__ == "__main__":
    for name, mod in STAGES:
        try:
            mod.test()
        except AssertionError:
            print(f"\nFAILED at kelmarsh {name}\n")
            traceback.print_exc()
            sys.exit(1)
    print("\nall kelmarsh checks passed\n")
