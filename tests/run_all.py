"""Run every stage-test suite.

    python3 tests/run_all.py              (from anywhere; no PYTHONPATH needed)

Each suite runs in its OWN process. That is the point: the suites are
independent, so a failure in one must not stop the others from reporting. A
crash or a hung import on one side cannot take the others down with it.

To run just one:

    python3 tests/scada/run_all.py
    python3 tests/vibration/run_all.py
    python3 tests/fusion/run_all.py

SMOKE TESTS ARE NOT RUN HERE. Stage tests are seconds and check the forward
pass, so they are cheap enough for every change. The smoke tests train, which
puts them in minutes, and they answer a different question -- can this thing
actually learn. Run them before spending cluster time, not before every commit:

    python3 tests/vibration/smoke_test.py
    python3 tests/fusion/smoke_test.py
"""

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

SUITES = [
    ("SCADA", os.path.join(_HERE, "scada", "run_all.py")),
    ("vibration", os.path.join(_HERE, "vibration", "run_all.py")),
    # after the encoders on purpose: fusion imports both, so if they are broken
    # the encoder suites should say so first rather than surfacing it as a
    # fusion failure
    ("fusion", os.path.join(_HERE, "fusion", "run_all.py")),
    # the loader depends on no model at all, so its position is arbitrary --
    # last because it is the only suite that can SKIP (data/ is gitignored) and
    # a skip is easiest to notice at the end
    ("kelmarsh", os.path.join(_HERE, "kelmarsh", "run_all.py")),
]

if __name__ == "__main__":
    results = []
    for name, path in SUITES:
        # flush: this process's stdout is block-buffered when piped, but the
        # children write straight through, so without it every header lands
        # after the output it is supposed to introduce
        print(f"\n{'=' * 60}\n  {name} suite\n{'=' * 60}", flush=True)
        # same interpreter that launched this, so a venv is respected
        results.append((name, subprocess.run([sys.executable, path]).returncode))

    print(f"\n{'=' * 60}\n  summary\n{'=' * 60}")
    for name, code in results:
        print(f"  {name:<12} {'PASS' if code == 0 else f'FAIL (exit {code})'}")

    sys.exit(0 if all(c == 0 for _, c in results) else 1)
