"""Run both encoder suites.

    python3 tests/run_all.py              (from anywhere; no PYTHONPATH needed)

Each suite runs in its OWN process. That is the point: the two encoders are
independent, so a failure in one must not stop the other from reporting. A
crash or a hung import on one side cannot take the other down with it.

To run just one:

    python3 tests/scada/run_all.py
    python3 tests/vibration/run_all.py
"""

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

SUITES = [
    ("SCADA", os.path.join(_HERE, "scada", "run_all.py")),
    ("vibration", os.path.join(_HERE, "vibration", "run_all.py")),
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
