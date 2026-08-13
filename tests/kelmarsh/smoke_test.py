"""Smoke test for the Kelmarsh loader.

The other suites test MODELS on data they synthesise. This tests a LOADER on
data it does not control, which fails differently: nothing raises, the tensors
have the right shape, and the numbers are quietly wrong. A transposed window, a
mask built after the NaNs were filled, a label horizon that overlaps its own
window -- all of those produce a perfectly well-formed batch.

Skips cleanly if the archives are not present, since they are gitignored:

    python3 tests/kelmarsh/smoke_test.py   (from anywhere; no PYTHONPATH needed)

Download with the Zenodo links in data_io/kelmarsh_io.py.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TESTS)
sys.path[:0] = [_HERE, _TESTS, _ROOT]

import numpy as np
import pandas as pd
import torch

from data_io.kelmarsh_io import (CHANNELS, N_CHANNELS, SAMPLE_PERIOD, HORIZON,
                                 load_turbine, make_windows, load_status,
                                 label_windows, load_farm)
from helpers import banner, report

DATA = os.path.join(_ROOT, "data", "kelmarsh")
YEAR = os.path.join(DATA, "scada_2016")


def _have_data():
    import glob
    return bool(glob.glob(os.path.join(YEAR, "Turbine_Data_Kelmarsh_1_*.csv")))


def _turbine1():
    import glob
    return glob.glob(os.path.join(YEAR, "Turbine_Data_Kelmarsh_1_*.csv"))[0]


# ---------------------------------------------------------------------------
def test_schema_is_frozen():
    banner("kelmarsh 0: the schema is frozen and complete")
    report("20 channels", float(N_CHANNELS), N_CHANNELS == 20, fmt="{:.0f}")
    report("no duplicates", float(len(set(CHANNELS))),
           len(set(CHANNELS)) == len(CHANNELS), fmt="{:.0f}")

    # channel_embed indexes into CHANNELS positionally, so a reorder silently
    # rewires every embedding. Pinning the ends catches the common accident of
    # someone inserting a channel rather than appending one.
    report("first is wind speed", 1.0, CHANNELS[0] == "Wind speed (m/s)",
           fmt="{:.0f}")
    report("last is drive train accel", 1.0,
           CHANNELS[-1] == "Drive train acceleration (mm/ss)", fmt="{:.0f}")

    thermal = [c for c in CHANNELS if "(°C)" in c]
    print(f"  thermal channels: {len(thermal)} of {N_CHANNELS}")
    # degradation shows up in the thermals; a schema that lost them would still
    # load, still train, and never work
    report("thermals present", float(len(thermal)), len(thermal) >= 8,
           fmt="{:.0f}")


def test_grid_is_regular():
    banner("kelmarsh 1: rows land on a gapless 10-minute grid")
    df = load_turbine(_turbine1())
    gaps = (df.index.to_series().diff().dropna() != SAMPLE_PERIOD).sum()
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} channels")
    # the CSV omits missing periods entirely; without the reindex, row i and
    # i+1 can be 10 minutes or 3 weeks apart and the TCN cannot tell
    report("no irregular gaps", float(gaps), gaps == 0, fmt="{:.0f}")
    report("columns are exactly CHANNELS", 1.0,
           list(df.columns) == CHANNELS, fmt="{:.0f}")
    report("index is monotonic", 1.0, bool(df.index.is_monotonic_increasing),
           fmt="{:.0f}")


def test_windows_are_well_formed():
    banner("kelmarsh 2: windows match what the encoder expects")
    df = load_turbine(_turbine1())
    x, mask, ends = make_windows(df, context_len=600, stride=144)

    print(f"  {tuple(x.shape)}   mask mean {mask.mean():.3f}")
    report("shape is (B, 20, 600)", 1.0,
           x.shape[1:] == (N_CHANNELS, 600), fmt="{:.0f}")
    report("mask matches x", 1.0, mask.shape == x.shape, fmt="{:.0f}")
    report("one end timestamp per window", float(len(ends)),
           len(ends) == len(x), fmt="{:.0f}")

    # THE ordering bug: build the mask from the NaN pattern, THEN fill. Reverse
    # those two lines and every mask is 1 and missing data reads as a real zero.
    report("x is 0 wherever mask is 0", 1.0,
           bool((x[mask == 0] == 0).all()), fmt="{:.0f}")
    report("mask is strictly 0 or 1", 1.0,
           bool(((mask == 0) | (mask == 1)).all()), fmt="{:.0f}")
    report("no NaN or inf survived", 1.0,
           bool(torch.isfinite(x).all()), fmt="{:.0f}")

    # a transposed window has the right shape and the wrong meaning: with 20
    # channels and 600 steps the axes cannot be confused by size, but the
    # per-channel spread tells you the time axis really is last
    spread = x.std(dim=-1).mean().item()
    report("channels vary along time", spread, spread > 1e-6, fmt="{:.3f}")


def test_windows_are_contiguous_in_time():
    banner("kelmarsh 3: a window is 600 CONSECUTIVE rows")
    df = load_turbine(_turbine1())
    _, _, ends = make_windows(df, context_len=600, stride=144)

    # 600 rows at 10 minutes is 4.2 days. If this drifts, someone changed
    # context_len or the sample period and the model's horizon changed with it.
    span = pd.Timedelta(minutes=10) * 599
    print(f"  one window spans {span}  ({span.total_seconds()/86400:.2f} days)")
    report("window spans 4.2 days", span.total_seconds() / 86400,
           abs(span.total_seconds() / 86400 - 4.16) < 0.05, fmt="{:.2f}")

    step = (ends[1] - ends[0]).total_seconds() / 86400
    report("stride 144 = 1 day between windows", step,
           abs(step - 1.0) < 0.01, fmt="{:.2f}")


def test_labels_look_right():
    banner("kelmarsh 4: labels come from forced outages, not battery tests")
    import glob
    stat = glob.glob(os.path.join(YEAR, "Status_Kelmarsh_1_*.csv"))[0]

    forced = load_status(stat)
    every = load_status(stat, category=None)
    print(f"  {len(forced)} forced outages of {len(every)} status events")
    # 258 of 751 Stop events in 2016 are "Battery test". Filtering on IEC
    # category rather than Status is what keeps routine maintenance out of the
    # positive class -- and keeps the generator-fan overloads, logged as
    # Warning, in it.
    report("filter removes most events", float(len(forced)),
           0 < len(forced) < len(every) * 0.5, fmt="{:.0f}")
    report("event times are sorted", 1.0,
           bool(forced.is_monotonic_increasing), fmt="{:.0f}")

    df = load_turbine(_turbine1())
    _, _, ends = make_windows(df)
    y = label_windows(ends, forced)
    print(f"  positive rate {y.mean():.3f} at a {HORIZON.days}-day horizon")
    report("labels are 0 or 1", 1.0,
           bool(((y == 0) | (y == 1)).all()), fmt="{:.0f}")
    report("both classes present", y.mean().item(),
           0.0 < y.mean().item() < 1.0, fmt="{:.3f}")

    # THE LEAK: the horizon must start strictly AFTER the window ends. Move an
    # event to the window's own last timestamp and it must NOT fire -- otherwise
    # the model can read the outage out of its own input.
    inside = pd.DatetimeIndex([ends[5]])
    report("event AT the window end does not fire",
           float(label_windows(ends, inside)[5]),
           label_windows(ends, inside)[5].item() == 0.0, fmt="{:.0f}")
    just_after = pd.DatetimeIndex([ends[5] + pd.Timedelta("10min")])
    report("event just after DOES fire",
           float(label_windows(ends, just_after)[5]),
           label_windows(ends, just_after)[5].item() == 1.0, fmt="{:.0f}")
    far_off = pd.DatetimeIndex([ends[5] + HORIZON + pd.Timedelta("1D")])
    report("event beyond the horizon does not fire",
           float(label_windows(ends, far_off)[5]),
           label_windows(ends, far_off)[5].item() == 0.0, fmt="{:.0f}")


def test_farm_assembles():
    banner("kelmarsh 5: six turbines assemble without crossing wires")
    X, M, E, T, Y = load_farm(YEAR)
    counts = torch.bincount(T)[1:]
    print(f"  {tuple(X.shape)}   turbines {counts.tolist()}")
    report("all six turbines present", float((counts > 0).sum()),
           int((counts > 0).sum()) == 6, fmt="{:.0f}")
    report("one label per window", float(len(Y)), len(Y) == len(X), fmt="{:.0f}")
    report("one timestamp per window", float(len(E)), len(E) == len(X),
           fmt="{:.0f}")
    report("one turbine id per window", float(len(T)), len(T) == len(X),
           fmt="{:.0f}")

    # labels are per turbine. If the status join leaked across machines every
    # turbine would carry the same positive rate; they emphatically do not --
    # 2016 spans 3% to 31%, which is itself the reason a leave-one-turbine-out
    # split is a bad idea here.
    rates = [Y[T == t].mean().item() for t in range(1, 7)]
    print("  positive rate per turbine: "
          + " ".join(f"{r:.3f}" for r in rates))
    report("rates differ across turbines", float(np.std(rates)),
           np.std(rates) > 0.01, fmt="{:.3f}")


def test():
    """Loader smoke test. Needs the Kelmarsh archives on disk.

    COST: about 20 seconds, dominated by parsing six 299-column CSVs. No model
    is built and nothing trains, so this is cheap enough to run on every change
    to the loader.
    """
    if not _have_data():
        print(f"\nSKIP: no Kelmarsh archives under {DATA}")
        print("      see the Zenodo link in data_io/kelmarsh_io.py\n")
        return

    test_schema_is_frozen()
    test_grid_is_regular()
    test_windows_are_well_formed()
    test_windows_are_contiguous_in_time()
    test_labels_look_right()
    test_farm_assembles()
    print("\nkelmarsh loader smoke test complete\n")


if __name__ == "__main__":
    test()
