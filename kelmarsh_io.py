"""Load Kelmarsh wind farm SCADA into ScadaTCNEncoder.

Data: https://zenodo.org/records/5841834  (Cubico, CC-BY-4.0)
6x Senvion MM92, 10-minute SCADA + status events, 2016 to mid-2021.

ONE CSV ROW IS ONE TIMESTEP. Greenbyte exported 10-minute averages, so a
600-step window spans 600 * 10 min = 4.2 DAYS, not 10 minutes. The encoder's
defaults were sized for 1 Hz SCADA where 600 steps meant 10 minutes; the tensor
shape is identical and the physical meaning is 600x larger. That is the right
scale for thermal degradation, and it makes dilation 128 reach back 21.3 hours
-- close enough to a day that the model can compare like hour with like.
"""

import glob
import os

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# FROZEN SCHEMA. Save this alongside any checkpoint.
#
# ScadaTCNEncoder.channel_embed indexes into this list POSITIONALLY. Reorder it
# between pretraining and finetuning and every channel embedding is silently
# wrong -- the model trains to mediocre and stays there, with nothing in the
# loss curve to tell you why. Append only; never insert or reorder.
#
# Names must match the CSV headers exactly, degree signs included.
# ---------------------------------------------------------------------------
CHANNELS = [
    # -- operating point: what the turbine was asked to do --------------------
    "Wind speed (m/s)",
    "Wind direction (°)",
    "Nacelle position (°)",
    "Rotor speed (RPM)",
    "Generator RPM (RPM)",
    "Blade angle (pitch position) A (°)",
    "Blade angle (pitch position) B (°)",
    "Blade angle (pitch position) C (°)",
    "Power (kW)",
    "Reactive power (kvar)",
    # -- drivetrain thermals: where degradation actually shows up -------------
    "Gear oil temperature (°C)",
    "Gear oil inlet temperature (°C)",
    "Front bearing temperature (°C)",
    "Rear bearing temperature (°C)",
    "Generator bearing front temperature (°C)",
    "Generator bearing rear temperature (°C)",
    "Stator temperature 1 (°C)",
    # -- context: a temperature only means something against a reference ------
    "Nacelle temperature (°C)",
    "Nacelle ambient temperature (°C)",
    # -- the closest thing to vibration in 10-minute data ---------------------
    "Drive train acceleration (mm/ss)",
]

N_CHANNELS = len(CHANNELS)          # 20, matching ScadaTCNEncoder's default
SAMPLE_PERIOD = pd.Timedelta("10min")


def load_turbine(csv_path):
    """One Turbine_Data_*.csv -> DataFrame on a gapless 10-minute grid.

    Rows absent from the export become explicit NaN rows, so downstream code
    sees one row per 10 minutes with no silent time jumps.
    """
    df = pd.read_csv(csv_path, skiprows=9, low_memory=False)
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    grid = pd.date_range(df.index.min(), df.index.max(), freq=SAMPLE_PERIOD)
    return df.reindex(grid)[CHANNELS]

def make_windows(df, context_len=600, stride=144, min_coverage=0.5):
    """DataFrame -> (x, mask, ends) ready for ScadaTCNEncoder.

    x, mask : (B, 20, context_len) float32,  mask 1 = observed, 0 = missing
    ends    : (B,) timestamps of each window's LAST row

    stride=144 is one day between window starts. The encoder default of 15 was
    sized for 1 Hz data; here it would mean 150 minutes and give windows that
    overlap 97.5%, which is fine as augmentation but makes an honest train/test
    split much harder to reason about.
    """
    values = df.values.astype("float32")            # (T_all, 20)
    mask = np.isfinite(values).astype("float32")
    values = np.nan_to_num(values, nan=0.0)         # missing reads as 0

    xs, ms, ends = [], [], []
    for i in range(0, len(df) - context_len + 1, stride):
        sl = slice(i, i + context_len)
        if mask[sl].mean() < min_coverage:          # window is mostly hole
            continue
        xs.append(values[sl].T)                     # (20, context_len)
        ms.append(mask[sl].T)
        ends.append(df.index[i + context_len - 1])
  
    return (torch.from_numpy(np.stack(xs)),
            torch.from_numpy(np.stack(ms)),
            pd.DatetimeIndex(ends))


# ---------------------------------------------------------------------------
# labels
#
# Filter on IEC CATEGORY, not on Status. The three generator-fan overloads --
# the most common genuine failure in 2016 -- are logged as `Warning`, so
# `Status == "Stop"` silently drops them. Meanwhile 258 of the 751 `Stop`
# events in 2016 are "Battery test", which is routine maintenance. Label every
# stop as a fault and you will build an excellent battery-test detector.
# ---------------------------------------------------------------------------
FORCED_OUTAGE = "Forced outage"
HORIZON = pd.Timedelta("7D")


def load_status(csv_path, category=FORCED_OUTAGE):
    """Status_*.csv -> DatetimeIndex of event START times.

    Only starts matter: the label asks when a failure BEGINS, so an outage that
    runs for nine days is one event, not nine days of them.
    """
    ev = pd.read_csv(csv_path, skiprows=9)
    if category is not None:
        ev = ev[ev["IEC category"] == category]
    return pd.DatetimeIndex(pd.to_datetime(ev["Timestamp start"])).sort_values()


def label_windows(ends, starts, horizon=HORIZON):
    """1 if a failure begins within `horizon` AFTER the window ends.

    Strictly after. Letting the horizon overlap the window would let the model
    read the outage out of its own input, which turns prediction into a lookup
    and produces a wonderful score that means nothing.
    """
    e = ends.values[:, None]                      # (B, 1)
    s = starts.values[None, :]                    # (1, n_events)
    hit = (s > e) & (s <= e + horizon.to_timedelta64())
    return torch.from_numpy(hit.any(axis=1).astype("float32"))


def load_farm(scada_dir, turbines=range(1, 7), **kw):
    """Every turbine in a year directory -> (x, mask, ends, turbine_id).

    turbine_id is carried through because it is the only honest way to build a
    leave-one-turbine-out split later, and because the six machines are far from
    interchangeable: their forced-outage rates over 2016 span 3% to 31%.
    """
    xs, ms, es, ids, ys = [], [], [], [], []
    for t in turbines:
        hits = glob.glob(os.path.join(
            scada_dir, f"Turbine_Data_Kelmarsh_{t}_*.csv"))
        if not hits:
            continue
        x, m, ends = make_windows(load_turbine(hits[0]), **kw)
        xs.append(x)
        ms.append(m)
        es.append(ends)
        ids.append(torch.full((len(x),), t, dtype=torch.long))

        # labels are per turbine: an outage on turbine 3 says nothing about
        # turbine 4, so the status file has to be joined inside this loop
        stat = glob.glob(os.path.join(scada_dir, f"Status_Kelmarsh_{t}_*.csv"))
        ys.append(label_windows(ends, load_status(stat[0])) if stat
                  else torch.zeros(len(x)))

    return (torch.cat(xs), torch.cat(ms),
            pd.DatetimeIndex(np.concatenate(es)),
            torch.cat(ids), torch.cat(ys))


if __name__ == "__main__":
    SCADA = "data/kelmarsh/scada_2016"

    df = load_turbine(glob.glob(
        os.path.join(SCADA, "Turbine_Data_Kelmarsh_1_*.csv"))[0])
    print(f"turbine 1 grid   {df.shape}   "
          f"gaps {(df.index.to_series().diff().dropna() != SAMPLE_PERIOD).sum()}")

    x, mask, ends = make_windows(df)
    print(f"turbine 1 windows {tuple(x.shape)}   mask mean {mask.mean():.3f}")
    print(f"  span  {ends.min()} -> {ends.max()}")
    # RevIN multiplies by the mask, but a nonzero value under mask=0 would still
    # be a bug: it means nan_to_num ran before the mask was built.
    print(f"  x is 0 wherever mask is 0: {bool((x[mask == 0] == 0).all())}")

    X, M, E, T, Y = load_farm(SCADA)
    print(f"\nfarm windows     {tuple(X.shape)}   mask mean {M.mean():.3f}")
    print(f"  per turbine    {torch.bincount(T)[1:].tolist()}")
    print(f"  dtype          {X.dtype}   finite {bool(torch.isfinite(X).all())}")

    print(f"\nlabels ({HORIZON.days}-day horizon, '{FORCED_OUTAGE}')")
    print(f"  positive rate  {Y.mean():.3f}   ({int(Y.sum())} of {len(Y)})")
    for t in torch.unique(T).tolist():
        yt = Y[T == t]
        print(f"    turbine {t}    {yt.mean():.3f}  ({int(yt.sum()):>3} of {len(yt)})")
    # what 'predict never fails' scores. If you report accuracy, this is the
    # number you are actually competing with -- which is why PR-AUC is the
    # metric and accuracy is not.
    print(f"  majority acc   {max(Y.mean(), 1 - Y.mean()):.3f}")