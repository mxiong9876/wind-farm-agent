"""
Getting data into ScadaEncoder and results out of it.

Covers the three things that actually break in practice:
  1. frozen channel ordering (the channel embedding indexes into it)
  2. alarm/state codes mapped to contiguous ids with an UNKNOWN bucket
  3. real patch timestamps when the SCADA grid has gaps
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from legacy.scada_encoder_patch import ScadaEncoder, patch_center_times


# ---------------------------------------------------------------------------
# 1. Freeze the schema. Save this alongside the checkpoint -- if the order
#    changes between pretraining and finetuning, every channel embedding is
#    silently wrong and the model will train to mediocre and stay there.
# ---------------------------------------------------------------------------
CONTINUOUS = [
    "wind_speed", "wind_dir", "nacelle_pos", "yaw_error",
    "rotor_rpm", "gen_rpm", "pitch_1", "pitch_2", "pitch_3",
    "active_power", "reactive_power",
    "gearbox_oil_temp", "gearbox_brg_temp_ds", "gearbox_brg_temp_nds",
    "gen_winding_u", "gen_winding_v", "gen_winding_w",
    "nacelle_temp", "ambient_temp", "tower_accel_fa",
]
CATEGORICAL = ["turbine_state", "alarm_code", "curtailment_flag"]
CARDINALITIES = (12, 256, 2)          # reserve index 0 as UNKNOWN in each
STATIC = ["doy_sin", "doy_cos", "hour_sin", "hour_cos",
          "op_hours", "load_cycles", "days_since_maint"]

SAMPLE_PERIOD = 1.0      # seconds between SCADA rows
CONTEXT_LEN = 600        # 10 minutes
PATCH_LEN, STRIDE = 30, 15


class ScadaWindowDataset(Dataset):
    """Slices a per-turbine SCADA dataframe into fixed windows.

    Expects a DatetimeIndex on a regular grid, with genuinely missing rows
    present as NaN rather than dropped -- the encoder uses missingness as
    signal, so silently collapsing gaps destroys information.
    """

    def __init__(self, df: pd.DataFrame, turbine_type: int = 0,
                 context_len: int = CONTEXT_LEN, step: int = 60,
                 static_scaler: dict | None = None):
        self.df = df
        self.turbine_type = turbine_type
        self.context_len = context_len
        self.starts = list(range(0, len(df) - context_len + 1, step))
        self.static_scaler = static_scaler or {}

        # (T, C) float arrays, NaN preserved
        self.cont = df[CONTINUOUS].to_numpy(dtype=np.float32)
        self.cat = df[CATEGORICAL].to_numpy(dtype=np.int64)
        self.static = df[STATIC].to_numpy(dtype=np.float32)
        # seconds since the start of the frame. Do NOT use .view("int64")/1e9 --
        # pandas 3.x stores datetimes at microsecond resolution by default, so
        # that silently yields times 1000x too small.
        self.times = (df.index - df.index[0]).total_seconds().to_numpy(dtype=np.float64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        e = s + self.context_len

        block = self.cont[s:e]                       # (T, C)
        mask = (~np.isnan(block)).astype(np.float32)
        block = np.nan_to_num(block, nan=0.0)

        # channels-first: the encoder wants (C, T)
        x = torch.from_numpy(block.T.copy())
        m = torch.from_numpy(mask.T.copy())

        # categoricals: take the value at the window's right edge ("now")
        cat = torch.from_numpy(self.cat[e - 1].copy())

        stat = torch.from_numpy(self.static[e - 1].copy())

        # real patch times from actual row timestamps, so gaps are honest
        t = self.times[s:e]
        centers = (np.arange((self.context_len - PATCH_LEN) // STRIDE + 1)
                   * STRIDE + PATCH_LEN // 2).astype(int)
        patch_t = torch.from_numpy((t[centers] - t[-1]).astype(np.float32))

        return {
            "x": x, "mask": m, "categorical": cat,
            "static_feats": stat, "patch_times": patch_t,
            "turbine_type": torch.tensor(self.turbine_type),
        }


def map_codes(series: pd.Series, vocab: dict) -> pd.Series:
    """Alarm codes are sparse and unbounded. Map known ones to 1..N,
    everything unseen to 0, so a novel fault code at inference does not
    index out of the embedding table."""
    return series.map(vocab).fillna(0).astype(np.int64)


# ---------------------------------------------------------------------------
# 2. Feed it
# ---------------------------------------------------------------------------
def build_encoder():
    return ScadaEncoder(
        n_continuous=len(CONTINUOUS),
        categorical_cardinalities=CARDINALITIES,
        context_len=CONTEXT_LEN,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        d_model=128,
        n_latents=32,
        n_static_feats=len(STATIC),
        sample_period=SAMPLE_PERIOD,
    )


def pretrain_step(enc, batch, opt):
    """Self-supervised: no labels needed, just normal-operation data."""
    loss = enc.masked_reconstruction_loss(batch["x"], batch["mask"], mask_ratio=0.4)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
    opt.step()
    return loss.item()


@torch.no_grad()
def encode(enc, batch):
    """Inference: dataframe window -> (B, 32, 128) tokens for fusion."""
    enc.eval()
    return enc(
        batch["x"], batch["mask"],
        categorical=batch["categorical"],
        turbine_type=batch["turbine_type"],
        patch_times=batch["patch_times"],
        static_feats=batch["static_feats"],
    )


if __name__ == "__main__":
    # synthetic SCADA standing in for a real feed
    n = 3000
    idx = pd.date_range("2026-01-01", periods=n, freq="1s")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(size=(n, len(CONTINUOUS))).astype(np.float32),
        columns=CONTINUOUS, index=idx,
    )
    df.loc[df.index[500:560], "gearbox_oil_temp"] = np.nan   # a comms dropout
    for c, card in zip(CATEGORICAL, CARDINALITIES):
        df[c] = rng.integers(0, card, n)
    for c in STATIC:
        df[c] = rng.normal(size=n).astype(np.float32)

    ds = ScadaWindowDataset(df, turbine_type=0, step=60)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    enc = build_encoder()
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-4, weight_decay=0.01)

    batch = next(iter(dl))
    print("windows in dataset:", len(ds))
    print("x", tuple(batch["x"].shape), "| mask", tuple(batch["mask"].shape))
    print("patch_times[0][:4]", batch["patch_times"][0][:4].tolist())
    print("pretrain loss:", round(pretrain_step(enc, batch, opt), 4))

    tokens = encode(enc, batch)
    print("-> fusion tokens:", tuple(tokens.shape))