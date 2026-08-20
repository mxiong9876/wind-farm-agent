# Wind Farm Agent

A multimodal condition-monitoring system for wind turbines. Sensor streams of
wildly different shapes and sample rates are encoded separately, fused into one
health state per turbine, and read out either by small task heads or by a frozen
language model that writes a maintenance assessment.

Built during a summer internship, 2026. Real data throughout is the **Kelmarsh
wind farm** open dataset — six Senvion MM92 turbines, 10-minute telemetry and
status logs, 2016 to mid-2021.

---

## Architecture

```
   SCADA          vibration          imagery          acoustic
 (20, 600)       (8, 25600)      (3, 518, 518)      (4, 16000)
 4.2 days at     ~1 s at          blade photo       microphone
 10-min rows     25.6 kHz                            array
      │               │                 │                │
 ScadaTCN      VibrationConv2d      RGBEncoder       StubEncoder
  Encoder         Encoder         (frozen DINOv2)   (placeholder)
      │               │                 │                │
  800 tokens      48 tokens        1370 tokens       64 tokens
      │               │                 │                │
      └───────────────┴────────┬────────┴────────────────┘
                               │
                  one PerceiverResampler per modality
                               │
                    32 latents each, 128 wide
                               │
                     MultiModalFusion
              (self-attention + presence masking)
                               │
                    health vector (128,)
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   diagnostic heads      control head          FusionProjector
   Linear(128, k)        (planned)             128 → 8 × 2048
   129 params each            │                 4.5M params
        │              ├ yaw offset  (deg)            │
  fault probability    ├ pitch + torque         Qwen3-VL (frozen)
  predicted temp       └ curtailment mode              │
  remaining life         {normal, derate, stop}  maintenance
                                                  assessment
```

**The two output paths have different requirements**, and the repository already
records the difference. `RevIN` normalizes by whole-window statistics, which is
deliberately non-causal — a change at t=500 moves the mean applied at t=100.
That is correct for the health path, which analyses a fixed historical window
after the fact. It is wrong for control, where future samples do not exist at
decision time.

So the control path needs three things the health path does not:

| | Health path | Control path |
|---|---|---|
| Normalization | `RevIN`, whole-window | running statistics |
| Causality | non-causal by design | must hold end to end |
| Window | 4.2 days of 10-minute rows | seconds to minutes |

The causality gap is measured, not assumed: `tests/scada/test_stage8_tokens.py`
prints the leak through the full encoder as INFO rather than asserting on it,
precisely because RevIN makes it expected. Swap in running normalization and
that INFO becomes an assertion — which is the test that gates the control path.

The sample-rate row is the awkward one. Yaw and curtailment decisions live
comfortably at 10-minute resolution; pitch and torque loops run at Hz, six
orders of magnitude faster than the health path's windows. Those three outputs
likely share the representation but not the input geometry.

**Nothing here is built yet.** The control head is planned; the diagnostic heads
and the language path exist.

**Why per-modality resamplers.** Encoders emit 48 to 1370 tokens — a 28× spread.
Concatenating raw tokens into one shared resampler would make imagery 1370 of
2282 keys, and a resampler at initialization behaves close to uniform averaging,
so vibration would start at a 2% contribution and have to climb out. One
resampler each removes the imbalance by construction.

**Two kinds of missing, kept separate.** A dead accelerometer on a turbine that
*has* vibration monitoring is handled inside that encoder by its channel mask.
A turbine with no vibration hardware at all is handled by the fusion's
`present` flags, which gate whole 32-latent blocks out of every attention. Conflating them means feeding an
encoder all-zeros and hoping the tokens come out neutral — they do not, because
"this sensor reads zero" is a different claim from "this sensor does not exist".

**Why the language model is frozen.** Only the 4.5M-parameter projector trains.
Gradient still flows backward *through* the frozen 2.1B model to reach it, so
the memory cost is a full backward pass — which is what decides where training
can run.

---

## Repository layout

```
models/      architecture — nn.Module classes, no training loops
data_io/     archives → tensors
scripts/     entry points (see scripts/README.md)
tests/       six suites, one per component
data/        raw archives (gitignored)
runs/        training outputs (gitignored)
figures/     plots promoted by hand (tracked)
legacy/      the previous patch-based encoder, superseded
```

### `models/`

| File | Lines | What |
|---|---:|---|
| `scada_encoder_tcn.py` | 184 | Dilated causal TCN over 20 channels × 600 steps. RevIN, channel embeddings, folding to `B×C` sequences. 848k parameters. |
| `vibration_encoder_2dconv.py` | 323 | Digital signal processing frontend (filterbank + envelope demodulation) into a 2-D convolutional trunk. Separates a bearing defect by its modulation when total energy is matched. 931k parameters. |
| `rgb_ViT.py` | 155 | Frozen DINOv2 ViT plus a LayerNorm and one Linear. 100k trainable against 86.6M frozen. |
| `multimodal_fusion.py` | 318 | The registry: N optional modalities, per-modality resamplers, presence masking, modality dropout, masked attention pooling. |
| `common.py` | 69 | Shared `PerceiverResampler`, `ResamplerLayer`, `ContinuousTimeEncoding`. |
| `vlm_bridge.py` | 264 | Projector + frozen Qwen3-VL. Splices the health vector in as soft tokens. |

Each file's module docstring carries its input/output contract and the reasoning
behind the choices that are not obvious.

### `data_io/`

`kelmarsh_io.py` — CSV archives to `(x, mask, timestamps, turbine_id, labels)`.
Frozen 20-channel schema, gapless 10-minute grid, windowing, and the
forced-outage label join.

Not named `io/`: a package called `io` at the repository root shadows Python's
standard-library `io` for anything that puts the root on `sys.path` — which
every test bootstrap does — and pandas, torch and numpy all import it.

### `tests/`

```bash
python3 tests/run_all.py          # all six suites
python3 tests/scada/run_all.py    # one suite
```

| Suite | Files | Covers |
|---|---:|---|
| `scada/` | 12 | 10 stage tests (causality, RevIN, folding, channel identity, resampler) + smoke test |
| `vibration/` | 5 | Signal-processing frontend, conv trunk, tokens, smoke test |
| `fusion/` | 5 | Routing, presence masking, modality dropout, smoke test |
| `kelmarsh/` | 2 | Loader: grid regularity, window shapes, label boundaries |
| `rgb/` | 2 | Frozen-backbone checks, variable image size, defect-signal probe |
| `vlm/` | 2 | The soft-token splice: scale matching, label masking, gradient routing |

**Stage tests** check the forward pass and run in seconds. **Smoke tests** train
and take minutes — they answer a different question: can this thing actually
learn. The last three suites skip cleanly when their data or checkpoint is
absent, so a fresh clone still gets a green suite.

---

## Results

**The SCADA encoder beats a 20-parameter linear baseline on real turbine data.**

| | R² on a held-out year |
|---|---|
| Frozen encoder, no training | 0.333 |
| **Ridge on 19 channel means — the bar** | **0.856** |
| Trained encoder | **0.941** |

That bar matters more than the headline. Twenty parameters and one closed-form
solve is what 3.5M parameters have to beat to justify themselves, and it is
computed on every run and drawn on every plot.

**The fusion carries both modalities, provably.** On a synthetic task where two
independent faults are each visible to only one sensor, information theory caps
any single modality at 0.790. The fused model reaches 0.948 — which cannot be
reached by one branch getting lucky.

**Two honest negatives.** Predicting forced outages from four days of SCADA does
not work (most Kelmarsh outages are grid loss, converter faults and emergency
stops, which have no thermal precursor). Residuals do not visibly rise before
faults across the 10 incidents available in the held-out year — too few to
conclude much, but worth reporting rather than burying.

---

## Status

| Component | State |
|---|---|
| SCADA encoder | ✅ trained and validated on real data |
| Vibration encoder | ✅ built and tested — synthetic data only |
| RGB encoder | ✅ built and tested — synthetic data only |
| Fusion | ✅ validated against an information-theoretic ceiling |
| Kelmarsh loader | ✅ real data, 24 assertions |
| Language-model bridge | ⚠️ wired and tested, **projector untrained** |
| Control head | ❌ planned — needs running-statistics normalization first |

The full pipeline runs end to end — four modalities through 2.1B parameters in
about 25 seconds on a laptop. The generated text is fluent noise, because the
projector has never been trained. Every shape, dtype, mask and scale is
verified; the meaning is not there yet.

**The one blocker** is paired (sensor window, text) training data.
`scripts/build_vlm_pairs.py` manufactures it from residuals, status logs and
templates; `scripts/train_vlm_projector.py` trains on it. Both are wired and
neither has been run.

---

## Setup

```bash
/opt/miniconda3/bin/python3          # the interpreter with PyTorch
./scripts/fetch_kelmarsh.sh          # ~6 GB of real data
python3 tests/run_all.py             # confirm the install
```

The project `.venv` is empty — use the conda interpreter. `python3` on this
machine resolves to a framework Python without PyTorch, which is the most common
way to get a confusing `ModuleNotFoundError` here.

Data: [Kelmarsh wind farm](https://zenodo.org/records/5841834), Cubico
Sustainable Investments, CC-BY-4.0.
