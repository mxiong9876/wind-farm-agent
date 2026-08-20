# scripts/

Everything you *run*. `models/` holds architecture (`nn.Module` classes and
nothing else), `data_io/` turns archives into tensors, `tests/` checks that both
behave. This directory is the entry points.

Run every script from the repository root with the conda interpreter — the
project `.venv` has no PyTorch:

```bash
/opt/miniconda3/bin/python3 scripts/<name>.py
```

Each script inserts the repository root on `sys.path` itself, so the working
directory only matters for the default relative paths (`data/kelmarsh`,
`runs/…`).

---

## Terms used below

Expanded once here, then used freely.

| Term | Meaning |
|---|---|
| **SCADA** — Supervisory Control And Data Acquisition | The turbine's own operational telemetry. At Kelmarsh, one row every 10 minutes per turbine: wind speed, power, temperatures, pitch angles. |
| **NBM** — Normal-Behaviour Modelling | Predict what a sensor *should* read given every other sensor, then treat the **residual** (actual − predicted) as the health signal. A component running hotter than its operating conditions justify is deviating from normal. |
| **Residual** | actual − predicted, in the sensor's own units (°C here). Near zero = behaving as conditions predict. Drifting upward over weeks = the degradation signature. |
| **PR-AUC** — Precision-Recall Area Under Curve | The metric for rare events. Its trivial baseline is the positive rate itself (~14% here), so unlike accuracy it cannot be gamed by predicting "healthy" forever. |
| **R²** — coefficient of determination | Share of variance explained. 0 = no better than guessing the mean; 1 = exact. Can go negative, which means worse than a constant — usually a sign that train and test are different distributions. |
| **ViT** — Vision Transformer | The image-encoder architecture behind DINOv2, used frozen in `models/rgb_ViT.py`. |
| **VLM** — Vision-Language Model | A model taking images *and* text and generating text. Here: a frozen Qwen3-VL that reads the fusion output as **soft tokens**. |
| **LLM** — Large Language Model | The language half of a VLM. Used loosely below when the vision tower is not involved. |
| **Soft tokens** | Vectors spliced into a language model's input sequence that were never in its vocabulary. The model attends over them exactly as if they were words. |
| **Projector** | The small trainable adapter mapping the 128-dimensional fusion vector into the language model's embedding width. ~4.5M parameters against a frozen 2.1B. |
| **Fusion / health vector** | `MultiModalFusion` collapses every present modality into one 128-dimensional vector per turbine-window. |
| **Held-out year** | A whole calendar year excluded from training. Splitting *within* a year puts summer in train and winter in test, and every channel here has an annual cycle. |

---

## Getting data

### `fetch_kelmarsh.sh`

Downloads the Kelmarsh wind farm archives from Zenodo (record 5841834,
CC-BY-4.0): six Senvion MM92 turbines, 10-minute SCADA plus status/event logs,
2016 to mid-2021. About 6 GB into `data/kelmarsh/`.

```bash
./scripts/fetch_kelmarsh.sh
```

Retries and resumes. **This matters more than it sounds**: a truncated archive
fails at `unzip` time, not download time, so a naive loop reports success and
leaves an empty directory. Every archive is verified with `unzip -t` before it
counts as done. Five of six years needed a retry on the first run.

---

## Training

### `train_nbm.py` — the experiment that worked

Trains the SCADA encoder to predict one sensor from the other nineteen, with a
whole year held out.

```bash
python3 scripts/train_nbm.py                                    # ~1 hr on Apple GPU
python3 scripts/train_nbm.py --epochs 3                         # quick look
python3 scripts/train_nbm.py --target "Front bearing temperature (°C)"
python3 scripts/train_nbm.py --plot-only                        # redraw from history
```

**Result: R² 0.941 on a held-out year, against 0.856 for ridge regression on
the 19 channel means.** That linear baseline is the bar — twenty parameters and
one closed-form solve — and the script computes it every run and draws it on
the plot labelled "← the bar". A result below that line means 3.5M parameters
bought nothing over averaging.

`--static-means` is the flag that makes the comparison fair. `RevIN` inside the
encoder strips each channel's window mean before the trunk sees it, which is
correct for condition monitoring (you want "hotter than its own baseline", not
absolute degrees) but leaves the encoder blind to exactly the quantity a
mean-regression task rewards. Passing the means back as static features gives
the encoder the baseline's entire input *plus* the temporal structure.

Writes `kelmarsh_nbm_history.csv` every epoch — not just at the end, so a run
killed at hour two keeps everything it computed.

### `train_kelmarsh.py` — the experiment that didn't

Trains the same encoder to predict forced outages within a 7-day horizon.

```bash
python3 scripts/train_kelmarsh.py --test-year 2019
```

**Result: no signal.** Training loss fell while held-out PR-AUC fell with it,
below the base rate by epoch 6. Kept because the negative result is
informative: most forced outages at Kelmarsh are grid loss, converter faults
and emergency stops, which have no thermal precursor. Nothing in four days of
temperature history predicts them, and no amount of architecture fixes that.

Runs a shuffled-label control automatically. If that does not collapse to the
base rate, the split leaked and the headline number is void.

### `run_nbm_suite.sh`

Three `train_nbm.py` runs back to back — the headline plus two robustness
checks (a different target, a different held-out year). Sequential on purpose:
one GPU, so parallel runs contend and finish no sooner.

```bash
EPOCHS=8 ./scripts/run_nbm_suite.sh
```

---

## Analysis

### `residual_analysis.py`

Asks whether NBM residuals rise *before* a fault is logged — the question the
whole health-monitoring story rests on.

```bash
python3 scripts/residual_analysis.py --ckpt runs/genbrg_static/run.pt
```

Aligns every thermal forced-outage incident at day 0 and averages the residuals
around it, for **both** the encoder and the linear baseline, against a control
of random no-fault dates.

**Result: no warning signal** across 10 incidents in the held-out year. Two
things make that honest rather than damning: 10 incidents is far too few to
conclude anything, and "generator fan overload" — the most common fault here —
is a protection trip that can happen in seconds, with the generator heating
*after* the fan stops rather than before.

One real finding did come out of it: the encoder's residuals are **half as
noisy** as the linear baseline's (standard deviation 0.592 vs 1.001). A tighter
normal-behaviour model means a smaller deviation is detectable, which is the
R² advantage showing up where it should.

### `turbine_report.py`

Turns model outputs into a text brief a person — or a language model — can read.

```bash
python3 scripts/turbine_report.py --turbine 3                  # brief only
python3 scripts/turbine_report.py --turbine 3 --at 2016-07-01
python3 scripts/turbine_report.py --turbine 3 --llm            # + written assessment
```

The brief needs no API key and no network: residual, its 30-day trend, the
operating context that explains it, the fault history to read it against, and
an explicit statement of the brief's own limitations (SCADA only, 10-minute
averages, "a residual is a deviation, not a diagnosis").

`--llm` sends that brief to Claude for a written assessment and needs
`ANTHROPIC_API_KEY`. The brief is the deliverable; the prose is a wrapper. If
the prose is ever wrong, the brief is what you check it against.

---

## The language-model path

Three scripts, run in order. **None has been trained yet** — the pipeline runs
end to end but the projector is untrained, so the generated text is fluent
noise. That is the expected state, not a bug.

### 1. `pipeline_demo.py` — proves the wiring

```bash
python3 scripts/pipeline_demo.py                    # shapes only, ~6 s
python3 scripts/pipeline_demo.py --vlm --generate   # + the language model, ~25 s
```

Four modalities → encoders → fusion → projector → Qwen3-VL → text, in one
forward pass. Shows the token-count imbalance the per-modality resamplers exist
to remove (48 to 1370 tokens in, 32 latents each out), and demonstrates
presence masking on the live stack by withholding modalities.

Real Kelmarsh SCADA; correctly-shaped noise for vibration, imagery and acoustic,
because only SCADA has real data on this project.

### 2. `build_vlm_pairs.py` — manufactures training data

```bash
python3 scripts/build_vlm_pairs.py --limit 200      # sanity check
python3 scripts/build_vlm_pairs.py                  # full run
```

Writes `(health vector, brief, target assessment)` triples to `runs/vlm_pairs/`.

**The one design decision that decides whether any of this works:** the target
says something the brief does not. The brief carries operating context and
fault history and never mentions the residual; the target carries the residual
finding and its trend. So the residual is reachable *only* through the health
vector.

Get that wrong — state the residual in the brief — and the model learns
brief → target, routes around the soft tokens entirely, and the loss still
falls. You would see a trained-looking run and a decorative adapter.

### 3. `train_vlm_projector.py` — trains the adapter

```bash
python3 scripts/train_vlm_projector.py --epochs 1 --limit 32   # smoke run
python3 scripts/train_vlm_projector.py                          # real run
python3 scripts/train_vlm_projector.py --model Qwen/Qwen3-VL-8B-Instruct
```

Trains 4.5M projector parameters with the 2.1B language model frozen. Gradient
still flows backward *through* the frozen model to reach the projector, so the
memory cost is a full backward pass even though almost nothing updates — which
is what decides where it can run. The 2B model fits on a laptop; the 8B does
not.

**Read the ablation, not the loss.** Every epoch the script re-scores the
validation set with health vectors shuffled between samples:

```
val 1.82   shuffled 1.84   → ignoring the health vector
val 1.82   shuffled 3.10   → genuinely conditioned on it
```

Shuffling preserves the marginal distribution, so any loss increase is
attributable to the *correspondence* between a health vector and its text. The
script prints a warning when the gap is under 0.15, because that failure mode
looks exactly like success on the loss curve alone.

---

## Conventions these scripts share

**Split by year, never within one.** Cutting a single year at 70% puts summer
in train and winter in test. Measured on 2016: a probe scored R² −12.5
predicting held-out ambient temperature and *below chance* on month. The
features were fine; the split made train and test different domains, and the
probe reported that as "no signal".

**Every result gets a baseline and a negative control.** A number alone means
nothing. `train_nbm.py` draws the linear baseline on its plot;
`train_kelmarsh.py` runs a shuffled-label control; `train_vlm_projector.py`
runs the shuffled-health ablation; `residual_analysis.py` uses random no-fault
dates. Skip these and you will believe a number that measures nothing.

**Write history every epoch, and flush.** Long runs get killed — a closed
laptop, a session ending, an out-of-memory kill. Writing results only on
completion throws away hours already computed, and block-buffered output means
a killed run leaves an empty log rather than the epochs that succeeded.

**Checkpoints and outputs live in `runs/`**, which is gitignored. Promote a
figure to `figures/` by hand if you want it tracked.
