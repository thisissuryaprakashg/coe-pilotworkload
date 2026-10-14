# New Data Pipeline + Nonlinear Mathematical Model for Pilot Workload Prediction

## Problem

Previous pipeline achieved **~40% accuracy** (4-class LOSO) with ordinal/multinomial logistic regression (linear model).

## What Went Wrong in Previous Pipeline

1. **Single summary over entire ~5min run** — collapses all temporal dynamics into one row. Workload evolves during flight approach; early vs late phases differ massively. This is the biggest issue.
2. **Too few features extracted per modality** — ECG: 10, EDA: 5, EMG: 3. Lots of signal information thrown away.
3. **Silent exception swallowing** — `except Exception: pass` everywhere, failures go unnoticed.
4. **Missing modalities never used** — Aircraft state (`lslxp11xpcac`), pilot head tracking (`lslxp11xpcplt`), oculomotor events (`ocuevts`) all ignored.
5. **Linear model ceiling** — Physiological responses are nonlinear (saturation, threshold effects). Logistic regression can't capture this.

## New Approach: 30-Second Temporal Windowing

Instead of 1 row per 5-min run, we split each run into **30-second windows** and extract features per window.

- Each ~5 min ILS run → **~10 windows** of 30 seconds
- ~408 runs × ~10 windows = **~4000+ samples** (10x more data)
- Each window inherits the run's difficulty label (difficulty is constant per run)
- LOSO by subject — all windows from a subject go to train or test together
- 30 sec is enough for time-domain HRV (RMSSD, SDNN, pNN50), EMG RMS, EDA, pupil stats
- LF/HF ratio dropped for windowed features (needs 2+ min) — kept only for run-level if needed

## Proposed Changes

### Step 0: Setup venv ✅ (done)

### Step 1: `new_data_pipeline.py`

Single self-contained script. For each run:
1. Load all streams (ECG, EDA, EMG, Eye, Aircraft, Head, PerfMetric time-series)
2. Align timestamps, split into 30-sec windows
3. Per window extract:

**ECG (per 30-sec window):**
- HR mean, std, range, slope (trend within window)
- RMSSD, SDNN, pNN50
- RR interval coefficient of variation
- Sample entropy of RR intervals (nonlinear HRV)

**EDA (per 30-sec window):**
- Tonic (SCL) mean, slope
- SCR count, mean amplitude, max amplitude
- Phasic energy (sum of squared phasic)

**EMG (per 30-sec window):**
- Flexor RMS, extensor RMS
- Co-contraction index (flexor × extensor correlation)
- Median frequency via PSD (fatigue marker)
- Burst count (threshold crossings)
- Forearm acceleration RMS

**Eye Tracking (per 30-sec window):**
- Pupil diameter: mean, std, skew, kurtosis (L+R averaged)
- Pupil velocity (rate of change)
- PERCLOS (% time eyes >80% closed)
- Blink rate, mean blink duration
- Gaze entropy X, Y (spatial scanning diversity)
- Convergence distance mean, std

**Aircraft State (per 30-sec window):**
- ILS glideslope deviation: mean, std, RMS
- ILS localizer deviation: mean, std, RMS
- Airspeed: mean, std, range
- Altitude rate: mean, std
- Roll/pitch variability (control effort)

**Head Tracking (per 30-sec window):**
- Head position RMS (movement magnitude)
- Head yaw/pitch/roll variability (scanning behavior)

**Oculomotor Events (per 30-sec window):**
- Fixation count, mean duration, std duration
- Saccade count, mean amplitude, mean peak velocity

4. Compute **resting baseline** (from task-rest level-000_run-001) — same windowing, take median across rest windows
5. Compute **delta features** = task_window - rest_baseline
6. Join `devSubjsFeatMat.csv` features at run level (replicated to each window)
7. Join `PerfMetrics.csv` ground truth difficulty labels
8. Apply cleaning rules (exclude bad runs, mask bad eye tracking)
9. Output: `new_master_features.csv`

**Metadata columns per row:** subject, session, run_folder, window_index, difficulty_level

### Step 2: `nonlinear_model.py`

Loads `new_master_features.csv`, fits nonlinear mathematical models via LOSO:

1. **Feature screening** — ANOVA F-test per feature, keep p < 0.05
2. **VIF pruning** — remove multicollinear features (VIF > 10)
3. **Standardization** — z-score within training folds only

**Models (all mathematical, no ML):**
- **Polynomial Logistic Regression (degree 2)** — quadratic terms + interactions of top features
- **Kernel Ridge Regression** — RBF kernel, regularized, closed-form solution
- **Spline Logistic Regression** — natural cubic spline basis expansion + logistic regression

**Evaluation (LOSO by subject):**
- 4-class strict accuracy
- Adjacent accuracy (±1 level)
- Binary accuracy (levels 1-2 vs 3-4)
- Macro F1, Cohen's Kappa

All results printed to terminal.

## File Summary

| # | File | What it does |
|---|------|-------------|
| 1 | `new_data_pipeline.py` | 30-sec windowed feature extraction from all modalities |
| 2 | `nonlinear_model.py` | Feature screening + nonlinear math model fitting + LOSO evaluation |

## Verification Plan

### Manual Verification
- Run `python new_data_pipeline.py` → check ~4000 rows, feature counts printed
- Run `python nonlinear_model.py` → compare accuracy against 40% baseline
- All output to terminal, no file writes
