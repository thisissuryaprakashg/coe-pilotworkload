# CogPilot Pilot Mental Workload Modeling — Project Specification

## 1. Project Goal

Predict pilot mental workload (operationalized as the experimentally-manipulated
**task difficulty level, 1–4**) from physiological signals, using:
1. A **mathematical model** (interpretable equation-based approach)
2. A **machine learning model** (classifier)
3. A **head-to-head comparison** of both, on identical inputs and identical
   train/test splits

This models the *cognitive workload* mechanism only. It does not model G-force
or thermal stress (no public fighter-specific dataset exists for those). This
scope limitation should be stated explicitly in the paper's intro/discussion,
not treated as a hidden weakness.

## 2. Dataset

**CogPilot** — PhysioNet, "Multimodal Physiological Monitoring During Virtual
Reality Piloting Tasks" (v1.0.0)
Link: https://physionet.org/content/virtual-reality-piloting/1.0.0/

- 35 subjects (sub-cp003 to sub-cp043), de-identified
- Task: simulated Instrument Landing System (ILS) approach, X-Plane 11, T-6
  Texan II aircraft, VR (HTC Vive Pro Eye) + HOTAS controls
- 12 ILS runs per subject (3 runs × 4 difficulty levels) + 2 rest runs
  (pre- and post-experiment, 5 min each)
- Synchronized via Lab Streaming Layer (LSL), sub-millisecond cross-stream sync

### 2.1 Directory structure
```
dataPackage/
  task-ils/
    PerfMetrics.csv                         <- summary performance/target labels
    sub-cpXXX/ses-YYYYMMDD/level-0NB_run-0NN/
      *_dat.csv                             <- raw time-series sensor samples
      *_hea.csv                             <- stream metadata (rates, sample count)
      *_perfmetric_*.csv                    <- per-run flight error time series
  task-rest/
    sub-cpXXX/ses-YYYYMMDD/level-000_run-00N/
      *_dat.csv / *_hea.csv                 <- rest baseline recordings
starterCode/
  data_feats/devSubjsFeatMat.csv            <- PRE-COMPUTED eye-tracking features
  assembleFeatureMatrix.py, extractSaccFix.py, trainModel.py  <- reusable starter code
```

### 2.2 File naming convention
```
sub-<SubjectID>_ses-<Date>_task-<Task>_stream-<StreamName>_feat-<FeatureType>_level-<Level>_run-<RunID>_<dat|hea>.csv
```
- `level-000` = rest baseline; `level-01B`…`level-04B` = difficulty 1–4
- `feat-chunk` = raw time series; `feat-perfmetric` = flight error time series

### 2.3 Difficulty protocol (ground truth for the target variable)
| Level | Wind | Clouds/Visibility | Turbulence |
|---|---|---|---|
| 1 | None | None / unlimited | None |
| 2 | 140°@10kt | Overcast 2500ft, 5mi | None |
| 3 | Layered, ~10kt | Overcast 1000ft, 3mi | Light, above 1000ft |
| 4 | Layered, 15-20kt gusts | Overcast 400ft, 1mi | Light, throughout |

### 2.4 Relevant sensor streams
| Stream | Hardware | Nominal Fs | Key columns |
|---|---|---|---|
| `lslshimmerecg` | Shimmer3 ECG (chest) | 512 Hz | `ecg_projection_ll_ra_mV`, `ecg_projection_la_ra_mV`, `ecg_projection_vx_rl_mV` |
| `lslshimmereda` | Shimmer3 GSR+ (hand) | 1024 Hz nominal / ~128 Hz effective | `eda_hand_l_kOhms`, `ppg_finger_mV` |
| `lslshimmeremg` | Shimmer3 EMG (forearm) | 512 Hz | `emg_wrist_flexor_mV`, `emg_wrist_extensor_mV`, forearm accelerometry |
| `lslhtcviveeye` | HTC Vive Pro Eye | 250 Hz nominal / 90-120 Hz effective | `pupil_diameter_l/r_mm`, `eye_openness_l/r`, gaze vectors, `validity_l/r` |
| `lslxp11_feat-perfmetric` | X-Plane 11 | 4-20 Hz | `glideslope_error_deg`, `localizer_error_deg`, `airspeed_error_kts`, `total_error` |

Timestamp conversion (MATLAB datenum → UNIX ms):
```
t_millis = (time_dn - 719529) * 86400 * 1000
```

## 3. Data Quality Audit (already completed — use these findings directly)

Source: full 487-row `DataQualitySummary` (419 ILS runs + 68 rest runs, 35 subjects).

### 3.1 Stream availability (whole-subject level)
| Stream | Subjects with NO data at all |
|---|---|
| Respiration (`respitrace`) | 20 of 35 subjects — **drop this stream from the feature set** (protocol/hardware change mid-collection, not random missingness) |
| Torso accelerometry | 2 of 35 subjects (cp003, cp009) — optional, not essential |
| EMG, EDA, ECG, Eye-tracking, X-Plane | 0 subjects — all subjects have at least some data |

### 3.2 Run-level quality failures (quality/validity % < 90) — EXCLUDE THESE SPECIFIC RUNS ONLY, not the whole subject
**ECG:**
- sub-cp009: level-01B_run-001, level-02B_run-003, level-03B_run-002, level-04B_run-004 (all 0%)
- sub-cp026: level-01B_run-001 (0%), level-02B_run-008 (68.84%), level-03B_run-002 (89.14%), level-03B_run-005 (77.99%), level-04B_run-009 (29.13%)
- sub-cp027: level-02B_run-010 (0%)

**EDA:**
- sub-cp028: level-01B_run-012 (0%)

**Eye-tracking:**
- sub-cp003: ALL 12 runs = 0% (matches this subject's absence from `devSubjsFeatMat.csv` — starter code already excluded them)
- sub-cp027: level-01B_run-001, level-03B_run-002 (both 0%)

### 3.3 Net effect of exclusions
- Total ILS runs: 419
- Runs excluded for bad ECG/EDA: 11
- **Runs retained: 408 (97.4%)** — this is a very minor loss; apply the exclusion at the row level, not by dropping subjects

### 3.4 Cleaning rules to implement
1. Drop `respiration` as a feature entirely.
2. Keep `torso-accelerometry` optional/secondary if used at all.
3. Exclude the 11 specific bad ECG/EDA runs listed in 3.2 (row-level filter).
4. Exclude sub-cp003 from eye-tracking-derived features only (its ECG/EDA data is fine and should stay in the main analysis).
5. Exclude sub-cp027's 2 bad eye-tracking runs from eye-tracking analysis only.

## 4. Feature Extraction

### 4.1 Already done — reuse, do not rebuild
`starterCode/data_feats/devSubjsFeatMat.csv` contains pre-computed eye-tracking
features per subject/run (34 subjects — sub-cp003 excluded, consistent with 3.2):
`overall_gaze_entropy_*`, `psd_max_*`, `psd_freq_of_max_*` (X/Y/Z, L/R eyes),
`eyes_closed_fraction_L/R`, `pupil_diam_mean/stdev/skew/kurt_L/R`,
`fix_dur_mean/stdev/skew/kurt`, `fix_density_mean/stdev/skew/kurt`,
`sac_main_seq_mean/stdev`, `sac_peak_vel_mean/stdev`.
Load this directly; join on `(Subject, Session, Run)`.

### 4.2 Must be built — ECG features
From `*_stream-lslshimmerecg_feat-chunk_*_dat.csv`, per run (and per rest run):
- Use `neurokit2.ecg_process(ecg_signal, sampling_rate=512)` for R-peak detection
- Extract: mean HR, SDNN, RMSSD (time-domain HRV), LF/HF ratio (frequency-domain HRV)
- Use one lead consistently (e.g. `ecg_projection_ll_ra_mV`, Lead II) unless a
  multi-lead comparison is explicitly wanted

### 4.3 Must be built — EDA/PPG features
From `*_stream-lslshimmereda_feat-chunk_*_dat.csv`, per run:
- Use `neurokit2.eda_process(eda_signal, sampling_rate=128)` (note: effective
  rate is ~128 Hz per the quality summary, not the 1024 Hz nominal rate — use
  the effective rate for processing)
- Extract: tonic EDA level (mean), phasic peak count, phasic peak amplitude (mean)
- Optionally extract PPG-derived HR from `ppg_finger_mV` as a cross-check against ECG-derived HR

### 4.4 Optional — EMG features
From `*_stream-lslshimmeremg_feat-chunk_*_dat.csv`, if used:
- Mean/RMS amplitude of `emg_wrist_flexor_mV`, `emg_wrist_extensor_mV`

### 4.5 Windowing decision
CogPilot already segments recordings by run = one difficulty level (duration
~5-10 min per run based on the quality summary's `*_dur_s` columns). Compute
one feature vector per (subject, run) as a first pass — i.e., aggregate over
the whole run — rather than sub-windowing into 30-60s epochs, unless the
per-run sample count proves insufficient for the modeling step. This keeps
the design consistent with the eye-tracking features in `devSubjsFeatMat.csv`,
which are also one row per run.

## 5. Target Variables and Baseline

### 5.1 Primary target
`difficulty` (1-4), from `PerfMetrics.csv` or parsed from the `level-0NB` folder name.

### 5.2 Secondary/validation target
`cumulative_total_error` from `PerfMetrics.csv` — CogPilot's unique advantage
(objective flight-performance ground truth). Use to sanity-check that predicted
workload correlates with actual performance degradation.

### 5.3 Personal baseline (for delta features)
Extract the same ECG/EDA features (Section 4.2/4.3) from each subject's
`task-rest/.../level-000_run-001` (pre-experiment rest, 5 min).
```
delta_HR  = task_run_HR  − subject's rest_run_001_HR
delta_EDA_tonic = task_run_EDA_tonic − subject's rest_run_001_EDA_tonic
```
(same pattern for HRV features). This is the leakage-safe personalization
step: it only uses information available without knowing the current
difficulty level (who the subject is + their own resting baseline), so it's
valid to use as a model input for a model trying to predict difficulty.

Both the mathematical model and the ML model use delta features as their
primary input representation. Also build a raw-feature version of the same
table as a comparison/ablation (does personalization actually help?).

## 6. Master Table Schema (after all merges)

```
subject | run | difficulty | HR | HRV_SDNN | HRV_RMSSD | HRV_LF_HF |
EDA_tonic | EDA_phasic_peak_count | EDA_phasic_peak_amp |
delta_HR | delta_HRV_SDNN | delta_HRV_RMSSD | delta_EDA_tonic |
[eye-tracking columns from devSubjsFeatMat.csv, subject != cp003] |
cumulative_total_error | quality_flag
```
One row per (subject, run). Apply the Section 3.4 exclusion rules to this
table before any modeling step.

## 7. Feature Selection

Run ANOVA (or Kruskal-Wallis if a normality check fails) on every feature
column against `difficulty` (4 groups). Retain only features with a
statistically significant difference across difficulty levels. This becomes
the final input feature set for both models in Section 8, ensuring both use
the same, literature-justified inputs.

## 8. Modeling (per instructor's requirement: ML model + math model + comparison)

**Shared setup for a fair comparison:**
- Same feature set (from Section 7, using delta features as primary)
- Same target (`difficulty`, 4-class)
- Same train/test splits: **LOSO (Leave-One-Subject-Out)** cross-validation —
  train on 34 subjects, test on the 1 held out, repeat for all 35, so results
  reflect generalization to a new, unseen pilot

### 8.1 Mathematical model
Ordinal or multinomial logistic regression (literature precedent: Wei et al.'s
discriminant-modelling approach):
```
P(difficulty level) = f(β0 + β1·delta_HR + β2·delta_HRV_SDNN + β3·delta_EDA_tonic + ...)
```
Report fitted coefficients (β values) — this is the interpretable,
standalone deliverable: which physiological deviations most strongly predict
higher workload, and by how much.

### 8.2 ML model
SVM, Random Forest, and/or Gradient Boosted Decision Trees (GBDT) — same
delta features, same LOSO splits, same target. These are the classifiers
used in the base paper this project extends, so using them keeps a direct,
citable point of comparison.

### 8.3 Comparison
Evaluate both on identical LOSO test folds. Report:
- Accuracy, F1-score (macro, given 4 balanced-ish classes), confusion matrix
- Where each model over/under-performs (e.g., does the mathematical model do
  as well at the extremes — level 1 vs level 4 — but worse in the middle,
  where the relationship may be less linear?)
- Interpretability trade-off discussion: the math model gives a clear,
  reportable equation; the ML model likely trades some interpretability for
  accuracy — this discussion is itself a real part of the paper's contribution,
  not filler.
- Raw-feature vs. delta-feature ablation (Section 5.3): does personalizing via
  subject's own rest baseline actually improve either model's accuracy? Report
  this explicitly rather than assuming it.

## 9. Explicit Scope Statement (include in intro and discussion, not buried)

This study models the *cognitive workload* axis of fighter-pilot stress using
a publicly available flight-training simulator dataset, since no public
fighter-pilot-specific physiological dataset exists. G-force load and thermal
load — the other two major stressor axes identified in the fighter-pilot
physiology literature — are explicitly out of scope; no accessible public
dataset permits modeling them. This limitation should be cited against the
G-tolerance/thermal-stress literature already reviewed for this project,
rather than left implicit.

## 10. Suggested Build Order

1. Modality/quality audit (Section 3) — **already done**, use as-is
2. ECG feature extraction script (Section 4.2)
3. EDA feature extraction script (Section 4.3)
4. Rest-baseline extraction + delta computation (Section 5.3)
5. Merge with `PerfMetrics.csv` (Section 5.1/5.2) and `devSubjsFeatMat.csv` (Section 4.1)
6. Apply exclusion rules (Section 3.4) to the merged master table
7. ANOVA feature screening (Section 7)
8. Fit mathematical model (Section 8.1)
9. Fit ML classifiers (Section 8.2)
10. Run LOSO comparison (Section 8.3)
11. Write up scope statement (Section 9) early, not as an afterthought
