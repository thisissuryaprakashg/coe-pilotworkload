# Validation Report — Synthetic Fighter-Pilot Mental Workload Dataset

Generator version: `v1.0` · Seed: `42` · Rows (windowed): **43,292** · Rows (mission-level): **4,553** · Pilots: **150**

This report follows Section 10 ("Validation plan") of the methodology document.

## 1. Construct validity — workload rises with intended scenario difficulty

Mean `latent_workload` by Chen-style difficulty tier:

| Tier | Mean latent workload |
|---|---|
| low | -0.231 |
| medium | 0.010 |
| high | 0.272 |

Monotonic increase confirmed: **True**.

## 2. Known-groups validity — flight phase separation

Mean `latent_workload` by flight phase (ascending):

| Phase | Mean latent workload |
|---|---|
| baseline_rest | -0.899 |
| cruise | -0.692 |
| climb | -0.621 |
| landing_rest | -0.549 |
| takeoff | -0.487 |
| attack_task | **0.747** |

The attack/task-execution phase — where Chen's and Mohanavelu's manipulated task load is enacted — is clearly separated from rest/transit phases, as expected.

## 3. Physiology validity — directional relationships with latent workload

| Feature | Correlation with latent workload | Expected direction | Match |
|---|---|---|---|
| HR (bpm) | +0.499 | increase | ✅ |
| SDNN (ms) | -0.346 | decrease | ✅ |
| RMSSD (ms) | -0.307 | decrease | ✅ |
| LF/HF ratio | +0.685 | increase | ✅ |
| EEG theta relative power | +0.539 | increase | ✅ |
| EEG alpha relative power | -0.423 | decrease | ✅ |
| Fixation duration (ms) | -0.331 | decrease | ✅ |
| Blink rate (bpm) | -0.453 | decrease | ✅ |
| Pupil diameter (mm) | +0.344 | increase | ✅ |

All nine physiological indicators move in the direction reported/expected in the source workload literature (Chen et al. 2025; Mohanavelu et al. 2022).

## 4. Performance validity

| Metric | Correlation with latent workload | Expected direction | Match |
|---|---|---|---|
| Engagement time (s) | +0.815 | increase | ✅ |
| Hit rate | -0.683 | decrease | ✅ |
| Altitude error (ft) | +0.832 | increase | ✅ |
| Tracking error (deg) | +0.854 | increase | ✅ |
| Navigation error (nm) | +0.854 | increase | ✅ |
| Mission completion | -0.612 | decrease | ✅ |

## 5. Inter-pilot variability

Standard deviation of mean HR across pilots, restricted to high-difficulty missions: **7.39 bpm** (non-zero, confirming identical missions do not produce identical physiology across pilots — pilot random effects and reactivity gains are preserved).

## 6. Cross-condition distinguishability (Mohanavelu 4-level design)

| Condition | Mean latent workload |
|---|---|
| NV_NoTask (normal vis., no cognitive task) | -0.220 |
| LV_NoTask (low vis., no cognitive task) | +0.007 |
| NV_CogTask (normal vis., cognitive task) | +0.097 |
| LV_CogTask (low vis., cognitive task) | **+0.305** |

Ordering matches the expected additive/compounding pattern: combined perceptual + cognitive load produces the highest workload, and each manipulation independently raises workload above the no-load baseline.

## 7. ML utility

- **Regression** (RandomForestRegressor, 200 trees, 80/20 split, physiology + performance features → `latent_workload`): **R² = 0.881**, MAE = 0.228.
- **Classification** (RandomForestClassifier, same features → `workload_class_3lvl`): **accuracy = 0.807**, macro-F1 = 0.806.

Scores indicate the generated physiology/performance features carry strong, learnable signal about the underlying workload state — sufficient for downstream ML modeling — without being deterministic (noise and pilot variability keep it below ceiling performance), which mirrors the imperfect predictability of real-world psychophysiological workload data.

## Caveats for the manuscript

- `latent_workload`, all `D_*_z` demand components, and `nasa_tlx_subjective_proxy` are **generative constructs**, not measured quantities — report them as the synthetic ground truth used to condition the generator, not as recovered real physiological facts.
- Regression/correlation weights in the generator (Section 6 of the methodology) are **author-specified, not fitted to real fighter-pilot data**. If a real dataset becomes available, recalibrate weights and rerun this validation (Version 4 in the roadmap).
- Absolute physiological values (e.g., baseline HR ranges) are plausible adult resting/task ranges but are **not drawn from the uploaded papers' raw data** — only the *direction and relative pattern* of workload effects is source-grounded (see `data_dictionary.csv` provenance column).
