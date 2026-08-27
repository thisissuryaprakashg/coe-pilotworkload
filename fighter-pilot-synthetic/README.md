# Synthetic Fighter-Pilot Mental Workload Dataset

A paper-grounded synthetic dataset generated from the causal pipeline described in
your methodology document, combining the workload-induction logic of:

- **Chen et al. (2025)** — ATA/ATG task, target/engagement difficulty tiers, multimodal measures
- **Mohanavelu et al. (2022)** — visibility × secondary-cognitive-task conditions, flight phases
- **Svensson et al. (1997)** — tactical/information complexity, simultaneous flight-navigation-tactical demand
- *(Gambiraza et al. 2026 startle/stress module: not included — flagged as a future optional extension)*

## Files

| File | Rows | Description |
|---|---|---|
| `synthetic_fighter_pilot_workload_windowed.csv` | 43,292 | **Primary dataset.** Time-windowed (segment-level) data: each mission is broken into flight phases (baseline rest, takeoff, climb, cruise, attack/task, landing rest), with the attack phase further split into 3–6 task windows. This is the "Version 2" windowed time-series-feature dataset from the methodology roadmap. |
| `synthetic_fighter_pilot_workload_mission_level.csv` | 4,553 | Mission-level aggregate (mean/max over all segments in a mission) — the "Version 1" tabular dataset, useful for simpler baseline models. |
| `data_dictionary.csv` | 50 columns | Column-by-column description **and provenance tag** (`SOURCE_GROUNDED` / `METHODOLOGICAL_INFERENCE` / `SYNTHETIC_ASSUMPTION`), per Section 9/13 of the methodology. |
| `validation_report.md` | — | Results of the Section-10 validation plan (construct validity, known-groups validity, physiology/performance direction checks, inter-pilot variability, cross-condition separability, ML utility). |
| `generate_dataset.py` | — | Fully reproducible generator script (seed = 42). |

## Generative pipeline implemented

```
Pilot baseline (150 synthetic pilots, experience group, resting physiology,
                idiosyncratic "reactivity gain" random effect)
        ↓
Mission generator (ATA/ATG type; Chen 3/1, 6/2, 9/3 difficulty tiers;
                    Mohanavelu visibility × cognitive-task; Svensson-style
                    threat density / info update rate / display complexity)
        ↓
Demand components (D_task, D_perceptual, D_cognitive, D_tactical, D_temporal),
        standardized and phase-weighted
        ↓
Latent workload:
    W = 0.28·D_task + 0.18·D_perceptual + 0.20·D_cognitive
        + 0.20·D_tactical + 0.14·D_temporal + pilot_effect + noise
        ↓
Performance (engagement time, hit rate, disengagement time, altitude/
             tracking/navigation error, mission completion)
   +
Physiology (HR, SDNN, RMSSD, LF/HF, EEG theta/alpha/delta relative power,
            fixation rate/duration, blink rate, pupil diameter),
   generated as baseline + pilot-gain × f(W) + noise
        ↓
Workload label (tertile-binned 3-level class; Mohanavelu-style 4-level
                perceptual/cognitive condition; subjective NASA-TLX proxy)
        ↓
Validation (see validation_report.md)
```

This matches the "Recommended dependency structure" and "Recommended mathematical
structure" sections of the methodology document exactly — scenario drives demand,
demand drives a single latent workload variable, and *both* performance and
physiology are generated as separate downstream consequences of that same latent
variable (never generated independently of each other, and never generated after
an arbitrary label).

## Suggested use in your paper

1. **Mathematical model section**: report the explicit generative equations above
   as your ground-truth data-generating process (DGP); this is the "known answer"
   you can use to validate that a fitted mathematical/statistical model (e.g.
   structural equation model, weighted regression) recovers the correct
   demand→workload relationships.
2. **ML model section**: use `synthetic_fighter_pilot_workload_windowed.csv` to
   train/test classifiers or regressors (`latent_workload`, `workload_class_3lvl`,
   or `nasa_tlx_subjective_proxy` as targets; physiology + performance columns as
   features). The validation report already shows a RandomForest baseline
   (R² = 0.881 regression, 80.7% accuracy classification) you can cite as a
   sanity-check baseline before your own model.
3. **Honesty/limitations paragraph**: use `data_dictionary.csv`'s provenance
   column directly — it tells you, per variable, whether to describe it as
   "based on Chen/Mohanavelu/Svensson's manipulation," "a methodological
   extension inspired by [paper]," or "a synthetic generative assumption."
   This lets you write a defensible "what is directly supported vs. proposed"
   subsection (Section 9 of your methodology doc) without extra work.
4. If you later obtain a real fighter-pilot physiological dataset, rerun
   `generate_dataset.py` after recalibrating `W_WEIGHTS` and the physiology
   response-gain constants against the real data (Version 4 in the roadmap:
   external realism validation).

## Reproducibility

Everything is seeded (`SEED = 42`). Re-running `generate_dataset.py` regenerates
byte-identical output. Change `N_PILOTS`, mission-count range, or `W_WEIGHTS` to
scale the dataset further or explore sensitivity of downstream validation results
to the (explicitly not-yet-calibrated) demand weights.
