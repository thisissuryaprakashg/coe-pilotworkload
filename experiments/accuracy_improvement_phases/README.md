# Accuracy Improvement Phases

## Overview

Three niche, novel techniques applied **on top of the existing models** (GBDT, RF, SVM, Logistic Regression) to improve workload prediction accuracy.

**Baseline:** 40.66% (GBDT on Oculomotor features, LOSO validation)

## Phase 1: Subject-Relative Physiological Normalization
**Technique:** Within-subject z-score centering + cross-modal interaction features

Each pilot's features are z-scored against their own mean/std, removing between-subject baseline differences. The same GBDT/RF/SVM/LR models are trained on the normalized features.

```bash
python experiments\accuracy_improvement_phases\phase1_subject_normalization\run_phase1.py
```

## Phase 2: Calibrated Stacked Generalization
**Technique:** Meta-learning over all 4 models × 2 feature subsets

All existing models' probability outputs (on both raw and enhanced features) are combined via a Logistic Regression meta-learner that learns which model to trust for each prediction.

```bash
python experiments\accuracy_improvement_phases\phase2_stacked_generalization\run_phase2.py
```

## Phase 3: Ordinal-Aware Post-Processing
**Technique:** Gaussian ordinal smoothing + cost-sensitive threshold tuning

Adjusts model probability outputs to respect the ordinal structure (Level 1 < 2 < 3 < 4) and optimizes decision thresholds to maximize quadratic weighted Cohen's Kappa.

```bash
python experiments\accuracy_improvement_phases\phase3_ordinal_postprocessing\run_phase3.py
```

## Compare All Phases
```bash
python experiments\accuracy_improvement_phases\comparison\compare_all_phases.py
```

## Key Principles
- ✅ **Same models** — GBDT, RF, SVM, LR configurations unchanged
- ✅ **Non-destructive** — All original `src/` files untouched
- ✅ **LOSO validation** — No subject leakage
- ✅ **Niche & Novel** — Research-grade techniques, not generic tuning
