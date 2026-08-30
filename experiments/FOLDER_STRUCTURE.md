# Folder Structure Overview

## Original Files (UNTOUCHED)
```
coe-pilotworkload/
├── data/processed/cleaned_multimodal_dataset.csv   ← Data source (408 samples)
├── src/modeling/
│   ├── train_and_compare_models.py                 ← Baseline GBDT/RF/SVM/LR (40.66%)
│   ├── math_model_ordinal_regression.py            ← Math model
│   └── [other models]
├── reports/
│   ├── enhanced_model_benchmark_results.csv        ← Baseline metrics
│   └── [other reports]
└── ACCURACY_IMPROVEMENT_PLAN.md                    ← Original plan document
```

## New Experiments Folder (PARALLEL)
```
experiments/accuracy_improvement_phases/
├── README.md
├── QUICK_START.md
├── FOLDER_STRUCTURE.md
│
├── phase1_subject_normalization/
│   ├── subject_relative_transform.py    ← Z-score normalization + interactions
│   ├── run_phase1.py                    ← Runs all 4 models on normalized features
│   └── results/
│       ├── phase1_per_fold_results.csv
│       ├── phase1_model_summary.csv
│       ├── phase1_per_subject_accuracy.csv
│       └── phase1_summary.txt
│
├── phase2_stacked_generalization/
│   ├── stacking_meta_learner.py         ← Meta-learner over all models
│   ├── run_phase2.py                    ← LOSO stacking evaluation
│   └── results/
│       ├── phase2_per_fold_results.csv
│       └── phase2_summary.txt
│
├── phase3_ordinal_postprocessing/
│   ├── ordinal_threshold_tuner.py       ← Ordinal smoothing + thresholds
│   ├── run_phase3.py                    ← LOSO ordinal evaluation
│   └── results/
│       ├── phase3_per_fold_results.csv
│       └── phase3_summary.txt
│
└── comparison/
    ├── compare_all_phases.py            ← Baseline vs all phases
    └── results/
        ├── comparison_all_phases.txt
        └── comparison_all_phases.csv
```

## Data Flow
```
cleaned_multimodal_dataset.csv
        │
        ├─→ src/modeling/train_and_compare_models.py
        │   Results: 40.66% baseline
        │
        └─→ experiments/accuracy_improvement_phases/
            │
            ├─→ Phase 1: Subject normalization → same models on better features
            ├─→ Phase 2: Stacked generalization → meta-learner over all models
            └─→ Phase 3: Ordinal post-processing → ordinal-aware correction
                    ↓
            comparison/compare_all_phases.py → side-by-side report
```

## Key Principle
**NO ORIGINAL FILES ARE MODIFIED.** Same models, enhanced techniques.
