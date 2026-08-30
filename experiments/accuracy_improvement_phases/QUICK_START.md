# Quick Start: Running Accuracy Improvement Experiments

## Prerequisites
```bash
cd "C:\Users\Adhithi C Iyer\Desktop\pilot"
venv\Scripts\Activate
cd coe-pilotworkload
```

## Run All Phases (Sequential)

### Phase 1: Subject-Relative Normalization (~3 minutes)
```bash
python experiments\accuracy_improvement_phases\phase1_subject_normalization\run_phase1.py
```
Runs all 4 existing models on raw vs normalized features. Results in `phase1_subject_normalization/results/`.

### Phase 2: Calibrated Stacked Generalization (~8 minutes)
```bash
python experiments\accuracy_improvement_phases\phase2_stacked_generalization\run_phase2.py
```
Meta-learner combines all models. Results in `phase2_stacked_generalization/results/`.

### Phase 3: Ordinal Post-Processing (~5 minutes)
```bash
python experiments\accuracy_improvement_phases\phase3_ordinal_postprocessing\run_phase3.py
```
Ordinal smoothing on ensemble output. Results in `phase3_ordinal_postprocessing/results/`.

### Compare Results
```bash
python experiments\accuracy_improvement_phases\comparison\compare_all_phases.py
```

## Folder Structure
```
experiments/accuracy_improvement_phases/
├── README.md
├── QUICK_START.md
├── phase1_subject_normalization/
│   ├── subject_relative_transform.py   ← Normalization pipeline
│   ├── run_phase1.py                   ← LOSO runner
│   └── results/
├── phase2_stacked_generalization/
│   ├── stacking_meta_learner.py        ← Stacking pipeline
│   ├── run_phase2.py                   ← LOSO runner
│   └── results/
├── phase3_ordinal_postprocessing/
│   ├── ordinal_threshold_tuner.py      ← Ordinal pipeline
│   ├── run_phase3.py                   ← LOSO runner
│   └── results/
└── comparison/
    ├── compare_all_phases.py           ← Side-by-side comparison
    └── results/
```

## What These Techniques Do (on the existing models)
1. **Phase 1**: Normalizes each pilot's physiology to their own baseline
2. **Phase 2**: Lets a meta-learner decide which model to trust
3. **Phase 3**: Smooths predictions to respect workload ordering (1<2<3<4)
