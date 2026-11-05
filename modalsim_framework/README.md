# MODALSIM94 Multi-Modal Cognitive Workload Framework

A modular, leak-free pipeline and model evaluation benchmark for the **MODALSIM94** dataset (94 participants, 3 cognitive workload tiers: `NBTask`, `1BTask`, `2BTask`).

---

## 1. Directory Structure

```text
modalsim_framework/
│
├── data_pipeline/
│   ├── __init__.py
│   ├── extract_pulse.py          # Cardiovascular & HRV metrics (10 features)
│   ├── extract_respiratory.py    # Respiration rate & amplitude (6 features)
│   ├── extract_eye.py            # Dikablis pupillometry & gaze dispersion (6 features)
│   ├── extract_performance.py    # Tunnel flight deviation errors (5 features)
│   └── assemble_master.py        # Leak-free subject-relative delta assembler
│
├── modeling/
│   ├── __init__.py
│   └── benchmark_models.py       # Strict 94-fold LOSO cross-validation benchmark
│
├── outputs/
│   ├── modalsim_master_features.csv      # Raw multi-modal features (282 rows x 39 cols)
│   └── modalsim_normalized_features.csv  # 27 Subject-relative delta features added (282 x 66)
│
└── README.md
```

---

## 2. Leakage Prevention Protocol

1. **Subject-Relative Deltas**: Baseline values are extracted solely from each participant's own `NBTask` (0-Back) session.
2. **Leave-One-Subject-Out (LOSO)**: 94 folds, where 100% of data from the test participant is held out.
3. **Train-Fold Only Imputation**: Missing values are imputed using medians computed strictly from the 93 training participants.
4. **Train-Fold Feature Scaling & Screening**: Standard scaling, ANOVA F-ranking, and Vectorized VIF collinearity pruning are computed strictly on training data.
5. **Out-of-Fold Spline Knots**: Natural cubic spline knots are fit on training data and evaluated out-of-fold.

---

## 3. Benchmark Results (94-Fold LOSO)

### Subject-Relative Delta Features (Leak-Free)
| Model | 3-Class Acc (%) | Adjacent Acc (%) | Binary Acc (Low vs High) | Macro F1 (%) | Cohen's Kappa |
|---|:---:|:---:|:---:|:---:|:---:|
| **Linear Logistic** | 68.1% | 96.1% | 87.6% | 66.0% | 0.521 |
| **Ordinal Logistic** | 69.9% | 96.1% | 87.6% | 68.2% | 0.548 |
| **Spline Logistic** | 70.2% | 98.2% | 91.1% | 68.0% | 0.553 |
| **XGBoost Classifier** | **74.8%** | **99.6%** | **99.3%** | **74.8%** | **0.622** |

*(Chance level = 33.3%)*

### Raw Absolute Sensor Features (Without Subject Normalization)
| Model | 3-Class Acc (%) | Adjacent Acc (%) | Binary Acc (%) | Cohen's Kappa |
|---|:---:|:---:|:---:|:---:|
| **Linear Logistic** | 45.4% | 85.5% | 66.0% | 0.181 |
| **Spline Logistic** | 44.7% | 84.0% | 64.2% | 0.170 |
| **XGBoost Classifier** | 40.4% | 84.4% | 61.0% | 0.106 |

> **Key Finding**: Subject-relative baseline normalization produces a massive **+34.4% accuracy surge** (from 40.4% to 74.8%), demonstrating that inter-individual physiological variability is the primary obstacle to cognitive load generalization.

---

## 4. Modality Ablation (3-Class LOSO)

| Modality Stream | Number of Features | 3-Class Accuracy | Adjacent Accuracy |
|---|:---:|:---:|:---:|
| **Flight Performance** | 5 | 52.1% | 89.0% |
| **Respiratory (BPM, RA)** | 6 | 56.7% | 89.4% |
| **Dikablis Eye Tracking** | 6 | 62.1% | 90.4% |
| **eSENSE Pulse (HRV)** | 10 | 64.5% | 96.8% |
| **Multi-Modal Combined (Linear)** | 27 | 68.1% | 96.1% |
| **Multi-Modal Combined (Spline)** | 27 | 70.2% | 98.2% |
| **Multi-Modal Combined (XGBoost)** | 27 | **74.8%** | **99.6%** |

---

## 5. How to Run

### Environment Setup
Navigate to the repository root directory and set the Python path:

```bash
# Linux / macOS
cd coe-pilotworkload
export PYTHONPATH="."

# Windows (PowerShell)
cd coe-pilotworkload
$env:PYTHONPATH="."

# Windows (Command Prompt)
cd coe-pilotworkload
set PYTHONPATH=.
```

### Step 1: Extract and Assemble Multi-Modal Features
```bash
python modalsim_framework/data_pipeline/assemble_master.py
```

### Step 2: Run Leak-Free LOSO Benchmark Models
```bash
python modalsim_framework/modeling/benchmark_models.py
```

