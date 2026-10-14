"""
nonlinear_model.py
==================
Nonlinear mathematical models for pilot workload prediction.
Models: Linear Logistic (baseline), Polynomial Logistic (deg 2),
        Kernel Ridge (RBF), QDA, Spline Logistic.
Evaluation: LOSO cross-validation, all output to terminal.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / 'outputs' / 'new_master_features.csv'

# 35 features from explore_features.py (no aircraft state features)
FEATURES = [
    'overall_gaze_entropy_RY', 'psd_max_LY', 'psd_max_RZ',
    'pupil_diam_kurt_R', 'fix_density_stdev', 'eye_pupil_std',
    'pupil_diam_skew_L', 'overall_gaze_entropy_RX',
    'eyes_closed_fraction_R', 'psd_freq_of_max_RY', 'psd_max_LX',
    'delta_extensor', 'emg_acc_rms', 'delta_flexor',
    'head_pitch_std', 'eye_blink_rate', 'delta_hr',
    'fix_density_kurt', 'fix_dur_stdev', 'fix_dur_mean',
    'sac_peak_vel_stdev', 'eye_pupil_velocity', 'sac_main_seq_mean',
    'psd_freq_of_max_LZ', 'fix_dur_kurt', 'delta_scl',
    'ecg_hr_mean', 'psd_freq_of_max_RZ', 'head_vel_rms',
    'eda_scr_count', 'pupil_diam_mean_R', 'emg_cocontraction',
    'ecg_sampen', 'eye_perclos', 'eye_pupil_skew',
]


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models():
    return {
        'Linear Logistic': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=5000, C=1.0,
            )),
        ]),

        'Polynomial Logistic (deg=2)': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale1', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scale2', StandardScaler()),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=5000, C=0.01,
            )),
        ]),

        'Kernel Ridge (RBF)': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', KernelRidge(alpha=1.0, kernel='rbf', gamma=None)),
        ]),

        'QDA': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', QuadraticDiscriminantAnalysis(reg_param=0.1)),
        ]),

        'Spline Logistic': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=5000, C=0.1,
            )),
        ]),
    }


# ============================================================
# LOSO CROSS-VALIDATION
# ============================================================

def loso_evaluate(df, model_name, template):
    subjects = sorted(df['subject'].unique())
    all_true, all_pred = [], []
    per_subj = []

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    for si, test_subj in enumerate(subjects):
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]

        X_train = train[FEATURES].values
        y_train = train['difficulty'].values.astype(int)
        X_test = test[FEATURES].values
        y_test = test['difficulty'].values.astype(int)

        model = clone(template)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # round for regression models (KernelRidge returns continuous)
        y_pred = np.clip(np.round(np.asarray(y_pred, dtype=float)), 1, 4).astype(int)

        subj_acc = accuracy_score(y_test, y_pred)
        per_subj.append((test_subj, subj_acc, len(y_test)))

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        if (si + 1) % 10 == 0 or si == len(subjects) - 1:
            print(f"  fold {si + 1}/{len(subjects)} done")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # metrics
    acc = accuracy_score(all_true, all_pred)
    adj = float(np.mean(np.abs(all_true - all_pred) <= 1))
    bt = (all_true >= 3).astype(int)
    bp = (all_pred >= 3).astype(int)
    bin_acc = accuracy_score(bt, bp)
    f1 = f1_score(all_true, all_pred, average='macro')
    kappa = cohen_kappa_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred, labels=[1, 2, 3, 4])

    # per-subject accuracy
    print(f"\nper-subject accuracy:")
    acc_vals = []
    for subj, sa, n in per_subj:
        bar = '#' * int(sa * 30)
        print(f"  {subj} | {sa:.3f} ({int(sa * n):3d}/{n:3d}) {bar}")
        acc_vals.append(sa)

    print(f"\n  mean +/- std: {np.mean(acc_vals):.4f} +/- {np.std(acc_vals):.4f}")
    print(f"  median:       {np.median(acc_vals):.4f}")
    print(f"  min/max:      {np.min(acc_vals):.4f} / {np.max(acc_vals):.4f}")

    print(f"\n--- RESULTS ---")
    print(f"4-class accuracy:   {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"adjacent accuracy:  {adj:.4f}  ({adj * 100:.1f}%)")
    print(f"binary accuracy:    {bin_acc:.4f}  ({bin_acc * 100:.1f}%)")
    print(f"macro F1:           {f1:.4f}")
    print(f"Cohen's kappa:      {kappa:.4f}")

    print(f"\nconfusion matrix:")
    print(f"  {'pred ->':>8s} {'1':>6s} {'2':>6s} {'3':>6s} {'4':>6s}")
    for i, row in enumerate(cm):
        total = sum(row)
        pct = row[i] / total * 100 if total > 0 else 0
        print(f"  L{i + 1}      {row[0]:6d} {row[1]:6d} {row[2]:6d} {row[3]:6d}  ({pct:.1f}% correct)")

    return {
        'model': model_name, 'acc': acc, 'adj_acc': adj,
        'bin_acc': bin_acc, 'f1': f1, 'kappa': kappa,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(INPUT_CSV)

    print("=" * 60)
    print("NONLINEAR MATHEMATICAL MODELS — LOSO")
    print("=" * 60)
    print(f"samples:  {len(df)}")
    print(f"features: {len(FEATURES)}")
    print(f"subjects: {df['subject'].nunique()}")
    print(f"classes:  {sorted(df['difficulty'].unique())}")
    print(f"class distribution: {dict(df['difficulty'].value_counts().sort_index())}")

    models = build_models()
    results = []

    for name, template in models.items():
        res = loso_evaluate(df, name, template)
        results.append(res)

    # comparison table
    print(f"\n\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'model':35s} {'4-class':>8s} {'adj_acc':>8s} {'binary':>8s} {'F1':>8s} {'kappa':>8s}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:35s} {r['acc']:8.4f} {r['adj_acc']:8.4f} "
              f"{r['bin_acc']:8.4f} {r['f1']:8.4f} {r['kappa']:8.4f}")

    best = max(results, key=lambda x: x['acc'])
    print(f"\nbest: {best['model']} | 4-class={best['acc'] * 100:.1f}% | "
          f"adjacent={best['adj_acc'] * 100:.1f}% | binary={best['bin_acc'] * 100:.1f}%")

    # chance level
    print(f"\nchance level: {1 / 4 * 100:.1f}% (4-class) | 50.0% (binary)")


if __name__ == '__main__':
    main()
