"""
benchmark_models.py
===================
Rigorous, leak-free Leave-One-Subject-Out (LOSO) cross-validation on MODALSIM94.

Models benchmarked:
1. Linear Logistic Regression (Multinomial L2)
2. Spline Logistic Regression (Cubic basis expansion on screened features)
3. Ordinal Logistic Regression (Proportional Odds Model)
4. XGBoost Classifier (Gradient Boosted Trees)

Strict Leak-Free Pipeline inside each LOSO fold:
- Train fold median imputation (zero information from test subject)
- Train fold StandardScaler
- Train fold ANOVA F-test feature screening
- Train fold Vectorized VIF collinearity pruning
- Full out-of-fold prediction and evaluation:
  - 3-Class Accuracy (0 vs 1 vs 2)
  - Adjacent Accuracy (|y_true - y_pred| <= 1)
  - Binary Accuracy (Low [0] vs High [1,2])
  - Macro F1 & Cohen's Kappa
  - Confusion Matrix
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---------------------------------------------------------
# Spline Basis Expansion (Leak-Free: Knots fit on Train only)
# ---------------------------------------------------------
def fit_transform_splines(X_train, X_test, n_knots=3):
    """
    Fits natural cubic spline basis knots on X_train and evaluates on both X_train and X_test.
    Completely avoids knot degeneracy and division-by-zero NaNs.
    """
    tr_cols = []
    te_cols = []
    for col_idx in range(X_train.shape[1]):
        col_tr = X_train[:, col_idx]
        col_te = X_test[:, col_idx]
        
        unique_vals = np.unique(col_tr)
        if len(unique_vals) < 4:
            tr_cols.append(col_tr[:, None])
            te_cols.append(col_te[:, None])
            continue
            
        knots = np.unique(np.percentile(col_tr, np.linspace(10, 90, n_knots)))
        if len(knots) < 3:
            tr_cols.append(col_tr[:, None])
            te_cols.append(col_te[:, None])
            continue
            
        basis_tr = [col_tr]
        basis_te = [col_te]
        k_last = knots[-1]
        k_penult = knots[-2]
        denom_penult = max(k_last - k_penult, 1e-4)
        
        for k in knots[:-2]:
            denom_k = max(k_last - k, 1e-4)
            d_k_tr = (np.maximum(0, col_tr - k)**3 - np.maximum(0, col_tr - k_last)**3) / denom_k
            d_penult_tr = (np.maximum(0, col_tr - k_penult)**3 - np.maximum(0, col_tr - k_last)**3) / denom_penult
            basis_tr.append(d_k_tr - d_penult_tr)
            
            d_k_te = (np.maximum(0, col_te - k)**3 - np.maximum(0, col_te - k_last)**3) / denom_k
            d_penult_te = (np.maximum(0, col_te - k_penult)**3 - np.maximum(0, col_te - k_last)**3) / denom_penult
            basis_te.append(d_k_te - d_penult_te)
            
        tr_cols.append(np.nan_to_num(np.column_stack(basis_tr), nan=0.0, posinf=0.0, neginf=0.0))
        te_cols.append(np.nan_to_num(np.column_stack(basis_te), nan=0.0, posinf=0.0, neginf=0.0))
        
    return np.column_stack(tr_cols), np.column_stack(te_cols)

# ---------------------------------------------------------
# Fast Vectorized VIF Screen
# ---------------------------------------------------------
def fast_vif_screen(X, max_vif=5.0):
    """Vectorized VIF feature selector via correlation matrix inversion."""
    keep_indices = list(range(X.shape[1]))
    while len(keep_indices) > 1:
        sub_X = X[:, keep_indices]
        corr = np.corrcoef(sub_X, rowvar=False)
        if np.isnan(corr).any():
            break
        try:
            inv_corr = np.linalg.inv(corr + 1e-6 * np.eye(corr.shape[0]))
            vifs = np.diag(inv_corr)
            max_v = np.max(vifs)
            if max_v > max_vif:
                drop_idx = np.argmax(vifs)
                del keep_indices[drop_idx]
            else:
                break
        except np.linalg.LinAlgError:
            break
    return keep_indices

# ---------------------------------------------------------
# Ordinal Logistic Regression (Proportional Odds)
# ---------------------------------------------------------
class OrdinalLogisticRegression:
    """Proportional Odds cumulative logit classifier for ordered classes."""
    def __init__(self, C=1.0):
        self.C = C
        self.binary_models = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.binary_models = []
        # Fit cumulative binary logits: P(Y <= k) for k in classes[:-1]
        for k in self.classes_[:-1]:
            y_binary = (y > k).astype(int)  # 1 if > k, 0 if <= k
            clf = LogisticRegression(C=self.C, max_iter=500, random_state=42)
            clf.fit(X, y_binary)
            self.binary_models.append(clf)
        return self

    def predict_proba(self, X):
        cum_probs = []
        for clf in self.binary_models:
            # clf.predict_proba returns [P(<=k), P(>k)]
            p_greater = clf.predict_proba(X)[:, 1]
            cum_probs.append(p_greater)
        
        # cum_probs[k] = P(Y > k)
        # P(Y=0) = 1 - P(Y > 0)
        # P(Y=1) = P(Y > 0) - P(Y > 1)
        # P(Y=2) = P(Y > 1)
        probs = np.zeros((X.shape[0], len(self.classes_)))
        probs[:, 0] = 1.0 - cum_probs[0]
        for i in range(1, len(self.classes_) - 1):
            probs[:, i] = np.maximum(0, cum_probs[i-1] - cum_probs[i])
        probs[:, -1] = cum_probs[-1]
        
        # Normalize rows to sum to 1
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

# ---------------------------------------------------------
# LOSO Cross-Validation Benchmark
# ---------------------------------------------------------
def run_loso_benchmark(csv_path, use_deltas=True, top_k_features=8):
    df = pd.read_csv(csv_path)
    print(f"\nLoaded dataset from: {csv_path}")
    print(f"Total rows: {len(df)} across {df['subject'].nunique()} subjects.")
    
    # Define candidate feature pools
    meta_cols = {'subject', 'scenario', 'target', 'MentalTLX', 'PhysicalTLX', 'TemporalTLX',
                 'PerformanceTLX', 'EffortTLX', 'FrustrationTLX', 'SART', 'KSS', 'SPS'}
    
    if use_deltas:
        feature_cols = [c for c in df.columns if c.startswith('delta_')]
    else:
        feature_cols = [c for c in df.columns if (c not in meta_cols and not c.startswith('delta_'))]
        
    print(f"Candidate feature pool ({'Delta' if use_deltas else 'Raw'}): {len(feature_cols)} features")

    subjects = df['subject'].unique()
    n_subjects = len(subjects)

    model_names = ['Linear Logistic', 'Spline Logistic', 'Ordinal Logistic']
    if HAS_XGB:
        model_names.append('XGBoost')

    predictions = {name: [] for name in model_names}
    ground_truth = []

    print(f"\nRunning {n_subjects}-Fold Leave-One-Subject-Out (LOSO) Cross-Validation...")

    for fold_i, test_subj in enumerate(subjects):
        train_df = df[df['subject'] != test_subj]
        test_df = df[df['subject'] == test_subj]

        X_train_raw = train_df[feature_cols].copy()
        X_test_raw = test_df[feature_cols].copy()
        y_train = train_df['target'].values
        y_test = test_df['target'].values
        ground_truth.extend(y_test)

        # 1. Train-only Median Imputation
        train_medians = X_train_raw.median().fillna(0.0)
        X_train_imp = np.nan_to_num(X_train_raw.fillna(train_medians).values, nan=0.0)
        X_test_imp = np.nan_to_num(X_test_raw.fillna(train_medians).values, nan=0.0)

        # 2. Train-only Standard Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # 3. Train-only ANOVA Screening
        f_scores, _ = f_classif(X_train_scaled, y_train)
        f_scores = np.nan_to_num(f_scores, nan=-1.0)
        ranked_indices = np.argsort(f_scores)[::-1]
        prelim_selected = ranked_indices[:min(top_k_features * 2, len(ranked_indices))]

        # 4. Train-only VIF Pruning
        kept_sub_indices = fast_vif_screen(X_train_scaled[:, prelim_selected], max_vif=5.0)
        final_selected = prelim_selected[kept_sub_indices][:top_k_features]

        X_tr_final = X_train_scaled[:, final_selected]
        X_te_final = X_test_scaled[:, final_selected]

        # Spline expansions (Train knots applied to test)
        X_tr_spline, X_te_spline = fit_transform_splines(X_tr_final, X_te_final, n_knots=3)

        # Model 1: Linear Logistic
        m_linear = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        m_linear.fit(X_tr_final, y_train)
        predictions['Linear Logistic'].extend(m_linear.predict(X_te_final))

        # Model 2: Spline Logistic
        m_spline = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        m_spline.fit(X_tr_spline, y_train)
        predictions['Spline Logistic'].extend(m_spline.predict(X_te_spline))

        # Model 3: Ordinal Logistic
        m_ordinal = OrdinalLogisticRegression(C=0.5)
        m_ordinal.fit(X_tr_final, y_train)
        predictions['Ordinal Logistic'].extend(m_ordinal.predict(X_te_final))

        # Model 4: XGBoost
        if HAS_XGB:
            m_xgb = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
                random_state=42
            )
            m_xgb.fit(X_tr_final, y_train)
            predictions['XGBoost'].extend(m_xgb.predict(X_te_final))

    ground_truth = np.array(ground_truth)
    
    # ---------------------------------------------------------
    # Metrics Compilation
    # ---------------------------------------------------------
    results = []
    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS: MODALSIM94 (Features: {'Subject-Deltas' if use_deltas else 'Raw Absolute'})")
    print("=" * 80)
    
    for name in model_names:
        preds = np.array(predictions[name])
        acc = accuracy_score(ground_truth, preds) * 100.0
        adj_acc = np.mean(np.abs(ground_truth - preds) <= 1) * 100.0
        
        # Binary: Low (0) vs High (1, 2)
        gt_bin = (ground_truth > 0).astype(int)
        pred_bin = (preds > 0).astype(int)
        bin_acc = accuracy_score(gt_bin, pred_bin) * 100.0
        
        f1_macro = f1_score(ground_truth, preds, average='macro') * 100.0
        kappa = cohen_kappa_score(ground_truth, preds)
        
        cm = confusion_matrix(ground_truth, preds)
        
        results.append({
            'Model': name,
            '3-Class Acc (%)': acc,
            'Adjacent Acc (%)': adj_acc,
            'Binary Acc (%)': bin_acc,
            'Macro F1 (%)': f1_macro,
            'Cohen Kappa': kappa,
            'Confusion Matrix': cm
        })

    summary_df = pd.DataFrame([
        {
            'Model': r['Model'],
            '3-Class Acc (%)': f"{r['3-Class Acc (%)']:.1f}%",
            'Adjacent Acc (%)': f"{r['Adjacent Acc (%)']:.1f}%",
            'Binary Acc (%)': f"{r['Binary Acc (%)']:.1f}%",
            'Macro F1 (%)': f"{r['Macro F1 (%)']:.1f}%",
            'Cohen Kappa': f"{r['Cohen Kappa']:.3f}"
        }
        for r in results
    ])
    
    print(summary_df.to_string(index=False))
    print("=" * 80)

    for r in results:
        print(f"\n--- Confusion Matrix: {r['Model']} ---")
        print("        Pred 0   Pred 1   Pred 2")
        for i, row in enumerate(r['Confusion Matrix']):
            print(f"True {i}:   {row[0]:6d}   {row[1]:6d}   {row[2]:6d}")

    return results

if __name__ == '__main__':
    root_path = Path(__file__).resolve().parents[1]
    norm_csv = root_path / 'outputs' / 'modalsim_normalized_features.csv'
    
    print(">>> 1. BENCHMARK WITH SUBJECT-RELATIVE DELTA FEATURES (LEAK-FREE) <<<")
    run_loso_benchmark(norm_csv, use_deltas=True, top_k_features=8)
    
    print("\n>>> 2. BENCHMARK WITH RAW ABSOLUTE SENSOR FEATURES <<<")
    run_loso_benchmark(norm_csv, use_deltas=False, top_k_features=8)
