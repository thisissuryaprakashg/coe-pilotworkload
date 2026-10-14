"""
step4_plsda.py
==============
Phase 2, Step 4: Replace ANOVA+VIF screening with PLS-DA.

Instead of univariate ANOVA followed by VIF, this fits a PLS-DA
(Partial Least Squares) transformation inside the LOSO fold.
This distills the ~120 features down to ~15 orthogonal components
that maximally covary with the difficulty label, capturing
cross-feature interactions that ANOVA misses.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import mode as scipy_mode

from improved_model import INPUT_CSV, META_COLS, EXCLUDE_PREFIXES

warnings.filterwarnings('ignore')

def run_plsda_model(n_comp=15):
    print("=" * 60)
    print(f"STEP 4: PLS-DA FEATURE EXTRACTION  [n_comp={n_comp}]")
    print("=" * 60)
    
    df = pd.read_csv(INPUT_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS and not c.startswith(EXCLUDE_PREFIXES)]
    subjects = sorted(df['subject'].unique())
    
    all_true_w, all_pred_w = [], []
    all_true_r, all_pred_r = [], []
    
    # Classifier pipeline
    template = Pipeline([
        ('scaler', StandardScaler()),
        ('spline', SplineTransformer(n_knots=4, degree=3, include_bias=False)),
        ('lr', LogisticRegression(solver='lbfgs', max_iter=5000, C=1.0))
    ])
    
    for si, test_subj in enumerate(subjects):
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]
        
        # 0. Impute missing values (fit on train only)
        # Using median imputation. If a column is entirely missing in train, fallback to 0.
        train_medians = train[feat_cols].median()
        X_train = train[feat_cols].fillna(train_medians).fillna(0).values
        y_train = train['difficulty'].values.astype(int)
        
        X_test = test[feat_cols].fillna(train_medians).fillna(0).values
        y_test = test['difficulty'].values.astype(int)
        
        # 1. Fit PLS on training data to find projection
        # We standardize X before PLS
        scaler_pls = StandardScaler()
        X_train_s = scaler_pls.fit_transform(X_train)
        X_test_s = scaler_pls.transform(X_test)
        
        # One-hot encode y for true PLS-DA
        y_train_dummy = pd.get_dummies(y_train).values
        
        # PLSRegression acts as PLS-DA when y is discrete/ordinal (or one-hot)
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train_s, y_train_dummy)
        
        X_train_pls = pls.transform(X_train_s)
        X_test_pls = pls.transform(X_test_s)
        
        # 2. Fit classifier on PLS components
        model = template
        model.fit(X_train_pls, y_train)
        
        y_pred_w = model.predict(X_test_pls)
        
        all_true_w.extend(y_test)
        all_pred_w.extend(y_pred_w)
        
        # 3. Run-level voting
        test_copy = test[['run_folder', 'difficulty']].copy()
        test_copy['pred'] = y_pred_w
        for run_name, rg in test_copy.groupby('run_folder'):
            run_true = int(rg['difficulty'].iloc[0])
            vote = int(scipy_mode(rg['pred'].values, keepdims=False).mode)
            all_true_r.append(run_true)
            all_pred_r.append(vote)
            
        print(f"  fold {si+1}/{len(subjects)} | subj: {test_subj} | PLS comps: {n_comp}")
        
    all_true_w = np.array(all_true_w)
    all_pred_w = np.array(all_pred_w)
    all_true_r = np.array(all_true_r)
    all_pred_r = np.array(all_pred_r)
    
    acc_w = accuracy_score(all_true_w, all_pred_w)
    adj_w = np.mean(np.abs(all_true_w - all_pred_w) <= 1)
    acc_r = accuracy_score(all_true_r, all_pred_r)
    adj_r = np.mean(np.abs(all_true_r - all_pred_r) <= 1)
    
    print(f"  4-class win: {acc_w*100:.1f}% | run: {acc_r*100:.1f}% | adj-win: {adj_w*100:.1f}% | adj-run: {adj_r*100:.1f}%")
    return acc_w, acc_r

if __name__ == '__main__':
    sweep = [8, 15, 25, 35]
    results = []
    for nc in sweep:
        acc_w, acc_r = run_plsda_model(n_comp=nc)
        results.append((nc, acc_w, acc_r))
    
    print("\n" + "=" * 60)
    print("PLS-DA COMPONENT SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'n_comp':>8} {'win-4cl':>10} {'run-4cl':>10}")
    print("-" * 32)
    for nc, aw, ar in results:
        print(f"{nc:>8} {aw*100:>9.1f}% {ar*100:>9.1f}%")
    print("\nBaseline (ANOVA+VIF screening): ~52.2% run-level")
