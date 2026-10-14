"""
step5_gmm_phenotyping.py
========================
Phase 3, Step 5: Physiological phenotyping via GMM clustering.

Clusters subjects into distinct phenotypes (e.g., high vs low HRV/arousal)
based purely on their resting baseline data. Trains a separate classifier
for each cluster. At test time, assigns the test subject to a cluster using
only their rest data, and predicts using that cluster's model.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import mode as scipy_mode

from src.modeling.improved_model import INPUT_CSV, META_COLS, EXCLUDE_PREFIXES, screen_features_silent
from src.data_pipeline.new_data_pipeline import compute_rest_baselines

warnings.filterwarnings('ignore')

def run_gmm_model():
    print("=" * 60)
    print("STEP 5: GMM PHENOTYPING (CLUSTER-SPECIFIC MODELS)")
    print("=" * 60)
    
    df = pd.read_csv(INPUT_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS and not c.startswith(EXCLUDE_PREFIXES)]
    subjects = sorted(df['subject'].unique())
    
    # 1. Load actual resting baselines
    print("Loading rest baselines...")
    baselines = compute_rest_baselines()
    
    # Convert to dataframe
    rest_df = pd.DataFrame.from_dict(baselines, orient='index')
    
    pheno_cols = list(rest_df.columns)
    
    n_clusters = 2
    
    template = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('spline', SplineTransformer(n_knots=4, degree=3, include_bias=False)),
        ('lr', LogisticRegression(solver='lbfgs', max_iter=5000, C=1.0))
    ])
    
    all_true_w, all_pred_w = [], []
    all_true_r, all_pred_r = [], []
    
    for si, test_subj in enumerate(subjects):
        train_subjs = [s for s in subjects if s != test_subj and s in rest_df.index]
        if test_subj not in rest_df.index:
            # Fallback if no rest data for test subj
            test_cluster = 0
            train_clusters = np.zeros(len(train_subjs))
        else:
            # Impute missing rest features (fit on train only)
            train_medians = rest_df.loc[train_subjs].median()
            X_train_prof = rest_df.loc[train_subjs].fillna(train_medians).values
            X_test_prof = rest_df.loc[[test_subj]].fillna(train_medians).values
            
            # Reduce to 2 most informative features (sympathetic/parasympathetic axis)
            # 34 points in 7D is too sparse for GMM — 34 points in 2D is tractable
            pheno_subset = ['rest_hr', 'rest_rmssd']
            avail = [c for c in pheno_subset if c in rest_df.columns]
            if avail:
                idx = [list(rest_df.columns).index(c) for c in avail]
                X_train_prof = X_train_prof[:, idx]
                X_test_prof = X_test_prof[:, idx]
            
            # Fit GMM on training subjects' rest profiles
            scaler_gmm = StandardScaler()
            X_train_prof_s = scaler_gmm.fit_transform(X_train_prof)
            X_test_prof_s = scaler_gmm.transform(X_test_prof)
            
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='diag',
                n_init=10,
                random_state=42
            )
            train_clusters = gmm.fit_predict(X_train_prof_s)
            test_cluster = gmm.predict(X_test_prof_s)[0]
            
            # Cluster split diagnostic
            split = np.bincount(train_clusters, minlength=n_clusters)
            if si == 0:
                print(f"  cluster split (fold 1): {split}")
        
        # Train a model for each cluster
        models = {}
        for c_idx in range(n_clusters):
            c_subjs = np.array(train_subjs)[train_clusters == c_idx]
            train_c = df[df['subject'].isin(c_subjs)]
            
            if len(train_c) == 0:
                continue
                
            feats_c = screen_features_silent(train_c, feat_cols)
            X_c = train_c[feats_c].values
            y_c = train_c['difficulty'].values.astype(int)
            
            m = clone(template)
            m.fit(X_c, y_c)
            models[c_idx] = (m, feats_c)
            
        # Predict test subject
        test = df[df['subject'] == test_subj]
        y_test = test['difficulty'].values.astype(int)
        
        # Fallback if cluster model is empty
        if test_cluster not in models:
            test_cluster = list(models.keys())[0] if models else 0
            
        if not models:
            print(f"  [WARN] No models trained for fold {si+1}")
            continue
            
        m, feats_c = models[test_cluster]
        X_test = test[feats_c].values
        
        y_pred_w = m.predict(X_test)
        
        all_true_w.extend(y_test)
        all_pred_w.extend(y_pred_w)
        
        # Run-level
        test_copy = test[['run_folder', 'difficulty']].copy()
        test_copy['pred'] = y_pred_w
        for run_name, rg in test_copy.groupby('run_folder'):
            run_true = int(rg['difficulty'].iloc[0])
            vote = int(scipy_mode(rg['pred'].values, keepdims=False).mode)
            all_true_r.append(run_true)
            all_pred_r.append(vote)
            
        print(f"  fold {si+1}/{len(subjects)} | subj: {test_subj} | cluster: {test_cluster} | feats: {len(feats_c)} | train_n: {len(train_subjs)}")
        
    all_true_w = np.array(all_true_w)
    all_pred_w = np.array(all_pred_w)
    all_true_r = np.array(all_true_r)
    all_pred_r = np.array(all_pred_r)
    
    acc_w = accuracy_score(all_true_w, all_pred_w)
    adj_w = np.mean(np.abs(all_true_w - all_pred_w) <= 1)
    acc_r = accuracy_score(all_true_r, all_pred_r)
    adj_r = np.mean(np.abs(all_true_r - all_pred_r) <= 1)
    
    print(f"\n{'=' * 60}")
    print("RESULTS (GMM PHENOTYPING + LOSO)")
    print(f"{'=' * 60}")
    print(f"WINDOW LEVEL:")
    print(f"  4-class: {acc_w*100:.1f}% | adjacent: {adj_w*100:.1f}%")
    print(f"RUN LEVEL:")
    print(f"  4-class: {acc_r*100:.1f}% | adjacent: {adj_r*100:.1f}%")

if __name__ == '__main__':
    run_gmm_model()
