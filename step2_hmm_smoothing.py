"""
step2_hmm_smoothing.py
======================
Phase 1, Step 2: HMM Smoothing at the Run Level

Replaces majority voting with a 4-state HMM using Viterbi decoding.
The transition matrix is estimated empirically from the training-fold
run sequences (with Laplace smoothing). This smooths out anomalous
window predictions without leaking data.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import json

from improved_model import screen_features_silent, INPUT_CSV, META_COLS, EXCLUDE_PREFIXES

warnings.filterwarnings('ignore')

def viterbi(obs_probs, trans_mat, start_probs):
    """
    Standard Viterbi algorithm.
    obs_probs: (N, K) array of emission probabilities P(x_t | y_t)
    trans_mat: (K, K) array of transition probabilities P(y_t | y_{t-1})
    start_probs: (K,) array of starting probabilities
    """
    n_obs, n_states = obs_probs.shape
    
    # We work in log space to prevent underflow
    log_obs = np.log(np.clip(obs_probs, 1e-12, 1.0))
    log_trans = np.log(np.clip(trans_mat, 1e-12, 1.0))
    log_start = np.log(np.clip(start_probs, 1e-12, 1.0))
    
    v = np.zeros((n_obs, n_states))
    ptr = np.zeros((n_obs, n_states), dtype=int)
    
    v[0, :] = log_start + log_obs[0, :]
    
    for t in range(1, n_obs):
        for s in range(n_states):
            # prob of transitioning to state s from all previous states k
            trans_probs = v[t-1, :] + log_trans[:, s]
            best_prev = np.argmax(trans_probs)
            v[t, s] = trans_probs[best_prev] + log_obs[t, s]
            ptr[t, s] = best_prev
            
    # backtrack
    path = np.zeros(n_obs, dtype=int)
    path[-1] = np.argmax(v[-1, :])
    for t in range(n_obs-2, -1, -1):
        path[t] = ptr[t+1, path[t+1]]
        
    return path

def run_hmm_model():
    print("=" * 60)
    print("STEP 2: HMM SMOOTHING (SPLINE LOGISTIC)")
    print("=" * 60)
    
    df = pd.read_csv(INPUT_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS and not c.startswith(EXCLUDE_PREFIXES)]
    subjects = sorted(df['subject'].unique())
    
    # Template model: matches improved_model.py best config (n_knots=5, C=0.1, single scaler)
    template = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
        ('lr', LogisticRegression(solver='lbfgs', max_iter=5000, C=0.1))
    ])
    
    all_true_w = []
    all_pred_w = []
    all_pred_hmm = []
    
    all_true_r = []
    all_pred_r_vote = []
    all_pred_r_prob = []
    all_pred_r_hmm = []
    
    for si, test_subj in enumerate(subjects):
        train = df[df['subject'] != test_subj].copy()
        test = df[df['subject'] == test_subj].copy()
        
        # 1. Feature Screening (Train only)
        features = screen_features_silent(train, feat_cols)
        
        X_train = train[features].values
        y_train = train['difficulty'].values.astype(int)
        X_test = test[features].values
        y_test = test['difficulty'].values.astype(int)
        
        # 2. Fit classifier
        model = template
        model.fit(X_train, y_train)
        
        # 3. Train HMM Transition Matrix
        # Count transitions in training runs (explicitly sorted by window)
        trans_counts = np.zeros((4, 4))
        for run_id, run_group in train.sort_values('window').groupby(['subject', 'run_folder']):
            y_run = run_group['difficulty'].values.astype(int) - 1 # 0-indexed
            for t in range(1, len(y_run)):
                trans_counts[y_run[t-1], y_run[t]] += 1
                
        # Laplace smoothing to allow state changes (just in case)
        trans_counts += 1.0 
        trans_mat = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        start_probs = np.ones(4) / 4.0 # Uniform start
        
        # 4. Predict on Test
        # Get class probabilities per window
        w_proba = model.predict_proba(X_test)
        
        # Predict individual windows (raw)
        y_pred_w = np.argmax(w_proba, axis=1) + 1
        all_true_w.extend(y_test)
        all_pred_w.extend(y_pred_w)
        
        # 5. HMM Decoding per Run
        test['pred_raw'] = y_pred_w
        for c in range(4):
            test[f'prob_{c}'] = w_proba[:, c]
            
        for run_name, rg in test.sort_values('window').groupby('run_folder'):
            run_true = int(rg['difficulty'].iloc[0])
            all_true_r.append(run_true)
            
            # Majority voting (baseline)
            vote = int(rg['pred_raw'].mode()[0])
            all_pred_r_vote.append(vote)
            
            # Probability average (baseline)
            avg_prob = rg[[f'prob_{c}' for c in range(4)]].mean(axis=0).values
            prob_vote = int(np.argmax(avg_prob) + 1)
            all_pred_r_prob.append(prob_vote)
            
            # HMM Smoothing
            obs_probs = rg[[f'prob_{c}' for c in range(4)]].values
            hmm_path = viterbi(obs_probs, trans_mat, start_probs)
            hmm_preds = hmm_path + 1 # back to 1-indexed
            
            all_pred_hmm.extend(hmm_preds)
            
            # Run-level HMM prediction (mode of smoothed path)
            # Since HMM usually locks into a single state, this is very stable
            hmm_run_vote = int(pd.Series(hmm_preds).mode()[0])
            all_pred_r_hmm.append(hmm_run_vote)
            
        print(f"  fold {si+1}/{len(subjects)} | subj: {test_subj} | feats: {len(features)}")
        
    # --- EVALUATION ---
    print(f"\n{'=' * 60}")
    print("RESULTS (LEAK-FREE LOSO)")
    print(f"{'=' * 60}")
    
    all_true_w = np.array(all_true_w)
    all_pred_w = np.array(all_pred_w)
    all_pred_hmm = np.array(all_pred_hmm)
    
    all_true_r = np.array(all_true_r)
    all_pred_r_vote = np.array(all_pred_r_vote)
    all_pred_r_prob = np.array(all_pred_r_prob)
    all_pred_r_hmm = np.array(all_pred_r_hmm)
    
    def metrics(true, pred):
        acc = accuracy_score(true, pred)
        adj = np.mean(np.abs(true - pred) <= 1)
        bin_acc = accuracy_score(true >= 3, pred >= 3)
        return acc, adj, bin_acc
        
    acc_w, adj_w, bin_w = metrics(all_true_w, all_pred_w)
    acc_h, adj_h, bin_h = metrics(all_true_w, all_pred_hmm)
    
    acc_rv, adj_rv, bin_rv = metrics(all_true_r, all_pred_r_vote)
    acc_rp, adj_rp, bin_rp = metrics(all_true_r, all_pred_r_prob)
    acc_rh, adj_rh, bin_rh = metrics(all_true_r, all_pred_r_hmm)
    
    print("\nWINDOW LEVEL (Raw independent vs HMM smoothed path):")
    print(f"  Raw Spline: 4-class={acc_w*100:.1f}% | adj={adj_w*100:.1f}% | bin={bin_w*100:.1f}%")
    print(f"  HMM Path:   4-class={acc_h*100:.1f}% | adj={adj_h*100:.1f}% | bin={bin_h*100:.1f}%")
    
    print("\nRUN LEVEL (Voting vs Prob-Avg vs HMM vote):")
    print(f"  Majority:   4-class={acc_rv*100:.1f}% | adj={adj_rv*100:.1f}% | bin={bin_rv*100:.1f}%")
    print(f"  Prob Avg:   4-class={acc_rp*100:.1f}% | adj={adj_rp*100:.1f}% | bin={bin_rp*100:.1f}%")
    print(f"  HMM Vote:   4-class={acc_rh*100:.1f}% | adj={adj_rh*100:.1f}% | bin={bin_rh*100:.1f}%")

if __name__ == "__main__":
    run_hmm_model()
