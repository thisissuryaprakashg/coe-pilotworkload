"""
ekf_workload.py
===============
Extended Kalman Filter (EKF) for Pilot Workload Estimation.
Evaluated on CogPilot 30-sec windowed physiological features under strict Leave-One-Subject-Out (LOSO).

Formulation:
  State vector: x_k = [w_k, dw_k]^T
    w_k  : continuous latent workload (1.0 to 4.0)
    dw_k : workload drift rate (rate of change)

  Process model:
    x_k = F @ x_{k-1} + w_{k-1},  w ~ N(0, Q)
    F = [[1, dt],
         [0,  1]]  where dt = 1 (1 window step = 30s)

  Observation model:
    z_k = h(w_k) + v_k,  v ~ N(0, R)
    h(w) maps scalar workload w to expected soft-probability vector.
    Jacobian H_k = dh/dx evaluated at x_{k|k-1}.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from scipy.stats import f_oneway, mode as scipy_mode

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / 'outputs' / 'enhanced_features.csv'
META_COLS = {'subject', 'session', 'run_folder', 'run_number', 'difficulty', 'window'}
EXCLUDE_PREFIXES = ('ac_',)

warnings.filterwarnings('ignore')

def fast_screen_features(df_train, feat_cols):
    """Fast, vectorized ANOVA + VIF screening on training data only."""
    nan_pct = df_train[feat_cols].isna().mean()
    keep = [c for c in feat_cols if nan_pct[c] <= 0.30]
    stds = df_train[keep].std()
    keep = [c for c in keep if stds[c] > 1e-10]

    df_clean = df_train[['difficulty'] + keep].copy()
    for c in keep:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())

    groups = [df_clean.loc[df_clean['difficulty'] == d] for d in [1, 2, 3, 4]]
    sig = []
    for feat in keep:
        try:
            _, p = f_oneway(*[g[feat].values for g in groups])
            if p < 0.05:
                sig.append(feat)
        except Exception:
            pass

    current = sig.copy()
    # Fast matrix inversion VIF (1000x faster than looping lstsq)
    for _ in range(80):
        if len(current) < 3:
            break
        X = df_clean[current].values
        m, s = X.mean(0), X.std(0)
        s[s < 1e-10] = 1.0
        Xz = (X - m) / s
        corr = (Xz.T @ Xz) / (len(Xz) - 1)
        
        try:
            inv_corr = np.linalg.inv(corr + np.eye(len(current)) * 1e-6)
        except np.linalg.LinAlgError:
            inv_corr = np.linalg.pinv(corr)
            
        vifs = np.diag(inv_corr)
        max_idx = np.argmax(vifs)
        max_vif = vifs[max_idx]
        if max_vif <= 10.0:
            break
        del current[max_idx]
        
    return current


class ExtendedKalmanFilterWorkload:
    def __init__(self, q_w=0.03, q_dw=0.005, r_scale=1.0, dt=1.0):
        self.dt = dt
        self.F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        self.Q = np.array([
            [q_w, 0.0],
            [0.0, q_dw]
        ])
        self.r_scale = r_scale

    def fit_observation_model(self, y_train, probs_train):
        w = y_train.astype(float)
        self.poly_coeffs = []
        residuals = []
        
        for c in range(probs_train.shape[1]):
            target = probs_train[:, c]
            p = np.polyfit(w, target, deg=2)
            self.poly_coeffs.append(p)
            pred = np.polyval(p, w)
            res = target - pred
            residuals.append(float(np.var(res) + 1e-4))
            
        self.R = np.diag(residuals) * self.r_scale

    def h(self, w):
        w_clipped = np.clip(w, 1.0, 4.0)
        vals = np.array([np.polyval(p, w_clipped) for p in self.poly_coeffs])
        vals = np.clip(vals, 1e-6, 1.0)
        return vals / np.sum(vals)

    def dh_dw(self, w):
        w_clipped = np.clip(w, 1.0, 4.0)
        dh_dw_vec = np.array([2.0 * p[0] * w_clipped + p[1] for p in self.poly_coeffs])
        H = np.zeros((len(self.poly_coeffs), 2))
        H[:, 0] = dh_dw_vec
        H[:, 1] = 0.0
        return H

    def filter_run(self, obs_sequence, initial_guess=2.5):
        N = len(obs_sequence)
        smoothed_w = np.zeros(N)
        filtered_classes = np.zeros(N, dtype=int)
        
        x_est = np.array([float(initial_guess), 0.0])
        P = np.array([
            [1.0, 0.0],
            [0.0, 0.2]
        ])
        
        for k in range(N):
            # Predict
            x_pred = self.F @ x_est
            x_pred[0] = np.clip(x_pred[0], 1.0, 4.0)
            P_pred = self.F @ P @ self.F.T + self.Q
            
            # Update
            z_obs = obs_sequence[k]
            z_pred = self.h(x_pred[0])
            y_innov = z_obs - z_pred
            
            H = self.dh_dw(x_pred[0])
            S = H @ P_pred @ H.T + self.R
            
            try:
                K = P_pred @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = P_pred @ H.T @ np.linalg.pinv(S)
                
            x_est = x_pred + K @ y_innov
            x_est[0] = np.clip(x_est[0], 1.0, 4.0)
            
            I_KH = np.eye(2) - K @ H
            P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
            
            smoothed_w[k] = x_est[0]
            filtered_classes[k] = int(np.clip(np.round(x_est[0]), 1, 4))
            
        return smoothed_w, filtered_classes


def evaluate_ekf_loso():
    print("=" * 70)
    print("EXTENDED KALMAN FILTER (EKF) WORKLOAD TRACKING")
    print("Dataset: CogPilot (Enhanced Features)")
    print("Validation: Leak-Free Leave-One-Subject-Out (LOSO)")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS and not any(c.startswith(p) for p in EXCLUDE_PREFIXES)]
    subjects = sorted(df['subject'].unique())
    print(f"Total Subjects: {len(subjects)} | Total Windows: {len(df)}")

    all_true_w = []
    all_pred_raw_w = []
    all_pred_ekf_w = []

    all_true_r = []
    all_pred_r_vote = []
    all_pred_r_prob = []
    all_pred_r_ekf = []

    ekf = ExtendedKalmanFilterWorkload(q_w=0.03, q_dw=0.005, r_scale=1.2)

    for si, test_subj in enumerate(subjects):
        train = df[df['subject'] != test_subj].copy()
        test = df[df['subject'] == test_subj].copy()

        # 1. Feature Screening strictly on training data
        features = fast_screen_features(train, feat_cols)

        X_train = train[features].values
        y_train = train['difficulty'].values.astype(int)
        X_test = test[features].values
        y_test = test['difficulty'].values.astype(int)

        # 2. Fit base classifier
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ('lr', LogisticRegression(solver='lbfgs', max_iter=1000, tol=1e-3, C=0.1))
        ])
        model.fit(X_train, y_train)

        # 3. Fit EKF Observation Model on training fold
        probs_train = model.predict_proba(X_train)
        ekf.fit_observation_model(y_train, probs_train)

        # 4. Predict on test fold
        w_proba = model.predict_proba(X_test)
        y_pred_raw = np.argmax(w_proba, axis=1) + 1

        all_true_w.extend(y_test)
        all_pred_raw_w.extend(y_pred_raw)

        # 5. EKF Run Filtering
        test['pred_raw'] = y_pred_raw
        for c in range(4):
            test[f'prob_{c}'] = w_proba[:, c]

        for run_name, rg in test.sort_values('window').groupby('run_folder'):
            run_true = int(rg['difficulty'].iloc[0])
            all_true_r.append(run_true)

            # Baseline 1: Majority Voting
            vote = int(scipy_mode(rg['pred_raw'].values, keepdims=False).mode)
            all_pred_r_vote.append(vote)

            # Baseline 2: Probability Average
            avg_prob = rg[[f'prob_{c}' for c in range(4)]].mean(axis=0).values
            prob_vote = int(np.argmax(avg_prob) + 1)
            all_pred_r_prob.append(prob_vote)

            # EKF Sequential Filter
            obs_seq = rg[[f'prob_{c}' for c in range(4)]].values
            init_val = float(np.argmax(obs_seq[0]) + 1)
            smoothed_w, ekf_classes = ekf.filter_run(obs_seq, initial_guess=init_val)

            all_pred_ekf_w.extend(ekf_classes)

            # Run-level EKF prediction (mean continuous state over run)
            ekf_run_pred = int(np.clip(np.round(np.mean(smoothed_w)), 1, 4))
            all_pred_r_ekf.append(ekf_run_pred)

        print(f"  Fold {si+1:2d}/{len(subjects)} | Subject: {test_subj:10s} | Feats: {len(features):2d}")

    # Metrics
    all_true_w = np.array(all_true_w)
    all_pred_raw_w = np.array(all_pred_raw_w)
    all_pred_ekf_w = np.array(all_pred_ekf_w)

    all_true_r = np.array(all_true_r)
    all_pred_r_vote = np.array(all_pred_r_vote)
    all_pred_r_prob = np.array(all_pred_r_prob)
    all_pred_r_ekf = np.array(all_pred_r_ekf)

    def calc_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        adj = float(np.mean(np.abs(y_true - y_pred) <= 1))
        b_true = (y_true >= 3).astype(int)
        b_pred = (y_pred >= 3).astype(int)
        bin_acc = accuracy_score(b_true, b_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        kappa = cohen_kappa_score(y_true, y_pred)
        return acc, adj, bin_acc, f1, kappa

    w_raw_acc, w_raw_adj, w_raw_bin, w_raw_f1, w_raw_k = calc_metrics(all_true_w, all_pred_raw_w)
    w_ekf_acc, w_ekf_adj, w_ekf_bin, w_ekf_f1, w_ekf_k = calc_metrics(all_true_w, all_pred_ekf_w)

    r_vote_acc, r_vote_adj, r_vote_bin, r_vote_f1, r_vote_k = calc_metrics(all_true_r, all_pred_r_vote)
    r_prob_acc, r_prob_adj, r_prob_bin, r_prob_f1, r_prob_k = calc_metrics(all_true_r, all_pred_r_prob)
    r_ekf_acc, r_ekf_adj, r_ekf_bin, r_ekf_f1, r_ekf_k = calc_metrics(all_true_r, all_pred_r_ekf)

    print("\n" + "=" * 70)
    print("FINAL RESULTS: BASELINES VS EXTENDED KALMAN FILTER (EKF)")
    print("=" * 70)
    print(f"\nWINDOW LEVEL ({len(all_true_w)} windows):")
    print(f"  Raw Spline:   4-class={w_raw_acc:.1%} | adjacent={w_raw_adj:.1%} | binary={w_raw_bin:.1%} | F1={w_raw_f1:.4f} | kappa={w_raw_k:.4f}")
    print(f"  EKF Smoothed: 4-class={w_ekf_acc:.1%} | adjacent={w_ekf_adj:.1%} | binary={w_ekf_bin:.1%} | F1={w_ekf_f1:.4f} | kappa={w_ekf_k:.4f}")

    print(f"\nRUN LEVEL ({len(all_true_r)} runs):")
    print(f"  Majority Vote (Baseline): 4-class={r_vote_acc:.1%} | adjacent={r_vote_adj:.1%} | binary={r_vote_bin:.1%} | F1={r_vote_f1:.4f} | kappa={r_vote_k:.4f}")
    print(f"  Prob Average (Baseline):  4-class={r_prob_acc:.1%} | adjacent={r_prob_adj:.1%} | binary={r_prob_bin:.1%} | F1={r_prob_f1:.4f} | kappa={r_prob_k:.4f}")
    print(f"  EKF Workload Filter:      4-class={r_ekf_acc:.1%} | adjacent={r_ekf_adj:.1%} | binary={r_ekf_bin:.1%} | F1={r_ekf_f1:.4f} | kappa={r_ekf_k:.4f}")

    print("\nEKF RUN-LEVEL CONFUSION MATRIX:")
    cm = confusion_matrix(all_true_r, all_pred_r_ekf, labels=[1, 2, 3, 4])
    print(f"  {'pred ->':>8s} {'1':>6s} {'2':>6s} {'3':>6s} {'4':>6s}")
    for i, row in enumerate(cm):
        pct = row[i] / max(1, sum(row)) * 100
        print(f"    L{i+1}     {row[0]:6d} {row[1]:6d} {row[2]:6d} {row[3]:6d}  ({pct:.1f}% correct)")

    print("=" * 70)


if __name__ == '__main__':
    evaluate_ekf_loso()
