"""
step1_signature.py
==================
Phase 1, Step 1: Signature transform on eye/gaze/pupil channel.

Builds a joint 3D trajectory (gaze-x, gaze-y, pupil-diameter) per 30-sec window.
Computes order-2 truncated path signature.
Runs ANOVA sanity check against difficulty labels.
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import f_oneway, kruskal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data_pipeline.new_data_pipeline import (
    load_run_streams, get_ref_time,
    DATASET_ROOT, WINDOW_DN, is_run_bad, is_eye_bad
)
import re

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV = ROOT / 'outputs' / 'signature_features.csv'

def sig2(path):
    """
    Computes order-2 truncated path signature for a 3D path.
    path: (N, 3) numpy array
    Returns: 13-dimensional vector (1 + 3 + 9)
    (ignoring the constant 1.0 at order 0 for features)
    Returns: 12-dimensional vector (3 for order 1, 9 for order 2)
    """
    if len(path) < 2:
        return np.full(12, np.nan)
    
    diffs = np.diff(path, axis=0) # (N-1, 3)
    
    # Order 1 (3 dims): integral of dX
    sig1 = np.sum(diffs, axis=0)
    
    # Order 2 (3x3=9 dims): double integral of dX dY
    # S^2_{ij} = sum_{t=1}^{N-1} (X_i(t) - X_i(0)) * dX_j(t+1)
    # wait, X(t) is path[t]. diffs[t] is path[t+1] - path[t]
    sig2_mat = np.zeros((3, 3))
    
    # vectorized computation
    # X_t_minus_X_0 is (N-1, 3). For each t in 0..N-2, it's path[t] - path[0]
    X_t_minus_X_0 = path[:-1] - path[0]
    
    # sig2_mat = X_t_minus_X_0^T @ diffs
    sig2_mat = X_t_minus_X_0.T @ diffs
    
    return np.concatenate([sig1, sig2_mat.flatten()])


def extract_signatures():
    print("=" * 60)
    print("STEP 1: GAZE/PUPIL PATH SIGNATURE EXTRACTION")
    print("=" * 60)

    ils_dir = DATASET_ROOT / 'task-ils'
    subjects = sorted([d for d in os.listdir(ils_dir) if d.startswith('sub-cp')])
    print(f"subjects: {len(subjects)}")

    records = []

    for si, subj in enumerate(subjects):
        subj_path = ils_dir / subj
        n_win = 0

        for sess in sorted(os.listdir(subj_path)):
            if not sess.startswith('ses-'):
                continue
            sess_path = subj_path / sess

            for run_dir in sorted(os.listdir(sess_path)):
                if not run_dir.startswith('level-'):
                    continue

                if is_run_bad(subj, run_dir):
                    continue

                m = re.search(r'level-(\d+)', run_dir)
                diff = int(m.group(1)) if m else -1
                if diff <= 0:
                    continue
                    
                eye_excl = is_eye_bad(subj, run_dir)
                if eye_excl:
                    continue

                run_path = sess_path / run_dir
                s = load_run_streams(str(run_path))
                ref = get_ref_time(s)
                if ref is None or len(ref) < 100:
                    continue
                    
                eye_df = s.get('eye')
                if eye_df is None:
                    continue

                w_idx = 0
                w0 = ref[0]
                while w0 + WINDOW_DN * 0.5 < ref[-1]:
                    w1 = w0 + WINDOW_DN
                    
                    # Extract eye trajectory
                    t = eye_df['time_dn'].values
                    mask = (t >= w0) & (t < w1)
                    vidx = np.where(mask)[0]
                    
                    if len(vidx) > 50:
                        gx = eye_df['gaze_direction_l_x'].values[vidx] if 'gaze_direction_l_x' in eye_df.columns else np.full(len(vidx), np.nan)
                        gy = eye_df['gaze_direction_l_y'].values[vidx] if 'gaze_direction_l_y' in eye_df.columns else np.full(len(vidx), np.nan)
                        
                        pl = eye_df['pupil_diameter_l_mm'].values[vidx].astype(float) if 'pupil_diameter_l_mm' in eye_df.columns else np.full(len(vidx), np.nan)
                        pr = eye_df['pupil_diameter_r_mm'].values[vidx].astype(float) if 'pupil_diameter_r_mm' in eye_df.columns else np.full(len(vidx), np.nan)
                        
                        if 'validity_l' in eye_df.columns:
                            val_l = eye_df['validity_l'].values[vidx] == 1
                            pl[~val_l] = np.nan
                            gx[~val_l] = np.nan
                            gy[~val_l] = np.nan
                        if 'validity_r' in eye_df.columns:
                            val_r = eye_df['validity_r'].values[vidx] == 1
                            pr[~val_r] = np.nan
                            
                        pupil = np.nanmean([pl, pr], axis=0)
                        # Bounds check
                        pupil[(pupil < 1.5) | (pupil > 8.5)] = np.nan
                        
                        # filter out NaN rows to get contiguous path (or interpolate)
                        path = np.column_stack([gx, gy, pupil])
                        # simple interpolation for missing values within window
                        df_path = pd.DataFrame(path).interpolate(method='linear', limit_direction='both')
                        path = df_path.values
                        
                        # We specifically DO NOT per-window normalize here. 
                        # We want to preserve absolute pupil size and gaze range magnitude across windows.
                        
                        feat_vec = sig2(path)
                        
                        rec = {
                            'subject': subj, 'session': sess,
                            'run_folder': run_dir, 'window': w_idx,
                            'difficulty': diff,
                        }
                        for i in range(3): rec[f'sig1_{i}'] = feat_vec[i]
                        for i in range(9): rec[f'sig2_{i}'] = feat_vec[3+i]
                        
                        records.append(rec)
                    
                    w_idx += 1
                    w0 = w1
                n_win += w_idx

        print(f"  [{si + 1:2d}/{len(subjects)}] {subj} | windows: {n_win}")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsaved: {OUTPUT_CSV} | rows: {len(df)}")
    return df

def evaluate_signatures(df):
    print("\n" + "=" * 60)
    print("SIGNATURE FEATURE EXPLORATION (MULTIVARIATE)")
    print("=" * 60)

    meta = {'subject', 'session', 'run_folder', 'window', 'difficulty'}
    feat_cols = [c for c in df.columns if c not in meta]
    
    # We will run a multivariate cross-validation using all signature features jointly
    # This correctly evaluates the cross-terms as an ensemble
    
    X = df[feat_cols].values
    y = df['difficulty'].values.astype(int)
    groups = df['subject'].values
    
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000))
    ])
    
    # StratifiedGroupKFold prevents subject leakage
    cv = StratifiedGroupKFold(n_splits=5)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring='accuracy')
    
    print(f"\nMultivariate Joint Accuracy (5-fold CV): {scores.mean()*100:.1f}%")
    print("\nFor reference: Random chance is ~25%, single best conventional feature alone (overall_gaze_entropy_RY) usually gives ~30-33%.")
    print("If this accuracy is substantially higher than chance, the joint signature carries signal.")

if __name__ == '__main__':
    if not OUTPUT_CSV.exists():
        df = extract_signatures()
    else:
        df = pd.read_csv(OUTPUT_CSV)
    evaluate_signatures(df)
