"""
step3_riemannian.py
===================
Phase 2, Step 3: Riemannian tangent-space features.

Extracts cross-modal covariance matrices per 30-second window.
Two evaluation tracks:
  PRIMARY:   7-channel (ECG, EDA, EMGf, EMGe, gazeX, gazeY, pupil)
             — evaluated over subjects with valid eye-tracking
  SECONDARY: 4-channel (ECG, EDA, EMGf, EMGe only)
             — evaluated over ALL subjects (no one dropped)

Interpolates to a common 64 Hz grid, computes OAS-shrunk covariance,
evaluates via PyRiemann TangentSpace in leak-free LOSO.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data_pipeline.new_data_pipeline import (
    load_run_streams, get_ref_time,
    DATASET_ROOT, WINDOW_DN, is_run_bad, is_eye_bad
)

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_7CH_COV = OUTPUT_DIR / 'covs_7ch.npy'
OUTPUT_7CH_META = OUTPUT_DIR / 'covs_7ch_meta.csv'
OUTPUT_4CH_COV = OUTPUT_DIR / 'covs_4ch.npy'
OUTPUT_4CH_META = OUTPUT_DIR / 'covs_4ch_meta.csv'

FS_COMMON = 64.0
MIN_VALID_FRAC = 0.30  # require 30% real samples per channel within a window

def interpolate_stream(t, y, t_new):
    """Returns (interpolated_signal, n_valid_samples)."""
    if len(t) < 2:
        return np.full(len(t_new), np.nan), 0
    valid = np.isfinite(y)
    n_valid = int(np.sum(valid))
    if n_valid < 2:
        return np.full(len(t_new), np.nan), 0
    t, y = t[valid], y[valid]
    
    t, unq = np.unique(t, return_index=True)
    y = y[unq]
    
    if len(t) < 2:
        return np.full(len(t_new), np.nan), 0
        
    f = interp1d(t, y, bounds_error=False, fill_value="extrapolate")
    return f(t_new), n_valid

def _channel_ok(interp_result, min_valid):
    """Check if an interpolated channel has enough real data."""
    signal, n_valid = interp_result
    if n_valid < min_valid:
        return False
    if np.all(~np.isfinite(signal)):
        return False
    return True

def _build_cov(X):
    """Standardize channels, compute OAS covariance, add SPD ridge."""
    # Interpolate remaining partial NaNs
    for i in range(X.shape[0]):
        bad = ~np.isfinite(X[i])
        if np.all(bad):
            return None  # should not happen if caller checked
        if np.any(bad):
            X[i, bad] = np.nanmean(X[i])
    
    # Standardize each channel (zero mean, unit variance)
    stds = np.std(X, axis=1, keepdims=True)
    stds[stds < 1e-10] = 1.0
    X = (X - np.mean(X, axis=1, keepdims=True)) / stds
    
    # OAS shrinkage covariance
    if HAS_PYRIEMANN:
        cov = Covariances(estimator='oas').fit_transform(X[np.newaxis, :, :])[0]
    else:
        cov = np.cov(X)
    
    # Single ridge for SPD guarantee (only place this is applied)
    cov += np.eye(cov.shape[0]) * 1e-4
    return cov

def extract_covariances():
    print("=" * 60)
    print("STEP 3: DUAL COVARIANCE EXTRACTION (7-CH + 4-CH)")
    print("=" * 60)
    print(f"  min valid fraction per channel: {MIN_VALID_FRAC:.0%}")
    
    ils_dir = DATASET_ROOT / 'task-ils'
    subjects = sorted([d for d in os.listdir(ils_dir) if d.startswith('sub-cp')])
    
    covs_7ch, meta_7ch = [], []
    covs_4ch, meta_4ch = [], []
    
    # Per-subject tracking
    subj_7ch_kept = {}
    subj_4ch_kept = {}
    
    for si, subj in enumerate(subjects):
        subj_path = ils_dir / subj
        subj_7ch_kept[subj] = 0
        subj_4ch_kept[subj] = 0
        
        for sess in sorted(os.listdir(subj_path)):
            if not sess.startswith('ses-'): continue
            sess_path = subj_path / sess
            
            for run_dir in sorted(os.listdir(sess_path)):
                if not run_dir.startswith('level-'): continue
                if is_run_bad(subj, run_dir): continue
                
                rm = re.search(r'level-(\d+)', run_dir)
                diff = int(rm.group(1)) if rm else -1
                if diff <= 0: continue
                
                s = load_run_streams(str(sess_path / run_dir))
                ref = get_ref_time(s)
                if ref is None or len(ref) < 100: continue
                
                eye_excl = is_eye_bad(subj, run_dir)
                
                w_idx = 0
                w0 = ref[0]
                while w0 + WINDOW_DN * 0.5 < ref[-1]:
                    w1 = w0 + WINDOW_DN
                    t_new = np.arange(w0, w1, 1.0 / FS_COMMON)
                    min_valid = int(len(t_new) * MIN_VALID_FRAC)
                    
                    rec = {'subject': subj, 'session': sess,
                           'run_folder': run_dir, 'window': w_idx,
                           'difficulty': diff}
                    
                    # ---- Extract 4 physio channels ----
                    ch4 = []  # (signal, n_valid) tuples
                    
                    # ECG
                    ecg_sig = np.full(len(t_new), np.nan)
                    ecg_nv = 0
                    if s.get('ecg') is not None:
                        t = s['ecg']['time_dn'].values
                        m = (t >= w0) & (t < w1)
                        candidates = ['ecg_projection_ll_ra_mV', 'ecg_projection_la_ra_mV', 'ecg_projection_vx_rl_mV']
                        c = next((col for col in candidates if col in s['ecg'].columns), None)
                        if c:
                            ecg_sig, ecg_nv = interpolate_stream(t[m], s['ecg'][c].values.astype(float)[m], t_new)
                    ch4.append((ecg_sig, ecg_nv))
                    
                    # EDA
                    eda_sig = np.full(len(t_new), np.nan)
                    eda_nv = 0
                    if s.get('eda') is not None:
                        t = s['eda']['time_dn'].values
                        m = (t >= w0) & (t < w1)
                        candidates = ['eda_hand_l_kOhms', 'eda_conductance_uS']
                        c = next((col for col in candidates if col in s['eda'].columns), None)
                        if c:
                            eda_sig, eda_nv = interpolate_stream(t[m], s['eda'][c].values.astype(float)[m], t_new)
                    ch4.append((eda_sig, eda_nv))
                    
                    # EMG Flexor
                    flex_sig = np.full(len(t_new), np.nan)
                    flex_nv = 0
                    if s.get('emg') is not None:
                        t = s['emg']['time_dn'].values
                        m = (t >= w0) & (t < w1)
                        if 'emg_wrist_flexor_mV' in s['emg'].columns:
                            flex_sig, flex_nv = interpolate_stream(t[m], s['emg']['emg_wrist_flexor_mV'].values[m].astype(float), t_new)
                    ch4.append((flex_sig, flex_nv))
                    
                    # EMG Extensor
                    ext_sig = np.full(len(t_new), np.nan)
                    ext_nv = 0
                    if s.get('emg') is not None:
                        t = s['emg']['time_dn'].values
                        m = (t >= w0) & (t < w1)
                        if 'emg_wrist_extensor_mV' in s['emg'].columns:
                            ext_sig, ext_nv = interpolate_stream(t[m], s['emg']['emg_wrist_extensor_mV'].values[m].astype(float), t_new)
                    ch4.append((ext_sig, ext_nv))
                    
                    # ---- 4-channel covariance (if all 4 physio channels pass threshold) ----
                    all_4_ok = all(_channel_ok(ch, min_valid) for ch in ch4)
                    if all_4_ok:
                        X4 = np.vstack([ch[0] for ch in ch4])
                        cov4 = _build_cov(X4)
                        if cov4 is not None:
                            covs_4ch.append(cov4)
                            meta_4ch.append(rec.copy())
                            subj_4ch_kept[subj] += 1
                    
                    # ---- 7-channel covariance (if 4 physio + 3 eye all pass threshold) ----
                    if all_4_ok and not eye_excl:
                        ch_eye = []
                        if s.get('eye') is not None:
                            eye = s['eye']
                            t = eye['time_dn'].values
                            m = (t >= w0) & (t < w1)
                            
                            gx_sig = np.full(len(t_new), np.nan)
                            gx_nv = 0
                            if 'gaze_direction_l_x' in eye.columns:
                                gx_sig, gx_nv = interpolate_stream(t[m], eye['gaze_direction_l_x'].values[m], t_new)
                            ch_eye.append((gx_sig, gx_nv))
                            
                            gy_sig = np.full(len(t_new), np.nan)
                            gy_nv = 0
                            if 'gaze_direction_l_y' in eye.columns:
                                gy_sig, gy_nv = interpolate_stream(t[m], eye['gaze_direction_l_y'].values[m], t_new)
                            ch_eye.append((gy_sig, gy_nv))
                            
                            pl = eye['pupil_diameter_l_mm'].values[m].astype(float) if 'pupil_diameter_l_mm' in eye.columns else np.full(np.sum(m), np.nan)
                            pr = eye['pupil_diameter_r_mm'].values[m].astype(float) if 'pupil_diameter_r_mm' in eye.columns else np.full(np.sum(m), np.nan)
                            pup = np.nanmean([pl, pr], axis=0)
                            pup_sig, pup_nv = interpolate_stream(t[m], pup, t_new)
                            ch_eye.append((pup_sig, pup_nv))
                        else:
                            ch_eye = [(np.full(len(t_new), np.nan), 0)] * 3
                        
                        all_eye_ok = all(_channel_ok(ch, min_valid) for ch in ch_eye)
                        if all_eye_ok:
                            X7 = np.vstack([ch[0] for ch in ch4] + [ch[0] for ch in ch_eye])
                            cov7 = _build_cov(X7)
                            if cov7 is not None:
                                covs_7ch.append(cov7)
                                meta_7ch.append(rec.copy())
                                subj_7ch_kept[subj] += 1
                    
                    w_idx += 1
                    w0 = w1
        
        s7 = subj_7ch_kept[subj]
        s4 = subj_4ch_kept[subj]
        tag = ""
        if s7 == 0 and s4 > 0:
            tag = " [eye-excluded]"
        elif s4 == 0:
            tag = " [WARN: 0 windows]"
        print(f"  [{si+1:2d}/{len(subjects)}] {subj} | 7ch: {s7} | 4ch: {s4}{tag}")
    
    # Save 7-channel set
    arr_7 = np.array(covs_7ch)
    np.save(OUTPUT_7CH_COV, arr_7)
    pd.DataFrame(meta_7ch).to_csv(OUTPUT_7CH_META, index=False)
    
    # Save 4-channel set
    arr_4 = np.array(covs_4ch)
    np.save(OUTPUT_4CH_COV, arr_4)
    pd.DataFrame(meta_4ch).to_csv(OUTPUT_4CH_META, index=False)
    
    n7_subj = sum(1 for v in subj_7ch_kept.values() if v > 0)
    n4_subj = sum(1 for v in subj_4ch_kept.values() if v > 0)
    
    print(f"\n7-channel: {len(arr_7)} windows from {n7_subj} subjects | shape: {arr_7.shape}")
    print(f"4-channel: {len(arr_4)} windows from {n4_subj} subjects | shape: {arr_4.shape}")
    
    eye_excl_subjs = [s for s in subj_7ch_kept if subj_7ch_kept[s] == 0 and subj_4ch_kept[s] > 0]
    if eye_excl_subjs:
        print(f"  eye-excluded subjects (in 4ch only): {eye_excl_subjs}")
    
    return (arr_7, pd.DataFrame(meta_7ch)), (arr_4, pd.DataFrame(meta_4ch))

def _run_loso(covs, meta, label, C_val=0.1):
    """Run LOSO evaluation on a covariance set."""
    if not HAS_PYRIEMANN:
        print("\npyriemann not installed.")
        return
    
    print(f"\n{'=' * 60}")
    print(f"RIEMANNIAN LOSO: {label}")
    print(f"{'=' * 60}")
    
    subjects = sorted(meta['subject'].unique())
    
    # TangentSpace -> StandardScaler -> LogisticRegression
    model = Pipeline([
        ('ts', TangentSpace(metric='riemann')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(solver='lbfgs', max_iter=5000, C=C_val))
    ])
    
    all_true, all_pred = [], []
    skipped = []
    
    for si, test_subj in enumerate(subjects):
        test_mask = meta['subject'] == test_subj
        train_mask = ~test_mask
        
        X_test = covs[test_mask]
        y_test = meta['difficulty'].values[test_mask].astype(int)
        
        if len(X_test) == 0:
            skipped.append(test_subj)
            print(f"  fold {si+1}/{len(subjects)} | subj: {test_subj} | SKIPPED (0 windows)")
            continue
        
        X_train = covs[train_mask]
        y_train = meta['difficulty'].values[train_mask].astype(int)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        
        print(f"  fold {si+1}/{len(subjects)} | subj: {test_subj} | n={len(X_test)} | acc: {accuracy_score(y_test, y_pred):.3f}")
    
    if len(all_true) == 0:
        print("\n  no predictions made")
        return
        
    acc = accuracy_score(all_true, all_pred)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    adj = np.mean(np.abs(all_true - all_pred) <= 1)
    
    print(f"\nRESULTS ({label}):")
    print(f"  subjects: {len(subjects) - len(skipped)}/{len(subjects)} evaluated | windows: {len(all_true)}")
    if skipped:
        print(f"  skipped: {skipped}")
    print(f"  4-class: {acc*100:.1f}% | adjacent: {adj*100:.1f}%")

if __name__ == '__main__':
    # Delete stale caches
    for p in [OUTPUT_7CH_COV, OUTPUT_7CH_META, OUTPUT_4CH_COV, OUTPUT_4CH_META]:
        if p.exists(): p.unlink()
    
    (covs_7, meta_7), (covs_4, meta_4) = extract_covariances()
    
    # Primary: 7-channel (eye + physio) — most subjects
    _run_loso(covs_7, meta_7, "7-CHANNEL (ECG+EDA+EMG+EYE)", C_val=0.1)
    
    # Secondary: 4-channel (physio only) — ALL subjects, no one dropped
    _run_loso(covs_4, meta_4, "4-CHANNEL (ECG+EDA+EMG ONLY)", C_val=0.1)
