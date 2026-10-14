"""
enhance_features.py
====================
Enhances the windowed feature matrix with:
1. Subject-wise z-scoring using rest baseline (mean + std from windowed rest)
2. Window position feature (0-1 within each run)
3. Temporal delta features (change from first window of each run)
4. Per-run temporal slope features
Uses left-out subject's own rest baseline — no data leakage.
Output: enhanced_features.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# import feature extractors from pipeline
from src.data_pipeline.new_data_pipeline import (
    ecg_window, eda_window, emg_window, eye_window,
    load_run_streams, get_ref_time,
    DATASET_ROOT, WINDOW_DN, EYE_EXCLUDED_SUBJECTS,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / 'outputs' / 'new_master_features.csv'
OUTPUT_CSV = ROOT / 'outputs' / 'enhanced_features.csv'

# features to z-score (must exist in both task and rest windows)
Z_FEATURES = [
    'ecg_hr_mean', 'ecg_hr_std', 'ecg_hr_range',
    'ecg_rmssd', 'ecg_sdnn', 'ecg_pnn50', 'ecg_rr_cv', 'ecg_sampen',
    'eda_scl_mean', 'eda_cond_std', 'eda_scr_count', 'eda_phasic_energy',
    'emg_flexor_rms', 'emg_extensor_rms', 'emg_cocontraction', 'emg_acc_rms',
    'eye_pupil_mean', 'eye_pupil_std', 'eye_blink_rate', 'eye_perclos',
]

# features for temporal dynamics (slope + delta from first window)
TEMPORAL_FEATURES = [
    'ecg_hr_mean', 'ecg_rmssd',
    'eda_scl_mean',
    'emg_flexor_rms', 'emg_extensor_rms',
    'eye_pupil_mean', 'eye_blink_rate',
    'head_pitch_std',
]


# ============================================================
# REST BASELINE WITH WINDOWING
# ============================================================

def compute_rest_windowed_stats():
    """Process rest recordings with 30-sec windowing.
    Returns dict: subject -> {feature: (mean, std)}"""
    rest_dir = DATASET_ROOT / 'task-rest'
    if not rest_dir.exists():
        print("  [WARN] task-rest not found")
        return {}

    subjects = sorted([d for d in os.listdir(rest_dir) if d.startswith('sub-cp')])
    all_stats = {}

    for subj in subjects:
        subj_path = rest_dir / subj
        for sess in sorted(os.listdir(subj_path)):
            if not sess.startswith('ses-'):
                continue
            rest_path = subj_path / sess / 'level-000_run-001'
            if not rest_path.exists():
                continue

            s = load_run_streams(str(rest_path))
            ref = get_ref_time(s)
            if ref is None or len(ref) < 100:
                continue

            # window the rest recording into 30-sec segments
            rest_records = []
            w0 = ref[0]
            while w0 + WINDOW_DN * 0.5 < ref[-1]:
                w1 = w0 + WINDOW_DN
                rec = {}
                rec.update(ecg_window(s.get('ecg'), w0, w1, s.get('ecg_fs', 512)))
                rec.update(eda_window(s.get('eda'), w0, w1, s.get('eda_fs', 128)))
                rec.update(emg_window(s.get('emg'), w0, w1, s.get('emg_fs', 512)))
                rec.update(eye_window(s.get('eye'), w0, w1, subj in EYE_EXCLUDED_SUBJECTS))
                rest_records.append(rec)
                w0 = w1

            if len(rest_records) < 3:
                continue

            rdf = pd.DataFrame(rest_records)
            stats = {}
            for col in rdf.columns:
                vals = rdf[col].dropna()
                if len(vals) >= 3:
                    m, s_val = float(vals.mean()), float(vals.std())
                    stats[col] = (m, s_val if s_val > 1e-10 else 1e-6)

            all_stats[subj] = stats

    print(f"  rest windowed stats: {len(all_stats)} subjects")
    return all_stats


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FEATURE ENHANCEMENT")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"input: {df.shape[0]} rows x {df.shape[1]} cols")

    # --- 1. REST-BASED Z-SCORING ---
    print("\n--- 1. SUBJECT Z-SCORING (rest baseline) ---")
    rest_stats = compute_rest_windowed_stats()

    n_z = 0
    for feat in Z_FEATURES:
        if feat not in df.columns:
            continue
        col_name = f'z_{feat}'
        df[col_name] = np.nan

        for subj, stats in rest_stats.items():
            if feat not in stats:
                continue
            mean, std = stats[feat]
            mask = df['subject'] == subj
            df.loc[mask, col_name] = (df.loc[mask, feat] - mean) / std

        valid_pct = df[col_name].notna().mean() * 100
        if valid_pct > 50:
            n_z += 1
        else:
            df.drop(columns=[col_name], inplace=True)

    print(f"  z-scored features added: {n_z}")

    # --- 2. WINDOW POSITION ---
    print("\n--- 2. WINDOW POSITION ---")
    df['window_position'] = 0.0

    for key, grp in df.groupby(['subject', 'session', 'run_folder']):
        max_w = grp['window'].max()
        if max_w > 0:
            df.loc[grp.index, 'window_position'] = grp['window'] / max_w
        else:
            df.loc[grp.index, 'window_position'] = 0.0

    print(f"  window_position range: [{df['window_position'].min():.2f}, {df['window_position'].max():.2f}]")

    # --- 3. TEMPORAL DELTA FROM FIRST WINDOW ---
    print("\n--- 3. TEMPORAL DELTA FEATURES ---")
    n_td = 0
    for feat in TEMPORAL_FEATURES:
        if feat not in df.columns:
            continue
        col_name = f'tdelta_{feat}'
        df[col_name] = np.nan

        for key, grp in df.groupby(['subject', 'session', 'run_folder']):
            sorted_grp = grp.sort_values('window')
            first_val = sorted_grp[feat].iloc[0]
            if np.isfinite(first_val):
                df.loc[sorted_grp.index, col_name] = sorted_grp[feat] - first_val

        valid_pct = df[col_name].notna().mean() * 100
        if valid_pct > 50:
            n_td += 1
        else:
            df.drop(columns=[col_name], inplace=True)

    print(f"  temporal delta features added: {n_td}")

    # --- 4. PER-RUN TEMPORAL SLOPE ---
    print("\n--- 4. TEMPORAL SLOPE FEATURES ---")
    n_ts = 0
    for feat in TEMPORAL_FEATURES:
        if feat not in df.columns:
            continue
        col_name = f'slope_{feat}'
        df[col_name] = np.nan

        for key, grp in df.groupby(['subject', 'session', 'run_folder']):
            sorted_grp = grp.sort_values('window')
            vals = sorted_grp[feat].values
            valid = np.isfinite(vals)
            if np.sum(valid) >= 4:
                x = np.arange(len(vals))[valid]
                y = vals[valid]
                slope = np.polyfit(x, y, 1)[0]
                df.loc[sorted_grp.index, col_name] = slope

        valid_pct = df[col_name].notna().mean() * 100
        if valid_pct > 50:
            n_ts += 1
        else:
            df.drop(columns=[col_name], inplace=True)

    print(f"  temporal slope features added: {n_ts}")

    # --- SUMMARY ---
    meta_cols = {'subject', 'session', 'run_folder', 'run_number', 'difficulty', 'window'}
    feat_cols = [c for c in df.columns if c not in meta_cols]

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"total features: {len(feat_cols)}")

    print(f"\nfeatures by type:")
    for prefix, label in [('ecg_', 'ECG'), ('eda_', 'EDA'), ('emg_', 'EMG'),
                          ('eye_', 'Eye'), ('head_', 'Head'), ('delta_', 'Delta'),
                          ('z_', 'Z-scored'), ('tdelta_', 'Temporal-delta'),
                          ('slope_', 'Temporal-slope'), ('window_', 'Window-pos')]:
        cols = [c for c in feat_cols if c.startswith(prefix)]
        if cols:
            pct = df[cols].notna().mean().mean() * 100
            print(f"  {label:18s}: {len(cols):3d} features | {pct:5.1f}% non-NaN")

    other = [c for c in feat_cols if not any(c.startswith(p) for p in
             ['ecg_', 'eda_', 'emg_', 'eye_', 'head_', 'ac_', 'delta_',
              'z_', 'tdelta_', 'slope_', 'window_'])]
    if other:
        pct = df[other].notna().mean().mean() * 100
        print(f"  {'Oculomotor':18s}: {len(other):3d} features | {pct:5.1f}% non-NaN")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsaved: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
