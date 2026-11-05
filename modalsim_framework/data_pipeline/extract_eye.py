"""
extract_eye.py
==============
Extracts pupillometry and oculomotor gaze dynamics from MODALSIM94 Dikablis files.
Features:
  - pupil_area_mean, pupil_area_std, pupil_area_range
  - pupil_height_mean, pupil_width_mean
  - gaze_dispersion_x, gaze_dispersion_y
  - blink_closure_fraction (pupil area <= 0 or missing)
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def extract_eye_features(modalsim_root):
    eye_dir = Path(modalsim_root) / 'Dikablis'
    files = glob.glob(str(eye_dir / 'Dikablis_*.csv'))
    
    records = []
    for f in files:
        fname = os.path.basename(f)
        parts = fname.replace('Dikablis_', '').replace('.csv', '').split('_')
        if len(parts) != 2:
            continue
        subj, scen = parts[0], parts[1]
        if scen not in ['NBTask', '1BTask', '2BTask']:
            continue
            
        try:
            df = pd.read_csv(f, sep=';', low_memory=False)
            
            # Numeric conversion
            num_cols = ['LEPupilArea', 'REPupilArea', 'LEPupilHeight', 'LEPupilWidth', 
                        'REPupilHeight', 'REPupilWidth', 'LEPupilX', 'LEPupilY', 'REPupilX', 'REPupilY']
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
                    
            # Combined pupil area (use copy=True so arrays are writable)
            p_left = df['LEPupilArea'].to_numpy(dtype=float, copy=True) if 'LEPupilArea' in df.columns else np.full(len(df), np.nan)
            p_right = df['REPupilArea'].to_numpy(dtype=float, copy=True) if 'REPupilArea' in df.columns else np.full(len(df), np.nan)
            
            # Valid pupil range in pixels/area (filter zero / negative)
            valid_l = (p_left > 10.0) & (p_left < 15000.0)
            valid_r = (p_right > 10.0) & (p_right < 15000.0)
            
            p_left[~valid_l] = np.nan
            p_right[~valid_r] = np.nan
            
            combined_pupil = np.nanmean(np.column_stack([p_left, p_right]), axis=1)
            valid_mask = np.isfinite(combined_pupil)
            
            n_valid = np.sum(valid_mask)
            if n_valid < 100:
                continue
                
            clean_pupil = combined_pupil[valid_mask]
            closure_frac = 1.0 - (n_valid / len(df))
            
            # Gaze coordinates
            gx = df['LEPupilX'].to_numpy(dtype=float, copy=True)[valid_mask] if 'LEPupilX' in df.columns else np.array([])
            gy = df['LEPupilY'].to_numpy(dtype=float, copy=True)[valid_mask] if 'LEPupilY' in df.columns else np.array([])
            
            rec = {
                'subject': subj,
                'scenario': scen,
                'eye_pupil_area_mean': float(np.mean(clean_pupil)),
                'eye_pupil_area_std': float(np.std(clean_pupil)),
                'eye_pupil_area_range': float(np.ptp(clean_pupil)),
                'eye_blink_closure_frac': float(closure_frac),
                'eye_gaze_dispersion_x': float(np.std(gx)) if len(gx) > 20 else np.nan,
                'eye_gaze_dispersion_y': float(np.std(gy)) if len(gy) > 20 else np.nan,
            }
            records.append(rec)
        except Exception as e:
            continue
            
    if not records:
        print("[Dikablis Eye] Warning: No records extracted.")
        return pd.DataFrame(columns=['subject', 'scenario', 'eye_pupil_area_mean', 'eye_pupil_area_std', 
                                     'eye_pupil_area_range', 'eye_blink_closure_frac', 
                                     'eye_gaze_dispersion_x', 'eye_gaze_dispersion_y'])
    eye_df = pd.DataFrame(records).drop_duplicates(subset=['subject', 'scenario'])
    print(f"[Dikablis Eye] Extracted {len(eye_df)} records across {eye_df['subject'].nunique()} participants.")
    return eye_df

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2] / 'MODALSIM94'
    df_eye = extract_eye_features(root)
    print(df_eye.head())

