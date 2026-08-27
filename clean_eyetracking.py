"""
clean_eyetracking.py
====================
Processes raw HTC Vive Pro Eye data:
- Drops samples where validity_l == 0 and validity_r == 0
- Filters pupil blinks & optical artifacts
- Extracts pupil diameter statistics, gaze direction dynamics, and spatial entropy
- Respects exclusion masks for sub-cp003 and sub-cp027 bad eye runs
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, kurtosis
from cleaning_rules import is_eye_excluded

def clean_and_extract_eye_features(df_eye: pd.DataFrame, subject: str, run_name: str) -> dict:
    """
    Cleans eye-tracking data per sample, extracts pupillometry and gaze features,
    and returns NaN if the run/subject is excluded from eye-tracking analysis.
    """
    empty_res = {
        'eye_valid': False,
        'pupil_diam_mean_mm': np.nan,
        'pupil_diam_std_mm': np.nan,
        'pupil_diam_skew': np.nan,
        'pupil_diam_kurt': np.nan,
        'gaze_entropy_x': np.nan,
        'gaze_entropy_y': np.nan,
        'gaze_valid_sample_fraction': 0.0
    }
    
    # Check if this subject/run is excluded from eye-tracking analysis
    if is_eye_excluded(subject, run_name):
        return empty_res
        
    if df_eye is None or len(df_eye) < 100:
        return empty_res

    # Check for validity columns
    val_l = df_eye['validity_l'].values if 'validity_l' in df_eye.columns else np.ones(len(df_eye))
    val_r = df_eye['validity_r'].values if 'validity_r' in df_eye.columns else np.ones(len(df_eye))
    
    # Step 6: Drop samples where validity_l == 0 AND validity_r == 0
    valid_samples = (val_l > 0) | (val_r > 0)
    valid_fraction = float(np.mean(valid_samples))
    
    if valid_fraction < 0.10 or np.sum(valid_samples) < 100:
        return empty_res
        
    df_valid = df_eye[valid_samples].copy()
    
    # Extract pupil diameters (combine Left & Right when valid)
    pupil_l = df_valid['pupil_diameter_l_mm'].values if 'pupil_diameter_l_mm' in df_valid.columns else np.array([])
    pupil_r = df_valid['pupil_diameter_r_mm'].values if 'pupil_diameter_r_mm' in df_valid.columns else np.array([])
    
    pupil_combined = []
    if len(pupil_l) > 0 and len(pupil_r) > 0:
        # Physiological human pupil range: 1.5 mm to 8.5 mm
        p_l_clean = np.where((pupil_l >= 1.5) & (pupil_l <= 8.5), pupil_l, np.nan)
        p_r_clean = np.where((pupil_r >= 1.5) & (pupil_r <= 8.5), pupil_r, np.nan)
        
        # Average both eyes where available
        with np.errstate(invalid='ignore'):
            pupil_combined = np.nanmean([p_l_clean, p_r_clean], axis=0)
        pupil_combined = pupil_combined[np.isfinite(pupil_combined)]
    elif len(pupil_l) > 0:
        pupil_combined = pupil_l[(pupil_l >= 1.5) & (pupil_l <= 8.5)]
    elif len(pupil_r) > 0:
        pupil_combined = pupil_r[(pupil_r >= 1.5) & (pupil_r <= 8.5)]
        
    if len(pupil_combined) < 50:
        return empty_res
        
    # Pupillometry summary metrics
    p_mean = float(np.mean(pupil_combined))
    p_std = float(np.std(pupil_combined))
    p_skew = float(skew(pupil_combined)) if len(pupil_combined) > 10 else np.nan
    p_kurt = float(kurtosis(pupil_combined)) if len(pupil_combined) > 10 else np.nan
    
    # Gaze Spatial Entropy
    entropy_x, entropy_y = np.nan, np.nan
    if 'gaze_direction_l_x' in df_valid.columns and 'gaze_direction_l_y' in df_valid.columns:
        gx = df_valid['gaze_direction_l_x'].dropna().values
        gy = df_valid['gaze_direction_l_y'].dropna().values
        
        if len(gx) > 50:
            hist_x, _ = np.histogram(gx, bins=20, density=True)
            hist_y, _ = np.histogram(gy, bins=20, density=True)
            
            p_x = hist_x[hist_x > 0]
            p_y = hist_y[hist_y > 0]
            
            entropy_x = float(entropy(p_x))
            entropy_y = float(entropy(p_y))

    return {
        'eye_valid': True,
        'pupil_diam_mean_mm': p_mean,
        'pupil_diam_std_mm': p_std,
        'pupil_diam_skew': p_skew,
        'pupil_diam_kurt': p_kurt,
        'gaze_entropy_x': entropy_x,
        'gaze_entropy_y': entropy_y,
        'gaze_valid_sample_fraction': valid_fraction
    }
