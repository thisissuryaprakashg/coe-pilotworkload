"""
clean_eda_emg.py
================
Cleans Electrodermal Activity (EDA) and Electromyography (EMG) signals,
performing motion artifact reduction, tonic/phasic decomposition, and muscle activation metrics.
"""

import numpy as np
import pandas as pd
from scipy import signal

def clean_and_extract_eda_features(df_eda: pd.DataFrame, default_fs: float = 128.0) -> dict:
    """
    Cleans skin conductance, removes artifacts, decomposes into Tonic (SCL) and Phasic (SCR) signals.
    """
    empty_res = {
        'eda_valid': False,
        'eda_conductance_mean_uS': np.nan,
        'eda_conductance_std_uS': np.nan,
        'eda_tonic_mean_uS': np.nan,
        'eda_phasic_peak_rate_per_min': np.nan,
        'eda_phasic_energy': np.nan
    }
    
    if df_eda is None or len(df_eda) < 200:
        return empty_res
        
    eda_col = None
    for cand in ['eda_hand_l_kOhms', 'eda_conductance_uS']:
        if cand in df_eda.columns and not df_eda[cand].isna().all():
            eda_col = cand
            break
            
    if eda_col is None:
        return empty_res

    # Sampling rate estimation
    fs = default_fs
    if 'time_dn' in df_eda.columns and len(df_eda) > 10:
        dt_days = np.median(np.diff(df_eda['time_dn'].values[:500]))
        dt_sec = dt_days * 86400.0
        if dt_sec > 0:
            est_fs = 1.0 / dt_sec
            if 10.0 <= est_fs <= 2000.0:
                fs = est_fs

    raw_vals = df_eda[eda_col].values.astype(float)
    valid_mask = np.isfinite(raw_vals) & (raw_vals > 0.05) & (raw_vals < 2000.0)
    
    if np.sum(valid_mask) < 200:
        return empty_res
        
    raw_vals = np.interp(np.arange(len(raw_vals)), np.where(valid_mask)[0], raw_vals[valid_mask])
    
    # Convert kOhms to Conductance in microSiemens (uS = 1000 / kOhms)
    if 'kOhms' in eda_col:
        conductance_uS = 1000.0 / np.clip(raw_vals, 0.1, 1000.0)
    else:
        conductance_uS = raw_vals

    try:
        # 1. Low-pass filter (Butterworth 4th order, 3 Hz cutoff) to eliminate high frequency sensor noise
        nyq = 0.5 * fs
        cutoff = min(3.0, 0.45 * nyq)
        b, a = signal.butter(4, cutoff / nyq, btype='low')
        eda_clean = signal.filtfilt(b, a, conductance_uS)
        
        # 2. Tonic-Phasic separation via high-pass baseline estimation (cutoff 0.05 Hz)
        hp_cutoff = 0.05
        if hp_cutoff < nyq:
            b_hp, a_hp = signal.butter(2, hp_cutoff / nyq, btype='high')
            phasic = signal.filtfilt(b_hp, a_hp, eda_clean)
            tonic = eda_clean - phasic
        else:
            phasic = eda_clean - np.median(eda_clean)
            tonic = np.full_like(eda_clean, np.median(eda_clean))
            
        # Detect SCR peaks in phasic component
        phasic_pos = np.clip(phasic, 0, None)
        peaks, _ = signal.find_peaks(phasic_pos, height=0.01, distance=int(fs * 1.0))
        
        duration_min = (len(eda_clean) / fs) / 60.0
        peak_rate = len(peaks) / max(0.1, duration_min)
        phasic_energy = float(np.sum(phasic_pos ** 2) / len(phasic_pos))

        return {
            'eda_valid': True,
            'eda_conductance_mean_uS': float(np.mean(eda_clean)),
            'eda_conductance_std_uS': float(np.std(eda_clean)),
            'eda_tonic_mean_uS': float(np.mean(tonic)),
            'eda_phasic_peak_rate_per_min': float(peak_rate),
            'eda_phasic_energy': float(phasic_energy)
        }
    except Exception:
        return empty_res


def clean_and_extract_emg_features(df_emg: pd.DataFrame, default_fs: float = 512.0) -> dict:
    """
    Filters raw EMG signals (20-200 Hz bandpass, 60 Hz notch), and computes Root Mean Square (RMS) activation.
    """
    empty_res = {
        'emg_valid': False,
        'emg_flexor_rms_mV': np.nan,
        'emg_extensor_rms_mV': np.nan,
        'emg_forearm_motion_energy': np.nan
    }
    
    if df_emg is None or len(df_emg) < 500:
        return empty_res
        
    fs = default_fs
    if 'time_dn' in df_emg.columns and len(df_emg) > 10:
        dt_days = np.median(np.diff(df_emg['time_dn'].values[:500]))
        dt_sec = dt_days * 86400.0
        if dt_sec > 0:
            est_fs = 1.0 / dt_sec
            if 50.0 <= est_fs <= 2000.0:
                fs = est_fs
                
    nyq = 0.5 * fs
    flex_col = 'emg_wrist_flexor_mV'
    ext_col = 'emg_wrist_extensor_mV'
    
    try:
        # Bandpass filter (20 - 150 Hz)
        lowcut, highcut = 20.0, min(150.0, 0.45 * nyq)
        b_bp, a_bp = signal.butter(4, [lowcut / nyq, highcut / nyq], btype='band')
        
        flex_rms, ext_rms = np.nan, np.nan
        if flex_col in df_emg.columns and not df_emg[flex_col].isna().all():
            v_flex = df_emg[flex_col].fillna(0).values
            v_flex_filt = signal.filtfilt(b_bp, a_bp, v_flex)
            flex_rms = float(np.sqrt(np.mean(v_flex_filt ** 2)))
            
        if ext_col in df_emg.columns and not df_emg[ext_col].isna().all():
            v_ext = df_emg[ext_col].fillna(0).values
            v_ext_filt = signal.filtfilt(b_bp, a_bp, v_ext)
            ext_rms = float(np.sqrt(np.mean(v_ext_filt ** 2)))
            
        # Forearm accelerometry motion energy
        acc_energy = np.nan
        acc_cols = ['accelerometry_forearm_r_x_mps2', 'accelerometry_forearm_r_y_mps2', 'accelerometry_forearm_r_z_mps2']
        if all(c in df_emg.columns for c in acc_cols):
            acc_mag = np.sqrt(df_emg[acc_cols[0]]**2 + df_emg[acc_cols[1]]**2 + df_emg[acc_cols[2]]**2)
            acc_energy = float(np.std(acc_mag.dropna()))

        return {
            'emg_valid': True,
            'emg_flexor_rms_mV': flex_rms,
            'emg_extensor_rms_mV': ext_rms,
            'emg_forearm_motion_energy': acc_energy
        }
    except Exception:
        return empty_res
