"""
clean_ecg.py
============
Processes raw ECG signals, performs R-peak detection, filters physiologically plausible
RR intervals [300ms - 2000ms], and extracts Heart Rate Variability (HRV) metrics.
Includes built-in Pan-Tompkins & SciPy filter fallbacks.
"""

import numpy as np
import pandas as pd
from scipy import signal



def detect_r_peaks_scipy(ecg_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Standard Pan-Tompkins QRS detector using pure SciPy:
    1. Bandpass filter (5 - 15 Hz)
    2. Derivative filter
    3. Squaring function
    4. Moving average integration window (~150ms)
    5. Peak detection with adaptive threshold
    """
    nyq = 0.5 * fs
    lowcut, highcut = 5.0, min(15.0, 0.45 * nyq)
    b_bp, a_bp = signal.butter(3, [lowcut / nyq, highcut / nyq], btype='band')
    filtered = signal.filtfilt(b_bp, a_bp, ecg_signal)
    
    # 2. Derivative
    diff_sig = np.gradient(filtered)
    
    # 3. Squaring
    squared = diff_sig ** 2
    
    # 4. Moving window integrator (150ms)
    win_len = int(max(3, 0.15 * fs))
    kernel = np.ones(win_len) / win_len
    integrated = np.convolve(squared, kernel, mode='same')
    
    # 5. Peak detection
    min_dist = int(0.30 * fs) # minimum 300ms between beats (max 200 bpm)
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    peaks, _ = signal.find_peaks(integrated, distance=min_dist, height=threshold)
    
    # Refine peaks to find the exact maximum in raw ECG within a +/- 50ms window
    refined_peaks = []
    win_search = int(0.05 * fs)
    for p in peaks:
        left = max(0, p - win_search)
        right = min(len(ecg_signal), p + win_search)
        if right > left:
            true_peak = left + np.argmax(ecg_signal[left:right])
            refined_peaks.append(true_peak)
            
    return np.array(refined_peaks, dtype=int)

def clean_and_extract_ecg_features(df_ecg: pd.DataFrame, default_fs: float = 512.0) -> dict:
    """
    Cleans raw ECG, detects R-peaks, removes implausible RR intervals (<300ms or >2000ms),
    and computes time and frequency domain HRV metrics.
    """
    empty_res = {
        'ecg_valid': False,
        'ecg_hr_mean': np.nan,
        'ecg_hr_std': np.nan,
        'ecg_rr_mean_ms': np.nan,
        'ecg_rr_sdnn_ms': np.nan,
        'ecg_rr_rmssd_ms': np.nan,
        'ecg_rr_pnn50': np.nan,
        'ecg_rr_lf_power': np.nan,
        'ecg_rr_hf_power': np.nan,
        'ecg_rr_lf_hf_ratio': np.nan,
        'ecg_num_clean_beats': 0
    }
    
    if df_ecg is None or len(df_ecg) < 1000:
        return empty_res
        
    # Determine sampling rate from timestamps if available
    fs = default_fs
    if 'time_dn' in df_ecg.columns and len(df_ecg) > 10:
        dt_days = np.median(np.diff(df_ecg['time_dn'].values[:1000]))
        dt_sec = dt_days * 86400.0
        if dt_sec > 0:
            est_fs = 1.0 / dt_sec
            if 50.0 <= est_fs <= 1000.0:
                fs = est_fs

    # Lead selection: Lead II (ll_ra) preferred, fallback to other channels
    ecg_col = None
    for cand in ['ecg_projection_ll_ra_mV', 'ecg_projection_la_ra_mV', 'ecg_projection_vx_rl_mV']:
        if cand in df_ecg.columns and not df_ecg[cand].isna().all():
            ecg_col = cand
            break
            
    if ecg_col is None:
        return empty_res

    raw_ecg = df_ecg[ecg_col].values.astype(float)
    
    # Remove NaNs / Infs
    valid_mask = np.isfinite(raw_ecg)
    if np.sum(valid_mask) < 1000:
        return empty_res
    raw_ecg = np.interp(np.arange(len(raw_ecg)), np.where(valid_mask)[0], raw_ecg[valid_mask])

    try:
        # Detect R-peaks using Pan-Tompkins QRS algorithm
        r_peaks = detect_r_peaks_scipy(raw_ecg, fs)
        
        if len(r_peaks) < 10:
            return empty_res
            
        # 3. Calculate RR intervals in milliseconds
        rr_intervals_ms = np.diff(r_peaks) / fs * 1000.0
        
        # 4. Filter physiologically plausible RR intervals [300ms, 2000ms] (30 - 200 bpm)
        valid_rr_mask = (rr_intervals_ms >= 300.0) & (rr_intervals_ms <= 2000.0)
        clean_rr = rr_intervals_ms[valid_rr_mask]
        
        if len(clean_rr) < 10:
            return empty_res
            
        # Compute instantaneous Heart Rate in BPM
        hr_bpm = 60000.0 / clean_rr
        
        # Time-Domain HRV features
        rr_mean = float(np.mean(clean_rr))
        rr_sdnn = float(np.std(clean_rr, ddof=1)) if len(clean_rr) > 1 else 0.0
        rr_diffs = np.diff(clean_rr)
        rr_rmssd = float(np.sqrt(np.mean(rr_diffs ** 2))) if len(rr_diffs) > 0 else 0.0
        rr_pnn50 = float(np.sum(np.abs(rr_diffs) > 50.0) / len(rr_diffs) * 100.0) if len(rr_diffs) > 0 else 0.0
        
        # Frequency-Domain HRV features via Welch PSD on 4 Hz interpolated tachogram
        lf_power, hf_power, lf_hf_ratio = np.nan, np.nan, np.nan
        if len(clean_rr) >= 25:
            try:
                cum_times_s = np.cumsum(clean_rr) / 1000.0
                total_t = cum_times_s[-1]
                t_uniform = np.arange(cum_times_s[0], total_t, 0.25) # 4 Hz
                if len(t_uniform) > 64:
                    rr_uniform = np.interp(t_uniform, cum_times_s, clean_rr)
                    rr_detrend = signal.detrend(rr_uniform)
                    
                    freqs, psd = signal.welch(rr_detrend, fs=4.0, nperseg=min(len(rr_detrend), 256))
                    
                    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
                    hf_band = (freqs > 0.15) & (freqs <= 0.40)
                    
                    df_freq = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                    lf_power = float(np.sum(psd[lf_band]) * df_freq)
                    hf_power = float(np.sum(psd[hf_band]) * df_freq)
                    
                    if hf_power > 1e-6:
                        lf_hf_ratio = float(lf_power / hf_power)
            except Exception:
                pass

        return {
            'ecg_valid': True,
            'ecg_hr_mean': float(np.mean(hr_bpm)),
            'ecg_hr_std': float(np.std(hr_bpm)),
            'ecg_rr_mean_ms': rr_mean,
            'ecg_rr_sdnn_ms': rr_sdnn,
            'ecg_rr_rmssd_ms': rr_rmssd,
            'ecg_rr_pnn50': rr_pnn50,
            'ecg_rr_lf_power': lf_power,
            'ecg_rr_hf_power': hf_power,
            'ecg_rr_lf_hf_ratio': lf_hf_ratio,
            'ecg_num_clean_beats': int(len(clean_rr))
        }
        
    except Exception:
        return empty_res
