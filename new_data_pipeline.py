"""
new_data_pipeline.py
====================
30-second windowed feature extraction from CogPilot physiological dataset.
Extracts features per window from: ECG, EDA, EMG, Eye, Head, Aircraft.
Computes resting baseline deltas. Joins pre-computed oculomotor features.
Output: new_master_features.csv
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.stats import skew, kurtosis, entropy
from pathlib import Path

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"c:\coe_new\multimodal-physiological-monitoring-during-virtual-reality-piloting-tasks-1.0.0\dataPackage"
)
OCULO_CSV = Path(
    r"c:\coe_new\multimodal-physiological-monitoring-during-virtual-reality-piloting-tasks-1.0.0"
    r"\starterCode\data_feats\devSubjsFeatMat.csv"
)
OUTPUT_CSV = Path(r"c:\coe_new\new_master_features.csv")

WINDOW_SEC = 30.0
WINDOW_DN = WINDOW_SEC / 86400.0  # 30 sec in MATLAB datenum (days)

# ============================================================
# CLEANING RULES
# ============================================================

BAD_RUNS = {
    ('sub-cp009', 'level-01B_run-001'), ('sub-cp009', 'level-02B_run-003'),
    ('sub-cp009', 'level-03B_run-002'), ('sub-cp009', 'level-04B_run-004'),
    ('sub-cp026', 'level-01B_run-001'), ('sub-cp026', 'level-02B_run-008'),
    ('sub-cp026', 'level-03B_run-002'), ('sub-cp026', 'level-03B_run-005'),
    ('sub-cp026', 'level-04B_run-009'),
    ('sub-cp027', 'level-02B_run-010'),
    ('sub-cp028', 'level-01B_run-012'),
}

EYE_EXCLUDED_SUBJECTS = {'sub-cp003'}

EYE_EXCLUDED_RUNS = {
    ('sub-cp027', 'level-01B_run-001'),
    ('sub-cp027', 'level-03B_run-002'),
}


def is_run_bad(subj, run):
    return (subj, run) in BAD_RUNS


def is_eye_bad(subj, run):
    return subj in EYE_EXCLUDED_SUBJECTS or (subj, run) in EYE_EXCLUDED_RUNS


# ============================================================
# SIGNAL UTILITIES
# ============================================================

def estimate_fs(time_dn, n=1000):
    if len(time_dn) < 10:
        return None
    dt = np.median(np.diff(time_dn[:min(n, len(time_dn))]))
    if dt <= 0:
        return None
    fs = 1.0 / (dt * 86400.0)
    return fs if 1.0 <= fs <= 5000.0 else None


def interp_nans(arr):
    valid = np.isfinite(arr)
    if np.sum(valid) < 10:
        return None
    return np.interp(np.arange(len(arr)), np.where(valid)[0], arr[valid])


def detect_r_peaks(ecg, fs):
    nyq = 0.5 * fs
    lo, hi = 5.0, min(15.0, 0.45 * nyq)
    b, a = sig.butter(3, [lo / nyq, hi / nyq], btype='band')
    filt = sig.filtfilt(b, a, ecg)
    sq = np.gradient(filt) ** 2
    win = int(max(3, 0.15 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode='same')

    peaks, _ = sig.find_peaks(
        integ, distance=int(0.3 * fs),
        height=np.mean(integ) + 0.5 * np.std(integ)
    )

    w = int(0.05 * fs)
    refined = []
    for p in peaks:
        lo_i, hi_i = max(0, p - w), min(len(ecg), p + w)
        if hi_i > lo_i:
            refined.append(lo_i + np.argmax(ecg[lo_i:hi_i]))
    return np.array(refined, dtype=int)


def sample_entropy(data, m=2, r_factor=0.2):
    N = len(data)
    if N < m + 3:
        return np.nan
    r = r_factor * np.std(data, ddof=1)
    if r < 1e-10:
        return np.nan

    def phi(dim):
        tpl = np.array([data[i:i + dim] for i in range(N - dim)])
        cnt = 0
        for i in range(len(tpl)):
            d = np.max(np.abs(tpl[i] - tpl[i + 1:]), axis=1)
            cnt += np.sum(d < r)
        return cnt

    B, A = phi(m), phi(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))


# ============================================================
# PER-WINDOW FEATURE EXTRACTORS
# ============================================================

def ecg_window(df, w0, w1, fs):
    P = 'ecg_'
    empty = {
        f'{P}hr_mean': np.nan, f'{P}hr_std': np.nan, f'{P}hr_range': np.nan,
        f'{P}rmssd': np.nan, f'{P}sdnn': np.nan, f'{P}pnn50': np.nan,
        f'{P}rr_cv': np.nan, f'{P}sampen': np.nan,
    }
    if df is None:
        return empty

    t = df['time_dn'].values
    mask = (t >= w0) & (t < w1)
    if np.sum(mask) < 500:
        return empty

    col = None
    for c in ['ecg_projection_ll_ra_mV', 'ecg_projection_la_ra_mV',
              'ecg_projection_vx_rl_mV']:
        if c in df.columns:
            col = c
            break
    if col is None:
        return empty

    raw = df[col].values[mask].astype(float)
    raw = interp_nans(raw)
    if raw is None:
        return empty

    try:
        peaks = detect_r_peaks(raw, fs)
    except Exception:
        return empty
    if len(peaks) < 5:
        return empty

    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr >= 300) & (rr <= 2000)]
    if len(rr) < 5:
        return empty

    hr = 60000.0 / rr
    rd = np.diff(rr)

    return {
        f'{P}hr_mean': float(np.mean(hr)),
        f'{P}hr_std': float(np.std(hr)),
        f'{P}hr_range': float(np.ptp(hr)),
        f'{P}rmssd': float(np.sqrt(np.mean(rd ** 2))) if len(rd) > 0 else np.nan,
        f'{P}sdnn': float(np.std(rr, ddof=1)),
        f'{P}pnn50': float(np.sum(np.abs(rd) > 50) / len(rd) * 100) if len(rd) > 0 else np.nan,
        f'{P}rr_cv': float(np.std(rr) / np.mean(rr)),
        f'{P}sampen': sample_entropy(rr),
    }


def eda_window(df, w0, w1, fs):
    P = 'eda_'
    empty = {
        f'{P}scl_mean': np.nan, f'{P}scl_slope': np.nan, f'{P}cond_std': np.nan,
        f'{P}scr_count': 0, f'{P}scr_amp_mean': np.nan, f'{P}phasic_energy': np.nan,
    }
    if df is None:
        return empty

    t = df['time_dn'].values
    mask = (t >= w0) & (t < w1)
    if np.sum(mask) < 100:
        return empty

    col = None
    for c in ['eda_hand_l_kOhms', 'eda_conductance_uS']:
        if c in df.columns:
            col = c
            break
    if col is None:
        return empty

    raw = df[col].values[mask].astype(float)
    valid = np.isfinite(raw) & (raw > 0.05) & (raw < 2000)
    if np.sum(valid) < 100:
        return empty
    raw = np.interp(np.arange(len(raw)), np.where(valid)[0], raw[valid])

    cond = 1000.0 / np.clip(raw, 0.1, 1000) if 'kOhms' in col else raw

    try:
        nyq = 0.5 * fs
        b, a = sig.butter(4, min(3.0, 0.45 * nyq) / nyq, btype='low')
        clean = sig.filtfilt(b, a, cond)
    except Exception:
        clean = cond

    scl = float(np.mean(clean))
    try:
        slope = float(np.polyfit(np.arange(len(clean)), clean, 1)[0] * fs)
    except Exception:
        slope = np.nan

    deriv = np.diff(clean) * fs
    dp = np.clip(deriv, 0, None)
    try:
        pks, props = sig.find_peaks(dp, height=0.01, distance=int(max(1, fs)))
        n_scr = len(pks)
        amp = float(np.mean(props['peak_heights'])) if n_scr > 0 else np.nan
    except Exception:
        n_scr, amp = 0, np.nan

    return {
        f'{P}scl_mean': scl,
        f'{P}scl_slope': slope,
        f'{P}cond_std': float(np.std(clean)),
        f'{P}scr_count': n_scr,
        f'{P}scr_amp_mean': amp,
        f'{P}phasic_energy': float(np.sum(dp ** 2) / max(1, len(dp))),
    }


def emg_window(df, w0, w1, fs):
    P = 'emg_'
    empty = {
        f'{P}flexor_rms': np.nan, f'{P}extensor_rms': np.nan,
        f'{P}cocontraction': np.nan, f'{P}acc_rms': np.nan,
    }
    if df is None:
        return empty

    t = df['time_dn'].values
    mask = (t >= w0) & (t < w1)
    if np.sum(mask) < 300:
        return empty

    nyq = 0.5 * fs
    lo, hi = 20.0, min(150.0, 0.45 * nyq)
    try:
        b, a = sig.butter(4, [lo / nyq, hi / nyq], btype='band')
    except Exception:
        return empty

    flex, ext = np.nan, np.nan

    if 'emg_wrist_flexor_mV' in df.columns:
        v = np.nan_to_num(df['emg_wrist_flexor_mV'].values[mask].astype(float))
        try:
            flex = float(np.sqrt(np.mean(sig.filtfilt(b, a, v) ** 2)))
        except Exception:
            pass

    if 'emg_wrist_extensor_mV' in df.columns:
        v = np.nan_to_num(df['emg_wrist_extensor_mV'].values[mask].astype(float))
        try:
            ext = float(np.sqrt(np.mean(sig.filtfilt(b, a, v) ** 2)))
        except Exception:
            pass

    cocon = float(flex * ext) if np.isfinite(flex) and np.isfinite(ext) else np.nan

    acc = np.nan
    ac = ['accelerometry_forearm_r_x_mps2', 'accelerometry_forearm_r_y_mps2',
          'accelerometry_forearm_r_z_mps2']
    if all(c in df.columns for c in ac):
        try:
            mag = np.sqrt(sum(df[c].values[mask].astype(float) ** 2 for c in ac))
            mag = mag[np.isfinite(mag)]
            acc = float(np.std(mag)) if len(mag) > 10 else np.nan
        except Exception:
            pass

    return {
        f'{P}flexor_rms': flex, f'{P}extensor_rms': ext,
        f'{P}cocontraction': cocon, f'{P}acc_rms': acc,
    }


def eye_window(df, w0, w1, excluded):
    P = 'eye_'
    empty = {
        f'{P}pupil_mean': np.nan, f'{P}pupil_std': np.nan,
        f'{P}pupil_skew': np.nan, f'{P}pupil_kurt': np.nan,
        f'{P}pupil_velocity': np.nan, f'{P}perclos': np.nan,
        f'{P}blink_rate': np.nan,
        f'{P}gaze_ent_x': np.nan, f'{P}gaze_ent_y': np.nan,
        f'{P}conv_mean': np.nan, f'{P}conv_std': np.nan,
    }
    if df is None or excluded:
        return empty

    t = df['time_dn'].values
    mask = (t >= w0) & (t < w1)
    idx = np.where(mask)[0]
    if len(idx) < 50:
        return empty

    # validity filter
    vl = df['validity_l'].values[idx] if 'validity_l' in df.columns else np.ones(len(idx))
    vr = df['validity_r'].values[idx] if 'validity_r' in df.columns else np.ones(len(idx))
    valid = (vl > 0) | (vr > 0)
    vidx = idx[valid]
    if len(vidx) < 30:
        return empty

    # pupil diameter (average L+R, bounds 1.5-8.5 mm)
    pl = df['pupil_diameter_l_mm'].values[vidx].astype(float) if 'pupil_diameter_l_mm' in df.columns else np.full(len(vidx), np.nan)
    pr = df['pupil_diameter_r_mm'].values[vidx].astype(float) if 'pupil_diameter_r_mm' in df.columns else np.full(len(vidx), np.nan)
    pl = np.where((pl >= 1.5) & (pl <= 8.5), pl, np.nan)
    pr = np.where((pr >= 1.5) & (pr <= 8.5), pr, np.nan)
    with np.errstate(invalid='ignore'):
        pupil = np.nanmean([pl, pr], axis=0)
    pupil = pupil[np.isfinite(pupil)]

    if len(pupil) < 20:
        return empty

    pmean = float(np.mean(pupil))
    pstd = float(np.std(pupil))
    pskew = float(skew(pupil)) if len(pupil) > 10 else np.nan
    pkurt = float(kurtosis(pupil)) if len(pupil) > 10 else np.nan
    pvel = float(np.mean(np.abs(np.diff(pupil)))) if len(pupil) > 5 else np.nan

    # PERCLOS (% time eyes < 20% open)
    perclos = np.nan
    if 'eye_openness_l' in df.columns:
        eo = df['eye_openness_l'].values[idx]
        eo_clean = eo[np.isfinite(eo)]
        if len(eo_clean) > 20:
            perclos = float(np.mean(eo_clean < 0.2) * 100)

    # blink rate (open->closed transitions per minute)
    br = np.nan
    if 'eye_openness_l' in df.columns:
        eo = np.nan_to_num(df['eye_openness_l'].values[idx], nan=1.0)
        closed = (eo < 0.3).astype(int)
        n_blinks = np.sum(np.diff(closed) == 1)
        br = float(n_blinks / (WINDOW_SEC / 60.0))

    # gaze entropy (histogram-based spatial distribution)
    gex, gey = np.nan, np.nan
    if 'gaze_direction_l_x' in df.columns:
        gx = df['gaze_direction_l_x'].values[vidx]
        gx = gx[np.isfinite(gx)]
        if len(gx) > 30:
            h, _ = np.histogram(gx, bins=20)
            h = h[h > 0]
            gex = float(entropy(h))
    if 'gaze_direction_l_y' in df.columns:
        gy = df['gaze_direction_l_y'].values[vidx]
        gy = gy[np.isfinite(gy)]
        if len(gy) > 30:
            h, _ = np.histogram(gy, bins=20)
            h = h[h > 0]
            gey = float(entropy(h))

    # convergence distance
    cm, cs = np.nan, np.nan
    if 'convergence_distance_mm' in df.columns:
        cd = df['convergence_distance_mm'].values[vidx].astype(float)
        cd = cd[np.isfinite(cd) & (cd > 0) & (cd < 10000)]
        if len(cd) > 10:
            cm, cs = float(np.mean(cd)), float(np.std(cd))

    return {
        f'{P}pupil_mean': pmean, f'{P}pupil_std': pstd,
        f'{P}pupil_skew': pskew, f'{P}pupil_kurt': pkurt,
        f'{P}pupil_velocity': pvel, f'{P}perclos': perclos,
        f'{P}blink_rate': br,
        f'{P}gaze_ent_x': gex, f'{P}gaze_ent_y': gey,
        f'{P}conv_mean': cm, f'{P}conv_std': cs,
    }


def head_window(df, w0, w1):
    P = 'head_'
    empty = {f'{P}vel_rms': np.nan, f'{P}yaw_std': np.nan, f'{P}pitch_std': np.nan}
    if df is None:
        return empty

    t = df['time_dn'].values
    mask = (t >= w0) & (t < w1)
    idx = np.where(mask)[0]
    if len(idx) < 10:
        return empty

    vel = np.nan
    pos_cols = ['pilot_head_position_f_m', 'pilot_head_position_h_m',
                'pilot_head_position_v_m']
    if all(c in df.columns for c in pos_cols):
        try:
            dp = np.sqrt(sum(
                np.diff(df[c].values[idx].astype(float)) ** 2 for c in pos_cols
            ))
            dp = dp[np.isfinite(dp)]
            vel = float(np.sqrt(np.mean(dp ** 2))) if len(dp) > 3 else np.nan
        except Exception:
            pass

    def col_std(col_name):
        if col_name not in df.columns:
            return np.nan
        v = df[col_name].values[idx].astype(float)
        v = v[np.isfinite(v)]
        return float(np.std(v)) if len(v) > 3 else np.nan

    return {
        f'{P}vel_rms': vel,
        f'{P}yaw_std': col_std('pilot_head_yaw_deg'),
        f'{P}pitch_std': col_std('pilot_head_pitch_deg'),
    }


def aircraft_window(df, w0, w1):
    """Aircraft state features. No ILS deflections (ground-truth contamination)."""
    P = 'ac_'
    empty = {
        f'{P}airspeed_std': np.nan, f'{P}climbrate_std': np.nan,
        f'{P}pitch_std': np.nan, f'{P}roll_std': np.nan, f'{P}agl_mean': np.nan,
    }
    if df is None:
        return empty

    t = df['time_dn'].values
    mask = (t >= w0) & (t < w1)
    idx = np.where(mask)[0]
    if len(idx) < 5:
        return empty

    def col_stat(col_name, func):
        if col_name not in df.columns:
            return np.nan
        v = df[col_name].values[idx].astype(float)
        v = v[np.isfinite(v)]
        return float(func(v)) if len(v) > 3 else np.nan

    return {
        f'{P}airspeed_std': col_stat('aircraft_indicated_airspeed_kias', np.std),
        f'{P}climbrate_std': col_stat('aircraft_climb_rate_mps', np.std),
        f'{P}pitch_std': col_stat('aircraft_pitch_deg', np.std),
        f'{P}roll_std': col_stat('aircraft_roll_deg', np.std),
        f'{P}agl_mean': col_stat('aircraft_agl_altitude_m', np.mean),
    }


# ============================================================
# STREAM LOADING
# ============================================================

def load_stream(run_path, keyword, default_fs, fs_lo, fs_hi):
    files = glob.glob(os.path.join(run_path, f'*{keyword}*_dat.csv'))
    if not files:
        return None, default_fs
    try:
        df = pd.read_csv(files[0])
        if len(df) < 10 or 'time_dn' not in df.columns:
            return None, default_fs
        fs = estimate_fs(df['time_dn'].values)
        if fs is None or not (fs_lo <= fs <= fs_hi):
            fs = default_fs
        return df, fs
    except Exception as e:
        print(f"      [WARN] {keyword}: {e}")
        return None, default_fs


def load_run_streams(run_path):
    s = {}
    s['ecg'], s['ecg_fs'] = load_stream(run_path, 'lslshimmerecg', 512, 50, 1000)
    s['eda'], s['eda_fs'] = load_stream(run_path, 'lslshimmereda', 128, 10, 2000)
    s['emg'], s['emg_fs'] = load_stream(run_path, 'lslshimmeremg', 512, 50, 2000)
    s['eye'], s['eye_fs'] = load_stream(run_path, 'lslhtcviveeye', 120, 30, 300)
    s['aircraft'], s['ac_fs'] = load_stream(run_path, 'lslxp11xpcac', 10, 1, 100)
    s['head'], s['head_fs'] = load_stream(run_path, 'lslxp11xpcplt', 10, 1, 100)
    return s


def get_ref_time(streams):
    for k in ['ecg', 'eye', 'eda', 'emg', 'aircraft', 'head']:
        df = streams.get(k)
        if df is not None and 'time_dn' in df.columns and len(df) > 50:
            return df['time_dn'].values
    return None


# ============================================================
# REST BASELINES
# ============================================================

def compute_rest_baselines():
    rest_dir = DATASET_ROOT / 'task-rest'
    if not rest_dir.exists():
        print("  [WARN] task-rest not found")
        return {}

    subjects = sorted([d for d in os.listdir(rest_dir) if d.startswith('sub-cp')])
    baselines = {}

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

            w0, w1 = ref[0], ref[-1]

            ecg = ecg_window(s.get('ecg'), w0, w1, s.get('ecg_fs', 512))
            eda = eda_window(s.get('eda'), w0, w1, s.get('eda_fs', 128))
            emg = emg_window(s.get('emg'), w0, w1, s.get('emg_fs', 512))
            eye = eye_window(s.get('eye'), w0, w1, subj in EYE_EXCLUDED_SUBJECTS)

            baselines[subj] = {
                'rest_hr': ecg['ecg_hr_mean'],
                'rest_rmssd': ecg['ecg_rmssd'],
                'rest_sdnn': ecg['ecg_sdnn'],
                'rest_scl': eda['eda_scl_mean'],
                'rest_flexor': emg['emg_flexor_rms'],
                'rest_extensor': emg['emg_extensor_rms'],
                'rest_pupil': eye['eye_pupil_mean'],
            }

    print(f"  rest baselines: {len(baselines)} subjects")
    return baselines


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("30-SEC WINDOWED FEATURE EXTRACTION")
    print("=" * 60)

    # PerfMetrics for ground truth labels
    perf_path = DATASET_ROOT / 'task-ils' / 'PerfMetrics.csv'
    perf_df = pd.read_csv(perf_path)
    perf_df['_subj'] = perf_df['subject'].apply(lambda x: f"sub-cp{int(x):03d}")
    perf_df['_run'] = perf_df['run'].astype(int)
    print(f"PerfMetrics: {len(perf_df)} rows")

    # rest baselines
    print("\n--- REST BASELINES ---")
    baselines = compute_rest_baselines()

    # process ILS runs
    print("\n--- ILS RUNS ---")
    ils_dir = DATASET_ROOT / 'task-ils'
    subjects = sorted([d for d in os.listdir(ils_dir) if d.startswith('sub-cp')])
    print(f"subjects: {len(subjects)}")

    records = []
    n_runs = 0
    n_excluded = 0

    for si, subj in enumerate(subjects):
        subj_windows = 0
        subj_path = ils_dir / subj

        for sess in sorted(os.listdir(subj_path)):
            if not sess.startswith('ses-'):
                continue
            sess_path = subj_path / sess
            runs = sorted([
                d for d in os.listdir(sess_path)
                if (sess_path / d).is_dir()
            ])

            for run_dir in runs:
                n_runs += 1

                if is_run_bad(subj, run_dir):
                    n_excluded += 1
                    continue

                # parse difficulty + run number from folder name
                m = re.search(r'level-(\d+)', run_dir)
                difficulty = int(m.group(1)) if m else -1
                m = re.search(r'run-(\d+)', run_dir)
                run_num = int(m.group(1)) if m else -1

                # match PerfMetrics difficulty (overrides folder parse)
                pr = perf_df[(perf_df['_subj'] == subj) & (perf_df['_run'] == run_num)]
                if len(pr) > 0:
                    difficulty = int(pr.iloc[0]['difficulty'])

                eye_excl = is_eye_bad(subj, run_dir)

                # load all streams
                s = load_run_streams(str(sess_path / run_dir))
                ref = get_ref_time(s)
                if ref is None or len(ref) < 100:
                    continue

                # 30-sec window loop
                w0 = ref[0]
                t_end = ref[-1]
                wi = 0

                while w0 + WINDOW_DN * 0.5 < t_end:
                    w1 = w0 + WINDOW_DN

                    rec = {
                        'subject': subj, 'session': sess,
                        'run_folder': run_dir, 'run_number': run_num,
                        'difficulty': difficulty, 'window': wi,
                    }

                    rec.update(ecg_window(s.get('ecg'), w0, w1, s.get('ecg_fs', 512)))
                    rec.update(eda_window(s.get('eda'), w0, w1, s.get('eda_fs', 128)))
                    rec.update(emg_window(s.get('emg'), w0, w1, s.get('emg_fs', 512)))
                    rec.update(eye_window(s.get('eye'), w0, w1, eye_excl))
                    rec.update(head_window(s.get('head'), w0, w1))
                    rec.update(aircraft_window(s.get('aircraft'), w0, w1))

                    records.append(rec)
                    w0 = w1
                    wi += 1

                subj_windows += wi

        print(f"  [{si + 1:2d}/{len(subjects)}] {subj} | windows: {subj_windows}")

    df = pd.DataFrame(records)
    print(f"\ntotal runs: {n_runs} | excluded: {n_excluded} | retained: {n_runs - n_excluded}")
    print(f"total windows: {len(df)}")

    # delta features (task window - rest baseline)
    delta_map = [
        ('rest_hr',        'ecg_hr_mean',      'delta_hr'),
        ('rest_rmssd',     'ecg_rmssd',         'delta_rmssd'),
        ('rest_sdnn',      'ecg_sdnn',          'delta_sdnn'),
        ('rest_scl',       'eda_scl_mean',      'delta_scl'),
        ('rest_flexor',    'emg_flexor_rms',    'delta_flexor'),
        ('rest_extensor',  'emg_extensor_rms',  'delta_extensor'),
        ('rest_pupil',     'eye_pupil_mean',    'delta_pupil'),
    ]

    for _, _, dk in delta_map:
        df[dk] = np.nan

    for subj, bl in baselines.items():
        mask = df['subject'] == subj
        for rk, tk, dk in delta_map:
            rv = bl.get(rk, np.nan)
            if rv is not None and np.isfinite(rv):
                df.loc[mask, dk] = df.loc[mask, tk] - rv

    # join run-level oculomotor features from devSubjsFeatMat.csv
    if OCULO_CSV.exists():
        odf = pd.read_csv(OCULO_CSV)
        if 'Unnamed: 0' in odf.columns:
            odf = odf.drop(columns=['Unnamed: 0'])
        odf = odf.rename(columns={
            'Subject': 'subject', 'Session': 'session', 'Run': 'run_folder'
        })
        n_ocol = len(odf.columns) - 3
        df = df.merge(odf, on=['subject', 'session', 'run_folder'], how='left')
        print(f"oculomotor features joined: {n_ocol}")
    else:
        print("[WARN] devSubjsFeatMat.csv not found")

    # summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")

    print(f"\ndifficulty distribution:")
    for d in sorted(df['difficulty'].unique()):
        n = (df['difficulty'] == d).sum()
        print(f"  level {d}: {n} windows")

    meta_cols = {'subject', 'session', 'run_folder', 'run_number', 'difficulty', 'window'}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    print(f"\nfeature columns: {len(feat_cols)}")

    print(f"\nnon-NaN coverage per modality:")
    for prefix, label in [('ecg_', 'ECG'), ('eda_', 'EDA'), ('emg_', 'EMG'),
                          ('eye_', 'Eye'), ('head_', 'Head'), ('ac_', 'Aircraft'),
                          ('delta_', 'Delta')]:
        cols = [c for c in feat_cols if c.startswith(prefix)]
        if cols:
            pct = df[cols].notna().mean().mean() * 100
            print(f"  {label:10s}: {len(cols):3d} features | {pct:5.1f}% non-NaN")

    ocu_prefixes = ('ecg_', 'eda_', 'emg_', 'eye_', 'head_', 'ac_', 'delta_')
    ocu_cols = [c for c in feat_cols if not any(c.startswith(p) for p in ocu_prefixes)]
    if ocu_cols:
        pct = df[ocu_cols].notna().mean().mean() * 100
        print(f"  {'Oculomotor':10s}: {len(ocu_cols):3d} features | {pct:5.1f}% non-NaN")

    # save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsaved: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
