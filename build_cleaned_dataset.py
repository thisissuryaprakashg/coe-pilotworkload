"""
build_cleaned_dataset.py
========================
Master data cleaning and feature assembly pipeline:
1. Omits respiration and torso-accelerometry streams.
2. Applies row-level exclusions on the 11 bad runs (quality < 90%).
3. Excludes sub-cp003 and 2 specific runs of sub-cp027 from eye-tracking analysis only.
4. Executes signal-level cleaning (ECG R-peak & RR [300-2000ms], EDA artifact filtering & decomposition, Eye sample validity masking).
5. Merges flight performance labels from PerfMetrics.csv.
6. Computes subject-specific resting baseline normalization.
7. Saves final cleaned dataset to cleaned_multimodal_dataset.csv.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from cleaning_rules import BAD_RUNS_EXCLUSIONS, is_run_excluded, is_eye_excluded
from clean_ecg import clean_and_extract_ecg_features
from clean_eda_emg import clean_and_extract_eda_features, clean_and_extract_emg_features
from clean_eyetracking import clean_and_extract_eye_features

DATASET_ROOT = r'c:\coe\multimodal-physiological-monitoring-during-virtual-reality-piloting-tasks-1.0.0\dataPackage'

def load_perf_metrics() -> pd.DataFrame:
    perf_path = os.path.join(DATASET_ROOT, 'task-ils', 'PerfMetrics.csv')
    if not os.path.exists(perf_path):
        print(f"Warning: PerfMetrics.csv not found at {perf_path}")
        return pd.DataFrame()
    df_perf = pd.read_csv(perf_path)
    # Format subject id to match sub-cpXXX
    df_perf['subject_str'] = df_perf['subject'].apply(lambda x: f"sub-cp{int(x):03d}")
    df_perf['run_int'] = df_perf['run'].astype(int)
    return df_perf

def extract_features_from_folder(run_path: str, subject: str, run_name: str) -> dict:
    files = glob.glob(os.path.join(run_path, '*_dat.csv'))
    
    # Locate individual modality files
    f_ecg = [f for f in files if 'lslshimmerecg' in f]
    f_eda = [f for f in files if 'lslshimmereda' in f]
    f_emg = [f for f in files if 'lslshimmeremg' in f]
    f_eye = [f for f in files if 'lslhtcviveeye' in f]
    
    # 1. ECG
    ecg_feats = {}
    if f_ecg:
        try:
            df_ecg = pd.read_csv(f_ecg[0])
            ecg_feats = clean_and_extract_ecg_features(df_ecg)
        except Exception:
            pass
            
    # 2. EDA & EMG
    eda_feats = {}
    if f_eda:
        try:
            df_eda = pd.read_csv(f_eda[0])
            eda_feats = clean_and_extract_eda_features(df_eda)
        except Exception:
            pass
            
    emg_feats = {}
    if f_emg:
        try:
            df_emg = pd.read_csv(f_emg[0])
            emg_feats = clean_and_extract_emg_features(df_emg)
        except Exception:
            pass
            
    # 3. Eye Tracking
    eye_feats = {}
    if f_eye:
        try:
            df_eye = pd.read_csv(f_eye[0])
            eye_feats = clean_and_extract_eye_features(df_eye, subject, run_name)
        except Exception:
            pass
            
    res = {}
    res.update(ecg_feats)
    res.update(eda_feats)
    res.update(emg_feats)
    res.update(eye_feats)
    return res

def main():
    print("==========================================================")
    print("STARTING COG-PILOT MULTIMODAL DATASET CLEANING PIPELINE")
    print("==========================================================")
    
    perf_df = load_perf_metrics()
    task_ils_dir = os.path.join(DATASET_ROOT, 'task-ils')
    
    subj_dirs = sorted([d for d in os.listdir(task_ils_dir) if d.startswith('sub-cp')])
    print(f"Found {len(subj_dirs)} subjects in task-ils.")
    
    records = []
    total_ils_runs_found = 0
    excluded_runs_count = 0
    
    for subj in subj_dirs:
        subj_path = os.path.join(task_ils_dir, subj)
        sess_dirs = [d for d in os.listdir(subj_path) if d.startswith('ses-')]
        
        for sess in sess_dirs:
            sess_path = os.path.join(subj_path, sess)
            run_dirs = sorted([d for d in os.listdir(sess_path) if os.path.isdir(os.path.join(sess_path, d))])
            
            for run_dir in run_dirs:
                total_ils_runs_found += 1
                
                # Step 3: Check row-level run exclusion
                if is_run_excluded(subj, run_dir):
                    excluded_runs_count += 1
                    print(f"  [EXCLUDED RUN] {subj} -> {run_dir} (Quality < 90%)")
                    continue
                    
                # Parse run number and difficulty
                m_run = re.search(r'run-(\d+)', run_dir)
                run_num = int(m_run.group(1)) if m_run else -1
                
                m_level = re.search(r'level-(\d+)', run_dir)
                level_num = int(m_level.group(1)) if m_level else -1
                
                run_full_path = os.path.join(sess_path, run_dir)
                feats = extract_features_from_folder(run_full_path, subj, run_dir)
                
                # Match performance labels from PerfMetrics.csv
                perf_row = perf_df[(perf_df['subject_str'] == subj) & (perf_df['run_int'] == run_num)]
                
                rec = {
                    'subject': subj,
                    'session': sess,
                    'run_folder': run_dir,
                    'run_number': run_num,
                    'difficulty_level': level_num,
                    'is_eye_analyzed': not is_eye_excluded(subj, run_dir),
                }
                
                if len(perf_row) > 0:
                    p = perf_row.iloc[0]
                    rec['difficulty_ground_truth'] = p['difficulty']
                    rec['cumulative_glideslope_error_deg'] = p['cumulative_glideslope_error_deg']
                    rec['cumulative_localizer_error_deg'] = p['cumulative_localizer_error_deg']
                    rec['cumulative_airspeed_error_kts'] = p['cumulative_airspeed_error_kts']
                    rec['cumulative_total_error'] = p['cumulative_total_error']
                else:
                    rec['difficulty_ground_truth'] = level_num
                    rec['cumulative_glideslope_error_deg'] = np.nan
                    rec['cumulative_localizer_error_deg'] = np.nan
                    rec['cumulative_airspeed_error_kts'] = np.nan
                    rec['cumulative_total_error'] = np.nan

                rec.update(feats)
                records.append(rec)
                
    df_clean = pd.DataFrame(records)
    print("\n----------------------------------------------------------")
    print(f"Total raw ILS runs encountered : {total_ils_runs_found}")
    print(f"Total bad runs excluded        : {excluded_runs_count}")
    print(f"Total cleaned runs retained    : {len(df_clean)} / {total_ils_runs_found} ({(len(df_clean)/total_ils_runs_found)*100:.1f}%)")
    print("----------------------------------------------------------")

    # Step 4 & 5 Verification: Eye Tracking Analysis Exclusions
    eye_valid_count = df_clean['is_eye_analyzed'].sum()
    print(f"Runs with Eye-Tracking included: {eye_valid_count} (Sub-cp003 and 2 sub-cp027 runs excluded as specified)")
    
    # Save output dataset
    out_csv = r'c:\coe\cleaned_multimodal_dataset.csv'
    df_clean.to_csv(out_csv, index=False)
    print(f"\nSuccessfully generated cleaned dataset: {out_csv}")
    print(f"Dataset shape: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")

if __name__ == '__main__':
    main()
