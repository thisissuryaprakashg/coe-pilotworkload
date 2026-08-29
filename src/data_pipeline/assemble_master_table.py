"""
assemble_master_table.py
========================
Implements Sections 4.1, 5.3, and 6 of cogpilot_project_spec.md:
1. Extracts resting-state physiological baselines from task-rest (level-000_run-001).
2. Computes personal delta features:
   - delta_HR = task_run_HR - rest_HR
   - delta_HRV_SDNN = task_run_SDNN - rest_SDNN
   - delta_HRV_RMSSD = task_run_RMSSD - rest_RMSSD
   - delta_HRV_LF_HF = task_run_LF_HF - rest_LF_HF
   - delta_EDA_tonic = task_run_EDA_tonic - rest_EDA_tonic
3. Joins the official pre-computed oculomotor feature matrix (devSubjsFeatMat.csv).
4. Assembles and exports master_feature_matrix.csv matching the Section 6 Schema.
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from clean_ecg import clean_and_extract_ecg_features
from clean_eda_emg import clean_and_extract_eda_features, clean_and_extract_emg_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get(
    'COGPILOT_DATASET_ROOT',
    PROJECT_ROOT / 'data' / 'raw' / 'cogpilot'
))
REST_DIR = DATASET_ROOT / 'dataPackage' / 'task-rest'
OFFICIAL_EYE_CSV = DATASET_ROOT / 'starterCode' / 'data_feats' / 'devSubjsFeatMat.csv'
CLEANED_ILS_CSV = PROJECT_ROOT / 'data' / 'processed' / 'cleaned_multimodal_dataset.csv'
OUTPUT_MASTER_CSV = PROJECT_ROOT / 'data' / 'processed' / 'master_feature_matrix.csv'

def extract_rest_baselines() -> pd.DataFrame:
    """Extracts resting baseline features from level-000_run-001 for all subjects."""
    print("Extracting resting baseline features from task-rest (level-000_run-001)...")
    subj_dirs = sorted([d for d in os.listdir(REST_DIR) if d.startswith('sub-cp')])
    
    rest_records = []
    for subj in subj_dirs:
        subj_path = os.path.join(REST_DIR, subj)
        sess_dirs = [d for d in os.listdir(subj_path) if d.startswith('ses-')]
        
        for sess in sess_dirs:
            sess_path = os.path.join(subj_path, sess)
            run1_path = os.path.join(sess_path, 'level-000_run-001')
            
            if not os.path.exists(run1_path):
                continue
                
            files = glob.glob(os.path.join(run1_path, '*_dat.csv'))
            f_ecg = [f for f in files if 'lslshimmerecg' in f]
            f_eda = [f for f in files if 'lslshimmereda' in f]
            f_emg = [f for f in files if 'lslshimmeremg' in f]
            f_eye = [f for f in files if 'lslhtcviveeye' in f]
            
            ecg_res, eda_res, emg_res, eye_res = {}, {}, {}, {}
            if f_ecg:
                try:
                    df_ecg = pd.read_csv(f_ecg[0])
                    ecg_res = clean_and_extract_ecg_features(df_ecg)
                except Exception:
                    pass
                    
            if f_eda:
                try:
                    df_eda = pd.read_csv(f_eda[0])
                    eda_res = clean_and_extract_eda_features(df_eda)
                except Exception:
                    pass

            if f_emg:
                try:
                    df_emg = pd.read_csv(f_emg[0])
                    emg_res = clean_and_extract_emg_features(df_emg)
                except Exception:
                    pass

            if f_eye:
                try:
                    from clean_eyetracking import clean_and_extract_eye_features
                    df_eye = pd.read_csv(f_eye[0])
                    eye_res = clean_and_extract_eye_features(df_eye, subj, 'level-000_run-001')
                except Exception:
                    pass
                    
            rec = {
                'subject': subj,
                'session': sess,
                'rest_HR': ecg_res.get('ecg_hr_mean', np.nan),
                'rest_HRV_SDNN': ecg_res.get('ecg_rr_sdnn_ms', np.nan),
                'rest_HRV_RMSSD': ecg_res.get('ecg_rr_rmssd_ms', np.nan),
                'rest_HRV_LF_HF': ecg_res.get('ecg_rr_lf_hf_ratio', np.nan),
                'rest_EDA_tonic': eda_res.get('eda_tonic_mean_uS', np.nan),
                'rest_EDA_conductance': eda_res.get('eda_conductance_mean_uS', np.nan),
                'rest_EMG_flexor': emg_res.get('emg_flexor_rms_mV', np.nan),
                'rest_EMG_extensor': emg_res.get('emg_extensor_rms_mV', np.nan),
                'rest_pupil_diam': eye_res.get('pupil_diam_mean_mm', np.nan)
            }
            rest_records.append(rec)
            
    df_rest = pd.DataFrame(rest_records)
    print(f"Extracted resting baselines for {len(df_rest)} subjects.")
    return df_rest

def main():
    print("==================================================================")
    print("ASSEMBLING MASTER FEATURE MATRIX (SECTIONS 4.1, 5.3, 6)")
    print("==================================================================")
    
    # 1. Load cleaned ILS dataset (408 runs)
    if not os.path.exists(CLEANED_ILS_CSV):
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_ILS_CSV}. Run build_cleaned_dataset.py first.")
    df_ils = pd.read_csv(CLEANED_ILS_CSV)
    print(f"Loaded cleaned ILS dataset: {len(df_ils)} runs.")
    
    # 2. Extract and merge rest baselines
    df_rest = extract_rest_baselines()
    df_merged = df_ils.merge(df_rest, on=['subject', 'session'], how='left')
    
    # 3. Compute Section 5.3 Personal Delta Features
    print("Computing subject-specific delta features (Delta = Task - Rest)...")
    df_merged['delta_HR'] = df_merged['ecg_hr_mean'] - df_merged['rest_HR']
    df_merged['delta_HRV_SDNN'] = df_merged['ecg_rr_sdnn_ms'] - df_merged['rest_HRV_SDNN']
    df_merged['delta_HRV_RMSSD'] = df_merged['ecg_rr_rmssd_ms'] - df_merged['rest_HRV_RMSSD']
    df_merged['delta_HRV_LF_HF'] = df_merged['ecg_rr_lf_hf_ratio'] - df_merged['rest_HRV_LF_HF']
    df_merged['delta_EDA_tonic'] = df_merged['eda_tonic_mean_uS'] - df_merged['rest_EDA_tonic']
    df_merged['delta_EDA_conductance'] = df_merged['eda_conductance_mean_uS'] - df_merged['rest_EDA_conductance']
    df_merged['delta_EMG_flexor'] = df_merged['emg_flexor_rms_mV'] - df_merged['rest_EMG_flexor']
    df_merged['delta_EMG_extensor'] = df_merged['emg_extensor_rms_mV'] - df_merged['rest_EMG_extensor']
    df_merged['delta_pupil_diam'] = df_merged['pupil_diam_mean_mm'] - df_merged['rest_pupil_diam']
    
    # 4. Join official pre-computed oculomotor feature matrix (devSubjsFeatMat.csv)
    if os.path.exists(OFFICIAL_EYE_CSV):
        print(f"Joining official oculomotor features from {OFFICIAL_EYE_CSV}...")
        df_eye_official = pd.read_csv(OFFICIAL_EYE_CSV)
        
        # Clean unnamed column if present
        if 'Unnamed: 0' in df_eye_official.columns:
            df_eye_official = df_eye_official.drop(columns=['Unnamed: 0'])
        elif '' in df_eye_official.columns:
            df_eye_official = df_eye_official.drop(columns=[''])
            
        # Match join keys: Subject, Session, Run
        df_eye_official = df_eye_official.rename(columns={
            'Subject': 'subject',
            'Session': 'session',
            'Run': 'run_folder'
        })
        
        # Prefix eye columns to keep clean namespace if desired, or join directly
        df_merged = df_merged.merge(df_eye_official, on=['subject', 'session', 'run_folder'], how='left')
        print(f"Joined {len(df_eye_official.columns) - 3} official oculomotor features.")
    else:
        print(f"Warning: {OFFICIAL_EYE_CSV} not found.")

    # 5. Export master feature matrix
    OUTPUT_MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(OUTPUT_MASTER_CSV, index=False)
    print("------------------------------------------------------------------")
    print(f"SUCCESS: Generated Master Feature Matrix -> {OUTPUT_MASTER_CSV}")
    print(f"Master Matrix Dimensions: {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")
    print("------------------------------------------------------------------")

if __name__ == '__main__':
    main()
