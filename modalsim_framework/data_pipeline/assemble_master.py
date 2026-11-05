"""
assemble_master.py
==================
Integrates all multi-modal streams for MODALSIM94:
  1. eSENSE_Pulse (HR, HRV metrics)
  2. eSENSE_Respiratory (Respiration rate & depth)
  3. Dikablis Eye Tracking (Pupillometry & gaze dispersion)
  4. Performance (Tunnel flight deviation errors)
  5. Questionnaires (NASA-TLX ground-truth mental demand)

Computes Subject-Relative Deltas (Delta = Task - Baseline_NBTask) with ZERO leakage.
Outputs saved to modalsim_framework/outputs/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

from modalsim_framework.data_pipeline.extract_pulse import extract_pulse_features
from modalsim_framework.data_pipeline.extract_respiratory import extract_respiratory_features
from modalsim_framework.data_pipeline.extract_eye import extract_eye_features
from modalsim_framework.data_pipeline.extract_performance import extract_performance_features

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = FRAMEWORK_ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = OUTPUT_DIR / 'modalsim_master_features.csv'
NORMALIZED_CSV = OUTPUT_DIR / 'modalsim_normalized_features.csv'

def assemble_modalsim_dataset(modalsim_root):
    print("=" * 70)
    print("MODALSIM94 MULTI-MODAL DATASET ASSEMBLY")
    print("=" * 70)

    # 1. Extract each modality
    df_pulse = extract_pulse_features(modalsim_root)
    df_resp = extract_respiratory_features(modalsim_root)
    df_eye = extract_eye_features(modalsim_root)
    df_perf = extract_performance_features(modalsim_root)

    # 2. Extract NASA-TLX & Demographics
    q_path = Path(modalsim_root) / 'QuestionnairesAndDemographic.csv'
    df_q = pd.read_csv(q_path, sep=';')
    
    # Map scenarios: 0: NBTask, 1: 1BTask, 2: 2BTask
    scen_map = {0: 'NBTask', 1: '1BTask', 2: '2BTask'}
    df_q['scenario'] = df_q['Scenario'].map(scen_map)
    df_q = df_q.dropna(subset=['scenario']).rename(columns={'PartID': 'subject'})

    tlx_cols = ['subject', 'scenario', 'MentalTLX', 'PhysicalTLX', 'TemporalTLX', 
                'PerformanceTLX', 'EffortTLX', 'FrustrationTLX', 'SART', 'KSS', 'SPS']
    avail_tlx = [c for c in tlx_cols if c in df_q.columns]
    df_tlx = df_q[avail_tlx].drop_duplicates(subset=['subject', 'scenario'])

    # 3. Outer merge to preserve all available participant data
    print("\n[Merging Modalities]")
    master = df_pulse.merge(df_resp, on=['subject', 'scenario'], how='outer')
    master = master.merge(df_eye, on=['subject', 'scenario'], how='outer')
    master = master.merge(df_perf, on=['subject', 'scenario'], how='outer')
    master = master.merge(df_tlx, on=['subject', 'scenario'], how='left')

    # Target integer mapping
    target_map = {'NBTask': 0, '1BTask': 1, '2BTask': 2}
    master['target'] = master['scenario'].map(target_map)
    master = master.dropna(subset=['target'])
    master['target'] = master['target'].astype(int)

    master.to_csv(MASTER_CSV, index=False)
    print(f"Master feature table saved to: {MASTER_CSV}")
    print(f"Shape: {master.shape[0]} rows x {master.shape[1]} columns ({master['subject'].nunique()} subjects)")

    # 4. Compute Subject-Relative Baseline Deltas (Zero Leakage)
    print("\n[Computing Subject-Relative Baseline Deltas]")
    meta_cols = {'subject', 'scenario', 'target', 'MentalTLX', 'PhysicalTLX', 'TemporalTLX',
                 'PerformanceTLX', 'EffortTLX', 'FrustrationTLX', 'SART', 'KSS', 'SPS'}
    sensor_features = [c for c in master.columns if c not in meta_cols]

    norm_df = master.copy()
    delta_cols = []

    for f in sensor_features:
        delta_col = f'delta_{f}'
        norm_df[delta_col] = np.nan
        delta_cols.append(delta_col)

    for subj, grp in norm_df.groupby('subject'):
        baseline_rows = grp[grp['target'] == 0]
        if len(baseline_rows) > 0:
            for f in sensor_features:
                b_val = baseline_rows[f].iloc[0]
                if pd.notna(b_val):
                    norm_df.loc[grp.index, f'delta_{f}'] = norm_df.loc[grp.index, f] - b_val

    norm_df.to_csv(NORMALIZED_CSV, index=False)
    print(f"Normalized feature table saved to: {NORMALIZED_CSV}")
    print(f"Added {len(delta_cols)} subject-relative delta features.")
    print("=" * 70)

    return norm_df

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2] / 'MODALSIM94'
    assemble_modalsim_dataset(root)

