"""
extract_pulse.py
================
Extracts and normalizes cardiovascular / HRV features from MODALSIM94 eSENSE_Pulse stream.
Features:
  - AverageHR, MinHR, MaxHR, DifferenceMinMax
  - SDNN, RMSDD, NN50, PNN50, StressIndex, AverageRR
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def extract_pulse_features(modalsim_root):
    pulse_dir = Path(modalsim_root) / 'eSENSE_Pulse'
    summary_file = pulse_dir / '0.GeneralPulseTable.csv'

    if summary_file.exists():
        df = pd.read_csv(summary_file, sep=';')
    else:
        raise FileNotFoundError(f"Pulse summary file not found: {summary_file}")

    # Standardize column types
    num_cols = ['MaxHR', 'MinHR', 'DifferenceMinMax', 'AverageHR', 'SDNN', 'RMSDD', 'NN50', 'StressIndex', 'AverageRR']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    df['PNN50'] = pd.to_numeric(
        df['PNN50'].astype(str).str.replace('%', '').str.replace(',', '.').str.strip(),
        errors='coerce'
    )

    # Clean scenario strings
    df['Scenario'] = df['Scenario'].astype(str).str.strip()
    df = df[df['Scenario'].isin(['NBTask', '1BTask', '2BTask'])].copy()

    # Rename to pulse_ prefix
    rename_dict = {
        'ParticipantID': 'subject',
        'Scenario': 'scenario',
        'AverageHR': 'pulse_hr_mean',
        'MaxHR': 'pulse_hr_max',
        'MinHR': 'pulse_hr_min',
        'DifferenceMinMax': 'pulse_hr_range',
        'SDNN': 'pulse_sdnn',
        'RMSDD': 'pulse_rmssd',
        'NN50': 'pulse_nn50',
        'PNN50': 'pulse_pnn50',
        'StressIndex': 'pulse_stress_index',
        'AverageRR': 'pulse_rr_mean'
    }
    df = df.rename(columns=rename_dict)
    keep_cols = list(rename_dict.values())
    df = df[keep_cols].drop_duplicates(subset=['subject', 'scenario'])
    
    print(f"[Pulse] Extracted {len(df)} records across {df['subject'].nunique()} participants.")
    return df

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2] / 'MODALSIM94'
    df_pulse = extract_pulse_features(root)
    print(df_pulse.head())

