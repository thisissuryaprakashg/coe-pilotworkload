"""
extract_respiratory.py
======================
Extracts respiratory features from MODALSIM94 eSENSE_Respiratory files.
Features:
  - resp_bpm_mean, resp_bpm_std, resp_bpm_min, resp_bpm_max
  - resp_ra_mean, resp_ra_std
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def extract_respiratory_features(modalsim_root):
    resp_dir = Path(modalsim_root) / 'eSENSE_Respiratory'
    files = glob.glob(str(resp_dir / '*_Respiratory.csv'))
    
    records = []
    for f in files:
        fname = os.path.basename(f)
        parts = fname.replace('_Respiratory.csv', '').split('_')
        if len(parts) != 2:
            continue
        subj, scen = parts[0], parts[1]
        if scen not in ['NBTask', '1BTask', '2BTask']:
            continue
            
        try:
            df = pd.read_csv(f, sep=';', low_memory=False)
            if 'BreathPerMinute' not in df.columns or 'RA' not in df.columns:
                continue
                
            bpm = pd.to_numeric(df['BreathPerMinute'].astype(str).str.replace(',', '.'), errors='coerce').dropna()
            ra = pd.to_numeric(df['RA'].astype(str).str.replace(',', '.'), errors='coerce').dropna()
            
            # Physiological plausibility filter for respiration: 4 to 60 breaths/min
            bpm = bpm[(bpm >= 4) & (bpm <= 60)]
            
            if len(bpm) < 5:
                continue
                
            rec = {
                'subject': subj,
                'scenario': scen,
                'resp_bpm_mean': float(bpm.mean()),
                'resp_bpm_std': float(bpm.std()),
                'resp_bpm_min': float(bpm.min()),
                'resp_bpm_max': float(bpm.max()),
                'resp_ra_mean': float(ra.mean()) if len(ra) > 5 else np.nan,
                'resp_ra_std': float(ra.std()) if len(ra) > 5 else np.nan
            }
            records.append(rec)
        except Exception as e:
            continue
            
    res_df = pd.DataFrame(records).drop_duplicates(subset=['subject', 'scenario'])
    print(f"[Respiratory] Extracted {len(res_df)} records across {res_df['subject'].nunique()} participants.")
    return res_df

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2] / 'MODALSIM94'
    df_resp = extract_respiratory_features(root)
    print(df_resp.head())

