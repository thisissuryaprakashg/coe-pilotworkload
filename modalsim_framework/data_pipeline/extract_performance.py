"""
extract_performance.py
======================
Extracts flight trajectory deviation errors from MODALSIM94 Performance files.
Features:
  - flight_error_mean: mean Euclidean deviation from tunnel centerline
  - flight_error_std: tracking instability
  - flight_error_max: peak deviation
  - flight_error_h_mean: mean horizontal drift
  - flight_error_v_mean: mean vertical drift
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def extract_performance_features(modalsim_root):
    perf_dir = Path(modalsim_root) / 'Performance'
    files = glob.glob(str(perf_dir / 'Performance_*.csv'))
    
    records = []
    for f in files:
        fname = os.path.basename(f)
        parts = fname.replace('Performance_', '').replace('.csv', '').split('_')
        if len(parts) != 2:
            continue
        subj, scen = parts[0], parts[1]
        if scen not in ['NBTask', '1BTask', '2BTask']:
            continue
            
        try:
            df = pd.read_csv(f, sep=';', low_memory=False)
            if 'HorizontalDistance' not in df.columns or 'VerticalDistance' not in df.columns:
                continue
                
            if df['HorizontalDistance'].dtype == object:
                h_dist = pd.to_numeric(df['HorizontalDistance'].str.replace(',', '.'), errors='coerce').dropna()
            else:
                h_dist = pd.to_numeric(df['HorizontalDistance'], errors='coerce').dropna()
                
            if df['VerticalDistance'].dtype == object:
                v_dist = pd.to_numeric(df['VerticalDistance'].str.replace(',', '.'), errors='coerce').dropna()
            else:
                v_dist = pd.to_numeric(df['VerticalDistance'], errors='coerce').dropna()
            
            common_len = min(len(h_dist), len(v_dist))
            if common_len < 50:
                continue
                
            h_vals = h_dist.iloc[:common_len].values
            v_vals = v_dist.iloc[:common_len].values
            
            # 3D tracking error from centerline
            eucl_err = np.sqrt(h_vals**2 + v_vals**2)
            
            rec = {
                'subject': subj,
                'scenario': scen,
                'flight_error_mean': float(np.mean(eucl_err)),
                'flight_error_std': float(np.std(eucl_err)),
                'flight_error_max': float(np.max(eucl_err)),
                'flight_error_h_mean': float(np.mean(np.abs(h_vals))),
                'flight_error_v_mean': float(np.mean(np.abs(v_vals)))
            }
            records.append(rec)
        except Exception:
            continue
            
    if not records:
        print("[Flight Performance] Warning: No records extracted.")
        return pd.DataFrame(columns=['subject', 'scenario', 'flight_error_mean', 'flight_error_std', 
                                     'flight_error_max', 'flight_error_h_mean', 'flight_error_v_mean'])
    perf_df = pd.DataFrame(records).drop_duplicates(subset=['subject', 'scenario'])
    print(f"[Flight Performance] Extracted {len(perf_df)} records across {perf_df['subject'].nunique()} participants.")
    return perf_df

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2] / 'MODALSIM94'
    df_perf = extract_performance_features(root)
    print(df_perf.head())

