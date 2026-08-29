"""
feature_screening.py
====================
Implements Section 7 of cogpilot_project_spec.md:
1. Performs One-Way ANOVA and Kruskal-Wallis non-parametric hypothesis testing 
   across the 4 task difficulty levels (1, 2, 3, 4) for all physiological features.
2. Checks normality per feature using Shapiro-Wilk.
3. Computes FDR (Benjamini-Hochberg) adjusted p-values.
4. Evaluates correlation with secondary target cumulative_total_error (Section 5.2).
5. Retains statistically significant physiological markers (p < 0.05).
6. Exports selected_features.json and feature_screening_report.csv for Section 8 modeling.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_CSV = PROJECT_ROOT / 'data' / 'processed' / 'master_feature_matrix.csv'
REPORT_CSV = PROJECT_ROOT / 'reports' / 'feature_screening_report.csv'
SELECTED_JSON = PROJECT_ROOT / 'reports' / 'selected_features.json'

# Non-feature metadata and identifier columns to exclude from statistical screening
EXCLUDE_COLS = {
    'subject', 'session', 'run_folder', 'run_number', 'difficulty_level',
    'difficulty_ground_truth', 'is_eye_analyzed', 'ecg_valid', 'eda_valid',
    'emg_valid', 'eye_valid', 'ecg_num_clean_beats', 'gaze_valid_sample_fraction',
    'cumulative_glideslope_error_deg', 'cumulative_localizer_error_deg',
    'cumulative_airspeed_error_kts', 'cumulative_total_error'
}

def fdr_correction(p_values):
    """Benjamini-Hochberg False Discovery Rate (FDR) q-value correction."""
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]
    q_values = np.zeros(n)
    
    q_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q_val = (sorted_p[i] * n) / rank
        q_min = min(q_min, q_val)
        q_values[sorted_indices[i]] = q_min
        
    return np.clip(q_values, 0.0, 1.0)

def main():
    print("========================================================================")
    print("SECTION 7: PHYSIOLOGICAL FEATURE SCREENING (ANOVA / KRUSKAL-WALLIS)")
    print("========================================================================")
    
    df = pd.read_csv(MASTER_CSV)
    print(f"Loaded master feature matrix: {df.shape[0]} runs x {df.shape[1]} columns.\n")
    
    # Identify numerical feature columns to screen
    candidate_features = [
        c for c in df.columns 
        if c not in EXCLUDE_COLS and np.issubdtype(df[c].dtype, np.number)
    ]
    print(f"Total candidate physiological features to evaluate: {len(candidate_features)}")
    
    # 4 Task Difficulty Groups
    groups_dict = {
        lvl: df[df['difficulty_ground_truth'] == lvl]
        for lvl in [1, 2, 3, 4]
    }
    
    flight_error = df['cumulative_total_error']
    
    results = []
    
    for feat in candidate_features:
        # Extract non-null data per difficulty group
        g1 = groups_dict[1][feat].dropna().values
        g2 = groups_dict[2][feat].dropna().values
        g3 = groups_dict[3][feat].dropna().values
        g4 = groups_dict[4][feat].dropna().values
        
        # Require at least 15 valid samples per group
        if min(len(g1), len(g2), len(g3), len(g4)) < 15:
            continue
            
        # Check normality via Shapiro-Wilk (subsample if N > 50 for stability)
        try:
            norm_p_vals = [
                stats.shapiro(g[:50])[1] if len(g) >= 3 else 0.0
                for g in [g1, g2, g3, g4]
            ]
            is_normal = all(p > 0.05 for p in norm_p_vals)
        except Exception:
            is_normal = False
            
        # Run One-Way ANOVA and Kruskal-Wallis
        try:
            f_stat, anova_p = stats.f_oneway(g1, g2, g3, g4)
        except Exception:
            f_stat, anova_p = np.nan, 1.0
            
        try:
            h_stat, kw_p = stats.kruskal(g1, g2, g3, g4)
        except Exception:
            h_stat, kw_p = np.nan, 1.0
            
        # Primary test selection based on normality
        test_used = "ANOVA (F)" if is_normal else "Kruskal-Wallis (H)"
        primary_p = anova_p if is_normal else kw_p
        test_stat = f_stat if is_normal else h_stat
        
        # Pearson and Spearman correlation with cumulative flight error
        valid_err_mask = df[feat].notna() & flight_error.notna()
        if np.sum(valid_err_mask) > 30:
            spearman_rho, spearman_p = stats.spearmanr(df.loc[valid_err_mask, feat], flight_error[valid_err_mask])
        else:
            spearman_rho, spearman_p = np.nan, np.nan
            
        # Categorize feature domain
        if feat.startswith('delta_'):
            domain = 'Personalized Delta'
        elif 'ecg' in feat or 'HR' in feat or 'rr_' in feat:
            domain = 'Cardiovascular (ECG/HRV)'
        elif 'eda' in feat:
            domain = 'Electrodermal (EDA)'
        elif 'emg' in feat:
            domain = 'Muscular/Motion (EMG)'
        else:
            domain = 'Oculomotor (Pupil/Gaze)'
            
        results.append({
            'Feature': feat,
            'Domain': domain,
            'Test_Used': test_used,
            'Test_Statistic': test_stat,
            'p_value': primary_p,
            'ANOVA_p': anova_p,
            'Kruskal_p': kw_p,
            'Mean_Level_1': float(np.mean(g1)),
            'Mean_Level_2': float(np.mean(g2)),
            'Mean_Level_3': float(np.mean(g3)),
            'Mean_Level_4': float(np.mean(g4)),
            'Corr_Flight_Error_rho': spearman_rho,
            'Corr_Flight_Error_p': spearman_p,
            'Valid_N': len(g1) + len(g2) + len(g3) + len(g4)
        })
        
    df_res = pd.DataFrame(results)
    
    # Compute FDR q-values
    df_res['q_value_FDR'] = fdr_correction(df_res['p_value'].values)
    df_res['Significant_p05'] = df_res['p_value'] < 0.05
    
    # Sort by statistical significance (lowest p-value first)
    df_res = df_res.sort_values('p_value').reset_index(drop=True)
    
    # Save complete statistical report
    df_res.to_csv(REPORT_CSV, index=False)
    print(f"Saved full statistical screening report to: {REPORT_CSV}")
    
    # Filter significant features
    sig_df = df_res[df_res['Significant_p05']].copy()
    
    # Save selected feature names to JSON for Section 8 modeling
    selected_feature_names = sig_df['Feature'].tolist()
    
    # Separate into primary Delta features, ECG/EDA/EMG, and Eye-tracking
    selected_dict = {
        'all_significant_features': selected_feature_names,
        'delta_features': [f for f in selected_feature_names if f.startswith('delta_')],
        'raw_autonomic_features': [f for f in selected_feature_names if not f.startswith('delta_') and ('ecg' in f or 'eda' in f or 'emg' in f or 'HR' in f)],
        'oculomotor_features': [f for f in selected_feature_names if 'gaze' in f or 'pupil' in f or 'fix' in f or 'sac' in f]
    }
    
    with open(SELECTED_JSON, 'w') as f:
        json.dump(selected_dict, f, indent=2)
    print(f"Saved selected feature subset to: {SELECTED_JSON}\n")
    
    # Display formatted terminal summary
    print("----------------------------------------------------------------------------------------------------------------")
    print(f"SCREENING RESULTS SUMMARY: {len(sig_df)} / {len(df_res)} features are statistically significant across Difficulty Levels (p < 0.05)")
    print("----------------------------------------------------------------------------------------------------------------")
    
    display_cols = ['Feature', 'Domain', 'Test_Used', 'Test_Statistic', 'p_value', 'q_value_FDR', 'Corr_Flight_Error_rho']
    top_sig = sig_df[display_cols].head(25)
    
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)
    print(top_sig.to_string(index=False, formatters={
        'Test_Statistic': '{:.3f}'.format,
        'p_value': '{:.2e}'.format,
        'q_value_FDR': '{:.2e}'.format,
        'Corr_Flight_Error_rho': '{:.3f}'.format
    }))
    
    print("\n----------------------------------------------------------------------------------------------------------------")
    print(f"Breakdown of Significant Markers:")
    for dom, grp in sig_df.groupby('Domain'):
        print(f"  - {dom:25s}: {len(grp)} significant features (e.g. {', '.join(grp['Feature'].head(3).tolist())})")
    print("================================================================================================================")

if __name__ == '__main__':
    main()
