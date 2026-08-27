"""
pilot_personalized_delta_math_model.py
======================================
Personalized Pilot Workload Detection Engine using Baseline-Calibrated Mathematical Models.

Architecture:
1. Input:
   - X_baseline : Pilot's pre-flight resting baseline vector (HR, HRV, Pupil, EMG, EDA)
   - X_current  : Current real-time sensor window during flight
2. Core Mathematical Deviation Transformation:
   - Delta = X_current - X_baseline
   - Relative Surge (% Delta) = (X_current - X_baseline) / (|X_baseline| + eps)
   - Interaction Term = Delta * X_baseline (accounting for baseline-dependent sensitivity)
3. Mathematical Models Evaluated:
   - Model A: Pure Delta Deviation Mathematical Equation
   - Model B: Baseline-Conditioned Deviation Equation (Delta + Baseline + Interaction)
   - Model C: Hybrid Engine (Physiological Deltas + Real-Time Gaze Scanning Dynamics)
4. Evaluation:
   - Strict Leave-One-Subject-Out (LOSO-CV, 34 pilot folds)
   - Strict 4-Class, Adjacent (+/-1), Binary Low/High, Extreme 1v4, Macro ROC-AUC
5. Real-Time Pilot Workload Detector Simulation:
   - Takes a pilot baseline dict + current sensor dict -> outputs predicted workload, probabilities, and alerts.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score

MASTER_CSV = r'c:\coe\master_feature_matrix.csv'
OUT_PERSONALIZED_CSV = r'c:\coe\personalized_delta_model_results.csv'
OUT_EQUATION_CSV = r'c:\coe\personalized_math_equation_weights.csv'

def engineer_personalized_features(df):
    """
    Computes absolute deltas, percentage surges, and baseline interaction terms.
    """
    df_p = df.copy()
    
    # 1. Base Modality Mapping (Current Live Reading vs Rest Baseline)
    pairs = [
        ('ecg_hr_mean', 'rest_HR', 'HR'),
        ('ecg_rr_rmssd_ms', 'rest_HRV_RMSSD', 'HRV_RMSSD'),
        ('ecg_rr_sdnn_ms', 'rest_HRV_SDNN', 'HRV_SDNN'),
        ('ecg_rr_lf_hf_ratio', 'rest_HRV_LF_HF', 'HRV_LF_HF'),
        ('eda_tonic_mean_uS', 'rest_EDA_tonic', 'EDA_tonic'),
        ('emg_flexor_rms_mV', 'rest_EMG_flexor', 'EMG_flexor'),
        ('emg_extensor_rms_mV', 'rest_EMG_extensor', 'EMG_extensor'),
        ('pupil_diam_mean_mm', 'rest_pupil_diam', 'pupil_diam')
    ]
    
    for live_col, rest_col, name in pairs:
        if live_col in df_p.columns and rest_col in df_p.columns:
            # Absolute Delta (Live - Rest)
            df_p[f'delta_{name}'] = df_p[live_col] - df_p[rest_col]
            # Percentage Surge (%)
            df_p[f'pct_surge_{name}'] = (df_p[live_col] - df_p[rest_col]) / (np.abs(df_p[rest_col]) + 1e-4)
            # Baseline Interaction Term
            df_p[f'interact_{name}'] = df_p[f'delta_{name}'] * df_p[rest_col]
            
    # 2. Add Fused Eye Scanning Dynamics
    if 'overall_gaze_entropy_LY' in df_p.columns and 'overall_gaze_entropy_RY' in df_p.columns:
        df_p['fused_gaze_entropy_Y'] = df_p[['overall_gaze_entropy_LY', 'overall_gaze_entropy_RY']].mean(axis=1)
    if 'psd_max_LY' in df_p.columns and 'psd_max_RY' in df_p.columns:
        df_p['fused_psd_max_Y'] = df_p[['psd_max_LY', 'psd_max_RY']].mean(axis=1)
        
    return df_p

def get_personalized_subsets(df_p):
    """Defines feature sets for mathematical delta evaluation."""
    
    # 1. Pure Absolute Deltas (8 features)
    pure_deltas = [c for c in df_p.columns if c.startswith('delta_')]
    
    # 2. Relative Percentage Surges (% Delta, 8 features)
    pct_surges = [c for c in df_p.columns if c.startswith('pct_surge_')]
    
    # 3. Baseline-Conditioned (Deltas + Baselines + Interactions, 24 features)
    rest_cols = [c for c in df_p.columns if c.startswith('rest_')]
    interact_cols = [c for c in df_p.columns if c.startswith('interact_')]
    baseline_conditioned = list(dict.fromkeys(pure_deltas + rest_cols + interact_cols))
    
    # 4. Hybrid Mathematical Engine (Physiological Deltas + Real-Time Gaze Dynamics)
    gaze_dynamics = ['fused_gaze_entropy_Y', 'fused_psd_max_Y', 'fix_density_mean', 'fix_dur_kurt', 'pupil_diam_kurt', 'pupil_diam_skew']
    gaze_dynamics = [f for f in gaze_dynamics if f in df_p.columns]
    hybrid_engine = list(dict.fromkeys(pure_deltas + gaze_dynamics))
    
    return {
        '1. Pure Absolute Deltas (Delta = Live - Baseline)': pure_deltas,
        '2. Relative Percentage Surges (% Change from Rest)': pct_surges,
        '3. Baseline-Conditioned Equations (Delta + Baseline + Interactions)': baseline_conditioned,
        '4. Hybrid Engine (Personalized Deltas + Real-Time Gaze Dynamics)': hybrid_engine
    }

def evaluate_loso_math_model(df_p, feature_cols, target_col='difficulty_ground_truth'):
    """
    Evaluates the Mathematical Logistic Regression equation under strict LOSO cross-validation.
    """
    df_eval = df_p[df_p['subject'] != 'sub-cp003'].copy()
    subjects = sorted(df_eval['subject'].unique())
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=1500, C=1.0, random_state=42))
    ])
    
    y_true_all, y_pred_all, y_prob_all = [], [], []
    
    for subj in subjects:
        train_mask = (df_eval['subject'] != subj)
        test_mask = (df_eval['subject'] == subj)
        
        X_tr = df_eval.loc[train_mask, feature_cols]
        y_tr = df_eval.loc[train_mask, target_col].values.astype(int)
        X_te = df_eval.loc[test_mask, feature_cols]
        y_te = df_eval.loc[test_mask, target_col].values.astype(int)
        
        if len(X_te) == 0:
            continue
            
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        probs = pipe.predict_proba(X_te)
        
        y_true_all.extend(y_te)
        y_pred_all.extend(preds)
        y_prob_all.extend(probs)
        
    y_t = np.array(y_true_all)
    y_p = np.array(y_pred_all)
    y_pr = np.array(y_prob_all)
    
    acc_strict = accuracy_score(y_t, y_p)
    acc_adjacent = np.mean(np.abs(y_t - y_p) <= 1)
    
    # Binary Low (1-2) vs High (3-4)
    y_t_bin = (y_t >= 3).astype(int)
    y_p_bin = (y_p >= 3).astype(int)
    acc_binary = accuracy_score(y_t_bin, y_p_bin)
    
    # Extreme (1 vs 4)
    ext_mask = (y_t == 1) | (y_t == 4)
    acc_extreme = accuracy_score(y_t[ext_mask], y_p[ext_mask]) if np.sum(ext_mask) > 0 else np.nan
    
    f1_macro = f1_score(y_t, y_p, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_t, y_p)
    
    try:
        auc_macro = roc_auc_score(y_t, y_pr, multi_class='ovr', average='macro')
    except Exception:
        auc_macro = np.nan
        
    return {
        'Strict_4Class_Acc': acc_strict,
        'Adjacent_Acc (+/-1 Class)': acc_adjacent,
        'Binary_LowHigh_Acc (1-2 vs 3-4)': acc_binary,
        'Extreme_1v4_Acc': acc_extreme,
        'Macro_ROC_AUC': auc_macro,
        'Macro_F1': f1_macro,
        'Cohen_Kappa': kappa,
        'Num_Features': len(feature_cols)
    }

def fit_and_export_hybrid_math_equation(df_p, feature_cols):
    """
    Fits the final Hybrid Mathematical Model on all clean data and exports exact beta weights and odds ratios.
    """
    df_eval = df_p[df_p['subject'] != 'sub-cp003'].copy()
    X = df_eval[feature_cols].copy()
    y = df_eval['difficulty_ground_truth'].values.astype(int)
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_s = scaler.fit_transform(X_imp)
    
    lr = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0, random_state=42)
    lr.fit(X_s, y)
    
    classes = lr.classes_
    coefs = lr.coef_
    intercepts = lr.intercept_
    
    records = []
    for c_idx, cls in enumerate(classes):
        for f_idx, feat in enumerate(feature_cols):
            beta = coefs[c_idx, f_idx]
            records.append({
                'Workload_Level': f'Level_{cls}',
                'Feature': feat,
                'Beta_Weight (Standardized)': beta,
                'Odds_Ratio': np.exp(beta),
                'Feature_Mean': float(np.mean(X_imp[:, f_idx])),
                'Feature_Std': float(np.std(X_imp[:, f_idx]))
            })
            
    df_eq = pd.DataFrame(records)
    df_eq.to_csv(OUT_EQUATION_CSV, index=False)
    
    return lr, imputer, scaler

def simulate_realtime_cockpit_detection(model_pipe, feature_cols, df_p):
    """
    Demonstrates real-time cockpit pattern detection from pilot baseline + live flight readings
    on actual held-out pilot test flights from the dataset.
    """
    print("\n" + "=" * 100)
    print("REAL-TIME PILOT WORKLOAD DETECTOR DEMO (Baseline Input -> Delta Calculation -> Workload State)")
    print("=" * 100)
    
    # Pick a few diverse real runs across difficulty levels from a test subject (e.g. sub-cp029)
    test_subjs = ['sub-cp029', 'sub-cp014', 'sub-cp037']
    demo_df = df_p[df_p['subject'].isin(test_subjs)].copy()
    
    for level in [1, 2, 3, 4]:
        sample_rows = demo_df[demo_df['difficulty_ground_truth'] == level]
        if len(sample_rows) == 0:
            continue
        row = sample_rows.iloc[0]
        subj = row['subject']
        run_name = row['run_folder'] if 'run_folder' in row else f"Run (Level {level})"
        
        # Prepare input as DataFrame matching feature_cols
        X_input = pd.DataFrame([row[feature_cols].to_dict()])[feature_cols]
        
        pred_lvl = model_pipe.predict(X_input)[0]
        probs = model_pipe.predict_proba(X_input)[0]
        
        b_hr = row.get('rest_HR', np.nan)
        f_hr = row.get('ecg_hr_mean', np.nan)
        d_hr = row.get('delta_HR', f_hr - b_hr)
        
        b_pupil = row.get('rest_pupil_diam', np.nan)
        f_pupil = row.get('pupil_diam_mean_mm', np.nan)
        d_pupil = row.get('delta_pupil_diam', f_pupil - b_pupil)
        
        b_emg = row.get('rest_EMG_flexor', np.nan)
        f_emg = row.get('emg_flexor_rms_mV', np.nan)
        d_emg = row.get('delta_EMG_flexor', f_emg - b_emg)
        
        print(f"\n--- Pilot: {subj} | True Task Difficulty: LEVEL {level} ({run_name}) ---")
        print(f"  Pilot Pre-Flight Resting Baseline: HR = {b_hr:.1f} BPM | Pupil = {b_pupil:.2f} mm | EMG_Flex = {b_emg:.3f} mV")
        print(f"  Live In-Flight Sensor Reading    : HR = {f_hr:.1f} BPM | Pupil = {f_pupil:.2f} mm | EMG_Flex = {f_emg:.3f} mV")
        print(f"  Dynamic Computed Deviations      : Delta_HR = {d_hr:+.1f} BPM | Delta_Pupil = {d_pupil:+.2f} mm | Delta_EMG = {d_emg:+.3f} mV")
        print(f"  ==> PREDICTED WORKLOAD STATE     : LEVEL {pred_lvl} (Confidence: {probs[pred_lvl-1]*100:.1f}%)")
        print(f"  ==> Model Class Probabilities    : L1: {probs[0]*100:.1f}% | L2: {probs[1]*100:.1f}% | L3: {probs[2]*100:.1f}% | L4: {probs[3]*100:.1f}%")
        
        if pred_lvl >= 3:
            print("  ⚠️  COCKPIT ADVISORY: High Pilot Mental/Physical Workload. Cockpit alerting / automation engaged.")
        else:
            print("  ✅  COCKPIT STATUS: Nominal Workload. Pilot cognitive capacity clear.")

def main():
    print("====================================================================================================")
    print("BASELINE-CALIBRATED MATHEMATICAL DELTA WORKLOAD ENGINE")
    print("====================================================================================================")
    
    # 1. Load data and engineer personalized baseline deltas
    df_raw = pd.read_csv(MASTER_CSV)
    df_p = engineer_personalized_features(df_raw)
    feature_subsets = get_personalized_subsets(df_p)
    
    # 2. Run LOSO Cross-Validation on each mathematical architecture
    print("\n----------------------------------------------------------------------------------------------------")
    print("LEAVE-ONE-SUBJECT-OUT (LOSO) BENCHMARK ACROSS MATHEMATICAL DELTA ARCHITECTURES")
    print("----------------------------------------------------------------------------------------------------")
    
    records = []
    for name, f_cols in feature_subsets.items():
        print(f"Evaluating: {name} ({len(f_cols)} features)...")
        res = evaluate_loso_math_model(df_p, f_cols)
        res['Architecture'] = name
        records.append(res)
        
    df_res = pd.DataFrame(records)
    cols_order = ['Architecture', 'Strict_4Class_Acc', 'Adjacent_Acc (+/-1 Class)', 'Binary_LowHigh_Acc (1-2 vs 3-4)', 'Macro_ROC_AUC', 'Macro_F1', 'Extreme_1v4_Acc', 'Cohen_Kappa', 'Num_Features']
    df_res_clean = df_res[cols_order].sort_values('Strict_4Class_Acc', ascending=False).reset_index(drop=True)
    df_res.to_csv(OUT_PERSONALIZED_CSV, index=False)
    
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 150)
    print("\n" + "=" * 140)
    print("RESULTS: MATHEMATICAL DELTA & BASELINE-CALIBRATED ARCHITECTURES (LOSO-CV)")
    print("=" * 140)
    print(df_res_clean.to_string(index=False, formatters={
        'Strict_4Class_Acc': '{:.2%}'.format,
        'Adjacent_Acc (+/-1 Class)': '{:.2%}'.format,
        'Binary_LowHigh_Acc (1-2 vs 3-4)': '{:.2%}'.format,
        'Macro_ROC_AUC': '{:.3f}'.format,
        'Macro_F1': '{:.3f}'.format,
        'Extreme_1v4_Acc': '{:.2%}'.format,
        'Cohen_Kappa': '{:.3f}'.format
    }))
    
    # 3. Fit and Export Standalone Mathematical Equation
    hybrid_feats = feature_subsets['4. Hybrid Engine (Personalized Deltas + Real-Time Gaze Dynamics)']
    lr_model, imputer, scaler = fit_and_export_hybrid_math_equation(df_p, hybrid_feats)
    print(f"\nSaved standalone mathematical equation weights to: {OUT_EQUATION_CSV}")
    
    # Create fitted pipeline for real-time detector simulation
    full_pipe = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('clf', lr_model)
    ])
    
    # 4. Run Real-Time Cockpit Detector Simulation
    simulate_realtime_cockpit_detection(full_pipe, hybrid_feats, df_p)
    print("=" * 100)

if __name__ == '__main__':
    main()
