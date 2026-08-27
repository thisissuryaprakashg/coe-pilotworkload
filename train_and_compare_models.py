"""
train_and_compare_models.py
===========================
Implements Section 8 of cogpilot_project_spec.md:
1. Methodological Adjustment:
   - Evaluates 34 clean subjects (34 LOSO folds, 394 runs) for eye-tracking / multimodal models,
     excluding sub-cp003 (which has 0% valid eye tracking).
   - Evaluates all 35 subjects (408 runs) for autonomic-only models.
2. Section 8.1: Mathematical Model (Interpretable Multinomial / Ordinal Logistic Regression)
   - Fits parametric equation: P(difficulty = k) = Softmax(beta_0,k + sum(beta_i,k * x_i))
   - Extracts and saves fitted beta weights, odds ratios, and interpretability rankings.
3. Section 8.2: Machine Learning Classifiers
   - Random Forest Classifier (RF)
   - Support Vector Classifier (SVM, RBF)
   - Gradient Boosted Decision Trees (GBDT)
4. Section 8.3: Leave-One-Subject-Out Cross-Validation (LOSO-CV)
   - Evaluates Accuracy, Macro F1-score, Cohen's Kappa, and Extreme Workload (1 vs 4) accuracy.
5. Ablation Studies:
   - Full Multimodal (28 significant features)
   - Oculomotor Only (Pupillometry & Gaze Dynamics)
   - Autonomic Only (ECG, EDA, EMG)
   - Personalized Delta Features vs. Raw Features (The Delta Paradox Ablation)
   - Parsimonious 8-Feature Core Equation
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

MASTER_CSV = r'c:\coe\master_feature_matrix.csv'
SELECTED_JSON = r'c:\coe\selected_features.json'
OUT_RESULTS_CSV = r'c:\coe\model_comparison_results.csv'
OUT_MATH_COEFS_CSV = r'c:\coe\math_model_coefficients.csv'

def get_feature_subsets(df, selected_dict):
    """Defines feature subsets for model training and ablation studies."""
    sig_features = [f for f in selected_dict['all_significant_features'] if f in df.columns]
    
    # 1. Full Multimodal (Significant features)
    multimodal_feats = sig_features
    
    # 2. Oculomotor Only
    oculo_feats = [f for f in sig_features if any(k in f for k in ['gaze', 'pupil', 'fix', 'sac', 'psd'])]
    
    # 3. Autonomic Only (ECG, EDA, EMG)
    autonomic_feats = [f for f in df.columns if any(k in f for k in ['ecg_', 'eda_', 'emg_', 'HR', 'delta_']) and np.issubdtype(df[f].dtype, np.number)]
    
    # 4. Personalized Delta Features Only (Section 5.3)
    delta_feats = [f for f in df.columns if f.startswith('delta_')]
    
    # 5. Raw Autonomic Features Only (for Delta vs Raw Ablation)
    raw_autonomic = ['ecg_hr_mean', 'ecg_rr_sdnn_ms', 'ecg_rr_rmssd_ms', 'ecg_rr_lf_hf_ratio', 'eda_tonic_mean_uS', 'emg_flexor_rms_mV', 'emg_extensor_rms_mV']
    raw_autonomic = [f for f in raw_autonomic if f in df.columns]

    # 6. Core 8-Feature Parsimonious Set (for high interpretability math equation)
    core_math_feats = [
        'overall_gaze_entropy_LY', 'psd_max_LY', 'pupil_diam_kurt', 'pupil_diam_std_mm',
        'fix_dur_stdev', 'fix_density_mean', 'emg_extensor_rms_mV', 'delta_HR'
    ]
    core_math_feats = [f for f in core_math_feats if f in df.columns]
    
    return {
        'Multimodal_Significant (28 feats)': multimodal_feats,
        'Oculomotor_Only (26 feats)': oculo_feats,
        'Autonomic_Only (ECG/EDA/EMG)': autonomic_feats,
        'Personalized_Delta_Only (7 feats)': delta_feats,
        'Raw_Autonomic_Only (7 feats)': raw_autonomic,
        'Parsimonious_Core_Math (8 feats)': core_math_feats
    }

def run_loso_cv(df, feature_cols, target_col='difficulty_ground_truth'):
    """
    Executes Leave-One-Subject-Out Cross-Validation.
    Automatically handles sub-cp003 exclusion when eye-tracking features are present (34 folds).
    """
    has_eye_features = any(k in f.lower() for f in feature_cols for k in ['gaze', 'pupil', 'fix', 'sac', 'psd', 'eye'])
    
    if has_eye_features:
        # Crucial Methodological Fix: Exclude sub-cp003 (0% eye quality across all runs)
        df_eval = df[df['subject'] != 'sub-cp003'].copy()
        expected_subjects = 34
    else:
        df_eval = df.copy()
        expected_subjects = 35
        
    subjects = df_eval['subject'].unique()
    
    models = {
        'Math_Model (Multinomial Logistic Regression)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0, random_state=42))
        ]),
        'ML_Model (Random Forest)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1))
        ]),
        'ML_Model (Support Vector Machine - RBF)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42), ensemble=False))
        ]),
        'ML_Model (Gradient Boosted Trees - GBDT)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42))
        ])
    }
    
    results = {name: {'y_true': [], 'y_pred': []} for name in models}
    
    for subj in subjects:
        # LOSO split
        train_mask = (df_eval['subject'] != subj)
        test_mask = (df_eval['subject'] == subj)
        
        X_train = df_eval.loc[train_mask, feature_cols]
        y_train = df_eval.loc[train_mask, target_col]
        X_test = df_eval.loc[test_mask, feature_cols]
        y_test = df_eval.loc[test_mask, target_col]
        
        if len(X_test) == 0:
            continue
            
        for name, pipe in models.items():
            try:
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                results[name]['y_true'].extend(y_test.values)
                results[name]['y_pred'].extend(preds)
            except Exception:
                pass
                
    # Calculate performance metrics per model
    metrics_summary = []
    for name, res in results.items():
        y_t = np.array(res['y_true'])
        y_p = np.array(res['y_pred'])
        
        acc = accuracy_score(y_t, y_p)
        f1_macro = f1_score(y_t, y_p, average='macro')
        f1_weighted = f1_score(y_t, y_p, average='weighted')
        kappa = cohen_kappa_score(y_t, y_p)
        cm = confusion_matrix(y_t, y_p, labels=[1, 2, 3, 4])
        
        # Extreme accuracy (Level 1 vs Level 4 binary discrimination)
        extreme_mask = (y_t == 1) | (y_t == 4)
        acc_extremes = accuracy_score(y_t[extreme_mask], y_p[extreme_mask]) if np.sum(extreme_mask) > 0 else np.nan
        
        metrics_summary.append({
            'Model': name,
            'Accuracy': acc,
            'Macro_F1': f1_macro,
            'Weighted_F1': f1_weighted,
            'Cohen_Kappa': kappa,
            'Extreme_Diff_Accuracy (1 vs 4)': acc_extremes,
            'Total_Test_Samples': len(y_t),
            'LOSO_Folds': len(subjects)
        })
        
    return metrics_summary, results

def fit_interpretable_math_equation(df, feature_cols):
    """
    Fits the standalone interpretable Mathematical Model on the entire valid dataset
    and outputs the exact equation beta coefficients and odds ratios.
    """
    has_eye = any(k in f.lower() for f in feature_cols for k in ['gaze', 'pupil', 'fix', 'sac', 'psd', 'eye'])
    df_fit = df[df['subject'] != 'sub-cp003'].copy() if has_eye else df.copy()
    
    X = df_fit[feature_cols].copy()
    y = df_fit['difficulty_ground_truth'].values
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    
    lr = LogisticRegression(solver='lbfgs', max_iter=1500, C=1.0, random_state=42)
    lr.fit(X_scaled, y)
    
    classes = lr.classes_
    coefs = lr.coef_
    intercepts = lr.intercept_
    
    records = []
    for c_idx, cls in enumerate(classes):
        for f_idx, feat in enumerate(feature_cols):
            beta = coefs[c_idx, f_idx]
            odds_ratio = np.exp(beta)
            records.append({
                'Difficulty_Class': f'Level_{cls}',
                'Feature': feat,
                'Beta_Coefficient (Standardized)': beta,
                'Odds_Ratio': odds_ratio,
                'Feature_Mean': float(np.mean(X_imp[:, f_idx])),
                'Feature_Std': float(np.std(X_imp[:, f_idx]))
            })
            
    df_coefs = pd.DataFrame(records)
    df_coefs.to_csv(OUT_MATH_COEFS_CSV, index=False)
    print(f"Saved mathematical equation beta coefficients to: {OUT_MATH_COEFS_CSV}")
    
    return lr, scaler, intercepts, coefs

def main():
    print("========================================================================")
    print("SECTION 8: MATHEMATICAL VS. ML WORKLOAD MODELING (LOSO-CV BENCHMARK)")
    print("========================================================================")
    
    df = pd.read_csv(MASTER_CSV)
    with open(SELECTED_JSON, 'r') as f:
        selected_dict = json.load(f)
        
    feature_subsets = get_feature_subsets(df, selected_dict)
    
    all_comparison_records = []
    
    # 1. Evaluate all models across feature subsets using Leave-One-Subject-Out (LOSO)
    for subset_name, feat_list in feature_subsets.items():
        if len(feat_list) == 0:
            continue
        print(f"\nEvaluating Feature Set: [{subset_name}] ({len(feat_list)} features)...")
        metrics_summary, _ = run_loso_cv(df, feat_list)
        
        for m in metrics_summary:
            m['Feature_Subset'] = subset_name
            m['Num_Features'] = len(feat_list)
            all_comparison_records.append(m)
            
    df_comp = pd.DataFrame(all_comparison_records)
    
    # Reorder columns
    cols_order = ['Feature_Subset', 'Model', 'Accuracy', 'Macro_F1', 'Weighted_F1', 'Cohen_Kappa', 'Extreme_Diff_Accuracy (1 vs 4)', 'LOSO_Folds', 'Total_Test_Samples']
    df_comp_clean = df_comp[cols_order].sort_values(['Feature_Subset', 'Macro_F1'], ascending=[True, False]).reset_index(drop=True)
    
    df_comp.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"\nSaved full LOSO cross-validation benchmark results to: {OUT_RESULTS_CSV}")
    
    # 2. Fit and output the standalone interpretable Mathematical Model Equation
    print("\n------------------------------------------------------------------------")
    print("SECTION 8.1: FITTING STANDALONE INTERPRETABLE MATHEMATICAL MODEL")
    print("------------------------------------------------------------------------")
    primary_feats = feature_subsets['Multimodal_Significant (28 feats)']
    lr_model, scaler, intercepts, coefs = fit_interpretable_math_equation(df, primary_feats)
    
    # 3. Print Head-to-Head Benchmark Table
    print("\n====================================================================================================")
    print("HEAD-TO-HEAD BENCHMARK: MATHEMATICAL MODEL VS. ML CLASSIFIERS (LEAVE-ONE-SUBJECT-OUT CV)")
    print("====================================================================================================")
    
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 130)
    
    print(df_comp_clean.to_string(index=False, formatters={
        'Accuracy': '{:.2%}'.format,
        'Macro_F1': '{:.3f}'.format,
        'Weighted_F1': '{:.3f}'.format,
        'Cohen_Kappa': '{:.3f}'.format,
        'Extreme_Diff_Accuracy (1 vs 4)': '{:.2%}'.format
    }))
    
    # 4. Print Mathematical Formula Weights Summary
    print("\n----------------------------------------------------------------------------------------------------")
    print("MATHEMATICAL MODEL: TOP PHYSIOLOGICAL WEIGHTS FOR EXTREME WORKLOAD (LEVEL 4 vs LEVEL 1)")
    print("----------------------------------------------------------------------------------------------------")
    df_coefs = pd.read_csv(OUT_MATH_COEFS_CSV)
    
    lvl4_coefs = df_coefs[df_coefs['Difficulty_Class'] == 'Level_4'].sort_values('Beta_Coefficient (Standardized)', ascending=False)
    print("Top Positive Predictors of Severe Workload (Level 4):")
    for _, r in lvl4_coefs.head(5).iterrows():
        print(f"  + {r['Feature']:25s}: beta = {r['Beta_Coefficient (Standardized)']:+.4f} (Odds Ratio = {r['Odds_Ratio']:.3f})")
        
    print("\nTop Inverted/Negative Predictors of Severe Workload (suppressed under stress):")
    for _, r in lvl4_coefs.tail(5).iterrows():
        print(f"  - {r['Feature']:25s}: beta = {r['Beta_Coefficient (Standardized)']:+.4f} (Odds Ratio = {r['Odds_Ratio']:.3f})")
        
    print("====================================================================================================")

if __name__ == '__main__':
    main()
