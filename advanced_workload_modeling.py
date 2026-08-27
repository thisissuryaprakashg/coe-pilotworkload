"""
advanced_workload_modeling.py
=============================
Enhanced Workload Modeling Suite implementing:
1. Bilateral Eye Feature Fusion (denoising left + right eye channels).
2. Oculomotor + Forearm EMG Fusion (Visual Cognition + Physical Stick Grip Strain).
3. Multi-Faceted Evaluation Metrics:
   - Strict 4-Class Accuracy
   - Macro F1-Score
   - Adjacent Accuracy (Tolerance +/- 1 class: |y_true - y_pred| <= 1)
   - Binary Low (1-2) vs High (3-4) Workload Accuracy
   - Extreme Binary Accuracy (Level 1 vs Level 4)
   - Multi-Class One-vs-Rest Macro ROC-AUC
4. Dual Granularity Evaluation:
   - Run-Level (394 runs, 34 LOSO folds)
   - Denoised Subject x Difficulty Aggregated (136 samples, 34 LOSO folds)
5. Soft-Voting Ensembles combining top probabilistic classifiers.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score

MASTER_CSV = r'c:\coe\master_feature_matrix.csv'
OUT_RESULTS_CSV = r'c:\coe\enhanced_model_benchmark_results.csv'
OUT_AGG_RESULTS_CSV = r'c:\coe\enhanced_aggregated_benchmark_results.csv'

def fuse_bilateral_eye_features(df):
    """
    Creates clean bilateral-averaged eye features instead of dropping L or R channels.
    Averages bilateral gaze entropies, PSD maxes, pupil stats, and blink rates.
    """
    df_fused = df.copy()
    
    # 1. Bilateral Gaze Entropies
    for axis in ['X', 'Y', 'Z']:
        lx = f'overall_gaze_entropy_L{axis}'
        rx = f'overall_gaze_entropy_R{axis}'
        if lx in df.columns and rx in df.columns:
            df_fused[f'fused_gaze_entropy_{axis}'] = df[[lx, rx]].mean(axis=1)
            
    # 2. Bilateral PSD Max Powers
    for axis in ['X', 'Y', 'Z']:
        lx = f'psd_max_L{axis}'
        rx = f'psd_max_R{axis}'
        if lx in df.columns and rx in df.columns:
            df_fused[f'fused_psd_max_{axis}'] = df[[lx, rx]].mean(axis=1)
            
    # 3. Bilateral Eyes Closed Fraction (Blink rate)
    if 'eyes_closed_fraction_L' in df.columns and 'eyes_closed_fraction_R' in df.columns:
        df_fused['fused_eyes_closed_fraction'] = df[['eyes_closed_fraction_L', 'eyes_closed_fraction_R']].mean(axis=1)
        
    # 4. Bilateral Pupil Stats
    for stat in ['mean', 'stdev', 'skew', 'kurt']:
        l_stat = f'pupil_diam_{stat}_L'
        r_stat = f'pupil_diam_{stat}_R'
        if l_stat in df.columns and r_stat in df.columns:
            df_fused[f'fused_pupil_diam_{stat}'] = df[[l_stat, r_stat]].mean(axis=1)
            
    return df_fused

def get_enhanced_feature_subsets(df):
    """Defines targeted feature subsets including Fused Oculomotor + EMG."""
    
    # Fused Oculomotor Features
    fused_oculo = [
        'fused_gaze_entropy_X', 'fused_gaze_entropy_Y', 'fused_gaze_entropy_Z',
        'fused_psd_max_X', 'fused_psd_max_Y', 'fused_psd_max_Z',
        'fused_eyes_closed_fraction',
        'fused_pupil_diam_mean', 'fused_pupil_diam_stdev', 'fused_pupil_diam_skew', 'fused_pupil_diam_kurt',
        'gaze_entropy_x', 'gaze_entropy_y', 'pupil_diam_std_mm', 'pupil_diam_kurt', 'pupil_diam_skew',
        'fix_density_mean', 'fix_density_stdev', 'fix_density_skew', 'fix_density_kurt',
        'fix_dur_mean', 'fix_dur_stdev', 'fix_dur_skew', 'fix_dur_kurt'
    ]
    fused_oculo = [f for f in fused_oculo if f in df.columns]
    
    # Forearm EMG Features
    emg_feats = ['emg_flexor_rms_mV', 'emg_extensor_rms_mV', 'delta_EMG_flexor', 'delta_EMG_extensor']
    emg_feats = [f for f in emg_feats if f in df.columns]
    
    # 1. Targeted Hypothesis: Fused Oculomotor + Forearm EMG (Cognitive Vision + Physical Grip)
    oculo_emg = list(dict.fromkeys(fused_oculo + emg_feats))
    
    # 2. Oculomotor Only Fused
    oculo_only = fused_oculo
    
    # 3. Full Multimodal Fused (Oculomotor + EMG + ECG + EDA + Deltas)
    autonomic_deltas = [f for f in df.columns if any(k in f for k in ['ecg_', 'eda_', 'emg_', 'delta_']) and np.issubdtype(df[f].dtype, np.number)]
    full_multimodal = list(dict.fromkeys(fused_oculo + autonomic_deltas))
    
    # 4. Autonomic Only
    autonomic_only = [f for f in df.columns if any(k in f for k in ['ecg_', 'eda_', 'emg_']) and not f.startswith('delta_') and np.issubdtype(df[f].dtype, np.number)]
    
    # 5. Personalized Delta Only
    delta_only = [f for f in df.columns if f.startswith('delta_')]
    
    # 6. Parsimonious 8-Feature Core Set
    parsimonious_core = [
        'fused_gaze_entropy_Y', 'fused_psd_max_Y', 'fused_pupil_diam_skew', 'fused_pupil_diam_kurt',
        'fix_density_mean', 'fix_dur_kurt', 'emg_flexor_rms_mV', 'delta_HR'
    ]
    parsimonious_core = [f for f in parsimonious_core if f in df.columns]
    
    return {
        'Oculomotor + Forearm EMG (Cognitive + Physical)': oculo_emg,
        'Fused Oculomotor Only': oculo_only,
        'Full Multimodal Fused': full_multimodal,
        'Parsimonious 8-Feature Core': parsimonious_core,
        'Autonomic Only (ECG/EDA/EMG)': autonomic_only,
        'Personalized Delta Only': delta_only
    }

def evaluate_loso_pipeline(df, feature_cols, target_col='difficulty_ground_truth', is_aggregated=False):
    """
    Executes Leave-One-Subject-Out Cross-Validation.
    Calculates: Strict Accuracy, Macro F1, Adjacent Accuracy, Binary Low/High, Extreme 1v4, and ROC-AUC.
    """
    has_eye = any(k in f.lower() for f in feature_cols for k in ['gaze', 'pupil', 'fix', 'sac', 'psd', 'eye', 'fused'])
    if has_eye:
        df_eval = df[df['subject'] != 'sub-cp003'].copy()
    else:
        df_eval = df.copy()
        
    subjects = sorted(df_eval['subject'].unique())
    
    # Define models with probability estimates
    models = {
        'Gradient Boosted Trees (GBDT)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.06, subsample=0.85, random_state=42))
        ]),
        'Random Forest (RF)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=3, random_state=42, n_jobs=-1))
        ]),
        'Support Vector Classifier (SVM RBF)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', CalibratedClassifierCV(SVC(kernel='rbf', C=1.2, gamma='scale', random_state=42), ensemble=False))
        ]),
        'Multinomial Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(solver='lbfgs', max_iter=1500, C=0.8, random_state=42))
        ])
    }
    
    # Results dictionary
    results = {name: {'y_true': [], 'y_pred': [], 'y_prob': []} for name in models}
    results['Soft-Voting Ensemble (GBDT + RF + SVM)'] = {'y_true': [], 'y_pred': [], 'y_prob': []}
    
    for subj in subjects:
        train_mask = (df_eval['subject'] != subj)
        test_mask = (df_eval['subject'] == subj)
        
        X_tr = df_eval.loc[train_mask, feature_cols]
        y_tr = df_eval.loc[train_mask, target_col].values.astype(int)
        X_te = df_eval.loc[test_mask, feature_cols]
        y_te = df_eval.loc[test_mask, target_col].values.astype(int)
        
        if len(X_te) == 0:
            continue
            
        fold_probs = []
        for name, pipe in models.items():
            try:
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_te)
                probs = pipe.predict_proba(X_te)
                
                results[name]['y_true'].extend(y_te)
                results[name]['y_pred'].extend(preds)
                results[name]['y_prob'].extend(probs)
                
                if name in ['Gradient Boosted Trees (GBDT)', 'Random Forest (RF)', 'Support Vector Classifier (SVM RBF)']:
                    fold_probs.append(probs)
            except Exception:
                pass
                
        # Compute Soft-Voting Ensemble
        if len(fold_probs) == 3:
            ens_prob = (fold_probs[0]*0.40 + fold_probs[1]*0.35 + fold_probs[2]*0.25)
            ens_preds = np.argmax(ens_prob, axis=1) + 1
            results['Soft-Voting Ensemble (GBDT + RF + SVM)']['y_true'].extend(y_te)
            results['Soft-Voting Ensemble (GBDT + RF + SVM)']['y_pred'].extend(ens_preds)
            results['Soft-Voting Ensemble (GBDT + RF + SVM)']['y_prob'].extend(ens_prob)

    metrics_list = []
    for model_name, res in results.items():
        if len(res['y_true']) == 0:
            continue
        y_t = np.array(res['y_true'])
        y_p = np.array(res['y_pred'])
        y_pr = np.array(res['y_prob'])
        
        # 1. Strict 4-Class Accuracy & Macro F1
        acc_strict = accuracy_score(y_t, y_p)
        f1_macro = f1_score(y_t, y_p, average='macro', zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)
        
        # 2. Adjacent Accuracy (+/- 1 class tolerance)
        acc_adjacent = np.mean(np.abs(y_t - y_p) <= 1)
        
        # 3. Binary Low (1-2) vs High (3-4) Accuracy
        y_t_bin = (y_t >= 3).astype(int)
        y_p_bin = (y_p >= 3).astype(int)
        acc_binary = accuracy_score(y_t_bin, y_p_bin)
        f1_binary = f1_score(y_t_bin, y_p_bin, average='macro', zero_division=0)
        
        # 4. Extreme Binary (Level 1 vs Level 4 only)
        ext_mask = (y_t == 1) | (y_t == 4)
        acc_extreme = accuracy_score(y_t[ext_mask], y_p[ext_mask]) if np.sum(ext_mask) > 0 else np.nan
        
        # 5. Multi-Class One-vs-Rest Macro ROC-AUC
        try:
            auc_macro = roc_auc_score(y_t, y_pr, multi_class='ovr', average='macro')
        except Exception:
            auc_macro = np.nan
            
        metrics_list.append({
            'Model': model_name,
            'Strict_4Class_Acc': acc_strict,
            'Macro_F1': f1_macro,
            'Adjacent_Acc (+/-1 Class)': acc_adjacent,
            'Binary_LowHigh_Acc (1-2 vs 3-4)': acc_binary,
            'Binary_LowHigh_F1': f1_binary,
            'Extreme_1v4_Acc': acc_extreme,
            'Macro_ROC_AUC': auc_macro,
            'Cohen_Kappa': kappa,
            'LOSO_Folds': len(subjects),
            'Total_Samples': len(y_t)
        })
        
    return metrics_list

def create_aggregated_dataset(df):
    """
    Computes the mean feature vector per (subject, difficulty_ground_truth),
    averaging out session-to-session run noise.
    """
    meta_cols = ['subject', 'difficulty_ground_truth', 'difficulty_level']
    num_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]
    
    df_agg = df.groupby(['subject', 'difficulty_ground_truth'])[num_cols].mean().reset_index()
    return df_agg

def main():
    print("====================================================================================================")
    print("ENHANCED MULTI-METRIC WORKLOAD MODELING SUITE (LOSO-CV BENCHMARK)")
    print("====================================================================================================")
    
    # 1. Load Master Table & Apply Bilateral Eye Fusion
    raw_df = pd.read_csv(MASTER_CSV)
    df_fused = fuse_bilateral_eye_features(raw_df)
    feature_subsets = get_enhanced_feature_subsets(df_fused)
    
    # ------------------------------------------------------------------------------------------------
    # PIPELINE 1: RUN-LEVEL EVALUATION (394 clean runs, 34 LOSO folds)
    # ------------------------------------------------------------------------------------------------
    print("\n----------------------------------------------------------------------------------------------------")
    print("PIPELINE 1: RUN-LEVEL EVALUATION (394 Runs, 34 Pilot Folds)")
    print("----------------------------------------------------------------------------------------------------")
    
    run_records = []
    for subset_name, feat_cols in feature_subsets.items():
        print(f"Evaluating Feature Set: [{subset_name}] ({len(feat_cols)} features)...")
        metrics = evaluate_loso_pipeline(df_fused, feat_cols)
        for m in metrics:
            m['Feature_Subset'] = subset_name
            m['Num_Features'] = len(feat_cols)
            run_records.append(m)
            
    df_run_results = pd.DataFrame(run_records)
    
    # Reorder columns
    cols_order = [
        'Feature_Subset', 'Model', 'Strict_4Class_Acc', 'Adjacent_Acc (+/-1 Class)', 
        'Binary_LowHigh_Acc (1-2 vs 3-4)', 'Macro_ROC_AUC', 'Macro_F1', 'Extreme_1v4_Acc', 'Cohen_Kappa', 'Num_Features'
    ]
    df_run_clean = df_run_results[cols_order].sort_values(['Feature_Subset', 'Strict_4Class_Acc'], ascending=[True, False]).reset_index(drop=True)
    df_run_results.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"Saved run-level benchmark to: {OUT_RESULTS_CSV}")

    # ------------------------------------------------------------------------------------------------
    # PIPELINE 2: DENOISED SUBJECT x DIFFICULTY AGGREGATED EVALUATION (136 Samples, 34 LOSO Folds)
    # ------------------------------------------------------------------------------------------------
    print("\n----------------------------------------------------------------------------------------------------")
    print("PIPELINE 2: DENOISED SUBJECT x DIFFICULTY AGGREGATED EVALUATION (136 Samples, 34 Pilot Folds)")
    print("----------------------------------------------------------------------------------------------------")
    
    df_agg = create_aggregated_dataset(df_fused)
    agg_records = []
    for subset_name, feat_cols in feature_subsets.items():
        print(f"Evaluating Aggregated Feature Set: [{subset_name}] ({len(feat_cols)} features)...")
        metrics = evaluate_loso_pipeline(df_agg, feat_cols, is_aggregated=True)
        for m in metrics:
            m['Feature_Subset'] = subset_name
            m['Num_Features'] = len(feat_cols)
            agg_records.append(m)
            
    df_agg_results = pd.DataFrame(agg_records)
    df_agg_clean = df_agg_results[cols_order].sort_values(['Feature_Subset', 'Strict_4Class_Acc'], ascending=[True, False]).reset_index(drop=True)
    df_agg_results.to_csv(OUT_AGG_RESULTS_CSV, index=False)
    print(f"Saved aggregated benchmark to: {OUT_AGG_RESULTS_CSV}")

    # ------------------------------------------------------------------------------------------------
    # PRINT RESULTS COMPARISONS
    # ------------------------------------------------------------------------------------------------
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 160)
    
    print("\n" + "=" * 140)
    print("SUMMARY: RUN-LEVEL BENCHMARK (394 Runs across 34 Pilot Folds)")
    print("=" * 140)
    print(df_run_clean.to_string(index=False, formatters={
        'Strict_4Class_Acc': '{:.2%}'.format,
        'Adjacent_Acc (+/-1 Class)': '{:.2%}'.format,
        'Binary_LowHigh_Acc (1-2 vs 3-4)': '{:.2%}'.format,
        'Macro_ROC_AUC': '{:.3f}'.format,
        'Macro_F1': '{:.3f}'.format,
        'Extreme_1v4_Acc': '{:.2%}'.format,
        'Cohen_Kappa': '{:.3f}'.format
    }))
    
    print("\n" + "=" * 140)
    print("SUMMARY: DENOISED AGGREGATED BENCHMARK (Subject x Difficulty Averaging, 136 Samples)")
    print("=" * 140)
    print(df_agg_clean.to_string(index=False, formatters={
        'Strict_4Class_Acc': '{:.2%}'.format,
        'Adjacent_Acc (+/-1 Class)': '{:.2%}'.format,
        'Binary_LowHigh_Acc (1-2 vs 3-4)': '{:.2%}'.format,
        'Macro_ROC_AUC': '{:.3f}'.format,
        'Macro_F1': '{:.3f}'.format,
        'Extreme_1v4_Acc': '{:.2%}'.format,
        'Cohen_Kappa': '{:.3f}'.format
    }))
    print("=" * 140)

if __name__ == '__main__':
    main()
