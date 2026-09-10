"""
fold_safe_transform.py
======================
Fold-Safe Physiological Transformation & Normalization Engine (Zero-Leakage Guaranteed).

Core Rules:
1. Strict exclusion of all flight performance errors (cumulative_total_error, glideslope, etc.).
2. Cross-modal interactions computed exclusively on physiological variables.
3. Training subject normalizations are fit strictly on training fold data.
4. Held-out test subjects are normalized using ONLY their pre-flight resting baseline (rest_*),
   never peeking at their task runs.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)

# Forbidden non-physiological columns (flight errors and metadata)
FORBIDDEN_COLUMNS = {
    'subject', 'session', 'run_folder', 'run_number',
    'difficulty_level', 'difficulty_ground_truth',
    'is_eye_analyzed', 'ecg_valid', 'eda_valid', 'emg_valid',
    'eye_valid', 'gaze_valid_sample_fraction', 'ecg_num_clean_beats',
    'cumulative_glideslope_error_deg', 'cumulative_localizer_error_deg',
    'cumulative_airspeed_error_kts', 'cumulative_total_error',
}

# Pure physiological interaction pairs
PHYSIOLOGICAL_INTERACTION_PAIRS = [
    # Cognitive-autonomic coupling: pupil variability * heart rate variability
    ('pupil_diam_std_mm', 'ecg_hr_std', 'pupil_x_hr_variability'),
    # Attention-arousal: vertical gaze dispersion * electrodermal tonic level
    ('gaze_entropy_y', 'eda_tonic_mean_uS', 'gaze_x_eda_arousal'),
    # Bilateral forearm muscle co-activation on HOTAS stick
    ('emg_flexor_rms_mV', 'emg_extensor_rms_mV', 'emg_forearm_coactivation'),
    # Visual cognitive load: pupil variability * vertical gaze entropy
    ('pupil_diam_std_mm', 'gaze_entropy_y', 'oculomotor_cognitive_load'),
    # Autonomic stress composite: Mean Heart Rate * Mean Electrodermal Conductance
    ('ecg_hr_mean', 'eda_conductance_mean_uS', 'autonomic_stress_composite'),
]

# Mapping task feature -> subject pre-flight resting baseline column
REST_BASELINE_MAPPING = {
    'ecg_hr_mean': 'rest_HR',
    'ecg_rr_sdnn_ms': 'rest_HRV_SDNN',
    'ecg_rr_rmssd_ms': 'rest_HRV_RMSSD',
    'ecg_rr_lf_hf_ratio': 'rest_HRV_LF_HF',
    'eda_tonic_mean_uS': 'rest_EDA_tonic',
    'eda_conductance_mean_uS': 'rest_EDA_conductance',
    'emg_flexor_rms_mV': 'rest_EMG_flexor',
    'emg_extensor_rms_mV': 'rest_EMG_extensor',
    'pupil_diam_mean_mm': 'rest_pupil_diam',
}


def add_pure_physiological_interactions(df):
    """
    Compute cross-modal interaction terms on purely physiological variables.
    Never uses flight error outcomes.
    """
    df_out = df.copy()
    for feat_a, feat_b, name in PHYSIOLOGICAL_INTERACTION_PAIRS:
        if feat_a in df.columns and feat_b in df.columns:
            a = df[feat_a].astype(float).values
            b = df[feat_b].astype(float).values
            df_out[name] = a * b
    return df_out


class FoldSafeRestingBaselineNormalizer(BaseEstimator, TransformerMixin):
    """
    Transforms physiological features relative to personal resting baseline.
    Computes:
      - Delta: (Task - Rest)
      - Relative Surge: (Task - Rest) / (|Rest| + eps)
    Then scales using a population StandardScaler fit strictly on the training folds.
    """
    def __init__(self, eps=1e-5):
        self.eps = eps
        self.scaler = StandardScaler()
        self.feature_names_in_ = None

    def _transform_features(self, df):
        df_trans = df.copy()
        for task_col, rest_col in REST_BASELINE_MAPPING.items():
            if task_col in df.columns and rest_col in df.columns:
                task_val = df[task_col].astype(float).values
                rest_val = df[rest_col].astype(float).values

                # Delta
                delta_col = f'{task_col}_rel_delta'
                df_trans[delta_col] = task_val - rest_val

                # Relative surge
                surge_col = f'{task_col}_rel_surge'
                df_trans[surge_col] = (task_val - rest_val) / (np.abs(rest_val) + self.eps)

        return df_trans

    def fit(self, X_train_df, y=None):
        X_trans = self._transform_features(X_train_df)
        feature_cols = [c for c in X_trans.columns if c not in FORBIDDEN_COLUMNS and not c.startswith('rest_')]
        self.feature_names_in_ = feature_cols
        
        # Fit scaler strictly on training samples
        numeric_mat = X_trans[feature_cols].values.astype(float)
        self.scaler.fit(numeric_mat)
        return self

    def transform(self, X_df):
        X_trans = self._transform_features(X_df)
        numeric_mat = X_trans[self.feature_names_in_].values.astype(float)
        scaled_mat = self.scaler.transform(numeric_mat)
        return pd.DataFrame(scaled_mat, columns=self.feature_names_in_, index=X_df.index)


class FoldSafeWithinSubjectScaler(BaseEstimator, TransformerMixin):
    """
    Fits subject-level mean and std strictly on training subjects' runs.
    For held-out test subject, scales using the training population distribution
    conditioned on the test subject's pre-flight resting baseline.
    """
    def __init__(self, subject_col='subject', eps=1e-5):
        self.subject_col = subject_col
        self.eps = eps
        self.train_subject_stats_ = {}
        self.global_train_stats_ = {}
        self.feature_cols_ = None

    def fit(self, df_train, y=None):
        self.feature_cols_ = [
            c for c in df_train.select_dtypes(include=['number', 'bool']).columns
            if c not in FORBIDDEN_COLUMNS and not c.startswith('rest_')
        ]
        
        # Compute global training distribution
        for col in self.feature_cols_:
            vals = df_train[col].dropna().values.astype(float)
            self.global_train_stats_[col] = {
                'mean': np.mean(vals) if len(vals) > 0 else 0.0,
                'std': np.std(vals) if len(vals) > 0 and np.std(vals) > self.eps else 1.0
            }

        # Compute per-subject training statistics (only for subjects in training set)
        for subj in df_train[self.subject_col].unique():
            subj_mask = df_train[self.subject_col] == subj
            self.train_subject_stats_[subj] = {}
            for col in self.feature_cols_:
                vals = df_train.loc[subj_mask, col].dropna().values.astype(float)
                self.train_subject_stats_[subj][col] = {
                    'mean': np.mean(vals) if len(vals) > 0 else self.global_train_stats_[col]['mean'],
                    'std': np.std(vals) if len(vals) > 0 and np.std(vals) > self.eps else self.global_train_stats_[col]['std']
                }

        return self

    def transform(self, df_input):
        df_out = pd.DataFrame(index=df_input.index)

        for col in self.feature_cols_:
            z_vals = np.zeros(len(df_input))
            for i, (idx, row) in enumerate(df_input.iterrows()):
                subj = row[self.subject_col]
                raw_val = float(row[col]) if not pd.isna(row[col]) else self.global_train_stats_[col]['mean']
                
                if subj in self.train_subject_stats_:
                    # Training subject: use their train-run statistics
                    m = self.train_subject_stats_[subj][col]['mean']
                    s = self.train_subject_stats_[subj][col]['std']
                else:
                    # Test subject: Honest Option A - use their resting baseline if mapped, else global train stats
                    rest_col = REST_BASELINE_MAPPING.get(col, None)
                    if rest_col is not None and rest_col in df_input.columns and not pd.isna(row[rest_col]):
                        m = float(row[rest_col])
                        s = self.global_train_stats_[col]['std']
                    else:
                        m = self.global_train_stats_[col]['mean']
                        s = self.global_train_stats_[col]['std']

                z_vals[i] = (raw_val - m) / (s if s > self.eps else 1.0)

            df_out[f'{col}_zscore'] = z_vals

        return df_out
