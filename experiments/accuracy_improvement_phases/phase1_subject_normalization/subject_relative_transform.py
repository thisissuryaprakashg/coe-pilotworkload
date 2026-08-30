"""
Subject-Relative Physiological Normalization
=============================================
Novel technique: Within-subject z-score centering for workload prediction.

Problem:
    Each pilot has a different physiological baseline. Pilot A rests at HR=55,
    Pilot B rests at HR=95. A raw HR=85 means "highly stressed" for A but
    "relaxed" for B. Standard across-population StandardScaler cannot fix this.

Solution:
    Z-score each subject's features against THEIR OWN mean/std across all runs.
    This transforms raw values into "deviations from personal baseline":
        z_HR = (HR - mean_HR_for_this_subject) / std_HR_for_this_subject

    Now z_HR=+1.5 means "1.5 std above MY baseline" regardless of who the pilot is.

    This is an unsupervised transform (uses no labels), so it's safe for LOSO.

Additionally adds cross-modal interaction features:
    - pupil_diam_std * ecg_hr_std      (cognitive-autonomic coupling)
    - gaze_entropy_y * eda_tonic_mean  (attention-arousal interaction)
    - emg_flexor_rms * emg_extensor_rms (bilateral muscle co-activation)

References:
    - Beatty & Lucero-Wagoner (2000): Pupillometric baseline normalization
    - Fairclough (2009): Individual differences in physiological stress response
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


# Features that benefit most from subject-relative normalization
# (high between-subject variance, low within-subject variance)
PHYSIO_FEATURES_TO_NORMALIZE = [
    # ECG / cardiac
    'ecg_hr_mean', 'ecg_hr_std', 'ecg_rr_mean_ms', 'ecg_rr_sdnn_ms',
    'ecg_rr_rmssd_ms', 'ecg_rr_pnn50', 'ecg_rr_lf_power', 'ecg_rr_hf_power',
    'ecg_rr_lf_hf_ratio',
    # EDA / electrodermal
    'eda_conductance_mean_uS', 'eda_conductance_std_uS', 'eda_tonic_mean_uS',
    'eda_phasic_peak_rate_per_min', 'eda_phasic_energy',
    # EMG / muscle
    'emg_flexor_rms_mV', 'emg_extensor_rms_mV', 'emg_forearm_motion_energy',
    # Pupil / oculomotor
    'pupil_diam_mean_mm', 'pupil_diam_std_mm', 'pupil_diam_skew', 'pupil_diam_kurt',
    # Gaze
    'gaze_entropy_x', 'gaze_entropy_y',
    # Flight performance
    'cumulative_glideslope_error_deg', 'cumulative_localizer_error_deg',
    'cumulative_airspeed_error_kts', 'cumulative_total_error',
]

# Cross-modal interaction features to create
INTERACTION_PAIRS = [
    # Cognitive-autonomic coupling: pupil changes * heart rate variability
    ('pupil_diam_std_mm', 'ecg_hr_std', 'pupil_x_hr_coupling'),
    # Attention-arousal: gaze dispersion * electrodermal arousal
    ('gaze_entropy_y', 'eda_tonic_mean_uS', 'gaze_x_eda_interaction'),
    # Bilateral muscle tension: co-activation pattern
    ('emg_flexor_rms_mV', 'emg_extensor_rms_mV', 'emg_coactivation'),
    # Cognitive load proxy: pupil variability * gaze complexity
    ('pupil_diam_std_mm', 'gaze_entropy_y', 'cognitive_load_proxy'),
    # Stress composite: heart rate * electrodermal activity
    ('ecg_hr_mean', 'eda_conductance_mean_uS', 'stress_composite'),
    # Flight difficulty proxy: total error * HR (effort under difficulty)
    ('cumulative_total_error', 'ecg_hr_mean', 'effort_x_difficulty'),
]


def subject_relative_zscore(df, feature_cols, subject_col='subject'):
    """
    Apply within-subject z-score normalization.

    For each subject, computes mean and std of each feature across all their
    runs (regardless of difficulty level), then z-scores.

    Args:
        df: DataFrame with subject column and feature columns
        feature_cols: list of feature column names to normalize
        subject_col: name of the subject identifier column

    Returns:
        DataFrame with z-scored features (same column names, '_zscore' suffix)
    """
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        logger.warning("No matching feature columns found for normalization")
        return df

    df_out = df.copy()

    for col in available_cols:
        zscore_col = f'{col}_zscore'
        df_out[zscore_col] = np.nan

        for subject in df[subject_col].unique():
            mask = df[subject_col] == subject
            values = df.loc[mask, col].values.astype(float)

            # Need at least 2 samples for std
            if np.sum(~np.isnan(values)) < 2:
                df_out.loc[mask, zscore_col] = 0.0
                continue

            subj_mean = np.nanmean(values)
            subj_std = np.nanstd(values)

            if subj_std < 1e-10:
                # No variance for this subject on this feature
                df_out.loc[mask, zscore_col] = 0.0
            else:
                df_out.loc[mask, zscore_col] = (values - subj_mean) / subj_std

    return df_out


def add_interaction_features(df):
    """
    Create cross-modal physiological interaction features.

    These capture non-linear relationships between different modalities
    (e.g., cognitive load shows as BOTH pupil dilation AND heart rate increase,
    but their product is more discriminative than either alone).

    Args:
        df: DataFrame with raw physiological features

    Returns:
        DataFrame with additional interaction columns
    """
    df_out = df.copy()

    for feat_a, feat_b, name in INTERACTION_PAIRS:
        if feat_a in df.columns and feat_b in df.columns:
            a = df[feat_a].astype(float).values
            b = df[feat_b].astype(float).values
            df_out[name] = a * b
            logger.debug(f"Created interaction feature: {name} = {feat_a} × {feat_b}")
        else:
            logger.debug(f"Skipping interaction {name}: missing {feat_a} or {feat_b}")

    return df_out


def build_enhanced_feature_matrix(df, subject_col='subject'):
    """
    Full pipeline: raw features → subject-normalized + interactions.

    Returns:
        df_enhanced: DataFrame with original + zscore + interaction features
        feature_groups: dict mapping group names to column lists
    """
    logger.info("Building enhanced feature matrix with subject-relative normalization...")

    # Step 1: Add interaction features (on raw data)
    df_enhanced = add_interaction_features(df)

    # Step 2: Apply subject-relative z-scoring
    df_enhanced = subject_relative_zscore(
        df_enhanced,
        PHYSIO_FEATURES_TO_NORMALIZE,
        subject_col
    )

    # Also z-score the interaction features
    interaction_cols = [name for _, _, name in INTERACTION_PAIRS if name in df_enhanced.columns]
    df_enhanced = subject_relative_zscore(
        df_enhanced,
        interaction_cols,
        subject_col
    )

    # Build feature groups for ablation
    excluded = {
        'subject', 'session', 'run_folder', 'run_number',
        'difficulty_level', 'difficulty_ground_truth',
        'is_eye_analyzed', 'ecg_valid', 'eda_valid', 'emg_valid',
        'eye_valid', 'gaze_valid_sample_fraction', 'ecg_num_clean_beats',
    }

    # Original raw features (same as baseline)
    raw_feature_cols = [
        col for col in df.select_dtypes(include=['number', 'bool']).columns
        if col not in excluded
    ]

    # Z-scored features
    zscore_cols = [c for c in df_enhanced.columns if c.endswith('_zscore')]

    # Interaction features
    interaction_feature_cols = [
        name for _, _, name in INTERACTION_PAIRS
        if name in df_enhanced.columns
    ]

    # Combined: raw + zscore + interactions
    all_enhanced_cols = raw_feature_cols + zscore_cols + interaction_feature_cols

    feature_groups = {
        'raw_only': raw_feature_cols,
        'zscore_only': zscore_cols,
        'interactions_only': interaction_feature_cols,
        'raw_plus_zscore': raw_feature_cols + zscore_cols,
        'full_enhanced': all_enhanced_cols,
    }

    logger.info(f"  Raw features: {len(raw_feature_cols)}")
    logger.info(f"  Z-scored features: {len(zscore_cols)}")
    logger.info(f"  Interaction features: {len(interaction_feature_cols)}")
    logger.info(f"  Total enhanced: {len(all_enhanced_cols)}")

    return df_enhanced, feature_groups


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Subject-Relative Transform module loaded.")
    print("Use in run_phase1.py to apply normalization to existing models.")
