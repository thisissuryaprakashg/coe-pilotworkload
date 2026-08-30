"""
Phase 1 Runner: Subject-Relative Normalization on Existing Models
=================================================================
Runs the SAME 4 models (GBDT, RF, SVM, Logistic Regression) from the baseline,
but with subject-relative z-scored features and cross-modal interactions.

Protocol:
    - LOSO cross-validation (34-35 folds, no subject leakage)
    - Subject-relative z-scoring is unsupervised (no label info), safe for LOSO
    - Reports per-model accuracy before/after normalization

Output:
    results/phase1_per_fold_results.csv
    results/phase1_per_subject_accuracy.csv
    results/phase1_summary.txt
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from subject_relative_transform import build_enhanced_feature_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    precision_score, recall_score, balanced_accuracy_score,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_models():
    """Return the SAME model configurations used in the baseline."""
    return {
        'Math_Model (Logistic Regression)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=1000, C=1.0, random_state=42
            ))
        ]),
        'ML_Model (Random Forest)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
            ))
        ]),
        'ML_Model (SVM RBF)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', CalibratedClassifierCV(
                SVC(kernel='rbf', C=1.0, random_state=42), ensemble=False
            ))
        ]),
        'ML_Model (GBDT)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42
            ))
        ]),
    }


def load_dataset():
    """Load dataset and apply subject-relative normalization."""
    workspace_root = Path(__file__).resolve().parents[3]

    # Try multiple candidate paths
    candidates = [
        workspace_root / 'data' / 'processed' / 'cleaned_multimodal_dataset.csv',
        workspace_root / 'cleaned_multimodal_dataset.csv',
    ]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        matches = list(workspace_root.rglob('cleaned_multimodal_dataset.csv'))
        if matches:
            data_path = matches[0]
        else:
            raise FileNotFoundError("cleaned_multimodal_dataset.csv not found")

    logger.info(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Apply subject-relative normalization + interactions
    df_enhanced, feature_groups = build_enhanced_feature_matrix(df, subject_col='subject')

    return df_enhanced, feature_groups


def run_loso_for_model(df, feature_cols, model_pipeline, model_name,
                       target_col='difficulty_ground_truth', subject_col='subject'):
    """
    Run LOSO cross-validation for a single model on given features.

    Returns:
        fold_results: list of per-fold metric dicts
        all_y_true, all_y_pred: aggregated predictions
    """
    subjects = df[subject_col].unique()
    fold_results = []
    all_y_true = []
    all_y_pred = []

    for subj in subjects:
        train_mask = df[subject_col] != subj
        test_mask = df[subject_col] == subj

        X_train = df.loc[train_mask, feature_cols].astype(float)
        y_train = df.loc[train_mask, target_col].astype(int)
        X_test = df.loc[test_mask, feature_cols].astype(float)
        y_test = df.loc[test_mask, target_col].astype(int)

        if len(X_test) == 0:
            continue

        try:
            # Clone the pipeline for each fold to avoid state leakage
            from sklearn.base import clone
            pipe = clone(model_pipeline)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            fold_results.append({
                'subject': str(subj),
                'model': model_name,
                'accuracy': acc,
                'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'kappa': cohen_kappa_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'n_test': len(y_test),
            })

            all_y_true.extend(y_test.values)
            all_y_pred.extend(y_pred)

        except Exception as e:
            logger.warning(f"  {model_name} failed on subject {subj}: {e}")
            continue

    return fold_results, np.array(all_y_true), np.array(all_y_pred)


def run_phase1():
    """Run all existing models on raw vs enhanced features."""
    logger.info("=" * 80)
    logger.info("PHASE 1: Subject-Relative Normalization on Existing Models")
    logger.info("=" * 80)

    df_enhanced, feature_groups = load_dataset()
    models = get_models()

    all_results = []

    # Feature sets to evaluate
    feature_sets = {
        'raw_only (baseline)': feature_groups['raw_only'],
        'raw + zscore': feature_groups['raw_plus_zscore'],
        'full_enhanced (raw + zscore + interactions)': feature_groups['full_enhanced'],
    }

    for feat_name, feat_cols in feature_sets.items():
        valid_cols = [c for c in feat_cols if c in df_enhanced.columns]
        logger.info(f"\n--- Feature Set: {feat_name} ({len(valid_cols)} features) ---")

        for model_name, model_pipe in models.items():
            logger.info(f"  Running {model_name}...")

            fold_results, y_true, y_pred = run_loso_for_model(
                df_enhanced, valid_cols, model_pipe, model_name
            )

            if len(fold_results) == 0:
                continue

            overall_acc = np.mean([r['accuracy'] for r in fold_results])
            overall_f1 = np.mean([r['macro_f1'] for r in fold_results])
            overall_kappa = np.mean([r['kappa'] for r in fold_results])

            logger.info(f"    Accuracy: {overall_acc:.4f} | F1: {overall_f1:.4f} | Kappa: {overall_kappa:.4f}")

            for r in fold_results:
                r['feature_set'] = feat_name
                r['n_features'] = len(valid_cols)
                all_results.append(r)

    return pd.DataFrame(all_results), df_enhanced, feature_groups


def save_results(results_df, output_dir):
    """Save Phase 1 results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold results
    results_csv = output_dir / 'phase1_per_fold_results.csv'
    results_df.to_csv(results_csv, index=False)
    logger.info(f"✓ Saved per-fold results: {results_csv}")

    # Per-model summary
    summary_data = []
    for (feat_set, model), group in results_df.groupby(['feature_set', 'model']):
        summary_data.append({
            'feature_set': feat_set,
            'model': model,
            'mean_accuracy': group['accuracy'].mean(),
            'std_accuracy': group['accuracy'].std(),
            'min_accuracy': group['accuracy'].min(),
            'max_accuracy': group['accuracy'].max(),
            'mean_macro_f1': group['macro_f1'].mean(),
            'mean_kappa': group['kappa'].mean(),
            'mean_balanced_acc': group['balanced_accuracy'].mean(),
            'n_folds': len(group),
        })

    summary_df = pd.DataFrame(summary_data).sort_values(
        ['feature_set', 'mean_accuracy'], ascending=[True, False]
    )
    summary_csv = output_dir / 'phase1_model_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"✓ Saved model summary: {summary_csv}")

    # Per-subject accuracy (best model on best feature set)
    best_config = summary_df.loc[summary_df['mean_accuracy'].idxmax()]
    best_feat = best_config['feature_set']
    best_model = best_config['model']

    best_results = results_df[
        (results_df['feature_set'] == best_feat) &
        (results_df['model'] == best_model)
    ][['subject', 'accuracy']].sort_values('accuracy')

    subj_csv = output_dir / 'phase1_per_subject_accuracy.csv'
    best_results.to_csv(subj_csv, index=False)
    logger.info(f"✓ Saved per-subject results: {subj_csv}")

    # Text summary
    summary_txt = output_dir / 'phase1_summary.txt'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 1: SUBJECT-RELATIVE NORMALIZATION - RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("TECHNIQUE: Within-subject z-score centering + cross-modal interactions\n")
        f.write("MODELS: Same GBDT, RF, SVM, LR as baseline (no model changes)\n")
        f.write("VALIDATION: LOSO (Leave-One-Subject-Out, no data leakage)\n\n")

        f.write("RESULTS BY FEATURE SET & MODEL:\n")
        f.write("-" * 80 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"  {row['feature_set']} | {row['model']}\n")
            f.write(f"    Accuracy: {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}\n")
            f.write(f"    Macro F1: {row['mean_macro_f1']:.4f} | Kappa: {row['mean_kappa']:.4f}\n")
            f.write(f"    Range: {row['min_accuracy']:.4f} - {row['max_accuracy']:.4f}\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"\nBEST CONFIGURATION:\n")
        f.write(f"  Model: {best_model}\n")
        f.write(f"  Feature Set: {best_feat}\n")
        f.write(f"  Accuracy: {best_config['mean_accuracy']:.4f}\n")

        # Compare baseline raw_only models
        baseline_rows = summary_df[summary_df['feature_set'].str.contains('baseline')]
        if not baseline_rows.empty:
            baseline_best = baseline_rows['mean_accuracy'].max()
            improvement = best_config['mean_accuracy'] - baseline_best
            f.write(f"\n  Baseline Best: {baseline_best:.4f}\n")
            f.write(f"  Improvement: +{improvement:.4f} (+{improvement*100:.2f}%)\n")

    logger.info(f"✓ Saved summary: {summary_txt}")

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: RESULTS COMPARISON")
    logger.info("=" * 80)

    # Pivot: rows=models, columns=feature_sets
    pivot = summary_df.pivot(index='model', columns='feature_set', values='mean_accuracy')
    logger.info("\nAccuracy by Model × Feature Set:")
    logger.info(pivot.to_string(float_format='{:.4f}'.format))
    logger.info("=" * 80)

    return summary_df


def main():
    results_df, df_enhanced, feature_groups = run_phase1()
    output_dir = Path(__file__).parent / 'results'
    summary_df = save_results(results_df, output_dir)

    logger.info("\n✓ Phase 1 complete!")
    logger.info("Next: Run Phase 2 (Calibrated Stacked Generalization)")


if __name__ == '__main__':
    main()
