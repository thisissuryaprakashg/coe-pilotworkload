"""
Phase 3 Runner: Ordinal-Aware Post-Processing
===============================================
Applies ordinal post-processing on top of the existing models.

Combines:
    - Phase 1: Subject-relative normalized features
    - Phase 3: Ordinal smoothing + threshold tuning on model outputs

Output:
    results/phase3_per_fold_results.csv
    results/phase3_summary.txt
"""

import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1_subject_normalization'))

from ordinal_threshold_tuner import OrdinalEnsemblePostProcessor
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
    balanced_accuracy_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_base_models():
    """Same model configurations as baseline."""
    return {
        'LR': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=1000, C=1.0, random_state=42
            ))
        ]),
        'RF': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
            ))
        ]),
        'SVM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', CalibratedClassifierCV(
                SVC(kernel='rbf', C=1.0, random_state=42), ensemble=False
            ))
        ]),
        'GBDT': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42
            ))
        ]),
    }


def load_dataset():
    """Load and enhance dataset."""
    workspace_root = Path(__file__).resolve().parents[3]
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
    df_enhanced, feature_groups = build_enhanced_feature_matrix(df)
    return df_enhanced, feature_groups


def run_ordinal_loso(df, feature_groups, target_col='difficulty_ground_truth',
                     subject_col='subject'):
    """
    Run LOSO with ordinal post-processing on all base models.
    """
    subjects = df[subject_col].unique()
    base_models = get_base_models()

    feature_sets = {
        'raw': feature_groups['raw_only'],
        'enhanced': feature_groups['full_enhanced'],
    }

    fold_results = []
    all_y_true = []
    all_y_pred = []

    logger.info(f"Running ordinal post-processing LOSO ({len(subjects)} folds)...")

    for fold_num, test_subject in enumerate(subjects, 1):
        train_mask = df[subject_col] != test_subject
        test_mask = df[subject_col] == test_subject

        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)

        y_train = df_train[target_col].astype(int).values
        y_test = df_test[target_col].astype(int).values

        if len(y_test) == 0:
            continue

        # Create ordinal ensemble with post-processing
        ordinal_ensemble = OrdinalEnsemblePostProcessor(
            base_models=base_models,
            feature_sets=feature_sets,
            n_classes=4,
            smoothing_sigma=0.8,
            optimize_thresholds=True,
        )

        # Fit on training data
        ordinal_ensemble.fit(df_train, y_train)

        # Predict on test data
        y_pred, probs = ordinal_ensemble.predict(df_test)

        acc = accuracy_score(y_test, y_pred)
        fold_results.append({
            'subject': str(test_subject),
            'accuracy': acc,
            'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'n_test': len(y_test),
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        if fold_num % 5 == 0 or fold_num == len(subjects):
            avg_acc = np.mean([r['accuracy'] for r in fold_results])
            logger.info(f"  Fold {fold_num}/{len(subjects)} - Running avg: {avg_acc:.4f}")

    return fold_results, np.array(all_y_true), np.array(all_y_pred)


def main():
    logger.info("=" * 80)
    logger.info("PHASE 3: Ordinal-Aware Post-Processing on Existing Models")
    logger.info("=" * 80)

    df_enhanced, feature_groups = load_dataset()

    fold_results, y_true, y_pred = run_ordinal_loso(df_enhanced, feature_groups)

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(fold_results)
    results_csv = output_dir / 'phase3_per_fold_results.csv'
    results_df.to_csv(results_csv, index=False)
    logger.info(f"✓ Saved: {results_csv}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, labels=[1, 2, 3, 4], average=None, zero_division=0)

    mean_acc = results_df['accuracy'].mean()
    std_acc = results_df['accuracy'].std()
    mean_f1 = results_df['macro_f1'].mean()
    mean_kappa = results_df['kappa'].mean()

    summary_txt = output_dir / 'phase3_summary.txt'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 3: ORDINAL-AWARE POST-PROCESSING - RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("TECHNIQUE: Ordinal Gaussian smoothing + cost-sensitive threshold tuning\n")
        f.write("BASE MODELS: All 4 existing models (LR, RF, SVM, GBDT)\n")
        f.write("FEATURES: Raw + Subject-Relative Enhanced (from Phase 1)\n\n")
        f.write(f"Overall Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%) ± {std_acc:.4f}\n")
        f.write(f"Macro F1: {mean_f1:.4f}\n")
        f.write(f"Cohen's Kappa: {mean_kappa:.4f}\n")
        f.write(f"Balanced Accuracy: {results_df['balanced_accuracy'].mean():.4f}\n")
        f.write(f"Per-Subject Range: {results_df['accuracy'].min():.4f} - {results_df['accuracy'].max():.4f}\n\n")

        f.write(f"Per-Class F1:\n")
        for i, f1 in enumerate(per_class_f1, 1):
            f.write(f"  Level {i}: {f1:.4f}\n")

        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  Predicted →  L1    L2    L3    L4\n")
        for i in range(4):
            f.write(f"  True L{i+1}:  {cm[i]}\n")

        f.write(f"\nBaseline: 40.66%\n")
        f.write(f"Improvement: +{(mean_acc*100 - 40.66):.2f}%\n")

    logger.info(f"✓ Saved: {summary_txt}")

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3 SUMMARY")
    logger.info(f"  Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%) ± {std_acc:.4f}")
    logger.info(f"  Macro F1: {mean_f1:.4f}")
    logger.info(f"  Kappa: {mean_kappa:.4f}")
    logger.info(f"  Per-class F1: {[f'{f:.3f}' for f in per_class_f1]}")
    logger.info(f"  Improvement over baseline: +{(mean_acc*100 - 40.66):.2f}%")
    logger.info("=" * 80)
    logger.info("\n✓ Phase 3 complete!")


if __name__ == '__main__':
    main()
