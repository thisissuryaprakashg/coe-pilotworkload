"""
Phase 2 Runner: Calibrated Stacked Generalization
===================================================
Combines all existing models' probability outputs via meta-learning.

Uses Phase 1 enhanced features as one of the feature subsets,
so Phase 2 builds on top of Phase 1.

Output:
    results/phase2_per_fold_results.csv
    results/phase2_summary.txt
"""

import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1_subject_normalization'))

from stacking_meta_learner import run_stacked_loso
from subject_relative_transform import build_enhanced_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset():
    """Load dataset with subject-relative normalization (from Phase 1)."""
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


def main():
    logger.info("=" * 80)
    logger.info("PHASE 2: Calibrated Stacked Generalization")
    logger.info("=" * 80)

    df_enhanced, feature_groups = load_dataset()

    # Feature subsets for base models — stacking uses BOTH raw and enhanced
    stacking_feature_sets = {
        'raw': feature_groups['raw_only'],
        'enhanced': feature_groups['full_enhanced'],
    }

    # Run stacked LOSO
    fold_results, y_true, y_pred = run_stacked_loso(
        df_enhanced, stacking_feature_sets
    )

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(fold_results)
    results_csv = output_dir / 'phase2_per_fold_results.csv'
    results_df.to_csv(results_csv, index=False)
    logger.info(f"✓ Saved: {results_csv}")

    # Summary
    mean_acc = results_df['accuracy'].mean()
    std_acc = results_df['accuracy'].std()
    mean_f1 = results_df['macro_f1'].mean()
    mean_kappa = results_df['kappa'].mean()

    summary_txt = output_dir / 'phase2_summary.txt'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 2: CALIBRATED STACKED GENERALIZATION - RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("TECHNIQUE: Meta-learning over 4 models × 2 feature sets = 8 base learners\n")
        f.write("META-LEARNER: Logistic Regression on 32 stacking features\n\n")
        f.write(f"Overall Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%) ± {std_acc:.4f}\n")
        f.write(f"Macro F1: {mean_f1:.4f}\n")
        f.write(f"Cohen's Kappa: {mean_kappa:.4f}\n")
        f.write(f"Per-Subject Range: {results_df['accuracy'].min():.4f} - {results_df['accuracy'].max():.4f}\n\n")
        f.write(f"Baseline: 40.66%\n")
        f.write(f"Improvement: +{(mean_acc*100 - 40.66):.2f}%\n")

    logger.info(f"✓ Saved: {summary_txt}")

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 SUMMARY")
    logger.info(f"  Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%) ± {std_acc:.4f}")
    logger.info(f"  Macro F1: {mean_f1:.4f}")
    logger.info(f"  Kappa: {mean_kappa:.4f}")
    logger.info(f"  Improvement over baseline: +{(mean_acc*100 - 40.66):.2f}%")
    logger.info("=" * 80)
    logger.info("\n✓ Phase 2 complete!")
    logger.info("Next: Run Phase 3 (Ordinal-Aware Post-Processing)")


if __name__ == '__main__':
    main()
