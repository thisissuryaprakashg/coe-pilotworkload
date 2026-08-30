"""
Comparison Script: Baseline vs All Phases
==========================================
Loads results from all phases and generates a comprehensive comparison.

Output:
    results/comparison_all_phases.txt
    results/comparison_all_phases.csv
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASELINE_ACCURACY = 0.4066
BASELINE_F1 = 0.405
BASELINE_KAPPA = 0.21


def load_phase_results(phase_dir, result_filename):
    """Load per-fold results CSV from a phase directory."""
    csv_path = phase_dir / 'results' / result_filename
    if not csv_path.exists():
        logger.warning(f"Results not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)


def main():
    phases_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all available phase results
    phases = {}

    # Phase 1
    phase1_df = load_phase_results(
        phases_dir / 'phase1_subject_normalization', 'phase1_per_fold_results.csv'
    )
    if phase1_df is not None:
        # Get best model on best feature set
        best_per_config = phase1_df.groupby(['feature_set', 'model']).agg(
            mean_acc=('accuracy', 'mean'),
            mean_f1=('macro_f1', 'mean'),
            mean_kappa=('kappa', 'mean'),
        ).reset_index()
        best_row = best_per_config.loc[best_per_config['mean_acc'].idxmax()]

        # Get fold-level results for that best config
        best_mask = (
            (phase1_df['feature_set'] == best_row['feature_set']) &
            (phase1_df['model'] == best_row['model'])
        )
        best_phase1 = phase1_df[best_mask]

        phases['Phase 1 (Subject Normalization)'] = {
            'df': best_phase1,
            'accuracy': best_row['mean_acc'],
            'f1': best_row['mean_f1'],
            'kappa': best_row['mean_kappa'],
            'details': f"{best_row['model']} on {best_row['feature_set']}",
        }

        # Also show all model comparisons
        phases['_phase1_all'] = best_per_config

    # Phase 2
    phase2_df = load_phase_results(
        phases_dir / 'phase2_stacked_generalization', 'phase2_per_fold_results.csv'
    )
    if phase2_df is not None:
        phases['Phase 2 (Stacked Generalization)'] = {
            'df': phase2_df,
            'accuracy': phase2_df['accuracy'].mean(),
            'f1': phase2_df['macro_f1'].mean(),
            'kappa': phase2_df['kappa'].mean(),
            'details': 'Meta-learner over 4 models × 2 feature sets',
        }

    # Phase 3
    phase3_df = load_phase_results(
        phases_dir / 'phase3_ordinal_postprocessing', 'phase3_per_fold_results.csv'
    )
    if phase3_df is not None:
        phases['Phase 3 (Ordinal Post-Processing)'] = {
            'df': phase3_df,
            'accuracy': phase3_df['accuracy'].mean(),
            'f1': phase3_df['macro_f1'].mean(),
            'kappa': phase3_df['kappa'].mean(),
            'details': 'Ordinal smoothing + threshold tuning on all models',
        }

    # Generate comparison report
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("ACCURACY IMPROVEMENT: COMPREHENSIVE PHASE COMPARISON")
    report_lines.append("=" * 90)
    report_lines.append("")
    report_lines.append(f"{'Phase':<45} {'Accuracy':>10} {'Macro F1':>10} {'Kappa':>10} {'Δ Acc':>10}")
    report_lines.append("-" * 90)
    report_lines.append(
        f"{'Baseline (GBDT Oculomotor)':<45} {BASELINE_ACCURACY:>10.4f} {BASELINE_F1:>10.4f} {BASELINE_KAPPA:>10.4f} {'---':>10}"
    )

    comparison_data = [{
        'Phase': 'Baseline',
        'Accuracy': BASELINE_ACCURACY,
        'Macro_F1': BASELINE_F1,
        'Kappa': BASELINE_KAPPA,
        'Improvement': 0,
    }]

    for phase_name, phase_info in phases.items():
        if phase_name.startswith('_'):
            continue

        delta = phase_info['accuracy'] - BASELINE_ACCURACY
        report_lines.append(
            f"{phase_name:<45} {phase_info['accuracy']:>10.4f} {phase_info['f1']:>10.4f} "
            f"{phase_info['kappa']:>10.4f} {delta:>+10.4f}"
        )

        comparison_data.append({
            'Phase': phase_name,
            'Accuracy': phase_info['accuracy'],
            'Macro_F1': phase_info['f1'],
            'Kappa': phase_info['kappa'],
            'Improvement': delta,
            'Details': phase_info['details'],
        })

    report_lines.append("-" * 90)
    report_lines.append("")

    # Phase 1 detailed model breakdown
    if '_phase1_all' in phases:
        report_lines.append("PHASE 1 DETAILED: All Models × Feature Sets")
        report_lines.append("-" * 90)
        p1_all = phases['_phase1_all'].sort_values('mean_acc', ascending=False)
        report_lines.append(
            f"{'Model':<40} {'Feature Set':<35} {'Accuracy':>10} {'F1':>8} {'Kappa':>8}"
        )
        for _, row in p1_all.iterrows():
            report_lines.append(
                f"{row['model']:<40} {row['feature_set']:<35} "
                f"{row['mean_acc']:>10.4f} {row['mean_f1']:>8.4f} {row['mean_kappa']:>8.4f}"
            )
        report_lines.append("")

    # Per-subject comparison (if Phase 3 available)
    if 'Phase 3 (Ordinal Post-Processing)' in phases:
        p3_df = phases['Phase 3 (Ordinal Post-Processing)']['df']
        report_lines.append("PER-SUBJECT ACCURACY (Phase 3 - Best Configuration)")
        report_lines.append("-" * 90)
        for _, row in p3_df.sort_values('accuracy').iterrows():
            bar = "█" * int(row['accuracy'] * 40)
            report_lines.append(f"  {row['subject']:<15} {row['accuracy']:.4f} {bar}")
        report_lines.append("")

    report_lines.append("=" * 90)

    report = "\n".join(report_lines)
    print(report)

    # Save text report
    report_path = output_dir / 'comparison_all_phases.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"✓ Saved: {report_path}")

    # Save CSV comparison
    comp_df = pd.DataFrame(comparison_data)
    comp_csv = output_dir / 'comparison_all_phases.csv'
    comp_df.to_csv(comp_csv, index=False)
    logger.info(f"✓ Saved: {comp_csv}")


if __name__ == '__main__':
    main()
