"""
explore_tda.py
==============
Quick ANOVA sanity check: do TDA features separate by difficulty level?
If eta-squared is significant, the features are worth wiring into modeling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import f_oneway, kruskal

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / 'outputs' / 'tda_features.csv'


def main():
    df = pd.read_csv(INPUT_CSV)

    print("=" * 70)
    print("TDA FEATURE EXPLORATION")
    print("=" * 70)
    print(f"rows: {len(df)}")

    meta = {'subject', 'session', 'run_folder', 'window', 'difficulty'}
    feat_cols = [c for c in df.columns if c not in meta]
    print(f"TDA features: {len(feat_cols)}")
    print(f"difficulty levels: {sorted(df['difficulty'].unique().tolist())}")

    # NaN check
    nan_pct = df[feat_cols].isna().mean()
    bad = [c for c in feat_cols if nan_pct[c] > 0.30]
    if bad:
        print(f"\nfeatures with >30% NaN (will skip): {len(bad)}")
        for c in bad:
            print(f"  {c:40s} {nan_pct[c] * 100:.1f}% NaN")
    keep = [c for c in feat_cols if nan_pct[c] <= 0.30]
    print(f"features retained: {len(keep)}")

    # impute median
    for c in keep:
        df[c] = df[c].fillna(df[c].median())

    # ANOVA + Kruskal-Wallis
    groups = [df.loc[df['difficulty'] == d] for d in [1, 2, 3, 4]]
    results = []

    for feat in keep:
        try:
            F, p_anova = f_oneway(*[g[feat].values for g in groups])
            _, p_kruskal = kruskal(*[g[feat].values for g in groups])
        except Exception:
            continue

        # eta-squared
        grand_mean = df[feat].mean()
        ss_between = sum(len(g) * (g[feat].mean() - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((df[feat].values - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        # direction: correlation with difficulty
        r = np.corrcoef(df['difficulty'].values, df[feat].values)[0, 1]

        results.append({
            'feature': feat,
            'F': F, 'p_anova': p_anova, 'p_kruskal': p_kruskal,
            'eta_sq': eta_sq, 'corr': r,
            'L1_mean': groups[0][feat].mean(),
            'L4_mean': groups[3][feat].mean(),
        })

    res = pd.DataFrame(results).sort_values('eta_sq', ascending=False)

    # significant features
    sig = res[res['p_anova'] < 0.05]
    nsig = res[res['p_anova'] >= 0.05]

    print(f"\n{'=' * 70}")
    print(f"SIGNIFICANT FEATURES (p < 0.05): {len(sig)} / {len(res)}")
    print(f"{'=' * 70}")
    print(f"\n{'feature':40s} {'eta_sq':>8s} {'p_anova':>10s} {'corr':>7s} {'L1_mean':>10s} {'L4_mean':>10s}")
    print("-" * 80)
    for _, row in sig.iterrows():
        direction = "+" if row['corr'] > 0 else "-"
        print(f"{row['feature']:40s} {row['eta_sq']:8.4f} {row['p_anova']:10.2e} "
              f"{row['corr']:+7.3f} {row['L1_mean']:10.3f} {row['L4_mean']:10.3f}")

    print(f"\n{'=' * 70}")
    print(f"NON-SIGNIFICANT (p >= 0.05): {len(nsig)} / {len(res)}")
    print(f"{'=' * 70}")
    for _, row in nsig.iterrows():
        print(f"  {row['feature']:40s} eta_sq={row['eta_sq']:.4f} p={row['p_anova']:.3f}")

    # comparison with conventional features
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH CONVENTIONAL FEATURES")
    print(f"{'=' * 70}")
    best_tda = sig.iloc[0] if len(sig) > 0 else None
    if best_tda is not None:
        print(f"best TDA feature: {best_tda['feature']} (eta_sq={best_tda['eta_sq']:.4f})")
        print(f"for reference: best conventional = overall_gaze_entropy_RY (eta_sq=0.0728)")
        ratio = best_tda['eta_sq'] / 0.0728 * 100
        print(f"TDA best = {ratio:.1f}% of conventional best")
    else:
        print("no significant TDA features found")

    # group means for top features
    if len(sig) > 0:
        print(f"\n{'=' * 70}")
        print("TOP 10 TDA FEATURES — PER-LEVEL MEANS")
        print(f"{'=' * 70}")
        top = sig.head(10)
        print(f"\n{'feature':40s} {'L1':>8s} {'L2':>8s} {'L3':>8s} {'L4':>8s} {'trend':>6s}")
        print("-" * 75)
        for _, row in top.iterrows():
            vals = [groups[i][row['feature']].mean() for i in range(4)]
            trend = "+" if row['corr'] > 0 else "-"
            print(f"{row['feature']:40s} {vals[0]:8.3f} {vals[1]:8.3f} {vals[2]:8.3f} {vals[3]:8.3f} {trend:>6s}")


if __name__ == '__main__':
    main()
