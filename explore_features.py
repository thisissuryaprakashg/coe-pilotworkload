"""
explore_features.py
===================
Feature quality analysis, statistical screening, correlation,
multicollinearity (VIF), and class separability analysis on the
windowed feature matrix from new_data_pipeline.py.

Run: python explore_features.py
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, spearmanr
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)

INPUT_CSV = Path(r"c:\coe_new\new_master_features.csv")
META_COLS = {'subject', 'session', 'run_folder', 'run_number', 'difficulty', 'window'}


def main():
    df = pd.read_csv(INPUT_CSV)
    print("=" * 70)
    print("FEATURE EXPLORATION")
    print("=" * 70)
    print(f"rows: {len(df)} | cols: {len(df.columns)}")

    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"feature columns: {len(feat_cols)}")
    print(f"difficulty levels: {sorted(df['difficulty'].unique())}")
    print(f"subjects: {df['subject'].nunique()}")

    # ==========================================================
    # 1. DATA QUALITY: NaN coverage + drop bad features
    # ==========================================================
    print("\n" + "=" * 70)
    print("1. DATA QUALITY (NaN analysis)")
    print("=" * 70)

    nan_pct = df[feat_cols].isna().mean() * 100
    print(f"\nfeatures with >30% NaN (will be dropped):")
    bad_feats = nan_pct[nan_pct > 30].sort_values(ascending=False)
    if len(bad_feats) > 0:
        for f, p in bad_feats.items():
            print(f"  {f:40s} {p:5.1f}% NaN")
    else:
        print("  (none)")

    # drop features with >30% NaN
    keep_feats = [c for c in feat_cols if nan_pct[c] <= 30]
    print(f"\nfeatures retained: {len(keep_feats)} / {len(feat_cols)}")

    # also check for zero-variance features
    stds = df[keep_feats].std()
    zero_var = stds[stds < 1e-10].index.tolist()
    if zero_var:
        print(f"zero-variance features dropped: {zero_var}")
        keep_feats = [c for c in keep_feats if c not in zero_var]

    print(f"final feature count: {len(keep_feats)}")

    # fill NaN with column median for analysis
    df_clean = df[['subject', 'difficulty'] + keep_feats].copy()
    for c in keep_feats:
        med = df_clean[c].median()
        df_clean[c] = df_clean[c].fillna(med)

    # ==========================================================
    # 2. ANOVA / KRUSKAL-WALLIS SCREENING
    # ==========================================================
    print("\n" + "=" * 70)
    print("2. STATISTICAL SCREENING (ANOVA + Kruskal-Wallis)")
    print("=" * 70)

    groups = [df_clean.loc[df_clean['difficulty'] == d] for d in [1, 2, 3, 4]]
    results = []

    for feat in keep_feats:
        g = [grp[feat].values for grp in groups]

        # ANOVA
        try:
            f_stat, p_anova = f_oneway(*g)
        except Exception:
            f_stat, p_anova = np.nan, 1.0

        # Kruskal-Wallis (nonparametric)
        try:
            h_stat, p_kw = kruskal(*g)
        except Exception:
            h_stat, p_kw = np.nan, 1.0

        # Eta-squared (effect size)
        grand_mean = df_clean[feat].mean()
        ss_between = sum(len(grp) * (grp[feat].mean() - grand_mean) ** 2 for grp in groups)
        ss_total = np.sum((df_clean[feat].values - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        # means per level
        means = [grp[feat].mean() for grp in groups]
        monotonic = all(means[i] <= means[i + 1] for i in range(3)) or \
                    all(means[i] >= means[i + 1] for i in range(3))

        results.append({
            'feature': feat,
            'F_stat': f_stat,
            'p_anova': p_anova,
            'H_stat': h_stat,
            'p_kw': p_kw,
            'eta_sq': eta_sq,
            'mean_L1': means[0],
            'mean_L2': means[1],
            'mean_L3': means[2],
            'mean_L4': means[3],
            'monotonic': monotonic,
        })

    res_df = pd.DataFrame(results).sort_values('p_anova')

    # significant features (p < 0.05)
    sig_feats = res_df[res_df['p_anova'] < 0.05]
    nonsig_feats = res_df[res_df['p_anova'] >= 0.05]

    print(f"\nsignificant (p<0.05): {len(sig_feats)} / {len(keep_feats)}")
    print(f"\n{'feature':45s} {'F':>8s} {'p_anova':>10s} {'eta_sq':>8s} {'mono':>5s} {'L1':>8s} {'L2':>8s} {'L3':>8s} {'L4':>8s}")
    print("-" * 150)

    for _, r in sig_feats.iterrows():
        mono = "YES" if r['monotonic'] else ""
        print(f"{r['feature']:45s} {r['F_stat']:8.2f} {r['p_anova']:10.2e} {r['eta_sq']:8.4f} {mono:>5s} "
              f"{r['mean_L1']:8.3f} {r['mean_L2']:8.3f} {r['mean_L3']:8.3f} {r['mean_L4']:8.3f}")

    print(f"\nnon-significant features ({len(nonsig_feats)}):")
    for _, r in nonsig_feats.head(10).iterrows():
        print(f"  {r['feature']:45s} p={r['p_anova']:.4f}")
    if len(nonsig_feats) > 10:
        print(f"  ... and {len(nonsig_feats) - 10} more")

    # ==========================================================
    # 3. CORRELATION ANALYSIS
    # ==========================================================
    print("\n" + "=" * 70)
    print("3. CORRELATION ANALYSIS")
    print("=" * 70)

    sig_feat_names = sig_feats['feature'].tolist()
    if len(sig_feat_names) < 2:
        print("  not enough significant features for correlation analysis")
        return

    corr_matrix = df_clean[sig_feat_names].corr()

    # find highly correlated pairs (|r| > 0.85)
    high_corr = []
    for i in range(len(sig_feat_names)):
        for j in range(i + 1, len(sig_feat_names)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.85:
                high_corr.append((sig_feat_names[i], sig_feat_names[j], r))

    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\nhighly correlated pairs (|r| > 0.85): {len(high_corr)}")
    if high_corr:
        print(f"\n{'feature_1':40s} {'feature_2':40s} {'r':>8s}")
        print("-" * 90)
        for f1, f2, r in high_corr:
            print(f"{f1:40s} {f2:40s} {r:8.3f}")

    # moderate correlations (0.6 < |r| < 0.85)
    mod_corr = []
    for i in range(len(sig_feat_names)):
        for j in range(i + 1, len(sig_feat_names)):
            r = corr_matrix.iloc[i, j]
            if 0.6 < abs(r) <= 0.85:
                mod_corr.append((sig_feat_names[i], sig_feat_names[j], r))

    print(f"\nmoderate correlations (0.6 < |r| <= 0.85): {len(mod_corr)}")
    if mod_corr:
        mod_corr.sort(key=lambda x: abs(x[2]), reverse=True)
        for f1, f2, r in mod_corr[:15]:
            print(f"  {f1:40s} {f2:40s} {r:8.3f}")

    # ==========================================================
    # 4. VIF ANALYSIS (multicollinearity)
    # ==========================================================
    print("\n" + "=" * 70)
    print("4. VIF ANALYSIS (multicollinearity)")
    print("=" * 70)

    from numpy.linalg import LinAlgError

    # standardize
    X = df_clean[sig_feat_names].values
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_z = (X - X_mean) / X_std

    def compute_vif(X_z, col_names):
        vifs = {}
        for i, name in enumerate(col_names):
            others = np.delete(X_z, i, axis=1)
            y = X_z[:, i]
            try:
                # VIF = 1 / (1 - R^2) where R^2 from regressing feature on all others
                beta = np.linalg.lstsq(others, y, rcond=None)[0]
                y_hat = others @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                vif = 1.0 / (1.0 - r2) if r2 < 1.0 else 999.0
            except (LinAlgError, Exception):
                vif = 999.0
            vifs[name] = vif
        return vifs

    # iterative VIF pruning (remove highest VIF until all < 10)
    current_feats = sig_feat_names.copy()
    removed_vif = []

    for iteration in range(50):
        if len(current_feats) < 3:
            break
        X_curr = df_clean[current_feats].values
        X_mean = np.mean(X_curr, axis=0)
        X_std = np.std(X_curr, axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_z = (X_curr - X_mean) / X_std

        vifs = compute_vif(X_z, current_feats)
        max_vif_feat = max(vifs, key=vifs.get)
        max_vif = vifs[max_vif_feat]

        if max_vif <= 10.0:
            break

        removed_vif.append((max_vif_feat, max_vif))
        current_feats.remove(max_vif_feat)

    print(f"\nVIF pruning: {len(sig_feat_names)} -> {len(current_feats)} features")
    if removed_vif:
        print(f"\nremoved (VIF > 10):")
        for f, v in removed_vif:
            print(f"  {f:45s} VIF={v:.1f}")

    # final VIF values
    if len(current_feats) >= 3:
        X_curr = df_clean[current_feats].values
        X_mean = np.mean(X_curr, axis=0)
        X_std = np.std(X_curr, axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_z = (X_curr - X_mean) / X_std
        final_vifs = compute_vif(X_z, current_feats)

        print(f"\nfinal feature set ({len(current_feats)} features, all VIF <= 10):")
        print(f"\n{'feature':45s} {'VIF':>8s} {'p_anova':>10s} {'eta_sq':>8s} {'monotonic':>10s}")
        print("-" * 85)
        for f in current_feats:
            row = res_df[res_df['feature'] == f].iloc[0]
            mono = "YES" if row['monotonic'] else ""
            print(f"{f:45s} {final_vifs[f]:8.2f} {row['p_anova']:10.2e} {row['eta_sq']:8.4f} {mono:>10s}")

    # ==========================================================
    # 5. CLASS SEPARABILITY (mean per level for top features)
    # ==========================================================
    print("\n" + "=" * 70)
    print("5. CLASS SEPARABILITY (top features by eta-squared)")
    print("=" * 70)

    top_feats = res_df[res_df['feature'].isin(current_feats)].sort_values('eta_sq', ascending=False)

    print(f"\n{'feature':45s} {'eta_sq':>8s} {'L1_mean':>10s} {'L2_mean':>10s} {'L3_mean':>10s} {'L4_mean':>10s} {'trend':>10s}")
    print("-" * 140)

    for _, r in top_feats.head(25).iterrows():
        means = [r['mean_L1'], r['mean_L2'], r['mean_L3'], r['mean_L4']]
        if all(means[i] <= means[i + 1] for i in range(3)):
            trend = "INCREASE"
        elif all(means[i] >= means[i + 1] for i in range(3)):
            trend = "DECREASE"
        else:
            trend = "NONLINEAR"
        print(f"{r['feature']:45s} {r['eta_sq']:8.4f} {means[0]:10.4f} {means[1]:10.4f} {means[2]:10.4f} {means[3]:10.4f} {trend:>10s}")

    # ==========================================================
    # 6. MODALITY CONTRIBUTION SUMMARY
    # ==========================================================
    print("\n" + "=" * 70)
    print("6. MODALITY CONTRIBUTION")
    print("=" * 70)

    modalities = [
        ('ecg_', 'ECG/HRV'), ('eda_', 'EDA'), ('emg_', 'EMG'),
        ('eye_', 'Eye/Pupil'), ('head_', 'Head'), ('ac_', 'Aircraft'),
        ('delta_', 'Delta'), ('overall_', 'Ocu-Entropy'), ('psd_', 'Ocu-PSD'),
        ('eyes_closed', 'Ocu-Blink'), ('pupil_diam', 'Ocu-Pupil'),
        ('fix_', 'Ocu-Fixation'), ('sac_', 'Ocu-Saccade'),
    ]

    print(f"\n{'modality':20s} {'total':>6s} {'significant':>12s} {'after_VIF':>10s} {'top_eta_sq':>12s}")
    print("-" * 65)

    for prefix, label in modalities:
        total = sum(1 for f in keep_feats if f.startswith(prefix))
        sig_count = sum(1 for f in sig_feat_names if f.startswith(prefix))
        vif_count = sum(1 for f in current_feats if f.startswith(prefix))
        top_eta = 0
        matching = res_df[res_df['feature'].apply(lambda x: x.startswith(prefix))]
        if len(matching) > 0:
            top_eta = matching['eta_sq'].max()
        if total > 0:
            print(f"{label:20s} {total:6d} {sig_count:12d} {vif_count:10d} {top_eta:12.4f}")

    # ==========================================================
    # 7. FEATURE LIST FOR MODELING
    # ==========================================================
    print("\n" + "=" * 70)
    print("7. RECOMMENDED FEATURE SET FOR MODELING")
    print("=" * 70)

    print(f"\nfeatures to use ({len(current_feats)}):")
    for i, f in enumerate(current_feats):
        print(f"  {i + 1:2d}. {f}")

    print(f"\ntotal samples: {len(df)}")
    print(f"samples per class: {dict(df['difficulty'].value_counts().sort_index())}")
    print(f"subjects: {df['subject'].nunique()}")
    print(f"feature:sample ratio: 1:{len(df) // len(current_feats)}")


if __name__ == '__main__':
    main()
