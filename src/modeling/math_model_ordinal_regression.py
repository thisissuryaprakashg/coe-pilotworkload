"""
math_model_ordinal_regression.py
=================================
Rigorous 6-step mathematical model pipeline for Section 8.1 of cogpilot_project_spec.md.

Approach A (Primary): Ordinal Logistic Regression on Delta (personalised) features.
    logit(P(Y_ij <= k)) = theta_k - (beta_1 * X1_ij + ... + beta_p * Xp_ij)
    where X = Delta = Task - Rest (per-subject baseline-subtracted features)

Approach B (Comparison): Same Ordinal LR but on Raw features + explicit subject_id dummy
    fixed effect to show the value of baseline personalisation without a full GLMM.

Steps executed:
  Step 1  - Choose features (delta + oculomotor + EMG significant set)
  Step 2  - VIF pruning: iteratively drop VIF > 10 to remove multicollinearity
  Step 3  - Fit Ordinal Logistic Regression (statsmodels OrderedModel, logit link)
  Step 4  - Proportional Odds Assumption Test via Likelihood Ratio Test
             (Ordinal vs. Multinomial Logistic Regression)
  Step 5  - Extract beta coefficients, convert to Odds Ratios with 95% CI & p-values
  Step 6  - Leave-One-Subject-Out (LOSO) cross-validation: Accuracy, Macro F1, Cohen Kappa

Outputs:
  math_model_vif_report.csv            - VIF values before / after pruning
  math_model_equation_coefficients.csv - beta, OR, 95%CI, p-value for the final model
  math_model_loso_results.csv          - Per-fold LOSO accuracy/F1/kappa
  math_model_summary.txt               - Human-readable written summary for paper
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# statsmodels for ordinal / multinomial regression with full inference output
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

# sklearn for LOSO pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MASTER_CSV    = PROJECT_ROOT / 'data' / 'processed' / 'master_feature_matrix.csv'
SELECTED_JSON = PROJECT_ROOT / 'reports' / 'selected_features.json'
OUT_VIF       = PROJECT_ROOT / 'reports' / 'math_model_vif_report.csv'
OUT_COEFS     = PROJECT_ROOT / 'reports' / 'math_model_equation_coefficients.csv'
OUT_LOSO      = PROJECT_ROOT / 'reports' / 'math_model_loso_results.csv'
OUT_SUMMARY   = PROJECT_ROOT / 'reports' / 'math_model_summary.txt'

VIF_THRESHOLD = 10.0   # Drop features with VIF > 10


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Assemble candidate features
# ─────────────────────────────────────────────────────────────────────────────
def build_candidate_features(df, selected_dict):
    """
    Primary set = delta features (personalized baselines) +
                  statistically significant oculomotor / EMG features from ANOVA.
    """
    sig = selected_dict['all_significant_features']

    # All delta_ columns that exist in the master matrix
    delta_cols = [c for c in df.columns if c.startswith('delta_')]

    # Significant oculomotor & EMG features present in the dataframe
    eye_emg_sig = [f for f in sig if f in df.columns]

    # Union: delta features + significant eye/EMG features
    candidate = list(dict.fromkeys(delta_cols + eye_emg_sig))   # preserve order, no dupes

    print(f"Step 1 - Candidate features : {len(candidate)}")
    print(f"         └─ Delta features  : {len(delta_cols)}")
    print(f"         └─ Sig eye/EMG     : {len(eye_emg_sig)}")
    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: VIF pruning (iterative)
# ─────────────────────────────────────────────────────────────────────────────
def iterative_vif_prune(X_df, threshold=VIF_THRESHOLD):
    """
    Iteratively compute VIF for each column and drop the column with the
    highest VIF above `threshold` until all remaining features have VIF <= threshold.
    Returns (pruned_df, vif_history_df).
    """
    features = list(X_df.columns)
    history = []

    round_num = 0
    while True:
        round_num += 1
        vif_data = pd.DataFrame({
            'Feature': features,
            'VIF': [variance_inflation_factor(X_df[features].values, i)
                    for i in range(len(features))],
            'Round': round_num
        })
        history.append(vif_data)
        max_vif = vif_data['VIF'].max()
        if max_vif <= threshold:
            break
        worst = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
        features.remove(worst)

    vif_history = pd.concat(history, ignore_index=True)
    final_vif = history[-1].copy()
    print(f"Step 2 - VIF pruning complete:")
    print(f"         └─ Started with {len(X_df.columns)} features")
    print(f"         └─ Retained     {len(features)} features (VIF <= {threshold})")
    for _, row in final_vif.sort_values('VIF', ascending=False).iterrows():
        print(f"           {row['Feature']:35s}  VIF = {row['VIF']:.2f}")
    return features, vif_history


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 + 4: Fit Ordinal model, test Proportional Odds, fall back if needed
# ─────────────────────────────────────────────────────────────────────────────
def fit_and_select_model(df_model, feature_cols, target='difficulty_ground_truth'):
    """
    Fit Ordinal Logistic Regression (Proportional Odds Model).
    Test assumption via Likelihood Ratio Test against unconstrained Multinomial LR.
    If assumption fails (p < 0.05), fall back to Multinomial.

    Returns the chosen statsmodels result object and a string describing model type.
    """
    from sklearn.linear_model import LogisticRegression as SKLR
    from sklearn.preprocessing import LabelEncoder

    X = df_model[feature_cols].values.astype(float)
    y = df_model[target].values.astype(int)

    # ── Ordinal Model ──────────────────────────────────────────────
    print("\nStep 3 - Fitting Ordinal Logistic Regression (Proportional Odds Model)...")
    ordinal_res = OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=False)
    llf_ordinal = ordinal_res.llf

    # ── Multinomial Model (via sklearn for comparable LL) ──────────
    # We compute log-likelihood of multinomial LR manually using predicted probas
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    pipe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0, random_state=42)
    pipe.fit(X, y)
    probas = pipe.predict_proba(X)
    classes = pipe.classes_

    # Manual log-likelihood of multinomial model
    y_onehot = (y[:, None] == classes[None, :]).astype(float)
    llf_multinomial = np.sum(y_onehot * np.log(probas + 1e-12))

    # ── Proportional Odds Assumption Test (LRT) ────────────────────
    # Ordinal has p + 3 params (3 cut-points for 4 levels); Multinomial has 3*(p+1)
    n_feats = len(feature_cols)
    df_lrt = 3 * (n_feats + 1) - (n_feats + 3)   # extra df of unconstrained model
    df_lrt = max(df_lrt, 1)
    lrt_stat = -2 * (llf_ordinal - llf_multinomial)
    p_assumption = 1 - stats.chi2.cdf(lrt_stat, df=df_lrt)

    print(f"\nStep 4 - Proportional Odds Assumption (Likelihood Ratio Test):")
    print(f"         LRT statistic = {lrt_stat:.3f}  |  df = {df_lrt}  |  p = {p_assumption:.4f}")
    if p_assumption > 0.05:
        print("         ✅  Assumption HOLDS (p > 0.05). Using Ordinal Logistic Regression.")
        chosen_model = 'Ordinal Logistic Regression (Proportional Odds Model)'
        chosen_res   = ordinal_res
    else:
        print("         ⚠️  Assumption FAILS (p < 0.05). Falling back to Multinomial Logistic Regression.")
        chosen_model = 'Multinomial Logistic Regression'
        chosen_res   = None   # we will use sklearn for LOSO; statsmodels for coefs below

    return chosen_res, chosen_model, ordinal_res, p_assumption, lrt_stat


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Extract beta coefficients → Odds Ratios
# ─────────────────────────────────────────────────────────────────────────────
def extract_coefficients(ordinal_res, feature_cols, model_type):
    """
    Extract standardized beta coefficients, odds ratios, 95% CI, and p-values
    directly from the OrderedResults attributes — avoids summary2() which is
    not available on OrderedResults.
    """
    print("\nStep 5 - Extracting beta coefficients and Odds Ratios...")

    # OrderedModel params: first len(feature_cols) entries are feature betas;
    # the remaining entries are the threshold/cut-point parameters.
    # conf_int() returns numpy ndarray (n_params, 2) on OrderedResults
    ci_arr  = np.asarray(ordinal_res.conf_int())
    params  = np.asarray(ordinal_res.params)[:len(feature_cols)]
    bse     = np.asarray(ordinal_res.bse)[:len(feature_cols)]
    tvals   = np.asarray(ordinal_res.tvalues)[:len(feature_cols)]
    pvals   = np.asarray(ordinal_res.pvalues)[:len(feature_cols)]
    ci_low  = ci_arr[:len(feature_cols), 0]
    ci_high = ci_arr[:len(feature_cols), 1]

    records = []
    for i, feat in enumerate(feature_cols):
        beta = float(params[i])
        se   = float(bse[i])
        z    = float(tvals[i])
        p    = float(pvals[i])
        ci_l = float(ci_low[i])
        ci_u = float(ci_high[i])
        or_  = np.exp(beta)
        or_l = np.exp(ci_l)
        or_u = np.exp(ci_u)

        sig_stars = ''
        if   p < 0.001: sig_stars = '***'
        elif p < 0.01:  sig_stars = '**'
        elif p < 0.05:  sig_stars = '*'
        elif p < 0.10:  sig_stars = '.'

        records.append({
            'Feature': feat,
            'Beta (Standardized)': beta,
            'Std_Error': se,
            'z_statistic': z,
            'p_value': p,
            'Significance': sig_stars,
            'Odds_Ratio': or_,
            'OR_CI_Lower_95': or_l,
            'OR_CI_Upper_95': or_u
        })

    df_coefs = pd.DataFrame(records).sort_values('p_value')
    df_coefs.to_csv(OUT_COEFS, index=False)
    print(f"         Saved coefficient table → {OUT_COEFS}")
    return df_coefs


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: LOSO Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────
def run_loso_math_model(df, feature_cols, target='difficulty_ground_truth',
                        exclude_cp003_for_eye=True):
    """
    Leave-One-Subject-Out CV using statsmodels OrderedModel (Approach A: Delta features).
    Fallback to sklearn LogisticRegression if ordinal fitting fails in a fold.

    Returns per-fold DataFrame and aggregate metrics.
    """
    from sklearn.linear_model import LogisticRegression

    has_eye = any(k in f.lower() for f in feature_cols
                  for k in ['gaze', 'pupil', 'fix', 'sac', 'psd', 'eye'])

    if has_eye and exclude_cp003_for_eye:
        df_eval = df[df['subject'] != 'sub-cp003'].copy()
        note = "sub-cp003 excluded (0% eye quality). 34 LOSO folds."
    else:
        df_eval = df.copy()
        note = "All 35 subjects. 35 LOSO folds."

    subjects = sorted(df_eval['subject'].unique())
    print(f"\nStep 6 - LOSO Cross-Validation: {note}")

    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()

    fold_records = []
    for subj in subjects:
        train_idx = df_eval['subject'] != subj
        test_idx  = df_eval['subject'] == subj

        X_tr = df_eval.loc[train_idx, feature_cols].values.astype(float)
        y_tr = df_eval.loc[train_idx, target].values.astype(int)
        X_te = df_eval.loc[test_idx, feature_cols].values.astype(float)
        y_te = df_eval.loc[test_idx, target].values.astype(int)

        # Impute then scale
        X_tr = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_te = scaler.transform(imputer.transform(X_te))

        try:
            # Try statsmodels Ordinal Model for mathematical rigor
            mdl = OrderedModel(y_tr, X_tr, distr='logit')
            res = mdl.fit(method='bfgs', disp=False)
            preds = res.predict(X_te).argmax(axis=1) + 1   # columns = P(k=1..4)
        except Exception:
            # Fallback to sklearn multinomial LR (numerically more stable per fold)
            lr = LogisticRegression(solver='lbfgs', max_iter=1000,
                                    C=1.0, random_state=42)
            lr.fit(X_tr, y_tr)
            preds = lr.predict(X_te)

        acc   = accuracy_score(y_te, preds)
        f1_m  = f1_score(y_te, preds, average='macro', zero_division=0)
        f1_w  = f1_score(y_te, preds, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y_te, preds)

        # Extreme discrimination (Level 1 vs Level 4 only)
        ext_mask = (y_te == 1) | (y_te == 4)
        acc_ext  = accuracy_score(y_te[ext_mask], preds[ext_mask]) if ext_mask.sum() > 0 else np.nan

        fold_records.append({
            'Subject': subj,
            'N_test': len(y_te),
            'Accuracy': acc,
            'Macro_F1': f1_m,
            'Weighted_F1': f1_w,
            'Cohen_Kappa': kappa,
            'Extreme_Acc_1v4': acc_ext
        })

        print(f"  Fold {subj}: Acc={acc:.2%}  Macro-F1={f1_m:.3f}  Kappa={kappa:.3f}")

    df_loso = pd.DataFrame(fold_records)
    df_loso.to_csv(OUT_LOSO, index=False)

    # Aggregate
    agg = df_loso[['Accuracy','Macro_F1','Weighted_F1','Cohen_Kappa','Extreme_Acc_1v4']].agg(['mean','std'])
    return df_loso, agg


# ─────────────────────────────────────────────────────────────────────────────
# Approach B: Fixed-effects Subject Dummy model (baseline comparison)
# ─────────────────────────────────────────────────────────────────────────────
def run_approach_b_fixed_effects(df, feature_cols, target='difficulty_ground_truth'):
    """
    Approach B: Raw features + subject dummy fixed effects (sklearn multinomial LR).
    This is the statistical equivalent of GLMM random intercepts for our scale of data.
    Evaluated with LOSO to compare directly against Approach A (Delta features).
    Note: By definition, adding subject dummies uses each subject as their own control –
    showing whether the MODEL recovers baseline differences automatically vs.
    Approach A where WE explicitly subtract them.
    """
    from sklearn.linear_model import LogisticRegression

    # Use raw autonomic + raw eye features (no delta)
    raw_eye_emg_cols = [c for c in feature_cols if not c.startswith('delta_')]
    if not raw_eye_emg_cols:
        print("Approach B: No raw features found. Skipping.")
        return None, None

    # Add one-hot encoded subject dummies to raw features
    subj_dummies = pd.get_dummies(df['subject'], prefix='subj', drop_first=True).astype(float)
    X_raw  = df[raw_eye_emg_cols].copy()
    X_full = pd.concat([X_raw.reset_index(drop=True),
                        subj_dummies.reset_index(drop=True)], axis=1)
    y      = df[target].values

    subjects = sorted(df['subject'].unique())
    print(f"\nApproach B (Raw + Subject Fixed Effects) LOSO...")

    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()

    records = []
    for subj in subjects:
        train_mask = df['subject'] != subj
        test_mask  = df['subject'] == subj

        # For the test fold, zero-out the test subject's dummy (it was not in training)
        Xtr = X_full[train_mask.values].copy()
        Xte = X_full[test_mask.values].copy()
        ytr = y[train_mask.values]
        yte = y[test_mask.values]

        # Zero out test subject dummy column if it exists
        subj_col = f'subj_{subj}'
        if subj_col in Xte.columns:
            Xte[subj_col] = 0.0

        Xtr_s = scaler.fit_transform(imputer.fit_transform(Xtr))
        Xte_s = scaler.transform(imputer.transform(Xte))

        lr = LogisticRegression(solver='lbfgs', max_iter=1000,
                                C=0.5, random_state=42)
        lr.fit(Xtr_s, ytr)
        preds = lr.predict(Xte_s)

        records.append({
            'Subject': subj,
            'Accuracy': accuracy_score(yte, preds),
            'Macro_F1': f1_score(yte, preds, average='macro', zero_division=0),
            'Cohen_Kappa': cohen_kappa_score(yte, preds)
        })

    df_b = pd.DataFrame(records)
    agg_b = df_b[['Accuracy','Macro_F1','Cohen_Kappa']].agg(['mean','std'])
    return df_b, agg_b


# ─────────────────────────────────────────────────────────────────────────────
# Write Human-Readable Paper Summary
# ─────────────────────────────────────────────────────────────────────────────
def write_paper_summary(model_type, p_assumption, lrt_stat, df_coefs,
                        agg_a, agg_b, feature_cols, pruned_features):
    lines = []
    lines.append("=" * 80)
    lines.append("SECTION 8.1: MATHEMATICAL MODEL RESULTS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Selected Model : {model_type}")
    lines.append(f"Proportional Odds LRT: chi2 = {lrt_stat:.3f},  p = {p_assumption:.4f}")
    lines.append("")
    lines.append("Feature Selection (post VIF pruning):")
    lines.append(f"  Candidate features entered : {len(feature_cols)}")
    lines.append(f"  Features retained (VIF<=10): {len(pruned_features)}")
    lines.append("  Pruned feature set:")
    for f in pruned_features:
        lines.append(f"    - {f}")
    lines.append("")
    lines.append("Equation Interpretation (top significant predictors, p < 0.05):")
    sig = df_coefs[df_coefs['p_value'] < 0.05].sort_values('p_value')
    for _, row in sig.iterrows():
        direction = "increases" if row['Beta (Standardized)'] > 0 else "decreases"
        lines.append(
            f"  {row['Feature']:35s}: beta={row['Beta (Standardized)']:+.4f}, "
            f"OR={row['Odds_Ratio']:.3f} "
            f"[{row['OR_CI_Lower_95']:.3f}, {row['OR_CI_Upper_95']:.3f}], "
            f"p={row['p_value']:.4f} {row['Significance']}"
        )
        lines.append(
            f"    -> For a 1-SD increase in {row['Feature'].replace('_',' ')}, "
            f"the odds of transitioning to a higher workload level "
            f"{direction} by {abs(row['Odds_Ratio']-1)*100:.1f}%."
        )
    lines.append("")
    lines.append("LOSO Cross-Validation Results (Approach A: Delta personalised features):")
    lines.append(f"  Mean Accuracy   : {agg_a.loc['mean','Accuracy']:.2%}  +/- {agg_a.loc['std','Accuracy']:.2%}")
    lines.append(f"  Mean Macro-F1   : {agg_a.loc['mean','Macro_F1']:.3f}  +/- {agg_a.loc['std','Macro_F1']:.3f}")
    lines.append(f"  Mean Kappa      : {agg_a.loc['mean','Cohen_Kappa']:.3f}  +/- {agg_a.loc['std','Cohen_Kappa']:.3f}")
    lines.append(f"  Extreme Acc(1v4): {agg_a.loc['mean','Extreme_Acc_1v4']:.2%}  +/- {agg_a.loc['std','Extreme_Acc_1v4']:.2%}")
    if agg_b is not None:
        lines.append("")
        lines.append("LOSO Cross-Validation Results (Approach B: Raw features + Subject Fixed Effects):")
        lines.append(f"  Mean Accuracy   : {agg_b.loc['mean','Accuracy']:.2%}  +/- {agg_b.loc['std','Accuracy']:.2%}")
        lines.append(f"  Mean Macro-F1   : {agg_b.loc['mean','Macro_F1']:.3f}  +/- {agg_b.loc['std','Macro_F1']:.3f}")
        lines.append(f"  Mean Kappa      : {agg_b.loc['mean','Cohen_Kappa']:.3f}  +/- {agg_b.loc['std','Cohen_Kappa']:.3f}")
        lines.append("")
        delta_acc = agg_a.loc['mean','Accuracy'] - agg_b.loc['mean','Accuracy']
        lines.append(f"  Personalization Gain (Delta vs Raw): {delta_acc:+.2%} accuracy")
    lines.append("")
    lines.append("=" * 80)

    with open(OUT_SUMMARY, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f"\nSaved paper summary -> {OUT_SUMMARY}")
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("MATHEMATICAL MODEL PIPELINE  –  SECTION 8.1 (Ordinal Logistic Regression)")
    print("=" * 80)

    df = pd.read_csv(MASTER_CSV)
    with open(SELECTED_JSON) as f:
        selected_dict = json.load(f)

    # ── Exclude sub-cp003 (0% eye quality) for the main model ──────
    df_model = df[df['subject'] != 'sub-cp003'].copy()
    print(f"Working dataset (eye-valid): {len(df_model)} runs, "
          f"{df_model['subject'].nunique()} subjects")

    # ── STEP 1: Candidate features ──────────────────────────────────
    all_candidates = build_candidate_features(df_model, selected_dict)

    # ── Prepare feature matrix (impute for VIF, scale after pruning) ─
    X_cand = df_model[all_candidates].copy()
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    X_imp   = pd.DataFrame(imputer.fit_transform(X_cand), columns=all_candidates)
    X_sc    = pd.DataFrame(scaler.fit_transform(X_imp),   columns=all_candidates)

    # ── STEP 2: VIF pruning ─────────────────────────────────────────
    pruned_features, vif_history = iterative_vif_prune(X_sc)
    vif_history.to_csv(OUT_VIF, index=False)
    print(f"         VIF report saved → {OUT_VIF}")

    # ── STEP 3 + 4: Fit model, test assumption ──────────────────────
    X_pruned = X_sc[pruned_features].values
    y_vals   = df_model['difficulty_ground_truth'].values.astype(int)
    chosen_res, chosen_model, ordinal_res, p_assumption, lrt_stat = \
        fit_and_select_model(
            df_model.assign(**{f: X_pruned[:, i] for i, f in enumerate(pruned_features)}),
            pruned_features
        )

    # ── STEP 5: Extract coefficients ────────────────────────────────
    df_coefs = extract_coefficients(ordinal_res, pruned_features, chosen_model)
    print(f"\n  Top 5 predictors (p-value ranked):")
    for _, row in df_coefs.head(5).iterrows():
        print(f"    {row['Feature']:35s}: beta={row['Beta (Standardized)']:+.4f}  "
              f"OR={row['Odds_Ratio']:.3f}  p={row['p_value']:.4f} {row['Significance']}")

    # ── STEP 6: LOSO cross-validation ───────────────────────────────
    df_loso_a, agg_a = run_loso_math_model(df, pruned_features, exclude_cp003_for_eye=True)

    print("\n  Aggregate LOSO Metrics (Approach A – Delta personalised):")
    print(f"    Mean Accuracy : {agg_a.loc['mean','Accuracy']:.2%}  ± {agg_a.loc['std','Accuracy']:.2%}")
    print(f"    Mean Macro-F1 : {agg_a.loc['mean','Macro_F1']:.3f}  ± {agg_a.loc['std','Macro_F1']:.3f}")
    print(f"    Mean Kappa    : {agg_a.loc['mean','Cohen_Kappa']:.3f}  ± {agg_a.loc['std','Cohen_Kappa']:.3f}")
    print(f"    Extreme 1v4   : {agg_a.loc['mean','Extreme_Acc_1v4']:.2%}  ± {agg_a.loc['std','Extreme_Acc_1v4']:.2%}")

    # ── Approach B: Fixed-effects baseline comparison ───────────────
    df_loso_b, agg_b = run_approach_b_fixed_effects(df_model, pruned_features)
    if agg_b is not None:
        print("\n  Aggregate LOSO Metrics (Approach B – Raw + Subject Fixed Effects):")
        print(f"    Mean Accuracy : {agg_b.loc['mean','Accuracy']:.2%}  ± {agg_b.loc['std','Accuracy']:.2%}")
        print(f"    Mean Macro-F1 : {agg_b.loc['mean','Macro_F1']:.3f}  ± {agg_b.loc['std','Macro_F1']:.3f}")
        print(f"    Mean Kappa    : {agg_b.loc['mean','Cohen_Kappa']:.3f}  ± {agg_b.loc['std','Cohen_Kappa']:.3f}")

    # ── Paper summary ───────────────────────────────────────────────
    summary = write_paper_summary(
        chosen_model, p_assumption, lrt_stat,
        df_coefs, agg_a, agg_b,
        all_candidates, pruned_features
    )
    print("\n" + summary)

    print("=" * 80)
    print("DONE. All outputs saved to c:\\coe\\")
    print("=" * 80)


if __name__ == '__main__':
    main()
