"""
improved_model.py
=================
Improved nonlinear mathematical models on enhanced features.
- Feature screening INSIDE LOSO loop (no leakage)
- LOSO cross-validation
- Window-level AND run-level majority voting
- Models: Linear Logistic, Spline Logistic, Ordinal Logistic
No data leakage anywhere.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import f_oneway, mode as scipy_mode
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.pipeline import Pipeline
from collections import Counter

warnings.filterwarnings('ignore')

INPUT_CSV = Path(r"c:\coe_new\enhanced_features.csv")
META_COLS = {'subject', 'session', 'run_folder', 'run_number', 'difficulty', 'window'}
EXCLUDE_PREFIXES = ('ac_',)


# ============================================================
# FEATURE SCREENING (silent, for inside LOSO loop)
# ============================================================

def screen_features_silent(df_train, feat_cols):
    """ANOVA + VIF on TRAINING data only. Returns selected feature list."""

    # 1. drop >30% NaN
    nan_pct = df_train[feat_cols].isna().mean()
    keep = [c for c in feat_cols if nan_pct[c] <= 0.30]

    # 2. drop zero-variance
    stds = df_train[keep].std()
    keep = [c for c in keep if stds[c] > 1e-10]

    # 3. ANOVA p < 0.05
    df_clean = df_train[['difficulty'] + keep].copy()
    for c in keep:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())

    groups = [df_clean.loc[df_clean['difficulty'] == d] for d in [1, 2, 3, 4]]
    sig = []
    for feat in keep:
        try:
            _, p = f_oneway(*[g[feat].values for g in groups])
            if p < 0.05:
                sig.append(feat)
        except Exception:
            pass

    # 4. iterative VIF pruning (VIF > 10)
    current = sig.copy()
    for _ in range(80):
        if len(current) < 3:
            break
        X = df_clean[current].values
        m, s = X.mean(0), X.std(0)
        s[s < 1e-10] = 1.0
        Xz = (X - m) / s

        max_vif, max_feat = 0, None
        for i, name in enumerate(current):
            others = np.delete(Xz, i, axis=1)
            y = Xz[:, i]
            beta = np.linalg.lstsq(others, y, rcond=None)[0]
            ss_res = np.sum((y - others @ beta) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else 999.0
            if vif > max_vif:
                max_vif, max_feat = vif, name
        if max_vif <= 10.0:
            break
        current.remove(max_feat)

    return current


# ============================================================
# ORDINAL LOGISTIC (cumulative logit model)
# ============================================================

class OrdinalLogistic:
    """3 binary models: P(Y>1), P(Y>2), P(Y>3)."""

    def __init__(self, C=1.0):
        self.C = C
        self.models = []
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        self.models = []
        for k in [1, 2, 3]:
            binary_y = (y > k).astype(int)
            if len(np.unique(binary_y)) < 2:
                self.models.append(None)
                continue
            lr = LogisticRegression(solver='lbfgs', max_iter=5000, C=self.C)
            lr.fit(X, binary_y)
            self.models.append(lr)

    def predict_proba(self, X):
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        probs = []
        for lr in self.models:
            if lr is None:
                probs.append(np.zeros(len(X)))
            else:
                probs.append(lr.predict_proba(X)[:, 1])

        p = np.column_stack([
            1 - probs[0],
            probs[0] - probs[1],
            probs[1] - probs[2],
            probs[2],
        ])
        p = np.clip(p, 0, None)
        row_sums = p.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-10] = 1.0
        return p / row_sums

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1) + 1


# ============================================================
# LOSO WITH PER-FOLD SCREENING + RUN-LEVEL VOTING
# ============================================================

def loso_evaluate(df, all_feats, model_name, template, use_proba=False):
    subjects = sorted(df['subject'].unique())

    all_true_w, all_pred_w = [], []
    all_true_r, all_pred_r_vote, all_pred_r_prob = [], [], []
    per_subj_w, per_subj_r = [], []
    feat_usage = Counter()

    for si, test_subj in enumerate(subjects):
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]

        # screen features on TRAINING DATA ONLY (no leakage)
        features = screen_features_silent(train, all_feats)
        for f in features:
            feat_usage[f] += 1

        X_train = train[features].values
        y_train = train['difficulty'].values.astype(int)
        X_test = test[features].values
        y_test = test['difficulty'].values.astype(int)

        model = clone(template) if not isinstance(template, OrdinalLogistic) else OrdinalLogistic(C=template.C)
        model.fit(X_train, y_train)

        y_pred_w = np.asarray(model.predict(X_test), dtype=float)
        y_pred_w = np.clip(np.round(y_pred_w), 1, 4).astype(int)

        # window probabilities
        if use_proba and hasattr(model, 'predict_proba'):
            try:
                w_proba = model.predict_proba(X_test)
            except Exception:
                w_proba = None
        elif isinstance(model, OrdinalLogistic):
            w_proba = model.predict_proba(X_test)
        else:
            w_proba = None

        all_true_w.extend(y_test.tolist())
        all_pred_w.extend(y_pred_w.tolist())

        subj_acc_w = accuracy_score(y_test, y_pred_w)
        per_subj_w.append((test_subj, subj_acc_w))

        # run-level aggregation
        test_copy = test[['run_folder', 'difficulty']].copy()
        test_copy['pred'] = y_pred_w
        if w_proba is not None:
            for c in range(4):
                test_copy[f'prob_{c}'] = w_proba[:, c]

        subj_run_true, subj_run_pred = [], []
        for run_name, rg in test_copy.groupby('run_folder'):
            run_true = int(rg['difficulty'].iloc[0])
            all_true_r.append(run_true)

            preds = rg['pred'].values
            vote = int(scipy_mode(preds, keepdims=False).mode)
            all_pred_r_vote.append(vote)

            if w_proba is not None and all(f'prob_{c}' in rg.columns for c in range(4)):
                avg_prob = np.mean(rg[[f'prob_{c}' for c in range(4)]].values, axis=0)
                prob_pred = int(np.argmax(avg_prob) + 1)
                all_pred_r_prob.append(prob_pred)
            else:
                all_pred_r_prob.append(vote)

            subj_run_true.append(run_true)
            subj_run_pred.append(vote)

        subj_acc_r = accuracy_score(subj_run_true, subj_run_pred) if subj_run_true else 0
        per_subj_r.append((test_subj, subj_acc_r, len(subj_run_true)))

        if (si + 1) % 10 == 0 or si == len(subjects) - 1:
            print(f"  fold {si + 1}/{len(subjects)} done | {len(features)} features selected")

    all_true_w = np.array(all_true_w)
    all_pred_w = np.array(all_pred_w)
    all_true_r = np.array(all_true_r)
    all_pred_r_vote = np.array(all_pred_r_vote)
    all_pred_r_prob = np.array(all_pred_r_prob)

    # print results
    print(f"\n{'=' * 65}")
    print(f"  {model_name}")
    print(f"{'=' * 65}")

    # feature stability
    n_folds = len(subjects)
    stable = [f for f, cnt in feat_usage.items() if cnt == n_folds]
    sometimes = [f for f, cnt in feat_usage.items() if cnt < n_folds]
    print(f"\n  feature selection stability:")
    print(f"    selected in all {n_folds} folds: {len(stable)} features")
    print(f"    selected in some folds:   {len(sometimes)} features")
    if sometimes:
        for f, cnt in sorted(feat_usage.items(), key=lambda x: x[1]):
            if cnt < n_folds:
                print(f"      {f:40s} {cnt}/{n_folds} folds")

    # --- WINDOW LEVEL ---
    acc_w = accuracy_score(all_true_w, all_pred_w)
    adj_w = float(np.mean(np.abs(all_true_w - all_pred_w) <= 1))
    bt, bp = (all_true_w >= 3).astype(int), (all_pred_w >= 3).astype(int)
    bin_w = accuracy_score(bt, bp)
    f1_w = f1_score(all_true_w, all_pred_w, average='macro')
    k_w = cohen_kappa_score(all_true_w, all_pred_w)
    cm_w = confusion_matrix(all_true_w, all_pred_w, labels=[1, 2, 3, 4])

    print(f"\n  WINDOW-LEVEL ({len(all_true_w)} windows):")
    print(f"  4-class:  {acc_w:.4f} ({acc_w * 100:.1f}%)")
    print(f"  adjacent: {adj_w:.4f} ({adj_w * 100:.1f}%)")
    print(f"  binary:   {bin_w:.4f} ({bin_w * 100:.1f}%)")
    print(f"  F1:       {f1_w:.4f}")
    print(f"  kappa:    {k_w:.4f}")
    print(f"  confusion:")
    print(f"  {'pred':>8s} {'1':>6s} {'2':>6s} {'3':>6s} {'4':>6s}")
    for i, row in enumerate(cm_w):
        pct = row[i] / max(1, sum(row)) * 100
        print(f"    L{i + 1}   {row[0]:6d} {row[1]:6d} {row[2]:6d} {row[3]:6d}  ({pct:.1f}%)")

    # --- RUN LEVEL (majority vote) ---
    acc_rv = accuracy_score(all_true_r, all_pred_r_vote)
    adj_rv = float(np.mean(np.abs(all_true_r - all_pred_r_vote) <= 1))
    bt_r, bp_r = (all_true_r >= 3).astype(int), (all_pred_r_vote >= 3).astype(int)
    bin_rv = accuracy_score(bt_r, bp_r)
    f1_rv = f1_score(all_true_r, all_pred_r_vote, average='macro')
    k_rv = cohen_kappa_score(all_true_r, all_pred_r_vote)
    cm_rv = confusion_matrix(all_true_r, all_pred_r_vote, labels=[1, 2, 3, 4])

    print(f"\n  RUN-LEVEL MAJORITY VOTE ({len(all_true_r)} runs):")
    print(f"  4-class:  {acc_rv:.4f} ({acc_rv * 100:.1f}%)")
    print(f"  adjacent: {adj_rv:.4f} ({adj_rv * 100:.1f}%)")
    print(f"  binary:   {bin_rv:.4f} ({bin_rv * 100:.1f}%)")
    print(f"  F1:       {f1_rv:.4f}")
    print(f"  kappa:    {k_rv:.4f}")
    print(f"  confusion:")
    print(f"  {'pred':>8s} {'1':>6s} {'2':>6s} {'3':>6s} {'4':>6s}")
    for i, row in enumerate(cm_rv):
        pct = row[i] / max(1, sum(row)) * 100
        print(f"    L{i + 1}   {row[0]:6d} {row[1]:6d} {row[2]:6d} {row[3]:6d}  ({pct:.1f}%)")

    # --- RUN LEVEL (probability average) ---
    acc_rp = accuracy_score(all_true_r, all_pred_r_prob)
    adj_rp = float(np.mean(np.abs(all_true_r - all_pred_r_prob) <= 1))
    bt_rp = (all_true_r >= 3).astype(int)
    bp_rp = (all_pred_r_prob >= 3).astype(int)
    bin_rp = accuracy_score(bt_rp, bp_rp)
    f1_rp = f1_score(all_true_r, all_pred_r_prob, average='macro')
    k_rp = cohen_kappa_score(all_true_r, all_pred_r_prob)

    print(f"\n  RUN-LEVEL PROB AVERAGE ({len(all_true_r)} runs):")
    print(f"  4-class:  {acc_rp:.4f} ({acc_rp * 100:.1f}%)")
    print(f"  adjacent: {adj_rp:.4f} ({adj_rp * 100:.1f}%)")
    print(f"  binary:   {bin_rp:.4f} ({bin_rp * 100:.1f}%)")
    print(f"  F1:       {f1_rp:.4f}")
    print(f"  kappa:    {k_rp:.4f}")

    # per-subject run accuracy
    print(f"\n  per-subject run-level accuracy:")
    for subj, sa, n in per_subj_r:
        bar = '#' * int(sa * 25)
        print(f"    {subj} | {sa:.3f} ({int(sa * n):2d}/{n:2d}) {bar}")
    acc_vals = [x[1] for x in per_subj_r]
    print(f"    mean: {np.mean(acc_vals):.4f} +/- {np.std(acc_vals):.4f}")

    return {
        'model': model_name,
        'w_acc': acc_w, 'w_adj': adj_w, 'w_bin': bin_w, 'w_f1': f1_w, 'w_kappa': k_w,
        'r_vote_acc': acc_rv, 'r_vote_adj': adj_rv, 'r_vote_bin': bin_rv,
        'r_prob_acc': acc_rp, 'r_prob_adj': adj_rp, 'r_prob_bin': bin_rp,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(INPUT_CSV)

    print("=" * 65)
    print("IMPROVED MODELS — LEAK-FREE (per-fold screening)")
    print("=" * 65)
    print(f"samples:  {len(df)} windows")
    n_runs = df.groupby(['subject', 'session', 'run_folder']).ngroups
    print(f"runs:     {n_runs}")
    print(f"subjects: {df['subject'].nunique()}")

    # candidate features (exclude ac_ and metadata)
    all_feats = [c for c in df.columns
                 if c not in META_COLS
                 and not any(c.startswith(p) for p in EXCLUDE_PREFIXES)]
    print(f"candidate features: {len(all_feats)}")
    print(f"feature screening: INSIDE each LOSO fold (train-only, no leakage)")

    # models
    models = {
        'Linear Logistic': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(solver='lbfgs', max_iter=5000, C=1.0)),
        ]),
        'Spline Logistic': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ('clf', LogisticRegression(solver='lbfgs', max_iter=5000, C=0.1)),
        ]),
        'Ordinal Logistic': OrdinalLogistic(C=1.0),
    }

    results = []
    for name, template in models.items():
        use_proba = hasattr(template, 'predict_proba') or isinstance(template, OrdinalLogistic)
        if isinstance(template, Pipeline):
            last = template.steps[-1][1]
            use_proba = hasattr(last, 'predict_proba')
        res = loso_evaluate(df, all_feats, name, template, use_proba=use_proba)
        results.append(res)

    # comparison
    print(f"\n\n{'=' * 90}")
    print("COMPARISON SUMMARY (LEAK-FREE)")
    print(f"{'=' * 90}")

    print(f"\n{'model':25s} | {'--- WINDOW ---':^30s} | {'--- RUN (vote) ---':^25s} | {'--- RUN (prob) ---':^25s}")
    print(f"{'':25s} | {'4-cls':>7s} {'adj':>7s} {'bin':>7s} | {'4-cls':>7s} {'adj':>7s} {'bin':>7s} | {'4-cls':>7s} {'adj':>7s} {'bin':>7s}")
    print("-" * 90)
    for r in results:
        print(f"{r['model']:25s} | {r['w_acc']:7.1%} {r['w_adj']:7.1%} {r['w_bin']:7.1%} "
              f"| {r['r_vote_acc']:7.1%} {r['r_vote_adj']:7.1%} {r['r_vote_bin']:7.1%} "
              f"| {r['r_prob_acc']:7.1%} {r['r_prob_adj']:7.1%} {r['r_prob_bin']:7.1%}")

    best_w = max(results, key=lambda x: x['w_acc'])
    best_r = max(results, key=lambda x: x['r_vote_acc'])
    print(f"\nbest window-level: {best_w['model']} ({best_w['w_acc'] * 100:.1f}%)")
    print(f"best run-level:    {best_r['model']} ({best_r['r_vote_acc'] * 100:.1f}%)")
    print(f"chance: 25.0% (4-class) | 50.0% (binary)")


if __name__ == '__main__':
    main()
