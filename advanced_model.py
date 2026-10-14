"""
advanced_model.py
=================
Two improvements over improved_model.py:
1. FULL subject normalization — z-score ALL features with rest equivalents
   (30 features vs previous 16), plus rest summary features as predictors.
2. Subject-level rest features as predictors (pseudo random intercepts) —
   gives the model subject-specific context without data leakage.
No data leakage. LOSO with per-fold feature screening.
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.stats import f_oneway, mode as scipy_mode
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.pipeline import Pipeline

from new_data_pipeline import (
    ecg_window, eda_window, emg_window, eye_window, head_window,
    load_run_streams, get_ref_time,
    DATASET_ROOT, WINDOW_DN, EYE_EXCLUDED_SUBJECTS,
)

warnings.filterwarnings('ignore')

INPUT_CSV = Path(r"c:\coe_new\new_master_features.csv")
META_COLS = {'subject', 'session', 'run_folder', 'run_number', 'difficulty', 'window'}
EXCLUDE_PREFIXES = ('ac_',)

# ALL features extractable from rest recordings (30 features)
ALL_REST_FEATURES = [
    'ecg_hr_mean', 'ecg_hr_std', 'ecg_hr_range',
    'ecg_rmssd', 'ecg_sdnn', 'ecg_pnn50', 'ecg_rr_cv', 'ecg_sampen',
    'eda_scl_mean', 'eda_scl_slope', 'eda_cond_std',
    'eda_scr_count', 'eda_scr_amp_mean', 'eda_phasic_energy',
    'emg_flexor_rms', 'emg_extensor_rms', 'emg_cocontraction', 'emg_acc_rms',
    'eye_pupil_mean', 'eye_pupil_std', 'eye_pupil_skew', 'eye_pupil_kurt',
    'eye_pupil_velocity', 'eye_perclos', 'eye_blink_rate',
    'eye_gaze_ent_x', 'eye_gaze_ent_y',
    'head_vel_rms', 'head_yaw_std', 'head_pitch_std',
]

TEMPORAL_FEATURES = [
    'ecg_hr_mean', 'ecg_rmssd', 'eda_scl_mean',
    'emg_flexor_rms', 'emg_extensor_rms',
    'eye_pupil_mean', 'eye_blink_rate', 'head_pitch_std',
]

# subject-level rest summary features (pseudo random intercepts)
REST_SUMMARY_KEYS = [
    'ecg_hr_mean', 'ecg_rmssd', 'ecg_sampen',
    'eda_scl_mean', 'eda_scr_count',
    'emg_flexor_rms', 'emg_extensor_rms',
    'eye_pupil_mean', 'eye_blink_rate',
    'head_pitch_std',
]


# ============================================================
# 1. REST WINDOWED STATS (ALL features)
# ============================================================

def compute_rest_stats_full():
    rest_dir = DATASET_ROOT / 'task-rest'
    subjects = sorted([d for d in os.listdir(rest_dir) if d.startswith('sub-cp')])
    all_stats = {}

    for subj in subjects:
        subj_path = rest_dir / subj
        for sess in sorted(os.listdir(subj_path)):
            if not sess.startswith('ses-'):
                continue
            rest_path = subj_path / sess / 'level-000_run-001'
            if not rest_path.exists():
                continue

            s = load_run_streams(str(rest_path))
            ref = get_ref_time(s)
            if ref is None or len(ref) < 100:
                continue

            records = []
            w0 = ref[0]
            while w0 + WINDOW_DN * 0.5 < ref[-1]:
                w1 = w0 + WINDOW_DN
                rec = {}
                rec.update(ecg_window(s.get('ecg'), w0, w1, s.get('ecg_fs', 512)))
                rec.update(eda_window(s.get('eda'), w0, w1, s.get('eda_fs', 128)))
                rec.update(emg_window(s.get('emg'), w0, w1, s.get('emg_fs', 512)))
                rec.update(eye_window(s.get('eye'), w0, w1, subj in EYE_EXCLUDED_SUBJECTS))
                rec.update(head_window(s.get('head'), w0, w1))
                records.append(rec)
                w0 = w1

            if len(records) < 3:
                continue

            rdf = pd.DataFrame(records)
            stats = {}
            for col in rdf.columns:
                vals = rdf[col].dropna()
                if len(vals) >= 3:
                    m, sv = float(vals.mean()), float(vals.std())
                    stats[col] = (m, max(sv, 1e-6))
            all_stats[subj] = stats

    print(f"  rest stats computed: {len(all_stats)} subjects")
    return all_stats


# ============================================================
# 2. FEATURE ENHANCEMENT
# ============================================================

def enhance_full(df, rest_stats):
    print("\n--- ENHANCEMENT ---")

    # --- Z-score ALL rest-available features ---
    n_z = 0
    for feat in ALL_REST_FEATURES:
        if feat not in df.columns:
            continue
        col = f'z_{feat}'
        df[col] = np.nan
        for subj, stats in rest_stats.items():
            if feat not in stats:
                continue
            mean, std = stats[feat]
            mask = df['subject'] == subj
            df.loc[mask, col] = (df.loc[mask, feat] - mean) / std
        if df[col].notna().mean() > 0.40:
            n_z += 1
        else:
            df.drop(columns=[col], inplace=True)
    print(f"  z-scored features: {n_z} / {len(ALL_REST_FEATURES)} attempted")

    # --- Window position ---
    df['window_position'] = 0.0
    for key, grp in df.groupby(['subject', 'session', 'run_folder']):
        mx = grp['window'].max()
        if mx > 0:
            df.loc[grp.index, 'window_position'] = grp['window'] / mx

    # --- Temporal delta ---
    n_td = 0
    for feat in TEMPORAL_FEATURES:
        if feat not in df.columns:
            continue
        col = f'tdelta_{feat}'
        df[col] = np.nan
        for key, grp in df.groupby(['subject', 'session', 'run_folder']):
            sg = grp.sort_values('window')
            fv = sg[feat].iloc[0]
            if np.isfinite(fv):
                df.loc[sg.index, col] = sg[feat] - fv
        if df[col].notna().mean() > 0.5:
            n_td += 1
        else:
            df.drop(columns=[col], inplace=True)
    print(f"  temporal deltas: {n_td}")

    # --- Temporal slope ---
    n_ts = 0
    for feat in TEMPORAL_FEATURES:
        if feat not in df.columns:
            continue
        col = f'slope_{feat}'
        df[col] = np.nan
        for key, grp in df.groupby(['subject', 'session', 'run_folder']):
            sg = grp.sort_values('window')
            vals = sg[feat].values
            valid = np.isfinite(vals)
            if np.sum(valid) >= 4:
                x = np.arange(len(vals))[valid]
                slope = np.polyfit(x, vals[valid], 1)[0]
                df.loc[sg.index, col] = slope
        if df[col].notna().mean() > 0.5:
            n_ts += 1
        else:
            df.drop(columns=[col], inplace=True)
    print(f"  temporal slopes: {n_ts}")

    # --- Subject-level rest summary features (pseudo random intercepts) ---
    n_rs = 0
    for feat in REST_SUMMARY_KEYS:
        col = f'rsumm_{feat}'
        df[col] = np.nan
        for subj, stats in rest_stats.items():
            if feat in stats:
                df.loc[df['subject'] == subj, col] = stats[feat][0]
        if df[col].notna().mean() > 0.5:
            n_rs += 1
        else:
            df.drop(columns=[col], inplace=True)
    print(f"  rest summary features (pseudo intercepts): {n_rs}")

    meta = {'subject', 'session', 'run_folder', 'run_number', 'difficulty', 'window'}
    feats = [c for c in df.columns if c not in meta]
    print(f"\n  total: {len(df)} rows x {len(feats)} features")
    return df


# ============================================================
# 3. FEATURE SCREENING (train-only, silent)
# ============================================================

def screen_silent(df_train, feat_cols):
    nan_pct = df_train[feat_cols].isna().mean()
    keep = [c for c in feat_cols if nan_pct[c] <= 0.30]
    stds = df_train[keep].std()
    keep = [c for c in keep if stds[c] > 1e-10]

    tmp = df_train[['difficulty'] + keep].copy()
    for c in keep:
        tmp[c] = tmp[c].fillna(tmp[c].median())

    groups = [tmp.loc[tmp['difficulty'] == d] for d in [1, 2, 3, 4]]
    sig = []
    for feat in keep:
        try:
            _, p = f_oneway(*[g[feat].values for g in groups])
            if p < 0.05:
                sig.append(feat)
        except Exception:
            pass

    current = sig.copy()
    for _ in range(100):
        if len(current) < 3:
            break
        X = tmp[current].values
        m, s = X.mean(0), X.std(0)
        s[s < 1e-10] = 1.0
        Xz = (X - m) / s
        max_vif, max_f = 0, None
        for i, name in enumerate(current):
            others = np.delete(Xz, i, axis=1)
            y = Xz[:, i]
            beta = np.linalg.lstsq(others, y, rcond=None)[0]
            ss_r = np.sum((y - others @ beta) ** 2)
            ss_t = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
            vif = 1 / (1 - r2) if r2 < 1 else 999
            if vif > max_vif:
                max_vif, max_f = vif, name
        if max_vif <= 10:
            break
        current.remove(max_f)
    return current


# ============================================================
# 4. ORDINAL LOGISTIC
# ============================================================

class OrdinalLogistic:
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
            by = (y > k).astype(int)
            if len(np.unique(by)) < 2:
                self.models.append(None)
                continue
            lr = LogisticRegression(solver='lbfgs', max_iter=5000, C=self.C)
            lr.fit(X, by)
            self.models.append(lr)

    def predict_proba(self, X):
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        probs = []
        for lr in self.models:
            probs.append(lr.predict_proba(X)[:, 1] if lr else np.zeros(len(X)))
        p = np.column_stack([1 - probs[0], probs[0] - probs[1],
                             probs[1] - probs[2], probs[2]])
        p = np.clip(p, 0, None)
        rs = p.sum(axis=1, keepdims=True)
        rs[rs < 1e-10] = 1.0
        return p / rs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1) + 1


# ============================================================
# 5. LOSO (leak-free, per-fold screening, run voting)
# ============================================================

def loso_evaluate(df, all_feats, model_name, template):
    subjects = sorted(df['subject'].unique())
    all_tw, all_pw = [], []
    all_tr, all_pr_v, all_pr_p = [], [], []
    per_subj_r = []
    feat_usage = Counter()

    for si, test_subj in enumerate(subjects):
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]

        features = screen_silent(train, all_feats)
        for f in features:
            feat_usage[f] += 1

        X_tr = train[features].values
        y_tr = train['difficulty'].values.astype(int)
        X_te = test[features].values
        y_te = test['difficulty'].values.astype(int)

        model = clone(template) if not isinstance(template, OrdinalLogistic) else OrdinalLogistic(C=template.C)
        model.fit(X_tr, y_tr)

        pw = np.clip(np.round(np.asarray(model.predict(X_te), dtype=float)), 1, 4).astype(int)

        # probabilities for voting
        wp = None
        if hasattr(model, 'predict_proba'):
            try:
                wp = model.predict_proba(X_te)
            except Exception:
                pass
        elif isinstance(model, OrdinalLogistic):
            wp = model.predict_proba(X_te)

        all_tw.extend(y_te.tolist())
        all_pw.extend(pw.tolist())

        # run-level
        tc = test[['run_folder', 'difficulty']].copy()
        tc['pred'] = pw
        if wp is not None:
            for c in range(4):
                tc[f'p{c}'] = wp[:, c]

        sr_t, sr_p = [], []
        for _, rg in tc.groupby('run_folder'):
            rt = int(rg['difficulty'].iloc[0])
            all_tr.append(rt)
            vote = int(scipy_mode(rg['pred'].values, keepdims=False).mode)
            all_pr_v.append(vote)
            if wp is not None and all(f'p{c}' in rg.columns for c in range(4)):
                ap = np.mean(rg[[f'p{c}' for c in range(4)]].values, axis=0)
                all_pr_p.append(int(np.argmax(ap) + 1))
            else:
                all_pr_p.append(vote)
            sr_t.append(rt)
            sr_p.append(vote)

        sa = accuracy_score(sr_t, sr_p) if sr_t else 0
        per_subj_r.append((test_subj, sa, len(sr_t)))

        if (si + 1) % 10 == 0 or si == len(subjects) - 1:
            print(f"  fold {si + 1}/{len(subjects)} | {len(features)} feats")

    tw, pw_a = np.array(all_tw), np.array(all_pw)
    tr, prv, prp = np.array(all_tr), np.array(all_pr_v), np.array(all_pr_p)

    print(f"\n{'=' * 65}")
    print(f"  {model_name}")
    print(f"{'=' * 65}")

    # stability
    n_folds = len(subjects)
    stable = sum(1 for _, c in feat_usage.items() if c == n_folds)
    print(f"\n  features: {stable} stable (all folds) + {len(feat_usage) - stable} partial")

    # window level
    aw = accuracy_score(tw, pw_a)
    ajw = float(np.mean(np.abs(tw - pw_a) <= 1))
    bw = accuracy_score((tw >= 3).astype(int), (pw_a >= 3).astype(int))
    fw = f1_score(tw, pw_a, average='macro')
    kw = cohen_kappa_score(tw, pw_a)
    cmw = confusion_matrix(tw, pw_a, labels=[1, 2, 3, 4])

    print(f"\n  WINDOW ({len(tw)}):")
    print(f"  4-class={aw:.1%} | adjacent={ajw:.1%} | binary={bw:.1%} | F1={fw:.4f} | kappa={kw:.4f}")
    print(f"  confusion:")
    for i, row in enumerate(cmw):
        pct = row[i] / max(1, sum(row)) * 100
        print(f"    L{i + 1}  {row[0]:5d} {row[1]:5d} {row[2]:5d} {row[3]:5d}  ({pct:.1f}%)")

    # run level vote
    arv = accuracy_score(tr, prv)
    ajrv = float(np.mean(np.abs(tr - prv) <= 1))
    brv = accuracy_score((tr >= 3).astype(int), (prv >= 3).astype(int))
    frv = f1_score(tr, prv, average='macro')
    krv = cohen_kappa_score(tr, prv)
    cmr = confusion_matrix(tr, prv, labels=[1, 2, 3, 4])

    print(f"\n  RUN VOTE ({len(tr)}):")
    print(f"  4-class={arv:.1%} | adjacent={ajrv:.1%} | binary={brv:.1%} | F1={frv:.4f} | kappa={krv:.4f}")
    print(f"  confusion:")
    for i, row in enumerate(cmr):
        pct = row[i] / max(1, sum(row)) * 100
        print(f"    L{i + 1}  {row[0]:5d} {row[1]:5d} {row[2]:5d} {row[3]:5d}  ({pct:.1f}%)")

    # run level prob
    arp = accuracy_score(tr, prp)
    ajrp = float(np.mean(np.abs(tr - prp) <= 1))
    brp = accuracy_score((tr >= 3).astype(int), (prp >= 3).astype(int))

    print(f"\n  RUN PROB ({len(tr)}):")
    print(f"  4-class={arp:.1%} | adjacent={ajrp:.1%} | binary={brp:.1%}")

    # per-subject
    print(f"\n  per-subject run accuracy:")
    for subj, sa, n in per_subj_r:
        bar = '#' * int(sa * 25)
        print(f"    {subj} | {sa:.3f} ({int(sa * n):2d}/{n:2d}) {bar}")
    vals = [x[1] for x in per_subj_r]
    print(f"    mean={np.mean(vals):.4f} +/- {np.std(vals):.4f} | median={np.median(vals):.4f}")

    return {
        'model': model_name,
        'w_acc': aw, 'w_adj': ajw, 'w_bin': bw,
        'r_acc': arv, 'r_adj': ajrv, 'r_bin': brv,
        'rp_acc': arp, 'rp_adj': ajrp, 'rp_bin': brp,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("ADVANCED MODELS — FULL NORMALIZATION + REST INTERCEPTS")
    print("=" * 65)

    df = pd.read_csv(INPUT_CSV)
    print(f"base: {df.shape[0]} rows x {df.shape[1]} cols")

    print("\n--- REST PROCESSING ---")
    rest_stats = compute_rest_stats_full()

    df = enhance_full(df, rest_stats)

    all_feats = [c for c in df.columns
                 if c not in META_COLS
                 and not any(c.startswith(p) for p in EXCLUDE_PREFIXES)]
    print(f"\ncandidate features: {len(all_feats)}")
    print(f"screening: INSIDE each LOSO fold (train-only)")

    models = {
        'Linear Logistic': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', StandardScaler()),
            ('clf', LogisticRegression(solver='lbfgs', max_iter=5000, C=1.0)),
        ]),
        'Spline Logistic': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', StandardScaler()),
            ('spl', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ('clf', LogisticRegression(solver='lbfgs', max_iter=5000, C=0.1)),
        ]),
        'Ordinal Logistic': OrdinalLogistic(C=1.0),
    }

    results = []
    for name, tmpl in models.items():
        res = loso_evaluate(df, all_feats, name, tmpl)
        results.append(res)

    # comparison
    print(f"\n\n{'=' * 90}")
    print("COMPARISON (LEAK-FREE)")
    print(f"{'=' * 90}")
    print(f"\n{'model':25s} | {'WINDOW':^22s} | {'RUN VOTE':^22s} | {'RUN PROB':^22s}")
    print(f"{'':25s} | {'4cl':>6s} {'adj':>6s} {'bin':>6s} | {'4cl':>6s} {'adj':>6s} {'bin':>6s} | {'4cl':>6s} {'adj':>6s} {'bin':>6s}")
    print("-" * 90)
    for r in results:
        print(f"{r['model']:25s} | {r['w_acc']:6.1%} {r['w_adj']:6.1%} {r['w_bin']:6.1%} "
              f"| {r['r_acc']:6.1%} {r['r_adj']:6.1%} {r['r_bin']:6.1%} "
              f"| {r['rp_acc']:6.1%} {r['rp_adj']:6.1%} {r['rp_bin']:6.1%}")

    best = max(results, key=lambda x: x['r_acc'])
    print(f"\nbest run-level: {best['model']} ({best['r_acc']:.1%})")
    print(f"previous best:  Spline Logistic = 52.2%")
    print(f"chance:         25.0% (4-class) | 50.0% (binary)")


if __name__ == '__main__':
    main()
