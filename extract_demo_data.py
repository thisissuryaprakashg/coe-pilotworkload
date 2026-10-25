"""
extract_demo_data.py
====================
Runs Spline Logistic LOSO (same pipeline as improved_model.py) and
extracts all data needed for demo_dashboard.html.

Output: demo_data.js  — loaded by demo_dashboard.html as window.DEMO_DATA
Run this first, then open demo_dashboard.html directly in any browser.
No server needed.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mode as scipy_mode
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

# Import feature screening from improved_model (read-only import, file not modified)
from improved_model import screen_features_silent

warnings.filterwarnings('ignore')

INPUT_CSV  = Path(r"c:\coe_new\enhanced_features.csv")
OUTPUT_JS  = Path(r"c:\coe_new\demo_data.js")

META_COLS        = {'subject', 'session', 'run_folder', 'run_number', 'difficulty', 'window'}
EXCLUDE_PREFIXES = ('ac_',)

def get_modality(feat):
    f = feat.lower()
    for key in ('gaze', 'pupil', 'blink', 'perclos', 'saccade', 'fixation'):
        if key in f:
            return 'Eye'
    for prefix, mod in [('ecg', 'ECG'), ('eda', 'EDA'), ('emg', 'EMG'),
                         ('eye', 'Eye'), ('head', 'Head')]:
        if f.startswith(prefix) or f.startswith(f'z_{prefix}') \
                or f.startswith(f'tdelta_{prefix}') or f.startswith(f'slope_{prefix}'):
            return mod
    return 'Other'

def main():
    print("=" * 60)
    print("EXTRACT DEMO DATA — Spline Logistic LOSO")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    all_feats = [c for c in df.columns
                 if c not in META_COLS
                 and not any(c.startswith(p) for p in EXCLUDE_PREFIXES)]

    subjects = sorted(df['subject'].unique())
    print(f"subjects: {len(subjects)} | windows: {len(df)} | candidate features: {len(all_feats)}")

    template = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler()),
        ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
        ('clf',    LogisticRegression(solver='lbfgs', max_iter=5000, C=0.1)),
    ])

    all_true_w, all_pred_w = [], []
    all_true_r, all_pred_r = [], []
    feat_usage      = Counter()
    per_subject     = []
    window_records  = []

    for si, test_subj in enumerate(subjects):
        train = df[df['subject'] != test_subj]
        test  = df[df['subject'] == test_subj]

        features = screen_features_silent(train, all_feats)
        for f in features:
            feat_usage[f] += 1

        X_train = train[features].values
        y_train = train['difficulty'].values.astype(int)
        X_test  = test[features].values
        y_test  = test['difficulty'].values.astype(int)

        model = clone(template)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        proba  = model.predict_proba(X_test)  # (n_windows, 4)

        test_r = test.reset_index(drop=True)
        for i in range(len(y_test)):
            window_records.append({
                'subject': test_subj,
                'run':     str(test_r['run_folder'].iloc[i]),
                'window':  int(test_r['window'].iloc[i]) if 'window' in test_r.columns else i,
                'true':    int(y_test[i]),
                'pred':    int(y_pred[i]),
                'p1':      round(float(proba[i, 0]), 4),
                'p2':      round(float(proba[i, 1]), 4),
                'p3':      round(float(proba[i, 2]), 4),
                'p4':      round(float(proba[i, 3]), 4),
            })

        all_true_w.extend(y_test.tolist())
        all_pred_w.extend(y_pred.tolist())

        test_copy = test[['run_folder', 'difficulty']].copy()
        test_copy['pred'] = y_pred

        subj_rt, subj_rp = [], []
        for _, rg in test_copy.groupby('run_folder'):
            rt = int(rg['difficulty'].iloc[0])
            rp = int(scipy_mode(rg['pred'].values, keepdims=False).mode)
            all_true_r.append(rt)
            all_pred_r.append(rp)
            subj_rt.append(rt)
            subj_rp.append(rp)

        subj_acc_r = accuracy_score(subj_rt, subj_rp) if subj_rt else 0.0
        subj_acc_w = accuracy_score(y_test, y_pred)

        per_subject.append({
            'subject':   test_subj,
            'run_acc':   round(float(subj_acc_r), 4),
            'win_acc':   round(float(subj_acc_w), 4),
            'n_runs':    len(subj_rt),
            'n_correct': int(round(subj_acc_r * len(subj_rt))),
        })

        print(f"  [{si+1:2d}/{len(subjects)}] {test_subj} | "
              f"run={subj_acc_r*100:.1f}% | win={subj_acc_w*100:.1f}% | feats={len(features)}")

    all_true_w = np.array(all_true_w)
    all_pred_w = np.array(all_pred_w)
    all_true_r = np.array(all_true_r)
    all_pred_r = np.array(all_pred_r)

    cm      = confusion_matrix(all_true_w, all_pred_w, labels=[1, 2, 3, 4]).tolist()
    win_acc = float(accuracy_score(all_true_w, all_pred_w))
    run_acc = float(accuracy_score(all_true_r, all_pred_r))
    adj_acc = float(np.mean(np.abs(all_true_r - all_pred_r) <= 1))
    bin_acc = float(accuracy_score((all_true_r >= 3).astype(int),
                                   (all_pred_r >= 3).astype(int)))

    n_folds  = len(subjects)
    feat_imp = []
    for feat, cnt in feat_usage.most_common(25):
        feat_imp.append({
            'feature':  feat,
            'folds':    cnt,
            'pct':      round(cnt / n_folds * 100, 1),
            'modality': get_modality(feat),
        })

    # Best subject for run replay
    best = max(per_subject, key=lambda x: x['run_acc'])
    best_windows = [r for r in window_records if r['subject'] == best['subject']]
    best_windows.sort(key=lambda r: (r['run'], r['window']))

    # Accuracy progression (actual experimental results from this project)
    progression = [
        {'label': 'Random Chance',       'value': 25.0},
        {'label': 'Feature Extraction',  'value': 34.0},
        {'label': '+ Spline Basis',      'value': 35.0},
        {'label': '+ Rest Z-scoring',    'value': 51.0},
        {'label': '+ Run-level Voting',  'value': round(run_acc * 100, 1)},
    ]

    payload = {
        'summary': {
            'total_windows': int(len(all_true_w)),
            'subjects':      int(len(subjects)),
            'runs':          int(len(all_true_r)),
            'window_acc':    round(win_acc * 100, 1),
            'run_acc':       round(run_acc * 100, 1),
            'adjacent_acc':  round(adj_acc * 100, 1),
            'binary_acc':    round(bin_acc * 100, 1),
        },
        'progression':        progression,
        'confusion_matrix':   cm,
        'per_subject':        sorted(per_subject, key=lambda x: x['run_acc'], reverse=True),
        'feature_importance': feat_imp,
        'best_subject': {
            'subject': best['subject'],
            'run_acc': round(best['run_acc'] * 100, 1),
            'windows': best_windows,
        },
    }

    OUTPUT_JS.write_text(f"window.DEMO_DATA = {json.dumps(payload, indent=2)};\n",
                         encoding='utf-8')

    print(f"\nSaved: {OUTPUT_JS}")
    print(f"Run acc:  {run_acc*100:.1f}% | Window acc: {win_acc*100:.1f}%")
    print(f"Best subject: {best['subject']} ({best['run_acc']*100:.1f}% run-level)")
    print(f"\nNow open demo_dashboard.html in your browser.")

if __name__ == '__main__':
    main()
