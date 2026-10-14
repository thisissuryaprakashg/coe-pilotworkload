"""
tda_features.py
===============
Topological Data Analysis features from raw ECG and EDA signals.
Uses persistent homology via:
  1. Sub/super-level set filtration (pure Python, H0)
  2. Takens embedding + ripser (H0 + H1 loops)
Output: tda_features.csv keyed by (subject, session, run_folder, window).
No data leakage — all features computed per window from raw signal only.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

from src.data_pipeline.new_data_pipeline import (
    load_run_streams, get_ref_time,
    DATASET_ROOT, WINDOW_DN, BAD_RUNS, is_run_bad,
)

try:
    from ripser import ripser as ripser_compute
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV = ROOT / 'outputs' / 'tda_features.csv'


# ============================================================
# H0 PERSISTENCE (pure Python, sublevel set filtration)
# ============================================================

def sublevel_h0(values):
    """H0 persistence via sublevel set filtration on 1D signal.
    Returns (N, 2) array of (birth, death) pairs."""
    n = len(values)
    if n < 3:
        return np.empty((0, 2))

    order = np.argsort(values, kind='stable')
    parent = np.arange(n)
    comp_birth = values.copy()
    visited = np.zeros(n, dtype=bool)
    pairs = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in order:
        visited[i] = True
        for nb in (i - 1, i + 1):
            if 0 <= nb < n and visited[nb]:
                ri, rn = find(i), find(nb)
                if ri != rn:
                    bi, bn = comp_birth[ri], comp_birth[rn]
                    if bi > bn:
                        pairs.append((bi, values[i]))
                        parent[ri] = rn
                    elif bn > bi:
                        pairs.append((bn, values[i]))
                        parent[rn] = ri
                    else:
                        pairs.append((bi, values[i]))
                        parent[ri] = rn

    dgm = np.array(pairs) if pairs else np.empty((0, 2))
    if len(dgm) > 0:
        dgm = dgm[dgm[:, 1] - dgm[:, 0] > 0]
    return dgm


def superlevel_h0(values):
    """H0 persistence via superlevel set filtration (captures peaks)."""
    dgm_neg = sublevel_h0(-values)
    if len(dgm_neg) == 0:
        return np.empty((0, 2))
    return np.column_stack([-dgm_neg[:, 1], -dgm_neg[:, 0]])


# ============================================================
# FEATURE EXTRACTION FROM PERSISTENCE DIAGRAM
# ============================================================

def pers_entropy(pers):
    if len(pers) == 0 or np.sum(pers) == 0:
        return 0.0
    p = pers / np.sum(pers)
    return float(-np.sum(p * np.log(p + 1e-12)))


def dgm_features(dgm, prefix):
    empty = {
        f'{prefix}_count': 0, f'{prefix}_max_pers': 0,
        f'{prefix}_mean_pers': 0, f'{prefix}_std_pers': 0,
        f'{prefix}_sum_pers': 0, f'{prefix}_entropy': 0,
        f'{prefix}_ratio': 0, f'{prefix}_n_signif': 0,
        f'{prefix}_mean_mid': 0, f'{prefix}_std_mid': 0,
    }
    if len(dgm) == 0:
        return empty

    pers = dgm[:, 1] - dgm[:, 0]
    mid = (dgm[:, 0] + dgm[:, 1]) / 2.0
    med = np.median(pers)

    return {
        f'{prefix}_count': len(pers),
        f'{prefix}_max_pers': float(np.max(pers)),
        f'{prefix}_mean_pers': float(np.mean(pers)),
        f'{prefix}_std_pers': float(np.std(pers)) if len(pers) > 1 else 0.0,
        f'{prefix}_sum_pers': float(np.sum(pers)),
        f'{prefix}_entropy': pers_entropy(pers),
        f'{prefix}_ratio': float(np.max(pers) / np.sum(pers)) if np.sum(pers) > 0 else 0.0,
        f'{prefix}_n_signif': int(np.sum(pers > med)) if med > 0 else 0,
        f'{prefix}_mean_mid': float(np.mean(mid)),
        f'{prefix}_std_mid': float(np.std(mid)) if len(mid) > 1 else 0.0,
    }


# ============================================================
# TAKENS EMBEDDING + RIPSER (H0 + H1)
# ============================================================

def takens_embed(signal, delay, dim):
    n = len(signal) - (dim - 1) * delay
    if n <= 0:
        return np.empty((0, dim))
    out = np.zeros((n, dim))
    for d in range(dim):
        out[:, d] = signal[d * delay: d * delay + n]
    return out


def takens_features(signal, delay, dim, prefix, max_pts=500):
    if not HAS_RIPSER or len(signal) < (dim - 1) * delay + 20:
        return {}

    cloud = takens_embed(signal, delay, dim)
    if len(cloud) > max_pts:
        idx = np.linspace(0, len(cloud) - 1, max_pts).astype(int)
        cloud = cloud[idx]

    # normalize point cloud
    mu, sigma = cloud.mean(0), cloud.std(0)
    sigma[sigma < 1e-10] = 1.0
    cloud = (cloud - mu) / sigma

    result = ripser_compute(cloud, maxdim=1)
    feats = {}
    for d, label in [(0, 'h0'), (1, 'h1')]:
        dgm = result['dgms'][d]
        finite = dgm[np.isfinite(dgm[:, 1])]
        feats.update(dgm_features(finite, f'{prefix}_tk{label}'))
    return feats


# ============================================================
# PER-WINDOW TDA
# ============================================================

def tda_ecg(ecg_df, w0, w1, fs=512):
    if ecg_df is None or len(ecg_df) == 0:
        return {}
    t = ecg_df.iloc[:, 0].values
    sig = ecg_df.iloc[:, 1].values
    seg = sig[(t >= w0) & (t < w1)]
    if len(seg) < int(fs * 5):
        return {}

    seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-10)

    # downsample to ~100 Hz for cleaner topology
    step = max(1, int(fs / 100))
    ds = seg[::step]

    feats = {}
    feats.update(dgm_features(sublevel_h0(ds), 'ecg_sub'))
    feats.update(dgm_features(superlevel_h0(ds), 'ecg_sup'))

    # Takens H0+H1 (captures periodicity)
    feats.update(takens_features(ds, delay=10, dim=2, prefix='ecg'))
    return feats


def tda_eda(eda_df, w0, w1, fs=128):
    if eda_df is None or len(eda_df) == 0:
        return {}
    t = eda_df.iloc[:, 0].values
    sig = eda_df.iloc[:, 1].values
    seg = sig[(t >= w0) & (t < w1)]
    if len(seg) < int(fs * 5):
        return {}

    seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-10)

    # downsample to ~32 Hz
    step = max(1, int(fs / 32))
    ds = seg[::step]

    feats = {}
    feats.update(dgm_features(sublevel_h0(ds), 'eda_sub'))
    feats.update(dgm_features(superlevel_h0(ds), 'eda_sup'))
    feats.update(takens_features(ds, delay=8, dim=2, prefix='eda'))
    return feats


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TDA FEATURE EXTRACTION (persistent homology)")
    print("=" * 60)
    print(f"ripser: {HAS_RIPSER}")

    ils_dir = DATASET_ROOT / 'task-ils'
    subjects = sorted([d for d in os.listdir(ils_dir) if d.startswith('sub-cp')])
    print(f"subjects: {len(subjects)}")

    records = []

    for si, subj in enumerate(subjects):
        subj_path = ils_dir / subj
        n_win = 0

        for sess in sorted(os.listdir(subj_path)):
            if not sess.startswith('ses-'):
                continue
            sess_path = subj_path / sess

            for run_dir in sorted(os.listdir(sess_path)):
                if not run_dir.startswith('level-'):
                    continue

                if is_run_bad(subj, run_dir):
                    continue

                # parse difficulty from folder name (same as pipeline)
                m = re.search(r'level-(\d+)', run_dir)
                diff = int(m.group(1)) if m else -1
                if diff <= 0:
                    continue

                run_path = sess_path / run_dir
                s = load_run_streams(str(run_path))
                ref = get_ref_time(s)
                if ref is None or len(ref) < 100:
                    continue

                w_idx = 0
                w0 = ref[0]
                while w0 + WINDOW_DN * 0.5 < ref[-1]:
                    w1 = w0 + WINDOW_DN

                    rec = {
                        'subject': subj, 'session': sess,
                        'run_folder': run_dir, 'window': w_idx,
                        'difficulty': diff,
                    }
                    rec.update(tda_ecg(s.get('ecg'), w0, w1, s.get('ecg_fs', 512)))
                    rec.update(tda_eda(s.get('eda'), w0, w1, s.get('eda_fs', 128)))

                    records.append(rec)
                    w_idx += 1
                    w0 = w1
                n_win += w_idx

        print(f"  [{si + 1:2d}/{len(subjects)}] {subj} | windows: {n_win}")

    df = pd.DataFrame(records)

    meta = {'subject', 'session', 'run_folder', 'window', 'difficulty'}
    feat_cols = [c for c in df.columns if c not in meta]

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"rows: {len(df)}")
    print(f"TDA features: {len(feat_cols)}")

    for pfx, label in [('ecg_sub', 'ECG sublevel H0'), ('ecg_sup', 'ECG superlevel H0'),
                        ('ecg_tkh0', 'ECG Takens H0'), ('ecg_tkh1', 'ECG Takens H1'),
                        ('eda_sub', 'EDA sublevel H0'), ('eda_sup', 'EDA superlevel H0'),
                        ('eda_tkh0', 'EDA Takens H0'), ('eda_tkh1', 'EDA Takens H1')]:
        cols = [c for c in feat_cols if c.startswith(pfx)]
        if cols:
            pct = df[cols].notna().mean().mean() * 100
            print(f"  {label:20s}: {len(cols):3d} features | {pct:5.1f}% coverage")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsaved: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
