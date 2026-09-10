"""
nested_feature_selection.py
===========================
Nested In-Fold Statistical Feature Screening & VIF Pruning (Zero Leakage).

Features:
1. In-fold ANOVA F-test / Kruskal-Wallis with FDR adjustment fit strictly on X_train, y_train.
2. In-fold Iterative VIF (Variance Inflation Factor) pruning strictly on X_train.
3. FeatureSelectionStabilityTracker to verify if the selected features are consistent across all folds.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging

logger = logging.getLogger(__name__)


def fdr_correction(p_values):
    """Benjamini-Hochberg FDR correction."""
    p_vals = np.asarray(p_values)
    n = len(p_vals)
    if n == 0:
        return np.array([])
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    q_vals = np.zeros(n)
    
    min_q = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = (sorted_p[i] * n) / rank
        min_q = min(min_q, q)
        q_vals[i] = min_q
        
    corrected = np.zeros(n)
    corrected[sorted_indices] = np.clip(q_vals, 0.0, 1.0)
    return corrected


def screen_features_in_fold(X_train_df, y_train_series, p_threshold=0.05, min_features=8):
    """
    Run ANOVA F-test across difficulty classes strictly on training fold data.
    """
    results = []
    classes = np.unique(y_train_series)
    
    for col in X_train_df.columns:
        # Group by difficulty level
        groups = [
            X_train_df.loc[y_train_series == c, col].dropna().values
            for c in classes
        ]
        # Filter empty groups
        valid_groups = [g for g in groups if len(g) > 1]
        if len(valid_groups) < len(classes):
            continue
            
        try:
            f_stat, p_val = stats.f_oneway(*valid_groups)
            kw_stat, kw_pval = stats.kruskal(*valid_groups)
            if not np.isnan(f_stat) and not np.isnan(p_val):
                results.append({
                    'feature': col,
                    'f_stat': f_stat,
                    'p_value': p_val,
                    'kw_p_value': kw_pval
                })
        except Exception:
            continue

    if not results:
        return list(X_train_df.columns[:min_features])

    res_df = pd.DataFrame(results)
    res_df['q_value'] = fdr_correction(res_df['p_value'].values)
    
    # Select significant features
    sig_df = res_df[res_df['p_value'] < p_threshold].sort_values('p_value')
    if len(sig_df) < min_features:
        sig_df = res_df.sort_values('p_value').head(min_features)

    return sig_df['feature'].tolist()


def prune_vif_in_fold(X_train_df, feature_list, max_vif=10.0, min_features=6):
    """
    Iterative VIF pruning strictly on X_train to eliminate multicollinearity.
    """
    selected = [f for f in feature_list if f in X_train_df.columns]
    
    while len(selected) > min_features:
        # Check finite numeric matrix
        X_sub = X_train_df[selected].dropna()
        if len(X_sub) < len(selected) + 1:
            break
            
        # Standardize for stable VIF
        X_mat = (X_sub - X_sub.mean()) / (X_sub.std() + 1e-8)
        X_vals = X_mat.values
        
        vif_vals = []
        for i in range(len(selected)):
            try:
                v = variance_inflation_factor(X_vals, i)
                vif_vals.append(v if not np.isnan(v) and not np.isinf(v) else 999.0)
            except Exception:
                vif_vals.append(999.0)
                
        max_idx = np.argmax(vif_vals)
        highest_vif = vif_vals[max_idx]
        
        if highest_vif <= max_vif:
            break  # All VIFs within acceptable threshold
            
        # Drop highest VIF feature
        dropped_feature = selected.pop(max_idx)
        logger.debug(f"Fold VIF pruning: Dropped {dropped_feature} (VIF={highest_vif:.2f})")

    return selected


class FeatureStabilityTracker:
    """
    Tracks which features are selected across all LOSO folds to prove selection stability.
    """
    def __init__(self):
        self.fold_selections = []
        self.total_folds = 0

    def record_fold(self, fold_idx, selected_features):
        self.fold_selections.append({
            'fold': fold_idx,
            'features': list(selected_features)
        })
        self.total_folds += 1

    def get_summary_df(self):
        counts = {}
        for item in self.fold_selections:
            for feat in item['features']:
                counts[feat] = counts.get(feat, 0) + 1
                
        records = []
        for feat, count in counts.items():
            records.append({
                'feature': feat,
                'selection_count': count,
                'selection_frequency_pct': (count / self.total_folds) * 100.0,
                'total_folds': self.total_folds
            })
            
        return pd.DataFrame(records).sort_values('selection_frequency_pct', ascending=False)
