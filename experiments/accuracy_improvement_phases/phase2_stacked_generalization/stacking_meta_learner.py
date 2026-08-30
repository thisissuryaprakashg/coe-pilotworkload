"""
Calibrated Stacked Generalization Meta-Learner
================================================
Novel technique: Combine all 4 existing models' calibrated probability outputs
across multiple feature subsets using a meta-learner.

Architecture:
    Layer 0 (Base models):
        - LR on raw features → 4 class probabilities
        - RF on raw features → 4 class probabilities
        - SVM on raw features → 4 class probabilities
        - GBDT on raw features → 4 class probabilities
        - LR on enhanced features → 4 class probabilities
        - RF on enhanced features → 4 class probabilities
        - SVM on enhanced features → 4 class probabilities
        - GBDT on enhanced features → 4 class probabilities
        Total: 8 models × 4 probabilities = 32 meta-features

    Layer 1 (Meta-learner):
        - Logistic Regression on 32 meta-features → final prediction
        - Learns which base model to trust for which feature modality

Why this is novel:
    Standard stacking uses one feature set. Cross-feature-subset stacking
    exploits the fact that GBDT excels on oculomotor features while LR excels
    on multimodal features. The meta-learner discovers these complementarities.

Protocol:
    For LOSO without leakage:
    1. Hold out test subject S
    2. On remaining subjects, use 5-fold CV to generate out-of-fold
       base predictions (stacking features for training)
    3. Train meta-learner on these stacking features
    4. Train base models on full training set → predict test subject
    5. Meta-learner predicts on test subject's base predictions
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import logging

logger = logging.getLogger(__name__)


def get_base_models():
    """Return the same model configurations as the baseline."""
    return {
        'LR': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=1000, C=1.0, random_state=42
            ))
        ]),
        'RF': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
            ))
        ]),
        'SVM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', CalibratedClassifierCV(
                SVC(kernel='rbf', C=1.0, random_state=42), ensemble=False
            ))
        ]),
        'GBDT': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42
            ))
        ]),
    }


def get_meta_learner():
    """
    Meta-learner: simple LR to combine base model probabilities.
    Using LR because:
    - Low variance (only 32 meta-features, need to avoid overfitting)
    - Interpretable (can see which base model gets most weight)
    - Fast
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='lbfgs', max_iter=1000, C=0.5, random_state=42
        ))
    ])


def generate_stacking_features(X_train, y_train, feature_sets, base_models,
                               n_inner_folds=5, random_state=42):
    """
    Generate out-of-fold base model predictions for stacking.

    Uses StratifiedKFold inner CV on the training set to create unbiased
    stacking features without data leakage.

    Args:
        X_train: dict mapping feature_set_name → DataFrame of features
        y_train: array of labels
        feature_sets: dict mapping name → list of column names
        base_models: dict mapping model_name → Pipeline
        n_inner_folds: number of inner CV folds

    Returns:
        meta_features: (n_train, n_models * n_feature_sets * n_classes) array
        meta_feature_names: list of feature names
    """
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))

    meta_features = np.zeros((n_samples, 0))
    meta_feature_names = []

    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)

    for feat_name, feat_cols in feature_sets.items():
        X_feat = X_train[feat_cols].astype(float).values

        for model_name, model_pipe in base_models.items():
            # Out-of-fold predictions
            oof_probs = np.zeros((n_samples, n_classes))

            for train_idx, val_idx in skf.split(X_feat, y_train):
                X_tr, X_val = X_feat[train_idx], X_feat[val_idx]
                y_tr = y_train[train_idx]

                pipe = clone(model_pipe)
                try:
                    pipe.fit(X_tr, y_tr)
                    probs = pipe.predict_proba(X_val)
                    oof_probs[val_idx] = probs
                except Exception as e:
                    logger.debug(f"Inner fold failed for {model_name}/{feat_name}: {e}")
                    # Fallback: uniform probs
                    oof_probs[val_idx] = 1.0 / n_classes

            meta_features = np.hstack([meta_features, oof_probs])
            for cls in range(n_classes):
                meta_feature_names.append(f"{model_name}_{feat_name}_class{cls+1}")

    return meta_features, meta_feature_names


def generate_test_stacking_features(X_train_full, y_train_full, X_test,
                                    feature_sets, base_models):
    """
    Generate stacking features for test data by training base models on
    the full training set and predicting on test.

    Returns:
        test_meta_features: (n_test, n_meta_features) array
    """
    n_test = len(X_test) if not isinstance(X_test, dict) else len(next(iter(X_test.values())))
    n_classes = len(np.unique(y_train_full))

    test_meta = np.zeros((n_test, 0))

    for feat_name, feat_cols in feature_sets.items():
        X_tr = X_train_full[feat_cols].astype(float).values
        X_te = X_test[feat_cols].astype(float).values

        for model_name, model_pipe in base_models.items():
            pipe = clone(model_pipe)
            try:
                pipe.fit(X_tr, y_train_full)
                probs = pipe.predict_proba(X_te)
            except Exception:
                probs = np.full((n_test, n_classes), 1.0 / n_classes)

            test_meta = np.hstack([test_meta, probs])

    return test_meta


def run_stacked_loso(df, feature_sets, target_col='difficulty_ground_truth',
                     subject_col='subject'):
    """
    Full LOSO evaluation with calibrated stacking.

    For each test subject:
    1. Generate out-of-fold stacking features on training set
    2. Train meta-learner on stacking features
    3. Generate test stacking features
    4. Meta-learner predicts on test

    Returns:
        fold_results: list of per-fold metrics
    """
    subjects = df[subject_col].unique()
    base_models = get_base_models()
    fold_results = []
    all_y_true = []
    all_y_pred = []

    logger.info(f"Running stacked LOSO ({len(subjects)} folds)...")

    for fold_num, test_subject in enumerate(subjects, 1):
        train_mask = df[subject_col] != test_subject
        test_mask = df[subject_col] == test_subject

        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)

        y_train = df_train[target_col].astype(int).values
        y_test = df_test[target_col].astype(int).values

        if len(y_test) == 0:
            continue

        # Step 1: Generate training stacking features (out-of-fold)
        train_meta, meta_names = generate_stacking_features(
            df_train, y_train, feature_sets, base_models
        )

        # Step 2: Generate test stacking features
        test_meta = generate_test_stacking_features(
            df_train, y_train, df_test, feature_sets, base_models
        )

        # Step 3: Train and predict with meta-learner
        meta_learner = get_meta_learner()
        try:
            meta_learner.fit(train_meta, y_train)
            y_pred = meta_learner.predict(test_meta)
        except Exception as e:
            logger.warning(f"  Meta-learner failed for {test_subject}: {e}")
            # Fallback: use GBDT on enhanced features
            fallback_cols = list(feature_sets.values())[-1]  # last feature set
            X_tr = df_train[fallback_cols].astype(float)
            X_te = df_test[fallback_cols].astype(float)
            fallback_pipe = clone(base_models['GBDT'])
            fallback_pipe.fit(X_tr, y_train)
            y_pred = fallback_pipe.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        fold_results.append({
            'subject': str(test_subject),
            'accuracy': acc,
            'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'n_test': len(y_test),
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        if fold_num % 5 == 0 or fold_num == len(subjects):
            avg_acc = np.mean([r['accuracy'] for r in fold_results])
            logger.info(f"  Fold {fold_num}/{len(subjects)} - Running avg accuracy: {avg_acc:.4f}")

    return fold_results, np.array(all_y_true), np.array(all_y_pred)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Stacking Meta-Learner module loaded.")
    print("Use in run_phase2.py for full evaluation.")
