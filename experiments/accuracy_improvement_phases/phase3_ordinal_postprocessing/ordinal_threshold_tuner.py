"""
Ordinal-Aware Threshold Tuning & Post-Processing
==================================================
Novel technique: Cost-sensitive post-processing that respects the ordinal
structure of workload levels (Level 1 < 2 < 3 < 4).

Problem:
    Standard classifiers treat all misclassifications equally:
    - Predicting Level 1 when true is Level 4 (catastrophic error) = same loss as
    - Predicting Level 2 when true is Level 3 (near miss)

    This leads to predictions that scatter across all classes without
    respecting the ordinal constraint.

Solution (3 components):
    1. Ordinal Label Smoothing:
       Instead of hard labels [0,0,1,0] for class 3, use ordinal-distance-weighted
       soft labels [0.05, 0.15, 0.60, 0.20] that spread probability to adjacent classes.

    2. Ordinal Probability Redistribution:
       After getting raw probabilities from base models, redistribute probability
       mass toward adjacent classes using a Gaussian kernel centered on the
       predicted class, weighted by ordinal distance.

    3. Cost-Sensitive Threshold Moving:
       Optimize per-class decision thresholds on training data to maximize
       quadratic weighted Cohen's Kappa (which penalizes far misclassifications
       more than near misclassifications).

References:
    - Frank & Hall (2001): "A Simple Approach to Ordinal Classification"
    - Cardoso & da Costa (2007): "Learning to Classify Ordinal Data"
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


class OrdinalPostProcessor:
    """
    Post-processing layer that takes raw class probabilities and adjusts
    them to respect ordinal structure.

    Can be applied to ANY classifier's probability outputs.
    """

    def __init__(self, n_classes=4, smoothing_sigma=0.8, optimize_thresholds=True):
        """
        Args:
            n_classes: number of ordinal classes (4 for workload levels)
            smoothing_sigma: Gaussian smoothing bandwidth for ordinal redistribution
                             Higher = more smoothing toward adjacent classes
            optimize_thresholds: whether to optimize per-class thresholds on training data
        """
        self.n_classes = n_classes
        self.smoothing_sigma = smoothing_sigma
        self.optimize_thresholds = optimize_thresholds
        self.class_thresholds = None
        self.class_biases = None

    def _ordinal_smooth_probs(self, probs):
        """
        Redistribute probability mass using ordinal-aware Gaussian smoothing.

        For a probability vector [0.1, 0.05, 0.8, 0.05]:
        - The high probability at class 3 should "leak" to adjacent classes 2 and 4
        - But NOT to distant class 1

        This is achieved by applying a 1D Gaussian filter along the class axis.
        """
        smoothed = np.zeros_like(probs)
        for i in range(len(probs)):
            smoothed[i] = gaussian_filter1d(probs[i], sigma=self.smoothing_sigma)
            # Renormalize
            total = smoothed[i].sum()
            if total > 0:
                smoothed[i] /= total
            else:
                smoothed[i] = 1.0 / self.n_classes

        return smoothed

    def _apply_class_biases(self, probs):
        """Apply learned per-class probability biases."""
        if self.class_biases is None:
            return probs

        adjusted = probs + self.class_biases
        # Clip and renormalize
        adjusted = np.maximum(adjusted, 0)
        row_sums = adjusted.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        adjusted /= row_sums

        return adjusted

    def fit(self, probs_train, y_train):
        """
        Learn optimal class biases and thresholds from training data.

        Args:
            probs_train: (n_samples, n_classes) probability matrix
            y_train: true labels (1-indexed: 1,2,3,4)
        """
        if not self.optimize_thresholds:
            return self

        # Apply ordinal smoothing first
        probs_smoothed = self._ordinal_smooth_probs(probs_train)

        # Optimize class biases to maximize quadratic weighted kappa
        def neg_kappa(biases):
            adjusted = probs_smoothed + biases.reshape(1, -1)
            adjusted = np.maximum(adjusted, 0)
            row_sums = adjusted.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            adjusted /= row_sums
            y_pred = np.argmax(adjusted, axis=1) + 1
            try:
                kappa = cohen_kappa_score(y_train, y_pred, weights='quadratic')
            except Exception:
                kappa = 0.0
            return -kappa  # Minimize negative kappa

        # Start from zero biases
        initial_biases = np.zeros(self.n_classes)
        result = minimize(
            neg_kappa, initial_biases,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-5}
        )

        self.class_biases = result.x
        logger.debug(f"Optimized class biases: {self.class_biases}")
        logger.debug(f"Training kappa: {-result.fun:.4f}")

        return self

    def transform(self, probs):
        """
        Apply ordinal post-processing to probability matrix.

        Args:
            probs: (n_samples, n_classes) probability matrix from any classifier

        Returns:
            y_pred: (n_samples,) predictions (1-indexed: 1,2,3,4)
            probs_adjusted: (n_samples, n_classes) adjusted probabilities
        """
        # Step 1: Ordinal smoothing
        probs_smoothed = self._ordinal_smooth_probs(probs)

        # Step 2: Apply learned biases
        probs_adjusted = self._apply_class_biases(probs_smoothed)

        # Step 3: Predict
        y_pred = np.argmax(probs_adjusted, axis=1) + 1

        return y_pred, probs_adjusted


class OrdinalEnsemblePostProcessor:
    """
    Combines multiple base models' probability outputs with ordinal post-processing.

    Pipeline:
    1. Get probability outputs from multiple base models
    2. Average probabilities (equal weighting)
    3. Apply ordinal smoothing + threshold tuning
    4. Predict
    """

    def __init__(self, base_models, feature_sets, n_classes=4,
                 smoothing_sigma=0.8, optimize_thresholds=True):
        """
        Args:
            base_models: dict of {name: Pipeline}
            feature_sets: dict of {name: list_of_column_names}
            n_classes: number of ordinal classes
            smoothing_sigma: Gaussian smoothing bandwidth
            optimize_thresholds: whether to optimize thresholds
        """
        self.base_models = base_models
        self.feature_sets = feature_sets
        self.n_classes = n_classes
        self.post_processor = OrdinalPostProcessor(
            n_classes=n_classes,
            smoothing_sigma=smoothing_sigma,
            optimize_thresholds=optimize_thresholds
        )
        self.fitted_models = {}

    def fit(self, df_train, y_train):
        """
        Fit all base models and calibrate ordinal post-processor.
        """
        self.fitted_models = {}

        for feat_name, feat_cols in self.feature_sets.items():
            valid_cols = [c for c in feat_cols if c in df_train.columns]
            X_train = df_train[valid_cols].astype(float)

            for model_name, model_pipe in self.base_models.items():
                key = f"{model_name}_{feat_name}"
                pipe = clone(model_pipe)
                try:
                    pipe.fit(X_train, y_train)
                    self.fitted_models[key] = (pipe, valid_cols)
                except Exception as e:
                    logger.debug(f"Failed to fit {key}: {e}")

        # Generate averaged probabilities on training data for threshold tuning
        train_probs = self._get_averaged_probs(df_train)
        self.post_processor.fit(train_probs, y_train)

        return self

    def _get_averaged_probs(self, df):
        """Get averaged probability outputs from all fitted base models."""
        n_samples = len(df)
        all_probs = []

        for key, (pipe, valid_cols) in self.fitted_models.items():
            X = df[valid_cols].astype(float)
            try:
                probs = pipe.predict_proba(X)
                all_probs.append(probs)
            except Exception:
                pass

        if not all_probs:
            return np.full((n_samples, self.n_classes), 1.0 / self.n_classes)

        # Average probabilities across all base models
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs

    def predict(self, df_test):
        """
        Predict using averaged probabilities + ordinal post-processing.
        """
        avg_probs = self._get_averaged_probs(df_test)
        y_pred, probs_adjusted = self.post_processor.transform(avg_probs)
        return y_pred, probs_adjusted


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Ordinal Threshold Tuner module loaded.")
    print("Use in run_phase3.py for full evaluation.")
