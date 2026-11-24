"""
Generate the MODALSIM94 LOSO dashboard payload.

Output: modalsim_demo_data.js
"""

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent / "modalsim_framework" / "modeling"))
from benchmark_models import OrdinalLogisticRegression, fit_transform_splines


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "modalsim_framework" / "outputs" / "modalsim_normalized_features.csv"
OUTPUT_JS = ROOT / "modalsim_demo_data.js"


def modality(feature):
    feature = feature.lower()
    if feature.startswith("delta_pulse") or "pulse" in feature:
        return "ECG"
    if "resp" in feature:
        return "EDA"
    if "eye" in feature:
        return "Eye"
    if "flight" in feature:
        return "Head"
    return "Other"


def fit_vif(X, max_vif=5.0):
    keep = list(range(X.shape[1]))
    while len(keep) > 1:
        sub = X[:, keep]
        corr = np.corrcoef(sub, rowvar=False)
        if np.isnan(corr).any():
            break
        try:
            inv = np.linalg.inv(corr + 1e-6 * np.eye(corr.shape[0]))
            vifs = np.diag(inv)
        except np.linalg.LinAlgError:
            break
        if np.max(vifs) <= max_vif:
            break
        del keep[np.argmax(vifs)]
    return keep


def main():
    warnings.filterwarnings("ignore")
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c.startswith("delta_")]
    subjects = sorted(df["subject"].unique())

    linear_preds = []
    spline_preds = []
    ordinal_preds = []
    truth = []
    feature_usage = Counter()
    per_subject = []
    replay_by_subject = {}

    for test_subject in subjects:
        train = df[df["subject"] != test_subject]
        test = df[df["subject"] == test_subject]

        train_medians = train[feature_cols].median().fillna(0.0)
        X_train = np.nan_to_num(train[feature_cols].fillna(train_medians).values)
        X_test = np.nan_to_num(test[feature_cols].fillna(train_medians).values)
        y_train = train["target"].values
        y_test = test["target"].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        scores = np.nan_to_num(f_classif(X_train, y_train)[0], nan=-1.0)
        prelim = np.argsort(scores)[::-1][:16]
        keep = fit_vif(X_train[:, prelim])[:8]
        selected = prelim[keep]
        X_train = X_train[:, selected]
        X_test = X_test[:, selected]
        feature_usage.update(feature_cols[i] for i in selected)

        linear = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        linear.fit(X_train, y_train)
        linear_pred = linear.predict(X_test)

        X_train_spline, X_test_spline = fit_transform_splines(X_train, X_test, n_knots=3)
        spline = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        spline.fit(X_train_spline, y_train)
        spline_pred = spline.predict(X_test_spline)

        ordinal = OrdinalLogisticRegression(C=0.5)
        ordinal.fit(X_train, y_train)
        ordinal_pred = ordinal.predict(X_test)
        spline_proba = spline.predict_proba(X_test_spline)

        truth.extend(y_test)
        linear_preds.extend(linear_pred)
        spline_preds.extend(spline_pred)
        ordinal_preds.extend(ordinal_pred)

        subject_acc = accuracy_score(y_test, spline_pred)
        replay_by_subject[str(test_subject)] = []
        for row, true, pred, probabilities in zip(
            test.itertuples(index=False), y_test, spline_pred, spline_proba
        ):
            replay_by_subject[str(test_subject)].append({
                "scenario": str(row.scenario),
                "true": int(true) + 1,
                "pred": int(pred) + 1,
                "p1": round(float(probabilities[0]), 4),
                "p2": round(float(probabilities[1]), 4),
                "p3": round(float(probabilities[2]), 4),
                "metrics": {
                    "Pulse HR": round(float(row.pulse_hr_mean), 2),
                    "Respiration": round(float(row.resp_bpm_mean), 2),
                    "Pupil area": round(float(row.eye_pupil_area_mean), 2),
                    "Flight error": round(float(row.flight_error_mean), 2),
                },
            })
        per_subject.append({
            "subject": str(test_subject),
            "run_acc": round(float(subject_acc), 4),
            "win_acc": round(float(subject_acc), 4),
            "n_runs": len(y_test),
            "n_correct": int(round(subject_acc * len(y_test))),
        })

    y_true = np.array(truth)
    linear_preds = np.array(linear_preds)
    spline_preds = np.array(spline_preds)
    ordinal_preds = np.array(ordinal_preds)

    cm = confusion_matrix(y_true, spline_preds, labels=[0, 1, 2]).tolist()
    spline_acc = accuracy_score(y_true, spline_preds)
    adjacent = np.mean(np.abs(y_true - spline_preds) <= 1)
    binary = accuracy_score((y_true > 0).astype(int), (spline_preds > 0).astype(int))

    methods = [
        {"name": "Spline Logistic", "tag": "Leak-free spline basis · LOSO", "acc": round(spline_acc * 100, 1),
         "badge": "close", "color": "#00d4ff", "verdict": "Interpretable nonlinear baseline using subject-relative delta features. The session replay uses this model's probabilities."},
        {"name": "Linear Logistic", "tag": "L2 logistic · LOSO", "acc": round(accuracy_score(y_true, linear_preds) * 100, 1),
         "badge": "close", "color": "#22c55e", "verdict": "Linear baseline using subject-relative delta features."},
        {"name": "Ordinal Logistic", "tag": "Ordered three-tier baseline · LOSO", "acc": round(accuracy_score(y_true, ordinal_preds) * 100, 1),
         "badge": "close", "color": "#3b82f6", "verdict": "Multinomial ordered baseline under the same LOSO protocol."},
        {"name": "XGBoost", "tag": "Documented nonlinear benchmark · LOSO", "acc": 74.8,
         "badge": "best", "color": "#a855f7", "verdict": "Best documented MODALSIM94 benchmark result using all subject-relative modalities."},
        {"name": "Random Chance", "tag": "3-class uniform baseline", "acc": 33.3,
         "badge": "chance", "color": "#475569", "verdict": "Reference floor for three workload tiers."},
    ]

    feature_importance = [
        {"feature": feature, "folds": count, "pct": round(count / len(subjects) * 100, 1),
         "modality": modality(feature)}
        for feature, count in feature_usage.most_common(27)
    ]

    best = max(per_subject, key=lambda row: row["run_acc"])
    payload = {
        "dataset": "MODALSIM94",
        "summary": {
            "total_windows": int(len(df)),
            "subjects": len(subjects),
            "runs": len(df),
            "window_acc": round(spline_acc * 100, 1),
            "run_acc": round(spline_acc * 100, 1),
            "adjacent_acc": round(adjacent * 100, 1),
            "binary_acc": round(binary * 100, 1),
        },
        "progression": [
            {"label": "Random Chance", "value": 33.3},
            {"label": "Raw Features", "value": 40.4},
            {"label": "Subject Deltas", "value": round(spline_acc * 100, 1)},
        ],
        "confusion_matrix": cm,
        "per_subject": sorted(per_subject, key=lambda row: row["run_acc"], reverse=True),
        "feature_importance": feature_importance,
        "methods": methods,
        "benchmark_leader": {
            "name": "XGBoost",
            "accuracy": 74.8,
            "note": "Documented all-modality LOSO benchmark; replay probabilities use the spline baseline."
        },
        "best_subject": {
            "subject": best["subject"],
            "run_acc": round(best["run_acc"] * 100, 1),
            "windows": replay_by_subject[best["subject"]],
        },
    }

    OUTPUT_JS.write_text(
        "window.MODALSIM_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    print("Saved:", OUTPUT_JS)
    print("Spline LOSO accuracy:", round(spline_acc * 100, 1), "%")


if __name__ == "__main__":
    main()
