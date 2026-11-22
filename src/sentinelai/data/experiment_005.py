from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed/cleaned_dataset.csv")
RESULTS = Path("research/results")
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
UNSEEN_CLASS = "PortScan"
TARGET_BENIGN_FPR = 0.01


def normalized_entropy(probabilities):
    """Return normalized predictive entropy in [0, 1]."""
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    denominator = np.log(probabilities.shape[1])
    return entropy / denominator if denominator > 0 else np.zeros(len(probabilities))


def empirical_percentile(reference_values, values):
    """Map values to upper-tail empirical percentiles using reference values."""
    reference = np.sort(np.asarray(reference_values, dtype=float))
    return np.searchsorted(reference, np.asarray(values, dtype=float), side="right") / len(reference)


def evaluate_signal(name, validation_scores, validation_labels, test_scores, test_labels,
                    test_labels_text, validation_labels_text):
    """Calibrate at target benign FPR and evaluate on untouched test records."""
    benign_validation = validation_scores[validation_labels_text == "BENIGN"]
    if len(benign_validation) == 0:
        raise ValueError("No benign validation records available for calibration.")

    threshold = float(np.quantile(benign_validation, 1.0 - TARGET_BENIGN_FPR,
                                  method="higher"))
    validation_predictions = (validation_scores > threshold).astype(int)
    test_predictions = (test_scores > threshold).astype(int)

    known_test_mask = test_labels == 0
    benign_test_mask = test_labels_text == "BENIGN"
    unseen_test_mask = test_labels == 1

    return {
        "signal": name,
        "threshold": threshold,
        "validation_benign_fpr": float(
            validation_predictions[validation_labels_text == "BENIGN"].mean()
        ),
        "test_benign_fpr": float(test_predictions[benign_test_mask].mean()),
        "test_all_known_fpr": float(test_predictions[known_test_mask].mean()),
        "portscan_recall": float(recall_score(
            test_labels[unseen_test_mask], test_predictions[unseen_test_mask],
            zero_division=0
        )),
        "portscan_precision": float(precision_score(
            test_labels, test_predictions, zero_division=0
        )),
        "portscan_f1": float(f1_score(
            test_labels, test_predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(test_labels, test_scores)),
        "average_precision": float(average_precision_score(test_labels, test_scores)),
        "prevalence_baseline": float(test_labels.mean()),
        "confusion_matrix_known_vs_portscan": confusion_matrix(
            test_labels, test_predictions
        ).tolist(),
    }


def main():
    start = time.time()
    print("Loading processed dataset...")
    df = pd.read_csv(DATA)

    X = df.drop(columns=["Label"])
    y = df["Label"].astype(str)

    if UNSEEN_CLASS not in set(y):
        raise ValueError("PortScan class not found.")

    known_mask = y != UNSEEN_CLASS
    X_known = X.loc[known_mask]
    y_known = y.loc[known_mask]
    X_unseen = X.loc[~known_mask]

    # First reserve 20% of known traffic, then split it equally into validation/test.
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_known, y_known, test_size=0.20, random_state=RANDOM_STATE,
        stratify=y_known
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.50, random_state=RANDOM_STATE + 1,
        stratify=y_holdout
    )

    print("Known training records:", len(X_train))
    print("Known validation records:", len(X_validation))
    print("Known test records:", len(X_test))
    print("Unseen PortScan records:", len(X_unseen))

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    isolation_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=200,
            max_samples=512,
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    print("\nTraining Random Forest on known training records...")
    rf.fit(X_train, y_train)
    print("Training Isolation Forest on known training records...")
    isolation_forest.fit(X_train)

    # Validation contains known classes only; PortScan is never used for calibration.
    validation_probabilities = rf.predict_proba(X_validation)
    test_probabilities = rf.predict_proba(X_test)
    unseen_probabilities = rf.predict_proba(X_unseen)

    validation_max_uncertainty = 1.0 - validation_probabilities.max(axis=1)
    test_max_uncertainty = 1.0 - test_probabilities.max(axis=1)
    unseen_max_uncertainty = 1.0 - unseen_probabilities.max(axis=1)

    validation_entropy = normalized_entropy(validation_probabilities)
    test_entropy = normalized_entropy(test_probabilities)
    unseen_entropy = normalized_entropy(unseen_probabilities)

    validation_if_anomaly = -isolation_forest.decision_function(X_validation)
    test_if_anomaly = -isolation_forest.decision_function(X_test)
    unseen_if_anomaly = -isolation_forest.decision_function(X_unseen)

    # Convert each signal to an empirical percentile using known validation traffic.
    # Higher percentile = more suspicious. PortScan is excluded from the reference.
    validation_max_pct = rankdata(validation_max_uncertainty, method="average") / len(validation_max_uncertainty)
    test_max_pct = empirical_percentile(validation_max_uncertainty, test_max_uncertainty)
    unseen_max_pct = empirical_percentile(validation_max_uncertainty, unseen_max_uncertainty)

    validation_entropy_pct = rankdata(validation_entropy, method="average") / len(validation_entropy)
    test_entropy_pct = empirical_percentile(validation_entropy, test_entropy)
    unseen_entropy_pct = empirical_percentile(validation_entropy, unseen_entropy)

    validation_if_pct = rankdata(validation_if_anomaly, method="average") / len(validation_if_anomaly)
    test_if_pct = empirical_percentile(validation_if_anomaly, test_if_anomaly)
    unseen_if_pct = empirical_percentile(validation_if_anomaly, unseen_if_anomaly)

    # Equal-weight hybrid; weights are fixed before inspecting unseen test outcomes.
    validation_hybrid = (validation_max_pct + validation_entropy_pct + validation_if_pct) / 3.0
    test_hybrid = (test_max_pct + test_entropy_pct + test_if_pct) / 3.0
    unseen_hybrid = (unseen_max_pct + unseen_entropy_pct + unseen_if_pct) / 3.0

    validation_labels = np.zeros(len(X_validation), dtype=int)
    validation_labels_text = y_validation.to_numpy()
    test_labels = np.concatenate([
        np.zeros(len(X_test), dtype=int),
        np.ones(len(X_unseen), dtype=int)
    ])
    test_labels_text = np.concatenate([y_test.to_numpy(), np.full(len(X_unseen), UNSEEN_CLASS)])
    test_group_text = np.concatenate([
        np.full(len(X_test), "known_test"),
        np.full(len(X_unseen), "unseen_portscan")
    ])

    # Combine known test and unseen PortScan predictions in the same order as labels.
    signals = {
        "random_forest_max_probability_uncertainty": (
            validation_max_uncertainty,
            np.concatenate([test_max_uncertainty, unseen_max_uncertainty])
        ),
        "random_forest_normalized_entropy": (
            validation_entropy,
            np.concatenate([test_entropy, unseen_entropy])
        ),
        "isolation_forest_anomaly_score": (
            validation_if_anomaly,
            np.concatenate([test_if_anomaly, unseen_if_anomaly])
        ),
        "hybrid_equal_weight_percentile": (
            validation_hybrid,
            np.concatenate([test_hybrid, unseen_hybrid])
        ),
    }

    metrics = {}
    prediction_frame = pd.DataFrame({
        "split": test_group_text,
        "true_label": test_labels_text,
        "true_group": test_labels,
    })

    for name, (validation_scores, test_scores) in signals.items():
        metrics[name] = evaluate_signal(
            name, validation_scores, validation_labels, test_scores,
            test_labels, test_labels_text, validation_labels_text
        )
        prediction_frame[name + "_score"] = test_scores
        prediction_frame[name + "_predicted_anomaly"] = (
            test_scores > metrics[name]["threshold"]
        ).astype(int)

    # RF closed-set known-class performance is reported separately from unknown detection.
    rf_test_predictions = rf.predict(X_test)
    rf_report = classification_report(
        y_test, rf_test_predictions, output_dict=True, zero_division=0
    )
    (RESULTS / "experiment_005_rf_known_class_report.txt").write_text(
        classification_report(y_test, rf_test_predictions, zero_division=0),
        encoding="utf-8"
    )

    prediction_frame.to_csv(
        RESULTS / "experiment_005_test_predictions.csv", index=False
    )

    results = {
        "experiment": "SentinelAI Experiment 005",
        "objective": "Compare RF uncertainty, predictive entropy, Isolation Forest, and a fixed equal-weight hybrid for held-out PortScan detection",
        "dataset": str(DATA),
        "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "unseen_class": UNSEEN_CLASS,
        "random_state": RANDOM_STATE,
        "split": "80% known training / 10% known validation / 10% known test; PortScan fully excluded from training and validation",
        "training_records": int(len(X_train)),
        "validation_records": int(len(X_validation)),
        "known_test_records": int(len(X_test)),
        "unseen_portscan_records": int(len(X_unseen)),
        "target_benign_fpr": TARGET_BENIGN_FPR,
        "hybrid_definition": "Equal-weight mean of validation-referenced empirical percentiles for RF maximum-probability uncertainty, normalized predictive entropy, and Isolation Forest anomaly score",
        "training_only_imputation": True,
        "random_forest_known_class": {
            "macro_f1": float(rf_report["macro avg"]["f1-score"]),
            "weighted_f1": float(rf_report["weighted avg"]["f1-score"]),
            "accuracy": float(rf_report["accuracy"]),
        },
        "unknown_detection_metrics": metrics,
        "runtime_seconds": round(time.time() - start, 2),
    }

    (RESULTS / "experiment_005_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\nEXPERIMENT 005 COMPLETE")
    for name, item in metrics.items():
        print(
            name,
            "| ROC-AUC:", round(item["roc_auc"], 4),
            "| AP:", round(item["average_precision"], 4),
            "| PortScan recall:", round(item["portscan_recall"], 4),
            "| Test benign FPR:", round(item["test_benign_fpr"], 4),
            "| Test all-known FPR:", round(item["test_all_known_fpr"], 4),
        )
    print("\nResults saved in:", RESULTS)


if __name__ == "__main__":
    main()
