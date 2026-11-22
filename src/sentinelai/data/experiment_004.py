from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA = Path("data/processed/cleaned_dataset.csv")
RESULTS = Path("research/results")
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
UNSEEN_CLASS = "PortScan"
BENIGN_CLASS = "BENIGN"
TARGET_BENIGN_FPR = 0.01


def main():
    start = time.time()

    print("Loading processed dataset...")
    df = pd.read_csv(DATA)

    X = df.drop(columns=["Label"])
    y = df["Label"].astype(str)

    if UNSEEN_CLASS not in set(y):
        raise ValueError(f"{UNSEEN_CLASS} class not found.")
    if BENIGN_CLASS not in set(y):
        raise ValueError(f"{BENIGN_CLASS} class not found.")

    # Keep every PortScan record outside training and validation.
    known_mask = y != UNSEEN_CLASS
    X_known = X.loc[known_mask]
    y_known = y.loc[known_mask]
    X_unseen = X.loc[~known_mask]
    y_unseen = y.loc[~known_mask]

    print("Known-class records:", len(X_known))
    print("Unseen PortScan records:", len(X_unseen))

    # Split known traffic into 80% train and 20% holdout.
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_known,
        y_known,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_known,
    )

    # Divide holdout equally into validation and final test sets.
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_holdout,
    )

    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Known test records:", len(X_test))

    # Fit preprocessing and anomaly detection on training data only.
    anomaly_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=200,
            max_samples=512,
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("\nTraining Isolation Forest on known training records...")
    anomaly_model.fit(X_train)

    # Calibrate the threshold using BENIGN validation records only.
    benign_val_mask = y_val == BENIGN_CLASS
    if int(benign_val_mask.sum()) < 2:
        raise ValueError("Not enough benign validation records to calibrate.")

    validation_scores = anomaly_model.decision_function(X_val)
    benign_validation_scores = validation_scores[benign_val_mask.to_numpy()]

    threshold = float(np.quantile(
        benign_validation_scores,
        TARGET_BENIGN_FPR,
        method="linear",
    ))

    validation_predictions = (validation_scores < threshold).astype(int)
    validation_benign_fpr = float(
        validation_predictions[benign_val_mask.to_numpy()].mean()
    )
    validation_known_fpr = float(validation_predictions.mean())

    print("\nCalibrated threshold:", threshold)
    print("Validation benign FPR:", validation_benign_fpr)

    # Final evaluation: known test records and completely unseen PortScan.
    test_scores = anomaly_model.decision_function(X_test)
    portscan_scores = anomaly_model.decision_function(X_unseen)

    test_predictions = (test_scores < threshold).astype(int)
    portscan_predictions = (portscan_scores < threshold).astype(int)

    y_true = np.concatenate([
        np.zeros(len(X_test), dtype=int),
        np.ones(len(X_unseen), dtype=int),
    ])
    predictions = np.concatenate([test_predictions, portscan_predictions])

    benign_test_mask = (y_test == BENIGN_CLASS).to_numpy()
    if int(benign_test_mask.sum()) == 0:
        raise ValueError("No benign records in the known test set.")

    benign_test_fpr = float(test_predictions[benign_test_mask].mean())
    known_test_fpr = float(test_predictions.mean())

    portscan_recall = float(
        recall_score(
            np.ones(len(X_unseen), dtype=int),
            portscan_predictions,
            zero_division=0,
        )
    )

    portscan_precision = float(
        precision_score(y_true, predictions, zero_division=0)
    )
    portscan_f1 = float(
        f1_score(y_true, predictions, zero_division=0)
    )

    metrics = {
        "experiment": "SentinelAI Experiment 004",
        "dataset": str(DATA),
        "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "unseen_class": UNSEEN_CLASS,
        "random_state": RANDOM_STATE,
        "split": "80% known training / 10% known validation / 10% known test; PortScan fully held out",
        "training_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "known_test_records": int(len(X_test)),
        "unseen_portscan_records": int(len(X_unseen)),
        "training_only_imputation": True,
        "threshold_calibration": {
            "method": "1st percentile of benign validation decision_function scores; flag scores below threshold",
            "target_benign_false_positive_rate": TARGET_BENIGN_FPR,
            "calibrated_threshold": threshold,
            "validation_benign_false_positive_rate": validation_benign_fpr,
            "validation_all_known_false_positive_rate": validation_known_fpr,
        },
        "test_metrics": {
            "benign_false_positive_rate": benign_test_fpr,
            "all_known_false_positive_rate": known_test_fpr,
            "portscan_recall": portscan_recall,
            "portscan_precision": portscan_precision,
            "portscan_f1": portscan_f1,
            "confusion_matrix_known_vs_portscan": confusion_matrix(
                y_true, predictions, labels=[0, 1]
            ).tolist(),
        },
        "runtime_seconds": round(time.time() - start, 2),
    }

    validation_output = pd.DataFrame({
        "split": "validation",
        "true_label": y_val.to_numpy(),
        "true_group": np.where(y_val.to_numpy() == UNSEEN_CLASS, 1, 0),
        "anomaly_score": validation_scores,
        "predicted_anomaly": validation_predictions,
    })
    validation_output.to_csv(
        RESULTS / "experiment_004_validation_predictions.csv",
        index=False,
    )

    test_output = pd.DataFrame({
        "split": "known_test",
        "true_label": y_test.to_numpy(),
        "true_group": 0,
        "anomaly_score": test_scores,
        "predicted_anomaly": test_predictions,
    })
    portscan_output = pd.DataFrame({
        "split": "unseen_portscan",
        "true_label": y_unseen.to_numpy(),
        "true_group": 1,
        "anomaly_score": portscan_scores,
        "predicted_anomaly": portscan_predictions,
    })
    pd.concat([test_output, portscan_output], ignore_index=True).to_csv(
        RESULTS / "experiment_004_test_predictions.csv",
        index=False,
    )

    (RESULTS / "experiment_004_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nEXPERIMENT 004 COMPLETE")
    print(json.dumps(metrics, indent=2))
    print("\nResults saved in:", RESULTS)


if __name__ == "__main__":
    main()
