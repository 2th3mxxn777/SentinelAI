from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed/cleaned_dataset.csv")
RESULTS = Path("research/results")
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
UNSEEN_CLASS = "PortScan"


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

    print("Known-class records:", len(X_known))
    print("Unseen PortScan records:", len(X_unseen))

    X_train, X_test, y_train, y_test = train_test_split(
        X_known,
        y_known,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_known
    )

    print("Training records:", len(X_train))
    print("Known-class test records:", len(X_test))

    # Model 1: supervised known-class classifier.
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    print("\nTraining Random Forest...")
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)

    rf_report = classification_report(
        y_test,
        rf_predictions,
        output_dict=True,
        zero_division=0
    )

    (RESULTS / "experiment_003_rf_report.txt").write_text(
        classification_report(
            y_test, rf_predictions, zero_division=0
        ),
        encoding="utf-8"
    )

    pd.DataFrame(
        confusion_matrix(y_test, rf_predictions),
        index=sorted(y_known.unique()),
        columns=sorted(y_known.unique())
    ).to_csv(RESULTS / "experiment_003_rf_confusion_matrix.csv")

    # Model 2: anomaly detector trained only on known training data.
    anomaly_model = Pipeline([
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

    print("\nTraining Isolation Forest on known training records...")
    anomaly_model.fit(X_train)

    # Evaluate known held-out records and unseen PortScan records.
    X_evaluation = pd.concat([X_test, X_unseen], axis=0)
    y_true = np.concatenate([
        np.zeros(len(X_test), dtype=int),
        np.ones(len(X_unseen), dtype=int)
    ])

    scores = anomaly_model.decision_function(X_evaluation)
    predictions = (scores < 0).astype(int)

    anomaly_metrics = {
        "known_test_records": int(len(X_test)),
        "unseen_portscan_records": int(len(X_unseen)),
        "portscan_recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "portscan_precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "portscan_f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "known_traffic_false_positive_rate": float(
            predictions[:len(X_test)].mean()
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
        "threshold": "IsolationForest decision_function < 0"
    }

    pd.DataFrame({
        "true_group": y_true,
        "anomaly_score": scores,
        "predicted_anomaly": predictions
    }).to_csv(
        RESULTS / "experiment_003_anomaly_predictions.csv",
        index=False
    )

    results = {
        "experiment": "SentinelAI Experiment 003",
        "dataset": str(DATA),
        "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "class_count": int(y.nunique()),
        "unseen_class": UNSEEN_CLASS,
        "random_state": RANDOM_STATE,
        "train_test_split": "80/20 stratified among known classes",
        "known_training_records": int(len(X_train)),
        "known_test_records": int(len(X_test)),
        "unseen_records": int(len(X_unseen)),
        "training_only_imputation": True,
        "random_forest": {
            "macro_f1": float(rf_report["macro avg"]["f1-score"]),
            "weighted_f1": float(rf_report["weighted avg"]["f1-score"]),
            "accuracy": float(rf_report["accuracy"]),
            "classification_report": rf_report
        },
        "isolation_forest": anomaly_metrics,
        "runtime_seconds": round(time.time() - start, 2)
    }

    (RESULTS / "experiment_003_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print("\nEXPERIMENT 003 COMPLETE")
    print(json.dumps(results, indent=2))
    print("\nResults saved in:", RESULTS)


if __name__ == "__main__":
    main()
