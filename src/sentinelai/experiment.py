"""Run a controlled withheld-attack-family experiment on flow CSV data."""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, balanced_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DROP_HINTS = ("ip", "flow id", "timestamp", "time", "date")


def load_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    frames = []
    for file in files:
        frame = pd.read_csv(file, low_memory=False)
        frame["_source_file"] = file.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def find_label_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested
    candidates = [c for c in df.columns if c.strip().lower() in {"label", "class", "attack"}]
    if candidates:
        return candidates[0]
    raise ValueError(f"Could not find label column. Columns: {list(df.columns)}")


def clean_features(df: pd.DataFrame, label_col: str):
    y = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col, "_source_file"], errors="ignore").copy()
    X.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in X.columns]
    drop = [c for c in X.columns if any(h in c.lower() for h in DROP_HINTS)]
    X = X.drop(columns=drop, errors="ignore")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Remove all-empty and constant columns; this is schema-only filtering.
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    return X, y, drop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/raw/cicids2017")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--holdout-label", default="PortScan")
    parser.add_argument("--benign-label", default="BENIGN")
    parser.add_argument("--false-positive-rate", type=float, default=0.02)
    parser.add_argument("--list-labels", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional quick smoke-test cap; 0 uses all rows.")
    args = parser.parse_args()

    data = load_csvs(Path(args.data_dir))
    label_col = find_label_column(data, args.label_column)
    data[label_col] = data[label_col].astype(str).str.strip()
    if args.list_labels:
        print(data[label_col].value_counts(dropna=False).to_string())
        return
    if args.max_rows:
        data = data.sample(min(args.max_rows, len(data)), random_state=RANDOM_STATE)
    if not 0 < args.false_positive_rate < 1:
        raise ValueError("--false-positive-rate must be between 0 and 1")

    X, labels, dropped = clean_features(data, label_col)
    benign = args.benign_label
    held = args.holdout_label
    if benign not in set(labels):
        raise ValueError(f"Benign label {benign!r} not found. Use --list-labels.")
    if held not in set(labels):
        raise ValueError(f"Holdout label {held!r} not found. Use --list-labels.")
    if held == benign:
        raise ValueError("The holdout label cannot be the benign label.")

    # Binary known-class training: benign=0, any represented known attack=1.
    known_mask = labels != held
    X_known, y_known = X.loc[known_mask], (labels.loc[known_mask] != benign).astype(int)
    if y_known.nunique() != 2:
        raise ValueError("Known training partition must contain benign and at least one known attack.")
    X_train, X_val, y_train, y_val = train_test_split(
        X_known, y_known, test_size=0.25, random_state=RANDOM_STATE, stratify=y_known
    )
    model = make_pipeline(SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample",
                               random_state=RANDOM_STATE, n_jobs=-1))
    model.fit(X_train, y_train)

    benign_train = X_train.loc[y_train == 0]
    anomaly = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
        IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_STATE, n_jobs=-1))
    anomaly.fit(benign_train)
    val_benign = X_val.loc[y_val == 0]
    val_scores = -anomaly.decision_function(val_benign)
    threshold = float(np.quantile(val_scores, 1 - args.false_positive_rate))

    # Evaluate all held-out attack records plus a validation slice of known records.
    held_idx = labels.index[labels == held]
    known_test_idx = X_val.index
    eval_idx = known_test_idx.append(held_idx)
    X_eval = X.loc[eval_idx]
    y_true = (labels.loc[eval_idx] != benign).astype(int).to_numpy()

    start = time.perf_counter()
    supervised = model.predict(X_eval)
    anomaly_score = -anomaly.decision_function(X_eval)
    hybrid = ((supervised == 1) | (anomaly_score >= threshold)).astype(int)
    elapsed = time.perf_counter() - start

    report = classification_report(y_true, hybrid, target_names=["benign", "attack"],
                                   output_dict=True, zero_division=0)
    held_positions = np.array([labels.loc[i] == held for i in eval_idx])
    held_true = y_true[held_positions]
    held_pred = hybrid[held_positions]
    metrics = {
        "dataset_rows": int(len(data)), "feature_count": int(X.shape[1]),
        "holdout_label": held, "benign_label": benign,
        "dropped_identifier_like_columns": dropped,
        "threshold_false_positive_quantile": args.false_positive_rate,
        "anomaly_threshold": threshold,
        "confusion_matrix_labels": ["benign", "attack"],
        "confusion_matrix": confusion_matrix(y_true, hybrid).tolist(),
        "classification_report": report,
        "withheld_attack_count": int(len(held_true)),
        "withheld_attack_recall": float(recall_score(held_true, held_pred, zero_division=0)) if len(held_true) else None,
        "withheld_attack_precision": float(precision_score(held_true, held_pred, zero_division=0)) if len(held_true) else None,
        "inference_seconds_total": elapsed,
        "inference_ms_per_flow": 1000 * elapsed / max(1, len(X_eval)),
        "random_seed": RANDOM_STATE,
        "warning": "Withheld dataset class is a controlled unknown-class proxy, not proof of real-world zero-day detection."
    }
    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions = pd.DataFrame({
        "dataset_index": eval_idx, "true_label": labels.loc[eval_idx].to_numpy(),
        "true_binary": y_true, "supervised_prediction": supervised,
        "anomaly_score": anomaly_score, "anomaly_threshold": threshold,
        "hybrid_alert": hybrid
    })
    predictions.to_csv(out / "predictions.csv", index=False)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"\nSaved metrics and predictions to {out.resolve()}")


if __name__ == "__main__":
    main()
