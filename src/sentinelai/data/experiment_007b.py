from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Experiment 007B
# Five-seed robustness test of Ordinal vs One-Hot encoding.
# The pilot sample is fixed (seed 42); split/model seeds vary.
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = (
    ROOT / "data" / "processed" / "unsw_nb15_experiment"
    / "unsw_nb15_conflict_free.csv"
)
OUTPUT_DIR = ROOT / "research" / "results" / "experiment_007b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_LABEL = "Reconnaissance"
SAMPLE_RANDOM_STATE = 42
MAX_ROWS = 300000
SEEDS = [42, 123, 2024, 7, 99]

FEATURE_COLUMNS = [
    "sport", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Sintpkt",
    "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]
CATEGORICAL_COLUMNS = ["proto", "state", "service"]


def load_fixed_pilot():
    print("Loading UNSW-NB15 dataset...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    print("Original rows:", len(df))

    # Match 007A: reproducible category-proportional pilot sample.
    if len(df) > MAX_ROWS:
        parts = []
        for label, group in df.groupby("attack_cat", dropna=False):
            n = max(1, round(MAX_ROWS * len(group) / len(df)))
            parts.append(
                group.sample(
                    n=min(n, len(group)),
                    random_state=SAMPLE_RANDOM_STATE,
                )
            )
        df = pd.concat(parts, ignore_index=True)

    # Match 007A target cleanup and feature-name correction.
    df["attack_cat"] = df["attack_cat"].fillna("").astype(str).str.strip()
    df["attack_cat"] = df["attack_cat"].replace("", "BENIGN")
    df = df.rename(columns={"ct_src_ ltm": "ct_src_ltm"})

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    categorical = [
        c for c in CATEGORICAL_COLUMNS if c in FEATURE_COLUMNS
    ]
    numeric = [c for c in FEATURE_COLUMNS if c not in categorical]

    for col in categorical:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Match 007A exactly: hash every raw column except the two targets.
    raw_feature_columns = [
        c for c in df.columns if c not in ["attack_cat", "Label"]
    ]
    groups = pd.util.hash_pandas_object(
        df[raw_feature_columns].astype(str),
        index=False,
    ).to_numpy()

    print("Fixed pilot rows:", len(df))
    print("Model features:", len(FEATURE_COLUMNS))
    print("Holdout category:", HOLDOUT_LABEL)
    print("Pilot sample seed:", SAMPLE_RANDOM_STATE)

    return df, groups, numeric, categorical


def make_split(df, groups, seed):
    unseen_mask = df["attack_cat"].eq(HOLDOUT_LABEL)
    known = df.loc[~unseen_mask].copy()
    known_groups = groups[~unseen_mask.to_numpy()]
    unseen = df.loc[unseen_mask].copy()

    if unseen.empty:
        raise ValueError(
            f"No rows found for held-out category: {HOLDOUT_LABEL}"
        )

    # Match 007A's two-stage StratifiedGroupKFold procedure.
    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    trainval_idx, test_idx = next(
        outer.split(
            known[FEATURE_COLUMNS],
            known["attack_cat"],
            groups=known_groups,
        )
    )

    trainval = known.iloc[trainval_idx]
    trainval_groups = known_groups[trainval_idx]

    inner = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed + 1,
    )
    train_relative, val_relative = next(
        inner.split(
            trainval[FEATURE_COLUMNS],
            trainval["attack_cat"],
            groups=trainval_groups,
        )
    )

    train_idx = trainval_idx[train_relative]
    val_idx = trainval_idx[val_relative]

    train = known.iloc[train_idx].copy()
    validation = known.iloc[val_idx].copy()
    known_test = known.iloc[test_idx].copy()
    test = pd.concat([known_test, unseen], ignore_index=True)

    # Verify the same three-way known-class group separation as 007A.
    train_groups = set(known_groups[train_idx])
    val_groups = set(known_groups[val_idx])
    test_groups = set(known_groups[test_idx])

    assert train_groups.isdisjoint(val_groups), "Train/validation group leakage"
    assert train_groups.isdisjoint(test_groups), "Train/test group leakage"
    assert val_groups.isdisjoint(test_groups), "Validation/test group leakage"

    split_info = {
        "seed": int(seed),
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "known_test_rows": int(len(known_test)),
        "unseen_test_rows": int(len(unseen)),
        "train_validation_test_group_leakage": False,
    }
    return train, validation, known_test, unseen, test, split_info


def normalized_entropy(probabilities):
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(
        probabilities * np.log(probabilities), axis=1
    ) / np.log(probabilities.shape[1])


def evaluate_encoding(
    encoding_name,
    encoder,
    seed,
    train,
    validation,
    known_test,
    unseen,
    test,
    numeric_columns,
    categorical_columns,
    split_info,
):
    print(f"  Seed {seed}: evaluating {encoding_name}...")
    start = time.time()

    # Match 007A: numerical imputation fitted only on training rows.
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    X_train_num = imputer.fit_transform(train[numeric_columns])
    X_val_num = imputer.transform(validation[numeric_columns])
    X_test_num = imputer.transform(test[numeric_columns])

    # Encoder is also fitted only on training rows.
    X_train_cat = encoder.fit_transform(train[categorical_columns])
    X_val_cat = encoder.transform(validation[categorical_columns])
    X_test_cat = encoder.transform(test[categorical_columns])

    if hasattr(X_train_cat, "toarray"):
        X_train_cat = X_train_cat.toarray()
        X_val_cat = X_val_cat.toarray()
        X_test_cat = X_test_cat.toarray()

    X_train = np.hstack([X_train_num, X_train_cat])
    X_val = np.hstack([X_val_num, X_val_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    y_train = train["attack_cat"].to_numpy()
    benign_val_mask = validation["attack_cat"].eq("BENIGN").to_numpy()
    if not np.any(benign_val_mask):
        raise ValueError(
            f"No benign validation rows for seed {seed}; "
            "cannot calibrate the 99th-percentile threshold."
        )

    model = RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    known_prob = model.predict_proba(X_test[:len(known_test)])
    known_pred = model.classes_[np.argmax(known_prob, axis=1)]

    known_accuracy = accuracy_score(
        known_test["attack_cat"], known_pred
    )
    known_macro_f1 = f1_score(
        known_test["attack_cat"],
        known_pred,
        average="macro",
        zero_division=0,
    )

    # Same normalized predictive entropy and benign validation threshold
    # as Experiment 007A.
    test_prob = model.predict_proba(X_test)
    test_entropy = normalized_entropy(test_prob)

    val_prob = model.predict_proba(X_val)
    val_entropy = normalized_entropy(val_prob)

    threshold = float(
        np.quantile(val_entropy[benign_val_mask], 0.99)
    )

    unknown_true = test["attack_cat"].eq(HOLDOUT_LABEL).to_numpy()
    unknown_pred = test_entropy >= threshold
    benign_test_mask = test["attack_cat"].eq("BENIGN").to_numpy()

    tn = int(np.sum((~unknown_true) & (~unknown_pred)))
    fp = int(np.sum((~unknown_true) & unknown_pred))
    fn = int(np.sum(unknown_true & (~unknown_pred)))
    tp = int(np.sum(unknown_true & unknown_pred))

    metrics = {
        "experiment": "007B",
        "encoding": encoding_name,
        "seed": int(seed),
        "pilot_sample_seed": SAMPLE_RANDOM_STATE,
        "holdout_category": HOLDOUT_LABEL,
        "pilot_rows": int(
            len(train) + len(validation) + len(known_test) + len(unseen)
        ),
        **split_info,
        "feature_count_before_encoding": int(len(FEATURE_COLUMNS)),
        "encoded_feature_count": int(X_train.shape[1]),
        "known_accuracy": float(known_accuracy),
        "known_macro_f1": float(known_macro_f1),
        "entropy_threshold": threshold,
        "validation_benign_fpr": float(
            np.mean(val_entropy[benign_val_mask] >= threshold)
        ),
        "test_benign_fpr": float(
            np.mean(test_entropy[benign_test_mask] >= threshold)
        ),
        "unseen_recall": float(
            recall_score(unknown_true, unknown_pred, zero_division=0)
        ),
        "unseen_precision": float(
            precision_score(unknown_true, unknown_pred, zero_division=0)
        ),
        "unseen_f1": float(
            f1_score(unknown_true, unknown_pred, zero_division=0)
        ),
        "unseen_roc_auc": float(
            roc_auc_score(unknown_true, test_entropy)
        ),
        "unseen_average_precision": float(
            average_precision_score(unknown_true, test_entropy)
        ),
        "confusion_matrix": {
            "true_known_pred_known": tn,
            "true_known_pred_unknown": fp,
            "true_unseen_pred_known": fn,
            "true_unseen_pred_unknown": tp,
        },
        "training_seconds": round(time.time() - start, 2),
    }

    report = classification_report(
        known_test["attack_cat"],
        known_pred,
        zero_division=0,
    )

    stem = f"seed_{seed}_{encoding_name.lower()}"
    with open(OUTPUT_DIR / f"{stem}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(
        OUTPUT_DIR / f"{stem}_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    print(
        f"    Known macro-F1={known_macro_f1:.4f}; "
        f"unseen F1={metrics['unseen_f1']:.4f}; "
        f"unseen recall={metrics['unseen_recall']:.4f}; "
        f"test benign FPR={metrics['test_benign_fpr']:.4f}; "
        f"time={metrics['training_seconds']}s"
    )
    return metrics


def summarize_results(results):
    metrics_to_summarize = [
        "known_accuracy",
        "known_macro_f1",
        "validation_benign_fpr",
        "test_benign_fpr",
        "unseen_recall",
        "unseen_precision",
        "unseen_f1",
        "unseen_roc_auc",
        "unseen_average_precision",
        "training_seconds",
    ]

    rows = []
    for encoding in ["Ordinal", "OneHot"]:
        subset = [r for r in results if r["encoding"] == encoding]
        for metric in metrics_to_summarize:
            values = np.asarray([r[metric] for r in subset], dtype=float)
            rows.append({
                "encoding": encoding,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_seeds": int(len(values)),
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "five_seed_summary.csv", index=False)

    # Paired One-Hot minus Ordinal differences for each seed.
    paired_rows = []
    for metric in metrics_to_summarize:
        ordinal_by_seed = {
            r["seed"]: r[metric] for r in results
            if r["encoding"] == "Ordinal"
        }
        onehot_by_seed = {
            r["seed"]: r[metric] for r in results
            if r["encoding"] == "OneHot"
        }
        differences = [
            onehot_by_seed[s] - ordinal_by_seed[s]
            for s in SEEDS
        ]
        paired_rows.append({
            "metric": metric,
            "onehot_minus_ordinal_mean": float(np.mean(differences)),
            "onehot_minus_ordinal_std": float(
                np.std(differences, ddof=1)
            ),
        })

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(
        OUTPUT_DIR / "paired_encoding_differences.csv",
        index=False,
    )

    summary = {
        "experiment": "007B",
        "purpose": (
            "Five-seed robustness comparison of Ordinal and One-Hot "
            "encoding using a fixed pilot sample."
        ),
        "pilot_sample_seed": SAMPLE_RANDOM_STATE,
        "seeds": SEEDS,
        "holdout_category": HOLDOUT_LABEL,
        "split_method": "StratifiedGroupKFold, 5 folds, two-stage",
        "outer_split_random_state": "seed",
        "inner_split_random_state": "seed + 1",
        "group_hash_columns": "All raw columns except attack_cat and Label",
        "results": results,
        "summary": summary_df.to_dict(orient="records"),
        "paired_differences_onehot_minus_ordinal": paired_rows,
    }

    with open(
        OUTPUT_DIR / "experiment_007b_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nFIVE-SEED SUMMARY")
    print(summary_df.to_string(index=False))
    print("\nPAIRED DIFFERENCES (One-Hot minus Ordinal)")
    print(paired_df.to_string(index=False))


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df, groups, numeric_columns, categorical_columns = load_fixed_pilot()
    all_results = []

    for seed in SEEDS:
        print(f"\nPreparing group-aware split for seed {seed}...")
        (
            train,
            validation,
            known_test,
            unseen,
            test,
            split_info,
        ) = make_split(df, groups, seed)

        print(
            "  Rows:",
            f"train={len(train):,}",
            f"validation={len(validation):,}",
            f"known_test={len(known_test):,}",
            f"unseen_test={len(unseen):,}",
        )
        print("  Group leakage check: PASSED")

        ordinal_result = evaluate_encoding(
            "Ordinal",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            seed,
            train,
            validation,
            known_test,
            unseen,
            test,
            numeric_columns,
            categorical_columns,
            split_info,
        )
        all_results.append(ordinal_result)

        onehot_result = evaluate_encoding(
            "OneHot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            ),
            seed,
            train,
            validation,
            known_test,
            unseen,
            test,
            numeric_columns,
            categorical_columns,
            split_info,
        )
        all_results.append(onehot_result)

    summarize_results(all_results)
    print("\nEXPERIMENT 007B COMPLETE")
    print("Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
