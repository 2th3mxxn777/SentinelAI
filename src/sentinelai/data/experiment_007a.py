
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
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = (
    ROOT / "data" / "processed" / "unsw_nb15_experiment"
    / "unsw_nb15_conflict_free.csv"
)
OUTPUT_DIR = ROOT / "research" / "results" / "experiment_007a"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_LABEL = "Reconnaissance"
RANDOM_STATE = 42
MAX_ROWS = 300000

print("Loading UNSW-NB15...")
df = pd.read_csv(DATA_FILE, low_memory=False)
print("Original rows:", len(df))

# Reproducible category-stratified pilot sample.
if len(df) > MAX_ROWS:
    parts = []
    for label, group in df.groupby("attack_cat", dropna=False):
        n = max(1, round(MAX_ROWS * len(group) / len(df)))
        parts.append(
            group.sample(
                n=min(n, len(group)),
                random_state=RANDOM_STATE,
            )
        )
    df = pd.concat(parts, ignore_index=True)

df["attack_cat"] = df["attack_cat"].fillna("").astype(str).str.strip()
df["attack_cat"] = df["attack_cat"].replace("", "BENIGN")

excluded = [
    "srcip", "dstip", "Stime", "Ltime",
    "stcpb", "dtcpb", "attack_cat", "Label",
]

df = df.rename(columns={"ct_src_ ltm": "ct_src_ltm"})

feature_columns = [
    "sport", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Sintpkt",
    "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]
feature_columns = [c for c in feature_columns if c in df.columns]

categorical_columns = ["proto", "state", "service"]
categorical_columns = [
    c for c in categorical_columns if c in feature_columns
]
numeric_columns = [
    c for c in feature_columns if c not in categorical_columns
]

for col in categorical_columns:
    df[col] = df[col].fillna("UNKNOWN").astype(str)

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# Hash raw features for group-aware splitting.
raw_feature_columns = [
    c for c in df.columns if c not in ["attack_cat", "Label"]
]
groups = pd.util.hash_pandas_object(
    df[raw_feature_columns].astype(str),
    index=False,
).to_numpy()

unseen_mask = df["attack_cat"].eq(HOLDOUT_LABEL)
known = df.loc[~unseen_mask].copy()
known_groups = groups[~unseen_mask.to_numpy()]
unseen = df.loc[unseen_mask].copy()

print("Pilot rows:", len(df))
print("Known rows:", len(known))
print("Unseen rows:", len(unseen))
print("Features:", len(feature_columns))

# Same deterministic group-aware split for both encoding methods.
outer = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)
trainval_idx, test_idx = next(
    outer.split(
        known[feature_columns],
        known["attack_cat"],
        groups=known_groups,
    )
)

trainval = known.iloc[trainval_idx]
trainval_groups = known_groups[trainval_idx]

inner = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=43,
)
train_relative, val_relative = next(
    inner.split(
        trainval[feature_columns],
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

# Explicit group-leakage checks.
train_groups = set(known_groups[train_idx])
val_groups = set(known_groups[val_idx])
test_groups = set(known_groups[test_idx])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print("Group leakage check: PASSED")

# Shared numerical preprocessing: fit on training data only.
imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True,
)
X_train_num = imputer.fit_transform(train[numeric_columns])
X_val_num = imputer.transform(validation[numeric_columns])
X_test_num = imputer.transform(test[numeric_columns])

y_train = train["attack_cat"].to_numpy()
y_val = validation["attack_cat"].to_numpy()

unknown_true = test["attack_cat"].eq(HOLDOUT_LABEL).to_numpy()
benign_val_mask = validation["attack_cat"].eq("BENIGN").to_numpy()

def evaluate_encoding(name, encoder):
    print(f"\nEvaluating {name}...")
    start = time.time()

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

    model = RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
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

    probabilities = np.clip(model.predict_proba(X_test), 1e-12, 1.0)
    entropy = -np.sum(
        probabilities * np.log(probabilities), axis=1
    ) / np.log(len(model.classes_))

    val_prob = np.clip(model.predict_proba(X_val), 1e-12, 1.0)
    val_entropy = -np.sum(
        val_prob * np.log(val_prob), axis=1
    ) / np.log(len(model.classes_))

    threshold = float(
        np.quantile(val_entropy[benign_val_mask], 0.99)
    )
    unknown_pred = entropy >= threshold

    tn = int(np.sum((~unknown_true) & (~unknown_pred)))
    fp = int(np.sum((~unknown_true) & unknown_pred))
    fn = int(np.sum(unknown_true & (~unknown_pred)))
    tp = int(np.sum(unknown_true & unknown_pred))

    metrics = {
        "experiment": "007A",
        "encoding": name,
        "pilot_rows": int(len(df)),
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "known_test_rows": int(len(known_test)),
        "unseen_test_rows": int(len(unseen)),
        "feature_count_before_encoding": int(len(feature_columns)),
        "encoded_feature_count": int(X_train.shape[1]),
        "known_accuracy": float(known_accuracy),
        "known_macro_f1": float(known_macro_f1),
        "entropy_threshold": threshold,
        "validation_benign_fpr": float(
            np.mean(val_entropy[benign_val_mask] >= threshold)
        ),
        "test_benign_fpr": float(
            np.mean(
                entropy[
                    test["attack_cat"].eq("BENIGN").to_numpy()
                ] >= threshold
            )
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
            roc_auc_score(unknown_true, entropy)
        ),
        "unseen_average_precision": float(
            average_precision_score(unknown_true, entropy)
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

    with open(
        OUTPUT_DIR / f"{name.lower()}_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    with open(
        OUTPUT_DIR / f"{name.lower()}_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    print(json.dumps(metrics, indent=2))
    return metrics

ordinal_result = evaluate_encoding(
    "Ordinal",
    OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ),
)

onehot_result = evaluate_encoding(
    "OneHot",
    OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    ),
)

comparison = {
    "experiment": "007A",
    "purpose": "Ordinal versus one-hot categorical encoding",
    "random_state": RANDOM_STATE,
    "holdout_category": HOLDOUT_LABEL,
    "results": [ordinal_result, onehot_result],
}

with open(
    OUTPUT_DIR / "encoding_comparison.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(comparison, f, indent=2)

print("\nEXPERIMENT 007A COMPLETE")
print("Results saved to:", OUTPUT_DIR)
