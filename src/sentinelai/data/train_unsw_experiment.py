
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = ROOT / "data" / "processed" / "unsw_nb15_experiment" / "unsw_nb15_conflict_free.csv"
OUTPUT_DIR = ROOT / "research" / "results" / "experiment_007"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_LABEL = "Reconnaissance"
RANDOM_STATE = 42
MAX_ROWS = 300000

print("Loading UNSW-NB15 experiment dataset...")
df = pd.read_csv(DATA_FILE, low_memory=False)
print("Loaded rows:", len(df))

# Keep a manageable, reproducible pilot sample.
# Sample within each attack category so rare categories are retained.
if len(df) > MAX_ROWS:
    sampled_parts = []
    for label, group in df.groupby("attack_cat", dropna=False):
        n = max(1, round(MAX_ROWS * len(group) / len(df)))
        sampled_parts.append(
            group.sample(n=min(n, len(group)), random_state=RANDOM_STATE)
        )
    df = pd.concat(sampled_parts, ignore_index=True)

print("Pilot rows:", len(df))

# Normalize whitespace in category names, without merging different labels.
df["attack_cat"] = df["attack_cat"].fillna("").astype(str).str.strip()
df["attack_cat"] = df["attack_cat"].replace("", "BENIGN")

# Exclude IP addresses, timestamps, targets and flow identity fields.
excluded = [
    "srcip", "dstip", "Stime", "Ltime",
    "stcpb", "dtcpb", "attack_cat", "Label"
]

feature_columns = [c for c in df.columns if c not in excluded]

# Numeric ports; non-numeric values become missing and are imputed from training data.
for col in ["sport", "dsport"]:
    if col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Clean feature names.
df = df.rename(columns={"ct_src_ ltm": "ct_src_ltm"})
feature_columns = [
    "sport", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Sintpkt",
    "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
]
feature_columns = [c for c in feature_columns if c in df.columns]

categorical_columns = ["proto", "state", "service"]
numeric_columns = [c for c in feature_columns if c not in categorical_columns]

for col in categorical_columns:
    df[col] = df[col].fillna("UNKNOWN").astype(str)

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

print("Model features:", len(feature_columns))

# Hash all original flow features to keep repeated flows in the same split.
raw_feature_columns = [
    c for c in df.columns
    if c not in ["attack_cat", "Label"]
]
groups = pd.util.hash_pandas_object(
    df[raw_feature_columns].astype(str),
    index=False
).to_numpy()

# Reserve Reconnaissance entirely for unseen-class testing.
unseen_mask = df["attack_cat"].eq(HOLDOUT_LABEL)
known = df.loc[~unseen_mask].copy()
known_groups = groups[~unseen_mask.to_numpy()]

print("Known-category rows:", len(known))
print("Unseen Reconnaissance rows:", int(unseen_mask.sum()))

# Assign known records to five group-aware folds.
print("Creating group-aware train/validation/test splits...")
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# First split: 80% known data for training/validation, 20% for testing.
outer_trainval_idx, test_idx = next(
    splitter.split(
        known[feature_columns],
        known["attack_cat"],
        groups=known_groups
    )
)

# Second split: separate training and validation without group leakage.
trainval = known.iloc[outer_trainval_idx]
trainval_groups = known_groups[outer_trainval_idx]

inner_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=43
)

train_relative, validation_relative = next(
    inner_splitter.split(
        trainval[feature_columns],
        trainval["attack_cat"],
        groups=trainval_groups
    )
)

train_idx = outer_trainval_idx[train_relative]
validation_idx = outer_trainval_idx[validation_relative]

train = known.iloc[train_idx].copy()
validation = known.iloc[validation_idx].copy()
known_test = known.iloc[test_idx].copy()

# Add all held-out Reconnaissance records to the test set.
unseen_test = df.loc[unseen_mask].copy()
test = pd.concat([known_test, unseen_test], ignore_index=True)

print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Known test rows:", len(known_test))
print("Unseen test rows:", len(unseen_test))

# Verify no identical feature groups cross the known-data splits.
train_groups = set(known_groups[train_idx])
validation_groups = set(known_groups[validation_idx])
test_groups = set(known_groups[test_idx])

assert train_groups.isdisjoint(validation_groups)
assert train_groups.isdisjoint(test_groups)
assert validation_groups.isdisjoint(test_groups)

print("Group leakage check: PASSED")

# Encode categorical features and impute missing numeric values.
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
])

X_train_num = preprocessor.fit_transform(train[numeric_columns])
X_val_num = preprocessor.transform(validation[numeric_columns])
X_test_num = preprocessor.transform(test[numeric_columns])

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_cat = encoder.fit_transform(train[categorical_columns])
X_val_cat = encoder.transform(validation[categorical_columns])
X_test_cat = encoder.transform(test[categorical_columns])

X_train = np.hstack([X_train_num, X_train_cat])
X_val = np.hstack([X_val_num, X_val_cat])
X_test = np.hstack([X_test_num, X_test_cat])

y_train = train["attack_cat"].to_numpy()
y_val = validation["attack_cat"].to_numpy()
y_test = test["attack_cat"].to_numpy()

# Train known-category classifier.
print("Training Random Forest...")
start = time.time()

model = RandomForestClassifier(
    n_estimators=150,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Training seconds:", round(time.time() - start, 2))

# Known-class classification performance.
known_probabilities = model.predict_proba(X_test[:len(known_test)])
known_predictions = model.classes_[np.argmax(known_probabilities, axis=1)]

known_accuracy = accuracy_score(
    known_test["attack_cat"], known_predictions
)
known_macro_f1 = f1_score(
    known_test["attack_cat"], known_predictions,
    average="macro", zero_division=0
)

print("\nKNOWN-CLASS CLASSIFICATION")
print("Accuracy:", round(known_accuracy, 6))
print("Macro F1:", round(known_macro_f1, 6))
print(classification_report(
    known_test["attack_cat"],
    known_predictions,
    zero_division=0
))

# Unknown detection: normalized predictive entropy.
probabilities = model.predict_proba(X_test)
probabilities = np.clip(probabilities, 1e-12, 1.0)

entropy = -np.sum(
    probabilities * np.log(probabilities), axis=1
) / np.log(len(model.classes_))

# Calibrate threshold using BENIGN validation records only.
val_probabilities = model.predict_proba(X_val)
val_probabilities = np.clip(val_probabilities, 1e-12, 1.0)

val_entropy = -np.sum(
    val_probabilities * np.log(val_probabilities), axis=1
) / np.log(len(model.classes_))

benign_val_mask = validation["attack_cat"].eq("BENIGN").to_numpy()
threshold = float(np.quantile(val_entropy[benign_val_mask], 0.99))

unknown_true = test["attack_cat"].eq(HOLDOUT_LABEL).to_numpy()
unknown_predicted = entropy >= threshold

tn, fp, fn, tp = confusion_matrix(
    unknown_true, unknown_predicted, labels=[False, True]
).ravel()

metrics = {
    "experiment": "007",
    "dataset": "UNSW-NB15",
    "holdout_category": HOLDOUT_LABEL,
    "pilot_rows": int(len(df)),
    "training_rows": int(len(train)),
    "validation_rows": int(len(validation)),
    "known_test_rows": int(len(known_test)),
    "unseen_test_rows": int(len(unseen_test)),
    "feature_count": int(len(feature_columns)),
    "model": "RandomForestClassifier",
    "estimators": 150,
    "known_accuracy": float(known_accuracy),
    "known_macro_f1": float(known_macro_f1),
    "entropy_threshold": threshold,
    "validation_benign_fpr": float(
        np.mean(val_entropy[benign_val_mask] >= threshold)
    ),
    "test_benign_fpr": float(
        np.mean(
            entropy[test["attack_cat"].eq("BENIGN").to_numpy()] >= threshold
        )
    ),
    "unseen_recall": float(recall_score(
        unknown_true, unknown_predicted, zero_division=0
    )),
    "unseen_precision": float(precision_score(
        unknown_true, unknown_predicted, zero_division=0
    )),
    "unseen_f1": float(f1_score(
        unknown_true, unknown_predicted, zero_division=0
    )),
    "unseen_roc_auc": float(roc_auc_score(unknown_true, entropy)),
    "unseen_average_precision": float(
        average_precision_score(unknown_true, entropy)
    ),
    "confusion_matrix": {
        "true_known_pred_known": int(tn),
        "true_known_pred_unknown": int(fp),
        "true_unseen_pred_known": int(fn),
        "true_unseen_pred_unknown": int(tp)
    }
}

output_file = OUTPUT_DIR / "experiment_007_metrics.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nUNSEEN-CLASS DETECTION RESULTS")
print(json.dumps(metrics, indent=2))
print("\nMetrics saved to:", output_file)
print("\nEXPERIMENT 007 COMPLETE")
