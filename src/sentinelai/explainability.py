"""Per-flow SHAP explanations for SentinelAI's supervised detector.

SHAP describes model behavior, not causality or proof of malicious intent.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from .experiment import clean_features, find_label_column, load_csvs, RANDOM_STATE


def _positive_class_shap_values(explainer, values):
    arr = np.asarray(explainer(values).values)
    if arr.ndim == 3:
        if arr.shape[0] == len(values) and arr.shape[1] == values.shape[1]:
            arr = arr[:, :, 1]
        elif arr.shape[1] == len(values) and arr.shape[2] == values.shape[1]:
            arr = arr[1, :, :]
        else:
            raise ValueError(f"Unexpected SHAP tensor shape: {arr.shape}")
    if arr.ndim != 2 or arr.shape != values.shape:
        raise ValueError(f"Unexpected SHAP output shape {arr.shape}")
    return arr


def build_explanations(data_dir: Path, output_dir: Path, holdout_label="PortScan",
                       benign_label="BENIGN", max_rows=30000, explain_count=25,
                       background_count=100):
    data = load_csvs(data_dir)
    label_col = find_label_column(data, None)
    data[label_col] = data[label_col].astype(str).str.strip()
    if max_rows and len(data) > max_rows:
        cap = max(1, max_rows // data[label_col].nunique())
        data = pd.concat([g.sample(min(len(g), cap), random_state=RANDOM_STATE)
                          for _, g in data.groupby(label_col, sort=True)], ignore_index=True)
    X, labels, dropped = clean_features(data, label_col)
    if holdout_label not in set(labels):
        raise ValueError(f"Holdout label {holdout_label!r} not found")
    if benign_label not in set(labels):
        raise ValueError(f"Benign label {benign_label!r} not found")
    known = labels != holdout_label
    X_known = X.loc[known]
    y_known = (labels.loc[known] != benign_label).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_known, y_known, test_size=0.25, random_state=RANDOM_STATE, stratify=y_known)
    pipeline = make_pipeline(SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=160, class_weight="balanced_subsample",
                               random_state=RANDOM_STATE, n_jobs=-1))
    pipeline.fit(X_train, y_train)
    imputer, forest = pipeline.named_steps["simpleimputer"], pipeline.named_steps["randomforestclassifier"]
    background = imputer.transform(X_train.sample(min(background_count, len(X_train)),
                                                   random_state=RANDOM_STATE))
    explainer = shap.TreeExplainer(forest, data=background,
        feature_perturbation="interventional", model_output="probability")
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    alert_pos = np.flatnonzero(preds == 1)
    alert_pos = alert_pos[np.argsort(probs[alert_pos])[::-1]][:explain_count]
    raw = X_test.iloc[alert_pos]
    values = imputer.transform(raw)
    shap_values = _positive_class_shap_values(explainer, values)
    base = np.asarray(explainer.expected_value).reshape(-1)
    base_attack = float(base[1] if len(base) > 1 else base[0])
    records = []
    feature_names = list(X.columns)
    for n, (idx, row, sv, prob, pos) in enumerate(zip(raw.index, raw.to_numpy(),
            shap_values, probs[alert_pos], alert_pos), 1):
        top = np.argsort(np.abs(sv))[::-1][:10]
        contribs = [{"feature": feature_names[j],
            "observed_value": None if pd.isna(row[j]) else float(row[j]),
            "shap_value_attack_probability": float(sv[j]),
            "direction": "increases_attack_probability" if sv[j] > 0 else "decreases_attack_probability"}
            for j in top]
        records.append({"explanation_id": n, "dataset_row_index": int(idx),
            "true_label": str(labels.loc[idx]), "predicted_label": "attack" if preds[pos] else "benign",
            "attack_probability": float(prob), "shap_base_attack_probability": base_attack,
            "shap_sum_all_features": float(np.sum(sv)),
            "shap_reconstructed_attack_probability": float(base_attack + np.sum(sv)),
            "shap_additivity_absolute_error": float(abs(base_attack + np.sum(sv) - prob)),
            "top_feature_contributions": contribs,
            "caveat": "SHAP attributes this model prediction; it does not establish causation or malicious intent."})
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"experiment":"SentinelAI SHAP explanation pilot",
        "dataset_rows_after_sampling":int(len(data)), "feature_count":int(X.shape[1]),
        "held_out_label_excluded_from_supervised_training":holdout_label,
        "known_training_rows":int(len(X_train)), "known_test_rows":int(len(X_test)),
        "explained_alerts":len(records),
        "model":"Median imputation + 160-tree RandomForestClassifier",
        "shap":{"method":"TreeExplainer","model_output":"probability",
            "feature_perturbation":"interventional","background_rows":int(len(background)),
            "meaning":"Positive SHAP values increase model attack probability; negative values decrease it."},
        "dropped_identifier_like_columns":dropped,"explanations":records,
        "limitations":["Only known-class test alerts are explained.",
            "The held-out family is excluded from supervised training; this run does not claim it was detected.",
            "SHAP explains model behavior, not causal mechanisms or ground-truth malicious intent.",
            "The pilot uses a capped sample, not a full-dataset benchmark."]}
    (output_dir/"shap_explanations.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    flat=[]
    for rec in records:
        for c in rec["top_feature_contributions"]:
            flat.append({"explanation_id":rec["explanation_id"],"true_label":rec["true_label"],
                "predicted_label":rec["predicted_label"],"attack_probability":rec["attack_probability"],**c})
    pd.DataFrame(flat).to_csv(output_dir/"shap_feature_contributions.csv",index=False)
    errors=[r["shap_additivity_absolute_error"] for r in records]
    return {"dataset_rows_after_sampling":len(data),"feature_count":int(X.shape[1]),
        "holdout_label":holdout_label,"explained_alerts":len(records),
        "max_additivity_error":max(errors) if errors else None,
        "mean_additivity_error":float(np.mean(errors)) if errors else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",default="datasets/raw/cicids2017")
    p.add_argument("--output-dir",default="experiments/results/explanations")
    p.add_argument("--holdout-label",default="PortScan")
    p.add_argument("--max-rows",type=int,default=30000)
    p.add_argument("--explain-count",type=int,default=25)
    p.add_argument("--background-count",type=int,default=100)
    a=p.parse_args()
    print(json.dumps(build_explanations(Path(a.data_dir),Path(a.output_dir),
        holdout_label=a.holdout_label,max_rows=a.max_rows,
        explain_count=a.explain_count,background_count=a.background_count),indent=2))
if __name__=="__main__": main()
