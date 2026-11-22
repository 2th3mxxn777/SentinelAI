from pathlib import Path
import json
import numpy as np
import pandas as pd

SOURCE = Path(r"C:\Users\parth\Downloads\cicids2017_stratified_pilot.csv")
OUTPUT = Path("data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    df = pd.read_csv(SOURCE)
    if "Label" not in df.columns:
        raise ValueError("Missing Label column")

    original_rows = len(df)
    hashes = pd.util.hash_pandas_object(
        df.drop(columns=["Label"]), index=False
    )

    audit = pd.DataFrame({
        "hash": hashes,
        "label": df["Label"].astype(str)
    })

    groups = audit.groupby("hash").agg(
        rows=("label", "size"),
        labels=("label", "nunique")
    )

    conflicts = groups[
        (groups["rows"] > 1) & (groups["labels"] > 1)
    ].index

    conflict_mask = hashes.isin(conflicts)
    safe = df.loc[~conflict_mask].copy()
    safe_hashes = hashes.loc[~conflict_mask]
    safe = safe.loc[~safe_hashes.duplicated()].copy()

    numeric = safe.select_dtypes(include=[np.number]).columns
    infinity_count = int(np.isinf(safe[numeric].to_numpy()).sum())

    safe[numeric] = safe[numeric].replace(
        [np.inf, -np.inf], np.nan
    )

    missing = safe.isna().sum()
    safe.to_csv(OUTPUT / "cleaned_dataset.csv", index=False)

    report = {
        "original_rows": int(original_rows),
        "repeated_feature_groups": int((groups["rows"] > 1).sum()),
        "consistent_repeated_groups": int(
            ((groups["rows"] > 1) & (groups["labels"] == 1)).sum()
        ),
        "conflicting_feature_groups": int(len(conflicts)),
        "conflicting_rows_excluded": int(conflict_mask.sum()),
        "processed_rows": int(len(safe)),
        "rows_removed": int(original_rows - len(safe)),
        "infinity_values_replaced": infinity_count,
        "missing_values": {
            str(k): int(v) for k, v in missing.items() if v > 0
        },
        "imputation_performed": False
    }

    (OUTPUT / "data_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("SENTINELAI PREPROCESSING COMPLETE")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
