
import json
from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[3]

    input_file = (
        project_root / "data" / "processed"
        / "unsw_nb15" / "unsw_nb15_combined.csv"
    )

    output_dir = (
        project_root / "data" / "processed"
        / "unsw_nb15_experiment"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading UNSW-NB15 dataset...")
    df = pd.read_csv(input_file, low_memory=False)

    if len(df.columns) != 49:
        raise ValueError(f"Expected 49 columns, got {len(df.columns)}")

    print("Original records:", len(df))

    categories = (
        df["attack_cat"].astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("BENIGN")
    )

    print("Auditing feature groups...")

    feature_hashes = pd.util.hash_pandas_object(
        df.iloc[:, :47], index=False
    )

    category_counts = categories.groupby(
        feature_hashes
    ).transform("nunique")

    conflict_mask = category_counts > 1

    conflict_groups = int(
        feature_hashes[conflict_mask].nunique()
    )
    excluded_records = int(conflict_mask.sum())

    clean_df = df.loc[~conflict_mask].copy()
    clean_categories = categories.loc[~conflict_mask]

    print("Conflicting feature groups:", conflict_groups)
    print("Records excluded:", excluded_records)
    print("Records retained:", len(clean_df))

    output_csv = output_dir / "unsw_nb15_conflict_free.csv"
    clean_df.to_csv(output_csv, index=False)

    counts = clean_categories.value_counts()
    counts.index.name = "attack_category"
    counts.name = "record_count"
    counts.to_csv(output_dir / "attack_category_counts.csv")

    report = {
        "dataset": "UNSW-NB15",
        "original_records": int(len(df)),
        "original_columns": int(len(df.columns)),
        "feature_columns_used_for_conflict_audit": 47,
        "conflicting_feature_groups": conflict_groups,
        "records_excluded": excluded_records,
        "records_retained": int(len(clean_df)),
        "consistent_duplicate_records_preserved": True,
        "original_dataset_modified": False,
        "reconnaissance_records_retained": int(
            (clean_categories == "Reconnaissance").sum()
        ),
        "category_counts": {
            str(k): int(v) for k, v in counts.items()
        },
    }

    with open(
        output_dir / "data_quality_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4)

    print("\nPREPARATION COMPLETE")
    print("Clean dataset:", output_csv)
    print("Reconnaissance records retained:",
          report["reconnaissance_records_retained"])
    print("Quality report saved.")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[3]

    input_file = (
        project_root / "data" / "processed"
        / "unsw_nb15" / "unsw_nb15_combined.csv"
    )

    output_dir = (
        project_root / "data" / "processed"
        / "unsw_nb15_experiment"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading UNSW-NB15 dataset...")
    df = pd.read_csv(input_file, low_memory=False)

    if len(df.columns) != 49:
        raise ValueError(f"Expected 49 columns, got {len(df.columns)}")

    print("Original records:", len(df))

    categories = (
        df["attack_cat"].astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("BENIGN")
    )

    print("Auditing feature groups...")

    feature_hashes = pd.util.hash_pandas_object(
        df.iloc[:, :47], index=False
    )

    category_counts = categories.groupby(
        feature_hashes
    ).transform("nunique")

    conflict_mask = category_counts > 1

    conflict_groups = int(
        feature_hashes[conflict_mask].nunique()
    )
    excluded_records = int(conflict_mask.sum())

    clean_df = df.loc[~conflict_mask].copy()
    clean_categories = categories.loc[~conflict_mask]

    print("Conflicting feature groups:", conflict_groups)
    print("Records excluded:", excluded_records)
    print("Records retained:", len(clean_df))

    output_csv = output_dir / "unsw_nb15_conflict_free.csv"
    clean_df.to_csv(output_csv, index=False)

    counts = clean_categories.value_counts()
    counts.index.name = "attack_category"
    counts.name = "record_count"
    counts.to_csv(output_dir / "attack_category_counts.csv")

    report = {
        "dataset": "UNSW-NB15",
        "original_records": int(len(df)),
        "original_columns": int(len(df.columns)),
        "feature_columns_used_for_conflict_audit": 47,
        "conflicting_feature_groups": conflict_groups,
        "records_excluded": excluded_records,
        "records_retained": int(len(clean_df)),
        "consistent_duplicate_records_preserved": True,
        "original_dataset_modified": False,
        "reconnaissance_records_retained": int(
            (clean_categories == "Reconnaissance").sum()
        ),
        "category_counts": {
            str(k): int(v) for k, v in counts.items()
        },
    }

    with open(
        output_dir / "data_quality_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4)

    print("\nPREPARATION COMPLETE")
    print("Clean dataset:", output_csv)
    print("Reconnaissance records retained:",
          report["reconnaissance_records_retained"])
    print("Quality report saved.")


if __name__ == "__main__":
    main()
