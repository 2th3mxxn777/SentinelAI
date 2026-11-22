
import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "unsw_nb15"
OUTPUT_DIR = ROOT / "data" / "processed" / "unsw_nb15"
FEATURE_FILE = DATA_DIR / "NUSW-NB15_features.csv"

SOURCE_FILES = [
    DATA_DIR / "UNSW-NB15_1.csv",
    DATA_DIR / "UNSW-NB15_2.csv",
    DATA_DIR / "UNSW-NB15_3.csv",
    DATA_DIR / "UNSW-NB15_4.csv",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_FILE, encoding="cp1252", newline="") as f:
        reader = csv.DictReader(f)
        columns = [row["Name"].strip() for row in reader]

    if len(columns) != 49:
        raise ValueError(f"Expected 49 columns, found {len(columns)}")

    frames = []

    for path in SOURCE_FILES:
        print(f"Reading {path.name}...", flush=True)

        frame = pd.read_csv(
            path,
            header=None,
            names=columns,
            encoding="cp1252",
            low_memory=False,
        )

        if frame.shape[1] != 49:
            raise ValueError(
                f"{path.name}: expected 49 columns, found {frame.shape[1]}"
            )

        frames.append(frame)
        print(f"Rows: {len(frame):,}", flush=True)

    data = pd.concat(frames, ignore_index=True)

    data["attack_cat"] = (
        data["attack_cat"].fillna("").astype(str).str.strip()
    )
    data["Label"] = pd.to_numeric(data["Label"], errors="coerce")

    output_file = OUTPUT_DIR / "unsw_nb15_combined.csv"
    data.to_csv(output_file, index=False)

    counts = (
        data.groupby(["attack_cat", "Label"], dropna=False)
        .size()
        .reset_index(name="records")
    )
    counts.to_csv(OUTPUT_DIR / "attack_category_counts.csv", index=False)

    report = {
        "dataset": "UNSW-NB15",
        "source_files": [p.name for p in SOURCE_FILES],
        "total_rows": int(len(data)),
        "total_columns": int(data.shape[1]),
        "columns": columns,
        "label_counts": {
            str(k): int(v)
            for k, v in data["Label"].value_counts(dropna=False).items()
        },
        "attack_categories": {
            str(k): int(v)
            for k, v in data["attack_cat"].value_counts(dropna=False).items()
        },
        "missing_attack_category": int((data["attack_cat"] == "").sum()),
        "missing_labels": int(data["Label"].isna().sum()),
    }

    with open(
        OUTPUT_DIR / "dataset_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nPREPARATION COMPLETE", flush=True)
    print("Rows:", f"{len(data):,}")
    print("Columns:", data.shape[1])
    print("\nAttack categories:")
    print(data["attack_cat"].value_counts(dropna=False))
    print("\nLabel counts:")
    print(data["Label"].value_counts(dropna=False))
    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "unsw_nb15"
OUTPUT_DIR = ROOT / "data" / "processed" / "unsw_nb15"
FEATURE_FILE = DATA_DIR / "NUSW-NB15_features.csv"

SOURCE_FILES = [
    DATA_DIR / "UNSW-NB15_1.csv",
    DATA_DIR / "UNSW-NB15_2.csv",
    DATA_DIR / "UNSW-NB15_3.csv",
    DATA_DIR / "UNSW-NB15_4.csv",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_FILE, encoding="cp1252", newline="") as f:
        reader = csv.DictReader(f)
        columns = [row["Name"].strip() for row in reader]

    if len(columns) != 49:
        raise ValueError(f"Expected 49 columns, found {len(columns)}")

    frames = []

    for path in SOURCE_FILES:
        print(f"Reading {path.name}...", flush=True)

        frame = pd.read_csv(
            path,
            header=None,
            names=columns,
            encoding="cp1252",
            low_memory=False,
        )

        if frame.shape[1] != 49:
            raise ValueError(
                f"{path.name}: expected 49 columns, found {frame.shape[1]}"
            )

        frames.append(frame)
        print(f"Rows: {len(frame):,}", flush=True)

    data = pd.concat(frames, ignore_index=True)

    data["attack_cat"] = (
        data["attack_cat"].fillna("").astype(str).str.strip()
    )
    data["Label"] = pd.to_numeric(data["Label"], errors="coerce")

    output_file = OUTPUT_DIR / "unsw_nb15_combined.csv"
    data.to_csv(output_file, index=False)

    counts = (
        data.groupby(["attack_cat", "Label"], dropna=False)
        .size()
        .reset_index(name="records")
    )
    counts.to_csv(OUTPUT_DIR / "attack_category_counts.csv", index=False)

    report = {
        "dataset": "UNSW-NB15",
        "source_files": [p.name for p in SOURCE_FILES],
        "total_rows": int(len(data)),
        "total_columns": int(data.shape[1]),
        "columns": columns,
        "label_counts": {
            str(k): int(v)
            for k, v in data["Label"].value_counts(dropna=False).items()
        },
        "attack_categories": {
            str(k): int(v)
            for k, v in data["attack_cat"].value_counts(dropna=False).items()
        },
        "missing_attack_category": int((data["attack_cat"] == "").sum()),
        "missing_labels": int(data["Label"].isna().sum()),
    }

    with open(
        OUTPUT_DIR / "dataset_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nPREPARATION COMPLETE", flush=True)
    print("Rows:", f"{len(data):,}")
    print("Columns:", data.shape[1])
    print("\nAttack categories:")
    print(data["attack_cat"].value_counts(dropna=False))
    print("\nLabel counts:")
    print(data["Label"].value_counts(dropna=False))
    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "unsw_nb15"
OUTPUT_DIR = ROOT / "data" / "processed" / "unsw_nb15"
FEATURE_FILE = DATA_DIR / "NUSW-NB15_features.csv"

SOURCE_FILES = [
    DATA_DIR / "UNSW-NB15_1.csv",
    DATA_DIR / "UNSW-NB15_2.csv",
    DATA_DIR / "UNSW-NB15_3.csv",
    DATA_DIR / "UNSW-NB15_4.csv",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_FILE, encoding="cp1252", newline="") as f:
        reader = csv.DictReader(f)
        columns = [row["Name"].strip() for row in reader]

    if len(columns) != 49:
        raise ValueError(f"Expected 49 columns, found {len(columns)}")

    frames = []

    for path in SOURCE_FILES:
        print(f"Reading {path.name}...", flush=True)

        frame = pd.read_csv(
            path,
            header=None,
            names=columns,
            encoding="cp1252",
            low_memory=False,
        )

        if frame.shape[1] != 49:
            raise ValueError(
                f"{path.name}: expected 49 columns, found {frame.shape[1]}"
            )

        frames.append(frame)
        print(f"Rows: {len(frame):,}", flush=True)

    data = pd.concat(frames, ignore_index=True)

    data["attack_cat"] = (
        data["attack_cat"].fillna("").astype(str).str.strip()
    )
    data["Label"] = pd.to_numeric(data["Label"], errors="coerce")

    output_file = OUTPUT_DIR / "unsw_nb15_combined.csv"
    data.to_csv(output_file, index=False)

    counts = (
        data.groupby(["attack_cat", "Label"], dropna=False)
        .size()
        .reset_index(name="records")
    )
    counts.to_csv(OUTPUT_DIR / "attack_category_counts.csv", index=False)

    report = {
        "dataset": "UNSW-NB15",
        "source_files": [p.name for p in SOURCE_FILES],
        "total_rows": int(len(data)),
        "total_columns": int(data.shape[1]),
        "columns": columns,
        "label_counts": {
            str(k): int(v)
            for k, v in data["Label"].value_counts(dropna=False).items()
        },
        "attack_categories": {
            str(k): int(v)
            for k, v in data["attack_cat"].value_counts(dropna=False).items()
        },
        "missing_attack_category": int((data["attack_cat"] == "").sum()),
        "missing_labels": int(data["Label"].isna().sum()),
    }

    with open(
        OUTPUT_DIR / "dataset_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nPREPARATION COMPLETE", flush=True)
    print("Rows:", f"{len(data):,}")
    print("Columns:", data.shape[1])
    print("\nAttack categories:")
    print(data["attack_cat"].value_counts(dropna=False))
    print("\nLabel counts:")
    print(data["Label"].value_counts(dropna=False))
    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "unsw_nb15"
OUTPUT_DIR = ROOT / "data" / "processed" / "unsw_nb15"
FEATURE_FILE = DATA_DIR / "NUSW-NB15_features.csv"

SOURCE_FILES = [
    DATA_DIR / "UNSW-NB15_1.csv",
    DATA_DIR / "UNSW-NB15_2.csv",
    DATA_DIR / "UNSW-NB15_3.csv",
    DATA_DIR / "UNSW-NB15_4.csv",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_FILE, encoding="cp1252", newline="") as f:
        reader = csv.DictReader(f)
        columns = [row["Name"].strip() for row in reader]

    if len(columns) != 49:
        raise ValueError(f"Expected 49 columns, found {len(columns)}")

    frames = []

    for path in SOURCE_FILES:
        print(f"Reading {path.name}...", flush=True)

        frame = pd.read_csv(
            path,
            header=None,
            names=columns,
            encoding="cp1252",
            low_memory=False,
        )

        if frame.shape[1] != 49:
            raise ValueError(
                f"{path.name}: expected 49 columns, found {frame.shape[1]}"
            )

        frames.append(frame)
        print(f"Rows: {len(frame):,}", flush=True)

    data = pd.concat(frames, ignore_index=True)

    data["attack_cat"] = (
        data["attack_cat"].fillna("").astype(str).str.strip()
    )
    data["Label"] = pd.to_numeric(data["Label"], errors="coerce")

    output_file = OUTPUT_DIR / "unsw_nb15_combined.csv"
    data.to_csv(output_file, index=False)

    counts = (
        data.groupby(["attack_cat", "Label"], dropna=False)
        .size()
        .reset_index(name="records")
    )
    counts.to_csv(OUTPUT_DIR / "attack_category_counts.csv", index=False)

    report = {
        "dataset": "UNSW-NB15",
        "source_files": [p.name for p in SOURCE_FILES],
        "total_rows": int(len(data)),
        "total_columns": int(data.shape[1]),
        "columns": columns,
        "label_counts": {
            str(k): int(v)
            for k, v in data["Label"].value_counts(dropna=False).items()
        },
        "attack_categories": {
            str(k): int(v)
            for k, v in data["attack_cat"].value_counts(dropna=False).items()
        },
        "missing_attack_category": int((data["attack_cat"] == "").sum()),
        "missing_labels": int(data["Label"].isna().sum()),
    }

    with open(
        OUTPUT_DIR / "dataset_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nPREPARATION COMPLETE", flush=True)
    print("Rows:", f"{len(data):,}")
    print("Columns:", data.shape[1])
    print("\nAttack categories:")
    print(data["attack_cat"].value_counts(dropna=False))
    print("\nLabel counts:")
    print(data["Label"].value_counts(dropna=False))
    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "unsw_nb15"
OUTPUT_DIR = ROOT / "data" / "processed" / "unsw_nb15"
FEATURE_FILE = DATA_DIR / "NUSW-NB15_features.csv"

SOURCE_FILES = [
    DATA_DIR / "UNSW-NB15_1.csv",
    DATA_DIR / "UNSW-NB15_2.csv",
    DATA_DIR / "UNSW-NB15_3.csv",
    DATA_DIR / "UNSW-NB15_4.csv",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_FILE, encoding="cp1252", newline="") as f:
        reader = csv.DictReader(f)
        columns = [row["Name"].strip() for row in reader]

    if len(columns) != 49:
        raise ValueError(f"Expected 49 columns, found {len(columns)}")

    frames = []

    for path in SOURCE_FILES:
        print(f"Reading {path.name}...", flush=True)

        frame = pd.read_csv(
            path,
            header=None,
            names=columns,
            encoding="cp1252",
            low_memory=False,
        )

        if frame.shape[1] != 49:
            raise ValueError(
                f"{path.name}: expected 49 columns, found {frame.shape[1]}"
            )

        frames.append(frame)
        print(f"Rows: {len(frame):,}", flush=True)

    data = pd.concat(frames, ignore_index=True)

    data["attack_cat"] = (
        data["attack_cat"].fillna("").astype(str).str.strip()
    )
    data["Label"] = pd.to_numeric(data["Label"], errors="coerce")

    output_file = OUTPUT_DIR / "unsw_nb15_combined.csv"
    data.to_csv(output_file, index=False)

    counts = (
        data.groupby(["attack_cat", "Label"], dropna=False)
        .size()
        .reset_index(name="records")
    )
    counts.to_csv(OUTPUT_DIR / "attack_category_counts.csv", index=False)

    report = {
        "dataset": "UNSW-NB15",
        "source_files": [p.name for p in SOURCE_FILES],
        "total_rows": int(len(data)),
        "total_columns": int(data.shape[1]),
        "columns": columns,
        "label_counts": {
            str(k): int(v)
            for k, v in data["Label"].value_counts(dropna=False).items()
        },
        "attack_categories": {
            str(k): int(v)
            for k, v in data["attack_cat"].value_counts(dropna=False).items()
        },
        "missing_attack_category": int((data["attack_cat"] == "").sum()),
        "missing_labels": int(data["Label"].isna().sum()),
    }

    with open(
        OUTPUT_DIR / "dataset_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nPREPARATION COMPLETE", flush=True)
    print("Rows:", f"{len(data):,}")
    print("Columns:", data.shape[1])
    print("\nAttack categories:")
    print(data["attack_cat"].value_counts(dropna=False))
    print("\nLabel counts:")
    print(data["Label"].value_counts(dropna=False))
    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "unsw_nb15"
OUTPUT_DIR = ROOT / "data" / "processed" / "unsw_nb15"
FEATURE_FILE = DATA_DIR / "NUSW-NB15_features.csv"

SOURCE_FILES = [
    DATA_DIR / "UNSW-NB15_1.csv",
    DATA_DIR / "UNSW-NB15_2.csv",
    DATA_DIR / "UNSW-NB15_3.csv",
    DATA_DIR / "UNSW-NB15_4.csv",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_FILE, encoding="cp1252", newline="") as f:
        reader = csv.DictReader(f)
        columns = [row["Name"].strip() for row in reader]

    if len(columns) != 49:
        raise ValueError(f"Expected 49 columns, found {len(columns)}")

    frames = []

    for path in SOURCE_FILES:
        print(f"Reading {path.name}...", flush=True)

        frame = pd.read_csv(
            path,
            header=None,
            names=columns,
            encoding="cp1252",
            low_memory=False,
        )

        if frame.shape[1] != 49:
            raise ValueError(
                f"{path.name}: expected 49 columns, found {frame.shape[1]}"
            )

        frames.append(frame)
        print(f"Rows: {len(frame):,}", flush=True)

    data = pd.concat(frames, ignore_index=True)

    data["attack_cat"] = (
        data["attack_cat"].fillna("").astype(str).str.strip()
    )
    data["Label"] = pd.to_numeric(data["Label"], errors="coerce")

    output_file = OUTPUT_DIR / "unsw_nb15_combined.csv"
    data.to_csv(output_file, index=False)

    counts = (
        data.groupby(["attack_cat", "Label"], dropna=False)
        .size()
        .reset_index(name="records")
    )
    counts.to_csv(OUTPUT_DIR / "attack_category_counts.csv", index=False)

    report = {
        "dataset": "UNSW-NB15",
        "source_files": [p.name for p in SOURCE_FILES],
        "total_rows": int(len(data)),
        "total_columns": int(data.shape[1]),
        "columns": columns,
        "label_counts": {
            str(k): int(v)
            for k, v in data["Label"].value_counts(dropna=False).items()
        },
        "attack_categories": {
            str(k): int(v)
            for k, v in data["attack_cat"].value_counts(dropna=False).items()
        },
        "missing_attack_category": int((data["attack_cat"] == "").sum()),
        "missing_labels": int(data["Label"].isna().sum()),
    }

    with open(
        OUTPUT_DIR / "dataset_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nPREPARATION COMPLETE", flush=True)
    print("Rows:", f"{len(data):,}")
    print("Columns:", data.shape[1])
    print("\nAttack categories:")
    print(data["attack_cat"].value_counts(dropna=False))
    print("\nLabel counts:")
    print(data["Label"].value_counts(dropna=False))
    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
