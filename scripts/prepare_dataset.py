#!/usr/bin/env python3
"""
Prepare labeled data for training: validate `text` / `label` columns and write
train, validation, and test CSVs under `data/processed/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = ("text", "label")


def validate_frame(df: pd.DataFrame, num_labels: int = 4) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain columns {list(REQUIRED_COLUMNS)}; missing: {missing}")
    if df["text"].isna().any():
        raise ValueError("Column 'text' contains null values; drop or fill them before splitting.")
    if df["label"].isna().any():
        raise ValueError("Column 'label' contains null values; drop or fill them before splitting.")
    try:
        labels = df["label"].astype(int)
    except (ValueError, TypeError) as e:
        raise ValueError("Column 'label' must be integer class indices.") from e
    invalid = labels[(labels < 0) | (labels >= num_labels)]
    if len(invalid) > 0:
        raise ValueError(
            f"Labels must be in [0, {num_labels - 1}]; found invalid values: "
            f"{sorted(invalid.unique().tolist())}"
        )


def stratified_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train + val + test must sum to 1.0, got {total}")
    train_val, test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df["label"],
        random_state=seed,
    )
    val_of_train_val = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_of_train_val,
        stratify=train_val["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate labeled CSV and write train/val/test splits for models/train.py"
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV with columns: text, label (int 0..num_labels-1)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for train.csv, val.csv, test.csv",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train fraction (default 0.7)")
    parser.add_argument("--val", type=float, default=0.15, help="Validation fraction (default 0.15)")
    parser.add_argument("--test", type=float, default=0.15, help="Test fraction (default 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--num-labels", type=int, default=4, help="Number of classes (default 4)")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remove duplicate texts")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    validate_frame(df, num_labels=args.num_labels)

    if args.remove_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=['text'])
        print(f"Removed {before - len(df)} duplicate texts")

    train_df, val_df, test_df = stratified_split(
        df, args.train, args.val, args.test, args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.output_dir / "train.csv", index=False)
    val_df.to_csv(args.output_dir / "val.csv", index=False)
    test_df.to_csv(args.output_dir / "test.csv", index=False)

    print(
        f"Wrote {len(train_df)} train, {len(val_df)} val, {len(test_df)} test rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
