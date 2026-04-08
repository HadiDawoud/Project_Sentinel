import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.dataset_io import load_labeled_csv, normalize_labeled_frame
from scripts.prepare_dataset import validate_frame
from scripts.split_manifest import build_manifest, sha256_and_size, write_manifest


def split_data(
    input_file: str,
    output_dir: str = "data/processed",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    *,
    num_labels: int = 4,
    drop_empty_text: bool = True,
    max_text_chars: int | None = None,
    write_manifest_file: bool = True,
):
    source_resolved = Path(input_file).resolve()
    source_sha256, source_size_bytes = sha256_and_size(source_resolved)

    try:
        df = load_labeled_csv(input_file)
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Failed to read {input_file} as UTF-8 (strict). Re-encode as UTF-8 or fix invalid bytes."
        ) from e

    rows_loaded = len(df)
    validate_frame(df, num_labels=num_labels)
    df, norm_stats = normalize_labeled_frame(
        df,
        drop_empty_text=drop_empty_text,
        max_text_chars=max_text_chars,
    )
    if norm_stats["dropped_empty"]:
        print(f"Dropped {norm_stats['dropped_empty']} row(s) with empty text after strip")
    if norm_stats["truncated_rows"]:
        print(
            f"Truncated text in {norm_stats['truncated_rows']} row(s) "
            f"to max_text_chars={max_text_chars}"
        )

    if len(df) == 0:
        raise ValueError("No rows left after filtering; nothing to split.")

    rows_after_normalize = len(df)
    df_before_split = df.copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=random_seed,
        stratify=df["label"],
    )

    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_seed,
        stratify=temp_df["label"],
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    if write_manifest_file:
        manifest = build_manifest(
            tool="split_data",
            source_resolved=source_resolved,
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            options={
                "train": train_ratio,
                "val": val_ratio,
                "test": test_ratio,
                "seed": random_seed,
                "num_labels": num_labels,
                "drop_empty_text": drop_empty_text,
                "max_text_chars": max_text_chars,
                "remove_duplicates": False,
            },
            rows_loaded=rows_loaded,
            norm_stats=norm_stats,
            rows_after_normalize=rows_after_normalize,
            rows_after_dedupe=len(df_before_split),
            duplicates_removed=0,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            df_before_split=df_before_split,
        )
        mp = write_manifest(output_path, manifest)
        print(f"Wrote manifest {mp}")

    print("Split complete:")
    print(f"  Train: {len(train_df)} samples ({train_ratio:.0%})")
    print(f"  Val:   {len(val_df)} samples ({val_ratio:.0%})")
    print(f"  Test:  {len(test_df)} samples ({test_ratio:.0%})")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-labels", type=int, default=4, help="Number of label classes")
    parser.add_argument(
        "--drop-empty-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows whose text is empty after strip (default: true)",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=None,
        metavar="N",
        help="Truncate text to N characters after strip (optional)",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write split_manifest.json next to the output CSVs",
    )
    args = parser.parse_args()

    split_data(
        args.input,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        num_labels=args.num_labels,
        drop_empty_text=args.drop_empty_text,
        max_text_chars=args.max_text_chars,
        write_manifest_file=not args.no_manifest,
    )
