"""
JSON manifest for reproducible train/val/test splits (source digest, options, label histograms).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MANIFEST_FILENAME = "split_manifest.json"
SCHEMA_VERSION = "1.0"


def sha256_and_size(path: Path) -> tuple[str, int]:
    """SHA-256 of raw file bytes and total byte length."""
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def label_counts_dict(df: pd.DataFrame) -> dict[str, int]:
    vc = df["label"].astype(int).value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}


def build_manifest(
    *,
    tool: str,
    source_resolved: Path,
    source_sha256: str,
    source_size_bytes: int,
    options: dict,
    rows_loaded: int,
    norm_stats: dict,
    rows_after_normalize: int,
    rows_after_dedupe: int,
    duplicates_removed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    df_before_split: pd.DataFrame,
) -> dict:
    """Assemble manifest dict (schema_version 1.0)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "source": {
            "path": str(source_resolved),
            "sha256": source_sha256,
            "size_bytes": source_size_bytes,
        },
        "options": options,
        "rows": {
            "loaded": rows_loaded,
            "dropped_empty_text": norm_stats.get("dropped_empty", 0),
            "truncated_text_rows": norm_stats.get("truncated_rows", 0),
            "after_normalize": rows_after_normalize,
            "duplicates_removed": duplicates_removed,
            "after_dedupe": rows_after_dedupe,
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "label_counts": {
            "before_split": label_counts_dict(df_before_split),
            "train": label_counts_dict(train_df),
            "val": label_counts_dict(val_df),
            "test": label_counts_dict(test_df),
        },
    }


def write_manifest(output_dir: Path, manifest: dict, filename: str = MANIFEST_FILENAME) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
