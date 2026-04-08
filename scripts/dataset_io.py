"""
Shared ingest helpers for labeled CSVs: UTF-8 with BOM strip, text cleanup.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_READ_ENCODING = "utf-8-sig"


def load_labeled_csv(path: Path | str) -> pd.DataFrame:
    """
    Read CSV as UTF-8; utf-8-sig strips a leading BOM if present.
    Raises UnicodeDecodeError on invalid bytes (strict).
    """
    p = Path(path)
    return pd.read_csv(p, encoding=DEFAULT_READ_ENCODING, encoding_errors="strict")


def normalize_labeled_frame(
    df: pd.DataFrame,
    *,
    drop_empty_text: bool,
    max_text_chars: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Strip whitespace on `text`, optionally drop blank rows, optionally truncate.

    Call after required columns exist and null checks on `text` / `label` pass.

    Returns (frame, stats) with keys dropped_empty, truncated_rows, max_chars.
    """
    stats: dict = {"dropped_empty": 0, "truncated_rows": 0, "max_chars": max_text_chars}
    out = df.copy()
    out["text"] = out["text"].astype(str).str.strip()

    blank = out["text"].str.len() == 0
    if blank.any() and not drop_empty_text:
        rows = out.index[blank].tolist()[:10]
        raise ValueError(
            "Column 'text' has empty or whitespace-only values after strip "
            f"(row indices, first 10): {rows}. Use --drop-empty-text (default) or fix the CSV."
        )

    if drop_empty_text and blank.any():
        stats["dropped_empty"] = int(blank.sum())
        out = out.loc[~blank].reset_index(drop=True)

    if max_text_chars is not None and max_text_chars > 0 and len(out) > 0:
        lengths = out["text"].str.len()
        long_mask = lengths > max_text_chars
        if long_mask.any():
            stats["truncated_rows"] = int(long_mask.sum())
            out = out.copy()
            out.loc[long_mask, "text"] = out.loc[long_mask, "text"].str.slice(0, max_text_chars)

    return out, stats
