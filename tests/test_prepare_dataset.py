import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_dataset import stratified_split, validate_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestValidateFrame:
    def test_accepts_valid(self):
        df = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d"],
                "label": [0, 1, 2, 3],
            }
        )
        validate_frame(df)

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"text": ["x"]})
        with pytest.raises(ValueError, match="missing"):
            validate_frame(df)

    def test_rejects_invalid_label_range(self):
        df = pd.DataFrame({"text": ["x"], "label": [4]})
        with pytest.raises(ValueError, match="Labels must be"):
            validate_frame(df)


class TestStratifiedSplit:
    def test_sizes_and_stratification(self):
        df = pd.DataFrame(
            {
                "text": [f"sample{i}" for i in range(100)],
                "label": [i % 4 for i in range(100)],
            }
        )
        train, val, test = stratified_split(df, 0.7, 0.15, 0.15, seed=42)
        assert len(train) + len(val) + len(test) == 100
        assert abs(len(train) - 70) <= 1
        assert abs(len(val) - 15) <= 1
        assert abs(len(test) - 15) <= 1
        for part in (train, val, test):
            assert set(part["label"].unique()) == {0, 1, 2, 3}


def test_cli_writes_three_files(tmp_path):
    inp = tmp_path / "in.csv"
    rows = []
    for lab in range(4):
        for _ in range(25):
            rows.append({"text": f"doc {lab}", "label": lab})
    pd.DataFrame(rows).to_csv(inp, index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_dataset.py"),
            str(inp),
            "-o",
            str(out),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "train.csv").exists()
    assert (out / "val.csv").exists()
    assert (out / "test.csv").exists()
