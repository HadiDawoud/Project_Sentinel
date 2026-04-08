import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.dataset_io import load_labeled_csv, normalize_labeled_frame
from scripts.prepare_dataset import stratified_split, validate_frame
from scripts.split_manifest import MANIFEST_FILENAME

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


class TestNormalizeLabeledFrame:
    def test_strips_and_drops_empty(self):
        df = pd.DataFrame({"text": ["  hi  ", "  ", "x"], "label": [0, 1, 2]})
        validate_frame(df)
        out, stats = normalize_labeled_frame(df, drop_empty_text=True)
        assert len(out) == 2
        assert out.iloc[0]["text"] == "hi"
        assert stats["dropped_empty"] == 1

    def test_no_drop_raises_on_blank(self):
        df = pd.DataFrame({"text": ["ok", "   "], "label": [0, 1]})
        validate_frame(df)
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            normalize_labeled_frame(df, drop_empty_text=False)

    def test_truncates_long_text(self):
        df = pd.DataFrame({"text": ["abcdef"], "label": [0]})
        validate_frame(df)
        out, stats = normalize_labeled_frame(df, drop_empty_text=True, max_text_chars=3)
        assert out.iloc[0]["text"] == "abc"
        assert stats["truncated_rows"] == 1


class TestLoadLabeledCsv:
    def test_utf8_bom_stripped_from_header(self, tmp_path):
        p = tmp_path / "bom.csv"
        p.write_bytes('text,label\n"hello",0\n'.encode("utf-8-sig"))
        df = load_labeled_csv(p)
        assert list(df.columns) == ["text", "label"]
        assert df.iloc[0]["text"] == "hello"


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
    mf = out / MANIFEST_FILENAME
    assert mf.exists()
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert manifest["tool"] == "prepare_dataset"
    assert manifest["source"]["sha256"] == hashlib.sha256(inp.read_bytes()).hexdigest()
    assert manifest["rows"]["train"] + manifest["rows"]["val"] + manifest["rows"]["test"] == 100
    assert sum(manifest["label_counts"]["before_split"].values()) == 100


def test_cli_no_manifest(tmp_path):
    inp = tmp_path / "in.csv"
    rows = []
    for lab in range(4):
        for _ in range(8):
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
            "--no-manifest",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert not (out / MANIFEST_FILENAME).exists()


def test_cli_rejects_non_utf8(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"text,label\n\xff,0\n")
    r = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_dataset.py"),
            str(bad),
            "-o",
            str(tmp_path / "out"),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "UTF-8" in (r.stderr + r.stdout)


def test_cli_no_drop_empty_text_fails(tmp_path):
    inp = tmp_path / "in.csv"
    rows = []
    for lab in range(4):
        for _ in range(8):
            rows.append({"text": f"doc {lab}", "label": lab})
    rows.append({"text": "   ", "label": 0})
    pd.DataFrame(rows).to_csv(inp, index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_dataset.py"),
            str(inp),
            "-o",
            str(out),
            "--no-drop-empty-text",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "empty" in (r.stderr + r.stdout).lower()
