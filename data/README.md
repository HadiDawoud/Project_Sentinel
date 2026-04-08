# Data layout

This project expects **labeled CSV** data for training and evaluation. Raw corpora and generated splits stay **out of git** by default (see `.gitignore`); only documentation and a tiny **format demo** are committed.

## Directories

| Path | Purpose |
|------|---------|
| `data/raw/` | Your source CSV (or exports) before splitting. Not committed. |
| `data/processed/` | `train.csv`, `val.csv`, `test.csv` produced by the split scripts. Not committed. |
| `data/examples/` | Committed examples that document the expected file shape only. |
| `data/rules/` | Keyword and pattern rules (committed). |

## CSV schema

Training scripts expect a header row with at least:

- **`text`** — UTF-8 string, one sample per row (no nulls). Files may include a UTF-8 BOM; readers strip it automatically.
- **`label`** — Integer class index. Default project labels: `0` … `3` (four classes). See `README.md` / `CONTEXT.md` for label meanings.

Optional columns may be present for your own tooling; split scripts only require `text` and `label`.

### Ingest options (`prepare_dataset.py` / `split_data.py`)

- **`--drop-empty-text` / `--no-drop-empty-text`** — By default, rows whose `text` is empty after trimming whitespace are **dropped** (count printed). Use `--no-drop-empty-text` to **fail** instead if any blank lines exist.
- **`--max-text-chars N`** — Optionally truncate `text` to `N` characters after strip (number of truncated rows is printed).
- Non-UTF-8 bytes cause a **clear error** (strict decode); re-encode the source file as UTF-8.

## Workflow

Stratified splitting needs **enough rows per class** in each fold (roughly on the order of tens of rows minimum for four classes with default `0.7 / 0.15 / 0.15` fractions). The committed `data/examples/labeled_format_demo.csv` is sized so `prepare_dataset.py` succeeds with defaults.

1. Place a labeled file under `data/raw/` locally (e.g. `my_corpus.csv`).
2. Run `scripts/prepare_dataset.py` (validates columns and labels, optional dedupe, stratified split):

   ```bash
   python scripts/prepare_dataset.py data/raw/my_corpus.csv -o data/processed
   ```

3. Train with `models/train.py` pointing at `data/processed/train.csv` and `val.csv`.

`scripts/split_data.py` is a lighter alternative if you already validated the CSV elsewhere.

## Ethics and safety

Do **not** commit real operational intelligence, PII, or unreviewed extremist content. Use private storage and access controls for sensitive corpora. The committed file under `data/examples/` uses **neutral placeholder text** only to show CSV structure.
