# Processed splits (local only)

After running `scripts/prepare_dataset.py` (or `scripts/split_data.py`), you should see:

- `train.csv`
- `val.csv`
- `test.csv`

These files are **gitignored** so experiment outputs and real data do not land in the repository. Keep a record of **which raw snapshot and which seed** produced a given split if you need reproducibility (later steps can add manifest files under a committed path).

See `data/README.md`.
