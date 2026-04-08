# Processed splits (local only)

After running `scripts/prepare_dataset.py` (or `scripts/split_data.py`), you should see:

- `train.csv`
- `val.csv`
- `test.csv`
- `split_manifest.json` — unless you passed `--no-manifest` (source digest, options, row counts, label histograms)

These files are **gitignored** so experiment outputs and real data do not land in the repository. Copy or archive **`split_manifest.json`** with checkpoints if you need an auditable link from a model back to the exact raw file bytes and split settings.

See `data/README.md`.
