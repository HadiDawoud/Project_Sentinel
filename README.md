# Project Sentinel

A hybrid ML/NLP system for detecting radical or extremist content in text using rule-based filtering and transformer classification.

## Quick Start

```bash
pip install -r requirements.txt
python sentinel.py "We must rise and eliminate those who oppose our cause by any means necessary."
```

## Architecture

- **Preprocessor** - Text cleaning, normalization
- **Rule Engine** - Keyword/pattern matching against YAML rules
- **Classifier** - DistilBERT transformer for multi-class classification
- **Fusion Layer** - Combines rule + ML scores into final threat assessment

## Classification Labels

| Score | Label |
|-------|-------|
| 0 | Non-Radical |
| 1 | Mildly Radical |
| 2 | Moderately Radical |
| 3 | Highly Radical |

## CLI Usage

```bash
# Single text
python sentinel.py "text to classify"

# Batch from file (JSON, JSONL, TXT)
python sentinel.py data/raw/input.json -o results.json

# With raw intermediate results
python sentinel.py "text" --raw
```

## API

```bash
uvicorn api.app:app --reload
```

## Configuration

Edit `config.yaml` to adjust weights, model settings, and rule paths. Update `data/rules/keywords.yaml` to extend keyword/pattern rules.

## Testing

```bash
pytest tests/
```
