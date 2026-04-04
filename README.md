# Project Sentinel

A hybrid ML/NLP system for detecting radical or extremist content in text using rule-based filtering and transformer classification.

Architecture, configuration, and ethics: [CONTEXT.md](CONTEXT.md).

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

## Responsible AI & Ethics

Project Sentinel is designed as an **assistive tool** for human moderators, not an autonomous decision-maker.

- **Human Review**: Results with `requires_human_review: true` must be manually verified.
- **Bias Awareness**: Be aware of potential biases in both keyword rules and ML models, especially regarding religious, political, and identity-related content.
- **Transparency**: Detailed reasoning is provided for each classification to ensure accountability.
- **Fairness**: The system includes a fairness evaluator to track and mitigate disproportionate flagging of sensitive categories.

See [CONTEXT.md](CONTEXT.md) for more on the project's ethical framework.
