.PHONY: install test lint api train clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make api        - Start API server"
	@echo "  make train      - Train model"
	@echo "  make clean      - Remove cache and log files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	@command -pylint 2>/dev/null && pylint sentinel/ || echo "pylint not installed"
	@command -ruff 2>/dev/null && ruff check sentinel/ || echo "ruff not installed"

api:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

train:
	python models/train.py --train data/processed/train.csv --val data/processed/val.csv

classify:
	python sentinel.py "$(TEXT)"

clean:
	rm -rf logs/*.log
	rm -rf __pycache__ sentinel/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
