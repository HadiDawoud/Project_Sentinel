import logging
import json
from datetime import datetime, timezone
from pathlib import Path


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class StructuredLogger:
    def __init__(self, name: str, audit_log_path: str = None):
        self.logger = logging.getLogger(name)
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None

    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)

    def log_classification(self, text: str, result: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text_length": len(text),
            "label": result.get("label"),
            "risk_score": result.get("risk_score"),
            "confidence": result.get("confidence"),
            "flagged_terms": result.get("flagged_terms", [])
        }
        self.logger.info(f"Classification: {json.dumps(entry)}")

        if self.audit_log_path:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
