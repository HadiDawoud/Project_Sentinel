import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


class StructuredLogger:
    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        audit_file: Optional[str] = None,
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)
        
        self._audit_logger = None
        if audit_file:
            Path(audit_file).parent.mkdir(parents=True, exist_ok=True)
            self._audit_logger = logging.getLogger(f"{name}.audit")
            self._audit_logger.setLevel(logging.INFO)
            self._audit_logger.propagate = False
            
            audit_handler = RotatingFileHandler(
                audit_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            audit_handler.setFormatter(JSONFormatter())
            self._audit_logger.addHandler(audit_handler)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        extra = {}
        if 'request_id' in kwargs:
            extra['request_id'] = kwargs.pop('request_id')
        if kwargs:
            extra['extra'] = kwargs
        
        if extra:
            self.logger.log(level, message, extra=extra)
        else:
            self.logger.log(level, message)
    
    def audit(self, event: str, data: Dict[str, Any], request_id: Optional[str] = None) -> None:
        if not self._audit_logger:
            return
        
        audit_entry = {
            'event': event,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            **data
        }
        
        record = self._audit_logger.makeRecord(
            self._audit_logger.name,
            logging.INFO,
            "",
            0,
            json.dumps(audit_entry),
            None,
            None
        )
        
        if request_id:
            record.request_id = request_id
        
        self._audit_logger.handle(record)
    
    def log_classification(
        self,
        text: str,
        label: str,
        risk_score: int,
        confidence: float,
        flagged_terms: list,
        requires_review: bool,
        request_id: Optional[str] = None,
        audit_id: Optional[str] = None
    ) -> None:
        self.audit(
            event='classification',
            data={
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'label': label,
                'risk_score': risk_score,
                'confidence': confidence,
                'flagged_terms': flagged_terms,
                'requires_human_review': requires_review,
                'audit_id': audit_id
            },
            request_id=request_id
        )
    
    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> None:
        self.error(
            f"Error: {str(error)}",
            error_type=type(error).__name__,
            context=context,
            request_id=request_id
        )


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    audit_file: Optional[str] = None,
    level: str = 'INFO'
) -> StructuredLogger:
    log_level = getattr(logging, level.upper(), logging.INFO)
    return StructuredLogger(
        name=name,
        log_file=log_file,
        audit_file=audit_file,
        level=log_level
    )
