import time
from typing import Dict, Any, Optional
from threading import Lock


class MetricsCollector:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._total_requests = 0
        self._total_errors = 0
        self._label_counts: Dict[str, int] = {}
        self._review_counts: Dict[str, int] = {}
        self._start_time = time.time()
        self._lock = Lock()

    def increment_requests(self) -> None:
        with self._lock:
            self._total_requests += 1

    def increment_errors(self) -> None:
        with self._lock:
            self._total_errors += 1

    def record_label(self, label: str) -> None:
        with self._lock:
            self._label_counts[label] = self._label_counts.get(label, 0) + 1

    def record_review(self, requires_review: bool) -> None:
        with self._lock:
            key = "review_required" if requires_review else "no_review"
            self._review_counts[key] = self._review_counts.get(key, 0) + 1

    def get_metrics(self) -> str:
        with self._lock:
            uptime_seconds = time.time() - self._start_time
            
            lines = [
                f"# HELP sentinel_requests_total Total number of classification requests",
                f"# TYPE sentinel_requests_total counter",
                f"sentinel_requests_total {self._total_requests}",
                "",
                f"# HELP sentinel_errors_total Total number of errors",
                f"# TYPE sentinel_errors_total counter",
                f"sentinel_errors_total {self._total_errors}",
                "",
                f"# HELP sentinel_uptime_seconds Seconds since metrics collection started",
                f"# TYPE sentinel_uptime_seconds gauge",
                f"sentinel_uptime_seconds {uptime_seconds}",
                "",
            ]
            
            for label, count in sorted(self._label_counts.items()):
                label_sanitized = label.replace("-", "_").replace(" ", "_").lower()
                lines.extend([
                    f"# HELP sentinel_label_total Total classifications per label",
                    f"# TYPE sentinel_label_total counter",
                    f'sentinel_label_total{{label="{label}"}} {count}',
                    "",
                ])
            
            for review_status, count in sorted(self._review_counts.items()):
                lines.extend([
                    f"# HELP sentinel_review_total Total classifications by review status",
                    f"# TYPE sentinel_review_total counter",
                    f'sentinel_review_total{{status="{review_status}"}} {count}',
                    "",
                ])
            
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._total_requests = 0
            self._total_errors = 0
            self._label_counts.clear()
            self._review_counts.clear()
            self._start_time = time.time()