import logging
import sys
import time
import uuid
import yaml
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field

from .preprocessor import TextPreprocessor
from .rule_engine import RuleEngine
from .classifier import RadicalClassifier
from .fusion import ScoreFusion
from .exceptions import ModelLoadError, PredictionError, ValidationError, PreprocessingError, CacheError, RuleEngineError
from .metrics import MetricsCollector
import contextlib


@dataclass
class BatchStats:
    total: int = 0
    processed: int = 0
    failed: int = 0
    elapsed_ms: float = 0.0
    items_per_second: float = 0.0


class SentinelPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        from .classifier import RadicalClassifier

        self.config = self._load_config(config_path)
        self.preprocessor = TextPreprocessor()
        self.rule_engine = RuleEngine(
            rules_path=self.config['rule_engine']['data_path']
        )
        
        model_cfg = self.config.get('model', {})
        pipe_cfg = self.config.get('pipeline', {})
        lazy_load = pipe_cfg.get('lazy_load_model', False)
        
        self.classifier = RadicalClassifier(
            model_name=model_cfg.get('name', 'distilbert-base-uncased'),
            num_labels=model_cfg.get('num_labels', 4),
            checkpoint_path=model_cfg.get('checkpoint_path'),
            lazy_load=lazy_load
        )
        
        rule_cfg = self.config.get('rule_engine', {})
        self.fusion = ScoreFusion(
            rule_weight=rule_cfg.get('weights', {}).get('high_risk', 0.3),
            ml_weight=1.0 - rule_cfg.get('weights', {}).get('high_risk', 0.3),
            amplification_factor=rule_cfg.get('amplification_factor', 1.5)
        )
        pipe_cfg = self.config.get('pipeline', {})
        self._classify_cache_max = max(0, int(pipe_cfg.get('classify_cache_size', 0)))
        self._cache_ttl_seconds = float(pipe_cfg.get('cache_ttl_seconds', 300))
        self._include_latency_ms = bool(pipe_cfg.get('include_latency_ms', False))
        self._batch_max_workers = int(pipe_cfg.get('batch_max_workers', 4))
        self._batch_chunk_size = int(pipe_cfg.get('batch_chunk_size', 8))
        self._classify_cache: OrderedDict[str, tuple] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_stats = BatchStats()
        self._metrics = MetricsCollector()
        self._setup_logging()

    def warmup(self, num_inferences: int = 3) -> Dict:
        self.classifier.warmup(num_inferences)
        return {"status": "warmup_complete", "model_loaded": self.classifier.is_loaded}

    def health_check(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {
            "status": "healthy",
            "components": {},
            "issues": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0"
        }
        
        preprocessor_status = "healthy"
        preprocessor_latency_ms = None
        try:
            t0 = time.perf_counter()
            test_result = self.preprocessor.preprocess("health check test")
            preprocessor_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            if not test_result.get("cleaned"):
                preprocessor_status = "degraded"
        except Exception as e:
            preprocessor_status = "unhealthy"
            health["issues"].append({"component": "preprocessor", "error": str(e)})
        health["components"]["preprocessor"] = {"status": preprocessor_status, "latency_ms": preprocessor_latency_ms}
        
        rule_engine_status = "healthy"
        rule_engine_latency_ms = None
        try:
            t0 = time.perf_counter()
            test_rule = self.rule_engine.analyze("health check test")
            rule_engine_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            if test_rule is None:
                rule_engine_status = "degraded"
        except Exception as e:
            rule_engine_status = "unhealthy"
            health["issues"].append({"component": "rule_engine", "error": str(e)})
        health["components"]["rule_engine"] = {"status": rule_engine_status, "latency_ms": rule_engine_latency_ms}
        
        classifier_status = "healthy"
        classifier_latency_ms = None
        classifier_model_info = {}
        if not self.classifier.is_loaded:
            classifier_status = "not_loaded"
        else:
            try:
                t0 = time.perf_counter()
                _ = self.classifier.predict("health check test")
                classifier_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                classifier_model_info = self.classifier.get_stats()
            except Exception as e:
                classifier_status = "unhealthy"
                health["issues"].append({"component": "classifier", "error": str(e)})
        health["components"]["classifier"] = {
            "status": classifier_status,
            "latency_ms": classifier_latency_ms,
            "model_info": classifier_model_info
        }
        
        health["cache"] = self.get_cache_stats()
        
        if any(s.get("status") in ("unhealthy", "not_loaded") for s in health["components"].values()):
            health["status"] = "degraded"
        if health["issues"]:
            health["status"] = "unhealthy"
        
        return health

    def _load_config(self, config_path: str) -> Dict:
        env_config_path = os.environ.get('SENTINEL_CONFIG_PATH', config_path)
        path = Path(env_config_path)
        if not path.exists():
            return self._default_config()
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        config = self._apply_env_overrides(config)
        self._validate_config(config)
        return config

    def _validate_config(self, config: Dict) -> None:
        from .exceptions import ConfigurationError
        
        cfg_model = config.get('model', {})
        if cfg_model.get('num_labels', 0) not in range(2, 10):
            raise ConfigurationError(f"Invalid model.num_labels: {cfg_model.get('num_labels')}")
        
        if cfg_model.get('batch_size', 0) > 64:
            raise ConfigurationError(f"model.batch_size exceeds maximum of 64: {cfg_model.get('batch_size')}")
        
        cfg_pipeline = config.get('pipeline', {})
        cache_size = cfg_pipeline.get('classify_cache_size', 0)
        if cache_size < 0 or cache_size > 10000:
            raise ConfigurationError(f"Invalid pipeline.classify_cache_size: {cache_size}")
        
        cache_ttl = cfg_pipeline.get('cache_ttl_seconds', 300)
        if cache_ttl < 0 or cache_ttl > 86400:
            raise ConfigurationError(f"Invalid pipeline.cache_ttl_seconds: {cache_ttl}")
        
        cfg_rule = config.get('rule_engine', {})
        weights = cfg_rule.get('weights', {})
        total_weight = sum(weights.values()) if weights else 0
        if total_weight > 1.5:
            raise ConfigurationError(f"Rule weights total exceeds 1.5: {total_weight}")
        
        data_path = cfg_rule.get('data_path', '')
        if data_path and not Path(data_path).exists() and not data_path.startswith('http'):
            raise ConfigurationError(f"Rules file not found: {data_path}")
    
    def _apply_env_overrides(self, config: Dict) -> Dict:
        if os.environ.get('SENTINEL_MODEL_NAME'):
            config.setdefault('model', {})['name'] = os.environ['SENTINEL_MODEL_NAME']
        if os.environ.get('SENTINEL_MODEL_PATH'):
            config.setdefault('model', {})['checkpoint_path'] = os.environ['SENTINEL_MODEL_PATH']
        if os.environ.get('SENTINEL_LOG_LEVEL'):
            config.setdefault('logging', {})['level'] = os.environ['SENTINEL_LOG_LEVEL']
        if os.environ.get('SENTINEL_CACHE_SIZE'):
            config.setdefault('pipeline', {})['classify_cache_size'] = int(os.environ['SENTINEL_CACHE_SIZE'])
        return config

    def _validate_input(self, text: str) -> None:
        if not text:
            raise ValidationError("Input text cannot be empty", {"text": repr(text)[:50]})
        if not isinstance(text, str):
            raise ValidationError(f"Input must be a string, got {type(text).__name__}", {"type": type(text).__name__})
        if not text.strip():
            raise ValidationError("Input text cannot be only whitespace", {"text": repr(text)[:50]})

    def _validate_batch_inputs(self, texts: List[str]) -> None:
        if not isinstance(texts, list):
            raise ValidationError(f"texts must be a list, got {type(texts).__name__}", {"type": type(texts).__name__})
        if len(texts) > 1000:
            raise ValidationError(f"Batch size exceeds maximum of 1000: {len(texts)}", {"size": len(texts)})
        for i, text in enumerate(texts):
            if not text:
                raise ValidationError(f"Empty text at index {i}", {"index": i})
            if not isinstance(text, str):
                raise ValidationError(f"Text at index {i} must be string, got {type(text).__name__}", {"index": i, "type": type(text).__name__})

    def _default_config(self) -> Dict:
        return {
            'rule_engine': {'data_path': 'data/rules/keywords.yaml'},
            'model': {
                'name': 'distilbert-base-uncased',
                'num_labels': 4
            },
            'pipeline': {'classify_cache_size': 0, 'cache_ttl_seconds': 300, 'include_latency_ms': False},
        }

    def _setup_logging(self) -> None:
        log_config = self.config.get('logging', {})
        self.log_level = log_config.get('level', 'INFO')
        self.log_file = log_config.get('file', 'logs/sentinel.log')
        self.log_console = log_config.get('console', True)
        self.audit_enabled = log_config.get('audit_enabled', True)
        self.audit_file = log_config.get('audit_file', 'logs/audit.log')
        Path('logs').mkdir(exist_ok=True)

        self._audit_logger = logging.getLogger('sentinel.pipeline.classify')
        self._audit_logger.propagate = False
        level = getattr(logging, str(self.log_level).upper(), logging.INFO)
        self._audit_logger.setLevel(level)
        if self.log_console and not self._audit_logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(message)s'))
            self._audit_logger.addHandler(h)

    def classify(self, text: str, return_raw: bool = False, request_id: Optional[str] = None, max_length: Optional[int] = None) -> Dict:
        """Classify a single text through the full pipeline.
        
        Args:
            text: Input text to classify
            return_raw: If True, include raw preprocessor/rule/ML results
            request_id: Optional correlation ID for tracing
            max_length: Optional max input length override
            
        Returns:
            Dict with label, risk_score, confidence, flagged_terms, etc.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        self._validate_input(text)
        
        input_max_length = max_length if max_length is not None else self.config.get('pipeline', {}).get('max_input_length', 10000)
        if input_max_length and len(text) > input_max_length:
            text = text[:input_max_length]

        if (
            self._classify_cache_max > 0
            and not return_raw
            and text in self._classify_cache
        ):
            cached_result, cached_time = self._classify_cache[text]
            if time.time() - cached_time <= self._cache_ttl_seconds:
                self._cache_hits += 1
                self._classify_cache.move_to_end(text)
                t0 = time.perf_counter()
                output = {
                    **cached_result,
                    'timestamp': datetime.now().isoformat(),
                    'input': text,
                    'audit_id': str(uuid.uuid4()),
                    'request_id': request_id,
                }
                if self._include_latency_ms:
                    output['latency_ms'] = round((time.perf_counter() - t0) * 1000, 2)
                self._log_result(output, request_id)
                self._metrics.increment_requests()
                self._metrics.record_label(cached_result.get('label', 'Unknown'))
                self._metrics.record_review(cached_result.get('requires_human_review', False))
                return output
            else:
                del self._classify_cache[text]
        
        self._cache_misses += 1
        
        t0 = time.perf_counter()
        preprocessed = self.preprocessor.preprocess(text)
        rule_result = self.rule_engine.analyze(preprocessed['cleaned'])
        
        ml_result: Dict[str, Any] = {"label": "Non-Radical", "confidence": 0.5, "probabilities": {"Non-Radical": 0.5, "Mildly Radical": 0.2, "Moderately Radical": 0.2, "Highly Radical": 0.1}}
        ml_warning = None
        try:
            ml_result = self.classifier.predict(text)
        except Exception as e:
            ml_warning = f"ML model unavailable: {str(e)}"
            self._audit_logger.warning(ml_warning)

        fused_result = self.fusion.fuse(rule_result, ml_result)
        if ml_warning:
            fused_result["reasoning"] = fused_result.get("reasoning", "") + f" ({ml_warning})"
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        output = {
            'timestamp': datetime.now().isoformat(),
            'audit_id': str(uuid.uuid4()),
            'request_id': request_id,
            'input': text,
            'label': fused_result['label'],
            'confidence': fused_result['confidence'],
            'risk_score': fused_result['risk_score'],
            'flagged_terms': fused_result['flagged_terms'],
            'requires_human_review': fused_result['requires_human_review'],
            'bias_metadata': fused_result['bias_metadata'],
            'reasoning': fused_result['reasoning'],
            'rule_amplification': fused_result['rule_amplification'],
        }
        if self._include_latency_ms:
            output['latency_ms'] = latency_ms

        if return_raw:
            output['raw'] = {
                'preprocessed': preprocessed,
                'rule_result': rule_result,
                'ml_result': ml_result,
            }

        self._log_result(output, request_id)
        
        self._metrics.increment_requests()
        self._metrics.record_label(output.get('label', 'Unknown'))
        self._metrics.record_review(output.get('requires_human_review', False))

        if self._classify_cache_max > 0 and not return_raw:
            stored = {
                k: v
                for k, v in output.items()
                if k not in ('timestamp', 'audit_id', 'latency_ms', 'request_id')
            }
            self._classify_cache[text] = (stored, time.time())
            self._classify_cache.move_to_end(text)
            while len(self._classify_cache) > self._classify_cache_max:
                self._classify_cache.popitem(last=False)

        return output

def classify_batch(
        self,
        texts: List[str],
        parallel: bool = False,
        request_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict]:
        """Classify multiple texts in batch.
        
        Args:
            texts: List of input texts to classify
            parallel: If True, use parallel processing (ThreadPoolExecutor)
            request_id: Optional correlation ID for tracing
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            List of classification results
        """
        if not texts:
            return []

        self._validate_batch_inputs(texts)

        if request_id is None:
            request_id = str(uuid.uuid4())

        if parallel:
            return self._classify_batch_parallel(texts, request_id, progress_callback)
        
        return self._classify_batch_sequential(texts, request_id, progress_callback)

    def _classify_batch_sequential(self, texts: List[str], request_id: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
        total = len(texts)
        t0 = time.perf_counter()
        preprocessed = [self.preprocessor.preprocess(t) for t in texts]
        rule_results = [self.rule_engine.analyze(p['cleaned']) for p in preprocessed]
        ml_batch = self.classifier.predict_batch(texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_item_ms = round(elapsed_ms / len(texts), 2) if self._include_latency_ms else None
        self._batch_stats.total = total
        self._batch_stats.processed = len(texts)
        self._batch_stats.elapsed_ms = elapsed_ms
        self._batch_stats.items_per_second = round(len(texts) / (elapsed_ms / 1000), 2) if elapsed_ms > 0 else 0

        outputs: List[Dict] = []
        for i, text in enumerate(texts):
            if total > 10 and i % max(1, total // 10) == 0:
                pct = round((i / total) * 100)
                print(f"\rProcessing: {pct}% ({i}/{total})", file=sys.stderr, end='', flush=True)
                if progress_callback:
                    progress_callback(i, total)
            rule_result = rule_results[i]
            ml_result = ml_batch[i]
            ml_for_fuse = {k: v for k, v in ml_result.items() if k != 'text'}
            fused_result = self.fusion.fuse(rule_result, ml_for_fuse)
            output = {
                'timestamp': datetime.now().isoformat(),
                'audit_id': str(uuid.uuid4()),
                'request_id': f"{request_id}_{i}",
                'input': text,
                'label': fused_result['label'],
                'confidence': fused_result['confidence'],
                'risk_score': fused_result['risk_score'],
                'flagged_terms': fused_result['flagged_terms'],
                'requires_human_review': fused_result['requires_human_review'],
                'bias_metadata': fused_result['bias_metadata'],
                'reasoning': fused_result['reasoning'],
                'rule_amplification': fused_result['rule_amplification'],
            }
            if per_item_ms is not None:
                output['latency_ms'] = per_item_ms
            self._log_result(output, request_id)
            outputs.append(output)

        if progress_callback:
            progress_callback(total, total)
        if total > 10:
            print(f"\rProcessing: 100% ({total}/{total})", file=sys.stderr)
        return outputs

    def _classify_batch_parallel(self, texts: List[str], request_id: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
        from concurrent.futures import ThreadPoolExecutor
        outputs: List[Dict] = []
        
        max_workers = min(self._batch_max_workers, len(texts))
        total = len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if progress_callback:
                completed = 0
                results = []
                futures = {executor.submit(self.classify, text, request_id=f"{request_id}_{i}"): i for i, text in enumerate(texts)}
                for future in futures:
                    results.append(future.result())
                    completed += 1
                    progress_callback(completed, total)
                results.sort(key=lambda x: int(x.get('request_id', '0').split('_')[-1]))
                outputs = results
            else:
                results = list(executor.map(
                    lambda args: self.classify(args[0], request_id=args[1]),
                    [(text, f"{request_id}_{i}") for i, text in enumerate(texts)]
                ))
                outputs = results
        
        return outputs

    def classify_batch_chunked(self, texts: List[str], request_id: Optional[str] = None) -> List[Dict]:
        if not texts:
            return []
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        total = len(texts)
        outputs: List[Dict] = []
        
        for start in range(0, total, self._batch_chunk_size):
            end = min(start + self._batch_chunk_size, total)
            chunk = texts[start:end]
            chunk_request_id = f"{request_id}_chunk_{start // self._batch_chunk_size}"
            chunk_results = self._classify_batch_parallel(chunk, chunk_request_id)
            outputs.extend(chunk_results)
        
        return outputs

    async def classify_batch_async(self, texts: List[str], request_id: Optional[str] = None) -> List[Dict]:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        if not texts:
            return []
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        loop = asyncio.get_event_loop()
        
        async def run_in_executor():
            executor = ThreadPoolExecutor(max_workers=self._batch_max_workers)
            return await loop.run_in_executor(
                executor,
                lambda: self._classify_batch_sequential(texts, request_id)
            )
        
        return await run_in_executor()

    ALLOWED_EXTENSIONS = {'.json', '.jsonl', '.txt'}
    ALLOWED_INPUT_DIRS = {'data/raw', 'data/processed', 'data/uploads'}
    
    def _validate_file_path(self, file_path: str) -> Path:
        resolved_path = Path(file_path).resolve()
        
        if '..' in Path(file_path).parts:
            raise ValueError(f"Path traversal detected in: {file_path}")
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        if not resolved_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if resolved_path.suffix not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {resolved_path.suffix}. Allowed: {self.ALLOWED_EXTENSIONS}")
        
        parent_dir = str(resolved_path.parent)
        is_allowed = any(
            parent_dir.startswith(str(Path(allowed_dir).resolve()))
            for allowed_dir in self.ALLOWED_INPUT_DIRS
        )
        if not is_allowed:
            raise ValueError(f"File must be in allowed directories: {self.ALLOWED_INPUT_DIRS}")
        
        return resolved_path

    def classify_from_file(self, file_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """Classify texts from a file.
        
        Args:
            file_path: Path to input file (.json, .jsonl, .txt)
            output_path: Optional path to write results
            
        Returns:
            List of classification results
        """
        path = self._validate_file_path(file_path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                texts = [item.get('text', item.get('content', '')) for item in data]
        elif path.suffix == '.jsonl':
            texts = []
            with open(path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    texts.append(item.get('text', item.get('content', '')))
        elif path.suffix == '.txt':
            with open(path, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        results = self.classify_batch(texts)
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results

    def get_cache_stats(self) -> Dict:
        return {
            'enabled': self._classify_cache_max > 0,
            'max_size': self._classify_cache_max,
            'ttl_seconds': self._cache_ttl_seconds,
            'current_size': len(self._classify_cache),
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': round(self._cache_hits / max(1, self._cache_hits + self._cache_misses), 3),
            'batch_stats': {
                'total': self._batch_stats.total,
                'processed': self._batch_stats.processed,
                'elapsed_ms': self._batch_stats.elapsed_ms,
                'items_per_second': self._batch_stats.items_per_second,
            } if self._batch_stats.total > 0 else None,
        }

    def get_prometheus_metrics(self) -> str:
        return self._metrics.get_metrics()

    def shutdown(self) -> Dict[str, Any]:
        cache_items = len(self._classify_cache)
        self._classify_cache.clear()
        
        if hasattr(self, 'classifier') and self.classifier is not None:
            try:
                self.classifier.unload()
            except Exception:
                pass
        
        stats = self.get_cache_stats()
        return {
            "status": "shutdown_complete",
            "cache_cleared": cache_items,
            "model_unloaded": True
        }

    def reset_cache(self) -> Dict:
        cleared_items = len(self._classify_cache)
        self._classify_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        return {'status': 'cleared', 'items_removed': cleared_items}

    def _log_result(self, result: Dict, request_id: Optional[str] = None) -> None:
        if request_id:
            result['request_id'] = request_id
        
        if self.log_console and self._audit_logger.handlers:
            log_entry = {
                'event': 'classification',
                'request_id': request_id,
                'result': result
            }
            line = json.dumps(log_entry, default=str)
            self._audit_logger.info(line)
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            except IOError as e:
                self._audit_logger.error(f"Failed to write to log file {self.log_file}: {e}")
            except Exception as e:
                self._audit_logger.error(f"Unexpected error writing to log file {self.log_file}: {e}")
        if self.audit_enabled and self.audit_file:
            try:
                with open(self.audit_file, 'a') as f:
                    audit_entry = {
                        'timestamp': result.get('timestamp'),
                        'request_id': request_id,
                        'audit_id': result.get('audit_id'),
                        'label': result.get('label'),
                        'risk_score': result.get('risk_score'),
                        'confidence': result.get('confidence'),
                        'requires_human_review': result.get('requires_human_review'),
                        'bias_metadata': result.get('bias_metadata'),
                    }
                    f.write(json.dumps(audit_entry) + '\n')
            except IOError as e:
                self._audit_logger.error(f"Failed to write to audit file {self.audit_file}: {e}")
            except Exception as e:
                self._audit_logger.error(f"Unexpected error writing to audit file {self.audit_file}: {e}")
