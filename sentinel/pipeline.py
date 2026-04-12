import logging
import sys
import time
import uuid
import yaml
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from .preprocessor import TextPreprocessor
from .rule_engine import RuleEngine
from .fusion import ScoreFusion


class SentinelPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        from .classifier import RadicalClassifier

        self.config = self._load_config(config_path)
        self.preprocessor = TextPreprocessor()
        self.rule_engine = RuleEngine(
            rules_path=self.config['rule_engine']['data_path']
        )
        pipe_cfg = self.config.get('pipeline', {})
        self._classify_cache_max = max(0, int(pipe_cfg.get('classify_cache_size', 0)))
        self._include_latency_ms = bool(pipe_cfg.get('include_latency_ms', False))
        self._batch_max_workers = int(pipe_cfg.get('batch_max_workers', 4))
        self._batch_chunk_size = int(pipe_cfg.get('batch_chunk_size', 8))
        self._classify_cache: OrderedDict[str, Dict] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._setup_logging()

    def warmup(self, num_inferences: int = 3) -> Dict:
        self.classifier.warmup(num_inferences)
        return {"status": "warmup_complete", "model_loaded": self.classifier.is_loaded}

    def _load_config(self, config_path: str) -> Dict:
        path = Path(config_path)
        if not path.exists():
            return self._default_config()
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _default_config(self) -> Dict:
        return {
            'rule_engine': {'data_path': 'data/rules/keywords.yaml'},
            'model': {
                'name': 'distilbert-base-uncased',
                'num_labels': 4
            },
            'pipeline': {'classify_cache_size': 0, 'include_latency_ms': False},
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

    def classify(self, text: str, return_raw: bool = False, request_id: Optional[str] = None) -> Dict:
        if request_id is None:
            request_id = str(uuid.uuid4())

        if (
            self._classify_cache_max > 0
            and not return_raw
            and text in self._classify_cache
        ):
            self._cache_hits += 1
            self._classify_cache.move_to_end(text)
            cached = self._classify_cache[text]
            t0 = time.perf_counter()
            output = {
                **cached,
                'timestamp': datetime.now().isoformat(),
                'input': text,
                'audit_id': str(uuid.uuid4()),
                'request_id': request_id,
            }
            if self._include_latency_ms:
                output['latency_ms'] = round((time.perf_counter() - t0) * 1000, 2)
            self._log_result(output, request_id)
            return output
        else:
            self._cache_misses += 1

        t0 = time.perf_counter()
        preprocessed = self.preprocessor.preprocess(text)
        rule_result = self.rule_engine.analyze(preprocessed['cleaned'])
        ml_result = self.classifier.predict(text)

        fused_result = self.fusion.fuse(rule_result, ml_result)
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

        if self._classify_cache_max > 0 and not return_raw:
            stored = {
                k: v
                for k, v in output.items()
                if k not in ('timestamp', 'audit_id', 'latency_ms', 'request_id')
            }
            self._classify_cache[text] = stored
            self._classify_cache.move_to_end(text)
            while len(self._classify_cache) > self._classify_cache_max:
                self._classify_cache.popitem(last=False)

        return output

    def classify_batch(self, texts: List[str], parallel: bool = False, request_id: Optional[str] = None) -> List[Dict]:
        if not texts:
            return []

        if request_id is None:
            request_id = str(uuid.uuid4())

        if parallel:
            return self._classify_batch_parallel(texts, request_id)
        
        return self._classify_batch_sequential(texts, request_id)

    def _classify_batch_sequential(self, texts: List[str], request_id: str) -> List[Dict]:
        total = len(texts)
        t0 = time.perf_counter()
        preprocessed = [self.preprocessor.preprocess(t) for t in texts]
        rule_results = [self.rule_engine.analyze(p['cleaned']) for p in preprocessed]
        ml_batch = self.classifier.predict_batch(texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_item_ms = round(elapsed_ms / len(texts), 2) if self._include_latency_ms else None

        outputs: List[Dict] = []
        for i, text in enumerate(texts):
            if total > 10 and i % max(1, total // 10) == 0:
                pct = round((i / total) * 100)
                print(f"\rProcessing: {pct}% ({i}/{total})", file=sys.stderr, end='', flush=True)
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

        if total > 10:
            print(f"\rProcessing: 100% ({total}/{total})", file=sys.stderr)
        return outputs

    def _classify_batch_parallel(self, texts: List[str], request_id: str) -> List[Dict]:
        from concurrent.futures import ThreadPoolExecutor
        outputs: List[Dict] = []
        
        max_workers = min(self._batch_max_workers, len(texts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
            'current_size': len(self._classify_cache),
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': round(self._cache_hits / max(1, self._cache_hits + self._cache_misses), 3),
        }

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
