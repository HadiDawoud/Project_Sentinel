import logging
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
        self.classifier = RadicalClassifier(
            model_name=self.config['model']['name'],
            num_labels=self.config['model']['num_labels'],
            checkpoint_path=self.config['model'].get('checkpoint_path')
        )
        self.fusion = ScoreFusion(
            rule_weight=self.config['rule_engine']['weights'].get('high_risk', 0.3),
            ml_weight=0.7,
            amplification_factor=self.config['rule_engine'].get('amplification_factor', 1.5)
        )
        pipe_cfg = self.config.get('pipeline', {})
        self._classify_cache_max = max(0, int(pipe_cfg.get('classify_cache_size', 0)))
        self._classify_cache: OrderedDict[str, Dict] = OrderedDict()
        self._setup_logging()

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
            'pipeline': {'classify_cache_size': 0},
        }

    def _setup_logging(self) -> None:
        log_config = self.config.get('logging', {})
        self.log_level = log_config.get('level', 'INFO')
        self.log_file = log_config.get('file', 'logs/sentinel.log')
        self.log_console = log_config.get('console', True)
        Path('logs').mkdir(exist_ok=True)

        self._audit_logger = logging.getLogger('sentinel.pipeline.classify')
        self._audit_logger.propagate = False
        level = getattr(logging, str(self.log_level).upper(), logging.INFO)
        self._audit_logger.setLevel(level)
        if self.log_console and not self._audit_logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(message)s'))
            self._audit_logger.addHandler(h)

    def classify(self, text: str, return_raw: bool = False) -> Dict:
        if (
            self._classify_cache_max > 0
            and not return_raw
            and text in self._classify_cache
        ):
            self._classify_cache.move_to_end(text)
            cached = self._classify_cache[text]
            output = {
                **cached,
                'timestamp': datetime.now().isoformat(),
                'input': text,
            }
            self._log_result(output)
            return output

        preprocessed = self.preprocessor.preprocess(text)
        rule_result = self.rule_engine.analyze(preprocessed['cleaned'])
        ml_result = self.classifier.predict(text)

        fused_result = self.fusion.fuse(rule_result, ml_result)

        output = {
            'timestamp': datetime.now().isoformat(),
            'input': text,
            'label': fused_result['label'],
            'confidence': fused_result['confidence'],
            'risk_score': fused_result['risk_score'],
            'flagged_terms': fused_result['flagged_terms'],
            'reasoning': fused_result['reasoning'],
            'rule_amplification': fused_result['rule_amplification'],
        }

        if return_raw:
            output['raw'] = {
                'preprocessed': preprocessed,
                'rule_result': rule_result,
                'ml_result': ml_result,
            }

        self._log_result(output)

        if self._classify_cache_max > 0 and not return_raw:
            stored = {k: v for k, v in output.items() if k != 'timestamp'}
            self._classify_cache[text] = stored
            self._classify_cache.move_to_end(text)
            while len(self._classify_cache) > self._classify_cache_max:
                self._classify_cache.popitem(last=False)

        return output

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        if not texts:
            return []

        preprocessed = [self.preprocessor.preprocess(t) for t in texts]
        rule_results = [self.rule_engine.analyze(p['cleaned']) for p in preprocessed]
        ml_batch = self.classifier.predict_batch(texts)

        outputs: List[Dict] = []
        for i, text in enumerate(texts):
            rule_result = rule_results[i]
            ml_result = ml_batch[i]
            ml_for_fuse = {k: v for k, v in ml_result.items() if k != 'text'}
            fused_result = self.fusion.fuse(rule_result, ml_for_fuse)
            output = {
                'timestamp': datetime.now().isoformat(),
                'input': text,
                'label': fused_result['label'],
                'confidence': fused_result['confidence'],
                'risk_score': fused_result['risk_score'],
                'flagged_terms': fused_result['flagged_terms'],
                'reasoning': fused_result['reasoning'],
                'rule_amplification': fused_result['rule_amplification'],
            }
            self._log_result(output)
            outputs.append(output)

        return outputs

    def classify_from_file(self, file_path: str, output_path: Optional[str] = None) -> List[Dict]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
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

    def _log_result(self, result: Dict) -> None:
        if self.log_console and self._audit_logger.handlers:
            line = json.dumps({'event': 'classification', 'result': result}, default=str)
            self._audit_logger.info(line)
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            except Exception:
                pass
