import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from .preprocessor import TextPreprocessor
from .rule_engine import RuleEngine
from .classifier import RadicalClassifier
from .fusion import ScoreFusion


class SentinelPipeline:
    def __init__(self, config_path: str = "config.yaml"):
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
            }
        }

    def _setup_logging(self) -> None:
        log_config = self.config.get('logging', {})
        self.log_level = log_config.get('level', 'INFO')
        self.log_file = log_config.get('file', 'logs/sentinel.log')
        Path('logs').mkdir(exist_ok=True)

    def classify(self, text: str, return_raw: bool = False) -> Dict:
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
            'rule_amplification': fused_result['rule_amplification']
        }
        
        if return_raw:
            output['raw'] = {
                'preprocessed': preprocessed,
                'rule_result': rule_result,
                'ml_result': ml_result
            }
        
        self._log_result(output)
        return output

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        return [self.classify(text) for text in texts]

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
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            except Exception:
                pass
