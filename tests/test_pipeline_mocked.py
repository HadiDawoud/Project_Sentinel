import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from sentinel.pipeline import SentinelPipeline
from sentinel.constants import LABEL_MAP


class TestSentinelPipelineMocked:
    @pytest.fixture
    def mock_pipeline_components(self):
        with patch('sentinel.pipeline.TextPreprocessor') as mock_preprocessor, \
             patch('sentinel.pipeline.RuleEngine') as mock_rule_engine, \
             patch('sentinel.pipeline.ScoreFusion') as mock_fusion, \
             patch('sentinel.pipeline.RadicalClassifier') as mock_classifier:
            
            mock_preprocessor_instance = MagicMock()
            mock_preprocessor_instance.preprocess.return_value = {
                'original': 'test text',
                'cleaned': 'test text',
                'tokens': ['test', 'text'],
                'token_count': 2
            }
            mock_preprocessor.return_value = mock_preprocessor_instance
            
            mock_rule_engine_instance = MagicMock()
            mock_rule_engine_instance.analyze.return_value = {
                'flagged': False,
                'matched_terms': [],
                'risk_score': 0,
                'has_high_risk_terms': False
            }
            mock_rule_engine.return_value = mock_rule_engine_instance
            
            mock_classifier_instance = MagicMock()
            mock_classifier_instance.predict.return_value = {
                'label': 'Non-Radical',
                'label_id': 0,
                'confidence': 0.95,
                'probabilities': {label: 0.25 for label in LABEL_MAP.values()}
            }
            mock_classifier.return_value = mock_classifier_instance
            
            mock_fusion_instance = MagicMock()
            mock_fusion_instance.fuse.return_value = {
                'label': 'Non-Radical',
                'risk_score': 0,
                'confidence': 0.95,
                'flagged_terms': [],
                'requires_human_review': False,
                'bias_metadata': {},
                'reasoning': 'Normal content',
                'rule_amplification': False
            }
            mock_fusion.return_value = mock_fusion_instance
            
            yield {
                'pipeline': None,
                'preprocessor': mock_preprocessor_instance,
                'rule_engine': mock_rule_engine_instance,
                'classifier': mock_classifier_instance,
                'fusion': mock_fusion_instance
            }

    def test_pipeline_initializes_with_lazy_load(self, mock_pipeline_components):
        pipeline = SentinelPipeline.__new__(SentinelPipeline)
        pipeline._setup_logging = MagicMock()
        pipeline._load_config = MagicMock(return_value={
            'rule_engine': {'data_path': 'data/rules/keywords.yaml', 'weights': {'high_risk': 0.3}},
            'model': {'name': 'distilbert-base-uncased', 'num_labels': 4},
            'pipeline': {'classify_cache_size': 0, 'include_latency_ms': False, 'lazy_load_model': True}
        })
        
        from sentinel.pipeline import TextPreprocessor, RuleEngine, ScoreFusion, RadicalClassifier
        
        with patch.object(TextPreprocessor, '__init__', return_value=None), \
             patch.object(RuleEngine, '__init__', return_value=None), \
             patch.object(ScoreFusion, '__init__', return_value=None), \
             patch.object(RadicalClassifier, '__init__', return_value=None):
            
            pipeline.__init__(pipeline._load_config('config.yaml'))

    def test_classify_returns_request_id(self, mock_pipeline_components):
        pipeline = SentinelPipeline.__new__(SentinelPipeline)
        pipeline._setup_logging = MagicMock()
        pipeline._load_config = MagicMock(return_value={
            'rule_engine': {'data_path': 'data/rules/keywords.yaml', 'weights': {'high_risk': 0.3}},
            'model': {'name': 'distilbert-base-uncased', 'num_labels': 4},
            'pipeline': {'classify_cache_size': 0, 'include_latency_ms': False}
        })
        
        pipeline.preprocessor = mock_pipeline_components['preprocessor']
        pipeline.rule_engine = mock_pipeline_components['rule_engine']
        pipeline.classifier = mock_pipeline_components['classifier']
        pipeline.fusion = mock_pipeline_components['fusion']
        pipeline._classify_cache_max = 0
        pipeline._include_latency_ms = False
        pipeline._classify_cache = {}
        pipeline._cache_hits = 0
        pipeline._cache_misses = 0
        pipeline.log_console = False
        pipeline.log_file = None
        pipeline.audit_enabled = False
        pipeline._audit_logger = MagicMock()
        
        result = pipeline.classify("test text", request_id="test-request-123")
        
        assert 'request_id' in result
        assert result['request_id'] == "test-request-123"

    def test_classify_batch_propagates_request_id(self, mock_pipeline_components):
        pipeline = SentinelPipeline.__new__(SentinelPipeline)
        pipeline._setup_logging = MagicMock()
        pipeline._load_config = MagicMock(return_value={
            'rule_engine': {'data_path': 'data/rules/keywords.yaml', 'weights': {'high_risk': 0.3}},
            'model': {'name': 'distilbert-base-uncased', 'num_labels': 4},
            'pipeline': {'classify_cache_size': 0, 'include_latency_ms': False}
        })
        
        pipeline.preprocessor = mock_pipeline_components['preprocessor']
        pipeline.rule_engine = mock_pipeline_components['rule_engine']
        pipeline.classifier = mock_pipeline_components['classifier']
        pipeline.classifier.predict_batch.return_value = [
            {'label': 'Non-Radical', 'label_id': 0, 'confidence': 0.95, 'probabilities': {}},
            {'label': 'Non-Radical', 'label_id': 0, 'confidence': 0.95, 'probabilities': {}}
        ]
        pipeline.fusion = mock_pipeline_components['fusion']
        pipeline._classify_cache_max = 0
        pipeline._include_latency_ms = False
        pipeline._classify_cache = {}
        pipeline._cache_hits = 0
        pipeline._cache_misses = 0
        pipeline.log_console = False
        pipeline.log_file = None
        pipeline.audit_enabled = False
        pipeline._audit_logger = MagicMock()
        
        result = pipeline.classify_batch(["text1", "text2"], request_id="batch-request-123")
        
        assert len(result) == 2
        assert result[0]['request_id'] == "batch-request-123_0"
        assert result[1]['request_id'] == "batch-request-123_1"

    def test_classify_from_file_path_validation(self, mock_pipeline_components):
        pipeline = SentinelPipeline.__new__(SentinelPipeline)
        pipeline._setup_logging = MagicMock()
        pipeline._load_config = MagicMock(return_value={
            'rule_engine': {'data_path': 'data/rules/keywords.yaml', 'weights': {'high_risk': 0.3}},
            'model': {'name': 'distilbert-base-uncased', 'num_labels': 4},
            'pipeline': {'classify_cache_size': 0, 'include_latency_ms': False}
        })
        
        pipeline.preprocessor = mock_pipeline_components['preprocessor']
        pipeline.rule_engine = mock_pipeline_components['rule_engine']
        pipeline.classifier = mock_pipeline_components['classifier']
        pipeline.classifier.predict_batch.return_value = []
        pipeline.fusion = mock_pipeline_components['fusion']
        pipeline._classify_cache_max = 0
        pipeline._include_latency_ms = False
        pipeline._classify_cache = {}
        pipeline._cache_hits = 0
        pipeline._cache_misses = 0
        pipeline.log_console = False
        pipeline.log_file = None
        pipeline.audit_enabled = False
        pipeline._audit_logger = MagicMock()
        
        from pathlib import Path
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                pipeline.classify_from_file("/nonexistent/path/file.json")


class TestPipelineWarmup:
    def test_warmup_calls_classifier_warmup(self):
        pipeline = SentinelPipeline.__new__(SentinelPipeline)
        pipeline._setup_logging = MagicMock()
        pipeline._load_config = MagicMock(return_value={
            'rule_engine': {'data_path': 'data/rules/keywords.yaml', 'weights': {'high_risk': 0.3}},
            'model': {'name': 'distilbert-base-uncased', 'num_labels': 4},
            'pipeline': {'classify_cache_size': 0, 'include_latency_ms': False}
        })
        
        pipeline.classifier = MagicMock()
        pipeline.classifier.is_loaded = True
        pipeline.classifier.warmup = MagicMock()
        
        result = pipeline.warmup(num_inferences=5)
        
        pipeline.classifier.warmup.assert_called_once_with(5)
        assert result['status'] == 'warmup_complete'
        assert result['model_loaded'] is True


class TestPipelineLogging:
    def test_log_result_includes_request_id(self):
        pipeline = SentinelPipeline.__new__(SentinelPipeline)
        pipeline._setup_logging = MagicMock()
        pipeline._load_config = MagicMock(return_value={
            'logging': {'level': 'INFO', 'file': None, 'console': False, 'audit_enabled': False}
        })
        
        pipeline.log_console = True
        pipeline.log_file = None
        pipeline.audit_enabled = False
        pipeline._audit_logger = MagicMock()
        pipeline._audit_logger.handlers = [MagicMock()]
        
        result = {'label': 'Non-Radical', 'audit_id': 'test-audit-id'}
        pipeline._log_result(result, request_id='test-request-id')
        
        pipeline._audit_logger.info.assert_called_once()
        call_args = pipeline._audit_logger.info.call_args[0][0]
        assert 'test-request-id' in call_args
