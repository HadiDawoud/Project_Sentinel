from .preprocessor import TextPreprocessor
from .rule_engine import RuleEngine
from .fusion import ScoreFusion
from .pipeline import SentinelPipeline
from .logger import StructuredLogger, get_logger

__version__ = '0.1.0'

__all__ = [
    'TextPreprocessor',
    'RuleEngine',
    'RadicalClassifier',
    'ScoreFusion',
    'SentinelPipeline',
    'StructuredLogger',
    'get_logger',
    '__version__',
]


def __getattr__(name: str):
    if name == 'RadicalClassifier':
        from .classifier import RadicalClassifier

        return RadicalClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
