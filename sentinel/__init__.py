from .preprocessor import TextPreprocessor
from .rule_engine import RuleEngine
from .fusion import ScoreFusion
from .pipeline import SentinelPipeline

__all__ = [
    'TextPreprocessor',
    'RuleEngine',
    'RadicalClassifier',
    'ScoreFusion',
    'SentinelPipeline',
]


def __getattr__(name: str):
    if name == 'RadicalClassifier':
        from .classifier import RadicalClassifier

        return RadicalClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
