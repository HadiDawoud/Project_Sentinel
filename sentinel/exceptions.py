class SentinelError(Exception):
    pass


class ConfigurationError(SentinelError):
    pass


class ModelLoadError(SentinelError):
    pass


class PredictionError(SentinelError):
    pass


class ValidationError(SentinelError):
    pass


class RuleEngineError(SentinelError):
    pass