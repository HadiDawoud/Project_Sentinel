class SentinelError(Exception):
    def __init__(self, message: str, error_code: str = "SENTINEL_ERROR", details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(SentinelError):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONFIG_ERROR", details)
        self.message = message
        self.error_code = "CONFIG_ERROR"
        self.details = details or {}


class ModelLoadError(SentinelError):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "MODEL_LOAD_ERROR", details)
        self.message = message
        self.error_code = "MODEL_LOAD_ERROR"
        self.details = details or {}


class PredictionError(SentinelError):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "PREDICTION_ERROR", details)
        self.message = message
        self.error_code = "PREDICTION_ERROR"
        self.details = details or {}


class ValidationError(SentinelError):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.message = message
        self.error_code = "VALIDATION_ERROR"
        self.details = details or {}


class RuleEngineError(SentinelError):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "RULE_ENGINE_ERROR", details)
        self.message = message
        self.error_code = "RULE_ENGINE_ERROR"
        self.details = details or {}


class PreprocessingError(SentinelError):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "PREPROCESSING_ERROR", details)
        self.message = message
        self.error_code = "PREPROCESSING_ERROR"
        self.details = details or {}


class CacheError(SentinelError):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CACHE_ERROR", details)
        self.message = message
        self.error_code = "CACHE_ERROR"
        self.details = details or {}