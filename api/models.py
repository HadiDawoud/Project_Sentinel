from typing import Dict, List, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class ClassificationRequest(BaseModel):
    text: str
    return_raw: bool = False


class ClassificationResponse(BaseModel):
    audit_id: str
    label: str
    confidence: float
    risk_score: int
    flagged_terms: List[str]
    reasoning: str
    latency_ms: Optional[float] = None


class BatchClassificationRequest(BaseModel):
    texts: List[str]


class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]


class CacheStatsResponse(BaseModel):
    enabled: bool
    max_size: int
    current_size: int
    hits: int
    misses: int
    hit_rate: float
