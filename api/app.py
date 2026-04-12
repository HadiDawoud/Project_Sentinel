import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from sentinel.pipeline import SentinelPipeline
from sentinel.constants import MAX_INPUT_LENGTH

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Project Sentinel API",
    description=(
        "Hybrid rule-based keyword/pattern matching plus DistilBERT classification for radicalization detection.\n\n"
        "### IMPORTANT: Responsible AI Usage\n"
        "- **Assistive, Not Authoritative**: This tool is designed to assist human moderators. It should NOT be used for automated decision-making without human oversight.\n"
        "- **Bias & Fairness**: Automated detection can exhibit bias. High-risk religious, political, and identity-related terms may trigger false positives.\n"
        "- **Human Review**: Results flagged with `requires_human_review: true` MUST be reviewed by a human expert.\n"
        "- **Context Matters**: This system has limited understanding of cultural and linguistic context. Review reasoning and flagged terms carefully."
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "classification", "description": "Single and batch text classification"},
        {"name": "meta", "description": "Service health and metadata"},
    ],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

pipeline = SentinelPipeline()


def _optional_api_key_auth(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    expected = os.environ.get("SENTINEL_API_KEY")
    if not expected:
        return
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if not secrets.compare_digest(
        x_api_key.encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_LENGTH,
        description="Text to classify (1-10000 characters)"
    )
    return_raw: bool = False


class BatchInput(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to classify (max 100 items)"
    )

    @field_validator('texts')
    @classmethod
    def validate_text_lengths(cls, v):
        for i, text in enumerate(v):
            if len(text) > MAX_INPUT_LENGTH:
                raise ValueError(f"Text at index {i} exceeds max length of {MAX_INPUT_LENGTH} characters")
            if len(text.strip()) == 0:
                raise ValueError(f"Text at index {i} is empty or whitespace only")
        return v


class ClassificationResult(BaseModel):
    audit_id: str
    label: str
    confidence: float
    risk_score: int
    flagged_terms: List[str]
    requires_human_review: bool
    bias_metadata: Optional[dict] = None
    reasoning: str
    latency_ms: Optional[float] = None


@app.get("/", tags=["meta"], summary="Health check")
async def root():
    return {"message": "Project Sentinel API", "status": "running"}


@app.get("/health", tags=["meta"], summary="Detailed health status")
async def health():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "endpoints": {
            "classify": "/classify",
            "batch": "/classify/batch",
            "file": "/classify/file"
        }
    }


@app.get("/docs/info", tags=["meta"], summary="Get API documentation info")
async def docs_info():
    return {
        "title": "Project Sentinel API",
        "description": "Hybrid rule-based + ML classification for radicalization detection",
        "version": "0.1.0",
        "swagger_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json"
    }


@app.post(
    "/classify",
    response_model=ClassificationResult,
    tags=["classification"],
    summary="Classify one text",
    response_description="Fused rule + ML label, confidence, and risk score",
)
@limiter.limit("60/minute")
async def classify(
    request: Request,
    input_data: TextInput,
    _auth: None = Depends(_optional_api_key_auth),
):
    try:
        result = pipeline.classify(input_data.text, return_raw=input_data.return_raw)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", tags=["classification"], summary="Classify multiple texts")
@limiter.limit("30/minute")
async def classify_batch(
    request: Request,
    input_data: BatchInput,
    _auth: None = Depends(_optional_api_key_auth),
):
    try:
        results = pipeline.classify_batch(input_data.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/file", tags=["classification"], summary="Classify from a local file path")
@limiter.limit("30/minute")
async def classify_file(
    request: Request,
    file_url: str,
    output_format: str = "json",
    _auth: None = Depends(_optional_api_key_auth),
):
    try:
        results = pipeline.classify_from_file(file_url)
        return {"results": results, "count": len(results)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats", tags=["meta"], summary="Get cache statistics")
async def cache_stats():
    return pipeline.get_cache_stats()


@app.get("/latency", tags=["meta"], summary="Get pipeline latency info")
async def latency_info():
    return {
        "include_latency_ms": pipeline._include_latency_ms,
        "cache_enabled": pipeline._classify_cache_max > 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
