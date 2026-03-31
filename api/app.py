import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from sentinel.pipeline import SentinelPipeline

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Project Sentinel API",
    description=(
        "Hybrid rule-based keyword/pattern matching plus DistilBERT classification. "
        "Use interactive docs at `/docs` (Swagger UI) or `/redoc`. "
        "If env `SENTINEL_API_KEY` is set, classification endpoints require header `X-API-Key`."
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
    text: str
    return_raw: bool = False


class BatchInput(BaseModel):
    texts: List[str]


class ClassificationResult(BaseModel):
    audit_id: str
    label: str
    confidence: float
    risk_score: int
    flagged_terms: List[str]
    reasoning: str


@app.get("/", tags=["meta"], summary="Health check")
async def root():
    return {"message": "Project Sentinel API", "status": "running"}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
