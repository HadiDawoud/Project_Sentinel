from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from sentinel.pipeline import SentinelPipeline

app = FastAPI(
    title="Project Sentinel API",
    description=(
        "Hybrid rule-based keyword/pattern matching plus DistilBERT classification. "
        "Use interactive docs at `/docs` (Swagger UI) or `/redoc`."
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "classification", "description": "Single and batch text classification"},
        {"name": "meta", "description": "Service health and metadata"},
    ],
)
pipeline = SentinelPipeline()


class TextInput(BaseModel):
    text: str
    return_raw: bool = False


class BatchInput(BaseModel):
    texts: List[str]


class ClassificationResult(BaseModel):
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
async def classify(input_data: TextInput):
    try:
        result = pipeline.classify(input_data.text, return_raw=input_data.return_raw)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", tags=["classification"], summary="Classify multiple texts")
async def classify_batch(input_data: BatchInput):
    try:
        results = pipeline.classify_batch(input_data.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/file", tags=["classification"], summary="Classify from a local file path")
async def classify_file(file_url: str, output_format: str = "json"):
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
