from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from sentinel.pipeline import SentinelPipeline

app = FastAPI(title="Project Sentinel API")
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


@app.get("/")
async def root():
    return {"message": "Project Sentinel API", "status": "running"}


@app.post("/classify", response_model=ClassificationResult)
async def classify(input_data: TextInput):
    try:
        result = pipeline.classify(input_data.text, return_raw=input_data.return_raw)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch")
async def classify_batch(input_data: BatchInput):
    try:
        results = pipeline.classify_batch(input_data.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/file")
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
