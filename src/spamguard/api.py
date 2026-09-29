"""FastAPI inference service for a trained SpamGuard lexical model."""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000, description="SMS text to score")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class BatchRequest(BaseModel):
    texts: list[Annotated[str, Field(min_length=1, max_length=5000)]] = Field(
        min_length=1, max_length=20
    )

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("each text must contain non-whitespace characters")
        return values


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("SPAMGUARD_MODEL", "artifacts/baseline/model.joblib"))
    if not model_path.is_file():
        raise RuntimeError(f"Model bundle not found at {model_path}; train the baseline first")
    app.state.bundle = joblib.load(model_path)
    yield


app = FastAPI(
    title="SpamGuard NLP API",
    description="Spam scoring for SMS text; the model is a research/demo classifier, not an automatic blocking control.",
    version="0.1.0",
    lifespan=lifespan,
)


def predict_one(text: str, bundle: dict) -> dict:
    pipeline = bundle["pipeline"]
    class_order = list(bundle.get("class_order", pipeline.classes_))
    if "spam" not in class_order:
        raise RuntimeError("model bundle has no spam class")
    probability = float(pipeline.predict_proba([text])[0][class_order.index("spam")])
    threshold = float(bundle["decision_threshold"])
    is_spam = probability >= threshold
    return {
        "label": "spam" if is_spam else "ham",
        "is_spam": is_spam,
        "spam_probability": probability,
        "decision_threshold": threshold,
        "model": bundle.get("model_name", "unknown"),
        "note": "Review uncertain messages; this demo should not automatically block or delete SMS messages.",
    }


@app.get("/health", tags=["service"])
def health(request: Request) -> dict:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": bundle.get("model_name", "unknown")}


@app.post("/predict", tags=["inference"])
def predict(payload: TextRequest, request: Request) -> dict:
    return predict_one(payload.text, request.app.state.bundle)


@app.post("/predict/batch", tags=["inference"])
def predict_batch(payload: BatchRequest, request: Request) -> dict:
    bundle = request.app.state.bundle
    pipeline = bundle["pipeline"]
    class_order = list(bundle.get("class_order", pipeline.classes_))
    probabilities = pipeline.predict_proba(payload.texts)[:, class_order.index("spam")]
    threshold = float(bundle["decision_threshold"])
    predictions = [
        {
            "label": "spam" if probability >= threshold else "ham",
            "is_spam": bool(probability >= threshold),
            "spam_probability": float(probability),
            "decision_threshold": threshold,
            "model": bundle.get("model_name", "unknown"),
        }
        for probability in probabilities
    ]
    return {"predictions": predictions}


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the SpamGuard FastAPI inference server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8019")))
    args = parser.parse_args()
    uvicorn.run("spamguard.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
