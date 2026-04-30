"""
PhishGuard AI - REST API Module
FastAPI application to serve the phishing URL classifier.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl

from data_prep import URLFeatureEngineer
from train_model import MODEL_PATH, load_model

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model_store: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load the trained model on startup and release resources on shutdown.

    The model is stored in the module-level `_model_store` dict so that
    route handlers can access it without relying on global variables directly.
    """
    try:
        _model_store["classifier"] = load_model(MODEL_PATH)
        print(f"[startup] Model loaded from: {MODEL_PATH.resolve()}")
    except FileNotFoundError as exc:
        print(f"[startup] WARNING — {exc}")
        print("[startup] Run train_model.py first to generate the model file.")
    yield
    _model_store.clear()
    print("[shutdown] Model unloaded.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PhishGuard AI",
    description="Phishing URL classifier powered by a Random Forest model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Schema for a single-URL classification request."""

    url: str = Field(
        ...,
        min_length=4,
        description="The full URL to classify (e.g. 'https://example.com/login').",
        examples=["http://paypal-secure-verify.com/account/confirm=true"],
    )


class PredictionResponse(BaseModel):
    """Schema for the classification result returned to the caller."""

    url: str = Field(..., description="The original URL that was analysed.")
    is_phishing: bool = Field(
        ..., description="True if the model classifies the URL as phishing."
    )
    phishing_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that the URL is a phishing page.",
    )
    features_extracted: Dict[str, Any] = Field(
        ..., description="Lexical features computed from the URL."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
async def health_check() -> Dict[str, str]:
    """
    Health-check endpoint.

    Returns a simple status payload so that load-balancers and CI pipelines
    can verify the service is running.
    """
    model_status = "loaded" if "classifier" in _model_store else "unavailable"
    return {"status": "ok", "model": model_status}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Classify a URL as phishing or benign",
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Classify a URL and return risk assessment.

    Steps performed:
    1. Validate that the model has been loaded.
    2. Extract lexical features from the URL using `URLFeatureEngineer`.
    3. Build a single-row DataFrame compatible with the trained model.
    4. Run `predict` and `predict_proba` to obtain class and confidence.
    5. Return a structured `PredictionResponse`.

    Args:
        request: A `PredictionRequest` body containing the target URL.

    Returns:
        `PredictionResponse` with classification result, probability and
        the feature dictionary used for the decision.

    Raises:
        HTTPException 503: If the model is not loaded.
        HTTPException 500: If feature extraction or inference fails.
    """
    classifier = _model_store.get("classifier")
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not available. "
                "Run train_model.py to train and save the model first."
            ),
        )

    # --- Feature extraction ---
    try:
        engineer = URLFeatureEngineer()
        feature_dict: Dict[str, Any] = engineer.extract_features(request.url)
        feature_df: pd.DataFrame = pd.DataFrame(
            [feature_dict], columns=URLFeatureEngineer.FEATURE_COLUMNS
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {exc}",
        ) from exc

    # --- Inference ---
    try:
        predicted_class: int = int(classifier.predict(feature_df)[0])
        phishing_proba: float = float(
            classifier.predict_proba(feature_df)[0][1]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {exc}",
        ) from exc

    return PredictionResponse(
        url=request.url,
        is_phishing=bool(predicted_class),
        phishing_probability=round(phishing_proba, 6),
        features_extracted=feature_dict,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )