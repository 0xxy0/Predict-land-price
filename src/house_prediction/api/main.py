import logging
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from house_prediction.api.schemas import PredictionRequest, PredictionResponse
from house_prediction.config import Settings
from house_prediction.logging_config import configure_logging
from house_prediction.training.metadata import load_metadata
from house_prediction.training.persistence import load_model

configure_logging()
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="House Price Prediction API", version="1.0.0")
    current_settings = settings or Settings()
    model_path = current_settings.model_path
    metadata_path = current_settings.metadata_path
    metadata = load_metadata(metadata_path) if metadata_path.exists() else None
    model = load_model(model_path) if model_path.exists() else None

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"message": "Welcome to the House Price Prediction API!"}

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/ready")
    async def ready_check() -> dict[str, str]:
        if model_path.exists():
            return {"status": "ready"}
        else:
            return {"status": "not_ready"}
        #

    @app.get("/model-info")
    async def model_info() -> dict[str, Any]:
        if metadata is None:
            raise HTTPException(status_code=503, detail="Model metadata not available")
        return metadata

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")

        request_frame = pd.DataFrame([request.model_dump()])
        prediction = model.predict(request_frame)[0]

        if metadata is None:
            raise HTTPException(status_code=503, detail="Model metadata not available")

        logger.info("Prediction request processed successfully.")

        return PredictionResponse(
            prediction=float(prediction),
            model_version=metadata["model_version"],
        )

    return app


app = create_app()


def main() -> None:
    settings = Settings()

    uvicorn.run(
        "house_prediction.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
