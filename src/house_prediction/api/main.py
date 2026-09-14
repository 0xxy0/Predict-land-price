import pandas as pd
from fastapi import FastAPI, HTTPException

from house_prediction.api.schemas import PredictionRequest, PredictionResponse
from house_prediction.config import Settings
from house_prediction.training.metadata import load_metadata
from house_prediction.training.persistence import load_model


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="House Price Prediction API", version="1.0.0")
    current_settings = settings or Settings()
    model_path = current_settings.model_path
    metadata_path = current_settings.metadata_path
    metadata = load_metadata(metadata_path) if metadata_path.exists() else None
    model = load_model(model_path) if model_path.exists() else None

    @app.get("/")
    async def root():
        return {"message": "Welcome to the House Price Prediction API!"}

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.get("/ready")
    async def ready_check():
        if model_path.exists():
            return {"status": "ready"}
        else:
            return {"status": "not_ready"}
        #

    @app.get("/model-info")
    async def model_info() -> dict:
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

        return PredictionResponse(
            prediction=float(prediction),
            model_version=metadata["model_version"],
        )

    return app


app = create_app()

def main() -> None:
    import uvicorn
    
    uvicorn.run(
        "house_prediction.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )