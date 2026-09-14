from house_prediction.api.schemas import PredictionRequest, PredictionResponse


def test_api_request_schema_validation() -> None:
    valid_data = {
        "date": "2023-01-01",
        "bedrooms": 3,
        "bathrooms": 2.0,
        "floors": 1.0,
        "waterfront": 0,
        "view": 1,
        "condition": 4,
        "sqft_living": 1500,
        "sqft_lot": 5000,
        "sqft_above": 1500,
        "sqft_basement": 0,
        "yr_built": 1990,
        "yr_renovated": 2005,
        "city": "Seattle",
        "state": "WA",
        "zip_code": "98101",
    }

    # Validate the request schema with valid data
    request = PredictionRequest(**valid_data)
    assert request.date == valid_data["date"]
    assert request.bedrooms == valid_data["bedrooms"]
    assert request.bathrooms == valid_data["bathrooms"]
    assert request.floors == valid_data["floors"]
    assert request.waterfront == valid_data["waterfront"]
    assert request.view == valid_data["view"]
    assert request.condition == valid_data["condition"]
    assert request.sqft_living == valid_data["sqft_living"]
    assert request.sqft_lot == valid_data["sqft_lot"]
    assert request.sqft_above == valid_data["sqft_above"]
    assert request.sqft_basement == valid_data["sqft_basement"]
    assert request.yr_built == valid_data["yr_built"]
    assert request.yr_renovated == valid_data["yr_renovated"]
    assert request.city == valid_data["city"]
    assert request.state == valid_data["state"]
    assert request.zip_code == valid_data["zip_code"]


def test_api_response_schema_validation() -> None:
    valid_data = {"prediction": 500000.0, "model_version": "v1.0"}

    # Validate the response schema with valid data
    response = PredictionResponse(**valid_data)
    assert response.prediction == valid_data["prediction"]
    assert response.model_version == valid_data["model_version"]
