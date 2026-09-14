from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Schema for the prediction request.
    """

    date: str = Field(..., description="Date of the house sale in YYYY-MM-DD format")
    bedrooms: float = Field(ge=0, description="Number of bedrooms in the house")
    bathrooms: float = Field(ge=0, description="Number of bathrooms in the house")
    floors: float = Field(ge=0.5, description="Number of floors in the house")
    waterfront: int = Field(ge=0, le=1, description="Whether the house has a waterfront view (0 or 1)")
    view: int = Field(ge=0, le=4, description="Quality of the view from the house (0-4)")
    condition: int = Field(ge=1, le=5, description="Condition of the house (1-5)")
    sqft_living: float = Field(gt=0, description="Square footage of the living area")
    sqft_lot: float = Field(gt=0, description="Square footage of the lot")
    sqft_above: float = Field(ge=0, description="Square footage of the house above ground level")
    sqft_basement: float = Field(ge=0, description="Square footage of the basement")
    yr_built: int = Field(..., description="Year the house was built")
    yr_renovated: int = Field(ge=0, description="Year the house was renovated")
    city: str = Field(..., description="City where the house is located")
    state: str = Field(..., description="State where the house is located")
    zip_code: str = Field(..., description="Zip code of the house")


class PredictionResponse(BaseModel):
    """
    Schema for the prediction response.
    """

    prediction: float = Field(..., description="Predicted price of the house")
    model_version: str = Field(..., description="Version of the model used for prediction")
