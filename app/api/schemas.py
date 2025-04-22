from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class PropertyFeatures(BaseModel):
    square_feet: float
    bedrooms: int
    bathrooms: float
    year_built: int
    location_zip: str
    lot_size: Optional[float] = None
    has_garage: bool = False
    has_pool: bool = False
    property_type: str
    description: Optional[str] = None


class PredictionRequest(BaseModel):
    property_id: Optional[str] = None
    features: PropertyFeatures
    include_explainability: bool = False
    model_version: Optional[str] = None


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class PredictionResponse(BaseModel):
    property_id: Optional[str] = None
    predicted_price: float
    prediction_interval: List[float] = Field(..., description="Lower and upper bounds of prediction interval")
    confidence_score: float
    model_version: str
    feature_importances: Optional[List[FeatureImportance]] = None