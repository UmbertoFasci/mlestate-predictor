from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from app.api.schemas import PredictionRequest, PredictionResponse
from app.models.ensemble import RealEstateEnsembleModel
from app.utils.explainer import generate_explanation
import uuid

router = APIRouter()

# Global model instance
model = RealEstateEnsembleModel()

@router.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest, background_tasks: BackgroundTasks):
    try:
        # Generate a property ID if not provided
        property_id = request.property_id or str(uuid.uuid4())
        
        # Get prediction from model
        prediction_result = model.predict(
            features=request.features.dict(),
            version=request.model_version
        )
        
        response = PredictionResponse(
            property_id=property_id,
            predicted_price=prediction_result["price"],
            prediction_interval=prediction_result["interval"],
            confidence_score=prediction_result["confidence"],
            model_version=prediction_result["model_version"]
        )
        
        # Add explainability if requested
        if request.include_explainability:
            feature_importances = generate_explanation(
                model=model, 
                features=request.features.dict(),
                prediction=prediction_result["price"]
            )
            response.feature_importances = feature_importances
            
        # Log prediction in background
        background_tasks.add_task(
            model.log_prediction,
            property_id=property_id,
            features=request.features.dict(),
            prediction=prediction_result
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict/with-image")
async def predict_with_image(
    features: PredictionRequest, 
    image: UploadFile = File(...),
    background_tasks: BackgroundTasks
):
    # TODO
    pass