from typing import Dict, List, Any
import numpy as np
import pandas as pd
import shap

from app.api.schemas import FeatureImportance


def generate_explanation(model, features: Dict[str, Any], prediction: float) -> List[FeatureImportance]:
    """Generate SHAP explanations for a prediction."""
    # Convert features to DataFrame
    features_df = pd.DataFrame([features])
    
    # Preprocess features
    preprocessor = model.current_model['preprocessor']
    X_processed = preprocessor.transform(features_df)
    
    # In this case we'll use the GradientBoostingRegressor
    base_model_name = 'gbm'
    base_model = None
    
    for name, model_obj in model.current_model['base_models']:
        if name == base_model_name:
            base_model = model_obj
            break
    
    if base_model is None:
        # Fallback to the first model
        _, base_model = model.current_model['base_models'][0]
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(base_model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_processed)
    
    # Get feature names
    feature_names = list(features.keys())
    
    # Create feature importance list
    importances = []
    abs_shap_values = np.abs(shap_values)[0]  # Get absolute values
    
    # Sort by importance
    indices = np.argsort(abs_shap_values)[::-1]
    
    # Take top 10 features or less if fewer features exist
    for i in range(min(10, len(indices))):
        idx = indices[i]
        if idx < len(feature_names):
            name = feature_names[idx]
            importance = float(abs_shap_values[idx])
            importances.append(FeatureImportance(feature=name, importance=importance))
    
    return importances