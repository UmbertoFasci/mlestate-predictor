import joblib
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from lightgbm import LGBMRegressor
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from app.models.preprocessors import (
    TextFeatureExtractor,
    ZipCodeFeatureExtractor,
    FeatureEngineeringTransformer
)


class RealEstateEnsembleModel:
    """Advanced ensemble model for real estate price prediction."""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or Path(__file__).parent / "trained_models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.current_model = self._load_latest_model()
        
    def _load_latest_model(self):
        """Load the latest trained model from disk."""
        model_files = list(self.model_dir.glob("ensemble_model_*.joblib"))
        if not model_files:
            # Return untrained model if no saved models exist
            return self._create_model()
        
        # Get the most recent model
        latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        return joblib.load(latest_model_path)
    
    def _create_model(self):
        """Create a new ensemble model."""
        # Define preprocessing for numerical features
        numeric_features = ['square_feet', 'bedrooms', 'bathrooms', 'year_built', 'lot_size']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('feature_eng', FeatureEngineeringTransformer())
        ])
        
        # Define preprocessing for categorical features
        categorical_features = ['property_type', 'has_garage', 'has_pool']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Define preprocessing for text features
        text_features = ['description']
        text_transformer = Pipeline(steps=[
            ('extractor', TextFeatureExtractor())
        ])
        
        # Define preprocessing for location features
        location_features = ['location_zip']
        location_transformer = Pipeline(steps=[
            ('extractor', ZipCodeFeatureExtractor())
        ])
        
        # Combine all preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('txt', text_transformer, text_features),
                ('loc', location_transformer, location_features)
            ])
        
        # Create base models
        base_models = [
            ('gbm', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
            ('lgbm', LGBMRegressor(n_estimators=100, random_state=42)),
            ('lasso', LassoCV(cv=5))
        ]
        
        # Create meta-learner
        meta_learner = xgb.XGBRegressor(n_estimators=50, random_state=42)
        
        # Return untrained model
        return {
            'preprocessor': preprocessor,
            'base_models': base_models,
            'meta_learner': meta_learner,
            'is_trained': False,
            'version': f"untrained_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'feature_importances': None
        }
    
    def train(self, X: pd.DataFrame, y: np.ndarray):
        """Train the ensemble model."""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit preprocessor
        preprocessor = self.current_model['preprocessor']
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        # Train base models
        base_predictions_train = np.zeros((X_train.shape[0], len(self.current_model['base_models'])))
        base_predictions_val = np.zeros((X_val.shape[0], len(self.current_model['base_models'])))
        
        for i, (name, model) in enumerate(self.current_model['base_models']):
            model.fit(X_train_processed, y_train)
            base_predictions_train[:, i] = model.predict(X_train_processed)
            base_predictions_val[:, i] = model.predict(X_val_processed)
        
        # Train meta-learner
        meta_learner = self.current_model['meta_learner']
        meta_learner.fit(base_predictions_train, y_train)
        
        # Calculate feature importances from base models
        feature_importances = self._calculate_feature_importances()
        
        # Update model
        self.current_model.update({
            'is_trained': True,
            'version': f"ensemble_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'feature_importances': feature_importances
        })
        
        # Save model
        self._save_model()
        
        return {
            'train_score': meta_learner.score(base_predictions_train, y_train),
            'val_score': meta_learner.score(base_predictions_val, y_val),
            'version': self.current_model['version']
        }
    
    def _calculate_feature_importances(self):
        """Calculate feature importances from base models."""
        # TODO
        pass
    
    def _save_model(self):
        """Save the current model to disk."""
        model_path = self.model_dir / f"{self.current_model['version']}.joblib"
        joblib.dump(self.current_model, model_path)
    
    def predict(self, features: Dict[str, Any], version: Optional[str] = None) -> Dict:
        """Generate price prediction with uncertainty estimates."""
        # Load specific version if requested
        if version and version != self.current_model['version']:
            model_path = self.model_dir / f"{version}.joblib"
            if not model_path.exists():
                raise ValueError(f"Model version {version} not found")
            model = joblib.load(model_path)
        else:
            model = self.current_model
        
        if not model['is_trained']:
            raise ValueError("Model is not trained yet")
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Preprocess features
        X_processed = model['preprocessor'].transform(features_df)
        
        # Get base model predictions
        base_predictions = np.zeros((1, len(model['base_models'])))
        base_model_outputs = []
        
        for i, (name, base_model) in enumerate(model['base_models']):
            pred = base_model.predict(X_processed)[0]
            base_predictions[0, i] = pred
            base_model_outputs.append(pred)
        
        # Get meta-learner prediction
        final_prediction = model['meta_learner'].predict(base_predictions)[0]
        
        # Calculate prediction interval (simplified approach)
        std_dev = np.std(base_model_outputs)
        lower_bound = final_prediction - 1.96 * std_dev
        upper_bound = final_prediction + 1.96 * std_dev
        
        # Calculate confidence score (simplified approach)
        # Higher agreement among base models = higher confidence
        normalized_std = std_dev / final_prediction if final_prediction > 0 else std_dev
        confidence_score = max(0, min(1, 1 - normalized_std))
        
        return {
            "price": float(final_prediction),
            "interval": [float(lower_bound), float(upper_bound)],
            "confidence": float(confidence_score),
            "model_version": model['version']
        }
    
    def log_prediction(self, property_id: str, features: Dict, prediction: Dict):
        """Log prediction for monitoring."""
        # TODO
        pass