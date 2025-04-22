import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Union


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from property descriptions."""
    
    def __init__(self, max_features=100):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def fit(self, X, y=None):
        # Handle missing descriptions
        descriptions = X.fillna('').astype(str)
        self.vectorizer.fit(descriptions)
        return self
    
    def transform(self, X):
        # Handle missing descriptions
        descriptions = X.fillna('').astype(str)
        return self.vectorizer.transform(descriptions)


class ZipCodeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from zip codes."""
    
    def __init__(self):
        self.zip_stats = {}
        
    def fit(self, X, y=None):
        unique_zips = X.unique()
        
        # Mock data
        for zip_code in unique_zips:
            self.zip_stats[zip_code] = {
                'median_income': np.random.uniform(30000, 200000),
                'population_density': np.random.uniform(100, 30000),
                'crime_rate': np.random.uniform(0, 100),
                'school_rating': np.random.uniform(1, 10)
            }
        
        return self
    
    def transform(self, X):
        # Create feature array from zip codes
        result = np.zeros((len(X), 4))
        
        for i, zip_code in enumerate(X):
            if zip_code in self.zip_stats:
                stats = self.zip_stats[zip_code]
                result[i, 0] = stats['median_income']
                result[i, 1] = stats['population_density']
                result[i, 2] = stats['crime_rate']
                result[i, 3] = stats['school_rating']
            # If zip code not found, features remain 0
                
        return result


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Engineer additional features from existing numeric features."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X is assumed to be a numpy array with columns:
        # [square_feet, bedrooms, bathrooms, year_built, lot_size]
        
        result = np.zeros((X.shape[0], X.shape[1] + 3))
        
        # Copy original features
        result[:, :X.shape[1]] = X
        
        # Add engineered features
        
        # Price per square foot (using current year)
        current_year = 2025
        house_age = current_year - X[:, 3]  # year_built is at idx 3
        result[:, X.shape[1]] = house_age
        
        # Square feet per bedroom
        sq_ft = X[:, 0]  # square_feet is at idx 0
        bedrooms = X[:, 1]  # bedrooms is at idx 1
        # Avoid division by zero
        bedrooms_safe = np.maximum(bedrooms, 1)
        result[:, X.shape[1] + 1] = sq_ft / bedrooms_safe
        
        # Bathroom to bedroom ratio
        bathrooms = X[:, 2]  # bathrooms is at idx 2
        result[:, X.shape[1] + 2] = bathrooms / bedrooms_safe
        
        return result