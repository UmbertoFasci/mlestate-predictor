from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional

class ModelRegistry:
    """Simple model registry to track model versions and metadata."""
    
    def __init__(self, registry_dir: str = None):
        self.registry_dir = Path(registry_dir) if registry_dir else Path(__file__).parent.parent / "models" / "registry"
        self.registry_dir.mkdir(exist_ok=True, parents=True)
        self.registry_file = self.registry_dir / "model_registry.json"
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load the model registry from disk."""
        if not self.registry_file.exists():
            return {"models": {}}
        
        with open(self.registry_file, "r") as f:
            return json.load(f)
    
    def _save_registry(self):
        """Save the model registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.models, f, indent=2)
    
    def register_model(self, 
                       model_version: str, 
                       model_path: str, 
                       metrics: Dict[str, float], 
                       description: Optional[str] = None):
        """Register a new model version."""
        self.models["models"][model_version] = {
            "version": model_version,
            "path": str(model_path),
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "description": description or "",
            "is_production": False
        }
        self._save_registry()
    
    def set_production_model(self, model_version: str):
        """Set a model as the production model."""
        if model_version not in self.models["models"]:
            raise ValueError(f"Model version {model_version} not found in registry")
        
        # Unset current production model
        for version, info in self.models["models"].items():
            self.models["models"][version]["is_production"] = False
        
        # Set new production model
        self.models["models"][model_version]["is_production"] = True
        self._save_registry()
    
    def get_production_model(self) -> Optional[Dict]:
        """Get the current production model."""
        for version, info in self.models["models"].items():
            if info["is_production"]:
                return info
        return None
    
    def get_model_info(self, model_version: str) -> Dict:
        """Get information about a specific model version."""
        if model_version not in self.models["models"]:
            raise ValueError(f"Model version {model_version} not found in registry")
        return self.models["models"][model_version]
    
    def list_models(self) -> List[Dict]:
        """List all registered models."""
        return list(self.models["models"].values())