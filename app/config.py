import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# API settings
API_PREFIX = "/api/v1"
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# Security settings
API_KEY = os.getenv("API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "development_secret_key")

# Model settings
MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))
DEFAULT_MODEL_VERSION = os.getenv("DEFAULT_MODEL_VERSION", "latest")

# Database settings (if you add one later)
DATABASE_URL = os.getenv("DATABASE_URL", "")