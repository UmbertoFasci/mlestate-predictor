from fastapi import FastAPI, HTTPException
from app.api.endpoints import router as api_router

app = FastAPI(
    title="MLEstate API",
    description="Advanced machine learning API for real estate price prediction",
    version="1.0.0"
)

app.include_router(api_rotuer, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}