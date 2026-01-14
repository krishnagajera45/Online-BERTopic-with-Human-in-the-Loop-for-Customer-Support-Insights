"""
FastAPI Main Application for TwCS Topic Modeling.

This module provides the REST API for:
- Querying topics and trends
- Getting drift alerts
- Performing inference
- HITL topic editing
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import topics, trends, alerts, inference, hitl, pipeline
from src.utils import load_config, setup_logger

logger = setup_logger(__name__, "logs/api.log")

# Load configuration
config = load_config()

# Create FastAPI app
app = FastAPI(
    title="TwCS Topic Modeling API",
    description="REST API for Twitter Customer Support topic modeling system",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(topics.router, prefix="/api/v1/topics", tags=["Topics"])
app.include_router(trends.router, prefix="/api/v1/trends", tags=["Trends"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Alerts"])
app.include_router(inference.router, prefix="/api/v1/infer", tags=["Inference"])
app.include_router(hitl.router, prefix="/api/v1/hitl", tags=["HITL"])
app.include_router(pipeline.router, prefix="/api/v1/pipeline/status", tags=["Pipeline"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "TwCS Topic Modeling API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting API server on {config.api.host}:{config.api.port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
        log_level="info"
    )

