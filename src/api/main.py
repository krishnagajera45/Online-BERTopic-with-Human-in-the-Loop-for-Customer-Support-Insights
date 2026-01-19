"""
FastAPI Main Application for TwCS Topic Modeling.

This module provides the REST API for:
- Querying topics and trends
- Getting drift alerts
- Performing inference
- HITL topic editing
"""
import time
import logging
from fastapi import FastAPI, Request
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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all API requests and responses to unified debug log."""
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"REQUEST: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000
    
    # Log response
    logger.info(f"RESPONSE: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration_ms:.2f}ms")
    
    return response


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
    logger.info("Health check endpoint called")
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    from src.utils.logging_config import get_unified_debug_log_path
    
    logger.info(f"Starting API server on {config.api.host}:{config.api.port}")
    
    # Configure uvicorn to use our unified logging
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "unified_file": {
                "class": "logging.FileHandler",
                "filename": get_unified_debug_log_path(),
                "formatter": "default",
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["unified_file", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["unified_file", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["unified_file", "console"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
        log_config=log_config,
    )

