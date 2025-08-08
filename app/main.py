import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings
from app.core.logger import logger

# Create FastAPI app
app = FastAPI(
    title="Hinglish Voice Bot",
    description="Voice bot for blue-collar candidate screening in Hinglish",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the main web interface."""
    return FileResponse("static/index.html")

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Hinglish Voice Bot starting up...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Host: {settings.host}:{settings.port}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Hinglish Voice Bot shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development"
    ) 