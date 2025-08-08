import sys
from loguru import logger
from app.core.config import settings


def setup_logging():
    """Configure structured logging for the application."""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # Add file handler for persistent logs
    logger.add(
        "logs/voice_bot.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        serialize=True  # JSON format for structured logging
    )
    
    # Add separate file for voice conversation logs
    logger.add(
        "logs/conversations.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="50 MB",
        retention="90 days",
        filter=lambda record: "conversation" in record["extra"],
        serialize=True
    )
    
    # Add performance metrics log
    logger.add(
        "logs/metrics.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        level="INFO",
        rotation="50 MB", 
        retention="30 days",
        filter=lambda record: "metrics" in record["extra"],
        serialize=True
    )
    
    return logger


# Initialize logger
setup_logging()

# Export configured logger
__all__ = ["logger"] 