"""Logging configuration for TTRPG Assistant MCP Server."""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from rich.console import Console
from rich.logging import RichHandler

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Check if running in MCP stdio mode - stdout is reserved for JSON-RPC
MCP_STDIO_MODE = os.environ.get("MCP_STDIO_MODE", "").lower() in ("true", "1", "yes")

# Configure structlog IMMEDIATELY at import time to use stderr in stdio mode
# This ensures any early logging doesn't corrupt stdout JSON-RPC communication
if MCP_STDIO_MODE:
    # Minimal stderr-only configuration for early imports
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,  # Allow reconfiguration later
    )

def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """
    Configure structured logging with rich console output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # In MCP stdio mode, all logging must go to stderr to keep stdout clean for JSON-RPC
    console_stream = sys.stderr if MCP_STDIO_MODE else None

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() and not MCP_STDIO_MODE else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    # In MCP stdio mode, use a simple stderr handler instead of RichHandler
    if MCP_STDIO_MODE:
        logging_config: Dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": level,
                    "formatter": "default",
                    "stream": "ext://sys.stderr",
                },
            },
            "root": {
                "level": level,
                "handlers": ["console"],
            },
            "loggers": {
                "ttrpg": {
                    "level": level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "chromadb": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "sentence_transformers": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    else:
        logging_config: Dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "rich.logging.RichHandler",
                    "level": level,
                    "formatter": "default",
                    "rich_tracebacks": True,
                    "markup": True,
                },
            },
            "root": {
                "level": level,
                "handlers": ["console"],
            },
            "loggers": {
                "ttrpg": {
                    "level": level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "chromadb": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "sentence_transformers": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    
    # Add file handler if log file is specified
    if log_file:
        log_path = LOG_DIR / log_file
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "detailed",
            "filename": str(log_path),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        }
        logging_config["root"]["handlers"].append("file")
        logging_config["loggers"]["ttrpg"]["handlers"].append("file")
    
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)