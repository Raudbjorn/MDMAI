"""
Structured logging system for MDMAI TTRPG Assistant.

This module provides comprehensive logging with rotation, structured formats,
performance metrics, and component-specific loggers.
"""

import asyncio
import json
import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog


class LogLevel(Enum):
    """Extended log levels for fine-grained control."""

    TRACE = 5  # More detailed than DEBUG
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    AUDIT = 35  # Between WARNING and ERROR
    PERFORMANCE = 25  # Between INFO and WARNING


class LogCategory(Enum):
    """Log categories for component-specific logging."""

    SYSTEM = "system"
    SERVICE = "service"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    MCP = "mcp"
    VECTOR_DB = "vector_db"
    CHARACTER = "character"
    CAMPAIGN = "campaign"
    COMBAT = "combat"
    RULES = "rules"
    AUDIT = "audit"


class LogConfig:
    """Centralized logging configuration."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 10,
        rotation_interval: str = "midnight",
        enable_performance: bool = True,
        enable_audit: bool = True,
    ) -> None:
        """
        Initialize logging configuration.

        Args:
            log_dir: Directory for log files (defaults to ~/.mdmai/logs or ./logs)
            log_level: Default log level
            enable_console: Enable console output
            enable_file: Enable file output
            enable_json: Enable JSON formatting
            max_bytes: Maximum size for log files
            backup_count: Number of backup files to keep
            rotation_interval: Rotation interval for time-based rotation
            enable_performance: Enable performance logging
            enable_audit: Enable audit logging
        """
        if log_dir is None:
            # Try user home directory first, fall back to local directory
            home_log_dir = Path.home() / ".mdmai" / "logs"
            local_log_dir = Path("./logs")
            
            # Use home directory if writable, otherwise use local directory
            try:
                home_log_dir.mkdir(parents=True, exist_ok=True)
                self.log_dir = home_log_dir
            except (OSError, PermissionError):
                self.log_dir = local_log_dir
        else:
            self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.rotation_interval = rotation_interval
        self.enable_performance = enable_performance
        self.enable_audit = enable_audit
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_log_file(self, component: str) -> Path:
        """Get log file path for component."""
        return self.log_dir / f"{component}.log"


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def __init__(self, include_extra: bool = True) -> None:
        """
        Initialize structured formatter.

        Args:
            include_extra: Include extra fields in log records
        """
        super().__init__()
        self.include_extra = include_extra
        self.default_keys = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "message", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName",
            "exc_info", "exc_text", "stack_info"
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data."""
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in self.default_keys
            }
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data, default=str)


class PerformanceLogHandler(logging.Handler):
    """Handler for performance metrics logging."""

    def __init__(self, metrics_file: Optional[Path] = None) -> None:
        """
        Initialize performance log handler.

        Args:
            metrics_file: Optional file path for metrics output
        """
        super().__init__()
        self.metrics_file = metrics_file
        self.metrics: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit performance log record."""
        if hasattr(record, "performance_data"):
            metric = {
                "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
                "operation": record.getMessage(),
                **record.performance_data,  # type: ignore
            }
            self.metrics.append(metric)
            
            if self.metrics_file:
                with open(self.metrics_file, "a") as f:
                    f.write(json.dumps(metric) + "\n")

    async def get_metrics(self) -> List[Dict[str, Any]]:
        """Get collected performance metrics."""
        async with self._lock:
            return list(self.metrics)

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self.metrics.clear()


class LoggerFactory:
    """Factory for creating component-specific loggers."""

    _instance: Optional["LoggerFactory"] = None
    _initialized: bool = False

    def __new__(cls) -> "LoggerFactory":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize logger factory."""
        if not self._initialized:
            self.config = LogConfig()
            self.loggers: Dict[str, logging.Logger] = {}
            self.performance_handler = PerformanceLogHandler()
            self._setup_root_logger()
            self._initialized = True

    def _setup_root_logger(self) -> None:
        """Setup root logger configuration."""
        root = logging.getLogger()
        root.setLevel(self.config.log_level)
        
        # Remove existing handlers
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if self.config.enable_json:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )
            root.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.enable_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.get_log_file("app"),
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count,
            )
            if self.config.enable_json:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )
            root.addHandler(file_handler)

    def get_logger(
        self,
        name: str,
        category: Optional[LogCategory] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> logging.Logger:
        """
        Get or create logger for component.

        Args:
            name: Logger name
            category: Log category
            extra_fields: Extra fields to include in all logs

        Returns:
            Configured logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            
            # Add category-specific handler if needed
            if category:
                handler = logging.handlers.RotatingFileHandler(
                    self.config.get_log_file(category.value),
                    maxBytes=self.config.max_bytes,
                    backupCount=self.config.backup_count,
                )
                handler.setFormatter(StructuredFormatter())
                logger.addHandler(handler)
            
            # Add performance handler if enabled
            if self.config.enable_performance:
                logger.addHandler(self.performance_handler)
            
            self.loggers[name] = logger
        
        logger = self.loggers[name]
        
        # Add extra fields to logger
        if extra_fields:
            for key, value in extra_fields.items():
                setattr(logger, key, value)
        
        return logger


class StructLogConfig:
    """Configuration for structlog integration."""

    @staticmethod
    def configure() -> None:
        """Configure structlog with custom processors."""
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
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    ]
                ),
                structlog.processors.EventRenamer("message"),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


@contextmanager
def log_context(
    logger: logging.Logger,
    operation: str,
    **context_fields: Any,
) -> Any:
    """
    Context manager for structured logging with automatic timing.

    Args:
        logger: Logger instance
        operation: Operation name
        **context_fields: Additional context fields

    Yields:
        Context with timing
    """
    start_time = time.perf_counter()
    logger.info(f"Starting {operation}", extra={"operation": operation, **context_fields})
    
    try:
        yield
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error(
            f"Failed {operation}: {e}",
            extra={
                "operation": operation,
                "duration_ms": duration * 1000,
                "error": str(e),
                **context_fields,
            },
            exc_info=True,
        )
        raise
    else:
        duration = time.perf_counter() - start_time
        logger.info(
            f"Completed {operation}",
            extra={
                "operation": operation,
                "duration_ms": duration * 1000,
                **context_fields,
            },
        )


class LogRotationManager:
    """Manages log rotation and cleanup."""

    def __init__(
        self,
        log_dir: Path,
        max_age_days: int = 30,
        max_total_size_gb: float = 10.0,
        check_interval: float = 3600.0,
    ) -> None:
        """
        Initialize log rotation manager.

        Args:
            log_dir: Directory containing log files
            max_age_days: Maximum age of log files in days
            max_total_size_gb: Maximum total size of logs in GB
            check_interval: Interval between cleanup checks in seconds
        """
        self.log_dir = log_dir
        self.max_age_days = max_age_days
        self.max_total_size_bytes = int(max_total_size_gb * 1024 * 1024 * 1024)
        self.check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start rotation manager."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._rotation_loop())

    async def stop(self) -> None:
        """Stop rotation manager."""
        self._running = False
        if self._task:
            await self._task

    async def _rotation_loop(self) -> None:
        """Main rotation loop."""
        while self._running:
            try:
                await self._cleanup_old_logs()
                await self._enforce_size_limit()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Error in rotation loop: {e}")
                await asyncio.sleep(60)  # Short delay on error

    async def _cleanup_old_logs(self) -> None:
        """Remove logs older than max_age_days."""
        cutoff_time = time.time() - (self.max_age_days * 24 * 3600)
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    logging.info(f"Removed old log file: {log_file}")
                except Exception as e:
                    logging.error(f"Failed to remove log file {log_file}: {e}")

    async def _enforce_size_limit(self) -> None:
        """Enforce maximum total size limit."""
        log_files = sorted(
            self.log_dir.glob("*.log*"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        
        total_size = sum(f.stat().st_size for f in log_files)
        
        while total_size > self.max_total_size_bytes and log_files:
            oldest_file = log_files.pop()
            file_size = oldest_file.stat().st_size
            
            try:
                oldest_file.unlink()
                total_size -= file_size
                logging.info(f"Removed log file for size limit: {oldest_file}")
            except Exception as e:
                logging.error(f"Failed to remove log file {oldest_file}: {e}")


class LogAnalyzer:
    """Analyze logs for patterns and issues."""

    def __init__(self, log_dir: Path) -> None:
        """
        Initialize log analyzer.

        Args:
            log_dir: Directory containing log files
        """
        self.log_dir = log_dir

    async def analyze_errors(
        self,
        time_range: Optional[tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze error patterns in logs.

        Args:
            time_range: Optional time range for analysis

        Returns:
            Analysis results with error patterns
        """
        errors: List[Dict[str, Any]] = []
        
        for log_file in self.log_dir.glob("*.log"):
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        if log_entry.get("level") in ["ERROR", "CRITICAL"]:
                            if time_range:
                                timestamp = datetime.fromisoformat(log_entry["timestamp"])
                                if not (time_range[0] <= timestamp <= time_range[1]):
                                    continue
                            errors.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        
        # Analyze patterns
        error_counts: Dict[str, int] = {}
        error_modules: Dict[str, int] = {}
        
        for error in errors:
            message = error.get("message", "")
            module = error.get("module", "unknown")
            
            # Count error messages
            error_counts[message] = error_counts.get(message, 0) + 1
            error_modules[module] = error_modules.get(module, 0) + 1
        
        return {
            "total_errors": len(errors),
            "unique_errors": len(error_counts),
            "top_errors": sorted(
                error_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "errors_by_module": error_modules,
            "time_range": time_range,
        }

    async def analyze_performance(
        self,
        operation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze performance metrics from logs.

        Args:
            operation: Optional operation to filter by

        Returns:
            Performance analysis results
        """
        metrics: List[Dict[str, Any]] = []
        
        # Read performance log
        perf_log = self.log_dir / "performance.log"
        if perf_log.exists():
            with open(perf_log, "r") as f:
                for line in f:
                    try:
                        metric = json.loads(line)
                        if operation and metric.get("operation") != operation:
                            continue
                        metrics.append(metric)
                    except json.JSONDecodeError:
                        continue
        
        if not metrics:
            return {"message": "No performance metrics found"}
        
        # Calculate statistics
        durations = [m.get("duration_ms", 0) for m in metrics if "duration_ms" in m]
        
        if durations:
            return {
                "operation": operation,
                "count": len(durations),
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "p50_duration_ms": sorted(durations)[len(durations) // 2],
                "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)],
                "p99_duration_ms": sorted(durations)[int(len(durations) * 0.99)],
            }
        
        return {"message": "No duration metrics found"}


# Global logger factory instance
logger_factory = LoggerFactory()

# Convenience function for getting loggers
def get_logger(
    name: str,
    category: Optional[LogCategory] = None,
    **extra_fields: Any,
) -> logging.Logger:
    """
    Get logger instance.

    Args:
        name: Logger name
        category: Log category
        **extra_fields: Extra fields for structured logging

    Returns:
        Configured logger
    """
    return logger_factory.get_logger(name, category, extra_fields)