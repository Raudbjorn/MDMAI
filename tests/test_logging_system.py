"""
Comprehensive tests for logging system.
"""

import asyncio
import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.logging_system import (
    LogAnalyzer,
    LogCategory,
    LogConfig,
    LogLevel,
    LogRotationManager,
    LoggerFactory,
    PerformanceLogHandler,
    StructuredFormatter,
    get_logger,
    log_context,
)


class TestLogConfig:
    """Test LogConfig class."""

    def test_log_config_initialization(self):
        """Test LogConfig initialization with custom values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(
                log_dir=tmpdir,
                log_level="DEBUG",
                enable_console=False,
                enable_file=True,
                enable_json=True,
                max_bytes=5000000,
                backup_count=5,
            )
            
            assert config.log_dir == Path(tmpdir)
            assert config.log_level == logging.DEBUG
            assert config.enable_console is False
            assert config.enable_file is True
            assert config.max_bytes == 5000000
            assert config.backup_count == 5

    def test_log_config_creates_directory(self):
        """Test that LogConfig creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"
            config = LogConfig(log_dir=str(log_path))
            
            assert log_path.exists()
            assert log_path.is_dir()

    def test_get_log_file_path(self):
        """Test getting log file path for component."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(log_dir=tmpdir)
            log_file = config.get_log_file("test_component")
            
            assert log_file == Path(tmpdir) / "test_component.log"


class TestStructuredFormatter:
    """Test StructuredFormatter class."""

    def test_structured_formatter_basic(self):
        """Test basic structured formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert data["module"] == "test"
        assert data["line"] == 10

    def test_structured_formatter_with_exception(self):
        """Test structured formatting with exception info."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert "exception" in data
        assert "ValueError: Test error" in data["exception"]

    def test_structured_formatter_with_extra_fields(self):
        """Test structured formatting with extra fields."""
        formatter = StructuredFormatter(include_extra=True)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.user_id = "user123"
        record.request_id = "req456"
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert data["extra"]["user_id"] == "user123"
        assert data["extra"]["request_id"] == "req456"


class TestPerformanceLogHandler:
    """Test PerformanceLogHandler class."""

    def test_performance_handler_initialization(self):
        """Test PerformanceLogHandler initialization."""
        with tempfile.NamedTemporaryFile(suffix=".log") as tmpfile:
            handler = PerformanceLogHandler(Path(tmpfile.name))
            
            assert handler.metrics_file == Path(tmpfile.name)
            assert len(handler.metrics) == 0

    def test_performance_handler_emit(self):
        """Test emitting performance metrics."""
        handler = PerformanceLogHandler()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Operation completed",
            args=(),
            exc_info=None,
        )
        record.performance_data = {
            "duration_ms": 125.5,
            "operation": "database_query",
        }
        
        handler.emit(record)
        
        assert len(handler.metrics) == 1
        assert handler.metrics[0]["operation"] == "Operation completed"
        assert handler.metrics[0]["duration_ms"] == 125.5

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting collected metrics."""
        handler = PerformanceLogHandler()
        
        # Add some metrics
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Operation {i}",
                args=(),
                exc_info=None,
            )
            record.performance_data = {"duration_ms": i * 100}
            handler.emit(record)
        
        metrics = await handler.get_metrics()
        assert len(metrics) == 3
        assert metrics[1]["duration_ms"] == 100


class TestLoggerFactory:
    """Test LoggerFactory singleton and logger creation."""

    def test_logger_factory_singleton(self):
        """Test LoggerFactory is a singleton."""
        factory1 = LoggerFactory()
        factory2 = LoggerFactory()
        
        assert factory1 is factory2

    def test_get_logger_basic(self):
        """Test getting a basic logger."""
        factory = LoggerFactory()
        logger = factory.get_logger("test.module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_with_category(self):
        """Test getting logger with category."""
        with tempfile.TemporaryDirectory() as tmpdir:
            factory = LoggerFactory()
            factory.config.log_dir = Path(tmpdir)
            
            logger = factory.get_logger(
                "test.module",
                category=LogCategory.DATABASE,
            )
            
            assert isinstance(logger, logging.Logger)
            # Check that category-specific handler was added
            assert any(
                isinstance(h, logging.handlers.RotatingFileHandler)
                for h in logger.handlers
            )

    def test_get_logger_caching(self):
        """Test logger caching."""
        factory = LoggerFactory()
        logger1 = factory.get_logger("test.cached")
        logger2 = factory.get_logger("test.cached")
        
        assert logger1 is logger2


class TestLogContext:
    """Test log_context context manager."""

    def test_log_context_success(self):
        """Test log_context with successful operation."""
        logger = logging.getLogger("test")
        
        with patch.object(logger, "info") as mock_info:
            with log_context(logger, "test_operation", user_id="123"):
                # Simulate some work
                pass
        
        # Check start and completion logs
        assert mock_info.call_count == 2
        start_call = mock_info.call_args_list[0]
        end_call = mock_info.call_args_list[1]
        
        assert "Starting test_operation" in start_call[0][0]
        assert "Completed test_operation" in end_call[0][0]

    def test_log_context_with_exception(self):
        """Test log_context with exception."""
        logger = logging.getLogger("test")
        
        with patch.object(logger, "info") as mock_info:
            with patch.object(logger, "error") as mock_error:
                with pytest.raises(ValueError):
                    with log_context(logger, "failing_operation"):
                        raise ValueError("Test error")
        
        # Check that error was logged
        assert mock_error.call_count == 1
        error_call = mock_error.call_args_list[0]
        assert "Failed failing_operation" in error_call[0][0]


class TestLogRotationManager:
    """Test LogRotationManager class."""

    @pytest.mark.asyncio
    async def test_rotation_manager_initialization(self):
        """Test LogRotationManager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = LogRotationManager(
                log_dir=Path(tmpdir),
                max_age_days=7,
                max_total_size_gb=1.0,
                check_interval=60.0,
            )
            
            assert manager.log_dir == Path(tmpdir)
            assert manager.max_age_days == 7
            assert manager.max_total_size_bytes == 1024 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_cleanup_old_logs(self):
        """Test cleanup of old log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            manager = LogRotationManager(log_dir=log_dir, max_age_days=7)
            
            # Create old log files
            old_file = log_dir / "test.log.1"
            old_file.touch()
            
            # Mock file modification time to be 10 days ago
            import os
            old_time = time.time() - (10 * 24 * 3600)
            os.utime(old_file, (old_time, old_time))
            
            # Create recent log file
            recent_file = log_dir / "test.log"
            recent_file.touch()
            
            await manager._cleanup_old_logs()
            
            assert not old_file.exists()
            assert recent_file.exists()

    @pytest.mark.asyncio
    async def test_enforce_size_limit(self):
        """Test enforcement of total size limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            manager = LogRotationManager(
                log_dir=log_dir,
                max_total_size_gb=0.000001,  # 1KB limit
            )
            
            # Create multiple log files exceeding limit
            for i in range(3):
                file = log_dir / f"test.log.{i}"
                file.write_text("x" * 500)  # 500 bytes each
            
            await manager._enforce_size_limit()
            
            # Should have removed oldest files to stay under limit
            remaining_files = list(log_dir.glob("*.log.*"))
            total_size = sum(f.stat().st_size for f in remaining_files)
            assert total_size <= 1024  # Under 1KB limit


class TestLogAnalyzer:
    """Test LogAnalyzer class."""

    @pytest.mark.asyncio
    async def test_analyze_errors(self):
        """Test error analysis from logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "test.log"
            
            # Write test log entries
            log_entries = [
                {"timestamp": datetime.utcnow().isoformat(), "level": "ERROR", 
                 "message": "Database error", "module": "db"},
                {"timestamp": datetime.utcnow().isoformat(), "level": "ERROR",
                 "message": "Database error", "module": "db"},
                {"timestamp": datetime.utcnow().isoformat(), "level": "WARNING",
                 "message": "Slow query", "module": "db"},
                {"timestamp": datetime.utcnow().isoformat(), "level": "ERROR",
                 "message": "Network timeout", "module": "network"},
            ]
            
            with open(log_file, "w") as f:
                for entry in log_entries:
                    f.write(json.dumps(entry) + "\n")
            
            analyzer = LogAnalyzer(log_dir)
            analysis = await analyzer.analyze_errors()
            
            assert analysis["total_errors"] == 3
            assert analysis["unique_errors"] == 2
            assert "Database error" in dict(analysis["top_errors"])

    @pytest.mark.asyncio
    async def test_analyze_performance(self):
        """Test performance analysis from logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            perf_log = log_dir / "performance.log"
            
            # Write performance metrics
            metrics = [
                {"operation": "db_query", "duration_ms": 150},
                {"operation": "db_query", "duration_ms": 200},
                {"operation": "db_query", "duration_ms": 100},
                {"operation": "api_call", "duration_ms": 500},
            ]
            
            with open(perf_log, "w") as f:
                for metric in metrics:
                    f.write(json.dumps(metric) + "\n")
            
            analyzer = LogAnalyzer(log_dir)
            analysis = await analyzer.analyze_performance("db_query")
            
            assert analysis["operation"] == "db_query"
            assert analysis["count"] == 3
            assert analysis["avg_duration_ms"] == 150
            assert analysis["min_duration_ms"] == 100
            assert analysis["max_duration_ms"] == 200


class TestIntegration:
    """Integration tests for logging system."""

    def test_complete_logging_flow(self):
        """Test complete logging flow with multiple components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup custom config
            factory = LoggerFactory()
            factory.config = LogConfig(
                log_dir=tmpdir,
                log_level="DEBUG",
                enable_json=True,
            )
            
            # Get loggers for different components
            db_logger = factory.get_logger("database", LogCategory.DATABASE)
            api_logger = factory.get_logger("api", LogCategory.NETWORK)
            
            # Log various messages
            db_logger.info("Database connected")
            db_logger.error("Query failed", extra={"query": "SELECT * FROM users"})
            api_logger.warning("API rate limit approaching")
            
            # Verify logs were written
            log_files = list(Path(tmpdir).glob("*.log"))
            assert len(log_files) > 0

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        handler = PerformanceLogHandler()
        logger = logging.getLogger("perf_test")
        logger.addHandler(handler)
        
        # Log performance metrics
        for i in range(5):
            record = logging.LogRecord(
                name="perf_test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Operation {i}",
                args=(),
                exc_info=None,
            )
            record.performance_data = {
                "duration_ms": (i + 1) * 100,
                "operation_type": "database",
            }
            logger.handle(record)
        
        # Get statistics
        metrics = await handler.get_metrics()
        assert len(metrics) == 5
        
        # Calculate average
        avg_duration = sum(m["duration_ms"] for m in metrics) / len(metrics)
        assert avg_duration == 300  # (100 + 200 + 300 + 400 + 500) / 5

    def test_get_logger_convenience_function(self):
        """Test get_logger convenience function."""
        logger = get_logger("test.convenience", LogCategory.SYSTEM)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.convenience"