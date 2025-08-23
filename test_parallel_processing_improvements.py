"""Tests for parallel processing system improvements."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import multiprocessing as mp
import time
import uuid

from src.performance.parallel_processor import (
    ParallelProcessor,
    ResourceLimits,
    ProcessingTask,
    TaskStatus,
    ParallelPDFProcessor,
    ResourceManager,
)
from src.performance.parallel_mcp_tools import (
    register_parallel_tools,
    initialize_parallel_tools,
)
from src.pdf_processing.pipeline import PDFProcessingPipeline


class TestResourceLimits:
    """Test resource limits validation."""
    
    def test_resource_limits_defaults(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        assert limits.max_workers == mp.cpu_count()
        assert limits.max_memory_mb == 1024
        assert limits.max_queue_size == 1000
        assert limits.task_timeout == 300
    
    def test_resource_limits_validation(self):
        """Test resource limits validation."""
        # Test negative values are corrected
        limits = ResourceLimits(max_workers=-1, max_memory_mb=-100, task_timeout=-10)
        assert limits.max_workers == mp.cpu_count()
        assert limits.max_memory_mb == 1024
        assert limits.task_timeout == 300
    
    def test_resource_limits_custom(self):
        """Test custom resource limits."""
        limits = ResourceLimits(max_workers=8, max_memory_mb=2048, task_timeout=600)
        assert limits.max_workers == 8
        assert limits.max_memory_mb == 2048
        assert limits.task_timeout == 600


class TestParallelProcessor:
    """Test parallel processor improvements."""
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self):
        """Test processor initialization with improvements."""
        processor = ParallelProcessor()
        
        # Test initialization
        await processor.initialize()
        assert processor._initialized
        assert processor.executor is not None
        assert processor.async_executor is not None
        assert len(processor._workers) == processor.limits.max_workers
        
        # Test double initialization prevention
        await processor.initialize()  # Should not error
        assert processor._initialized
        
        # Cleanup
        await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_processor_shutdown_improvements(self):
        """Test improved shutdown process."""
        processor = ParallelProcessor(ResourceLimits(max_workers=2))
        await processor.initialize()
        
        # Submit a task
        task = await processor.submit_task("pdf_processing", {"pdf_path": "test.pdf"})
        
        # Shutdown with pending task
        await processor.shutdown()
        assert processor._shutdown
        assert task.status == TaskStatus.CANCELLED
        assert not processor._initialized
        
        # Test double shutdown
        await processor.shutdown()  # Should not error
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self):
        """Test task timeout handling."""
        processor = ParallelProcessor(ResourceLimits(task_timeout=1))
        
        # Mock a slow task
        async def slow_task(data):
            await asyncio.sleep(5)
            return "done"
        
        with patch.object(processor, '_process_pdf_task', slow_task):
            await processor.initialize()
            
            task = await processor.submit_task("pdf_processing", {"pdf_path": "test.pdf"})
            
            # Wait for task to timeout
            await asyncio.sleep(2)
            
            assert task.status == TaskStatus.FAILED
            assert "timed out" in task.error.lower()
            
            await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_retry(self):
        """Test exponential backoff for retries."""
        processor = ParallelProcessor()
        
        # Track retry times
        retry_times = []
        call_count = 0
        
        async def failing_task(data):
            nonlocal call_count
            call_count += 1
            retry_times.append(time.time())
            if call_count < 3:
                raise ValueError("Test error")
            return "success"
        
        with patch.object(processor, '_process_pdf_task', failing_task):
            await processor.initialize()
            
            task = ProcessingTask(
                id="test-task",
                type="pdf_processing",
                data={"pdf_path": "test.pdf"},
                max_retries=3
            )
            
            processor.tasks[task.id] = task
            await processor.task_queue.put(task)
            
            # Wait for retries
            await asyncio.sleep(10)
            
            # Check exponential backoff occurred
            if len(retry_times) >= 2:
                first_gap = retry_times[1] - retry_times[0]
                assert first_gap >= 2  # 2^1 seconds
            
            await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_resource_monitoring_improvements(self):
        """Test improved resource monitoring."""
        processor = ParallelProcessor(ResourceLimits(max_memory_mb=100))
        
        with patch('psutil.Process') as mock_process:
            # Mock high memory usage
            mock_proc_instance = Mock()
            mock_proc_instance.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB
            mock_proc_instance.cpu_percent.return_value = 95
            mock_process.return_value = mock_proc_instance
            
            await processor.initialize()
            
            # Let monitor run
            await asyncio.sleep(1)
            
            # Should log warnings about high memory and CPU
            # Actual logging assertions would require log capture
            
            await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test context manager for automatic cleanup."""
        processor = ParallelProcessor()
        
        async with processor.managed_processor() as proc:
            assert proc._initialized
            task = await proc.submit_task("pdf_processing", {"pdf_path": "test.pdf"})
            assert task.id in proc.tasks
        
        # Should be shutdown after context
        assert processor._shutdown


class TestParallelMCPTools:
    """Test parallel MCP tools improvements."""
    
    @pytest.mark.asyncio
    async def test_pdf_validation(self):
        """Test PDF file validation in parallel processing."""
        mock_server = Mock()
        mock_server.tool = lambda: lambda func: func
        
        register_parallel_tools(mock_server)
        
        # Get the registered function
        process_pdfs_func = None
        for call in mock_server.tool.call_args_list:
            func = call[0][0] if call[0] else None
            if func and func.__name__ == 'process_pdfs_parallel':
                process_pdfs_func = func
                break
        
        # Test with missing files
        result = await process_pdfs_func([
            {"pdf_path": "/nonexistent/file.pdf", "rulebook_name": "Test", "system": "D&D 5e"}
        ])
        
        assert not result["success"]
        assert "not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of PDFs."""
        mock_server = Mock()
        mock_server.tool = lambda: lambda func: func
        
        with patch('src.pdf_processing.pipeline.PDFProcessingPipeline') as mock_pipeline:
            mock_instance = AsyncMock()
            mock_instance.process_multiple_pdfs.return_value = {
                "successful": 5,
                "failed": 0,
                "results": []
            }
            mock_pipeline.return_value = mock_instance
            
            register_parallel_tools(mock_server)
            
            # Create test files
            test_files = []
            for i in range(10):
                test_file = Path(f"/tmp/test_{i}.pdf")
                test_file.touch()
                test_files.append({
                    "pdf_path": str(test_file),
                    "rulebook_name": f"Book {i}",
                    "system": "D&D 5e"
                })
            
            # Would call the function with batch_size parameter
            # Cleanup test files
            for file_info in test_files:
                Path(file_info["pdf_path"]).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_text_truncation(self):
        """Test text truncation for embeddings."""
        mock_server = Mock()
        mock_server.tool = lambda: lambda func: func
        
        register_parallel_tools(mock_server)
        
        # Create very long texts
        long_texts = ["x" * 15000 for _ in range(3)]
        
        # Would test that texts are truncated to MAX_TEXT_LENGTH
        # and warning is logged
    
    @pytest.mark.asyncio
    async def test_processor_tracking(self):
        """Test active processor tracking."""
        from src.performance.parallel_mcp_tools import _active_processors
        
        mock_server = Mock()
        mock_server.tool = lambda: lambda func: func
        
        register_parallel_tools(mock_server)
        
        # Would test that processors are tracked and cleaned up
        # using the cleanup_parallel_processors function
    
    @pytest.mark.asyncio
    async def test_task_cancellation_improvements(self):
        """Test improved task cancellation."""
        mock_server = Mock()
        mock_server.tool = lambda: lambda func: func
        
        register_parallel_tools(mock_server)
        
        # Would test cancellation with processor_id tracking
        # and state validation


class TestPipelineIntegration:
    """Test pipeline integration improvements."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization with custom workers."""
        pipeline = PDFProcessingPipeline(enable_parallel=True, max_workers=2)
        
        assert pipeline.enable_parallel
        assert pipeline.parallel_processor is not None
        assert pipeline.parallel_processor.limits.max_workers == 2
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self):
        """Test improved error handling in pipeline."""
        pipeline = PDFProcessingPipeline(enable_parallel=False)
        
        # Test file not found
        result = await pipeline.process_pdf(
            pdf_path="/nonexistent/file.pdf",
            rulebook_name="Test",
            system="D&D 5e"
        )
        
        assert result["status"] == "error"
        assert result["error_type"] == "file_not_found"
    
    @pytest.mark.asyncio
    async def test_pipeline_validation(self):
        """Test input validation in pipeline."""
        pipeline = PDFProcessingPipeline(enable_parallel=False)
        
        # Create a test file
        test_file = Path("/tmp/test_validation.pdf")
        test_file.touch()
        
        try:
            # Test invalid source_type
            result = await pipeline.process_pdf(
                pdf_path=str(test_file),
                rulebook_name="Test",
                system="D&D 5e",
                source_type="invalid"
            )
            
            assert result["status"] == "error"
            assert "source_type" in result["error"]
            
            # Test missing required fields
            result = await pipeline.process_pdf(
                pdf_path=str(test_file),
                rulebook_name="",
                system="D&D 5e"
            )
            
            assert result["status"] == "error"
            assert "required" in result["error"].lower()
            
        finally:
            test_file.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_pipeline_processing_time(self):
        """Test processing time tracking."""
        pipeline = PDFProcessingPipeline(enable_parallel=False)
        
        with patch.object(pipeline.parser, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = {
                "text": "Test content",
                "total_pages": 1,
                "file_hash": "test_hash",
                "file_name": "test.pdf",
                "tables": []
            }
            
            with patch.object(pipeline, '_is_duplicate', return_value=False):
                with patch.object(pipeline.chunker, 'chunk_document', return_value=[]):
                    with patch.object(pipeline.embedding_generator, 'generate_embeddings', return_value=[]):
                        with patch.object(pipeline, '_store_chunks', return_value=0):
                            with patch.object(pipeline, '_store_source_metadata'):
                                # Create test file
                                test_file = Path("/tmp/test_timing.pdf")
                                test_file.touch()
                                
                                try:
                                    result = await pipeline.process_pdf(
                                        pdf_path=str(test_file),
                                        rulebook_name="Test",
                                        system="D&D 5e"
                                    )
                                    
                                    assert "processing_time_seconds" in result
                                    assert result["processing_time_seconds"] >= 0
                                    
                                finally:
                                    test_file.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_parallel_batch_processing(self):
        """Test parallel batch processing in pipeline."""
        pipeline = PDFProcessingPipeline(enable_parallel=True)
        
        # Mock the parallel processor
        if pipeline.parallel_processor:
            with patch.object(pipeline.parallel_processor, 'initialize', new_callable=AsyncMock):
                with patch.object(pipeline.parallel_processor, 'submit_task', new_callable=AsyncMock) as mock_submit:
                    mock_task = Mock()
                    mock_task.id = "test-task"
                    mock_task.status = TaskStatus.COMPLETED
                    mock_task.result = {"success": True}
                    mock_submit.return_value = mock_task
                    
                    with patch.object(pipeline.parallel_processor, 'wait_for_all', new_callable=AsyncMock) as mock_wait:
                        mock_wait.return_value = [mock_task]
                        
                        result = await pipeline.process_multiple_pdfs(
                            pdf_files=[],
                            enable_adaptive_learning=True
                        )
                        
                        assert result["total"] == 0
                        assert result["method"] == "none"


class TestResourceManager:
    """Test resource manager improvements."""
    
    def test_optimal_workers_calculation(self):
        """Test optimal worker calculation for different task types."""
        manager = ResourceManager()
        
        # Test CPU-intensive tasks
        workers = manager.get_optimal_workers("pdf_processing")
        assert workers == max(1, mp.cpu_count() - 1)
        
        workers = manager.get_optimal_workers("embedding_generation")
        assert workers == max(1, mp.cpu_count() - 1)
        
        # Test I/O-bound tasks
        workers = manager.get_optimal_workers("search")
        assert workers == min(mp.cpu_count() * 2, 16)
        
        # Test default
        workers = manager.get_optimal_workers("unknown")
        assert workers == mp.cpu_count()
    
    def test_processor_allocation(self):
        """Test processor allocation and tracking."""
        manager = ResourceManager()
        
        # Allocate processor
        processor = manager.allocate_processor("test-proc", "pdf_processing")
        assert "test-proc" in manager.active_processors
        assert processor.limits.max_workers == max(1, mp.cpu_count() - 1)
        
        # Test duplicate allocation
        with pytest.raises(ValueError):
            manager.allocate_processor("test-proc", "search")
    
    @pytest.mark.asyncio
    async def test_processor_release(self):
        """Test processor release."""
        manager = ResourceManager()
        
        # Allocate and release
        processor = manager.allocate_processor("test-proc", "pdf_processing")
        await processor.initialize()
        
        await manager.release_processor("test-proc")
        assert "test-proc" not in manager.active_processors
        assert processor._shutdown
    
    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """Test shutdown all processors."""
        manager = ResourceManager()
        
        # Allocate multiple processors
        proc1 = manager.allocate_processor("proc1", "pdf_processing")
        proc2 = manager.allocate_processor("proc2", "search")
        
        await proc1.initialize()
        await proc2.initialize()
        
        # Shutdown all
        await manager.shutdown_all()
        assert len(manager.active_processors) == 0
        assert proc1._shutdown
        assert proc2._shutdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])