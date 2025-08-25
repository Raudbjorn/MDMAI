"""Performance tests for critical operations in TTRPG Assistant.

This module provides comprehensive performance testing for:
- Search operations (semantic and hybrid)
- PDF processing and chunking
- Database query optimization
- Cache performance
- Embedding generation
- Parallel processing
- Memory usage monitoring
- Response time benchmarks
"""

import asyncio
import gc
import json
import os
import sys
import tempfile
import time
import tracemalloc
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import psutil
import pytest
import pytest_asyncio
import pytest_benchmark

from config.settings import settings
from src.core.database import ChromaDBManager
from src.pdf_processing.pipeline import PDFProcessingPipeline
from src.performance.cache_manager import GlobalCacheManager
from src.performance.parallel_processor import ParallelProcessor
from src.performance.performance_monitor import PerformanceMonitor
from src.search.search_service import SearchService


class PerformanceMetrics:
    """Helper class to track performance metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_peak = None
        self.operations_count = 0
        self.errors_count = 0
        
    def start(self):
        """Start tracking metrics."""
        self.start_time = time.perf_counter()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        tracemalloc.start()
        
    def stop(self):
        """Stop tracking and calculate metrics."""
        self.end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        self.memory_peak = peak / 1024 / 1024  # MB
        tracemalloc.stop()
        
    @property
    def elapsed_time(self):
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
        
    @property
    def throughput(self):
        """Get operations per second."""
        if self.elapsed_time > 0:
            return self.operations_count / self.elapsed_time
        return 0
        
    @property
    def memory_used(self):
        """Get memory used in MB."""
        if self.memory_peak and self.memory_start:
            return self.memory_peak - self.memory_start
        return 0
        
    def report(self) -> Dict[str, Any]:
        """Generate performance report."""
        return {
            "elapsed_time": f"{self.elapsed_time:.3f}s",
            "throughput": f"{self.throughput:.2f} ops/sec",
            "memory_used": f"{self.memory_used:.2f} MB",
            "memory_peak": f"{self.memory_peak:.2f} MB",
            "operations": self.operations_count,
            "errors": self.errors_count,
            "success_rate": f"{(1 - self.errors_count/max(self.operations_count, 1)) * 100:.1f}%"
        }


class TestSearchPerformance:
    """Test search operation performance."""
    
    @pytest_asyncio.fixture
    async def search_db(self, tmp_path):
        """Create and populate database for search testing."""
        with patch.object(settings, "chroma_db_path", tmp_path / "perf_search_db"):
            with patch.object(settings, "embedding_model", "all-MiniLM-L6-v2"):
                db = ChromaDBManager()
                
                # Generate test documents
                documents = []
                for i in range(1000):  # 1000 documents for performance testing
                    doc_type = ["spell", "monster", "item", "rule"][i % 4]
                    documents.append({
                        "id": f"perf_doc_{i}",
                        "content": f"This is {doc_type} document {i} with content about {doc_type} mechanics and rules. " * 5,
                        "metadata": {
                            "type": doc_type,
                            "level": i % 20,
                            "page": i % 300,
                            "rulebook": f"Book_{i % 10}"
                        }
                    })
                
                # Batch add documents
                db.batch_add_documents("rulebooks", documents)
                
                yield db
                
                if hasattr(db, "cleanup"):
                    await db.cleanup()
    
    @pytest.mark.asyncio
    async def test_single_search_performance(self, search_db, benchmark):
        """Benchmark single search operation."""
        search_service = SearchService(search_db)
        
        async def search_operation():
            return await search_service.search(
                query="spell damage fire",
                collection_name="rulebooks",
                max_results=10
            )
        
        # Warm up
        await search_operation()
        
        # Benchmark
        def sync_search():
            return asyncio.get_event_loop().run_until_complete(search_operation())
        result = benchmark(sync_search)
        
        assert len(result) > 0
        # Assert search completes in reasonable time
        assert benchmark.stats["mean"] < 0.5  # Less than 500ms average
    
    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self, search_db):
        """Test performance of concurrent searches."""
        search_service = SearchService(search_db)
        metrics = PerformanceMetrics()
        
        queries = [
            "fire spell damage",
            "monster dragon",
            "healing potion",
            "combat rules",
            "magic item sword",
        ] * 20  # 100 total queries
        
        metrics.start()
        
        # Run concurrent searches
        tasks = []
        for query in queries:
            task = asyncio.create_task(
                search_service.search(
                    query=query,
                    collection_name="rulebooks",
                    max_results=5
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.operations_count = len(results)
        metrics.errors_count = sum(1 for r in results if isinstance(r, Exception))
        metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nConcurrent Search Performance: {report}")
        
        assert metrics.elapsed_time < 10  # 100 searches in less than 10 seconds
        assert metrics.throughput > 10  # More than 10 searches per second
        assert metrics.errors_count == 0  # No errors
    
    @pytest.mark.asyncio
    async def test_filtered_search_performance(self, search_db):
        """Test performance of searches with metadata filters."""
        search_service = SearchService(search_db)
        metrics = PerformanceMetrics()
        
        metrics.start()
        
        # Test various filtered searches
        filters = [
            {"type": "spell"},
            {"type": "monster", "level": {"$gte": 10}},
            {"rulebook": "Book_5"},
            {"type": "item", "level": {"$lte": 5}},
        ]
        
        for filter_dict in filters * 5:  # 20 filtered searches
            result = await search_service.search(
                query="power",
                collection_name="rulebooks",
                max_results=10,
                metadata_filter=filter_dict
            )
            metrics.operations_count += 1
        
        metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nFiltered Search Performance: {report}")
        
        assert metrics.elapsed_time < 5  # 20 filtered searches in less than 5 seconds
        assert metrics.throughput > 4  # More than 4 filtered searches per second
    
    @pytest.mark.asyncio
    async def test_search_with_caching_performance(self, search_db):
        """Test search performance with caching enabled."""
        search_service = SearchService(search_db)
        metrics_uncached = PerformanceMetrics()
        metrics_cached = PerformanceMetrics()
        
        queries = ["fire spell", "dragon monster", "healing potion"] * 10
        
        # Test without cache (first run)
        metrics_uncached.start()
        for query in queries:
            await search_service.search(query, "rulebooks", max_results=5)
            metrics_uncached.operations_count += 1
        metrics_uncached.stop()
        
        # Test with cache (second run, same queries)
        metrics_cached.start()
        for query in queries:
            await search_service.search(query, "rulebooks", max_results=5)
            metrics_cached.operations_count += 1
        metrics_cached.stop()
        
        print(f"\nUncached Search: {metrics_uncached.report()}")
        print(f"Cached Search: {metrics_cached.report()}")
        
        # Cached should be significantly faster
        assert metrics_cached.elapsed_time < metrics_uncached.elapsed_time * 0.5


class TestPDFProcessingPerformance:
    """Test PDF processing pipeline performance."""
    
    @pytest.fixture
    def mock_pdf_content(self):
        """Generate mock PDF content for testing."""
        pages = []
        for i in range(100):  # 100 pages
            page_content = f"Page {i+1} content. " * 100  # ~500 words per page
            pages.append(page_content)
        return "\n\n".join(pages)
    
    @pytest.mark.asyncio
    async def test_pdf_processing_speed(self, mock_pdf_content, tmp_path):
        """Test PDF processing pipeline speed."""
        pipeline = PDFProcessingPipeline()
        metrics = PerformanceMetrics()
        
        # Mock PDF extraction
        with patch.object(pipeline.parser, "extract_text_from_pdf") as mock_extract:
            mock_extract.return_value = {
                "text": mock_pdf_content,
                "total_pages": 100,
                "file_hash": "test_hash",
                "file_name": "test.pdf",
                "tables": []
            }
            
            with patch.object(pipeline, "_is_duplicate", return_value=False):
                with patch.object(pipeline.embedding_generator, "generate_embeddings") as mock_embed:
                    # Mock embedding generation (return dummy embeddings)
                    mock_embed.return_value = [[0.1] * 384] * 100  # Dummy embeddings
                    
                    metrics.start()
                    
                    result = await pipeline.process_pdf(
                        pdf_path="test.pdf",
                        rulebook_name="Performance Test Book",
                        system="D&D 5e"
                    )
                    
                    metrics.operations_count = result.get("total_chunks", 0)
                    metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nPDF Processing Performance: {report}")
        
        assert result["status"] == "success"
        assert metrics.elapsed_time < 30  # Process 100 pages in less than 30 seconds
        assert metrics.operations_count > 0  # Chunks were created
    
    @pytest.mark.asyncio
    async def test_chunking_performance(self):
        """Test document chunking performance."""
        from src.pdf_processing.content_chunker import ContentChunker
        
        chunker = ContentChunker()
        metrics = PerformanceMetrics()
        
        # Generate large document
        large_document = " ".join([f"Sentence {i}." for i in range(10000)])  # ~10k sentences
        
        metrics.start()
        
        chunks = chunker.chunk_document(
            text=large_document,
            metadata={"test": True},
            chunk_size=1000,
            chunk_overlap=200
        )
        
        metrics.operations_count = len(chunks)
        metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nChunking Performance: {report}")
        
        assert len(chunks) > 0
        assert metrics.elapsed_time < 5  # Chunk large document in less than 5 seconds
        assert metrics.throughput > 10  # More than 10 chunks per second
    
    @pytest.mark.asyncio
    async def test_embedding_generation_performance(self):
        """Test embedding generation performance."""
        from src.pdf_processing.embedding_generator import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        metrics = PerformanceMetrics()
        
        # Generate test texts
        texts = [f"Test document {i} with some content about magic and spells." for i in range(100)]
        
        metrics.start()
        
        embeddings = generator.generate_embeddings(texts)
        
        metrics.operations_count = len(embeddings)
        metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nEmbedding Generation Performance: {report}")
        
        assert len(embeddings) == 100
        assert metrics.elapsed_time < 10  # Generate 100 embeddings in less than 10 seconds
        assert metrics.throughput > 10  # More than 10 embeddings per second


class TestDatabasePerformance:
    """Test database operation performance."""
    
    @pytest_asyncio.fixture
    async def perf_db(self, tmp_path):
        """Create database for performance testing."""
        with patch.object(settings, "chroma_db_path", tmp_path / "perf_db"):
            with patch.object(settings, "embedding_model", "all-MiniLM-L6-v2"):
                db = ChromaDBManager()
                yield db
                if hasattr(db, "cleanup"):
                    await db.cleanup()
    
    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, perf_db):
        """Test bulk document insertion performance."""
        metrics = PerformanceMetrics()
        
        # Generate test documents
        documents = []
        for i in range(1000):
            documents.append({
                "id": f"bulk_doc_{i}",
                "content": f"Document {i} content " * 10,
                "metadata": {"index": i, "batch": "test"}
            })
        
        metrics.start()
        
        # Bulk insert
        perf_db.batch_add_documents("rulebooks", documents)
        
        metrics.operations_count = len(documents)
        metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nBulk Insert Performance: {report}")
        
        assert metrics.elapsed_time < 10  # Insert 1000 docs in less than 10 seconds
        assert metrics.throughput > 100  # More than 100 inserts per second
    
    @pytest.mark.asyncio
    async def test_query_performance_scaling(self, perf_db):
        """Test how query performance scales with database size."""
        metrics_small = PerformanceMetrics()
        metrics_large = PerformanceMetrics()
        
        # Test with small dataset (100 docs)
        small_docs = [
            {
                "id": f"small_{i}",
                "content": f"Small dataset document {i}",
                "metadata": {"size": "small"}
            }
            for i in range(100)
        ]
        perf_db.batch_add_documents("rulebooks", small_docs)
        
        metrics_small.start()
        for _ in range(10):
            perf_db.search("rulebooks", "document", n_results=5)
            metrics_small.operations_count += 1
        metrics_small.stop()
        
        # Test with large dataset (add 900 more docs)
        large_docs = [
            {
                "id": f"large_{i}",
                "content": f"Large dataset document {i}",
                "metadata": {"size": "large"}
            }
            for i in range(900)
        ]
        perf_db.batch_add_documents("rulebooks", large_docs)
        
        metrics_large.start()
        for _ in range(10):
            perf_db.search("rulebooks", "document", n_results=5)
            metrics_large.operations_count += 1
        metrics_large.stop()
        
        print(f"\nSmall Dataset Query: {metrics_small.report()}")
        print(f"Large Dataset Query: {metrics_large.report()}")
        
        # Performance shouldn't degrade too much
        assert metrics_large.elapsed_time < metrics_small.elapsed_time * 3
    
    @pytest.mark.asyncio
    async def test_update_performance(self, perf_db):
        """Test document update performance."""
        metrics = PerformanceMetrics()
        
        # Add initial documents
        for i in range(100):
            perf_db.add_document(
                "campaigns",
                f"update_doc_{i}",
                f"Initial content {i}",
                {"version": 1}
            )
        
        metrics.start()
        
        # Update all documents
        for i in range(100):
            perf_db.update_document(
                "campaigns",
                f"update_doc_{i}",
                f"Updated content {i}",
                {"version": 2, "updated": True}
            )
            metrics.operations_count += 1
        
        metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nUpdate Performance: {report}")
        
        assert metrics.elapsed_time < 10  # Update 100 docs in less than 10 seconds
        assert metrics.throughput > 10  # More than 10 updates per second


class TestCachePerformance:
    """Test cache system performance."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager for testing."""
        manager = GlobalCacheManager()
        yield manager
        manager.shutdown()
    
    def test_cache_hit_performance(self, cache_manager):
        """Test cache hit performance."""
        metrics = PerformanceMetrics()
        
        # Populate cache
        for i in range(1000):
            cache_manager.set(f"key_{i}", f"value_{i}")
        
        metrics.start()
        
        # Test cache hits
        for i in range(1000):
            value = cache_manager.get(f"key_{i}")
            if value:
                metrics.operations_count += 1
        
        metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nCache Hit Performance: {report}")
        
        assert metrics.operations_count == 1000  # All hits
        assert metrics.elapsed_time < 0.1  # 1000 hits in less than 100ms
        assert metrics.throughput > 10000  # More than 10k hits per second
    
    def test_cache_miss_performance(self, cache_manager):
        """Test cache miss performance."""
        metrics = PerformanceMetrics()
        
        metrics.start()
        
        # Test cache misses
        for i in range(1000):
            value = cache_manager.get(f"missing_key_{i}")
            metrics.operations_count += 1
        
        metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nCache Miss Performance: {report}")
        
        assert metrics.elapsed_time < 0.1  # 1000 misses in less than 100ms
        assert metrics.throughput > 10000  # More than 10k operations per second
    
    def test_cache_eviction_performance(self, cache_manager):
        """Test cache eviction performance."""
        metrics = PerformanceMetrics()
        
        # Configure small cache to trigger evictions
        cache_manager.config.update_config(max_memory_mb=10)
        
        metrics.start()
        
        # Add many items to trigger eviction
        for i in range(10000):
            cache_manager.set(f"evict_key_{i}", f"value_{i}" * 100)
            metrics.operations_count += 1
        
        metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nCache Eviction Performance: {report}")
        
        assert metrics.elapsed_time < 5  # Handle evictions efficiently
        assert metrics.memory_used < 50  # Memory usage controlled


class TestParallelProcessingPerformance:
    """Test parallel processing performance."""
    
    @pytest.fixture
    def parallel_processor(self):
        """Create parallel processor for testing."""
        return ParallelProcessor(max_workers=4)
    
    @pytest.mark.asyncio
    async def test_parallel_task_execution(self, parallel_processor):
        """Test parallel task execution performance."""
        metrics = PerformanceMetrics()
        
        # Define CPU-bound task
        def cpu_task(n):
            """Simulate CPU-bound task."""
            result = 0
            for i in range(n * 1000):
                result += i
            return result
        
        metrics.start()
        
        # Execute tasks in parallel
        tasks = [cpu_task for _ in range(100)]
        args = [(1000,) for _ in range(100)]
        
        results = await parallel_processor.run_parallel(tasks, args)
        
        metrics.operations_count = len(results)
        metrics.stop()
        
        # Performance assertions
        report = metrics.report()
        print(f"\nParallel Processing Performance: {report}")
        
        assert len(results) == 100
        assert metrics.elapsed_time < 30  # 100 CPU tasks in less than 30 seconds
    
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self, parallel_processor):
        """Compare parallel vs sequential execution."""
        metrics_sequential = PerformanceMetrics()
        metrics_parallel = PerformanceMetrics()
        
        # Define task
        def task(n):
            time.sleep(0.01)  # Simulate work
            return n * 2
        
        # Sequential execution
        metrics_sequential.start()
        results_seq = []
        for i in range(50):
            results_seq.append(task(i))
            metrics_sequential.operations_count += 1
        metrics_sequential.stop()
        
        # Parallel execution
        metrics_parallel.start()
        tasks = [task for _ in range(50)]
        args = [(i,) for i in range(50)]
        results_par = await parallel_processor.run_parallel(tasks, args)
        metrics_parallel.operations_count = len(results_par)
        metrics_parallel.stop()
        
        print(f"\nSequential Execution: {metrics_sequential.report()}")
        print(f"Parallel Execution: {metrics_parallel.report()}")
        
        # Parallel should be significantly faster
        assert metrics_parallel.elapsed_time < metrics_sequential.elapsed_time * 0.5


class TestMemoryPerformance:
    """Test memory usage and management."""
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, tmp_path):
        """Test for memory leaks in core operations."""
        # Track memory over multiple iterations
        memory_samples = []
        
        with patch.object(settings, "chroma_db_path", tmp_path / "memory_test"):
            db = ChromaDBManager()
            search_service = SearchService(db)
            
            # Perform operations multiple times
            for iteration in range(5):
                gc.collect()  # Force garbage collection
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Perform operations
                for i in range(100):
                    # Add document
                    db.add_document(
                        "rulebooks",
                        f"mem_test_{iteration}_{i}",
                        f"Content {i}",
                        {"iter": iteration}
                    )
                    
                    # Search
                    await search_service.search("test", "rulebooks", max_results=5)
                
                gc.collect()
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(memory_after - memory_before)
            
            if hasattr(db, "cleanup"):
                await db.cleanup()
        
        # Check for memory leak (memory shouldn't grow significantly)
        avg_memory_growth = sum(memory_samples) / len(memory_samples)
        print(f"\nAverage memory growth per iteration: {avg_memory_growth:.2f} MB")
        
        # Memory growth should be minimal
        assert avg_memory_growth < 50  # Less than 50MB growth per iteration
    
    def test_large_document_memory_usage(self):
        """Test memory usage with large documents."""
        from src.pdf_processing.content_chunker import ContentChunker
        
        chunker = ContentChunker()
        
        # Create large document (10MB of text)
        large_text = "x" * (10 * 1024 * 1024)
        
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process large document
        chunks = chunker.chunk_document(
            text=large_text,
            metadata={},
            chunk_size=1000,
            chunk_overlap=200
        )
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        print(f"\nLarge document memory usage: {memory_used:.2f} MB")
        
        # Memory usage should be reasonable
        assert memory_used < 100  # Less than 100MB for 10MB document
        assert len(chunks) > 0


class TestResponseTimePerformance:
    """Test end-to-end response time performance."""
    
    @pytest.mark.asyncio
    async def test_search_response_time(self, tmp_path):
        """Test search operation response time."""
        with patch.object(settings, "chroma_db_path", tmp_path / "response_test"):
            db = ChromaDBManager()
            search_service = SearchService(db)
            
            # Add test documents
            for i in range(100):
                db.add_document(
                    "rulebooks",
                    f"response_doc_{i}",
                    f"Document content {i}",
                    {"index": i}
                )
            
            # Measure response times
            response_times = []
            
            for _ in range(20):
                start = time.perf_counter()
                await search_service.search("content", "rulebooks", max_results=10)
                response_times.append(time.perf_counter() - start)
            
            # Calculate statistics
            avg_response = sum(response_times) / len(response_times)
            max_response = max(response_times)
            min_response = min(response_times)
            p95_response = sorted(response_times)[int(len(response_times) * 0.95)]
            
            print(f"\nSearch Response Times:")
            print(f"  Average: {avg_response*1000:.2f}ms")
            print(f"  Min: {min_response*1000:.2f}ms")
            print(f"  Max: {max_response*1000:.2f}ms")
            print(f"  P95: {p95_response*1000:.2f}ms")
            
            # Performance requirements
            assert avg_response < 0.2  # Average less than 200ms
            assert p95_response < 0.5  # 95th percentile less than 500ms
            assert max_response < 1.0  # Max less than 1 second
            
            if hasattr(db, "cleanup"):
                await db.cleanup()


class TestOptimizationPerformance:
    """Test performance optimization features."""
    
    @pytest.mark.asyncio
    async def test_database_optimization_impact(self, tmp_path):
        """Test impact of database optimization."""
        with patch.object(settings, "chroma_db_path", tmp_path / "optimize_test"):
            db = ChromaDBManager()
            
            # Add many documents
            for i in range(500):
                db.add_document(
                    "rulebooks",
                    f"opt_doc_{i}",
                    f"Document {i} content",
                    {"index": i}
                )
            
            # Measure performance before optimization
            times_before = []
            for _ in range(10):
                start = time.perf_counter()
                db.search("rulebooks", "content", n_results=10)
                times_before.append(time.perf_counter() - start)
            
            # Run optimization
            if db.optimizer:
                db.optimizer.optimize_collection("rulebooks")
            
            # Measure performance after optimization
            times_after = []
            for _ in range(10):
                start = time.perf_counter()
                db.search("rulebooks", "content", n_results=10)
                times_after.append(time.perf_counter() - start)
            
            avg_before = sum(times_before) / len(times_before)
            avg_after = sum(times_after) / len(times_after)
            
            print(f"\nOptimization Impact:")
            print(f"  Before: {avg_before*1000:.2f}ms average")
            print(f"  After: {avg_after*1000:.2f}ms average")
            print(f"  Improvement: {((avg_before - avg_after) / avg_before * 100):.1f}%")
            
            # Optimization should maintain or improve performance
            assert avg_after <= avg_before * 1.1  # At worst 10% slower
            
            if hasattr(db, "cleanup"):
                await db.cleanup()


# Benchmark comparison fixture
@pytest.fixture(scope="module")
def performance_summary():
    """Collect and summarize all performance metrics."""
    summary = {
        "search": {},
        "database": {},
        "cache": {},
        "parallel": {},
        "memory": {}
    }
    yield summary
    
    # Print summary at end of test run
    print("\n" + "="*50)
    print("PERFORMANCE TEST SUMMARY")
    print("="*50)
    
    for category, metrics in summary.items():
        if metrics:
            print(f"\n{category.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")