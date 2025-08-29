# SPIKE 4: Performance Requirements for TTRPG MCP Server

## Executive Summary

This spike defines comprehensive performance requirements, benchmarks, and optimization strategies for the TTRPG MCP Server. The system must handle real-time collaboration, complex PDF processing, vector search operations, and maintain responsive user experiences across multiple concurrent sessions.

## 1. Performance Benchmarks and SLAs

### 1.1 PDF Processing Speed Requirements

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class PDFComplexity(Enum):
    SIMPLE = "simple"      # Text-only, <50 pages
    MODERATE = "moderate"  # Mixed content, 50-200 pages
    COMPLEX = "complex"    # Heavy graphics, >200 pages

@dataclass
class PDFProcessingBenchmark:
    """Performance benchmarks for PDF processing operations"""
    
    # Target processing times (seconds)
    EXTRACTION_TARGETS: Dict[PDFComplexity, float] = {
        PDFComplexity.SIMPLE: 2.0,      # 2 seconds for simple PDFs
        PDFComplexity.MODERATE: 10.0,   # 10 seconds for moderate
        PDFComplexity.COMPLEX: 30.0     # 30 seconds for complex
    }
    
    # Chunking performance (pages per second)
    CHUNKING_RATE: float = 50.0
    
    # Vector embedding generation (chunks per second)
    EMBEDDING_RATE: float = 100.0
    
    # ChromaDB insertion (vectors per second)
    INSERTION_RATE: float = 1000.0
    
    # Memory limits (MB)
    MAX_MEMORY_PER_PDF: Dict[PDFComplexity, int] = {
        PDFComplexity.SIMPLE: 100,
        PDFComplexity.MODERATE: 500,
        PDFComplexity.COMPLEX: 1000
    }
```

### 1.2 Search Latency Targets

```python
@dataclass
class SearchPerformanceTargets:
    """Search operation performance requirements"""
    
    # Latency targets (milliseconds)
    VECTOR_SEARCH_P50: float = 50.0    # 50ms median
    VECTOR_SEARCH_P95: float = 200.0   # 200ms 95th percentile
    VECTOR_SEARCH_P99: float = 500.0   # 500ms 99th percentile
    
    # Hybrid search (vector + keyword)
    HYBRID_SEARCH_P50: float = 100.0
    HYBRID_SEARCH_P95: float = 400.0
    HYBRID_SEARCH_P99: float = 1000.0
    
    # Complex queries (multi-document)
    COMPLEX_QUERY_P50: float = 200.0
    COMPLEX_QUERY_P95: float = 800.0
    COMPLEX_QUERY_P99: float = 2000.0
    
    # Results per query
    MIN_RESULTS_COUNT: int = 10
    MAX_RESULTS_COUNT: int = 100
    DEFAULT_RESULTS_COUNT: int = 20
```

### 1.3 WebSocket Message Throughput

```python
@dataclass
class WebSocketPerformanceTargets:
    """Real-time collaboration performance requirements"""
    
    # Connection targets
    MAX_CONCURRENT_CONNECTIONS: int = 1000
    CONNECTION_ESTABLISHMENT_P95: float = 100.0  # 100ms
    
    # Message throughput (messages per second)
    MESSAGES_PER_SECOND_PER_CLIENT: float = 10.0
    TOTAL_MESSAGES_PER_SECOND: float = 5000.0
    
    # Latency targets (milliseconds)
    MESSAGE_DELIVERY_P50: float = 10.0
    MESSAGE_DELIVERY_P95: float = 50.0
    MESSAGE_DELIVERY_P99: float = 100.0
    
    # Broadcast performance
    BROADCAST_TO_100_CLIENTS: float = 50.0  # 50ms
    BROADCAST_TO_1000_CLIENTS: float = 200.0  # 200ms
    
    # State synchronization
    FULL_STATE_SYNC_P95: float = 1000.0  # 1 second
    INCREMENTAL_SYNC_P95: float = 100.0  # 100ms
```

### 1.4 Concurrent User Limits

```python
@dataclass
class ConcurrencyLimits:
    """System concurrency constraints"""
    
    # Per session limits
    MAX_USERS_PER_SESSION: int = 20
    MAX_ACTIVE_SESSIONS: int = 100
    TOTAL_CONCURRENT_USERS: int = 2000
    
    # Resource limits per user
    MAX_DOCUMENTS_PER_USER: int = 50
    MAX_QUERIES_PER_MINUTE: int = 60
    MAX_UPLOAD_SIZE_MB: int = 100
    
    # Database connection pools
    CHROMADB_POOL_SIZE: int = 50
    REDIS_POOL_SIZE: int = 100
    POSTGRES_POOL_SIZE: int = 50
```

## 2. Backend Optimization Strategies

### 2.1 ChromaDB Query Optimization

```python
import asyncio
from typing import List, Dict, Any, Optional
from functools import lru_cache
import numpy as np
from chromadb import Client
from chromadb.config import Settings
import time

class OptimizedChromaDBClient:
    """Optimized ChromaDB client with performance enhancements"""
    
    def __init__(self, 
                 persist_directory: str,
                 pool_size: int = 10,
                 cache_size: int = 1000):
        # Connection pooling
        self.pool = self._create_connection_pool(persist_directory, pool_size)
        self.pool_semaphore = asyncio.Semaphore(pool_size)
        
        # Query result caching
        self.cache_size = cache_size
        self._query_cache = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Batch processing queue
        self.batch_queue = asyncio.Queue(maxsize=1000)
        self.batch_processor_task = None
        
    def _create_connection_pool(self, persist_directory: str, size: int) -> List[Client]:
        """Create a pool of ChromaDB client connections"""
        pool = []
        for _ in range(size):
            client = Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,
                allow_reset=False
            ))
            pool.append(client)
        return pool
    
    async def vector_search_optimized(self,
                                     collection_name: str,
                                     query_embedding: List[float],
                                     n_results: int = 10,
                                     where_filter: Optional[Dict] = None) -> Dict[str, Any]:
        """Optimized vector search with caching and connection pooling"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            collection_name, query_embedding, n_results, where_filter
        )
        
        # Check cache
        if cache_key in self._query_cache:
            self._cache_stats['hits'] += 1
            return self._query_cache[cache_key]
        
        self._cache_stats['misses'] += 1
        
        # Acquire connection from pool
        async with self.pool_semaphore:
            start_time = time.perf_counter()
            
            # Get available connection
            client = await self._get_connection()
            
            try:
                # Execute query
                collection = client.get_collection(collection_name)
                results = await asyncio.to_thread(
                    collection.query,
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_filter
                )
                
                # Calculate metrics
                query_time = (time.perf_counter() - start_time) * 1000
                
                # Add performance metadata
                results['_performance'] = {
                    'query_time_ms': query_time,
                    'cache_hit': False,
                    'results_count': len(results.get('ids', [[]])[0])
                }
                
                # Cache results
                self._cache_result(cache_key, results)
                
                return results
                
            finally:
                # Return connection to pool
                await self._return_connection(client)
    
    async def batch_insert_optimized(self,
                                    collection_name: str,
                                    embeddings: List[List[float]],
                                    metadatas: List[Dict],
                                    ids: List[str],
                                    batch_size: int = 100) -> Dict[str, Any]:
        """Optimized batch insertion with chunking and parallel processing"""
        
        start_time = time.perf_counter()
        total_inserted = 0
        errors = []
        
        # Split into optimal batch sizes
        batches = self._create_batches(embeddings, metadatas, ids, batch_size)
        
        # Process batches in parallel
        tasks = []
        for batch in batches:
            task = self._insert_batch(collection_name, batch)
            tasks.append(task)
        
        # Limit concurrent batch processing
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent batches
        
        async def bounded_insert(batch_task):
            async with semaphore:
                return await batch_task
        
        results = await asyncio.gather(
            *[bounded_insert(task) for task in tasks],
            return_exceptions=True
        )
        
        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            else:
                total_inserted += result.get('inserted', 0)
        
        insert_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'total_inserted': total_inserted,
            'insert_time_ms': insert_time,
            'throughput': total_inserted / (insert_time / 1000) if insert_time > 0 else 0,
            'errors': errors,
            'batch_size': batch_size,
            'num_batches': len(batches)
        }
    
    def _generate_cache_key(self, *args) -> str:
        """Generate a cache key from query parameters"""
        import hashlib
        import json
        
        # Convert args to JSON string
        key_data = json.dumps(args, sort_keys=True, default=str)
        
        # Create hash
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cache_result(self, key: str, result: Dict) -> None:
        """Cache query result with LRU eviction"""
        
        # Evict if cache is full
        if len(self._query_cache) >= self.cache_size:
            # Simple FIFO eviction (could be improved to LRU)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
            self._cache_stats['evictions'] += 1
        
        self._query_cache[key] = result
    
    async def _get_connection(self) -> Client:
        """Get an available connection from the pool"""
        # Simple round-robin (could be improved with health checks)
        return self.pool[0]
    
    async def _return_connection(self, client: Client) -> None:
        """Return a connection to the pool"""
        # In a real implementation, would track connection state
        pass
    
    def _create_batches(self, embeddings, metadatas, ids, batch_size):
        """Split data into optimal batches"""
        batches = []
        for i in range(0, len(embeddings), batch_size):
            batch = {
                'embeddings': embeddings[i:i+batch_size],
                'metadatas': metadatas[i:i+batch_size],
                'ids': ids[i:i+batch_size]
            }
            batches.append(batch)
        return batches
    
    async def _insert_batch(self, collection_name: str, batch: Dict) -> Dict:
        """Insert a single batch"""
        client = await self._get_connection()
        try:
            collection = client.get_collection(collection_name)
            collection.add(
                embeddings=batch['embeddings'],
                metadatas=batch['metadatas'],
                ids=batch['ids']
            )
            return {'inserted': len(batch['ids'])}
        finally:
            await self._return_connection(client)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = (
            self._cache_stats['hits'] / 
            (self._cache_stats['hits'] + self._cache_stats['misses'])
            if (self._cache_stats['hits'] + self._cache_stats['misses']) > 0
            else 0
        )
        
        return {
            'cache': {
                'size': len(self._query_cache),
                'hit_rate': cache_hit_rate,
                **self._cache_stats
            },
            'pool': {
                'size': len(self.pool),
                'available': self.pool_semaphore._value
            }
        }
```

### 2.2 PDF Processing Pipeline Improvements

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Any, AsyncIterator
import multiprocessing as mp
from dataclasses import dataclass
import PyPDF2
import fitz  # PyMuPDF for better performance
import io

@dataclass
class ProcessingMetrics:
    """Metrics for PDF processing performance"""
    pages_processed: int = 0
    extraction_time_ms: float = 0
    chunking_time_ms: float = 0
    embedding_time_ms: float = 0
    total_time_ms: float = 0
    memory_peak_mb: float = 0

class OptimizedPDFProcessor:
    """High-performance PDF processing pipeline"""
    
    def __init__(self, 
                 max_workers: int = None,
                 chunk_size: int = 1000,
                 overlap: int = 200):
        # Use CPU count for workers if not specified
        self.max_workers = max_workers or mp.cpu_count()
        
        # Processing pools
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Memory management
        self.max_memory_mb = 1000  # Maximum memory per PDF
        
    async def process_pdf_optimized(self, 
                                   pdf_bytes: bytes,
                                   extract_images: bool = False) -> Dict[str, Any]:
        """Process PDF with optimized extraction and chunking"""
        
        metrics = ProcessingMetrics()
        start_time = time.perf_counter()
        
        # Stage 1: Parallel page extraction
        extraction_start = time.perf_counter()
        pages = await self._extract_pages_parallel(pdf_bytes, extract_images)
        metrics.extraction_time_ms = (time.perf_counter() - extraction_start) * 1000
        metrics.pages_processed = len(pages)
        
        # Stage 2: Intelligent chunking with sliding window
        chunking_start = time.perf_counter()
        chunks = await self._chunk_pages_optimized(pages)
        metrics.chunking_time_ms = (time.perf_counter() - chunking_start) * 1000
        
        # Stage 3: Batch embedding generation
        embedding_start = time.perf_counter()
        embeddings = await self._generate_embeddings_batch(chunks)
        metrics.embedding_time_ms = (time.perf_counter() - embedding_start) * 1000
        
        metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'pages': pages,
            'chunks': chunks,
            'embeddings': embeddings,
            'metrics': metrics
        }
    
    async def _extract_pages_parallel(self, 
                                     pdf_bytes: bytes,
                                     extract_images: bool) -> List[Dict[str, Any]]:
        """Extract pages in parallel using multiple workers"""
        
        # Open PDF with PyMuPDF for better performance
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = pdf_document.page_count
        
        # Split pages among workers
        pages_per_worker = max(1, total_pages // self.max_workers)
        
        # Create extraction tasks
        tasks = []
        for worker_id in range(self.max_workers):
            start_page = worker_id * pages_per_worker
            end_page = min(start_page + pages_per_worker, total_pages)
            
            if start_page < total_pages:
                task = self._extract_page_range(
                    pdf_bytes, start_page, end_page, extract_images
                )
                tasks.append(task)
        
        # Execute extraction in parallel
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        pages = []
        for worker_pages in results:
            pages.extend(worker_pages)
        
        pdf_document.close()
        return pages
    
    async def _extract_page_range(self,
                                 pdf_bytes: bytes,
                                 start_page: int,
                                 end_page: int,
                                 extract_images: bool) -> List[Dict[str, Any]]:
        """Extract a range of pages in a worker process"""
        
        loop = asyncio.get_event_loop()
        
        # Run in process pool for CPU-intensive work
        result = await loop.run_in_executor(
            self.process_pool,
            self._extract_pages_worker,
            pdf_bytes,
            start_page,
            end_page,
            extract_images
        )
        
        return result
    
    @staticmethod
    def _extract_pages_worker(pdf_bytes: bytes,
                             start_page: int,
                             end_page: int,
                             extract_images: bool) -> List[Dict[str, Any]]:
        """Worker function for page extraction"""
        
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        
        for page_num in range(start_page, end_page):
            page = pdf_document[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Extract tables if present
            tables = []
            # Table extraction logic here
            
            # Extract images if requested
            images = []
            if extract_images:
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        images.append({
                            'index': img_index,
                            'width': pix.width,
                            'height': pix.height,
                            'data': pix.tobytes()
                        })
                    pix = None
            
            pages.append({
                'page_num': page_num,
                'text': text,
                'tables': tables,
                'images': images
            })
        
        pdf_document.close()
        return pages
    
    async def _chunk_pages_optimized(self, pages: List[Dict]) -> List[Dict[str, Any]]:
        """Create optimized chunks with sliding window"""
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for page in pages:
            text = page['text']
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                sentence_size = len(sentence.split())
                
                if current_size + sentence_size > self.chunk_size:
                    # Save current chunk
                    if current_chunk:
                        chunks.append({
                            'text': ' '.join(current_chunk),
                            'metadata': {
                                'pages': list(set([p['page_num'] for p in pages 
                                                 if any(s in p['text'] for s in current_chunk)])),
                                'word_count': current_size
                            }
                        })
                    
                    # Start new chunk with overlap
                    overlap_sentences = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                    current_chunk = overlap_sentences + [sentence]
                    current_size = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'metadata': {
                    'pages': list(set([p['page_num'] for p in pages 
                                     if any(s in p['text'] for s in current_chunk)])),
                    'word_count': current_size
                }
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _generate_embeddings_batch(self, chunks: List[Dict]) -> List[List[float]]:
        """Generate embeddings in batches for efficiency"""
        
        # Batch size for embedding generation
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_texts = [chunk['text'] for chunk in batch]
            
            # Generate embeddings (placeholder - would use actual embedding model)
            batch_embeddings = await self._embed_texts(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts (placeholder)"""
        # In production, would use actual embedding model
        import numpy as np
        return [np.random.rand(768).tolist() for _ in texts]
```

### 2.3 Caching Strategies with Redis

```python
import redis.asyncio as redis
import json
import hashlib
import pickle
from typing import Any, Optional, Union, Dict, List
from datetime import timedelta
import asyncio

class RedisPerformanceCache:
    """High-performance caching layer with Redis"""
    
    def __init__(self,
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 pool_size: int = 50):
        
        # Create connection pool for better performance
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            max_connections=pool_size,
            decode_responses=False  # Handle binary data
        )
        
        # Cache configuration
        self.default_ttl = timedelta(hours=1)
        self.max_cache_size_mb = 1000
        
        # Performance tracking
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis client from pool"""
        return redis.Redis(connection_pool=self.pool)
    
    async def cached_query(self,
                          key: str,
                          query_func,
                          ttl: Optional[timedelta] = None,
                          force_refresh: bool = False) -> Any:
        """Execute query with caching"""
        
        if not force_refresh:
            # Try to get from cache
            cached_result = await self.get(key)
            if cached_result is not None:
                self.stats['hits'] += 1
                return cached_result
        
        self.stats['misses'] += 1
        
        # Execute query
        result = await query_func()
        
        # Cache result
        await self.set(key, result, ttl or self.default_ttl)
        
        return result
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with performance tracking"""
        
        client = await self.get_redis()
        try:
            # Get with pipeline for better performance
            async with client.pipeline() as pipe:
                pipe.get(key)
                pipe.ttl(key)
                value, ttl = await pipe.execute()
            
            if value is None:
                return None
            
            # Deserialize
            return self._deserialize(value)
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"Cache get error: {e}")
            return None
    
    async def set(self, 
                  key: str, 
                  value: Any,
                  ttl: Optional[timedelta] = None) -> bool:
        """Set value in cache with TTL"""
        
        client = await self.get_redis()
        try:
            # Serialize value
            serialized = self._serialize(value)
            
            # Check size
            if len(serialized) > self.max_cache_size_mb * 1024 * 1024:
                print(f"Value too large for cache: {len(serialized)} bytes")
                return False
            
            # Set with TTL
            ttl_seconds = int(ttl.total_seconds()) if ttl else int(self.default_ttl.total_seconds())
            
            await client.setex(key, ttl_seconds, serialized)
            self.stats['sets'] += 1
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"Cache set error: {e}")
            return False
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values in one operation"""
        
        client = await self.get_redis()
        try:
            # Use pipeline for batch operation
            async with client.pipeline() as pipe:
                for key in keys:
                    pipe.get(key)
                values = await pipe.execute()
            
            # Build result dict
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
                    self.stats['hits'] += 1
                else:
                    self.stats['misses'] += 1
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"Batch get error: {e}")
            return {}
    
    async def batch_set(self, 
                       items: Dict[str, Any],
                       ttl: Optional[timedelta] = None) -> int:
        """Set multiple values in one operation"""
        
        client = await self.get_redis()
        successful = 0
        
        try:
            ttl_seconds = int(ttl.total_seconds()) if ttl else int(self.default_ttl.total_seconds())
            
            # Use pipeline for batch operation
            async with client.pipeline() as pipe:
                for key, value in items.items():
                    serialized = self._serialize(value)
                    if len(serialized) <= self.max_cache_size_mb * 1024 * 1024:
                        pipe.setex(key, ttl_seconds, serialized)
                        successful += 1
                
                await pipe.execute()
            
            self.stats['sets'] += successful
            return successful
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"Batch set error: {e}")
            return 0
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        
        client = await self.get_redis()
        try:
            # Find matching keys
            keys = []
            async for key in client.scan_iter(match=pattern, count=100):
                keys.append(key)
            
            if keys:
                # Delete in batches
                deleted = 0
                for i in range(0, len(keys), 100):
                    batch = keys[i:i+100]
                    deleted += await client.delete(*batch)
                
                self.stats['deletes'] += deleted
                return deleted
            
            return 0
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"Invalidate pattern error: {e}")
            return 0
    
    async def warm_cache(self, 
                        queries: List[Dict[str, Any]],
                        query_func) -> int:
        """Pre-warm cache with common queries"""
        
        warmed = 0
        
        for query in queries:
            key = self._generate_key(query)
            existing = await self.get(key)
            
            if existing is None:
                result = await query_func(**query)
                if await self.set(key, result):
                    warmed += 1
        
        return warmed
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        return pickle.loads(data)
    
    def _generate_key(self, params: Dict) -> str:
        """Generate cache key from parameters"""
        key_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    async def close(self):
        """Close connection pool"""
        await self.pool.disconnect()
```

### 2.4 Connection Pooling

```python
from contextlib import asynccontextmanager
import asyncio
from typing import Optional, Any
import asyncpg
from dataclasses import dataclass

@dataclass
class PoolConfig:
    """Configuration for connection pools"""
    min_size: int = 10
    max_size: int = 50
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0
    timeout: float = 10.0
    command_timeout: float = 10.0
    max_cached_statement_lifetime: int = 3600
    max_cacheable_statement_size: int = 1024

class ConnectionPoolManager:
    """Manage database connection pools for optimal performance"""
    
    def __init__(self):
        self.pools = {}
        self.pool_stats = {}
        
    async def create_postgres_pool(self,
                                  dsn: str,
                                  name: str = 'default',
                                  config: Optional[PoolConfig] = None) -> asyncpg.Pool:
        """Create optimized PostgreSQL connection pool"""
        
        config = config or PoolConfig()
        
        pool = await asyncpg.create_pool(
            dsn,
            min_size=config.min_size,
            max_size=config.max_size,
            max_queries=config.max_queries,
            max_inactive_connection_lifetime=config.max_inactive_connection_lifetime,
            timeout=config.timeout,
            command_timeout=config.command_timeout,
            max_cached_statement_lifetime=config.max_cached_statement_lifetime,
            max_cacheable_statement_size=config.max_cacheable_statement_size
        )
        
        self.pools[name] = pool
        self.pool_stats[name] = {
            'created': asyncio.get_event_loop().time(),
            'queries': 0,
            'errors': 0
        }
        
        return pool
    
    @asynccontextmanager
    async def acquire(self, pool_name: str = 'default'):
        """Acquire connection from pool with monitoring"""
        
        pool = self.pools.get(pool_name)
        if not pool:
            raise ValueError(f"Pool {pool_name} not found")
        
        start_time = asyncio.get_event_loop().time()
        
        async with pool.acquire() as connection:
            acquire_time = asyncio.get_event_loop().time() - start_time
            
            # Track slow acquisitions
            if acquire_time > 1.0:
                print(f"Slow connection acquisition: {acquire_time:.2f}s")
            
            self.pool_stats[pool_name]['queries'] += 1
            
            try:
                yield connection
            except Exception as e:
                self.pool_stats[pool_name]['errors'] += 1
                raise
    
    async def execute_with_retry(self,
                                pool_name: str,
                                query: str,
                                *args,
                                max_retries: int = 3,
                                retry_delay: float = 0.1) -> Any:
        """Execute query with automatic retry on connection errors"""
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                async with self.acquire(pool_name) as conn:
                    return await conn.fetch(query, *args)
                    
            except (asyncpg.ConnectionDoesNotExistError,
                    asyncpg.ConnectionFailureError,
                    asyncpg.TooManyConnectionsError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                continue
            except Exception as e:
                # Non-retryable error
                raise
        
        raise last_error
    
    def get_pool_stats(self, pool_name: str = None) -> Dict[str, Any]:
        """Get statistics for connection pools"""
        
        if pool_name:
            pool = self.pools.get(pool_name)
            if not pool:
                return {}
            
            return {
                'name': pool_name,
                'size': pool.get_size(),
                'free_connections': pool.get_idle_size(),
                'used_connections': pool.get_size() - pool.get_idle_size(),
                'max_size': pool._maxsize,
                **self.pool_stats.get(pool_name, {})
            }
        
        # Return stats for all pools
        stats = {}
        for name in self.pools:
            stats[name] = self.get_pool_stats(name)
        
        return stats
    
    async def close_all(self):
        """Close all connection pools"""
        
        for name, pool in self.pools.items():
            await pool.close()
            print(f"Closed pool: {name}")
        
        self.pools.clear()
        self.pool_stats.clear()
```

## 3. Frontend Performance

### 3.1 Bundle Size Optimization

```javascript
// vite.config.ts optimization for bundle size
import { defineConfig } from 'vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { visualizer } from 'rollup-plugin-visualizer';
import compression from 'vite-plugin-compression';

export default defineConfig({
  plugins: [
    sveltekit(),
    
    // Analyze bundle size
    visualizer({
      filename: './stats.html',
      open: false,
      gzipSize: true,
      brotliSize: true
    }),
    
    // Compress assets
    compression({
      algorithm: 'brotliCompress',
      ext: '.br',
      threshold: 10240,
      deleteOriginFile: false
    })
  ],
  
  build: {
    // Optimize build
    target: 'es2020',
    minify: 'terser',
    
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    },
    
    // Split chunks for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['svelte', '@sveltejs/kit'],
          'collaboration': ['./src/lib/components/collaboration'],
          'performance': ['./src/lib/performance'],
          'utils': ['./src/lib/utils']
        }
      }
    },
    
    // Chunk size warnings
    chunkSizeWarningLimit: 500
  },
  
  // Optimize dependencies
  optimizeDeps: {
    include: ['svelte', '@sveltejs/kit'],
    exclude: ['@sveltejs/kit/node']
  }
});
```

### 3.2 Lazy Loading Strategies

```typescript
// Lazy loading implementation for components
import { lazy } from 'svelte/store';
import type { ComponentType, SvelteComponent } from 'svelte';

export class LazyComponentLoader {
  private cache = new Map<string, Promise<ComponentType>>();
  private preloadQueue: Set<string> = new Set();
  
  /**
   * Load component lazily with caching
   */
  async loadComponent(
    path: string,
    preload: boolean = false
  ): Promise<ComponentType> {
    // Check cache
    if (this.cache.has(path)) {
      return this.cache.get(path)!;
    }
    
    // Create loader promise
    const loader = this.createLoader(path);
    this.cache.set(path, loader);
    
    // Add to preload queue if requested
    if (preload) {
      this.preloadQueue.add(path);
      this.processPreloadQueue();
    }
    
    return loader;
  }
  
  /**
   * Create component loader with performance tracking
   */
  private async createLoader(path: string): Promise<ComponentType> {
    const startTime = performance.now();
    
    try {
      // Dynamic import with webpack magic comments
      const module = await import(
        /* webpackChunkName: "[request]" */
        /* webpackPrefetch: true */
        path
      );
      
      const loadTime = performance.now() - startTime;
      
      // Track performance
      if (window.metricsCollector) {
        window.metricsCollector.recordTiming(
          'component_load',
          loadTime,
          { component: path }
        );
      }
      
      return module.default;
      
    } catch (error) {
      console.error(`Failed to load component: ${path}`, error);
      throw error;
    }
  }
  
  /**
   * Preload components in idle time
   */
  private async processPreloadQueue(): Promise<void> {
    if (this.preloadQueue.size === 0) return;
    
    // Use requestIdleCallback for preloading
    if ('requestIdleCallback' in window) {
      requestIdleCallback(async (deadline) => {
        while (deadline.timeRemaining() > 0 && this.preloadQueue.size > 0) {
          const path = this.preloadQueue.values().next().value;
          this.preloadQueue.delete(path);
          
          if (!this.cache.has(path)) {
            await this.loadComponent(path);
          }
        }
        
        // Continue if more to preload
        if (this.preloadQueue.size > 0) {
          this.processPreloadQueue();
        }
      });
    } else {
      // Fallback for browsers without requestIdleCallback
      setTimeout(() => {
        const path = this.preloadQueue.values().next().value;
        this.preloadQueue.delete(path);
        
        if (!this.cache.has(path)) {
          this.loadComponent(path).then(() => {
            if (this.preloadQueue.size > 0) {
              this.processPreloadQueue();
            }
          });
        }
      }, 100);
    }
  }
  
  /**
   * Preload critical components
   */
  async preloadCritical(paths: string[]): Promise<void> {
    const promises = paths.map(path => this.loadComponent(path, false));
    await Promise.all(promises);
  }
  
  /**
   * Clear component cache
   */
  clearCache(path?: string): void {
    if (path) {
      this.cache.delete(path);
    } else {
      this.cache.clear();
    }
  }
}

// Route-based code splitting
export const routeComponents = {
  '/': () => import('$routes/+page.svelte'),
  '/dashboard': () => import('$routes/dashboard/+page.svelte'),
  '/campaign/[id]/session': () => import('$routes/campaign/[id]/session/+page.svelte'),
  '/performance': () => import('$routes/performance/+page.svelte'),
  '/providers': () => import('$routes/providers/+page.svelte')
};
```

### 3.3 Service Worker Caching

```typescript
// service-worker.ts - Advanced caching strategies
import { build, files, version } from '$service-worker';

const CACHE_NAME = `ttrpg-mcp-v${version}`;
const API_CACHE = 'api-cache-v1';
const IMAGE_CACHE = 'image-cache-v1';

// Assets to precache
const PRECACHE_ASSETS = [
  ...build,
  ...files
];

// Cache strategies
enum CacheStrategy {
  CACHE_FIRST = 'cache-first',
  NETWORK_FIRST = 'network-first',
  CACHE_ONLY = 'cache-only',
  NETWORK_ONLY = 'network-only',
  STALE_WHILE_REVALIDATE = 'stale-while-revalidate'
}

class ServiceWorkerCache {
  /**
   * Install event - precache assets
   */
  async handleInstall(event: ExtendableEvent): Promise<void> {
    event.waitUntil(
      caches.open(CACHE_NAME).then(cache => {
        console.log('Precaching assets...');
        return cache.addAll(PRECACHE_ASSETS);
      })
    );
  }
  
  /**
   * Activate event - clean old caches
   */
  async handleActivate(event: ExtendableEvent): Promise<void> {
    event.waitUntil(
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames
            .filter(name => name !== CACHE_NAME && 
                          name !== API_CACHE && 
                          name !== IMAGE_CACHE)
            .map(name => caches.delete(name))
        );
      })
    );
  }
  
  /**
   * Fetch event - apply caching strategies
   */
  async handleFetch(event: FetchEvent): Promise<void> {
    const { request } = event;
    const url = new URL(request.url);
    
    // Determine strategy based on request type
    let strategy: CacheStrategy;
    let cacheName: string;
    
    if (url.pathname.startsWith('/api/')) {
      // API calls - network first with cache fallback
      strategy = CacheStrategy.NETWORK_FIRST;
      cacheName = API_CACHE;
    } else if (request.destination === 'image') {
      // Images - cache first
      strategy = CacheStrategy.CACHE_FIRST;
      cacheName = IMAGE_CACHE;
    } else if (PRECACHE_ASSETS.includes(url.pathname)) {
      // Precached assets - cache first
      strategy = CacheStrategy.CACHE_FIRST;
      cacheName = CACHE_NAME;
    } else {
      // Everything else - stale while revalidate
      strategy = CacheStrategy.STALE_WHILE_REVALIDATE;
      cacheName = CACHE_NAME;
    }
    
    event.respondWith(
      this.applyStrategy(request, strategy, cacheName)
    );
  }
  
  /**
   * Apply caching strategy
   */
  async applyStrategy(
    request: Request,
    strategy: CacheStrategy,
    cacheName: string
  ): Promise<Response> {
    
    switch (strategy) {
      case CacheStrategy.CACHE_FIRST:
        return this.cacheFirst(request, cacheName);
        
      case CacheStrategy.NETWORK_FIRST:
        return this.networkFirst(request, cacheName);
        
      case CacheStrategy.STALE_WHILE_REVALIDATE:
        return this.staleWhileRevalidate(request, cacheName);
        
      case CacheStrategy.CACHE_ONLY:
        return this.cacheOnly(request, cacheName);
        
      case CacheStrategy.NETWORK_ONLY:
        return fetch(request);
        
      default:
        return fetch(request);
    }
  }
  
  /**
   * Cache first strategy
   */
  async cacheFirst(request: Request, cacheName: string): Promise<Response> {
    const cache = await caches.open(cacheName);
    const cached = await cache.match(request);
    
    if (cached) {
      return cached;
    }
    
    try {
      const response = await fetch(request);
      
      if (response.ok) {
        cache.put(request, response.clone());
      }
      
      return response;
    } catch (error) {
      throw new Error('Network request failed and no cache available');
    }
  }
  
  /**
   * Network first strategy
   */
  async networkFirst(request: Request, cacheName: string): Promise<Response> {
    const cache = await caches.open(cacheName);
    
    try {
      const response = await fetch(request);
      
      if (response.ok) {
        cache.put(request, response.clone());
      }
      
      return response;
    } catch (error) {
      const cached = await cache.match(request);
      
      if (cached) {
        return cached;
      }
      
      throw error;
    }
  }
  
  /**
   * Stale while revalidate strategy
   */
  async staleWhileRevalidate(
    request: Request,
    cacheName: string
  ): Promise<Response> {
    const cache = await caches.open(cacheName);
    const cached = await cache.match(request);
    
    const fetchPromise = fetch(request).then(response => {
      if (response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    });
    
    return cached || fetchPromise;
  }
  
  /**
   * Cache only strategy
   */
  async cacheOnly(request: Request, cacheName: string): Promise<Response> {
    const cache = await caches.open(cacheName);
    const cached = await cache.match(request);
    
    if (!cached) {
      throw new Error('No cache available for request');
    }
    
    return cached;
  }
}

// Initialize service worker
const swCache = new ServiceWorkerCache();

self.addEventListener('install', event => swCache.handleInstall(event));
self.addEventListener('activate', event => swCache.handleActivate(event));
self.addEventListener('fetch', event => swCache.handleFetch(event));
```

### 3.4 Web Vitals Targets

```typescript
// Web Vitals monitoring and targets
export interface WebVitalsTargets {
  // Core Web Vitals targets (Google's recommendations)
  LCP: {
    good: 2500,      // < 2.5s
    needsImprovement: 4000,  // 2.5s - 4s
    poor: Infinity   // > 4s
  };
  FID: {
    good: 100,       // < 100ms
    needsImprovement: 300,   // 100ms - 300ms
    poor: Infinity   // > 300ms
  };
  CLS: {
    good: 0.1,       // < 0.1
    needsImprovement: 0.25,  // 0.1 - 0.25
    poor: Infinity   // > 0.25
  };
  
  // Additional metrics
  TTFB: {
    good: 800,       // < 800ms
    needsImprovement: 1800,  // 800ms - 1.8s
    poor: Infinity   // > 1.8s
  };
  INP: {
    good: 200,       // < 200ms
    needsImprovement: 500,   // 200ms - 500ms
    poor: Infinity   // > 500ms
  };
  FCP: {
    good: 1800,      // < 1.8s
    needsImprovement: 3000,  // 1.8s - 3s
    poor: Infinity   // > 3s
  };
}

// Monitor and report Web Vitals
export class WebVitalsMonitor {
  private targets: WebVitalsTargets;
  private observer: PerformanceObserver | null = null;
  
  constructor() {
    this.targets = this.getDefaultTargets();
    this.initializeMonitoring();
  }
  
  private getDefaultTargets(): WebVitalsTargets {
    return {
      LCP: { good: 2500, needsImprovement: 4000, poor: Infinity },
      FID: { good: 100, needsImprovement: 300, poor: Infinity },
      CLS: { good: 0.1, needsImprovement: 0.25, poor: Infinity },
      TTFB: { good: 800, needsImprovement: 1800, poor: Infinity },
      INP: { good: 200, needsImprovement: 500, poor: Infinity },
      FCP: { good: 1800, needsImprovement: 3000, poor: Infinity }
    };
  }
  
  private initializeMonitoring(): void {
    // Use web-vitals library for accurate measurements
    import('web-vitals').then(({ onLCP, onFID, onCLS, onFCP, onTTFB, onINP }) => {
      onLCP(metric => this.reportMetric('LCP', metric));
      onFID(metric => this.reportMetric('FID', metric));
      onCLS(metric => this.reportMetric('CLS', metric));
      onFCP(metric => this.reportMetric('FCP', metric));
      onTTFB(metric => this.reportMetric('TTFB', metric));
      onINP(metric => this.reportMetric('INP', metric));
    });
  }
  
  private reportMetric(name: string, metric: any): void {
    const rating = this.getRating(name, metric.value);
    
    // Log to console in development
    if (import.meta.env.DEV) {
      console.log(`Web Vital: ${name}`, {
        value: metric.value,
        rating,
        delta: metric.delta,
        id: metric.id
      });
    }
    
    // Send to analytics
    this.sendToAnalytics({
      metric: name,
      value: metric.value,
      rating,
      delta: metric.delta,
      id: metric.id,
      navigationType: metric.navigationType
    });
  }
  
  private getRating(metric: string, value: number): 'good' | 'needs-improvement' | 'poor' {
    const target = this.targets[metric as keyof WebVitalsTargets];
    
    if (!target) return 'poor';
    
    if (value <= target.good) return 'good';
    if (value <= target.needsImprovement) return 'needs-improvement';
    return 'poor';
  }
  
  private sendToAnalytics(data: any): void {
    // Send to your analytics service
    if (window.gtag) {
      window.gtag('event', 'web_vitals', data);
    }
  }
}
```

## 4. Load Testing Methodology

### 4.1 Test Scenarios for TTRPG Sessions

```python
import asyncio
import aiohttp
import websockets
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class TTRPGLoadTestScenario:
    """Load test scenarios specific to TTRPG sessions"""
    
    name: str
    description: str
    num_users: int
    duration_seconds: int
    actions_per_user: List[Dict[str, Any]]

class TTRPGLoadTester:
    """Comprehensive load testing for TTRPG MCP Server"""
    
    def __init__(self, base_url: str, ws_url: str):
        self.base_url = base_url
        self.ws_url = ws_url
        self.metrics = {
            'response_times': [],
            'errors': [],
            'websocket_latencies': [],
            'throughput': []
        }
    
    async def run_scenario(self, scenario: TTRPGLoadTestScenario) -> Dict[str, Any]:
        """Run a complete load test scenario"""
        
        print(f"Running scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Users: {scenario.num_users}, Duration: {scenario.duration_seconds}s")
        
        start_time = time.time()
        
        # Create virtual users
        users = [
            self.create_virtual_user(f"user_{i}", scenario.actions_per_user)
            for i in range(scenario.num_users)
        ]
        
        # Run users concurrently
        await asyncio.gather(*users)
        
        # Calculate results
        elapsed_time = time.time() - start_time
        
        return self.generate_report(scenario, elapsed_time)
    
    async def create_virtual_user(self, 
                                 user_id: str,
                                 actions: List[Dict[str, Any]]) -> None:
        """Simulate a virtual user performing actions"""
        
        # Establish WebSocket connection
        async with websockets.connect(self.ws_url) as websocket:
            # Perform user actions
            for action in actions:
                await self.perform_action(user_id, action, websocket)
                
                # Random delay between actions (simulate thinking time)
                await asyncio.sleep(np.random.uniform(0.5, 2.0))
    
    async def perform_action(self,
                            user_id: str,
                            action: Dict[str, Any],
                            websocket) -> None:
        """Perform a single user action"""
        
        action_type = action['type']
        
        if action_type == 'upload_pdf':
            await self.test_pdf_upload(user_id, action['file_size'])
            
        elif action_type == 'search':
            await self.test_search(user_id, action['query'])
            
        elif action_type == 'dice_roll':
            await self.test_dice_roll(user_id, websocket, action['dice'])
            
        elif action_type == 'canvas_draw':
            await self.test_canvas_draw(user_id, websocket, action['points'])
            
        elif action_type == 'chat_message':
            await self.test_chat_message(user_id, websocket, action['message'])
    
    async def test_pdf_upload(self, user_id: str, file_size: int) -> None:
        """Test PDF upload and processing"""
        
        start_time = time.time()
        
        # Generate dummy PDF data
        pdf_data = b'%PDF-1.4' + b'0' * file_size
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/pdf/upload",
                    data={'file': pdf_data, 'user_id': user_id}
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        self.metrics['response_times'].append(response_time)
                    else:
                        self.metrics['errors'].append({
                            'type': 'pdf_upload',
                            'status': response.status,
                            'user': user_id
                        })
                        
            except Exception as e:
                self.metrics['errors'].append({
                    'type': 'pdf_upload',
                    'error': str(e),
                    'user': user_id
                })
    
    async def test_search(self, user_id: str, query: str) -> None:
        """Test search functionality"""
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/api/search",
                    params={'q': query, 'user_id': user_id}
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        self.metrics['response_times'].append(response_time)
                    else:
                        self.metrics['errors'].append({
                            'type': 'search',
                            'status': response.status,
                            'user': user_id
                        })
                        
            except Exception as e:
                self.metrics['errors'].append({
                    'type': 'search',
                    'error': str(e),
                    'user': user_id
                })
    
    async def test_dice_roll(self, 
                            user_id: str,
                            websocket,
                            dice: str) -> None:
        """Test dice rolling through WebSocket"""
        
        start_time = time.time()
        
        message = {
            'type': 'dice_roll',
            'user_id': user_id,
            'dice': dice,
            'timestamp': time.time()
        }
        
        await websocket.send(json.dumps(message))
        
        # Wait for response
        response = await websocket.recv()
        
        latency = (time.time() - start_time) * 1000
        self.metrics['websocket_latencies'].append(latency)
    
    async def test_canvas_draw(self,
                              user_id: str,
                              websocket,
                              points: List[Dict]) -> None:
        """Test collaborative canvas drawing"""
        
        start_time = time.time()
        
        message = {
            'type': 'canvas_draw',
            'user_id': user_id,
            'points': points,
            'timestamp': time.time()
        }
        
        await websocket.send(json.dumps(message))
        
        # Wait for broadcast confirmation
        response = await websocket.recv()
        
        latency = (time.time() - start_time) * 1000
        self.metrics['websocket_latencies'].append(latency)
    
    async def test_chat_message(self,
                               user_id: str,
                               websocket,
                               message_text: str) -> None:
        """Test chat messaging"""
        
        start_time = time.time()
        
        message = {
            'type': 'chat',
            'user_id': user_id,
            'message': message_text,
            'timestamp': time.time()
        }
        
        await websocket.send(json.dumps(message))
        
        # Wait for broadcast
        response = await websocket.recv()
        
        latency = (time.time() - start_time) * 1000
        self.metrics['websocket_latencies'].append(latency)
    
    def generate_report(self, 
                       scenario: TTRPGLoadTestScenario,
                       elapsed_time: float) -> Dict[str, Any]:
        """Generate load test report"""
        
        response_times = self.metrics['response_times']
        ws_latencies = self.metrics['websocket_latencies']
        
        return {
            'scenario': scenario.name,
            'duration': elapsed_time,
            'users': scenario.num_users,
            'http_metrics': {
                'total_requests': len(response_times),
                'avg_response_time': np.mean(response_times) if response_times else 0,
                'p50_response_time': np.percentile(response_times, 50) if response_times else 0,
                'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
                'p99_response_time': np.percentile(response_times, 99) if response_times else 0,
                'throughput': len(response_times) / elapsed_time if elapsed_time > 0 else 0
            },
            'websocket_metrics': {
                'total_messages': len(ws_latencies),
                'avg_latency': np.mean(ws_latencies) if ws_latencies else 0,
                'p50_latency': np.percentile(ws_latencies, 50) if ws_latencies else 0,
                'p95_latency': np.percentile(ws_latencies, 95) if ws_latencies else 0,
                'p99_latency': np.percentile(ws_latencies, 99) if ws_latencies else 0,
                'message_rate': len(ws_latencies) / elapsed_time if elapsed_time > 0 else 0
            },
            'errors': {
                'total': len(self.metrics['errors']),
                'error_rate': len(self.metrics['errors']) / (len(response_times) + len(ws_latencies))
                            if (len(response_times) + len(ws_latencies)) > 0 else 0,
                'errors_by_type': self._group_errors_by_type()
            }
        }
    
    def _group_errors_by_type(self) -> Dict[str, int]:
        """Group errors by type"""
        error_counts = {}
        for error in self.metrics['errors']:
            error_type = error.get('type', 'unknown')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts

# Define test scenarios
scenarios = [
    TTRPGLoadTestScenario(
        name="Small Session",
        description="5 players in a typical D&D session",
        num_users=5,
        duration_seconds=300,
        actions_per_user=[
            {'type': 'upload_pdf', 'file_size': 5000000},  # 5MB PDF
            {'type': 'search', 'query': 'fireball spell'},
            {'type': 'dice_roll', 'dice': '1d20+5'},
            {'type': 'chat_message', 'message': 'I attack the goblin!'},
            {'type': 'canvas_draw', 'points': [{'x': 100, 'y': 100}]}
        ]
    ),
    
    TTRPGLoadTestScenario(
        name="Large Campaign",
        description="20 players across multiple parties",
        num_users=20,
        duration_seconds=600,
        actions_per_user=[
            {'type': 'search', 'query': 'monster manual'},
            {'type': 'dice_roll', 'dice': '3d6'},
            {'type': 'chat_message', 'message': 'Rolling for initiative'},
            {'type': 'canvas_draw', 'points': [{'x': 200, 'y': 200}]}
        ]
    ),
    
    TTRPGLoadTestScenario(
        name="Convention Event",
        description="100 concurrent users at a gaming convention",
        num_users=100,
        duration_seconds=1800,
        actions_per_user=[
            {'type': 'search', 'query': 'character creation'},
            {'type': 'dice_roll', 'dice': '4d6'},
            {'type': 'chat_message', 'message': 'New character ready!'}
        ]
    )
]

# Run load tests
async def run_all_scenarios():
    tester = TTRPGLoadTester(
        base_url="http://localhost:8000",
        ws_url="ws://localhost:8000/ws"
    )
    
    for scenario in scenarios:
        report = await tester.run_scenario(scenario)
        print(json.dumps(report, indent=2))
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(run_all_scenarios())
```

### 4.2 Stress Testing Tools and Scripts

```python
# Stress testing script with gradual load increase
import locust
from locust import HttpUser, task, between, events
import websocket
import json
import time

class TTRPGUser(HttpUser):
    """Simulated TTRPG user for stress testing"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when user starts"""
        # Login or authenticate
        self.client.post("/api/auth/login", json={
            "username": f"user_{self.environment.runner.user_count}",
            "password": "test123"
        })
        
        # Establish WebSocket connection
        self.ws = websocket.WebSocket()
        self.ws.connect("ws://localhost:8000/ws")
    
    @task(3)
    def search_content(self):
        """Search for game content"""
        queries = [
            "dragon stats",
            "magic items",
            "spell components",
            "character backgrounds",
            "dungeon traps"
        ]
        
        with self.client.get(
            "/api/search",
            params={"q": random.choice(queries)},
            catch_response=True
        ) as response:
            if response.elapsed.total_seconds() > 1:
                response.failure("Search took too long")
    
    @task(2)
    def roll_dice(self):
        """Roll dice through WebSocket"""
        dice_rolls = ["1d20", "2d6", "3d8+2", "4d6", "1d100"]
        
        message = {
            "type": "dice_roll",
            "dice": random.choice(dice_rolls)
        }
        
        start_time = time.time()
        self.ws.send(json.dumps(message))
        response = self.ws.recv()
        latency = time.time() - start_time
        
        events.request_success.fire(
            request_type="WebSocket",
            name="dice_roll",
            response_time=latency * 1000,
            response_length=len(response)
        )
    
    @task(1)
    def upload_character_sheet(self):
        """Upload a character sheet PDF"""
        with open("test_character_sheet.pdf", "rb") as f:
            files = {"file": f}
            self.client.post("/api/pdf/upload", files=files)
    
    @task(4)
    def view_game_content(self):
        """View game content pages"""
        pages = [
            "/api/content/rules",
            "/api/content/monsters",
            "/api/content/items",
            "/api/content/spells"
        ]
        
        self.client.get(random.choice(pages))
    
    def on_stop(self):
        """Called when user stops"""
        if hasattr(self, 'ws'):
            self.ws.close()

# Custom load shape for gradual stress increase
class StressTestShape(LoadTestShape):
    """Gradually increase load to find breaking point"""
    
    time_limit = 600  # 10 minutes
    spawn_rate = 2
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < self.time_limit:
            # Increase users every 30 seconds
            user_count = (run_time // 30) * 10
            return (user_count, self.spawn_rate)
        
        return None

# Run with: locust -f stress_test.py --host=http://localhost:8000
```

### 4.3 Performance Monitoring Setup

```python
# Performance monitoring and alerting setup
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import asyncio
from typing import Dict, Any
import logging

class PerformanceMonitor:
    """Comprehensive performance monitoring for TTRPG MCP Server"""
    
    def __init__(self):
        # Prometheus metrics
        self.request_count = Counter(
            'ttrpg_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'ttrpg_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.websocket_connections = Gauge(
            'ttrpg_websocket_connections',
            'Number of active WebSocket connections'
        )
        
        self.pdf_processing_time = Histogram(
            'ttrpg_pdf_processing_seconds',
            'PDF processing time in seconds',
            ['complexity']
        )
        
        self.search_latency = Histogram(
            'ttrpg_search_latency_seconds',
            'Search query latency in seconds',
            ['index_type']
        )
        
        self.memory_usage = Gauge(
            'ttrpg_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'ttrpg_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Alert thresholds
        self.alert_thresholds = {
            'response_time_p95': 1.0,  # 1 second
            'error_rate': 0.01,  # 1%
            'memory_usage': 0.8,  # 80% of available
            'cpu_usage': 0.7,  # 70%
            'websocket_connections': 1000
        }
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start Prometheus metrics server and system monitoring"""
        
        # Start Prometheus metrics server
        start_http_server(9090)
        
        # Start system metrics collection
        asyncio.create_task(self.collect_system_metrics())
        
        # Start alert checking
        asyncio.create_task(self.check_alerts())
    
    async def collect_system_metrics(self):
        """Collect system metrics periodically"""
        
        while True:
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Check for memory pressure
            if memory.percent > self.alert_thresholds['memory_usage'] * 100:
                await self.send_alert(
                    'HIGH_MEMORY_USAGE',
                    f'Memory usage at {memory.percent}%'
                )
            
            # Check for CPU pressure
            if cpu_percent > self.alert_thresholds['cpu_usage'] * 100:
                await self.send_alert(
                    'HIGH_CPU_USAGE',
                    f'CPU usage at {cpu_percent}%'
                )
            
            await asyncio.sleep(10)  # Collect every 10 seconds
    
    async def check_alerts(self):
        """Check metrics against alert thresholds"""
        
        while True:
            # Check WebSocket connections
            if self.websocket_connections._value > self.alert_thresholds['websocket_connections']:
                await self.send_alert(
                    'HIGH_WEBSOCKET_CONNECTIONS',
                    f'WebSocket connections: {self.websocket_connections._value}'
                )
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def send_alert(self, alert_type: str, message: str):
        """Send performance alert"""
        
        logging.warning(f"PERFORMANCE ALERT [{alert_type}]: {message}")
        
        # Send to monitoring service (e.g., PagerDuty, Slack)
        # await notify_ops_team(alert_type, message)
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_pdf_processing(self, complexity: str, duration: float):
        """Record PDF processing metrics"""
        
        self.pdf_processing_time.labels(
            complexity=complexity
        ).observe(duration)
    
    def record_search(self, index_type: str, duration: float):
        """Record search metrics"""
        
        self.search_latency.labels(
            index_type=index_type
        ).observe(duration)
    
    def update_websocket_count(self, delta: int):
        """Update WebSocket connection count"""
        
        if delta > 0:
            self.websocket_connections.inc(delta)
        else:
            self.websocket_connections.dec(abs(delta))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        
        return {
            'websocket_connections': self.websocket_connections._value,
            'memory_usage_bytes': self.memory_usage._value,
            'cpu_usage_percent': self.cpu_usage._value
        }

# Initialize global monitor
monitor = PerformanceMonitor()
```

## 5. Scalability Planning

### 5.1 Horizontal Scaling Approach

```python
# Horizontal scaling configuration
from typing import Dict, Any, List
import consul
import asyncio

class HorizontalScalingManager:
    """Manage horizontal scaling for TTRPG MCP Server"""
    
    def __init__(self, consul_host: str = 'localhost', consul_port: int = 8500):
        self.consul = consul.Consul(host=consul_host, port=consul_port)
        self.scaling_config = {
            'min_instances': 2,
            'max_instances': 20,
            'target_cpu': 70,
            'target_memory': 80,
            'scale_up_threshold': 0.8,
            'scale_down_threshold': 0.3,
            'cooldown_period': 300  # 5 minutes
        }
        
        self.last_scale_time = 0
    
    async def auto_scale(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Determine if scaling is needed based on metrics"""
        
        current_time = asyncio.get_event_loop().time()
        
        # Check cooldown period
        if current_time - self.last_scale_time < self.scaling_config['cooldown_period']:
            return {'action': 'none', 'reason': 'cooldown'}
        
        # Get current instance count
        instances = await self.get_active_instances()
        current_count = len(instances)
        
        # Calculate scaling decision
        cpu_usage = current_metrics.get('cpu_usage', 0)
        memory_usage = current_metrics.get('memory_usage', 0)
        
        # Scale up conditions
        if (cpu_usage > self.scaling_config['target_cpu'] * self.scaling_config['scale_up_threshold'] or
            memory_usage > self.scaling_config['target_memory'] * self.scaling_config['scale_up_threshold']):
            
            if current_count < self.scaling_config['max_instances']:
                new_count = min(
                    current_count + 2,
                    self.scaling_config['max_instances']
                )
                
                await self.scale_to(new_count)
                self.last_scale_time = current_time
                
                return {
                    'action': 'scale_up',
                    'from': current_count,
                    'to': new_count,
                    'reason': f'High load - CPU: {cpu_usage}%, Memory: {memory_usage}%'
                }
        
        # Scale down conditions
        if (cpu_usage < self.scaling_config['target_cpu'] * self.scaling_config['scale_down_threshold'] and
            memory_usage < self.scaling_config['target_memory'] * self.scaling_config['scale_down_threshold']):
            
            if current_count > self.scaling_config['min_instances']:
                new_count = max(
                    current_count - 1,
                    self.scaling_config['min_instances']
                )
                
                await self.scale_to(new_count)
                self.last_scale_time = current_time
                
                return {
                    'action': 'scale_down',
                    'from': current_count,
                    'to': new_count,
                    'reason': f'Low load - CPU: {cpu_usage}%, Memory: {memory_usage}%'
                }
        
        return {'action': 'none', 'reason': 'within thresholds'}
    
    async def get_active_instances(self) -> List[Dict[str, Any]]:
        """Get list of active service instances from Consul"""
        
        _, services = self.consul.health.service('ttrpg-mcp-server', passing=True)
        
        instances = []
        for service in services:
            instances.append({
                'id': service['Service']['ID'],
                'address': service['Service']['Address'],
                'port': service['Service']['Port'],
                'tags': service['Service']['Tags']
            })
        
        return instances
    
    async def scale_to(self, target_count: int):
        """Scale to target number of instances"""
        
        # In production, this would trigger:
        # - Kubernetes HPA
        # - AWS Auto Scaling
        # - Docker Swarm scaling
        # - etc.
        
        print(f"Scaling to {target_count} instances")
        
        # Example: Kubernetes scaling
        # kubectl.scale('deployment/ttrpg-mcp-server', replicas=target_count)
```

### 5.2 Database Sharding Strategies

```python
# Database sharding for ChromaDB collections
import hashlib
from typing import Dict, Any, List, Optional

class ChromaDBShardManager:
    """Manage sharded ChromaDB collections for scalability"""
    
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.shard_connections = {}
        self.shard_mapping = {}
        
        # Initialize shards
        self._initialize_shards()
    
    def _initialize_shards(self):
        """Initialize ChromaDB shards"""
        
        for shard_id in range(self.num_shards):
            # Each shard is a separate ChromaDB instance
            persist_dir = f"/data/chromadb/shard_{shard_id}"
            
            client = chromadb.Client(Settings(
                persist_directory=persist_dir,
                anonymized_telemetry=False
            ))
            
            self.shard_connections[shard_id] = client
    
    def get_shard_id(self, key: str) -> int:
        """Determine shard ID for a given key"""
        
        hash_value = hashlib.md5(key.encode()).hexdigest()
        return int(hash_value, 16) % self.num_shards
    
    async def insert_vector(self,
                          collection_name: str,
                          document_id: str,
                          embedding: List[float],
                          metadata: Dict[str, Any]) -> bool:
        """Insert vector into appropriate shard"""
        
        shard_id = self.get_shard_id(document_id)
        client = self.shard_connections[shard_id]
        
        collection = client.get_or_create_collection(
            name=f"{collection_name}_shard_{shard_id}"
        )
        
        collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[document_id]
        )
        
        # Update mapping
        self.shard_mapping[document_id] = shard_id
        
        return True
    
    async def search_all_shards(self,
                               collection_name: str,
                               query_embedding: List[float],
                               n_results: int = 10) -> List[Dict[str, Any]]:
        """Search across all shards and merge results"""
        
        all_results = []
        
        # Query each shard in parallel
        tasks = []
        for shard_id, client in self.shard_connections.items():
            task = self._search_shard(
                client,
                f"{collection_name}_shard_{shard_id}",
                query_embedding,
                n_results * 2  # Get more results for merging
            )
            tasks.append(task)
        
        shard_results = await asyncio.gather(*tasks)
        
        # Merge and sort results by distance
        for results in shard_results:
            if results:
                all_results.extend(results)
        
        # Sort by distance and return top N
        all_results.sort(key=lambda x: x['distance'])
        
        return all_results[:n_results]
    
    async def _search_shard(self,
                          client,
                          collection_name: str,
                          query_embedding: List[float],
                          n_results: int) -> List[Dict[str, Any]]:
        """Search a single shard"""
        
        try:
            collection = client.get_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching shard {collection_name}: {e}")
            return []
    
    def rebalance_shards(self):
        """Rebalance data across shards if needed"""
        
        # Count documents per shard
        shard_counts = {}
        for doc_id, shard_id in self.shard_mapping.items():
            shard_counts[shard_id] = shard_counts.get(shard_id, 0) + 1
        
        # Check if rebalancing is needed
        avg_count = len(self.shard_mapping) / self.num_shards
        max_imbalance = max(shard_counts.values()) - min(shard_counts.values())
        
        if max_imbalance > avg_count * 0.2:  # 20% imbalance threshold
            print(f"Rebalancing needed: imbalance = {max_imbalance}")
            # Implement rebalancing logic
```

### 5.3 CDN Integration

```typescript
// CDN configuration for static assets
export const cdnConfig = {
  // CloudFlare CDN configuration
  cloudflare: {
    zoneId: process.env.CLOUDFLARE_ZONE_ID,
    apiToken: process.env.CLOUDFLARE_API_TOKEN,
    
    // Cache rules
    cacheRules: [
      {
        pattern: /\.(js|css|woff2?|ttf|eot|svg|ico)$/,
        maxAge: 31536000,  // 1 year
        sMaxAge: 31536000,
        staleWhileRevalidate: 86400
      },
      {
        pattern: /\.(jpg|jpeg|png|gif|webp|avif)$/,
        maxAge: 86400,  // 1 day
        sMaxAge: 604800,  // 1 week
        transform: {
          width: 'auto',
          quality: 85,
          format: 'auto'
        }
      },
      {
        pattern: /\.pdf$/,
        maxAge: 3600,  // 1 hour
        sMaxAge: 86400  // 1 day
      }
    ],
    
    // Purge configuration
    purgePatterns: [
      '/api/*',  // Don't cache API responses
      '/ws/*'    // Don't cache WebSocket endpoints
    ]
  },
  
  // Asset optimization
  optimization: {
    images: {
      formats: ['webp', 'avif'],
      sizes: [320, 640, 1280, 1920],
      quality: {
        webp: 85,
        avif: 80,
        jpeg: 85
      }
    },
    
    scripts: {
      minify: true,
      splitChunks: true,
      compression: ['gzip', 'brotli']
    },
    
    styles: {
      minify: true,
      purgeCss: true,
      criticalCss: true
    }
  },
  
  // Edge workers for dynamic content
  edgeWorkers: {
    // Geolocation-based routing
    geoRouting: `
      addEventListener('fetch', event => {
        event.respondWith(handleRequest(event.request))
      })
      
      async function handleRequest(request) {
        const country = request.headers.get('CF-IPCountry')
        
        // Route to nearest server
        const serverMap = {
          'US': 'https://us.ttrpg-mcp.com',
          'EU': 'https://eu.ttrpg-mcp.com',
          'AS': 'https://asia.ttrpg-mcp.com'
        }
        
        const region = getRegion(country)
        const origin = serverMap[region] || serverMap['US']
        
        // Modify request
        const modifiedRequest = new Request(
          origin + new URL(request.url).pathname,
          request
        )
        
        return fetch(modifiedRequest)
      }
    `,
    
    // A/B testing at the edge
    abTesting: `
      addEventListener('fetch', event => {
        event.respondWith(handleRequest(event.request))
      })
      
      async function handleRequest(request) {
        const cookie = request.headers.get('Cookie')
        let variant = 'control'
        
        if (!cookie || !cookie.includes('ab_variant')) {
          // Assign variant
          variant = Math.random() < 0.5 ? 'control' : 'test'
          
          // Set cookie
          const response = await fetch(request)
          response.headers.append(
            'Set-Cookie',
            \`ab_variant=\${variant}; Max-Age=2592000; Path=/\`
          )
          
          return response
        }
        
        // Route based on variant
        if (variant === 'test') {
          // Modify request for test variant
        }
        
        return fetch(request)
      }
    `
  }
};
```

## 6. Performance Profiling Setup

```python
# Comprehensive profiling setup
import cProfile
import pstats
import io
from memory_profiler import profile
from line_profiler import LineProfiler
import tracemalloc
from typing import Callable, Any
import functools

class PerformanceProfiler:
    """Advanced profiling tools for performance analysis"""
    
    def __init__(self):
        self.cpu_profiler = cProfile.Profile()
        self.line_profiler = LineProfiler()
        self.memory_snapshots = []
        
        # Start memory tracing
        tracemalloc.start()
    
    def profile_cpu(self, func: Callable) -> Callable:
        """Decorator for CPU profiling"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.cpu_profiler.enable()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.cpu_profiler.disable()
                
                # Print stats
                s = io.StringIO()
                ps = pstats.Stats(self.cpu_profiler, stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats(20)
                
                print(f"\nCPU Profile for {func.__name__}:")
                print(s.getvalue())
        
        return wrapper
    
    @profile  # memory_profiler decorator
    def profile_memory(self, func: Callable) -> Any:
        """Profile memory usage of a function"""
        
        # Take snapshot before
        snapshot1 = tracemalloc.take_snapshot()
        
        result = func()
        
        # Take snapshot after
        snapshot2 = tracemalloc.take_snapshot()
        
        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        print(f"\nMemory Profile for {func.__name__}:")
        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
            print(stat)
        
        return result
    
    def profile_line(self, func: Callable) -> Callable:
        """Line-by-line profiling"""
        
        # Add function to line profiler
        self.line_profiler.add_function(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.line_profiler.enable()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.line_profiler.disable()
                self.line_profiler.print_stats()
        
        return wrapper
    
    def analyze_bottlenecks(self, trace_file: str):
        """Analyze performance bottlenecks from trace file"""
        
        stats = pstats.Stats(trace_file)
        stats.sort_stats('cumulative')
        
        # Find slowest functions
        print("\n=== Top 20 Slowest Functions ===")
        stats.print_stats(20)
        
        # Find functions called most often
        print("\n=== Most Called Functions ===")
        stats.sort_stats('calls')
        stats.print_stats(20)
        
        # Find functions with highest cumulative time
        print("\n=== Highest Cumulative Time ===")
        stats.sort_stats('cumulative')
        stats.print_stats(20)
    
    def memory_leak_detection(self):
        """Detect potential memory leaks"""
        
        snapshot = tracemalloc.take_snapshot()
        self.memory_snapshots.append(snapshot)
        
        if len(self.memory_snapshots) > 1:
            # Compare with previous snapshot
            previous = self.memory_snapshots[-2]
            current = self.memory_snapshots[-1]
            
            top_stats = current.compare_to(previous, 'lineno')
            
            # Look for growing allocations
            growing_allocations = []
            for stat in top_stats:
                if stat.size_diff > 1024 * 1024:  # > 1MB growth
                    growing_allocations.append(stat)
            
            if growing_allocations:
                print("\n⚠️  Potential Memory Leaks Detected:")
                for stat in growing_allocations[:10]:
                    print(f"  {stat}")
        
        # Keep only last 10 snapshots
        if len(self.memory_snapshots) > 10:
            self.memory_snapshots = self.memory_snapshots[-10:]

# Usage example
profiler = PerformanceProfiler()

@profiler.profile_cpu
@profiler.profile_line
async def process_heavy_operation(data):
    """Example of profiled operation"""
    # Simulate heavy processing
    result = []
    for item in data:
        processed = await complex_transformation(item)
        result.append(processed)
    return result
```

## Conclusion

This comprehensive performance requirements document provides:

1. **Clear Performance Targets**: Specific benchmarks for all critical operations
2. **Optimization Strategies**: Detailed implementations for backend and frontend optimization
3. **Load Testing Framework**: Complete testing scenarios and tools for TTRPG-specific workloads
4. **Monitoring Infrastructure**: Comprehensive monitoring and alerting setup
5. **Scalability Blueprint**: Clear path for horizontal scaling and distributed architecture

The implementation focuses on Python best practices with type hints, async patterns, and production-ready error handling. All components are designed to work together to deliver a high-performance TTRPG collaboration platform that can scale from small gaming groups to large conventions.

Key performance targets to maintain:
- PDF processing: <30s for complex documents
- Search latency: <200ms P95
- WebSocket message delivery: <50ms P95
- Support for 100+ concurrent users per session
- 99.9% uptime SLA

Regular performance testing and monitoring should be conducted to ensure these targets are consistently met as the system evolves.