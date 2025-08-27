"""High-performance context serialization and compression system."""

import asyncio
import json
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import msgpack
from pydantic import BaseModel

from .models import CompressionType, Context

logger = logging.getLogger(__name__)


class SerializationFormat(str):
    """Supported serialization formats."""
    
    JSON = "json"
    MSGPACK = "msgpack"
    PICKLE = "pickle"


class ContextSerializer:
    """High-performance context serialization with multiple format support."""
    
    def __init__(
        self,
        default_format: SerializationFormat = SerializationFormat.MSGPACK,
        enable_async: bool = True,
        thread_pool_size: int = 4,
    ):
        self.default_format = default_format
        self.enable_async = enable_async
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size) if enable_async else None
        
        # Performance tracking
        self._serialization_stats = {
            "json": {"count": 0, "total_time": 0.0, "total_size": 0},
            "msgpack": {"count": 0, "total_time": 0.0, "total_size": 0},
            "pickle": {"count": 0, "total_time": 0.0, "total_size": 0},
        }
        
        logger.info(
            "Context serializer initialized",
            default_format=default_format,
            async_enabled=enable_async,
            thread_pool_size=thread_pool_size,
        )
    
    def serialize(
        self,
        obj: Union[Context, BaseModel, Dict[str, Any]],
        format_type: Optional[SerializationFormat] = None,
    ) -> str:
        """Serialize object to string format."""
        format_type = format_type or self.default_format
        start_time = time.time()
        
        try:
            # Convert to serializable format
            if isinstance(obj, BaseModel):
                data = obj.dict()
            elif isinstance(obj, dict):
                data = obj
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")
            
            # Serialize based on format
            if format_type == SerializationFormat.JSON:
                result = json.dumps(data, default=self._json_serializer, ensure_ascii=False)
            elif format_type == SerializationFormat.MSGPACK:
                packed = msgpack.packb(data, default=self._msgpack_serializer, use_bin_type=True)
                result = packed.decode('latin1')  # Store as string
            elif format_type == SerializationFormat.PICKLE:
                pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                result = pickled.decode('latin1')  # Store as string
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_serialization(format_type, execution_time, len(result))
            
            logger.debug(
                "Object serialized",
                format=format_type,
                size=len(result),
                execution_time=execution_time,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
    
    def deserialize(
        self,
        data: str,
        target_type: Type[Union[Context, BaseModel]] = Context,
        format_type: Optional[SerializationFormat] = None,
    ) -> Union[Context, BaseModel, Dict[str, Any]]:
        """Deserialize string data to object."""
        format_type = format_type or self.default_format
        start_time = time.time()
        
        try:
            # Deserialize based on format
            if format_type == SerializationFormat.JSON:
                obj_data = json.loads(data)
            elif format_type == SerializationFormat.MSGPACK:
                packed = data.encode('latin1')
                obj_data = msgpack.unpackb(packed, raw=False, strict_map_key=False)
            elif format_type == SerializationFormat.PICKLE:
                pickled = data.encode('latin1')
                obj_data = pickle.loads(pickled)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Convert to target type if specified
            if target_type and target_type != dict:
                if issubclass(target_type, BaseModel):
                    result = target_type(**obj_data)
                else:
                    result = target_type(obj_data)
            else:
                result = obj_data
            
            # Track performance
            execution_time = time.time() - start_time
            
            logger.debug(
                "Object deserialized",
                format=format_type,
                target_type=target_type.__name__ if target_type else "dict",
                execution_time=execution_time,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise
    
    async def serialize_async(
        self,
        obj: Union[Context, BaseModel, Dict[str, Any]],
        format_type: Optional[SerializationFormat] = None,
    ) -> str:
        """Asynchronously serialize object."""
        if not self.enable_async or not self.thread_pool:
            return self.serialize(obj, format_type)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool, self.serialize, obj, format_type
        )
    
    async def deserialize_async(
        self,
        data: str,
        target_type: Type[Union[Context, BaseModel]] = Context,
        format_type: Optional[SerializationFormat] = None,
    ) -> Union[Context, BaseModel, Dict[str, Any]]:
        """Asynchronously deserialize data."""
        if not self.enable_async or not self.thread_pool:
            return self.deserialize(data, target_type, format_type)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool, self.deserialize, data, target_type, format_type
        )
    
    def serialize_batch(
        self,
        objects: List[Union[Context, BaseModel, Dict[str, Any]]],
        format_type: Optional[SerializationFormat] = None,
    ) -> List[str]:
        """Serialize multiple objects efficiently."""
        format_type = format_type or self.default_format
        start_time = time.time()
        
        try:
            results = []
            for obj in objects:
                serialized = self.serialize(obj, format_type)
                results.append(serialized)
            
            execution_time = time.time() - start_time
            logger.debug(
                "Batch serialization completed",
                count=len(objects),
                format=format_type,
                execution_time=execution_time,
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Batch serialization failed: {e}")
            raise
    
    async def serialize_batch_async(
        self,
        objects: List[Union[Context, BaseModel, Dict[str, Any]]],
        format_type: Optional[SerializationFormat] = None,
        batch_size: int = 10,
    ) -> List[str]:
        """Asynchronously serialize multiple objects in batches."""
        if not self.enable_async or not self.thread_pool:
            return self.serialize_batch(objects, format_type)
        
        results = []
        
        # Process in batches to avoid overwhelming the thread pool
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            batch_tasks = [
                self.serialize_async(obj, format_type) for obj in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    def _json_serializer(self, obj) -> Any:
        """Custom JSON serializer for complex types."""
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # Custom objects
            return obj.__dict__
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        else:
            return str(obj)
    
    def _msgpack_serializer(self, obj) -> Any:
        """Custom msgpack serializer for complex types."""
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return obj
        else:
            return str(obj)
    
    def _track_serialization(self, format_type: str, execution_time: float, size: int) -> None:
        """Track serialization performance metrics."""
        if format_type in self._serialization_stats:
            stats = self._serialization_stats[format_type]
            stats["count"] += 1
            stats["total_time"] += execution_time
            stats["total_size"] += size
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get serialization performance statistics."""
        stats = {}
        for format_type, data in self._serialization_stats.items():
            if data["count"] > 0:
                stats[format_type] = {
                    "count": data["count"],
                    "avg_time": data["total_time"] / data["count"],
                    "avg_size": data["total_size"] / data["count"],
                    "total_time": data["total_time"],
                    "total_size": data["total_size"],
                }
            else:
                stats[format_type] = data
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        logger.info("Context serializer cleaned up")


class ContextCompressor:
    """High-performance context compression with multiple algorithm support."""
    
    def __init__(
        self,
        enable_async: bool = True,
        thread_pool_size: int = 4,
        compression_level: Dict[CompressionType, int] = None,
    ):
        self.enable_async = enable_async
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size) if enable_async else None
        
        # Default compression levels for optimal performance/size tradeoff
        self.compression_levels = compression_level or {
            CompressionType.GZIP: 6,
            CompressionType.LZ4: 1,
            CompressionType.ZSTD: 3,
            CompressionType.BROTLI: 6,
        }
        
        # Performance tracking
        self._compression_stats = {
            "gzip": {"count": 0, "total_time": 0.0, "total_saved": 0},
            "lz4": {"count": 0, "total_time": 0.0, "total_saved": 0},
            "zstd": {"count": 0, "total_time": 0.0, "total_saved": 0},
            "brotli": {"count": 0, "total_time": 0.0, "total_saved": 0},
        }
        
        # Import compression libraries
        self._import_compression_libs()
        
        logger.info(
            "Context compressor initialized",
            async_enabled=enable_async,
            thread_pool_size=thread_pool_size,
            available_algorithms=list(self.available_algorithms),
        )
    
    def _import_compression_libs(self) -> None:
        """Import available compression libraries."""
        self.available_algorithms = set()
        
        # Standard library compression
        try:
            import gzip
            self.gzip = gzip
            self.available_algorithms.add(CompressionType.GZIP)
        except ImportError:
            logger.warning("gzip not available")
        
        # LZ4 compression
        try:
            import lz4.frame
            self.lz4 = lz4.frame
            self.available_algorithms.add(CompressionType.LZ4)
        except ImportError:
            logger.warning("lz4 not available")
        
        # Zstandard compression (recommended)
        try:
            import zstandard as zstd
            self.zstd = zstd
            self.available_algorithms.add(CompressionType.ZSTD)
        except ImportError:
            logger.warning("zstandard not available")
        
        # Brotli compression
        try:
            import brotli
            self.brotli = brotli
            self.available_algorithms.add(CompressionType.BROTLI)
        except ImportError:
            logger.warning("brotli not available")
    
    def compress(
        self,
        data: Union[str, bytes],
        compression_type: CompressionType = CompressionType.ZSTD,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress data using specified algorithm."""
        start_time = time.time()
        
        # Convert to bytes if string
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        original_size = len(data_bytes)
        
        # Skip compression if data is too small
        if original_size < 100:
            return data_bytes, {
                "compressed": False,
                "compression_type": "none",
                "original_size": original_size,
                "compressed_size": original_size,
                "compression_ratio": 1.0,
                "execution_time": 0.0,
            }
        
        try:
            # Compress based on algorithm
            if compression_type == CompressionType.GZIP and CompressionType.GZIP in self.available_algorithms:
                compressed_data = self.gzip.compress(
                    data_bytes, compresslevel=self.compression_levels[CompressionType.GZIP]
                )
            elif compression_type == CompressionType.LZ4 and CompressionType.LZ4 in self.available_algorithms:
                compressed_data = self.lz4.compress(
                    data_bytes, compression_level=self.compression_levels[CompressionType.LZ4]
                )
            elif compression_type == CompressionType.ZSTD and CompressionType.ZSTD in self.available_algorithms:
                compressor = self.zstd.ZstdCompressor(level=self.compression_levels[CompressionType.ZSTD])
                compressed_data = compressor.compress(data_bytes)
            elif compression_type == CompressionType.BROTLI and CompressionType.BROTLI in self.available_algorithms:
                compressed_data = self.brotli.compress(
                    data_bytes, quality=self.compression_levels[CompressionType.BROTLI]
                )
            else:
                # Fallback to no compression
                logger.warning(f"Compression type {compression_type} not available, using no compression")
                compressed_data = data_bytes
                compression_type = CompressionType.NONE
            
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            # Only use compressed version if it's significantly smaller
            if compression_ratio < 1.1 and compression_type != CompressionType.NONE:
                compressed_data = data_bytes
                compression_type = CompressionType.NONE
                compressed_size = original_size
                compression_ratio = 1.0
            
            execution_time = time.time() - start_time
            
            # Track performance
            self._track_compression(
                compression_type.value, execution_time, original_size - compressed_size
            )
            
            stats = {
                "compressed": compression_type != CompressionType.NONE,
                "compression_type": compression_type.value,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "execution_time": execution_time,
            }
            
            logger.debug(
                "Data compressed",
                compression_type=compression_type.value,
                original_size=original_size,
                compressed_size=compressed_size,
                ratio=f"{compression_ratio:.2f}x",
                execution_time=execution_time,
            )
            
            return compressed_data, stats
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Return uncompressed data as fallback
            return data_bytes, {
                "compressed": False,
                "compression_type": "none",
                "original_size": original_size,
                "compressed_size": original_size,
                "compression_ratio": 1.0,
                "execution_time": time.time() - start_time,
                "error": str(e),
            }
    
    def decompress(
        self,
        compressed_data: bytes,
        compression_type: CompressionType,
    ) -> str:
        """Decompress data using specified algorithm."""
        start_time = time.time()
        
        try:
            if compression_type == CompressionType.NONE:
                decompressed_bytes = compressed_data
            elif compression_type == CompressionType.GZIP:
                decompressed_bytes = self.gzip.decompress(compressed_data)
            elif compression_type == CompressionType.LZ4:
                decompressed_bytes = self.lz4.decompress(compressed_data)
            elif compression_type == CompressionType.ZSTD:
                decompressor = self.zstd.ZstdDecompressor()
                decompressed_bytes = decompressor.decompress(compressed_data)
            elif compression_type == CompressionType.BROTLI:
                decompressed_bytes = self.brotli.decompress(compressed_data)
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")
            
            # Convert to string
            result = decompressed_bytes.decode('utf-8')
            
            execution_time = time.time() - start_time
            
            logger.debug(
                "Data decompressed",
                compression_type=compression_type.value,
                compressed_size=len(compressed_data),
                decompressed_size=len(decompressed_bytes),
                execution_time=execution_time,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
    
    async def compress_async(
        self,
        data: Union[str, bytes],
        compression_type: CompressionType = CompressionType.ZSTD,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Asynchronously compress data."""
        if not self.enable_async or not self.thread_pool:
            return self.compress(data, compression_type)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool, self.compress, data, compression_type
        )
    
    async def decompress_async(
        self,
        compressed_data: bytes,
        compression_type: CompressionType,
    ) -> str:
        """Asynchronously decompress data."""
        if not self.enable_async or not self.thread_pool:
            return self.decompress(compressed_data, compression_type)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool, self.decompress, compressed_data, compression_type
        )
    
    def get_best_compression(self, data: Union[str, bytes]) -> CompressionType:
        """Determine the best compression algorithm for given data."""
        if len(data) < 100:
            return CompressionType.NONE
        
        # Test different algorithms and pick the best ratio
        best_algorithm = CompressionType.NONE
        best_ratio = 1.0
        
        for algorithm in self.available_algorithms:
            try:
                _, stats = self.compress(data, algorithm)
                if stats["compression_ratio"] > best_ratio:
                    best_ratio = stats["compression_ratio"]
                    best_algorithm = algorithm
            except Exception:
                continue
        
        return best_algorithm
    
    def _track_compression(self, compression_type: str, execution_time: float, bytes_saved: int) -> None:
        """Track compression performance metrics."""
        if compression_type in self._compression_stats:
            stats = self._compression_stats[compression_type]
            stats["count"] += 1
            stats["total_time"] += execution_time
            stats["total_saved"] += bytes_saved
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        stats = {}
        for compression_type, data in self._compression_stats.items():
            if data["count"] > 0:
                stats[compression_type] = {
                    "count": data["count"],
                    "avg_time": data["total_time"] / data["count"],
                    "avg_saved": data["total_saved"] / data["count"],
                    "total_time": data["total_time"],
                    "total_saved": data["total_saved"],
                }
            else:
                stats[compression_type] = data
        
        return {
            "compression_stats": stats,
            "available_algorithms": [alg.value for alg in self.available_algorithms],
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        logger.info("Context compressor cleaned up")