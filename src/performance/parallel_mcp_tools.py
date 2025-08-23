"""MCP tools for parallel processing operations."""

from typing import Any, Dict, List, Optional

from config.logging_config import get_logger

logger = get_logger(__name__)

# Global instances (initialized in main.py)
_parallel_processor = None
_resource_manager = None
_pdf_processor = None


def initialize_parallel_tools(parallel_processor=None, resource_manager=None, pdf_processor=None):
    """Initialize parallel processing tools with dependencies."""
    global _parallel_processor, _resource_manager, _pdf_processor
    _parallel_processor = parallel_processor
    _resource_manager = resource_manager
    _pdf_processor = pdf_processor
    logger.info("Parallel processing tools initialized")


def register_parallel_tools(mcp_server):
    """Register parallel processing tools with the MCP server."""
    
    @mcp_server.tool()
    async def process_pdfs_parallel(
        pdf_files: List[Dict[str, str]],
        enable_adaptive_learning: bool = True,
    ) -> Dict[str, Any]:
        """
        Process multiple PDF files in parallel for faster extraction.
        
        Args:
            pdf_files: List of PDF file information, each containing:
                - pdf_path: Path to the PDF file
                - rulebook_name: Name of the rulebook
                - system: Game system (e.g., "D&D 5e")
                - source_type: Type of source ("rulebook" or "flavor")
            enable_adaptive_learning: Whether to use adaptive learning patterns
            
        Returns:
            Processing results including success/failure counts and statistics
        """
        try:
            if not pdf_files:
                return {
                    "success": False,
                    "error": "No PDF files provided",
                }
            
            # Use the PDF processing pipeline
            from src.pdf_processing.pipeline import PDFProcessingPipeline
            
            pipeline = PDFProcessingPipeline(enable_parallel=True)
            results = await pipeline.process_multiple_pdfs(
                pdf_files=pdf_files,
                enable_adaptive_learning=enable_adaptive_learning
            )
            
            return {
                "success": True,
                "message": f"Processed {results['total']} PDFs in parallel",
                "successful": results.get("successful", 0),
                "failed": results.get("failed", 0),
                "processing_time": results.get("processing_time", 0),
                "method": results.get("method", "unknown"),
                "statistics": results.get("statistics", {}),
                "results": results.get("results", []),
            }
            
        except Exception as e:
            logger.error(f"Parallel PDF processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}",
            }
    
    @mcp_server.tool()
    async def batch_search_parallel(
        queries: List[Dict[str, Any]],
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute multiple search queries in parallel.
        
        Args:
            queries: List of search queries, each containing:
                - query: Search query text
                - collection: Collection to search (default: "rulebooks")
                - max_results: Maximum results per query (default: 5)
            max_workers: Maximum parallel workers (uses system default if not specified)
            
        Returns:
            Combined search results from all queries
        """
        try:
            if not queries:
                return {
                    "success": False,
                    "error": "No queries provided",
                }
            
            from src.performance.parallel_processor import ParallelProcessor, ResourceLimits
            
            # Create processor for search operations
            processor = ParallelProcessor(
                ResourceLimits(max_workers=max_workers or 4)
            )
            
            await processor.initialize()
            
            try:
                # Submit search task
                task = await processor.submit_task(
                    "search",
                    {"queries": queries}
                )
                
                # Wait for completion
                completed_task = await processor.wait_for_task(
                    task.id,
                    timeout=60  # 1 minute timeout for searches
                )
                
                if completed_task.status.value == "completed":
                    return {
                        "success": True,
                        "message": f"Executed {len(queries)} searches in parallel",
                        "results": completed_task.result.get("results", []),
                        "query_count": completed_task.result.get("query_count", 0),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Search task failed: {completed_task.error}",
                    }
                    
            finally:
                await processor.shutdown()
                
        except Exception as e:
            logger.error(f"Parallel search failed: {str(e)}")
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
            }
    
    @mcp_server.tool()
    async def batch_embeddings_parallel(
        texts: List[str],
        batch_size: int = 10,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for multiple texts in parallel.
        
        Args:
            texts: List of text strings to generate embeddings for
            batch_size: Number of texts to process per batch
            max_workers: Maximum parallel workers
            
        Returns:
            Generated embeddings and processing statistics
        """
        try:
            if not texts:
                return {
                    "success": False,
                    "error": "No texts provided",
                }
            
            from src.performance.parallel_processor import ParallelProcessor, ResourceLimits
            
            # Create processor for embedding generation
            processor = ParallelProcessor(
                ResourceLimits(max_workers=max_workers or 4)
            )
            
            await processor.initialize()
            
            try:
                # Submit embedding task
                task = await processor.submit_task(
                    "embedding_generation",
                    {"chunks": texts, "batch_size": batch_size}
                )
                
                # Wait for completion
                completed_task = await processor.wait_for_task(
                    task.id,
                    timeout=300  # 5 minutes timeout
                )
                
                if completed_task.status.value == "completed":
                    return {
                        "success": True,
                        "message": f"Generated embeddings for {len(texts)} texts",
                        "embeddings": completed_task.result.get("embeddings", []),
                        "count": completed_task.result.get("count", 0),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Embedding task failed: {completed_task.error}",
                    }
                    
            finally:
                await processor.shutdown()
                
        except Exception as e:
            logger.error(f"Parallel embedding generation failed: {str(e)}")
            return {
                "success": False,
                "error": f"Embedding generation failed: {str(e)}",
            }
    
    @mcp_server.tool()
    async def batch_database_operations(
        operations: List[Dict[str, Any]],
        batch_size: int = 100,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute multiple database operations in parallel batches.
        
        Args:
            operations: List of database operations, each containing:
                - type: Operation type ("add_document", "update_document", "delete_document")
                - data: Operation-specific data
            batch_size: Number of operations per batch
            max_workers: Maximum parallel workers
            
        Returns:
            Operation results and statistics
        """
        try:
            if not operations:
                return {
                    "success": False,
                    "error": "No operations provided",
                }
            
            from src.performance.parallel_processor import ParallelProcessor, ResourceLimits
            
            # Create processor for batch operations
            processor = ParallelProcessor(
                ResourceLimits(max_workers=max_workers or 4)
            )
            
            await processor.initialize()
            
            try:
                # Submit batch task
                task = await processor.submit_task(
                    "batch_operation",
                    {
                        "operations": operations,
                        "batch_size": batch_size
                    }
                )
                
                # Wait for completion
                completed_task = await processor.wait_for_task(
                    task.id,
                    timeout=600  # 10 minutes timeout
                )
                
                if completed_task.status.value == "completed":
                    result = completed_task.result
                    return {
                        "success": True,
                        "message": f"Executed {len(operations)} operations in batches",
                        "processed": result.get("processed", 0),
                        "failed": result.get("failed", 0),
                        "errors": result.get("errors", []),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Batch operation failed: {completed_task.error}",
                    }
                    
            finally:
                await processor.shutdown()
                
        except Exception as e:
            logger.error(f"Batch database operations failed: {str(e)}")
            return {
                "success": False,
                "error": f"Batch operations failed: {str(e)}",
            }
    
    @mcp_server.tool()
    async def get_parallel_processing_stats() -> Dict[str, Any]:
        """
        Get statistics about parallel processing operations.
        
        Returns:
            Current parallel processing statistics and resource usage
        """
        try:
            from src.performance.parallel_processor import ResourceManager
            import psutil
            
            # Get system resource information
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            stats = {
                "success": True,
                "system": {
                    "cpu_count": cpu_count,
                    "cpu_usage_percent": cpu_percent,
                    "memory_total_mb": memory.total / (1024 * 1024),
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "memory_usage_percent": memory.percent,
                },
                "recommendations": [],
            }
            
            # Add recommendations based on system state
            if cpu_percent > 80:
                stats["recommendations"].append(
                    "High CPU usage detected. Consider reducing parallel workers."
                )
            
            if memory.percent > 85:
                stats["recommendations"].append(
                    "High memory usage detected. Consider processing smaller batches."
                )
            
            optimal_workers = {
                "pdf_processing": max(1, cpu_count - 1),
                "embedding_generation": max(1, cpu_count - 1),
                "search": min(cpu_count * 2, 16),
                "batch_operations": cpu_count,
            }
            
            stats["optimal_workers"] = optimal_workers
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get parallel processing stats: {str(e)}")
            return {
                "success": False,
                "error": f"Stats retrieval failed: {str(e)}",
            }
    
    @mcp_server.tool()
    async def cancel_parallel_task(
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Cancel a running parallel processing task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            Cancellation status
        """
        try:
            if not _parallel_processor:
                return {
                    "success": False,
                    "error": "No active parallel processor",
                }
            
            # Get task status
            status = _parallel_processor.get_task_status(task_id)
            
            if not status:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found",
                }
            
            # Mark task as cancelled
            task = _parallel_processor.tasks.get(task_id)
            if task:
                from src.performance.parallel_processor import TaskStatus
                task.status = TaskStatus.CANCELLED
                
                return {
                    "success": True,
                    "message": f"Task {task_id} cancelled",
                    "previous_status": status.value,
                }
            
            return {
                "success": False,
                "error": "Failed to cancel task",
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel task: {str(e)}")
            return {
                "success": False,
                "error": f"Cancellation failed: {str(e)}",
            }
    
    logger.info("Parallel processing MCP tools registered successfully")