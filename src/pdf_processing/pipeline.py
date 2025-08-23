"""Main PDF processing pipeline that integrates all components."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import time

from config.logging_config import get_logger
from config.settings import settings
from src.core.database import get_db_manager
from src.pdf_processing.pdf_parser import PDFParser, PDFProcessingError
from src.pdf_processing.content_chunker import ContentChunker, ContentChunk
from src.pdf_processing.adaptive_learning import AdaptiveLearningSystem
from src.pdf_processing.embedding_generator import EmbeddingGenerator
from src.performance.parallel_processor import ParallelProcessor, ResourceLimits, TaskStatus

logger = get_logger(__name__)


class PDFProcessingPipeline:
    """Orchestrates the complete PDF processing workflow."""
    
    def __init__(self, enable_parallel: bool = True, max_workers: Optional[int] = None):
        """Initialize the PDF processing pipeline.
        
        Args:
            enable_parallel: Whether to enable parallel processing
            max_workers: Maximum number of parallel workers
        """
        self.parser = PDFParser()
        self.chunker = ContentChunker()
        self.adaptive_system = AdaptiveLearningSystem()
        self.embedding_generator = EmbeddingGenerator()
        self.db = get_db_manager()
        self.enable_parallel = enable_parallel
        self.parallel_processor = None
        self._initialized = False
        
        if enable_parallel:
            import multiprocessing as mp
            self.parallel_processor = ParallelProcessor(
                ResourceLimits(
                    max_workers=max_workers or min(mp.cpu_count() - 1, 4),
                    max_memory_mb=2048,
                    task_timeout=600
                )
            )
    
    async def process_pdf(
        self,
        pdf_path: str,
        rulebook_name: str,
        system: str,
        source_type: str = "rulebook",
        enable_adaptive_learning: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a PDF file through the complete pipeline.
        
        Args:
            pdf_path: Path to PDF file
            rulebook_name: Name of the rulebook
            system: Game system (e.g., "D&D 5e")
            source_type: Type of source ("rulebook" or "flavor")
            enable_adaptive_learning: Whether to use adaptive learning
            
        Returns:
            Processing results and statistics
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            pdf_path_obj = Path(pdf_path)
            if not pdf_path_obj.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if not rulebook_name or not system:
                raise ValueError("rulebook_name and system are required")
            
            if source_type not in ["rulebook", "flavor"]:
                raise ValueError("source_type must be 'rulebook' or 'flavor'")
            
            logger.info(
                f"Starting PDF processing",
                pdf=pdf_path,
                rulebook=rulebook_name,
                system=system,
            )
            
            # Step 1: Extract content from PDF
            logger.info("Step 1: Extracting PDF content")
            try:
                pdf_content = await asyncio.to_thread(self.parser.extract_text_from_pdf, pdf_path)
            except PDFProcessingError as e:
                logger.error(f"Failed to extract PDF content: {e}")
                raise
            
            # Check for duplicate
            if self._is_duplicate(pdf_content["file_hash"]):
                logger.warning(f"Duplicate PDF detected", hash=pdf_content["file_hash"])
                return {
                    "status": "duplicate",
                    "message": f"This PDF has already been processed",
                    "file_hash": pdf_content["file_hash"],
                }
            
            # Prepare source metadata
            source_id = str(uuid.uuid4())
            source_metadata = {
                "source_id": source_id,
                "rulebook_name": rulebook_name,
                "system": system,
                "source_type": source_type,
                "file_name": pdf_content["file_name"],
                "file_hash": pdf_content["file_hash"],
            }
            
            # Step 2: Apply adaptive learning patterns
            if enable_adaptive_learning and settings.enable_adaptive_learning:
                logger.info("Step 2: Applying adaptive learning")
                self._apply_adaptive_patterns(pdf_content, system)
            
            # Step 3: Chunk the content
            logger.info("Step 3: Chunking content")
            chunks = await asyncio.to_thread(self.chunker.chunk_document, pdf_content, source_metadata)
            
            # Step 4: Generate embeddings
            logger.info("Step 4: Generating embeddings")
            embeddings = await asyncio.to_thread(self.embedding_generator.generate_embeddings, chunks)
            
            # Validate embeddings
            if not self.embedding_generator.validate_embeddings(embeddings):
                logger.warning("Some embeddings failed validation")
            
            # Step 5: Store in database
            logger.info("Step 5: Storing in database")
            stored_count = self._store_chunks(chunks, embeddings, source_type)
            
            # Step 6: Learn from this document
            if enable_adaptive_learning and settings.enable_adaptive_learning:
                logger.info("Step 6: Learning from document")
                self.adaptive_system.learn_from_document(pdf_content, system)
            
            # Store source metadata
            self._store_source_metadata(source_metadata)
            
            # Generate processing statistics
            processing_time = time.time() - start_time
            stats = {
                "status": "success",
                "source_id": source_id,
                "rulebook_name": rulebook_name,
                "system": system,
                "total_pages": pdf_content["total_pages"],
                "total_chunks": len(chunks),
                "stored_chunks": stored_count,
                "tables_extracted": len(pdf_content.get("tables", [])),
                "embeddings_generated": len(embeddings),
                "file_hash": pdf_content["file_hash"],
                "processing_time_seconds": round(processing_time, 2),
            }
            
            logger.info(
                f"PDF processing complete",
                **stats
            )
            
            return stats
            
        except FileNotFoundError as e:
            logger.error(f"PDF file not found: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": "file_not_found",
                "pdf_path": pdf_path,
            }
        except PDFProcessingError as e:
            logger.error(f"PDF processing error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": "pdf_processing_error",
                "pdf_path": pdf_path,
            }
        except Exception as e:
            logger.error(f"Unexpected error during PDF processing", error=str(e), exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": "unexpected_error",
                "pdf_path": pdf_path,
                "processing_time_seconds": round(time.time() - start_time, 2),
            }
    
    def _is_duplicate(self, file_hash: str) -> bool:
        """
        Check if a PDF has already been processed.
        
        Args:
            file_hash: SHA256 hash of the PDF file
            
        Returns:
            True if duplicate
        """
        # Check both rulebooks and flavor_sources collections for duplicates
        for collection_name in ["rulebooks", "flavor_sources"]:
            existing = self.db.list_documents(
                collection_name=collection_name,
                metadata_filter={"file_hash": file_hash},
                limit=1,
            )
            if len(existing) > 0:
                return True
        
        return False
    
    async def process_multiple_pdfs(
        self,
        pdf_files: List[Dict[str, str]],
        enable_adaptive_learning: bool = True,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple PDFs in parallel.
        
        Args:
            pdf_files: List of dicts with pdf_path, rulebook_name, system, source_type
            enable_adaptive_learning: Whether to use adaptive learning
            max_workers: Maximum number of parallel workers
            
        Returns:
            Processing results for all PDFs
        """
        start_time = time.time()
        
        if not pdf_files:
            return {
                "results": [],
                "total": 0,
                "successful": 0,
                "failed": 0,
                "method": "none",
                "processing_time": 0,
            }
        
        if not self.enable_parallel or not self.parallel_processor:
            # Fall back to sequential processing
            logger.info(f"Processing {len(pdf_files)} PDFs sequentially")
            results = []
            successful = 0
            failed = 0
            
            for pdf_info in pdf_files:
                result = await self.process_pdf(
                    **pdf_info,
                    enable_adaptive_learning=enable_adaptive_learning
                )
                results.append(result)
                if result.get("status") == "success":
                    successful += 1
                else:
                    failed += 1
            
            return {
                "results": results,
                "total": len(results),
                "successful": successful,
                "failed": failed,
                "method": "sequential",
                "processing_time": round(time.time() - start_time, 2),
            }
        
        # Initialize parallel processor if not already initialized
        if not self._initialized:
            await self.parallel_processor.initialize()
            self._initialized = True
        
        try:
            start_time = time.time()
            
            # Submit PDF processing tasks
            tasks = []
            for pdf_info in pdf_files:
                task = await self.parallel_processor.submit_task(
                    "pdf_processing",
                    pdf_info
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            completed_tasks = await self.parallel_processor.wait_for_all(
                [t.id for t in tasks],
                timeout=300  # 5 minutes timeout
            )
            
            # Process results
            successful = 0
            failed = 0
            results = []
            
            for task in completed_tasks:
                if task.status.value == "completed":
                    successful += 1
                    results.append({
                        "pdf_path": task.data["pdf_path"],
                        "status": "success",
                        "result": task.result
                    })
                else:
                    failed += 1
                    results.append({
                        "pdf_path": task.data["pdf_path"],
                        "status": "failed",
                        "error": task.error
                    })
            
            processing_time = time.time() - start_time
            stats = self.parallel_processor.get_statistics()
            
            return {
                "results": results,
                "successful": successful,
                "failed": failed,
                "total": len(pdf_files),
                "processing_time": processing_time,
                "statistics": stats,
                "method": "parallel"
            }
            
        finally:
            await self.parallel_processor.shutdown()
            # Reinitialize for future use
            if self.enable_parallel:
                self.parallel_processor = ParallelProcessor(
                    ResourceLimits(max_workers=4)
                )
    
    def _apply_adaptive_patterns(self, pdf_content: Dict[str, Any], system: str):
        """
        Apply learned patterns to enhance extraction.
        
        Args:
            pdf_content: Extracted PDF content
            system: Game system
        """
        for page in pdf_content.get("pages", []):
            text = page.get("text", "")
            
            # Apply learned patterns
            extracted = self.adaptive_system.apply_learned_patterns(text, system)
            
            if extracted:
                # Add extracted structured data to page metadata
                if "extracted_data" not in page:
                    page["extracted_data"] = []
                page["extracted_data"].append(extracted)
    
    def _store_chunks(
        self,
        chunks: List[ContentChunk],
        embeddings: List[List[float]],
        source_type: str
    ) -> int:
        """
        Store chunks and embeddings in the database.
        
        Args:
            chunks: List of content chunks
            embeddings: List of embedding vectors
            source_type: Type of source
            
        Returns:
            Number of chunks stored
        """
        collection_name = "flavor_sources" if source_type == "flavor" else "rulebooks"
        stored_count = 0
        
        for chunk, embedding in zip(chunks, embeddings):
            try:
                # Prepare document for storage
                document_id = chunk.id
                content = chunk.content
                metadata = {
                    **chunk.metadata,
                    "chunk_type": chunk.chunk_type,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "char_count": chunk.char_count,
                    "word_count": chunk.word_count,
                }
                
                # Store in database
                self.db.add_document(
                    collection_name=collection_name,
                    document_id=document_id,
                    content=content,
                    metadata=metadata,
                    embedding=embedding,
                )
                
                stored_count += 1
                
            except Exception as e:
                logger.error(
                    f"Failed to store chunk",
                    chunk_id=chunk.id,
                    error=str(e),
                )
        
        return stored_count
    
    def _store_source_metadata(self, source_metadata: Dict[str, Any]):
        """
        Store source metadata for tracking.
        
        Args:
            source_metadata: Source metadata dictionary
        """
        try:
            # Determine collection based on source type
            collection_name = "flavor_sources" if source_metadata.get("source_type") == "flavor" else "rulebooks"
            
            # Store as a special document in the database
            self.db.add_document(
                collection_name=collection_name,
                document_id=f"source_{source_metadata['source_id']}",
                content=f"Source: {source_metadata['rulebook_name']}",
                metadata={
                    **source_metadata,
                    "document_type": "source_metadata",
                },
            )
        except Exception as e:
            logger.error(f"Failed to store source metadata", error=str(e))
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about PDF processing.
        
        Returns:
            Processing statistics
        """
        stats = {
            "adaptive_learning": self.adaptive_system.get_extraction_stats(),
            "embedding_model": self.embedding_generator.get_model_info(),
            "database_stats": {},
        }
        
        # Get database statistics
        for collection_name in ["rulebooks", "flavor_sources"]:
            stats["database_stats"][collection_name] = self.db.get_collection_stats(
                collection_name
            )
        
        return stats
    
    def reprocess_with_corrections(
        self,
        source_id: str,
        corrections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reprocess a source with manual corrections.
        
        Args:
            source_id: Source identifier
            corrections: Manual corrections to apply
            
        Returns:
            Reprocessing results
        """
        try:
            logger.info(f"Reprocessing with corrections", source_id=source_id)
            
            # Get source metadata
            source_doc = self.db.get_document(
                collection_name="rulebooks",
                document_id=f"source_{source_id}",
            )
            
            if not source_doc:
                return {
                    "status": "error",
                    "error": f"Source {source_id} not found",
                }
            
            metadata = source_doc["metadata"]
            system = metadata["system"]
            
            # Apply corrections to adaptive learning
            self.adaptive_system._apply_corrections(corrections, system)
            
            # TODO: Implement full reprocessing if needed
            
            return {
                "status": "success",
                "message": "Corrections applied to adaptive learning system",
                "source_id": source_id,
            }
            
        except Exception as e:
            logger.error(f"Reprocessing failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }