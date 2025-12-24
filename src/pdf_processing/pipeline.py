"""Main document processing pipeline that integrates all components.

Supports PDF, EPUB, and MOBI document formats.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from returns.result import Result

from config.logging_config import get_logger
from config.settings import settings
from src.core.database import get_db_manager
from src.pdf_processing.adaptive_learning import AdaptiveLearningSystem
from src.pdf_processing.content_chunker import ContentChunk, ContentChunker
from src.pdf_processing.embedding_generator import EmbeddingGenerator
from src.pdf_processing.pdf_parser import PDFParser, PDFProcessingError
from src.pdf_processing.ebook_parser import EbookParser
from src.pdf_processing.document_parser import UnifiedDocumentParser
from src.performance.parallel_processor import ParallelProcessor, ResourceLimits
from src.utils.file_size_handler import FileSizeCategory, FileSizeHandler
from src.utils.security import InputSanitizer, InputValidationError, validate_path

logger = get_logger(__name__)


class DocumentProcessingPipeline:
    """Orchestrates the complete document processing workflow for PDF, EPUB, and MOBI files."""

    def __init__(self, enable_parallel: bool = True, max_workers: Optional[int] = None, 
                 prompt_for_ollama: bool = True, model_name: Optional[str] = None):
        """Initialize the document processing pipeline.

        Args:
            enable_parallel: Whether to enable parallel processing
            max_workers: Maximum number of parallel workers
            prompt_for_ollama: Whether to prompt user for Ollama model selection
            model_name: Specific model to use for embeddings (e.g., "nomic-embed-text")
        """
        # Initialize parsers
        self.pdf_parser = PDFParser()
        self.ebook_parser = EbookParser()
        self.unified_parser = UnifiedDocumentParser()
        
        # For backward compatibility
        self.parser = self.pdf_parser
        self.chunker = ContentChunker()
        self.adaptive_system = AdaptiveLearningSystem()
        
        # Initialize embedding generator
        self.embedding_generator = self._init_embedding_generator(model_name, prompt_for_ollama)
            
        self.db = get_db_manager()
        self.enable_parallel = enable_parallel
        self.parallel_processor = None
        self._initialized = False

        if enable_parallel:
            import multiprocessing as mp

            self.parallel_processor = ParallelProcessor(
                ResourceLimits(
                    max_workers=max_workers or min(max(1, mp.cpu_count() - 1), 4),
                    max_memory_mb=2048,
                    task_timeout=600,
                )
            )

    async def process_document(
        self,
        document_path: str,
        rulebook_name: str,
        system: str,
        source_type: str = "rulebook",
        enable_adaptive_learning: bool = True,
        skip_size_check: bool = False,
        user_confirmed: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a document file (PDF, EPUB, or MOBI) through the complete pipeline.

        Args:
            document_path: Path to document file
            rulebook_name: Name of the rulebook
            system: Game system (e.g., "D&D 5e")
            source_type: Type of source ("rulebook" or "flavor")
            enable_adaptive_learning: Whether to use adaptive learning
            skip_size_check: Skip file size validation
            user_confirmed: Whether user has already confirmed large file processing

        Returns:
            Processing results and statistics
        """
        start_time = time.time()

        try:
            # Validate inputs
            document_path_obj = self._validate_document_inputs(document_path, rulebook_name, system, source_type)

            # Handle file size validation
            size_result = self._handle_file_size(document_path_obj, skip_size_check, user_confirmed)
            if size_result:
                return size_result

            logger.info(
                "Starting document processing",
                document=document_path,
                rulebook=rulebook_name,
                system=system,
                format=document_path_obj.suffix[1:],
            )

            # Step 1: Extract content from document
            logger.info("Step 1: Extracting document content")
            try:
                document_content = await asyncio.to_thread(
                    self.unified_parser.extract_text_from_document,
                    str(document_path_obj),
                    skip_size_check=skip_size_check,
                    user_confirmed=user_confirmed,
                )
            except (PDFProcessingError, Exception) as e:
                logger.error(f"Failed to extract document content: {e}")
                raise

            # Check for duplicate
            if self._is_duplicate(document_content["file_hash"]):
                logger.warning("Duplicate document detected", hash=document_content["file_hash"])
                return {
                    "status": "duplicate",
                    "message": "This document has already been processed",
                    "file_hash": document_content["file_hash"],
                }

            # Prepare source metadata
            source_id = str(uuid.uuid4())
            source_metadata = {
                "source_id": source_id,
                "rulebook_name": rulebook_name,
                "system": system,
                "source_type": source_type,
                "file_name": document_content["file_name"],
                "file_hash": document_content["file_hash"],
                "document_type": document_content.get("document_type", "pdf"),
            }

            # Step 2: Apply adaptive learning patterns
            if enable_adaptive_learning and settings.enable_adaptive_learning:
                logger.info("Step 2: Applying adaptive learning")
                self._apply_adaptive_patterns(document_content, system)

            # Step 3: Chunk the content
            logger.info("Step 3: Chunking content")
            chunks = await asyncio.to_thread(
                self.chunker.chunk_document, document_content, source_metadata
            )

            # Step 4: Generate embeddings
            logger.info("Step 4: Generating embeddings")
            embeddings = await asyncio.to_thread(
                self.embedding_generator.generate_embeddings, chunks
            )

            # Validate embeddings
            if not self.embedding_generator.validate_embeddings(embeddings):
                logger.warning("Some embeddings failed validation")

            # Step 5: Store in database
            logger.info("Step 5: Storing in database")
            stored_count = self._store_chunks(chunks, embeddings, source_type)

            # Step 6: Learn from this document
            if enable_adaptive_learning and settings.enable_adaptive_learning:
                logger.info("Step 6: Learning from document")
                self.adaptive_system.learn_from_document(document_content, system)

            # Store source metadata
            self._store_source_metadata(source_metadata)

            # Generate processing statistics
            processing_time = time.time() - start_time
            stats = {
                "status": "success",
                "source_id": source_id,
                "rulebook_name": rulebook_name,
                "system": system,
                "document_type": document_content.get("document_type", "pdf"),
                "total_pages": document_content["total_pages"],
                "total_chunks": len(chunks),
                "stored_chunks": stored_count,
                "tables_extracted": len(document_content.get("tables", [])),
                "embeddings_generated": len(embeddings),
                "file_hash": document_content["file_hash"],
                "processing_time_seconds": round(processing_time, 2),
            }

            logger.info("Document processing complete", **stats)

            return stats

        except FileNotFoundError as e:
            logger.error(f"Document file not found: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": "file_not_found",
                "document_path": document_path,
            }
        except (PDFProcessingError, Exception) as e:
            logger.error(f"Document processing error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": "document_processing_error",
                "document_path": document_path,
            }
        except Exception as e:
            logger.error("Unexpected error during document processing", error=str(e), exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": "unexpected_error",
                "document_path": document_path,
                "processing_time_seconds": round(time.time() - start_time, 2),
            }
    
    async def process_pdf(
        self,
        pdf_path: str,
        rulebook_name: str,
        system: str,
        source_type: str = "rulebook",
        enable_adaptive_learning: bool = True,
        skip_size_check: bool = False,
        user_confirmed: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a PDF file through the complete pipeline.
        
        This method is maintained for backward compatibility.
        It delegates to process_document().

        Args:
            pdf_path: Path to PDF file
            rulebook_name: Name of the rulebook
            system: Game system (e.g., "D&D 5e")
            source_type: Type of source ("rulebook" or "flavor")
            enable_adaptive_learning: Whether to use adaptive learning
            skip_size_check: Skip file size validation
            user_confirmed: Whether user has already confirmed large file processing

        Returns:
            Processing results and statistics
        """
        # Backward compatibility: delegate to process_document
        return await self.process_document(
            pdf_path,
            rulebook_name,
            system,
            source_type,
            enable_adaptive_learning,
            skip_size_check,
            user_confirmed,
        )

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
    
    def _handle_file_size(self, pdf_path_obj, skip_size_check: bool, user_confirmed: bool):
        """Handle file size validation and adjustment."""
        if skip_size_check or user_confirmed:
            return None
            
        file_info = FileSizeHandler.get_file_info(pdf_path_obj)
        category = FileSizeCategory(file_info["category"])

        if file_info["requires_confirmation"]:
            logger.info(f"Large file detected: {file_info['name']}")
            return {
                "success": False,
                "requires_confirmation": True,
                "file_info": file_info,
                "confirmation_message": FileSizeHandler.generate_confirmation_message(file_info),
                "message": f"File '{file_info['name']}' is {file_info['size_formatted']}. Confirmation required.",
            }

        if category in [FileSizeCategory.LARGE, FileSizeCategory.VERY_LARGE]:
            recommendations = FileSizeHandler.get_processing_recommendations(file_info)
            logger.info(f"Adjusting parameters for large file: {recommendations}")
            if hasattr(self.chunker, "max_chunk_size"):
                self.chunker.max_chunk_size = recommendations["chunk_size"]
                
        return None

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
                    **pdf_info, enable_adaptive_learning=enable_adaptive_learning
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
                task = await self.parallel_processor.submit_task("pdf_processing", pdf_info)
                tasks.append(task)

            # Wait for all tasks to complete
            completed_tasks = await self.parallel_processor.wait_for_all(
                [t.id for t in tasks], timeout=300  # 5 minutes timeout
            )

            # Process results
            successful = 0
            failed = 0
            results = []

            for task in completed_tasks:
                if task.status.value == "completed":
                    successful += 1
                    results.append(
                        {
                            "pdf_path": task.data["pdf_path"],
                            "status": "success",
                            "result": task.result,
                        }
                    )
                else:
                    failed += 1
                    results.append(
                        {"pdf_path": task.data["pdf_path"], "status": "failed", "error": task.error}
                    )

            processing_time = time.time() - start_time
            stats = self.parallel_processor.get_statistics()

            return {
                "results": results,
                "successful": successful,
                "failed": failed,
                "total": len(pdf_files),
                "processing_time": processing_time,
                "statistics": stats,
                "method": "parallel",
            }

        finally:
            pass

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
        self, chunks: List[ContentChunk], embeddings: List[List[float]], source_type: str
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

        # Prepare batch data for more efficient storage
        batch_documents = []
        batch_embeddings = []
        batch_ids = []
        batch_metadata = []

        for chunk, embedding in zip(chunks, embeddings):
            batch_ids.append(chunk.id)
            batch_documents.append(chunk.content)
            batch_embeddings.append(embedding)
            batch_metadata.append(
                {
                    **chunk.metadata,
                    "chunk_type": chunk.chunk_type,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "char_count": chunk.char_count,
                    "word_count": chunk.word_count,
                }
            )

        # Store in batch with error handling
        try:
            if hasattr(self.db, "add_documents"):
                # Use batch operation if available
                self.db.add_documents(
                    collection_name=collection_name,
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    ids=batch_ids,
                    metadatas=batch_metadata,
                )
                stored_count = len(batch_ids)
            else:
                # Fall back to individual operations with error handling
                for doc_id, content, metadata, embedding in zip(
                    batch_ids, batch_documents, batch_metadata, batch_embeddings
                ):
                    try:
                        self.db.add_document(
                            collection_name=collection_name,
                            document_id=doc_id,
                            content=content,
                            metadata=metadata,
                            embedding=embedding,
                        )
                        stored_count += 1
                    except Exception as e:
                        logger.error(
                            "Failed to store chunk",
                            chunk_id=doc_id,
                            error=str(e),
                        )
        except Exception as e:
            logger.error(
                "Failed to store batch of chunks",
                count=len(chunks),
                error=str(e),
            )
            # Re-raise for proper error handling upstream
            raise RuntimeError(f"Batch storage failed: {e}") from e

        return stored_count

    def _store_source_metadata(self, source_metadata: Dict[str, Any]):
        """
        Store source metadata for tracking.

        Args:
            source_metadata: Source metadata dictionary
        """
        try:
            # Determine collection based on source type
            collection_name = (
                "flavor_sources" if source_metadata.get("source_type") == "flavor" else "rulebooks"
            )

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
            logger.error("Failed to store source metadata", error=str(e))

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
            stats["database_stats"][collection_name] = self.db.get_collection_stats(collection_name)

        return stats

    def reprocess_with_corrections(
        self, source_id: str, corrections: Dict[str, Any]
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
            logger.info("Reprocessing with corrections", source_id=source_id)

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
            logger.error("Reprocessing failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _init_embedding_generator(self, model_name: Optional[str], prompt_for_ollama: bool) -> EmbeddingGenerator:
        """Initialize embedding generator with model selection."""
        if model_name:
            return EmbeddingGenerator(model_name=model_name)
        elif prompt_for_ollama and not hasattr(PDFProcessingPipeline, '_embedding_generator_prompted'):
            PDFProcessingPipeline._embedding_generator_prompted = True
            return EmbeddingGenerator.prompt_and_create()
        else:
            return EmbeddingGenerator()
    
    def _validate_document_inputs(self, document_path: str, rulebook_name: str, system: str, source_type: str):
        """Validate and sanitize input parameters for any document type."""
        try:
            document_path_obj = validate_path(document_path, must_exist=True)
            supported_formats = ['.pdf', '.epub', '.mobi', '.azw', '.azw3']
            if document_path_obj.suffix.lower() not in supported_formats:
                raise ValueError(
                    f"File must be a supported document format. "
                    f"Supported: {', '.join(supported_formats)}. "
                    f"Got: {document_path_obj.suffix}"
                )
        except Exception as e:
            logger.error(f"Path validation failed: {e}")
            raise ValueError(f"Invalid document path: {e}") from e

        sanitizer = InputSanitizer()
        try:
            rulebook_name = sanitizer.validate_system_name(rulebook_name)
            system = sanitizer.validate_system_name(system)
        except InputValidationError as e:
            logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Invalid input: {e}") from e

        if not rulebook_name or not system:
            raise ValueError("rulebook_name and system are required")

        if source_type not in ["rulebook", "flavor"]:
            raise ValueError("source_type must be 'rulebook' or 'flavor'")
            
        return document_path_obj
    
    def _validate_inputs(self, pdf_path: str, rulebook_name: str, system: str, source_type: str):
        """Legacy validation method for PDFs only. Maintained for backward compatibility."""
        try:
            pdf_path_obj = validate_path(pdf_path, must_exist=True)
            if pdf_path_obj.suffix.lower() != ".pdf":
                raise ValueError(f"File must be a PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Path validation failed: {e}")
            raise ValueError(f"Invalid PDF path: {e}") from e

        sanitizer = InputSanitizer()
        try:
            rulebook_name = sanitizer.validate_system_name(rulebook_name)
            system = sanitizer.validate_system_name(system)
        except InputValidationError as e:
            logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Invalid input: {e}") from e

        if not rulebook_name or not system:
            raise ValueError("rulebook_name and system are required")

        if source_type not in ["rulebook", "flavor"]:
            raise ValueError("source_type must be 'rulebook' or 'flavor'")
            
        return pdf_path_obj

    async def _extract_pdf_content(self, pdf_path_obj: Path, skip_size_check: bool, user_confirmed: bool) -> Result[Dict[str, Any], str]:
        """Extract content from PDF file asynchronously."""
        try:
            pdf_content = await asyncio.to_thread(
                self.parser.extract_text_from_pdf,
                str(pdf_path_obj),
                skip_size_check=skip_size_check,
                user_confirmed=user_confirmed,
            )
            return Result.ok(pdf_content)
            
        except PDFProcessingError as e:
            error_msg = f"PDF processing error: {str(e)}"
            logger.error(error_msg)
            return Result.err(error_msg)
        except Exception as e:
            error_msg = f"Failed to extract PDF content: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.err(error_msg)
    
    def _create_source_metadata(self, source_id: str, rulebook_name: str, system: str, source_type: str, pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        """Create source metadata dictionary."""
        return {
            "source_id": source_id,
            "rulebook_name": rulebook_name,
            "system": system,
            "source_type": source_type,
            "file_name": pdf_content["file_name"],
            "file_hash": pdf_content["file_hash"],
            "total_pages": pdf_content["total_pages"],
            "created_at": time.time(),
        }
    
    async def _chunk_content_async(self, pdf_content: Dict[str, Any], source_metadata: Dict[str, Any]) -> Result[List[ContentChunk], str]:
        """Chunk document content asynchronously."""
        try:
            chunks = await asyncio.to_thread(
                self.chunker.chunk_document, 
                pdf_content, 
                source_metadata
            )
            return Result.ok(chunks)
            
        except Exception as e:
            error_msg = f"Failed to chunk content: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.err(error_msg)
    
    async def _generate_embeddings_async(self, chunks: List[ContentChunk]) -> Result[List[List[float]], str]:
        """Generate embeddings for chunks asynchronously."""
        try:
            embeddings = await asyncio.to_thread(
                self.embedding_generator.generate_embeddings, 
                chunks
            )
            
            # Validate embeddings
            if not await asyncio.to_thread(self.embedding_generator.validate_embeddings, embeddings):
                logger.warning("Some embeddings failed validation")
            
            return Result.ok(embeddings)
            
        except Exception as e:
            error_msg = f"Failed to generate embeddings: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.err(error_msg)
    
    async def _learn_from_document_async(self, pdf_content: Dict[str, Any], system: str) -> Result[None, str]:
        """Learn from document for adaptive system asynchronously."""
        if not self.adaptive_system:
            return Result.ok(None)
            
        try:
            await asyncio.to_thread(
                self.adaptive_system.learn_from_document, 
                pdf_content, 
                system
            )
            return Result.ok(None)
            
        except Exception as e:
            error_msg = f"Failed to learn from document: {str(e)}"
            logger.error(error_msg, system=system, exc_info=True)
            return Result.err(error_msg)


# Backward compatibility alias
PDFProcessingPipeline = DocumentProcessingPipeline
