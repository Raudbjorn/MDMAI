"""Main PDF processing pipeline that integrates all components."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from config.logging_config import get_logger
from config.settings import settings
from src.core.database import get_db_manager
from src.pdf_processing.pdf_parser import PDFParser
from src.pdf_processing.content_chunker import ContentChunker, ContentChunk
from src.pdf_processing.adaptive_learning import AdaptiveLearningSystem
from src.pdf_processing.embedding_generator import EmbeddingGenerator

logger = get_logger(__name__)


class PDFProcessingPipeline:
    """Orchestrates the complete PDF processing workflow."""
    
    def __init__(self):
        """Initialize the PDF processing pipeline."""
        self.parser = PDFParser()
        self.chunker = ContentChunker()
        self.adaptive_system = AdaptiveLearningSystem()
        self.embedding_generator = EmbeddingGenerator()
        self.db = get_db_manager()
    
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
        try:
            logger.info(
                f"Starting PDF processing",
                pdf=pdf_path,
                rulebook=rulebook_name,
                system=system,
            )
            
            # Step 1: Extract content from PDF
            logger.info("Step 1: Extracting PDF content")
            pdf_content = await asyncio.to_thread(self.parser.extract_text_from_pdf, pdf_path)
            
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
            }
            
            logger.info(
                f"PDF processing complete",
                **stats
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"PDF processing failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "pdf_path": pdf_path,
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