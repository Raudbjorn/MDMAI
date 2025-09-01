"""MCP tool definitions for document processing (PDF, EPUB, MOBI)."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from config.logging_config import get_logger
from src.pdf_processing.pipeline import DocumentProcessingPipeline

logger = get_logger(__name__)


# Schema definitions using Pydantic for validation
class ProcessDocumentInput(BaseModel):
    """Input schema for processing a document."""
    
    document_path: str = Field(
        ...,
        description="Path to the document file (PDF, EPUB, or MOBI)",
        min_length=1
    )
    rulebook_name: str = Field(
        ...,
        description="Name of the rulebook or source material",
        min_length=1
    )
    system: str = Field(
        ...,
        description="Game system (e.g., 'D&D 5e', 'Pathfinder')",
        min_length=1
    )
    source_type: str = Field(
        default="rulebook",
        description="Type of source: 'rulebook' or 'flavor'",
        pattern="^(rulebook|flavor)$"
    )
    enable_adaptive_learning: bool = Field(
        default=True,
        description="Whether to use adaptive learning for better extraction"
    )
    skip_size_check: bool = Field(
        default=False,
        description="Skip file size validation checks"
    )
    user_confirmed: bool = Field(
        default=False,
        description="Whether user has confirmed processing of large files"
    )
    
    @validator('document_path')
    def validate_document_path(cls, v):
        """Validate that the document path exists and has a supported extension."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Document file not found: {v}")
        
        supported_extensions = {'.pdf', '.epub', '.mobi', '.azw', '.azw3'}
        if path.suffix.lower() not in supported_extensions:
            raise ValueError(
                f"Unsupported document format: {path.suffix}. "
                f"Supported formats: {', '.join(sorted(supported_extensions))}"
            )
        return v


class BatchProcessDocumentsInput(BaseModel):
    """Input schema for batch processing multiple documents."""
    
    documents: List[Dict[str, Any]] = Field(
        ...,
        description="List of document configurations to process",
        min_items=1
    )
    enable_adaptive_learning: bool = Field(
        default=True,
        description="Whether to use adaptive learning for all documents"
    )
    max_workers: Optional[int] = Field(
        default=None,
        description="Maximum number of parallel workers (None for auto)",
        ge=1,
        le=16
    )


class ExtractDocumentTextInput(BaseModel):
    """Input schema for extracting text from a document."""
    
    document_path: str = Field(
        ...,
        description="Path to the document file",
        min_length=1
    )
    skip_size_check: bool = Field(
        default=False,
        description="Skip file size validation"
    )
    user_confirmed: bool = Field(
        default=False,
        description="Whether user has confirmed processing of large files"
    )
    extract_tables: bool = Field(
        default=True,
        description="Whether to extract and format tables"
    )
    extract_metadata: bool = Field(
        default=True,
        description="Whether to extract document metadata"
    )


class DocumentProcessingResult(BaseModel):
    """Output schema for document processing results."""
    
    status: str = Field(..., description="Processing status: 'success', 'error', 'duplicate'")
    source_id: Optional[str] = Field(None, description="Unique identifier for the processed source")
    rulebook_name: Optional[str] = Field(None, description="Name of the processed rulebook")
    system: Optional[str] = Field(None, description="Game system")
    document_type: Optional[str] = Field(None, description="Type of document: 'pdf', 'epub', 'mobi'")
    total_pages: Optional[int] = Field(None, description="Total number of pages/sections processed")
    total_chunks: Optional[int] = Field(None, description="Number of content chunks created")
    stored_chunks: Optional[int] = Field(None, description="Number of chunks stored in database")
    tables_extracted: Optional[int] = Field(None, description="Number of tables extracted")
    embeddings_generated: Optional[int] = Field(None, description="Number of embeddings generated")
    file_hash: Optional[str] = Field(None, description="SHA256 hash of the document")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to process")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")
    error_type: Optional[str] = Field(None, description="Type of error encountered")
    message: Optional[str] = Field(None, description="Additional message or details")


class DocumentExtractionResult(BaseModel):
    """Output schema for document text extraction."""
    
    success: bool = Field(..., description="Whether extraction was successful")
    file_name: str = Field(..., description="Name of the document file")
    file_hash: str = Field(..., description="SHA256 hash of the document")
    document_type: str = Field(..., description="Type of document detected")
    total_pages: int = Field(..., description="Total number of pages/sections")
    total_characters: int = Field(..., description="Total character count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    tables: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted tables")
    toc: Optional[List[Dict[str, Any]]] = Field(None, description="Table of contents")
    error: Optional[str] = Field(None, description="Error message if extraction failed")


# MCP Tool Classes
class ProcessDocumentTool:
    """MCP tool for processing a single document."""
    
    name = "process_document"
    description = "Process a document (PDF, EPUB, or MOBI) for TTRPG content extraction and indexing"
    
    def __init__(self):
        self.pipeline = None
    
    async def initialize(self):
        """Initialize the document processing pipeline."""
        if not self.pipeline:
            self.pipeline = DocumentProcessingPipeline(
                enable_parallel=True,
                prompt_for_ollama=False  # Use default model for MCP
            )
    
    async def execute(self, input_data: ProcessDocumentInput) -> DocumentProcessingResult:
        """
        Execute document processing.
        
        Args:
            input_data: Validated input parameters
            
        Returns:
            Processing results
        """
        await self.initialize()
        
        try:
            result = await self.pipeline.process_document(
                document_path=input_data.document_path,
                rulebook_name=input_data.rulebook_name,
                system=input_data.system,
                source_type=input_data.source_type,
                enable_adaptive_learning=input_data.enable_adaptive_learning,
                skip_size_check=input_data.skip_size_check,
                user_confirmed=input_data.user_confirmed,
            )
            
            return DocumentProcessingResult(**result)
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            return DocumentProcessingResult(
                status="error",
                error=str(e),
                error_type=type(e).__name__,
                message=f"Failed to process document: {input_data.document_path}"
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": ProcessDocumentInput.schema(),
            "output_schema": DocumentProcessingResult.schema(),
        }


class BatchProcessDocumentsTool:
    """MCP tool for processing multiple documents in batch."""
    
    name = "batch_process_documents"
    description = "Process multiple documents (PDF, EPUB, MOBI) in parallel"
    
    def __init__(self):
        self.pipeline = None
    
    async def initialize(self):
        """Initialize the document processing pipeline."""
        if not self.pipeline:
            self.pipeline = DocumentProcessingPipeline(
                enable_parallel=True,
                prompt_for_ollama=False
            )
    
    async def execute(self, input_data: BatchProcessDocumentsInput) -> Dict[str, Any]:
        """
        Execute batch document processing.
        
        Args:
            input_data: Validated input parameters
            
        Returns:
            Batch processing results
        """
        await self.initialize()
        
        try:
            # Validate each document configuration
            validated_docs = []
            for doc_config in input_data.documents:
                # Ensure required fields are present
                if not all(k in doc_config for k in ['document_path', 'rulebook_name', 'system']):
                    raise ValueError(f"Missing required fields in document config: {doc_config}")
                
                # Rename 'document_path' to 'pdf_path' for backward compatibility
                if 'document_path' in doc_config:
                    doc_config['pdf_path'] = doc_config.pop('document_path')
                
                validated_docs.append(doc_config)
            
            result = await self.pipeline.process_multiple_pdfs(
                pdf_files=validated_docs,
                enable_adaptive_learning=input_data.enable_adaptive_learning,
                max_workers=input_data.max_workers,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Batch document processing failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "results": [],
                "total": 0,
                "successful": 0,
                "failed": 0,
            }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": BatchProcessDocumentsInput.schema(),
        }


class ExtractDocumentTextTool:
    """MCP tool for extracting text from documents without full processing."""
    
    name = "extract_document_text"
    description = "Extract text, tables, and metadata from a document without processing for embeddings"
    
    def __init__(self):
        self.parser = None
    
    async def initialize(self):
        """Initialize the document parser."""
        if not self.parser:
            from src.pdf_processing.document_parser import UnifiedDocumentParser
            self.parser = UnifiedDocumentParser()
    
    async def execute(self, input_data: ExtractDocumentTextInput) -> DocumentExtractionResult:
        """
        Execute document text extraction.
        
        Args:
            input_data: Validated input parameters
            
        Returns:
            Extraction results
        """
        await self.initialize()
        
        try:
            # Extract content
            content = await asyncio.to_thread(
                self.parser.extract_text_from_document,
                input_data.document_path,
                skip_size_check=input_data.skip_size_check,
                user_confirmed=input_data.user_confirmed,
            )
            
            # Calculate total characters
            total_chars = sum(page.get('char_count', 0) for page in content.get('pages', []))
            
            # Prepare result
            result = DocumentExtractionResult(
                success=True,
                file_name=content['file_name'],
                file_hash=content['file_hash'],
                document_type=content.get('document_type', 'unknown'),
                total_pages=content['total_pages'],
                total_characters=total_chars,
            )
            
            if input_data.extract_metadata:
                result.metadata = content.get('metadata', {})
            
            if input_data.extract_tables:
                result.tables = content.get('tables', [])
            
            # Include TOC if available
            if 'toc' in content:
                result.toc = content['toc']
            
            return result
            
        except Exception as e:
            logger.error(f"Document extraction failed: {e}", exc_info=True)
            return DocumentExtractionResult(
                success=False,
                file_name=Path(input_data.document_path).name,
                file_hash="",
                document_type="unknown",
                total_pages=0,
                total_characters=0,
                error=str(e),
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": ExtractDocumentTextInput.schema(),
            "output_schema": DocumentExtractionResult.schema(),
        }


class GetSupportedFormatsTool:
    """MCP tool for getting supported document formats."""
    
    name = "get_supported_formats"
    description = "Get list of supported document formats for processing"
    
    async def execute(self) -> Dict[str, List[str]]:
        """
        Get supported formats.
        
        Returns:
            Dictionary with supported formats
        """
        return {
            "supported_formats": [".pdf", ".epub", ".mobi", ".azw", ".azw3"],
            "primary_formats": [".pdf", ".epub", ".mobi"],
            "kindle_formats": [".azw", ".azw3"],
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {},
            "output_schema": {
                "type": "object",
                "properties": {
                    "supported_formats": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "primary_formats": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "kindle_formats": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                }
            }
        }


# Registry of available MCP tools
MCP_DOCUMENT_TOOLS = {
    "process_document": ProcessDocumentTool,
    "batch_process_documents": BatchProcessDocumentsTool,
    "extract_document_text": ExtractDocumentTextTool,
    "get_supported_formats": GetSupportedFormatsTool,
}


def get_tool(tool_name: str) -> Optional[Any]:
    """
    Get an MCP tool instance by name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool instance or None if not found
    """
    tool_class = MCP_DOCUMENT_TOOLS.get(tool_name)
    if tool_class:
        return tool_class()
    return None


def list_tools() -> List[Dict[str, str]]:
    """
    List all available MCP document processing tools.
    
    Returns:
        List of tool information
    """
    tools = []
    for name, tool_class in MCP_DOCUMENT_TOOLS.items():
        tool = tool_class()
        tools.append({
            "name": tool.name,
            "description": tool.description,
        })
    return tools