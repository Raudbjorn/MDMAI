"""MCP (Model Context Protocol) tools for document processing.

This module provides MCP-compliant tools for processing various document formats
including PDF, EPUB, and MOBI files for TTRPG content extraction and indexing.
"""

from .document_tools import (
    ProcessDocumentTool,
    BatchProcessDocumentsTool,
    ExtractDocumentTextTool,
    GetSupportedFormatsTool,
    MCP_DOCUMENT_TOOLS,
    get_tool,
    list_tools,
    ProcessDocumentInput,
    BatchProcessDocumentsInput,
    ExtractDocumentTextInput,
    DocumentProcessingResult,
    DocumentExtractionResult,
)

__all__ = [
    # Tools
    "ProcessDocumentTool",
    "BatchProcessDocumentsTool",
    "ExtractDocumentTextTool",
    "GetSupportedFormatsTool",
    
    # Registry and utilities
    "MCP_DOCUMENT_TOOLS",
    "get_tool",
    "list_tools",
    
    # Input schemas
    "ProcessDocumentInput",
    "BatchProcessDocumentsInput",
    "ExtractDocumentTextInput",
    
    # Output schemas
    "DocumentProcessingResult",
    "DocumentExtractionResult",
]