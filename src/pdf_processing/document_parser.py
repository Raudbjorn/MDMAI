"""Unified document parser interface for multiple document formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def extract_text_from_document(
        self, 
        document_path: str, 
        skip_size_check: bool = False, 
        user_confirmed: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text content from a document file.

        Args:
            document_path: Path to the document file
            skip_size_check: Skip file size validation
            user_confirmed: Whether user has already confirmed large file processing

        Returns:
            Dictionary containing extracted content and metadata
        """
        pass

    @abstractmethod
    def extract_tables_as_markdown(self, tables: list[Dict]) -> list[str]:
        """
        Convert extracted tables to markdown format.

        Args:
            tables: List of table dictionaries

        Returns:
            List of markdown-formatted tables
        """
        pass

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported file formats.

        Returns:
            List of supported file extensions
        """
        return []


class DocumentParserProtocol(Protocol):
    """Protocol defining the interface for document parsers."""

    def extract_text_from_document(
        self, 
        document_path: str, 
        skip_size_check: bool = False, 
        user_confirmed: bool = False
    ) -> Dict[str, Any]:
        """Extract text from document."""
        ...

    def extract_tables_as_markdown(self, tables: list[Dict]) -> list[str]:
        """Convert tables to markdown."""
        ...


class UnifiedDocumentParser:
    """
    Unified parser that delegates to appropriate format-specific parser.
    
    This class provides a single interface for parsing multiple document formats
    including PDF, EPUB, and MOBI files.
    """

    def __init__(self):
        """Initialize the unified document parser with format-specific parsers."""
        from src.pdf_processing.pdf_parser import PDFParser
        from src.pdf_processing.ebook_parser import EbookParser
        
        self.pdf_parser = PDFParser()
        self.ebook_parser = EbookParser()
        
        # Map file extensions to parsers
        self.parser_map = {
            '.pdf': self.pdf_parser,
            '.epub': self.ebook_parser,
            '.mobi': self.ebook_parser,
            '.azw': self.ebook_parser,
            '.azw3': self.ebook_parser,
        }

    def extract_text_from_document(
        self, 
        document_path: str, 
        skip_size_check: bool = False, 
        user_confirmed: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text content from a document file.

        Automatically detects the file format and uses the appropriate parser.

        Args:
            document_path: Path to the document file
            skip_size_check: Skip file size validation
            user_confirmed: Whether user has already confirmed large file processing

        Returns:
            Dictionary containing extracted content and metadata
        """
        path = Path(document_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {document_path}")
        
        # Get file extension
        file_extension = path.suffix.lower()
        
        # Select appropriate parser
        parser = self.parser_map.get(file_extension)
        
        if not parser:
            raise ValueError(
                f"Unsupported document format: {file_extension}. "
                f"Supported formats: {', '.join(self.parser_map.keys())}"
            )
        
        # Extract content using the appropriate parser
        if file_extension == '.pdf':
            result = parser.extract_text_from_pdf(
                document_path, 
                skip_size_check=skip_size_check,
                user_confirmed=user_confirmed
            )
        else:
            # Ebook formats
            result = parser.extract_text_from_ebook(
                document_path,
                skip_size_check=skip_size_check,
                user_confirmed=user_confirmed
            )
        
        # Add document type to result
        result['document_type'] = file_extension[1:]  # Remove the dot
        
        return result

    def extract_tables_as_markdown(self, tables: list[Dict]) -> list[str]:
        """
        Convert extracted tables to markdown format.

        Args:
            tables: List of table dictionaries

        Returns:
            List of markdown-formatted tables
        """
        # Tables have the same format regardless of source document type
        # So we can use either parser
        return self.pdf_parser.extract_tables_as_markdown(tables)

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported file formats.

        Returns:
            List of supported file extensions
        """
        return list(self.parser_map.keys())

    def is_format_supported(self, file_path: str) -> bool:
        """
        Check if a file format is supported.

        Args:
            file_path: Path to the file

        Returns:
            True if the format is supported, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in self.parser_map