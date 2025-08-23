"""PDF parsing and extraction module for TTRPG Assistant."""

import hashlib
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
import PyPDF2
from PyPDF2 import PdfReader

from config.logging_config import get_logger
from config.settings import settings
from src.utils.exceptions import (
    InvalidPDFError,
    PDFParseError,
    PDFProcessingException as PDFProcessingError,
    PDFReadError,
    PDFSizeError,
)
from src.utils.file_size_handler import FileSizeCategory, FileSizeHandler

logger = get_logger(__name__)


class PDFParser:
    """Handles PDF parsing and content extraction."""

    def __init__(self):
        """Initialize PDF parser."""
        self.current_file = None
        self.metadata = {}
        self.content_pages = []

    def extract_text_from_pdf(
        self, pdf_path: str, skip_size_check: bool = False, user_confirmed: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing extracted content and metadata
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

        # Check file size and get recommendations
        if not skip_size_check:
            file_info = FileSizeHandler.get_file_info(pdf_path)
            category = FileSizeCategory(file_info["category"])

            # For excessive files, require explicit confirmation
            if category == FileSizeCategory.EXCESSIVE and not user_confirmed:
                raise PDFSizeError(
                    f"PDF file is excessively large: {file_info['size_formatted']}. "
                    f"User confirmation required for files over 500MB.",
                    {
                        "file_size": file_info["size_bytes"],
                        "size_formatted": file_info["size_formatted"],
                        "path": str(pdf_path),
                        "requires_confirmation": True,
                        "estimated_time": file_info["estimated_time"],
                    },
                )

            # Log warning for large files
            if category in [FileSizeCategory.LARGE, FileSizeCategory.VERY_LARGE]:
                logger.warning(
                    f"Processing large PDF: {file_info['size_formatted']} "
                    f"(estimated time: {file_info['estimated_time']})"
                )

        logger.info(f"Starting PDF extraction: {pdf_path}")

        try:
            # Extract with both libraries for better coverage
            pypdf_content = self._extract_with_pypdf(pdf_path)
            plumber_content = self._extract_with_pdfplumber(pdf_path)

            # Merge results, preferring pdfplumber for tables
            result = {
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "file_hash": self._calculate_file_hash(pdf_path),
                "metadata": pypdf_content["metadata"],
                "total_pages": pypdf_content["total_pages"],
                "pages": self._merge_page_content(pypdf_content["pages"], plumber_content["pages"]),
                "tables": plumber_content.get("tables", []),
                "toc": self._extract_table_of_contents(pypdf_content),
            }

            logger.info(
                f"PDF extraction complete",
                file=pdf_path.name,
                pages=result["total_pages"],
                tables=len(result["tables"]),
            )

            return result

        except FileNotFoundError as e:
            raise PDFReadError(f"PDF file not found: {pdf_path}", {"path": str(pdf_path)}) from e
        except PermissionError as e:
            logger.error(f"Permission denied accessing PDF: {e}")
            raise PDFReadError(f"Permission denied: {pdf_path}", {"path": str(pdf_path)}) from e
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"Invalid PDF format: {e}")
            raise InvalidPDFError(f"Invalid PDF format: {pdf_path}", {"path": str(pdf_path)}) from e
        except Exception as e:
            logger.error(f"Unexpected error processing PDF: {e}")
            raise PDFProcessingError(
                f"Failed to process {pdf_path}: {e}", {"path": str(pdf_path)}
            ) from e

    def _extract_with_pypdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract content using PyPDF2.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted content dictionary
        """
        try:
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)

                # Extract metadata
                metadata = {
                    "title": reader.metadata.get("/Title", "") if reader.metadata else "",
                    "author": reader.metadata.get("/Author", "") if reader.metadata else "",
                    "subject": reader.metadata.get("/Subject", "") if reader.metadata else "",
                    "creator": reader.metadata.get("/Creator", "") if reader.metadata else "",
                    "producer": reader.metadata.get("/Producer", "") if reader.metadata else "",
                    "creation_date": (
                        str(reader.metadata.get("/CreationDate", "")) if reader.metadata else ""
                    ),
                    "modification_date": (
                        str(reader.metadata.get("/ModDate", "")) if reader.metadata else ""
                    ),
                }

                # Extract text from each page
                pages = []
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    pages.append(
                        {
                            "page_number": page_num,
                            "text": text,
                            "char_count": len(text),
                        }
                    )

                # Try to extract outline (TOC)
                outline = []
                if reader.outline:
                    outline = self._process_outline(reader.outline)

                return {
                    "metadata": metadata,
                    "total_pages": len(reader.pages),
                    "pages": pages,
                    "outline": outline,
                }

        except Exception as e:
            logger.error(f"PyPDF2 extraction failed", error=str(e))
            return {
                "metadata": {},
                "total_pages": 0,
                "pages": [],
                "outline": [],
            }

    def _extract_with_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract content using pdfplumber (better for tables).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted content dictionary
        """
        try:
            pages = []
            tables = []

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text() or ""

                    # Extract tables from this page
                    page_tables = page.extract_tables()

                    if page_tables:
                        for table_idx, table in enumerate(page_tables):
                            tables.append(
                                {
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "data": table,
                                    "rows": len(table),
                                    "cols": len(table[0]) if table else 0,
                                }
                            )

                    pages.append(
                        {
                            "page_number": page_num,
                            "text": text,
                            "char_count": len(text),
                            "has_tables": len(page_tables) > 0,
                        }
                    )

            return {
                "pages": pages,
                "tables": tables,
            }

        except Exception as e:
            logger.error(f"pdfplumber extraction failed", error=str(e))
            return {
                "pages": [],
                "tables": [],
            }

    def _merge_page_content(
        self, pypdf_pages: List[Dict], plumber_pages: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Merge content from both extraction methods.

        Args:
            pypdf_pages: Pages extracted with PyPDF2
            plumber_pages: Pages extracted with pdfplumber

        Returns:
            Merged page content
        """
        merged = []

        for i in range(max(len(pypdf_pages), len(plumber_pages))):
            page_data = {
                "page_number": i + 1,
                "text": "",
                "char_count": 0,
                "has_tables": False,
            }

            # Get text from both sources, prefer longer text
            pypdf_text = pypdf_pages[i]["text"] if i < len(pypdf_pages) else ""
            plumber_text = plumber_pages[i]["text"] if i < len(plumber_pages) else ""

            # Use the extraction with more content
            if len(plumber_text) > len(pypdf_text):
                page_data["text"] = plumber_text
            else:
                page_data["text"] = pypdf_text

            page_data["char_count"] = len(page_data["text"])

            # Add table flag from pdfplumber
            if i < len(plumber_pages):
                page_data["has_tables"] = plumber_pages[i].get("has_tables", False)

            merged.append(page_data)

        return merged

    def _extract_table_of_contents(self, pypdf_content: Dict) -> List[Dict[str, Any]]:
        """
        Extract table of contents from PDF outline.

        Args:
            pypdf_content: Content extracted with PyPDF2

        Returns:
            List of TOC entries
        """
        toc = []

        if "outline" in pypdf_content and pypdf_content["outline"]:
            for entry in pypdf_content["outline"]:
                if isinstance(entry, dict):
                    toc.append(entry)

        return toc

    def _process_outline(self, outline, level=0) -> List[Dict[str, Any]]:
        """
        Process PDF outline recursively.

        Args:
            outline: PDF outline object
            level: Current nesting level

        Returns:
            Processed outline entries
        """
        result = []

        for item in outline:
            if isinstance(item, list):
                # Nested outline
                result.extend(self._process_outline(item, level + 1))
            elif hasattr(item, "title"):
                entry = {
                    "title": item.title,
                    "level": level,
                }
                # Try to get page number
                if hasattr(item, "page") and hasattr(item.page, "idnum"):
                    entry["page"] = item.page.idnum
                result.append(entry)

        return result

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA256 hash of file for deduplication.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of file hash
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot calculate hash: {file_path} not found")

        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except IOError as e:
            logger.error(f"Failed to calculate file hash: {e}")
            raise PDFProcessingError(f"Cannot hash file {file_path}: {e}") from e

    def extract_tables_as_markdown(self, tables: List[Dict]) -> List[str]:
        """
        Convert extracted tables to markdown format.

        Args:
            tables: List of table dictionaries

        Returns:
            List of markdown-formatted tables
        """
        markdown_tables = []

        for table_info in tables:
            table = table_info["data"]
            if not table or not table[0]:
                continue

            # Build markdown table
            md_lines = []

            # Header row
            header = "| " + " | ".join(str(cell) if cell else "" for cell in table[0]) + " |"
            md_lines.append(header)

            # Separator row
            separator = "|" + "|".join([" --- " for _ in table[0]]) + "|"
            md_lines.append(separator)

            # Data rows
            for row in table[1:]:
                row_text = "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |"
                md_lines.append(row_text)

            markdown_tables.append("\n".join(md_lines))

        return markdown_tables
