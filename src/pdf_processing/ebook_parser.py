"""Ebook (EPUB and MOBI) parsing and extraction module for TTRPG Assistant."""

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

try:
    import ebooklib
    from ebooklib import epub
    EBOOKLIB_AVAILABLE = True
except ImportError:
    EBOOKLIB_AVAILABLE = False

try:
    import mobi
    MOBI_AVAILABLE = True
except ImportError:
    MOBI_AVAILABLE = False

from bs4 import BeautifulSoup

from config.logging_config import get_logger
from src.utils.exceptions import (
    InvalidPDFError as InvalidEbookError,
    PDFProcessingException as EbookProcessingError,
    PDFReadError as EbookReadError,
    PDFSizeError as EbookSizeError,
)
from src.utils.file_size_handler import FileSizeCategory, FileSizeHandler

logger = get_logger(__name__)


class EbookParser:
    """Handles ebook (EPUB and MOBI) parsing and content extraction."""

    def __init__(self):
        """Initialize ebook parser."""
        self.current_file = None
        self.metadata = {}
        self.content_pages = []
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required libraries are available."""
        if not EBOOKLIB_AVAILABLE:
            logger.warning("ebooklib not available. EPUB support will be limited.")
        if not MOBI_AVAILABLE:
            logger.warning("mobi library not available. MOBI support will be limited.")

    def extract_text_from_ebook(
        self, ebook_path: str, skip_size_check: bool = False, user_confirmed: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text content from an ebook file (EPUB or MOBI).

        Args:
            ebook_path: Path to the ebook file
            skip_size_check: Skip file size validation
            user_confirmed: Whether user has already confirmed large file processing

        Returns:
            Dictionary containing extracted content and metadata
        """
        ebook_path = Path(ebook_path)

        if not ebook_path.exists():
            raise FileNotFoundError(f"Ebook file not found: {ebook_path}")

        file_extension = ebook_path.suffix.lower()
        if file_extension not in ['.epub', '.mobi', '.azw', '.azw3']:
            raise ValueError(f"File is not a supported ebook format: {ebook_path}")

        # Check file size and get recommendations
        if not skip_size_check:
            file_info = FileSizeHandler.get_file_info(ebook_path)
            category = FileSizeCategory(file_info["category"])

            # For excessive files, require explicit confirmation
            if category == FileSizeCategory.EXCESSIVE and not user_confirmed:
                raise EbookSizeError(
                    f"Ebook file is excessively large: {file_info['size_formatted']}. "
                    f"User confirmation required for files over 500MB.",
                    {
                        "file_size": file_info["size_bytes"],
                        "size_formatted": file_info["size_formatted"],
                        "path": str(ebook_path),
                        "requires_confirmation": True,
                        "estimated_time": file_info["estimated_time"],
                    },
                )

            # Log warning for large files
            if category in [FileSizeCategory.LARGE, FileSizeCategory.VERY_LARGE]:
                logger.warning(
                    f"Processing large ebook: {file_info['size_formatted']} "
                    f"(estimated time: {file_info['estimated_time']})"
                )

        logger.info(f"Starting ebook extraction: {ebook_path}")

        try:
            if file_extension == '.epub':
                return self._extract_epub(ebook_path)
            elif file_extension in ['.mobi', '.azw', '.azw3']:
                return self._extract_mobi(ebook_path)
            else:
                raise ValueError(f"Unsupported ebook format: {file_extension}")

        except FileNotFoundError as e:
            raise EbookReadError(f"Ebook file not found: {ebook_path}", {"path": str(ebook_path)}) from e
        except PermissionError as e:
            logger.error(f"Permission denied accessing ebook: {e}")
            raise EbookReadError(f"Permission denied: {ebook_path}", {"path": str(ebook_path)}) from e
        except Exception as e:
            logger.error(f"Unexpected error processing ebook: {e}")
            raise EbookProcessingError(
                f"Failed to process {ebook_path}: {e}", {"path": str(ebook_path)}
            ) from e

    def _extract_epub(self, epub_path: Path) -> Dict[str, Any]:
        """
        Extract content from EPUB file.

        Args:
            epub_path: Path to EPUB file

        Returns:
            Extracted content dictionary
        """
        if not EBOOKLIB_AVAILABLE:
            return self._extract_epub_fallback(epub_path)

        try:
            book = epub.read_epub(str(epub_path))
            
            # Extract metadata
            metadata = self._extract_epub_metadata(book)
            
            # Extract content
            pages = []
            tables = []
            images = []
            toc = []
            
            # Extract table of contents
            toc = self._extract_epub_toc(book)
            
            # Extract all items
            page_num = 1
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    
                    # Extract text
                    text = soup.get_text(separator='\n', strip=True)
                    
                    # Extract tables from HTML
                    html_tables = soup.find_all('table')
                    for table_idx, table in enumerate(html_tables):
                        table_data = self._parse_html_table(table)
                        if table_data:
                            tables.append({
                                "page": page_num,
                                "table_index": table_idx,
                                "data": table_data,
                                "rows": len(table_data),
                                "cols": len(table_data[0]) if table_data else 0,
                            })
                    
                    # Extract images
                    for img in soup.find_all('img'):
                        img_src = img.get('src', '')
                        if img_src:
                            images.append({
                                "page": page_num,
                                "src": img_src,
                                "alt": img.get('alt', ''),
                            })
                    
                    if text:
                        pages.append({
                            "page_number": page_num,
                            "text": text,
                            "char_count": len(text),
                            "has_tables": len(html_tables) > 0,
                            "has_images": len(soup.find_all('img')) > 0,
                        })
                        page_num += 1
            
            result = {
                "file_path": str(epub_path),
                "file_name": epub_path.name,
                "file_hash": self._calculate_file_hash(epub_path),
                "metadata": metadata,
                "total_pages": len(pages),
                "pages": pages,
                "tables": tables,
                "images": images,
                "toc": toc,
                "format": "epub",
            }
            
            logger.info(
                "EPUB extraction complete",
                file=epub_path.name,
                pages=result["total_pages"],
                tables=len(result["tables"]),
                images=len(result["images"]),
            )
            
            return result

        except Exception as e:
            logger.error(f"Error extracting EPUB with ebooklib: {e}")
            # Fall back to basic extraction
            return self._extract_epub_fallback(epub_path)

    def _extract_epub_fallback(self, epub_path: Path) -> Dict[str, Any]:
        """
        Fallback EPUB extraction using basic ZIP reading.

        Args:
            epub_path: Path to EPUB file

        Returns:
            Extracted content dictionary
        """
        pages = []
        metadata = {}
        tables = []
        toc = []
        
        try:
            with zipfile.ZipFile(epub_path, 'r') as zip_file:
                # Try to extract metadata from content.opf
                for file_name in zip_file.namelist():
                    if file_name.endswith('.opf'):
                        with zip_file.open(file_name) as opf_file:
                            opf_content = opf_file.read().decode('utf-8', errors='ignore')
                            metadata = self._parse_opf_metadata(opf_content)
                        break
                
                # Extract HTML/XHTML content
                page_num = 1
                html_files = sorted([f for f in zip_file.namelist() 
                                   if f.endswith(('.html', '.xhtml', '.htm'))])
                
                for html_file in html_files:
                    with zip_file.open(html_file) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Extract text
                        text = soup.get_text(separator='\n', strip=True)
                        
                        # Extract tables
                        html_tables = soup.find_all('table')
                        for table_idx, table in enumerate(html_tables):
                            table_data = self._parse_html_table(table)
                            if table_data:
                                tables.append({
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "data": table_data,
                                    "rows": len(table_data),
                                    "cols": len(table_data[0]) if table_data else 0,
                                })
                        
                        if text:
                            pages.append({
                                "page_number": page_num,
                                "text": text,
                                "char_count": len(text),
                                "has_tables": len(html_tables) > 0,
                            })
                            page_num += 1
        
        except Exception as e:
            logger.error(f"Error in EPUB fallback extraction: {e}")
            raise
        
        return {
            "file_path": str(epub_path),
            "file_name": epub_path.name,
            "file_hash": self._calculate_file_hash(epub_path),
            "metadata": metadata,
            "total_pages": len(pages),
            "pages": pages,
            "tables": tables,
            "toc": toc,
            "format": "epub",
        }

    def _extract_mobi(self, mobi_path: Path) -> Dict[str, Any]:
        """
        Extract content from MOBI file.

        Args:
            mobi_path: Path to MOBI file

        Returns:
            Extracted content dictionary
        """
        pages = []
        metadata = {}
        tables = []
        toc = []
        
        try:
            # If mobi library is available, use it
            if MOBI_AVAILABLE:
                tempdir, html_file = mobi.extract(str(mobi_path))
                
                # Read the extracted HTML
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                
                # Parse HTML content
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract metadata from HTML if available
                if soup.find('meta', {'name': 'Author'}):
                    metadata['author'] = soup.find('meta', {'name': 'Author'}).get('content', '')
                if soup.find('title'):
                    metadata['title'] = soup.find('title').get_text()
                
                # Process content
                pages, tables = self._process_html_content(soup)
                
                # Clean up temporary files
                import shutil
                shutil.rmtree(tempdir)
            else:
                # Fallback: try to extract as much as possible from raw MOBI
                pages, metadata = self._extract_mobi_fallback(mobi_path)
        
        except Exception as e:
            logger.error(f"Error extracting MOBI: {e}")
            # Use fallback extraction
            pages, metadata = self._extract_mobi_fallback(mobi_path)
        
        return {
            "file_path": str(mobi_path),
            "file_name": mobi_path.name,
            "file_hash": self._calculate_file_hash(mobi_path),
            "metadata": metadata,
            "total_pages": len(pages),
            "pages": pages,
            "tables": tables,
            "toc": toc,
            "format": "mobi",
        }

    def _extract_mobi_fallback(self, mobi_path: Path) -> Tuple[List[Dict], Dict]:
        """
        Fallback MOBI extraction using basic file reading.

        Args:
            mobi_path: Path to MOBI file

        Returns:
            Tuple of (pages, metadata)
        """
        pages = []
        metadata = {}
        
        try:
            with open(mobi_path, 'rb') as f:
                content = f.read()
                
                # Try to extract some basic text (MOBI files often have readable text)
                text_content = content.decode('utf-8', errors='ignore')
                
                # Clean up the text
                text_content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text_content)
                text_content = re.sub(r'\s+', ' ', text_content)
                
                # Split into chunks (approximating pages)
                chunk_size = 3000  # Approximate characters per page
                chunks = [text_content[i:i+chunk_size] 
                         for i in range(0, len(text_content), chunk_size)]
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        pages.append({
                            "page_number": i + 1,
                            "text": chunk,
                            "char_count": len(chunk),
                            "has_tables": False,
                        })
                
                # Try to extract title from content
                title_match = re.search(r'^(.{1,100})', text_content.strip())
                if title_match:
                    metadata['title'] = title_match.group(1).strip()
        
        except Exception as e:
            logger.error(f"Error in MOBI fallback extraction: {e}")
        
        return pages, metadata

    def _process_html_content(self, soup: BeautifulSoup) -> Tuple[List[Dict], List[Dict]]:
        """
        Process HTML content from ebook.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Tuple of (pages, tables)
        """
        pages = []
        tables = []
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Split content into sections (using headers as breaks)
        sections = []
        current_section = []
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'table']):
            if element.name in ['h1', 'h2', 'h3'] and current_section:
                # Start new section
                sections.append(current_section)
                current_section = [element]
            else:
                current_section.append(element)
        
        if current_section:
            sections.append(current_section)
        
        # Process sections into pages
        page_num = 1
        for section in sections:
            text_parts = []
            section_tables = []
            
            for element in section:
                if element.name == 'table':
                    # Extract table
                    table_data = self._parse_html_table(element)
                    if table_data:
                        section_tables.append(table_data)
                        text_parts.append(f"[Table with {len(table_data)} rows]")
                else:
                    text = element.get_text(separator=' ', strip=True)
                    if text:
                        text_parts.append(text)
            
            # Combine text
            full_text = '\n'.join(text_parts)
            
            # Add tables to global list
            for idx, table_data in enumerate(section_tables):
                tables.append({
                    "page": page_num,
                    "table_index": idx,
                    "data": table_data,
                    "rows": len(table_data),
                    "cols": len(table_data[0]) if table_data else 0,
                })
            
            if full_text.strip():
                pages.append({
                    "page_number": page_num,
                    "text": full_text,
                    "char_count": len(full_text),
                    "has_tables": len(section_tables) > 0,
                })
                page_num += 1
        
        return pages, tables

    def _parse_html_table(self, table_element) -> List[List[str]]:
        """
        Parse HTML table element into data structure.

        Args:
            table_element: BeautifulSoup table element

        Returns:
            List of rows, each row is a list of cell values
        """
        table_data = []
        
        try:
            rows = table_element.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if row_data:
                    table_data.append(row_data)
        except Exception as e:
            logger.debug(f"Error parsing table: {e}")
        
        return table_data

    def _extract_epub_metadata(self, book) -> Dict[str, str]:
        """
        Extract metadata from EPUB book object.

        Args:
            book: ebooklib EPUB book object

        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        try:
            # Standard Dublin Core metadata
            metadata['title'] = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else ''
            metadata['author'] = book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else ''
            metadata['publisher'] = book.get_metadata('DC', 'publisher')[0][0] if book.get_metadata('DC', 'publisher') else ''
            metadata['language'] = book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else ''
            metadata['subject'] = book.get_metadata('DC', 'subject')[0][0] if book.get_metadata('DC', 'subject') else ''
            metadata['date'] = book.get_metadata('DC', 'date')[0][0] if book.get_metadata('DC', 'date') else ''
            metadata['identifier'] = book.get_metadata('DC', 'identifier')[0][0] if book.get_metadata('DC', 'identifier') else ''
        except Exception as e:
            logger.debug(f"Error extracting EPUB metadata: {e}")
        
        return metadata

    def _parse_opf_metadata(self, opf_content: str) -> Dict[str, str]:
        """
        Parse metadata from OPF (Open Packaging Format) file.

        Args:
            opf_content: OPF file content as string

        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        try:
            root = ET.fromstring(opf_content)
            
            # Define namespaces
            namespaces = {
                'dc': 'http://purl.org/dc/elements/1.1/',
                'opf': 'http://www.idpf.org/2007/opf'
            }
            
            # Extract metadata
            for field in ['title', 'creator', 'publisher', 'language', 'subject', 'date', 'identifier']:
                element = root.find(f'.//dc:{field}', namespaces)
                if element is not None and element.text:
                    metadata[field if field != 'creator' else 'author'] = element.text
        
        except Exception as e:
            logger.debug(f"Error parsing OPF metadata: {e}")
        
        return metadata

    def _extract_epub_toc(self, book) -> List[Dict[str, Any]]:
        """
        Extract table of contents from EPUB book.

        Args:
            book: ebooklib EPUB book object

        Returns:
            List of TOC entries
        """
        toc = []
        
        try:
            for item in book.toc:
                if isinstance(item, tuple):
                    # Handle nested TOC structure
                    section, children = item
                    toc.append({
                        'title': section.title,
                        'href': section.href,
                        'level': 0,
                    })
                    for child in children:
                        toc.append({
                            'title': child.title,
                            'href': child.href,
                            'level': 1,
                        })
                else:
                    # Simple TOC item
                    toc.append({
                        'title': item.title,
                        'href': item.href,
                        'level': 0,
                    })
        except Exception as e:
            logger.debug(f"Error extracting EPUB TOC: {e}")
        
        return toc

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
            raise EbookProcessingError(f"Cannot hash file {file_path}: {e}") from e

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