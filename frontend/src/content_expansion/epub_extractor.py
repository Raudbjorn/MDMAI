"""
EPUB content extractor for ebook processing.

This module provides functionality to extract and parse content from EPUB files,
including text, metadata, and structural information.
"""

import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from xml.etree import ElementTree as ET
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EPUBMetadata:
    """Container for EPUB metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    language: Optional[str] = None
    isbn: Optional[str] = None
    publication_date: Optional[str] = None
    description: Optional[str] = None
    subjects: List[str] = field(default_factory=list)
    contributors: List[str] = field(default_factory=list)


@dataclass
class EPUBChapter:
    """Container for EPUB chapter content."""
    title: str
    content: str
    order: int
    file_path: str
    
    
class EPUBExtractor:
    """Extract content from EPUB files."""
    
    NAMESPACES = {
        'opf': 'http://www.idpf.org/2007/opf',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'xhtml': 'http://www.w3.org/1999/xhtml',
        'epub': 'http://www.idpf.org/2007/ops'
    }
    
    def __init__(self, file_path: Path):
        """
        Initialize EPUB extractor.
        
        Args:
            file_path: Path to the EPUB file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {file_path}")
        if not zipfile.is_zipfile(self.file_path):
            raise ValueError(f"File is not a valid EPUB (ZIP) file: {file_path}")
            
        self.zip_file: Optional[zipfile.ZipFile] = None
        self.opf_path: Optional[str] = None
        self.opf_dir: Optional[str] = None
        self.metadata: EPUBMetadata = EPUBMetadata()
        self.chapters: List[EPUBChapter] = []
        
    def __enter__(self):
        """Context manager entry."""
        self.zip_file = zipfile.ZipFile(self.file_path, 'r')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.zip_file:
            self.zip_file.close()
            
    def extract(self) -> Dict[str, Any]:
        """
        Extract all content from the EPUB file.
        
        Returns:
            Dictionary containing metadata and chapter content
        """
        with self:
            self._find_opf_file()
            self._parse_metadata()
            self._extract_chapters()
            
        return {
            'metadata': self._metadata_to_dict(),
            'chapters': [
                {
                    'title': chapter.title,
                    'content': chapter.content,
                    'order': chapter.order
                }
                for chapter in sorted(self.chapters, key=lambda x: x.order)
            ],
            'total_chapters': len(self.chapters)
        }
        
    def _find_opf_file(self):
        """Find the OPF (Open Packaging Format) file in the EPUB."""
        # First, try to find META-INF/container.xml
        try:
            container_xml = self.zip_file.read('META-INF/container.xml')
            root = ET.fromstring(container_xml)
            
            # Find the OPF file path
            for rootfile in root.findall('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile'):
                if rootfile.get('media-type') == 'application/oebps-package+xml':
                    self.opf_path = rootfile.get('full-path')
                    self.opf_dir = str(Path(self.opf_path).parent)
                    if self.opf_dir == '.':
                        self.opf_dir = ''
                    break
        except KeyError:
            # Fallback: look for .opf file
            for file_name in self.zip_file.namelist():
                if file_name.endswith('.opf'):
                    self.opf_path = file_name
                    self.opf_dir = str(Path(file_name).parent)
                    if self.opf_dir == '.':
                        self.opf_dir = ''
                    break
                    
        if not self.opf_path:
            raise ValueError("Could not find OPF file in EPUB")
            
    def _parse_metadata(self):
        """Parse metadata from the OPF file."""
        opf_content = self.zip_file.read(self.opf_path)
        root = ET.fromstring(opf_content)
        
        # Parse Dublin Core metadata
        metadata_elem = root.find('.//opf:metadata', self.NAMESPACES)
        if metadata_elem is not None:
            # Title
            title_elem = metadata_elem.find('dc:title', self.NAMESPACES)
            if title_elem is not None:
                self.metadata.title = title_elem.text
                
            # Author(s)
            creator_elem = metadata_elem.find('dc:creator', self.NAMESPACES)
            if creator_elem is not None:
                self.metadata.author = creator_elem.text
                
            # Publisher
            publisher_elem = metadata_elem.find('dc:publisher', self.NAMESPACES)
            if publisher_elem is not None:
                self.metadata.publisher = publisher_elem.text
                
            # Language
            language_elem = metadata_elem.find('dc:language', self.NAMESPACES)
            if language_elem is not None:
                self.metadata.language = language_elem.text
                
            # ISBN
            for identifier_elem in metadata_elem.findall('dc:identifier', self.NAMESPACES):
                if identifier_elem.get('{http://www.idpf.org/2007/opf}scheme') == 'ISBN':
                    self.metadata.isbn = identifier_elem.text
                    break
                    
            # Publication date
            date_elem = metadata_elem.find('dc:date', self.NAMESPACES)
            if date_elem is not None:
                self.metadata.publication_date = date_elem.text
                
            # Description
            description_elem = metadata_elem.find('dc:description', self.NAMESPACES)
            if description_elem is not None:
                self.metadata.description = description_elem.text
                
            # Subjects
            for subject_elem in metadata_elem.findall('dc:subject', self.NAMESPACES):
                if subject_elem.text:
                    self.metadata.subjects.append(subject_elem.text)
                    
            # Contributors
            for contributor_elem in metadata_elem.findall('dc:contributor', self.NAMESPACES):
                if contributor_elem.text:
                    self.metadata.contributors.append(contributor_elem.text)
                    
    def _extract_chapters(self):
        """Extract chapter content from the EPUB."""
        opf_content = self.zip_file.read(self.opf_path)
        root = ET.fromstring(opf_content)
        
        # Get spine (reading order)
        spine = root.find('.//opf:spine', self.NAMESPACES)
        if spine is None:
            raise ValueError("Could not find spine in OPF")
            
        # Get manifest (file listings)
        manifest = root.find('.//opf:manifest', self.NAMESPACES)
        if manifest is None:
            raise ValueError("Could not find manifest in OPF")
            
        # Create id to href mapping
        id_to_href = {}
        for item in manifest.findall('opf:item', self.NAMESPACES):
            item_id = item.get('id')
            href = item.get('href')
            if item_id and href:
                id_to_href[item_id] = href
                
        # Extract chapters in spine order
        for order, itemref in enumerate(spine.findall('opf:itemref', self.NAMESPACES)):
            idref = itemref.get('idref')
            if idref in id_to_href:
                href = id_to_href[idref]
                file_path = Path(self.opf_dir) / href if self.opf_dir else href
                
                try:
                    content = self.zip_file.read(str(file_path))
                    text_content, title = self._extract_text_from_html(content)
                    
                    if text_content.strip():  # Only add non-empty chapters
                        chapter = EPUBChapter(
                            title=title or f"Chapter {order + 1}",
                            content=text_content,
                            order=order,
                            file_path=str(file_path)
                        )
                        self.chapters.append(chapter)
                except Exception as e:
                    logger.warning(f"Failed to extract chapter from {file_path}: {e}")
                    
    def _extract_text_from_html(self, html_content: bytes) -> tuple[str, Optional[str]]:
        """
        Extract text content from HTML/XHTML.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Tuple of (text_content, title)
        """
        try:
            # Parse HTML
            root = ET.fromstring(html_content)
            
            # Try to find title
            title = None
            for tag in ['h1', 'h2', 'h3', 'title']:
                title_elem = root.find(f'.//{{{self.NAMESPACES["xhtml"]}}}{tag}')
                if title_elem is None:
                    title_elem = root.find(f'.//{tag}')
                if title_elem is not None and title_elem.text:
                    title = title_elem.text.strip()
                    break
                    
            # Extract all text
            text_parts = []
            self._extract_text_recursive(root, text_parts)
            
            # Clean up text
            text = ' '.join(text_parts)
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = text.strip()
            
            return text, title
            
        except Exception as e:
            logger.warning(f"Failed to parse HTML content: {e}")
            # Fallback: try to extract text with regex
            text = re.sub(r'<[^>]+>', ' ', html_content.decode('utf-8', errors='ignore'))
            text = re.sub(r'\s+', ' ', text)
            return text.strip(), None
            
    def _extract_text_recursive(self, element, text_parts: List[str]):
        """Recursively extract text from XML elements."""
        if element.text:
            text_parts.append(element.text)
            
        for child in element:
            self._extract_text_recursive(child, text_parts)
            if child.tail:
                text_parts.append(child.tail)
                
    def _metadata_to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'title': self.metadata.title,
            'author': self.metadata.author,
            'publisher': self.metadata.publisher,
            'language': self.metadata.language,
            'isbn': self.metadata.isbn,
            'publication_date': self.metadata.publication_date,
            'description': self.metadata.description,
            'subjects': self.metadata.subjects,
            'contributors': self.metadata.contributors
        }
        
    def get_text_content(self) -> str:
        """
        Get all text content as a single string.
        
        Returns:
            Combined text from all chapters
        """
        with self:
            self._find_opf_file()
            self._extract_chapters()
            
        return '\n\n'.join(
            chapter.content
            for chapter in sorted(self.chapters, key=lambda x: x.order)
        )