"""
MOBI/AZW content extractor for ebook processing.

This module provides functionality to extract and parse content from MOBI and AZW files,
including text, metadata, and structural information.
"""

import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


@dataclass
class MOBIMetadata:
    """Container for MOBI metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    description: Optional[str] = None
    isbn: Optional[str] = None
    publish_date: Optional[str] = None
    language: Optional[str] = None
    contributor: Optional[str] = None
    rights: Optional[str] = None
    subject: Optional[str] = None
    
    
@dataclass
class MOBIHeader:
    """Container for MOBI header information."""
    identifier: bytes
    header_length: int
    mobi_type: int
    text_encoding: int
    unique_id: int
    file_version: int
    first_non_book_index: int
    full_name_offset: int
    full_name_length: int
    language_code: int
    input_language: int
    output_language: int
    min_version: int
    first_image_index: int
    first_huff_record: int
    huff_record_count: int
    huff_table_offset: int
    huff_table_length: int
    exth_flag: int
    drm_offset: int
    drm_count: int
    drm_size: int
    drm_flags: int
    

class MOBIExtractor:
    """Extract content from MOBI/AZW files."""
    
    # MOBI magic numbers
    MOBI_MAGIC = b'MOBI'
    EXTH_MAGIC = b'EXTH'
    
    # Text encodings
    ENCODING_UTF8 = 65001
    ENCODING_LATIN1 = 1252
    
    # EXTH record types
    EXTH_TYPES = {
        100: 'author',
        101: 'publisher',
        103: 'description',
        104: 'isbn',
        106: 'publish_date',
        108: 'contributor',
        109: 'rights',
        110: 'subject',
        503: 'title'
    }
    
    def __init__(self, file_path: Path):
        """
        Initialize MOBI extractor.
        
        Args:
            file_path: Path to the MOBI/AZW file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"MOBI file not found: {file_path}")
            
        self.file_handle: Optional[BinaryIO] = None
        self.metadata = MOBIMetadata()
        self.header: Optional[MOBIHeader] = None
        self.palm_header: Dict[str, Any] = {}
        self.text_records: List[bytes] = []
        
    def __enter__(self):
        """Context manager entry."""
        self.file_handle = open(self.file_path, 'rb')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.file_handle:
            self.file_handle.close()
            
    def extract(self) -> Dict[str, Any]:
        """
        Extract all content from the MOBI file.
        
        Returns:
            Dictionary containing metadata and text content
        """
        with self:
            self._read_palm_database_header()
            self._read_palm_record_info()
            self._read_mobi_header()
            
            if self.header and self.header.exth_flag & 0x40:
                self._read_exth_header()
                
            self._extract_text_records()
            text_content = self._decode_text_records()
            
        return {
            'metadata': self._metadata_to_dict(),
            'content': text_content,
            'encoding': self._get_encoding_name()
        }
        
    def _read_palm_database_header(self):
        """Read the Palm Database header."""
        self.file_handle.seek(0)
        
        # Read Palm Database header (78 bytes)
        self.palm_header['name'] = self.file_handle.read(32).decode('utf-8', errors='ignore').rstrip('\x00')
        self.palm_header['attributes'] = struct.unpack('>H', self.file_handle.read(2))[0]
        self.palm_header['version'] = struct.unpack('>H', self.file_handle.read(2))[0]
        self.palm_header['creation_date'] = struct.unpack('>I', self.file_handle.read(4))[0]
        self.palm_header['modification_date'] = struct.unpack('>I', self.file_handle.read(4))[0]
        self.palm_header['last_backup_date'] = struct.unpack('>I', self.file_handle.read(4))[0]
        self.palm_header['modification_number'] = struct.unpack('>I', self.file_handle.read(4))[0]
        self.palm_header['app_info_offset'] = struct.unpack('>I', self.file_handle.read(4))[0]
        self.palm_header['sort_info_offset'] = struct.unpack('>I', self.file_handle.read(4))[0]
        self.palm_header['type'] = self.file_handle.read(4)
        self.palm_header['creator'] = self.file_handle.read(4)
        self.palm_header['unique_seed'] = struct.unpack('>I', self.file_handle.read(4))[0]
        self.palm_header['next_record_list'] = struct.unpack('>I', self.file_handle.read(4))[0]
        self.palm_header['num_records'] = struct.unpack('>H', self.file_handle.read(2))[0]
        
    def _read_palm_record_info(self):
        """Read Palm record information."""
        self.palm_header['records'] = []
        
        for _ in range(self.palm_header['num_records']):
            record = {
                'offset': struct.unpack('>I', self.file_handle.read(4))[0],
                'attributes': self.file_handle.read(1)[0],
                'unique_id': struct.unpack('>I', b'\x00' + self.file_handle.read(3))[0]
            }
            self.palm_header['records'].append(record)
            
    def _read_mobi_header(self):
        """Read the MOBI header."""
        if not self.palm_header['records']:
            raise ValueError("No records found in Palm database")
            
        # Go to first record
        first_record_offset = self.palm_header['records'][0]['offset']
        self.file_handle.seek(first_record_offset)
        
        # Read PalmDOC header (16 bytes)
        palmdoc_header = {
            'compression': struct.unpack('>H', self.file_handle.read(2))[0],
            'unused': struct.unpack('>H', self.file_handle.read(2))[0],
            'text_length': struct.unpack('>I', self.file_handle.read(4))[0],
            'record_count': struct.unpack('>H', self.file_handle.read(2))[0],
            'record_size': struct.unpack('>H', self.file_handle.read(2))[0],
            'current_position': struct.unpack('>I', self.file_handle.read(4))[0]
        }
        
        # Check for MOBI header
        mobi_identifier = self.file_handle.read(4)
        if mobi_identifier != self.MOBI_MAGIC:
            logger.warning("MOBI header not found, file may be plain PalmDOC")
            return
            
        # Read MOBI header
        header_length = struct.unpack('>I', self.file_handle.read(4))[0]
        
        self.header = MOBIHeader(
            identifier=mobi_identifier,
            header_length=header_length,
            mobi_type=struct.unpack('>I', self.file_handle.read(4))[0],
            text_encoding=struct.unpack('>I', self.file_handle.read(4))[0],
            unique_id=struct.unpack('>I', self.file_handle.read(4))[0],
            file_version=struct.unpack('>I', self.file_handle.read(4))[0],
            first_non_book_index=struct.unpack('>I', self.file_handle.read(4))[0],
            full_name_offset=struct.unpack('>I', self.file_handle.read(4))[0],
            full_name_length=struct.unpack('>I', self.file_handle.read(4))[0],
            language_code=struct.unpack('>I', self.file_handle.read(4))[0],
            input_language=struct.unpack('>I', self.file_handle.read(4))[0],
            output_language=struct.unpack('>I', self.file_handle.read(4))[0],
            min_version=struct.unpack('>I', self.file_handle.read(4))[0],
            first_image_index=struct.unpack('>I', self.file_handle.read(4))[0],
            first_huff_record=struct.unpack('>I', self.file_handle.read(4))[0],
            huff_record_count=struct.unpack('>I', self.file_handle.read(4))[0],
            huff_table_offset=struct.unpack('>I', self.file_handle.read(4))[0],
            huff_table_length=struct.unpack('>I', self.file_handle.read(4))[0],
            exth_flag=struct.unpack('>I', self.file_handle.read(4))[0],
            drm_offset=struct.unpack('>I', self.file_handle.read(4))[0],
            drm_count=struct.unpack('>I', self.file_handle.read(4))[0],
            drm_size=struct.unpack('>I', self.file_handle.read(4))[0],
            drm_flags=struct.unpack('>I', self.file_handle.read(4))[0]
        )
        
        # Get full name if available
        if self.header.full_name_offset and self.header.full_name_length:
            current_pos = self.file_handle.tell()
            name_offset = first_record_offset + self.header.full_name_offset
            self.file_handle.seek(name_offset)
            full_name = self.file_handle.read(self.header.full_name_length)
            self.metadata.title = full_name.decode('utf-8', errors='ignore').rstrip('\x00')
            self.file_handle.seek(current_pos)
            
    def _read_exth_header(self):
        """Read the EXTH (extended) header."""
        # EXTH header should follow MOBI header
        first_record_offset = self.palm_header['records'][0]['offset']
        exth_offset = first_record_offset + 16 + self.header.header_length
        self.file_handle.seek(exth_offset)
        
        # Check for EXTH identifier
        exth_identifier = self.file_handle.read(4)
        if exth_identifier != self.EXTH_MAGIC:
            logger.warning("EXTH header not found where expected")
            return
            
        header_length = struct.unpack('>I', self.file_handle.read(4))[0]
        record_count = struct.unpack('>I', self.file_handle.read(4))[0]
        
        # Read EXTH records
        for _ in range(record_count):
            record_type = struct.unpack('>I', self.file_handle.read(4))[0]
            record_length = struct.unpack('>I', self.file_handle.read(4))[0]
            record_data = self.file_handle.read(record_length - 8)
            
            # Process known record types
            if record_type in self.EXTH_TYPES:
                field_name = self.EXTH_TYPES[record_type]
                value = record_data.decode('utf-8', errors='ignore').rstrip('\x00')
                setattr(self.metadata, field_name, value)
                
    def _extract_text_records(self):
        """Extract text records from the MOBI file."""
        if not self.header:
            # Try to extract as PalmDOC
            num_text_records = len(self.palm_header['records']) - 1
        else:
            num_text_records = min(
                self.header.first_non_book_index,
                len(self.palm_header['records'])
            ) - 1
            
        # Read text records
        for i in range(1, num_text_records + 1):
            record_start = self.palm_header['records'][i]['offset']
            
            if i < len(self.palm_header['records']) - 1:
                record_end = self.palm_header['records'][i + 1]['offset']
            else:
                # Last record - read to end of file
                self.file_handle.seek(0, 2)  # Seek to end
                record_end = self.file_handle.tell()
                
            self.file_handle.seek(record_start)
            record_data = self.file_handle.read(record_end - record_start)
            self.text_records.append(record_data)
            
    def _decode_text_records(self) -> str:
        """
        Decode text records into readable text.
        
        Returns:
            Decoded text content
        """
        # Combine all text records
        combined_data = b''.join(self.text_records)
        
        # Determine encoding
        if self.header:
            if self.header.text_encoding == self.ENCODING_UTF8:
                encoding = 'utf-8'
            elif self.header.text_encoding == self.ENCODING_LATIN1:
                encoding = 'cp1252'
            else:
                encoding = 'utf-8'  # Default
        else:
            encoding = 'utf-8'
            
        # Decode text
        text = combined_data.decode(encoding, errors='ignore')
        
        # Clean up text
        text = self._clean_text(text)
        
        return text
        
    def _clean_text(self, text: str) -> str:
        """
        Clean up extracted text.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove null characters
        text = text.replace('\x00', '')
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n?', '\n', text)
        
        return text.strip()
        
    def _get_encoding_name(self) -> str:
        """Get human-readable encoding name."""
        if not self.header:
            return 'unknown'
            
        if self.header.text_encoding == self.ENCODING_UTF8:
            return 'UTF-8'
        elif self.header.text_encoding == self.ENCODING_LATIN1:
            return 'Latin-1'
        else:
            return f'Unknown ({self.header.text_encoding})'
            
    def _metadata_to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'title': self.metadata.title or self.palm_header.get('name'),
            'author': self.metadata.author,
            'publisher': self.metadata.publisher,
            'description': self.metadata.description,
            'isbn': self.metadata.isbn,
            'publish_date': self.metadata.publish_date,
            'language': self.metadata.language,
            'contributor': self.metadata.contributor,
            'rights': self.metadata.rights,
            'subject': self.metadata.subject
        }
        
    def get_text_content(self) -> str:
        """
        Get all text content as a single string.
        
        Returns:
            Extracted text content
        """
        with self:
            self._read_palm_database_header()
            self._read_palm_record_info()
            self._read_mobi_header()
            self._extract_text_records()
            return self._decode_text_records()