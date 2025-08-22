"""Source validation system for quality assurance."""

import os
import re
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .models import Source, SourceQuality, QualityLevel, SourceMetadata, SourceType

logger = logging.getLogger(__name__)


class SourceValidator:
    """Validate and assess quality of source documents."""
    
    # Minimum requirements
    MIN_PAGE_COUNT = 1
    MIN_TEXT_LENGTH = 100
    MIN_CHUNK_LENGTH = 50
    MAX_CHUNK_LENGTH = 5000
    MIN_EXTRACTION_RATE = 0.5  # At least 50% of pages should have content
    
    # Quality thresholds
    EXCELLENT_EXTRACTION_RATE = 0.95
    GOOD_EXTRACTION_RATE = 0.80
    FAIR_EXTRACTION_RATE = 0.60
    
    # File size limits (in MB)
    MAX_FILE_SIZE = 500
    WARNING_FILE_SIZE = 100
    
    # Supported formats
    SUPPORTED_FORMATS = ['.pdf', '.txt', '.md', '.epub']
    
    # Required metadata fields by source type
    REQUIRED_METADATA = {
        SourceType.RULEBOOK: ['title', 'system', 'page_count'],
        SourceType.FLAVOR: ['title', 'system'],
        SourceType.SUPPLEMENT: ['title', 'system', 'page_count'],
        SourceType.ADVENTURE: ['title', 'system', 'page_count'],
        SourceType.SETTING: ['title', 'system'],
        SourceType.BESTIARY: ['title', 'system'],
        SourceType.SPELL_COMPENDIUM: ['title', 'system'],
        SourceType.ITEM_CATALOG: ['title', 'system'],
    }
    
    def __init__(self):
        """Initialize the source validator."""
        self.validation_cache = {}
    
    def validate_source_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a source file before processing.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {}
        }
        
        try:
            path = Path(file_path)
            
            # Check file existence
            if not path.exists():
                results["valid"] = False
                results["errors"].append(f"File not found: {file_path}")
                return results
            
            # Check file format
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                results["valid"] = False
                results["errors"].append(
                    f"Unsupported format: {path.suffix}. "
                    f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
                )
            
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            results["file_info"]["size_mb"] = round(file_size_mb, 2)
            
            if file_size_mb > self.MAX_FILE_SIZE:
                results["valid"] = False
                results["errors"].append(
                    f"File too large: {file_size_mb:.2f}MB "
                    f"(max: {self.MAX_FILE_SIZE}MB)"
                )
            elif file_size_mb > self.WARNING_FILE_SIZE:
                results["warnings"].append(
                    f"Large file: {file_size_mb:.2f}MB. "
                    "Processing may take longer."
                )
            
            # Calculate file hash
            results["file_info"]["hash"] = self._calculate_file_hash(file_path)
            results["file_info"]["format"] = path.suffix.lower()
            results["file_info"]["name"] = path.name
            
            # Check file readability
            if not os.access(file_path, os.R_OK):
                results["valid"] = False
                results["errors"].append("File is not readable")
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation error: {str(e)}")
        
        return results
    
    def validate_metadata(
        self,
        metadata: SourceMetadata,
        source_type: Optional[SourceType] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate source metadata completeness and correctness.
        
        Args:
            metadata: Source metadata to validate
            source_type: Expected source type
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check required fields
        if not metadata.title or metadata.title == "Unknown":
            errors.append("Title is required")
        
        if not metadata.system or metadata.system == "Unknown":
            errors.append("System is required")
        
        # Check source type specific requirements
        if source_type or metadata.source_type:
            check_type = source_type or metadata.source_type
            required_fields = self.REQUIRED_METADATA.get(check_type, [])
            
            for field in required_fields:
                value = getattr(metadata, field, None)
                if not value or (field == 'page_count' and value == 0):
                    errors.append(f"'{field}' is required for {check_type.value}")
        
        # Validate specific fields
        if metadata.page_count < 0:
            errors.append("Page count cannot be negative")
        
        if metadata.publication_date:
            if not self._validate_date(metadata.publication_date):
                warnings.append(f"Invalid publication date format: {metadata.publication_date}")
        
        if metadata.isbn:
            if not self._validate_isbn(metadata.isbn):
                warnings.append(f"Invalid ISBN format: {metadata.isbn}")
        
        # Check for completeness
        completeness = self._calculate_metadata_completeness(metadata)
        if completeness < 0.3:
            warnings.append(f"Low metadata completeness: {completeness:.0%}")
        
        is_valid = len(errors) == 0
        
        return is_valid, errors, warnings
    
    def validate_content(
        self,
        source: Source,
        content_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> SourceQuality:
        """
        Validate and assess content quality.
        
        Args:
            source: Source object to validate
            content_chunks: Optional content chunks to validate
            
        Returns:
            SourceQuality object with assessment results
        """
        quality = SourceQuality()
        
        try:
            # Use provided chunks or source's chunks
            chunks = content_chunks or source.content_chunks
            
            if not chunks:
                quality.issues.append("No content chunks found")
                quality.text_quality = 0.0
                quality.structure_quality = 0.0
                quality.calculate_overall_score()
                return quality
            
            # Basic metrics
            quality.total_chunks = len(chunks)
            quality.total_pages = source.metadata.page_count or 0
            
            # Validate each chunk
            valid_chunks = 0
            total_length = 0
            empty_chunks = 0
            malformed_chunks = 0
            
            for chunk in chunks:
                validation = self._validate_chunk(chunk)
                
                if validation['valid']:
                    valid_chunks += 1
                    total_length += validation['length']
                else:
                    if validation['empty']:
                        empty_chunks += 1
                    if validation['malformed']:
                        malformed_chunks += 1
            
            quality.valid_chunks = valid_chunks
            
            # Calculate average chunk length
            if valid_chunks > 0:
                quality.avg_chunk_length = total_length / valid_chunks
            
            # Calculate quality scores
            
            # Text quality based on valid chunks
            if quality.total_chunks > 0:
                quality.text_quality = valid_chunks / quality.total_chunks
            
            # Structure quality based on extraction rate
            if quality.total_pages > 0:
                quality.extracted_pages = min(quality.total_chunks, quality.total_pages)
                extraction_rate = quality.extracted_pages / quality.total_pages
                
                if extraction_rate >= self.EXCELLENT_EXTRACTION_RATE:
                    quality.structure_quality = 1.0
                elif extraction_rate >= self.GOOD_EXTRACTION_RATE:
                    quality.structure_quality = 0.8
                elif extraction_rate >= self.FAIR_EXTRACTION_RATE:
                    quality.structure_quality = 0.6
                elif extraction_rate >= self.MIN_EXTRACTION_RATE:
                    quality.structure_quality = 0.4
                else:
                    quality.structure_quality = 0.2
                    quality.issues.append(
                        f"Low extraction rate: {extraction_rate:.0%}"
                    )
            
            # Metadata completeness
            quality.metadata_completeness = self._calculate_metadata_completeness(
                source.metadata
            )
            
            # Content coverage (simplified - could be enhanced)
            quality.content_coverage = self._assess_content_coverage(source)
            
            # Report issues
            if empty_chunks > quality.total_chunks * 0.1:
                quality.issues.append(
                    f"High number of empty chunks: {empty_chunks}/{quality.total_chunks}"
                )
            
            if malformed_chunks > 0:
                quality.warnings.append(
                    f"Found {malformed_chunks} malformed chunks"
                )
            
            if quality.avg_chunk_length < self.MIN_CHUNK_LENGTH:
                quality.issues.append(
                    f"Average chunk length too short: {quality.avg_chunk_length:.0f}"
                )
            elif quality.avg_chunk_length > self.MAX_CHUNK_LENGTH:
                quality.warnings.append(
                    f"Average chunk length very long: {quality.avg_chunk_length:.0f}"
                )
            
            # Calculate overall score
            quality.calculate_overall_score()
            
        except Exception as e:
            logger.error(f"Content validation error: {str(e)}")
            quality.issues.append(f"Validation error: {str(e)}")
            quality.calculate_overall_score()
        
        return quality
    
    def validate_duplicate(
        self,
        file_hash: str,
        existing_sources: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if source is a duplicate.
        
        Args:
            file_hash: Hash of the file to check
            existing_sources: List of existing source metadata
            
        Returns:
            Duplicate source info if found, None otherwise
        """
        for source in existing_sources:
            if source.get('file_hash') == file_hash:
                return {
                    "is_duplicate": True,
                    "source_id": source.get('id'),
                    "title": source.get('title'),
                    "system": source.get('system')
                }
        
        return None
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _validate_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single content chunk."""
        result = {
            "valid": True,
            "empty": False,
            "malformed": False,
            "length": 0
        }
        
        # Check structure
        if not isinstance(chunk, dict):
            result["valid"] = False
            result["malformed"] = True
            return result
        
        # Check content
        content = chunk.get('content', '')
        if not content or not isinstance(content, str):
            result["valid"] = False
            result["empty"] = True
            return result
        
        # Clean and check length
        cleaned_content = content.strip()
        if len(cleaned_content) < self.MIN_CHUNK_LENGTH:
            result["valid"] = False
            result["empty"] = True
            return result
        
        result["length"] = len(cleaned_content)
        
        # Check for obvious extraction errors
        if self._is_garbled_text(cleaned_content):
            result["valid"] = False
            result["malformed"] = True
        
        return result
    
    def _is_garbled_text(self, text: str) -> bool:
        """Check if text appears to be garbled or corrupted."""
    @lru_cache(maxsize=128)
    def _is_garbled_text(self, text: str) -> bool:
        """Check if text appears to be garbled or corrupted."""
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
        if special_char_ratio > 0.5:
            return True
        
        # Check for excessive numbers
        number_ratio = len(re.findall(r'\d', text)) / len(text)
        if number_ratio > 0.5:
            return True
        
        # Check for repeating characters
        if re.search(r'(.)\1{10,}', text):
            return True
        
        return False
    
    def _validate_date(self, date_str: str) -> bool:
        """Validate date format."""
        # Accept various date formats
        date_patterns = [
            r'^\d{4}$',  # Year only
            r'^\d{4}-\d{2}$',  # Year-Month
            r'^\d{4}-\d{2}-\d{2}$',  # ISO date
            r'^\d{2}/\d{2}/\d{4}$',  # US date
            r'^\d{2}-\d{2}-\d{4}$',  # Alt date
        ]
        
        return any(re.match(pattern, date_str) for pattern in date_patterns)
    
    def _validate_isbn(self, isbn: str) -> bool:
        """Validate ISBN format."""
        # Remove hyphens and spaces
        isbn_clean = isbn.replace('-', '').replace(' ', '')
        
        # Check ISBN-10 or ISBN-13
        if len(isbn_clean) == 10:
            return isbn_clean[:-1].isdigit()
        elif len(isbn_clean) == 13:
            return isbn_clean.isdigit()
        
        return False
    
    def _calculate_metadata_completeness(self, metadata: SourceMetadata) -> float:
        """Calculate how complete the metadata is."""
        fields = [
            'title', 'system', 'author', 'publisher',
            'publication_date', 'edition', 'version',
            'isbn', 'description', 'page_count'
        ]
        
        filled = 0
        for field in fields:
            value = getattr(metadata, field, None)
            if value and value != "Unknown" and value != 0:
                filled += 1
        
        return filled / len(fields)
    
    def _assess_content_coverage(self, source: Source) -> float:
        """Assess how well content covers expected categories."""
        if not source.metadata.categories:
            return 0.5  # Default if no categories specified
        
        # This is a simplified assessment
        # Could be enhanced with actual content analysis
        expected_categories = len(source.metadata.categories)
        if expected_categories == 0:
            return 0.5
        
        # Check if chunks mention category keywords
        covered = 0
        for category in source.metadata.categories:
            category_name = category.value if hasattr(category, 'value') else str(category)
            
            # Check if any chunk mentions this category
            for chunk in source.content_chunks[:10]:  # Sample first 10 chunks
                if category_name.lower() in chunk.get('content', '').lower():
                    covered += 1
                    break
        
        return min(1.0, (covered / expected_categories) + 0.3)  # Baseline + coverage