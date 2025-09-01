"""
Unified analyzer for processing multiple ebook formats.

This module provides a unified interface for extracting and analyzing content
from various ebook formats including EPUB, MOBI, AZW, and PDF.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union
from enum import Enum

try:
    from .epub_extractor import EPUBExtractor
    from .mobi_extractor import MOBIExtractor
    from .ebook_analyzer import EbookAnalyzer, ContentAnalysis
except ImportError:
    # Fallback for direct imports
    from epub_extractor import EPUBExtractor
    from mobi_extractor import MOBIExtractor
    from ebook_analyzer import EbookAnalyzer, ContentAnalysis

# Try to import PDF analyzer if available
try:
    from .pdf_analyzer import PDFAnalyzer
    PDF_SUPPORT = True
except ImportError:
    try:
        from pdf_analyzer import PDFAnalyzer
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False
    
logger = logging.getLogger(__name__)


class EbookFormat(Enum):
    """Supported ebook formats."""
    EPUB = "epub"
    MOBI = "mobi"
    AZW = "azw"
    AZW3 = "azw3"
    PDF = "pdf"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_extension(cls, file_path: Union[str, Path]) -> 'EbookFormat':
        """
        Determine format from file extension.
        
        Args:
            file_path: Path to the ebook file
            
        Returns:
            EbookFormat enum value
        """
        path = Path(file_path)
        extension = path.suffix.lower().lstrip('.')
        
        format_map = {
            'epub': cls.EPUB,
            'mobi': cls.MOBI,
            'azw': cls.AZW,
            'azw3': cls.AZW3,
            'pdf': cls.PDF
        }
        
        return format_map.get(extension, cls.UNKNOWN)
        

class UnifiedEbookAnalyzer:
    """Unified interface for analyzing various ebook formats."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the unified analyzer.
        
        Args:
            cache_dir: Optional directory for caching analysis results
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.analyzer = EbookAnalyzer(cache_dir=self.cache_dir)
        
    def process_ebook(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process an ebook file of any supported format.
        
        Args:
            file_path: Path to the ebook file
            
        Returns:
            Dictionary containing extracted content and analysis
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Determine format
        ebook_format = EbookFormat.from_extension(file_path)
        
        if ebook_format == EbookFormat.UNKNOWN:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        # Extract content based on format
        extracted_data = self._extract_content(file_path, ebook_format)
        
        # Analyze content
        content_for_analysis = self._prepare_content_for_analysis(extracted_data)
        analysis = self.analyzer.analyze_content(content_for_analysis)
        
        # Generate summary if not too large
        if analysis.statistics.word_count < 50000:  # Limit for performance
            summary = self.analyzer.generate_summary(
                content_for_analysis if isinstance(content_for_analysis, str)
                else self.analyzer._chapters_to_text(content_for_analysis),
                max_sentences=5
            )
            analysis.summary = summary
            
        # Combine results
        return {
            'file_path': str(file_path),
            'format': ebook_format.value,
            'metadata': extracted_data.get('metadata', {}),
            'content': extracted_data.get('content') or extracted_data.get('chapters'),
            'analysis': {
                'statistics': {
                    'word_count': analysis.statistics.word_count,
                    'sentence_count': analysis.statistics.sentence_count,
                    'paragraph_count': analysis.statistics.paragraph_count,
                    'average_word_length': analysis.statistics.average_word_length,
                    'average_sentence_length': analysis.statistics.average_sentence_length,
                    'unique_words': analysis.statistics.unique_words,
                    'lexical_diversity': analysis.statistics.lexical_diversity
                },
                'key_themes': analysis.key_themes,
                'frequent_words': analysis.frequent_words,
                'reading_level': analysis.reading_level,
                'estimated_reading_time': analysis.estimated_reading_time,
                'summary': analysis.summary,
                'chapters_analyzed': analysis.chapters_analyzed
            }
        }
        
    def _extract_content(self, file_path: Path, ebook_format: EbookFormat) -> Dict[str, Any]:
        """
        Extract content from ebook based on format.
        
        Args:
            file_path: Path to the ebook file
            ebook_format: Format of the ebook
            
        Returns:
            Extracted content dictionary
        """
        if ebook_format == EbookFormat.EPUB:
            return self._extract_epub(file_path)
        elif ebook_format in [EbookFormat.MOBI, EbookFormat.AZW, EbookFormat.AZW3]:
            return self._extract_mobi(file_path)
        elif ebook_format == EbookFormat.PDF:
            return self._extract_pdf(file_path)
        else:
            raise ValueError(f"Unsupported format: {ebook_format}")
            
    def _extract_epub(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from EPUB file."""
        try:
            with EPUBExtractor(file_path) as extractor:
                return extractor.extract()
        except Exception as e:
            logger.error(f"Failed to extract EPUB content: {e}")
            raise
            
    def _extract_mobi(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from MOBI/AZW file."""
        try:
            with MOBIExtractor(file_path) as extractor:
                return extractor.extract()
        except Exception as e:
            logger.error(f"Failed to extract MOBI content: {e}")
            raise
            
    def _extract_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from PDF file."""
        if not PDF_SUPPORT:
            raise NotImplementedError("PDF support not available. Install pdf_analyzer module.")
            
        try:
            analyzer = PDFAnalyzer()
            result = analyzer.analyze_pdf(str(file_path))
            
            # Convert to standard format
            return {
                'metadata': result.get('metadata', {}),
                'content': result.get('content', ''),
                'chapters': None  # PDFs don't have chapter structure
            }
        except Exception as e:
            logger.error(f"Failed to extract PDF content: {e}")
            raise
            
    def _prepare_content_for_analysis(self, extracted_data: Dict[str, Any]) -> Union[str, list]:
        """
        Prepare extracted content for analysis.
        
        Args:
            extracted_data: Extracted content dictionary
            
        Returns:
            Content suitable for analysis (text or chapter list)
        """
        # Check for chapters (EPUB format)
        if 'chapters' in extracted_data and extracted_data['chapters']:
            return extracted_data['chapters']
            
        # Check for direct content (MOBI/PDF format)
        if 'content' in extracted_data and extracted_data['content']:
            return extracted_data['content']
            
        # Fallback to empty content
        return ""
        
    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported ebook formats.
        
        Returns:
            List of supported format extensions
        """
        formats = ['epub', 'mobi', 'azw', 'azw3']
        if PDF_SUPPORT:
            formats.append('pdf')
        return formats
        
    def quick_extract_text(self, file_path: Union[str, Path]) -> str:
        """
        Quickly extract just the text content from an ebook.
        
        Args:
            file_path: Path to the ebook file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        ebook_format = EbookFormat.from_extension(file_path)
        
        if ebook_format == EbookFormat.EPUB:
            with EPUBExtractor(file_path) as extractor:
                return extractor.get_text_content()
                
        elif ebook_format in [EbookFormat.MOBI, EbookFormat.AZW, EbookFormat.AZW3]:
            with MOBIExtractor(file_path) as extractor:
                return extractor.get_text_content()
                
        elif ebook_format == EbookFormat.PDF and PDF_SUPPORT:
            analyzer = PDFAnalyzer()
            result = analyzer.analyze_pdf(str(file_path))
            return result.get('content', '')
            
        else:
            raise ValueError(f"Unsupported format for quick extraction: {ebook_format}")
            
    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract just the metadata from an ebook.
        
        Args:
            file_path: Path to the ebook file
            
        Returns:
            Metadata dictionary
        """
        file_path = Path(file_path)
        ebook_format = EbookFormat.from_extension(file_path)
        
        extracted_data = self._extract_content(file_path, ebook_format)
        return extracted_data.get('metadata', {})