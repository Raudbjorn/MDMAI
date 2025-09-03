#!/usr/bin/env python3
"""
Enhanced PDF analysis script for TTRPG content extraction with Result pattern.

This module provides robust PDF processing with:
- Result/Either pattern for error handling
- Comprehensive OCR configuration
- Enhanced progress tracking with ETA
- Detailed extraction reports
- Configurable processing settings
"""

import argparse
import asyncio
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from returns.result import Failure, Result, Success

# PDF processing libraries
try:
    import pypdf
except ImportError:
    import PyPDF2 as pypdf

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Add parent directories to path for imports
parent_dir = Path(__file__).parent.parent.parent
if parent_dir.exists():
    sys.path.insert(0, str(parent_dir))

# Local imports
from src.core.result_pattern import (
    AppError, ErrorKind, collect_results, database_error,
    validation_error, with_result
)
from frontend.src.content_expansion.models import (
    ExtractedContent, ProcessingState, SourceAttribution,
    ExtractionConfidence, TTRPGGenre
)
from frontend.src.content_expansion.genre_classifier import GenreClassifier
from frontend.src.content_expansion.content_extractor import ContentExtractor
from frontend.src.content_expansion.processing_helpers import (
    CacheManager, BatchProcessor, ReportGenerator
)


# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction_enhanced.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for PDF processing."""
    output_dir: Path = field(default_factory=lambda: Path("./extracted_content"))
    cache_dir: Path = field(default_factory=lambda: Path("./extraction_cache"))
    use_multiprocessing: bool = True
    max_workers: Optional[int] = None
    use_ocr: bool = True
    ocr_dpi: int = 150
    ocr_max_pages: int = 50
    ocr_timeout: int = 300  # seconds per page
    min_text_length: int = 100
    sample_size: int = 10000  # characters for genre classification
    chars_per_page: int = 3000  # characters per page for text splitting
    save_intermediate: bool = True
    verbose: bool = False
    retry_failed: bool = True
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir),
            "use_multiprocessing": self.use_multiprocessing,
            "max_workers": self.max_workers,
            "use_ocr": self.use_ocr,
            "ocr_dpi": self.ocr_dpi,
            "ocr_max_pages": self.ocr_max_pages,
            "ocr_timeout": self.ocr_timeout,
            "min_text_length": self.min_text_length,
            "sample_size": self.sample_size,
            "chars_per_page": self.chars_per_page,
            "save_intermediate": self.save_intermediate,
            "verbose": self.verbose,
            "retry_failed": self.retry_failed,
            "max_retries": self.max_retries
        }


@dataclass
class ExtractionMetrics:
    """Metrics for extraction quality and performance."""
    extraction_method: str
    text_length: int
    pages_processed: int
    extraction_time: float
    ocr_pages: int = 0
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EnhancedPDFAnalyzer:
    """Enhanced PDF analyzer with Result pattern and improved error handling."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the enhanced PDF analyzer.
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        
        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.genre_classifier = GenreClassifier()
        self.processing_state: Optional[ProcessingState] = None
        
        # Initialize helper classes
        self.cache_manager = CacheManager(self.config.cache_dir)
        
        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "partial": 0,
            "total_races": 0,
            "total_classes": 0,
            "total_npcs": 0,
            "total_equipment": 0,
            "genres_found": set(),
            "extraction_methods": {"pypdf": 0, "pdfplumber": 0, "ocr": 0},
            "processing_time": 0,
            "total_pages": 0,
            "ocr_pages": 0,
            "average_confidence": 0.0
        }
        
        # Progress tracking
        self.start_time: Optional[datetime] = None
        self.processed_sizes: List[Tuple[float, int]] = []  # (time, bytes)
        
        # Initialize batch processor and report generator with stats
        self.batch_processor = BatchProcessor(self.config, self.stats)
        self.report_generator = ReportGenerator(self.config, self.stats)
        
        logger.info(f"Initialized PDFAnalyzer with config: {self.config.to_dict()}")
    
    @with_result(error_kind=ErrorKind.PROCESSING)
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, str, ExtractionMetrics]:
        """
        Extract text from a PDF using multiple methods with metrics.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Result containing (text, method, metrics) or AppError
        """
        metrics = ExtractionMetrics(
            extraction_method="none",
            text_length=0,
            pages_processed=0,
            extraction_time=0
        )
        
        start_time = time.time()
        text = ""
        method = "none"
        
        # Try pypdf first (fastest)
        try:
            logger.debug(f"Trying pypdf for {pdf_path.name}")
            text, pages = self._extract_with_pypdf(pdf_path)
            
            if len(text.strip()) > self.config.min_text_length:
                method = "pypdf"
                metrics.pages_processed = pages
                logger.debug(f"Successfully extracted {len(text)} chars using pypdf")
        except Exception as e:
            logger.warning(f"pypdf failed for {pdf_path.name}: {e}")
        
        # Try pdfplumber if pypdf didn't work well
        if pdfplumber and len(text.strip()) < self.config.min_text_length:
            try:
                logger.debug(f"Trying pdfplumber for {pdf_path.name}")
                text, pages = self._extract_with_pdfplumber(pdf_path)
                
                if len(text.strip()) > self.config.min_text_length:
                    method = "pdfplumber"
                    metrics.pages_processed = pages
                    logger.debug(f"Successfully extracted {len(text)} chars using pdfplumber")
            except Exception as e:
                logger.warning(f"pdfplumber failed for {pdf_path.name}: {e}")
        
        # Try OCR as last resort
        if self.config.use_ocr and OCR_AVAILABLE and len(text.strip()) < self.config.min_text_length:
            try:
                logger.info(f"Attempting OCR for {pdf_path.name} (this may take a while)")
                ocr_result = self._ocr_pdf(pdf_path)
                
                if isinstance(ocr_result, Success):
                    ocr_text, ocr_pages = ocr_result.unwrap()
                    if len(ocr_text.strip()) > self.config.min_text_length:
                        text = ocr_text
                        method = "ocr"
                        metrics.ocr_pages = ocr_pages
                        metrics.pages_processed = ocr_pages
                        logger.info(f"Successfully extracted {len(text)} chars using OCR")
            except Exception as e:
                logger.warning(f"OCR failed for {pdf_path.name}: {e}")
        
        # Update metrics
        metrics.extraction_method = method
        metrics.text_length = len(text)
        metrics.extraction_time = time.time() - start_time
        
        if len(text.strip()) < self.config.min_text_length:
            logger.warning(f"Could not extract meaningful text from {pdf_path.name}")
        
        return text, method, metrics
    
    def _extract_with_pypdf(self, pdf_path: Path) -> Tuple[str, int]:
        """Extract text using pypdf."""
        text = ""
        pages = 0
        
        with open(pdf_path, 'rb') as file:
            if hasattr(pypdf, 'PdfReader'):
                reader = pypdf.PdfReader(file)
                num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    pages += 1
            else:
                reader = pypdf.PdfFileReader(file)
                num_pages = reader.numPages
                for page_num in range(num_pages):
                    page = reader.getPage(page_num)
                    page_text = page.extractText()
                    if page_text:
                        text += page_text + "\n"
                    pages += 1
        
        return text, pages
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Tuple[str, int]:
        """Extract text using pdfplumber."""
        text = ""
        pages = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                pages += 1
        
        return text, pages
    
    @with_result(error_kind=ErrorKind.PROCESSING)
    def _ocr_pdf(self, pdf_path: Path) -> Tuple[str, int]:
        """
        Perform OCR on a PDF file with timeout and page limits.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Result containing (text, pages_processed) or AppError
        """
        if not OCR_AVAILABLE:
            return "", 0
        
        text = ""
        pages_processed = 0
        
        try:
            # Convert PDF to images with configured DPI
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=self.config.ocr_dpi,
                first_page=1,
                last_page=self.config.ocr_max_pages,
                thread_count=2  # Limit thread usage
            )
            
            # OCR each page with timeout
            for i, image in enumerate(images):
                logger.debug(f"OCR processing page {i+1}/{len(images)} of {pdf_path.name}")
                
                try:
                    # Set timeout for OCR
                    page_text = pytesseract.image_to_string(
                        image,
                        timeout=self.config.ocr_timeout
                    )
                    text += page_text + "\n"
                    pages_processed += 1
                except RuntimeError as e:
                    if "timeout" in str(e).lower():
                        logger.warning(f"OCR timeout on page {i+1} of {pdf_path.name}")
                        break
                    raise
                
                # Stop if we have enough text
                if len(text) > 50000:  # 50k chars is usually enough
                    logger.debug(f"Stopping OCR early - sufficient text extracted")
                    break
        
        except Exception as e:
            logger.error(f"OCR error for {pdf_path.name}: {e}")
            raise
        
        return text, pages_processed
    
    @with_result(error_kind=ErrorKind.PROCESSING)
    def analyze_single_pdf(self, pdf_path: Path) -> ExtractedContent:
        """
        Analyze a single PDF and extract content with Result pattern.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Result containing ExtractedContent or AppError
        """
        logger.info(f"Analyzing {pdf_path.name}")
        start_time = time.time()
        
        # Check cache first
        cache_key = self.cache_manager.get_cache_key(pdf_path, self.config.to_dict())
        cache_result = self.cache_manager.load(cache_key)
        if isinstance(cache_result, Success):
            cached_content = cache_result.unwrap()
            if cached_content:
                logger.debug(f"Loaded from cache: {pdf_path.name}")
                return cached_content
        
        # Extract text with metrics
        extraction_result = self.extract_text_from_pdf(pdf_path)
        if isinstance(extraction_result, Failure):
            raise Exception(f"Text extraction failed: {extraction_result.failure()}")
        
        text, extraction_method, metrics = extraction_result.unwrap()
        
        if not text or len(text.strip()) < self.config.min_text_length:
            raise ValueError(f"Insufficient text extracted from {pdf_path.name}")
        
        # Classify genre
        genre_result = self._classify_genre(pdf_path, text)
        if isinstance(genre_result, Failure):
            raise Exception(f"Genre classification failed: {genre_result.failure()}")
        
        genre, confidence = genre_result.unwrap()
        metrics.confidence_scores["genre"] = confidence
        
        logger.info(f"Classified {pdf_path.name} as {genre.name} (confidence: {confidence:.2f})")
        
        # Extract content
        content_result = self._extract_content(
            pdf_path, text, genre, extraction_method, metrics
        )
        
        if isinstance(content_result, Failure):
            raise Exception(f"Content extraction failed: {content_result.failure()}")
        
        extracted = content_result.unwrap()
        
        # Add timing
        metrics.extraction_time = time.time() - start_time
        extracted.extraction_metadata.update(metrics.to_dict())
        
        # Cache the results
        if self.config.save_intermediate:
            cache_key = self.cache_manager.get_cache_key(pdf_path, self.config.to_dict())
            cache_save_result = self.cache_manager.save(cache_key, extracted)
            if isinstance(cache_save_result, Failure):
                logger.warning(f"Failed to cache results: {cache_save_result.failure()}")
        
        # Save to JSON
        save_result = self._save_extracted_content(extracted)
        if isinstance(save_result, Failure):
            logger.warning(f"Failed to save JSON: {save_result.failure()}")
        
        # Log summary
        summary = extracted.get_summary()
        logger.info(f"Extracted from {pdf_path.name}: {summary}")
        
        return extracted
    
    @with_result(error_kind=ErrorKind.PROCESSING)
    def _classify_genre(self, pdf_path: Path, text: str) -> Tuple[TTRPGGenre, float]:
        """Classify PDF genre with error handling."""
        genre, confidence = self.genre_classifier.classify(
            pdf_path,
            title=pdf_path.stem,
            text_content=text[:self.config.sample_size]
        )
        return genre, confidence
    
    @with_result(error_kind=ErrorKind.PROCESSING)
    def _extract_content(
        self,
        pdf_path: Path,
        text: str,
        genre: TTRPGGenre,
        extraction_method: str,
        metrics: ExtractionMetrics
    ) -> ExtractedContent:
        """Extract content from text with error handling."""
        # Initialize content extractor
        extractor = ContentExtractor(genre)
        
        # Create extracted content container
        extracted = ExtractedContent(
            pdf_path=pdf_path,
            pdf_name=pdf_path.name,
            genre=genre
        )
        
        # Split text into manageable chunks
        pages = self._split_into_pages(text)
        total_pages = len(pages)
        
        for page_num, page_text in enumerate(pages[:100], 1):  # Limit to first 100 pages
            if not page_text.strip():
                continue
            
            # Create source attribution
            source = SourceAttribution(
                pdf_path=pdf_path,
                pdf_name=pdf_path.name,
                page_number=page_num,
                confidence=self._calculate_confidence(metrics),
                extraction_method=extraction_method
            )
            
            # Extract different content types with error handling
            extraction_tasks = [
                ("races", lambda: extractor.extract_races(page_text, page_num, source)),
                ("classes", lambda: extractor.extract_classes(page_text, page_num, source)),
                ("npcs", lambda: extractor.extract_npcs(page_text, page_num, source)),
                ("equipment", lambda: extractor.extract_equipment(page_text, page_num, source))
            ]
            
            for content_type, extract_func in extraction_tasks:
                try:
                    items = extract_func()
                    getattr(extracted, content_type).extend(items)
                except Exception as e:
                    logger.debug(f"{content_type} extraction error on page {page_num}: {e}")
                    metrics.error_count += 1
            
            # Show progress for large PDFs
            if total_pages > 50 and page_num % 10 == 0:
                progress = (page_num / min(total_pages, 100)) * 100
                logger.debug(f"Processing {pdf_path.name}: {progress:.1f}% complete")
        
        # Add metadata
        extracted.extraction_metadata = {
            "extraction_method": extraction_method,
            "genre_confidence": metrics.confidence_scores.get("genre", 0),
            "text_length": len(text),
            "pages_processed": metrics.pages_processed,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "errors_encountered": metrics.error_count
        }
        
        return extracted
    
    def _split_into_pages(self, text: str, chars_per_page: Optional[int] = None) -> List[str]:
        """Split text into page-like chunks."""
        # Try to split by common page markers first
        page_markers = ['\f', '\n\n\n\n', 'Page ', '- Page ']
        
        for marker in page_markers:
            if marker in text:
                pages = text.split(marker)
                if len(pages) > 1:
                    return pages
        
        # Fallback to character-based splitting
        if chars_per_page is None:
            chars_per_page = self.config.chars_per_page
        
        pages = []
        for i in range(0, len(text), chars_per_page):
            pages.append(text[i:i + chars_per_page])
        
        return pages
    
    def _calculate_confidence(self, metrics: ExtractionMetrics) -> ExtractionConfidence:
        """Calculate extraction confidence based on metrics."""
        if metrics.extraction_method == "ocr":
            return ExtractionConfidence.LOW
        elif metrics.error_count > 5:
            return ExtractionConfidence.LOW
        elif metrics.extraction_method == "pypdf" and metrics.error_count == 0:
            return ExtractionConfidence.HIGH
        else:
            return ExtractionConfidence.MEDIUM
    
    
    @with_result(error_kind=ErrorKind.DATABASE)
    def _save_extracted_content(self, content: ExtractedContent) -> None:
        """Save extracted content to JSON file."""
        output_file = self.config.output_dir / f"{content.pdf_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(content.to_dict(), f, indent=2)
        
        logger.debug(f"Saved extracted content to {output_file}")
    
    
    def process_batch(
        self,
        pdf_directory: str,
        resume: bool = True,
        pattern: str = "*.pdf"
    ) -> Result[Dict[str, Any], AppError]:
        """
        Process multiple PDFs in batch with resumable support.
        
        Args:
            pdf_directory: Directory containing PDF files
            resume: Whether to resume from previous state
            pattern: Glob pattern for PDF files
        
        Returns:
            Result containing processing report or AppError
        """
        pdf_dir = Path(pdf_directory)
        
        if not pdf_dir.exists():
            return Failure(validation_error(f"Directory not found: {pdf_directory}"))
        
        pdf_files = list(pdf_dir.glob(pattern))
        
        if not pdf_files:
            return Failure(validation_error(f"No PDF files found matching pattern: {pattern}"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Initialize or load processing state
        state_result = self._initialize_processing_state(pdf_files, resume)
        if isinstance(state_result, Failure):
            return state_result
        
        self.processing_state = state_result.unwrap()
        
        # Filter already processed files if resuming
        if resume:
            pdf_files = self._filter_processed_files(pdf_files)
            logger.info(f"Resuming with {len(pdf_files)} remaining PDFs")
        
        # Process PDFs
        self.start_time = datetime.utcnow()
        self.batch_processor.set_start_time(self.start_time)
        self.batch_processor.set_processing_state(self.processing_state)
        
        if self.config.use_multiprocessing and len(pdf_files) > 1:
            results = self.batch_processor.process_parallel(
                pdf_files,
                self._analyze_pdf_worker,
                self.report_generator.update_stats
            )
        else:
            results = self.batch_processor.process_sequential(
                pdf_files,
                self.analyze_single_pdf,
                self.report_generator.update_stats
            )
        
        # Update statistics and generate report
        report = self.report_generator.generate_report(
            results,
            self.processing_state,
            self.start_time
        )
        
        # Save final state
        self._save_processing_state()
        
        return Success(report)
    
    def _initialize_processing_state(
        self,
        pdf_files: List[Path],
        resume: bool
    ) -> Result[ProcessingState, AppError]:
        """Initialize or load processing state."""
        state_file = self.config.cache_dir / "processing_state.json"
        
        if resume and state_file.exists():
            try:
                logger.info("Resuming from previous state")
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    return Success(ProcessingState.from_dict(state_data))
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        
        return Success(ProcessingState(total_pdfs=len(pdf_files)))
    
    def _filter_processed_files(self, pdf_files: List[Path]) -> List[Path]:
        """Filter out already processed files."""
        processed_names = set(
            self.processing_state.successful_pdfs +
            (self.processing_state.failed_pdfs if not self.config.retry_failed else [])
        )
        return [p for p in pdf_files if p.name not in processed_names]
    
    
    @staticmethod
    def _analyze_pdf_worker(
        pdf_path: Path,
        config: ProcessingConfig
    ) -> Optional[ExtractedContent]:
        """Worker function for parallel processing."""
        analyzer = EnhancedPDFAnalyzer(config)
        
        try:
            result = analyzer.analyze_single_pdf(pdf_path)
            if isinstance(result, Success):
                return result.unwrap()
            else:
                logger.error(f"Worker failed for {pdf_path.name}: {result.failure()}")
                return None
        except Exception as e:
            logger.error(f"Worker exception for {pdf_path.name}: {e}")
            return None
    
    
    def _save_processing_state(self) -> None:
        """Save current processing state."""
        if self.processing_state:
            self.processing_state.last_checkpoint = datetime.utcnow()
            state_file = self.config.cache_dir / "processing_state.json"
            
            with open(state_file, 'w') as f:
                json.dump(self.processing_state.to_dict(), f, indent=2)
            
            logger.debug("Saved processing state checkpoint")
    
    
    def clear_cache(self) -> Result[int, AppError]:
        """Clear the extraction cache."""
        return self.cache_manager.clear()


def main():
    """Enhanced command-line interface for PDF analyzer."""
    parser = argparse.ArgumentParser(
        description="Extract TTRPG content from PDFs with advanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process all PDFs in a directory:
    %(prog)s ./pdfs --output-dir ./output
  
  Process with OCR disabled for speed:
    %(prog)s ./pdfs --no-ocr
  
  Resume interrupted processing:
    %(prog)s ./pdfs --resume
  
  Process single PDF with verbose output:
    %(prog)s --single ./myfile.pdf --verbose
  
  Use custom OCR settings:
    %(prog)s ./pdfs --ocr-dpi 300 --ocr-max-pages 20
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "pdf_directory",
        nargs='?',
        help="Directory containing PDF files"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        default="./extracted_content",
        help="Output directory for extracted content (default: ./extracted_content)"
    )
    parser.add_argument(
        "--cache-dir",
        default="./extraction_cache",
        help="Cache directory for processing state (default: ./extraction_cache)"
    )
    
    # Processing options
    parser.add_argument(
        "--single",
        help="Process a single PDF file"
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob pattern for PDF files (default: *.pdf)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed PDFs"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per PDF (default: 3)"
    )
    
    # Performance options
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process PDFs sequentially instead of parallel"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers"
    )
    
    # OCR options
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR for image-based PDFs"
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=150,
        help="DPI for OCR image conversion (default: 150)"
    )
    parser.add_argument(
        "--ocr-max-pages",
        type=int,
        default=50,
        help="Maximum pages to OCR per PDF (default: 50)"
    )
    parser.add_argument(
        "--ocr-timeout",
        type=int,
        default=300,
        help="OCR timeout per page in seconds (default: 300)"
    )
    
    # Other options
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=100,
        help="Minimum text length to consider extraction successful (default: 100)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before processing"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.single and not args.pdf_directory:
        parser.error("Either pdf_directory or --single must be specified")
    
    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = ProcessingConfig(
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        use_multiprocessing=not args.sequential,
        max_workers=args.max_workers,
        use_ocr=not args.no_ocr,
        ocr_dpi=args.ocr_dpi,
        ocr_max_pages=args.ocr_max_pages,
        ocr_timeout=args.ocr_timeout,
        min_text_length=args.min_text_length,
        verbose=args.verbose,
        retry_failed=args.retry_failed,
        max_retries=args.max_retries
    )
    
    # Initialize analyzer
    analyzer = EnhancedPDFAnalyzer(config)
    
    # Clear cache if requested
    if args.clear_cache:
        clear_result = analyzer.clear_cache()
        if isinstance(clear_result, Failure):
            logger.error(f"Failed to clear cache: {clear_result.failure()}")
            sys.exit(1)
    
    # Process PDFs
    if args.single:
        # Process single PDF
        pdf_path = Path(args.single)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            sys.exit(1)
        
        result = analyzer.analyze_single_pdf(pdf_path)
        
        if isinstance(result, Success):
            content = result.unwrap()
            print(f"\nExtracted content summary:")
            print(json.dumps(content.get_summary(), indent=2))
            print(f"\nGenre: {content.genre.name}")
            print(f"Extraction method: {content.extraction_metadata.get('extraction_method')}")
            print(f"Pages processed: {content.extraction_metadata.get('pages_processed')}")
            print(f"Confidence: {content.extraction_metadata.get('genre_confidence', 0):.2f}")
        else:
            print(f"Failed to extract content: {result.failure()}")
            sys.exit(1)
    
    else:
        # Process batch
        if not Path(args.pdf_directory).exists():
            logger.error(f"Directory not found: {args.pdf_directory}")
            sys.exit(1)
        
        result = analyzer.process_batch(
            args.pdf_directory,
            resume=not args.no_resume,
            pattern=args.pattern
        )
        
        if isinstance(result, Success):
            report = result.unwrap()
            
            # Print summary
            print("\n" + "=" * 60)
            print("EXTRACTION COMPLETE")
            print("=" * 60)
            
            summary = report["processing_summary"]
            print(f"Total Processed: {summary['total_processed']}")
            print(f"Successful: {summary['successful']}")
            print(f"Partial: {summary['partial']}")
            print(f"Failed: {summary['failed']}")
            print(f"Processing Time: {summary['processing_time_seconds']:.1f} seconds")
            print(f"Average Time per PDF: {summary['average_time_per_pdf']:.1f} seconds")
            
            print("\nContent Extracted:")
            for key, value in report["content_extracted"].items():
                print(f"  {key}: {value}")
            
            print(f"\nGenres Found: {', '.join(report['genres_found'])}")
            
            print("\nExtraction Methods Used:")
            for method, count in report["extraction_methods"].items():
                print(f"  {method}: {count}")
            
            print("\nQuality Metrics:")
            for metric, value in report["quality_metrics"].items():
                print(f"  {metric}: {value:.2%}")
            
            if report["file_lists"]["failures"]:
                print(f"\nFailed PDFs ({len(report['file_lists']['failures'])}):")
                for pdf in report["file_lists"]["failures"][:10]:
                    print(f"  - {pdf}")
                if len(report["file_lists"]["failures"]) > 10:
                    print(f"  ... and {len(report['file_lists']['failures']) - 10} more")
        
        else:
            logger.error(f"Batch processing failed: {result.failure()}")
            sys.exit(1)


if __name__ == "__main__":
    main()