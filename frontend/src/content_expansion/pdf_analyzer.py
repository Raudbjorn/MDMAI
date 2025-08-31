"""
Main PDF analysis script for TTRPG content extraction.

This module provides the main entry point for processing TTRPG PDFs,
with support for batch processing, error handling, and resumable operations.
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import hashlib

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

# Local imports
from .models import (
    ExtractedContent, ProcessingState, SourceAttribution,
    ExtractionConfidence, TTRPGGenre
)
from .genre_classifier import GenreClassifier
from .content_extractor import ContentExtractor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PDFAnalyzer:
    """Main class for analyzing and extracting content from TTRPG PDFs."""
    
    def __init__(self, output_dir: str = "./extracted_content",
                 cache_dir: str = "./extraction_cache",
                 use_multiprocessing: bool = True,
                 max_workers: Optional[int] = None):
        """
        Initialize the PDF analyzer.
        
        Args:
            output_dir: Directory to store extracted content
            cache_dir: Directory to store processing cache
            use_multiprocessing: Whether to use multiprocessing for batch operations
            max_workers: Maximum number of worker processes
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers or mp.cpu_count()
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.genre_classifier = GenreClassifier()
        self.processing_state = None
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_races": 0,
            "total_classes": 0,
            "total_npcs": 0,
            "total_equipment": 0,
            "genres_found": set(),
            "extraction_methods": {"pypdf": 0, "pdfplumber": 0, "ocr": 0},
            "processing_time": 0
        }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, str]:
        """
        Extract text from a PDF using multiple methods.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Tuple of (extracted_text, extraction_method)
        """
        text = ""
        method = "none"
        
        # Try pypdf first (fastest)
        try:
            logger.debug(f"Trying pypdf for {pdf_path.name}")
            with open(pdf_path, 'rb') as file:
                if hasattr(pypdf, 'PdfReader'):
                    reader = pypdf.PdfReader(file)
                else:
                    reader = pypdf.PdfFileReader(file)
                
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if len(text.strip()) > 100:
                method = "pypdf"
                logger.debug(f"Successfully extracted {len(text)} chars using pypdf")
                return text, method
        except Exception as e:
            logger.warning(f"pypdf failed for {pdf_path.name}: {e}")
        
        # Try pdfplumber (better for complex layouts)
        if pdfplumber:
            try:
                logger.debug(f"Trying pdfplumber for {pdf_path.name}")
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                if len(text.strip()) > 100:
                    method = "pdfplumber"
                    logger.debug(f"Successfully extracted {len(text)} chars using pdfplumber")
                    return text, method
            except Exception as e:
                logger.warning(f"pdfplumber failed for {pdf_path.name}: {e}")
        
        # Try OCR as last resort (slowest but works for scanned PDFs)
        if OCR_AVAILABLE and len(text.strip()) < 100:
            try:
                logger.info(f"Attempting OCR for {pdf_path.name} (this may take a while)")
                text = self._ocr_pdf(pdf_path)
                if len(text.strip()) > 100:
                    method = "ocr"
                    logger.info(f"Successfully extracted {len(text)} chars using OCR")
                    return text, method
            except Exception as e:
                logger.warning(f"OCR failed for {pdf_path.name}: {e}")
        
        logger.warning(f"Could not extract meaningful text from {pdf_path.name}")
        return text, method
    
    def _ocr_pdf(self, pdf_path: Path, max_pages: int = 50) -> str:
        """
        Perform OCR on a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to OCR
        
        Returns:
            Extracted text
        """
        if not OCR_AVAILABLE:
            return ""
        
        text = ""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=150,  # Lower DPI for faster processing
                first_page=1,
                last_page=min(max_pages, 50)
            )
            
            # OCR each page
            for i, image in enumerate(images):
                logger.debug(f"OCR processing page {i+1} of {pdf_path.name}")
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"
                
                # Stop if we have enough text
                if len(text) > 10000:
                    break
        
        except Exception as e:
            logger.error(f"OCR error for {pdf_path.name}: {e}")
        
        return text
    
    def analyze_single_pdf(self, pdf_path: Path) -> Optional[ExtractedContent]:
        """
        Analyze a single PDF and extract content.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            ExtractedContent object or None if extraction failed
        """
        logger.info(f"Analyzing {pdf_path.name}")
        
        try:
            # Check cache first
            cache_file = self._get_cache_file(pdf_path)
            if cache_file.exists():
                logger.debug(f"Loading from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Extract text
            text, extraction_method = self.extract_text_from_pdf(pdf_path)
            
            if not text or len(text.strip()) < 100:
                logger.warning(f"Insufficient text extracted from {pdf_path.name}")
                return None
            
            # Classify genre
            genre, confidence = self.genre_classifier.classify(
                pdf_path, 
                title=pdf_path.stem,
                text_content=text[:10000]  # Use first 10k chars for classification
            )
            
            logger.info(f"Classified {pdf_path.name} as {genre.name} (confidence: {confidence:.2f})")
            
            # Initialize content extractor for the detected genre
            extractor = ContentExtractor(genre)
            
            # Create extracted content container
            extracted = ExtractedContent(
                pdf_path=pdf_path,
                pdf_name=pdf_path.name,
                genre=genre
            )
            
            # Extract content page by page (simulate for now)
            pages = text.split('\n\n')  # Simple page splitting
            
            for page_num, page_text in enumerate(pages[:100], 1):  # Limit to first 100 pages
                if not page_text.strip():
                    continue
                
                # Create source attribution
                source = SourceAttribution(
                    pdf_path=pdf_path,
                    pdf_name=pdf_path.name,
                    page_number=page_num,
                    confidence=ExtractionConfidence.MEDIUM if confidence > 0.5 else ExtractionConfidence.LOW,
                    extraction_method=extraction_method
                )
                
                # Extract different content types
                try:
                    races = extractor.extract_races(page_text, page_num, source)
                    extracted.races.extend(races)
                except Exception as e:
                    logger.debug(f"Race extraction error on page {page_num}: {e}")
                
                try:
                    classes = extractor.extract_classes(page_text, page_num, source)
                    extracted.classes.extend(classes)
                except Exception as e:
                    logger.debug(f"Class extraction error on page {page_num}: {e}")
                
                try:
                    npcs = extractor.extract_npcs(page_text, page_num, source)
                    extracted.npcs.extend(npcs)
                except Exception as e:
                    logger.debug(f"NPC extraction error on page {page_num}: {e}")
                
                try:
                    equipment = extractor.extract_equipment(page_text, page_num, source)
                    extracted.equipment.extend(equipment)
                except Exception as e:
                    logger.debug(f"Equipment extraction error on page {page_num}: {e}")
            
            # Add metadata
            extracted.extraction_metadata = {
                "extraction_method": extraction_method,
                "genre_confidence": confidence,
                "text_length": len(text),
                "pages_processed": len(pages),
                "extraction_timestamp": datetime.utcnow().isoformat()
            }
            
            # Log summary
            summary = extracted.get_summary()
            logger.info(f"Extracted from {pdf_path.name}: {summary}")
            
            # Cache the results
            with open(cache_file, 'wb') as f:
                pickle.dump(extracted, f)
            
            # Save to JSON
            self._save_extracted_content(extracted)
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error analyzing {pdf_path.name}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def process_batch(self, pdf_directory: str, resume: bool = True) -> Dict[str, Any]:
        """
        Process multiple PDFs in batch with resumable support.
        
        Args:
            pdf_directory: Directory containing PDF files
            resume: Whether to resume from previous state
        
        Returns:
            Processing statistics
        """
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Initialize or load processing state
        state_file = self.cache_dir / "processing_state.json"
        
        if resume and state_file.exists():
            logger.info("Resuming from previous state")
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                self.processing_state = ProcessingState.from_dict(state_data)
        else:
            self.processing_state = ProcessingState(total_pdfs=len(pdf_files))
        
        # Filter already processed files
        if resume:
            processed_names = set(self.processing_state.successful_pdfs + 
                                self.processing_state.failed_pdfs)
            pdf_files = [p for p in pdf_files if p.name not in processed_names]
            logger.info(f"Resuming with {len(pdf_files)} remaining PDFs")
        
        # Process PDFs
        start_time = datetime.utcnow()
        
        if self.use_multiprocessing and len(pdf_files) > 1:
            results = self._process_batch_parallel(pdf_files)
        else:
            results = self._process_batch_sequential(pdf_files)
        
        # Update statistics
        for pdf_path, extracted in results:
            if extracted:
                self.processing_state.successful_pdfs.append(pdf_path.name)
                self._update_stats(extracted)
            else:
                self.processing_state.failed_pdfs.append(pdf_path.name)
            
            self.processing_state.processed_pdfs += 1
            
            # Save state periodically
            if self.processing_state.processed_pdfs % 10 == 0:
                self._save_processing_state()
        
        # Final save
        self._save_processing_state()
        
        # Calculate processing time
        end_time = datetime.utcnow()
        self.stats["processing_time"] = (end_time - start_time).total_seconds()
        
        # Generate report
        report = self._generate_report()
        
        # Save report
        report_file = self.output_dir / f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing complete. Report saved to {report_file}")
        
        return report
    
    def _process_batch_sequential(self, pdf_files: List[Path]) -> List[Tuple[Path, Optional[ExtractedContent]]]:
        """Process PDFs sequentially."""
        results = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_path.name}")
            self.processing_state.current_pdf = pdf_path.name
            
            extracted = self.analyze_single_pdf(pdf_path)
            results.append((pdf_path, extracted))
            
            # Show progress
            self._show_progress(i, len(pdf_files))
        
        return results
    
    def _process_batch_parallel(self, pdf_files: List[Path]) -> List[Tuple[Path, Optional[ExtractedContent]]]:
        """Process PDFs in parallel using multiprocessing."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(self._analyze_pdf_worker, pdf_path): pdf_path
                for pdf_path in pdf_files
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                completed += 1
                
                try:
                    extracted = future.result(timeout=300)  # 5 minute timeout per PDF
                    results.append((pdf_path, extracted))
                    logger.info(f"Completed {pdf_path.name} ({completed}/{len(pdf_files)})")
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path.name}: {e}")
                    results.append((pdf_path, None))
                
                # Show progress
                self._show_progress(completed, len(pdf_files))
        
        return results
    
    @staticmethod
    def _analyze_pdf_worker(pdf_path: Path) -> Optional[ExtractedContent]:
        """Worker function for parallel processing."""
        analyzer = PDFAnalyzer(use_multiprocessing=False)
        return analyzer.analyze_single_pdf(pdf_path)
    
    def _get_cache_file(self, pdf_path: Path) -> Path:
        """Get cache file path for a PDF."""
        # Create hash of file path and modification time
        file_stats = f"{pdf_path.absolute()}_{pdf_path.stat().st_mtime}"
        file_hash = hashlib.md5(file_stats.encode()).hexdigest()
        return self.cache_dir / f"{file_hash}.pkl"
    
    def _save_extracted_content(self, content: ExtractedContent):
        """Save extracted content to JSON file."""
        output_file = self.output_dir / f"{content.pdf_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(content.to_dict(), f, indent=2)
        
        logger.debug(f"Saved extracted content to {output_file}")
    
    def _save_processing_state(self):
        """Save current processing state."""
        if self.processing_state:
            self.processing_state.last_checkpoint = datetime.utcnow()
            state_file = self.cache_dir / "processing_state.json"
            
            with open(state_file, 'w') as f:
                json.dump(self.processing_state.to_dict(), f, indent=2)
    
    def _update_stats(self, content: ExtractedContent):
        """Update statistics with extracted content."""
        self.stats["successful"] += 1
        self.stats["total_races"] += len(content.races)
        self.stats["total_classes"] += len(content.classes)
        self.stats["total_npcs"] += len(content.npcs)
        self.stats["total_equipment"] += len(content.equipment)
        self.stats["genres_found"].add(content.genre.name)
        
        # Track extraction method
        method = content.extraction_metadata.get("extraction_method", "unknown")
        if method in self.stats["extraction_methods"]:
            self.stats["extraction_methods"][method] += 1
    
    def _show_progress(self, current: int, total: int):
        """Display progress bar."""
        progress = current / total
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f'\rProgress: [{bar}] {current}/{total} ({progress*100:.1f}%)', end='', flush=True)
        
        if current == total:
            print()  # New line when complete
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate processing report."""
        return {
            "processing_summary": {
                "total_processed": self.processing_state.processed_pdfs,
                "successful": len(self.processing_state.successful_pdfs),
                "failed": len(self.processing_state.failed_pdfs),
                "processing_time_seconds": self.stats["processing_time"],
                "start_time": self.processing_state.start_time.isoformat(),
                "end_time": datetime.utcnow().isoformat()
            },
            "content_extracted": {
                "total_races": self.stats["total_races"],
                "total_classes": self.stats["total_classes"],
                "total_npcs": self.stats["total_npcs"],
                "total_equipment": self.stats["total_equipment"]
            },
            "genres_found": list(self.stats["genres_found"]),
            "extraction_methods": self.stats["extraction_methods"],
            "failed_pdfs": self.processing_state.failed_pdfs,
            "successful_pdfs": self.processing_state.successful_pdfs[:10]  # First 10 for brevity
        }
    
    def clear_cache(self):
        """Clear the extraction cache."""
        cache_files = self.cache_dir.glob("*.pkl")
        count = 0
        
        for cache_file in cache_files:
            cache_file.unlink()
            count += 1
        
        # Also clear state file
        state_file = self.cache_dir / "processing_state.json"
        if state_file.exists():
            state_file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cache files")


def main():
    """Main entry point for the PDF analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract TTRPG content from PDFs")
    parser.add_argument("pdf_directory", help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="./extracted_content",
                       help="Output directory for extracted content")
    parser.add_argument("--cache-dir", default="./extraction_cache",
                       help="Cache directory for processing state")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh instead of resuming")
    parser.add_argument("--sequential", action="store_true",
                       help="Process PDFs sequentially instead of parallel")
    parser.add_argument("--max-workers", type=int,
                       help="Maximum number of parallel workers")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear cache before processing")
    parser.add_argument("--single", help="Process a single PDF file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize analyzer
    analyzer = PDFAnalyzer(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_multiprocessing=not args.sequential,
        max_workers=args.max_workers
    )
    
    # Clear cache if requested
    if args.clear_cache:
        analyzer.clear_cache()
    
    # Process PDFs
    if args.single:
        # Process single PDF
        pdf_path = Path(args.single)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            sys.exit(1)
        
        result = analyzer.analyze_single_pdf(pdf_path)
        if result:
            print(f"\nExtracted content summary:")
            print(json.dumps(result.get_summary(), indent=2))
        else:
            print("Failed to extract content from PDF")
    else:
        # Process batch
        if not Path(args.pdf_directory).exists():
            logger.error(f"Directory not found: {args.pdf_directory}")
            sys.exit(1)
        
        report = analyzer.process_batch(
            args.pdf_directory,
            resume=not args.no_resume
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)
        print(f"Total Processed: {report['processing_summary']['total_processed']}")
        print(f"Successful: {report['processing_summary']['successful']}")
        print(f"Failed: {report['processing_summary']['failed']}")
        print(f"Processing Time: {report['processing_summary']['processing_time_seconds']:.1f} seconds")
        print("\nContent Extracted:")
        for key, value in report['content_extracted'].items():
            print(f"  {key}: {value}")
        print(f"\nGenres Found: {', '.join(report['genres_found'])}")


if __name__ == "__main__":
    main()