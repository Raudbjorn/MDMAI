"""
Helper classes for PDF processing to support modular architecture.

This module contains specialized helper classes that handle specific
aspects of PDF processing, keeping the main analyzer class focused.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from returns.result import Failure, Result, Success

from src.core.result_pattern import AppError, ErrorKind, database_error, with_result
from frontend.src.content_expansion.models import ExtractedContent, ProcessingState

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of extracted content for efficient reprocessing."""
    
    def __init__(self, cache_dir: Path):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for storing cache files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @with_result(error_kind=ErrorKind.DATABASE)
    def load(self, cache_key: str) -> Optional[ExtractedContent]:
        """
        Load cached content.
        
        Args:
            cache_key: Unique key for the cached content
            
        Returns:
            Result containing ExtractedContent or None if not cached
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return ExtractedContent.from_dict(data)
            except Exception as e:
                logger.warning(f"Cache read error for {cache_key}: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
        
        return None
    
    @with_result(error_kind=ErrorKind.DATABASE)
    def save(self, cache_key: str, content: ExtractedContent) -> None:
        """
        Save content to cache.
        
        Args:
            cache_key: Unique key for the cached content
            content: ExtractedContent to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(content.to_dict(), f, indent=2)
        
        logger.debug(f"Cached content with key: {cache_key}")
    
    def clear(self) -> Result[int, AppError]:
        """
        Clear all cache files.
        
        Returns:
            Result containing count of cleared files or AppError
        """
        try:
            # Clear both old pickle and new JSON cache files
            cache_files = list(self.cache_dir.glob("*.pkl"))
            cache_files.extend(list(self.cache_dir.glob("*.json")))
            count = 0
            
            for cache_file in cache_files:
                if cache_file.name != "processing_state.json":  # Don't delete state file
                    cache_file.unlink()
                    count += 1
            
            logger.info(f"Cleared {count} cache files")
            return Success(count)
        
        except Exception as e:
            return Failure(database_error(f"Failed to clear cache: {e}"))
    
    def get_cache_key(self, pdf_path: Path, config_dict: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for a PDF and configuration.
        
        Args:
            pdf_path: Path to the PDF file
            config_dict: Configuration dictionary
            
        Returns:
            Unique cache key
        """
        import hashlib
        file_stats = f"{pdf_path.absolute()}_{pdf_path.stat().st_mtime}_{config_dict}"
        return hashlib.md5(file_stats.encode()).hexdigest()


class BatchProcessor:
    """Handles batch processing of multiple PDFs with progress tracking."""
    
    def __init__(self, config, stats: Dict[str, Any]):
        """
        Initialize the batch processor.
        
        Args:
            config: ProcessingConfig instance
            stats: Shared statistics dictionary
        """
        self.config = config
        self.stats = stats
        self.start_time: Optional[datetime] = None
        self.processing_state: Optional[ProcessingState] = None
    
    def process_sequential(
        self,
        pdf_files: List[Path],
        process_func,
        update_stats_func
    ) -> List[Tuple[Path, Optional[ExtractedContent]]]:
        """
        Process PDFs sequentially with progress tracking.
        
        Args:
            pdf_files: List of PDF paths to process
            process_func: Function to process a single PDF
            update_stats_func: Function to update statistics
            
        Returns:
            List of (pdf_path, extracted_content) tuples
        """
        results = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_path.name}")
            if self.processing_state:
                self.processing_state.current_pdf = pdf_path.name
            
            # Process with retries
            extracted = None
            for attempt in range(self.config.max_retries):
                try:
                    result = process_func(pdf_path)
                    if isinstance(result, Success):
                        extracted = result.unwrap()
                        break
                    elif attempt < self.config.max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    logger.error(f"Error processing {pdf_path.name}: {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)
            
            results.append((pdf_path, extracted))
            
            # Update state
            if self.processing_state:
                if extracted:
                    self.processing_state.successful_pdfs.append(pdf_path.name)
                    update_stats_func(extracted)
                else:
                    self.processing_state.failed_pdfs.append(pdf_path.name)
                
                self.processing_state.processed_pdfs += 1
            
            # Show progress with ETA
            self._show_progress_with_eta(i, len(pdf_files), pdf_path)
            
            # Save state periodically
            if i % 5 == 0:
                self._save_processing_state()
        
        return results
    
    def process_parallel(
        self,
        pdf_files: List[Path],
        worker_func,
        update_stats_func
    ) -> List[Tuple[Path, Optional[ExtractedContent]]]:
        """
        Process PDFs in parallel with progress tracking.
        
        Args:
            pdf_files: List of PDF paths to process
            worker_func: Worker function for parallel processing
            update_stats_func: Function to update statistics
            
        Returns:
            List of (pdf_path, extracted_content) tuples
        """
        results = []
        max_workers = self.config.max_workers or mp.cpu_count()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(worker_func, pdf_path, self.config): pdf_path
                for pdf_path in pdf_files
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                completed += 1
                
                try:
                    extracted = future.result(timeout=600)  # 10 minute timeout
                    results.append((pdf_path, extracted))
                    
                    if self.processing_state:
                        if extracted:
                            self.processing_state.successful_pdfs.append(pdf_path.name)
                            update_stats_func(extracted)
                        else:
                            self.processing_state.failed_pdfs.append(pdf_path.name)
                    
                    logger.info(f"Completed {pdf_path.name} ({completed}/{len(pdf_files)})")
                
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path.name}: {e}")
                    results.append((pdf_path, None))
                    if self.processing_state:
                        self.processing_state.failed_pdfs.append(pdf_path.name)
                
                if self.processing_state:
                    self.processing_state.processed_pdfs = completed
                
                # Show progress with ETA
                self._show_progress_with_eta(completed, len(pdf_files), pdf_path)
                
                # Save state periodically
                if completed % 5 == 0:
                    self._save_processing_state()
        
        return results
    
    def set_processing_state(self, state: ProcessingState) -> None:
        """Set the processing state for tracking."""
        self.processing_state = state
    
    def set_start_time(self, start_time: datetime) -> None:
        """Set the start time for ETA calculation."""
        self.start_time = start_time
    
    def _show_progress_with_eta(
        self,
        current: int,
        total: int,
        current_file: Path
    ) -> None:
        """Display progress bar with ETA calculation."""
        progress = current / total
        
        # Calculate ETA
        if self.start_time and current > 0:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            rate = current / elapsed
            remaining = (total - current) / rate if rate > 0 else 0
            eta = datetime.utcnow() + timedelta(seconds=remaining)
            eta_str = eta.strftime("%H:%M:%S")
        else:
            eta_str = "calculating..."
        
        # Build progress bar
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # File size for rate calculation
        file_size = current_file.stat().st_size / (1024 * 1024)  # MB
        
        status = (
            f'\r[{bar}] {current}/{total} ({progress*100:.1f}%) | '
            f'ETA: {eta_str} | Current: {current_file.name} ({file_size:.1f}MB)'
        )
        
        print(status, end='', flush=True)
        
        if current == total:
            print()  # New line when complete
    
    def _save_processing_state(self) -> None:
        """Save current processing state."""
        if self.processing_state:
            self.processing_state.last_checkpoint = datetime.utcnow()
            state_file = self.config.cache_dir / "processing_state.json"
            
            with open(state_file, 'w') as f:
                json.dump(self.processing_state.to_dict(), f, indent=2)
            
            logger.debug("Saved processing state checkpoint")


class ReportGenerator:
    """Generates comprehensive reports from processing results."""
    
    def __init__(self, config, stats: Dict[str, Any]):
        """
        Initialize the report generator.
        
        Args:
            config: ProcessingConfig instance
            stats: Statistics dictionary
        """
        self.config = config
        self.stats = stats
    
    def generate_report(
        self,
        results: List[Tuple[Path, Optional[ExtractedContent]]],
        processing_state: ProcessingState,
        start_time: datetime
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive processing report.
        
        Args:
            results: List of processing results
            processing_state: Current processing state
            start_time: Processing start time
            
        Returns:
            Detailed report dictionary
        """
        # Calculate final statistics
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        self.stats["failed"] = len(processing_state.failed_pdfs)
        
        # Categorize results
        full_success = []
        partial_success = []
        failures = []
        
        for pdf_path, content in results:
            if content:
                summary = content.get_summary()
                total_items = sum(summary.values())
                if total_items > 0:
                    full_success.append(pdf_path.name)
                else:
                    partial_success.append(pdf_path.name)
                    self.stats["partial"] += 1
            else:
                failures.append(pdf_path.name)
        
        # Generate detailed report
        report = {
            "processing_summary": {
                "total_processed": processing_state.processed_pdfs,
                "successful": len(full_success),
                "partial": len(partial_success),
                "failed": len(failures),
                "processing_time_seconds": processing_time,
                "average_time_per_pdf": (
                    processing_time / processing_state.processed_pdfs
                    if processing_state.processed_pdfs > 0 else 0
                ),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "content_extracted": {
                "total_races": self.stats["total_races"],
                "total_classes": self.stats["total_classes"],
                "total_npcs": self.stats["total_npcs"],
                "total_equipment": self.stats["total_equipment"],
                "total_pages": self.stats["total_pages"],
                "average_confidence": round(self.stats["average_confidence"], 3)
            },
            "extraction_methods": self.stats["extraction_methods"],
            "genres_found": list(self.stats["genres_found"]),
            "quality_metrics": {
                "full_extraction_rate": (
                    len(full_success) / processing_state.processed_pdfs
                    if processing_state.processed_pdfs > 0 else 0
                ),
                "partial_extraction_rate": (
                    len(partial_success) / processing_state.processed_pdfs
                    if processing_state.processed_pdfs > 0 else 0
                ),
                "failure_rate": (
                    len(failures) / processing_state.processed_pdfs
                    if processing_state.processed_pdfs > 0 else 0
                ),
                "ocr_usage_rate": (
                    self.stats["extraction_methods"].get("ocr", 0) /
                    processing_state.processed_pdfs
                    if processing_state.processed_pdfs > 0 else 0
                )
            },
            "file_lists": {
                "full_success": full_success[:20],  # First 20 for brevity
                "partial_success": partial_success[:20],
                "failures": failures
            },
            "configuration": self.config.to_dict()
        }
        
        # Save report
        report_file = self.config.output_dir / (
            f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing complete. Report saved to {report_file}")
        
        return report
    
    def update_stats(self, content: ExtractedContent) -> None:
        """
        Update statistics with extracted content.
        
        Args:
            content: ExtractedContent to add to statistics
        """
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
        
        # Track pages
        self.stats["total_pages"] += content.extraction_metadata.get("pages_processed", 0)
        
        # Track confidence
        confidence = content.extraction_metadata.get("genre_confidence", 0)
        if confidence > 0:
            current_avg = self.stats["average_confidence"]
            count = self.stats["successful"]
            self.stats["average_confidence"] = (
                (current_avg * (count - 1) + confidence) / count
            )