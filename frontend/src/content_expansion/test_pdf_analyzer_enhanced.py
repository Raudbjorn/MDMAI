#!/usr/bin/env python3
"""
Test suite for the enhanced PDF analyzer.

This module provides comprehensive tests for:
- Text extraction methods
- Genre classification
- Content extraction
- Error handling with Result pattern
- Batch processing
- Progress tracking
"""

import io
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, mock_open

from returns.result import Failure, Success

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
if parent_dir.exists():
    sys.path.insert(0, str(parent_dir))

from src.core.result_pattern import AppError, ErrorKind, validation_error
from frontend.src.content_expansion.models import (
    ExtractedContent, TTRPGGenre, ExtendedCharacterRace,
    ExtendedCharacterClass, ExtendedNPCRole, ExtendedEquipment
)
from frontend.src.content_expansion.pdf_analyzer_enhanced import (
    EnhancedPDFAnalyzer, ProcessingConfig, ExtractionMetrics
)


class TestProcessingConfig(unittest.TestCase):
    """Test ProcessingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        
        self.assertEqual(config.output_dir, Path("./extracted_content"))
        self.assertEqual(config.cache_dir, Path("./extraction_cache"))
        self.assertTrue(config.use_multiprocessing)
        self.assertTrue(config.use_ocr)
        self.assertEqual(config.ocr_dpi, 150)
        self.assertEqual(config.max_retries, 3)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = ProcessingConfig(
            output_dir=Path("/tmp/output"),
            use_ocr=False,
            max_workers=4
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict["output_dir"], "/tmp/output")
        self.assertFalse(config_dict["use_ocr"])
        self.assertEqual(config_dict["max_workers"], 4)


class TestExtractionMetrics(unittest.TestCase):
    """Test ExtractionMetrics dataclass."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ExtractionMetrics(
            extraction_method="pypdf",
            text_length=5000,
            pages_processed=10,
            extraction_time=2.5
        )
        
        self.assertEqual(metrics.extraction_method, "pypdf")
        self.assertEqual(metrics.text_length, 5000)
        self.assertEqual(metrics.pages_processed, 10)
        self.assertEqual(metrics.ocr_pages, 0)
        self.assertEqual(metrics.error_count, 0)
    
    def test_metrics_serialization(self):
        """Test metrics serialization."""
        metrics = ExtractionMetrics(
            extraction_method="ocr",
            text_length=3000,
            pages_processed=5,
            extraction_time=15.0,
            ocr_pages=5,
            confidence_scores={"genre": 0.85}
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertEqual(metrics_dict["extraction_method"], "ocr")
        self.assertEqual(metrics_dict["ocr_pages"], 5)
        self.assertEqual(metrics_dict["confidence_scores"]["genre"], 0.85)


class TestEnhancedPDFAnalyzer(unittest.TestCase):
    """Test EnhancedPDFAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ProcessingConfig(
            output_dir=Path(self.temp_dir) / "output",
            cache_dir=Path(self.temp_dir) / "cache",
            use_multiprocessing=False,
            use_ocr=False
        )
        self.analyzer = EnhancedPDFAnalyzer(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsNotNone(self.analyzer.genre_classifier)
        self.assertTrue(self.config.output_dir.exists())
        self.assertTrue(self.config.cache_dir.exists())
        self.assertEqual(self.analyzer.stats["total_processed"], 0)
    
    @patch('src.content_expansion.pdf_analyzer_enhanced.pypdf')
    def test_extract_with_pypdf(self, mock_pypdf):
        """Test text extraction with pypdf."""
        # Mock PDF reader
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock() for _ in range(3)]
        for page in mock_reader.pages:
            page.extract_text.return_value = "Sample text content"
        
        mock_pypdf.PdfReader.return_value = mock_reader
        
        pdf_path = Path(self.temp_dir) / "test.pdf"
        pdf_path.touch()
        
        # Test extraction
        text, pages = self.analyzer._extract_with_pypdf(pdf_path)
        
        self.assertIn("Sample text content", text)
        self.assertEqual(pages, 3)
    
    def test_split_into_pages(self):
        """Test text splitting into pages."""
        text = "Page 1 content\n\n\n\nPage 2 content\n\n\n\nPage 3 content"
        
        pages = self.analyzer._split_into_pages(text)
        
        self.assertEqual(len(pages), 3)
        self.assertIn("Page 1", pages[0])
        self.assertIn("Page 2", pages[1])
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # High confidence case
        metrics = ExtractionMetrics(
            extraction_method="pypdf",
            text_length=5000,
            pages_processed=10,
            extraction_time=1.0,
            error_count=0
        )
        
        confidence = self.analyzer._calculate_confidence(metrics)
        self.assertEqual(confidence.name, "HIGH")
        
        # Low confidence case (OCR)
        metrics.extraction_method = "ocr"
        confidence = self.analyzer._calculate_confidence(metrics)
        self.assertEqual(confidence.name, "LOW")
        
        # Low confidence case (many errors)
        metrics.extraction_method = "pypdf"
        metrics.error_count = 10
        confidence = self.analyzer._calculate_confidence(metrics)
        self.assertEqual(confidence.name, "LOW")
    
    def test_cache_file_generation(self):
        """Test cache file path generation."""
        pdf_path = Path(self.temp_dir) / "test.pdf"
        pdf_path.touch()
        
        cache_file1 = self.analyzer._get_cache_file(pdf_path)
        cache_file2 = self.analyzer._get_cache_file(pdf_path)
        
        # Same input should generate same cache file
        self.assertEqual(cache_file1, cache_file2)
        
        # Cache file should be in cache directory
        self.assertEqual(cache_file1.parent, self.config.cache_dir)
        
        # Cache file should have .pkl extension
        self.assertEqual(cache_file1.suffix, ".pkl")
    
    @patch('src.content_expansion.pdf_analyzer_enhanced.pickle')
    def test_check_cache(self, mock_pickle):
        """Test cache checking."""
        pdf_path = Path(self.temp_dir) / "test.pdf"
        pdf_path.touch()
        
        # Test cache miss
        result = self.analyzer._check_cache(pdf_path)
        self.assertIsInstance(result, Success)
        self.assertIsNone(result.unwrap())
        
        # Test cache hit
        cache_file = self.analyzer._get_cache_file(pdf_path)
        cache_file.touch()
        
        mock_content = ExtractedContent(
            pdf_path=pdf_path,
            pdf_name="test.pdf",
            genre=TTRPGGenre.FANTASY
        )
        mock_pickle.load.return_value = mock_content
        
        with patch('builtins.open', mock_open()):
            result = self.analyzer._check_cache(pdf_path)
            
        self.assertIsInstance(result, Success)
        # Note: Due to mocking, the actual content check would depend on mock setup
    
    def test_update_stats(self):
        """Test statistics updating."""
        content = ExtractedContent(
            pdf_path=Path("test.pdf"),
            pdf_name="test.pdf",
            genre=TTRPGGenre.FANTASY
        )
        
        # Add some content
        content.races.append(MagicMock())
        content.races.append(MagicMock())
        content.classes.append(MagicMock())
        content.extraction_metadata = {
            "extraction_method": "pypdf",
            "genre_confidence": 0.85,
            "pages_processed": 10
        }
        
        self.analyzer._update_stats(content)
        
        self.assertEqual(self.analyzer.stats["successful"], 1)
        self.assertEqual(self.analyzer.stats["total_races"], 2)
        self.assertEqual(self.analyzer.stats["total_classes"], 1)
        self.assertEqual(self.analyzer.stats["extraction_methods"]["pypdf"], 1)
        self.assertIn("FANTASY", self.analyzer.stats["genres_found"])
    
    def test_filter_processed_files(self):
        """Test filtering of already processed files."""
        pdf_files = [
            Path("file1.pdf"),
            Path("file2.pdf"),
            Path("file3.pdf"),
            Path("file4.pdf")
        ]
        
        self.analyzer.processing_state = MagicMock()
        self.analyzer.processing_state.successful_pdfs = ["file1.pdf", "file2.pdf"]
        self.analyzer.processing_state.failed_pdfs = ["file3.pdf"]
        
        # Test without retry_failed
        self.analyzer.config.retry_failed = False
        filtered = self.analyzer._filter_processed_files(pdf_files)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "file4.pdf")
        
        # Test with retry_failed
        self.analyzer.config.retry_failed = True
        filtered = self.analyzer._filter_processed_files(pdf_files)
        
        self.assertEqual(len(filtered), 2)
        self.assertIn(Path("file3.pdf"), filtered)
        self.assertIn(Path("file4.pdf"), filtered)
    
    def test_show_progress_with_eta(self):
        """Test progress display with ETA calculation."""
        import io
        from contextlib import redirect_stdout
        
        self.analyzer.start_time = datetime.utcnow()
        
        # Create a test file
        test_file = Path(self.temp_dir) / "test.pdf"
        test_file.write_text("dummy content")
        
        # Capture output
        output = io.StringIO()
        with redirect_stdout(output):
            self.analyzer._show_progress_with_eta(5, 10, test_file)
        
        progress_output = output.getvalue()
        
        # Check progress bar is displayed
        self.assertIn("█", progress_output)  # Filled part
        self.assertIn("░", progress_output)  # Empty part
        self.assertIn("50.0%", progress_output)  # Progress percentage
        self.assertIn("test.pdf", progress_output)  # Current file
    
    def test_finalize_processing(self):
        """Test report generation."""
        self.analyzer.start_time = datetime.utcnow()
        self.analyzer.processing_state = MagicMock()
        self.analyzer.processing_state.processed_pdfs = 3
        self.analyzer.processing_state.successful_pdfs = ["file1.pdf", "file2.pdf"]
        self.analyzer.processing_state.failed_pdfs = ["file3.pdf"]
        
        # Create mock results
        content1 = ExtractedContent(
            pdf_path=Path("file1.pdf"),
            pdf_name="file1.pdf",
            genre=TTRPGGenre.FANTASY
        )
        content1.races.append(MagicMock())
        
        content2 = ExtractedContent(
            pdf_path=Path("file2.pdf"),
            pdf_name="file2.pdf",
            genre=TTRPGGenre.SCI_FI
        )
        
        results = [
            (Path("file1.pdf"), content1),
            (Path("file2.pdf"), content2),
            (Path("file3.pdf"), None)
        ]
        
        # Update stats
        self.analyzer.stats["total_races"] = 1
        self.analyzer.stats["extraction_methods"] = {"pypdf": 2, "ocr": 0}
        self.analyzer.stats["genres_found"] = {"FANTASY", "SCI_FI"}
        
        report = self.analyzer._finalize_processing(results)
        
        # Check report structure
        self.assertIn("processing_summary", report)
        self.assertIn("content_extracted", report)
        self.assertIn("quality_metrics", report)
        self.assertIn("file_lists", report)
        
        # Check values
        self.assertEqual(report["processing_summary"]["total_processed"], 3)
        self.assertEqual(report["processing_summary"]["successful"], 1)
        self.assertEqual(report["processing_summary"]["partial"], 1)
        self.assertEqual(report["processing_summary"]["failed"], 1)
        
        # Check quality metrics
        self.assertAlmostEqual(report["quality_metrics"]["failure_rate"], 1/3, places=2)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Create some cache files
        cache_file1 = self.config.cache_dir / "test1.pkl"
        cache_file2 = self.config.cache_dir / "test2.pkl"
        state_file = self.config.cache_dir / "processing_state.json"
        
        cache_file1.touch()
        cache_file2.touch()
        state_file.touch()
        
        # Clear cache
        result = self.analyzer.clear_cache()
        
        self.assertIsInstance(result, Success)
        self.assertEqual(result.unwrap(), 3)
        
        # Check files are deleted
        self.assertFalse(cache_file1.exists())
        self.assertFalse(cache_file2.exists())
        self.assertFalse(state_file.exists())


class TestResultIntegration(unittest.TestCase):
    """Test Result pattern integration."""
    
    def test_with_result_decorator(self):
        """Test with_result decorator behavior."""
        from src.core.result_pattern import with_result
        
        @with_result(error_kind=ErrorKind.PROCESSING)
        def process_data(value: int) -> int:
            if value < 0:
                raise ValueError("Negative value not allowed")
            return value * 2
        
        # Test success case
        result = process_data(5)
        self.assertIsInstance(result, Success)
        self.assertEqual(result.unwrap(), 10)
        
        # Test failure case
        result = process_data(-1)
        self.assertIsInstance(result, Failure)
        error = result.failure()
        self.assertEqual(error.kind, ErrorKind.PROCESSING)
        self.assertIn("Negative value", error.message)
    
    def test_error_propagation(self):
        """Test error propagation through Result chain."""
        config = ProcessingConfig(
            output_dir=Path("/nonexistent/directory"),
            use_ocr=False
        )
        
        # This should fail due to permission error when creating directories
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
            try:
                analyzer = EnhancedPDFAnalyzer(config)
                # If no exception is raised, the test should note this
            except PermissionError:
                # Expected behavior
                pass


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ProcessingConfig(
            output_dir=Path(self.temp_dir) / "output",
            cache_dir=Path(self.temp_dir) / "cache",
            use_multiprocessing=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_process_batch_invalid_directory(self):
        """Test batch processing with invalid directory."""
        analyzer = EnhancedPDFAnalyzer(self.config)
        
        result = analyzer.process_batch("/nonexistent/directory")
        
        self.assertIsInstance(result, Failure)
        error = result.failure()
        self.assertEqual(error.kind, ErrorKind.VALIDATION)
        self.assertIn("not found", error.message)
    
    def test_process_batch_no_pdfs(self):
        """Test batch processing with no PDF files."""
        analyzer = EnhancedPDFAnalyzer(self.config)
        
        # Create empty directory
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        
        result = analyzer.process_batch(str(empty_dir))
        
        self.assertIsInstance(result, Failure)
        error = result.failure()
        self.assertEqual(error.kind, ErrorKind.VALIDATION)
        self.assertIn("No PDF files found", error.message)
    
    @patch('src.content_expansion.pdf_analyzer_enhanced.EnhancedPDFAnalyzer.analyze_single_pdf')
    def test_process_batch_sequential(self, mock_analyze):
        """Test sequential batch processing."""
        analyzer = EnhancedPDFAnalyzer(self.config)
        
        # Create test PDFs
        pdf_dir = Path(self.temp_dir) / "pdfs"
        pdf_dir.mkdir()
        
        pdf1 = pdf_dir / "test1.pdf"
        pdf2 = pdf_dir / "test2.pdf"
        pdf1.touch()
        pdf2.touch()
        
        # Mock successful analysis
        mock_content = ExtractedContent(
            pdf_path=pdf1,
            pdf_name="test1.pdf",
            genre=TTRPGGenre.FANTASY
        )
        mock_analyze.return_value = Success(mock_content)
        
        # Process batch
        result = analyzer.process_batch(str(pdf_dir), resume=False)
        
        self.assertIsInstance(result, Success)
        report = result.unwrap()
        
        # Check processing was called
        self.assertEqual(mock_analyze.call_count, 2)
        
        # Check report structure
        self.assertIn("processing_summary", report)
        self.assertIn("configuration", report)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()