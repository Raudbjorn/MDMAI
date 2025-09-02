# Task 24.1: PDF Content Analysis Script

## Overview

This enhanced PDF content analysis system provides robust, production-ready extraction of TTRPG content from PDF files. It implements comprehensive error handling using the Result/Either pattern, supports multiple extraction methods including OCR fallback, and provides detailed progress tracking with resumable processing.

## Features

### Core Functionality
- **Batch Processing**: Process entire directories of PDFs with parallel execution
- **Genre Classification**: Intelligent classification based on content and filename patterns
- **Pattern Matching**: Extract races, classes, NPCs, and equipment using sophisticated regex patterns
- **OCR Fallback**: Automatic OCR for image-based PDFs using Tesseract
- **Progress Tracking**: Real-time progress with ETA calculation and resumable processing

### Enhanced Features
- **Result Pattern Integration**: Comprehensive error handling using `returns` library
- **Configurable Settings**: YAML-based configuration for all processing parameters
- **Detailed Metrics**: Track extraction quality, confidence scores, and performance
- **Smart Caching**: Intelligent caching with TTL and size management
- **Retry Logic**: Automatic retry with exponential backoff for failed extractions

## Installation

### Requirements
```bash
pip install pypdf pdfplumber pytesseract pdf2image pillow returns pyyaml
```

### System Dependencies
For OCR support:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler

# Windows
# Download and install from:
# - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# - Poppler: http://blog.alivate.com.au/poppler-windows/
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
# Process all PDFs in a directory
python pdf_analyzer_enhanced.py ./pdfs --output-dir ./output

# Process with OCR disabled for speed
python pdf_analyzer_enhanced.py ./pdfs --no-ocr

# Resume interrupted processing
python pdf_analyzer_enhanced.py ./pdfs --resume

# Process single PDF with verbose output
python pdf_analyzer_enhanced.py --single ./myfile.pdf --verbose
```

#### Advanced Options
```bash
# Custom OCR settings
python pdf_analyzer_enhanced.py ./pdfs --ocr-dpi 300 --ocr-max-pages 20

# Parallel processing with specific worker count
python pdf_analyzer_enhanced.py ./pdfs --max-workers 4

# Retry failed PDFs from previous run
python pdf_analyzer_enhanced.py ./pdfs --retry-failed --max-retries 5

# Custom file pattern
python pdf_analyzer_enhanced.py ./pdfs --pattern "D&D*.pdf"
```

### Python API

```python
from pathlib import Path
from frontend.src.content_expansion.pdf_analyzer_enhanced import (
    EnhancedPDFAnalyzer, ProcessingConfig
)

# Configure analyzer
config = ProcessingConfig(
    output_dir=Path("./output"),
    use_ocr=True,
    ocr_dpi=200,
    max_workers=4
)

# Initialize analyzer
analyzer = EnhancedPDFAnalyzer(config)

# Process single PDF
result = analyzer.analyze_single_pdf(Path("rulebook.pdf"))
if result.is_success():
    content = result.unwrap()
    print(f"Extracted: {content.get_summary()}")
else:
    error = result.failure()
    print(f"Failed: {error}")

# Process batch
batch_result = analyzer.process_batch("./pdfs", resume=True)
if batch_result.is_success():
    report = batch_result.unwrap()
    print(f"Processed: {report['processing_summary']}")
```

## Configuration

The system uses a YAML configuration file (`config.yaml`) for detailed settings:

```yaml
# Key configuration sections
processing:
  use_multiprocessing: true
  max_workers: null  # Auto-detect CPU cores
  retry_failed: true
  max_retries: 3

ocr:
  enabled: true
  dpi: 150
  max_pages: 50
  optimize_for: "speed"  # or "accuracy"

genre_classification:
  confidence_threshold: 0.5
  fallback_genre: "GENERIC"
```

## Output Format

### Extracted Content JSON
```json
{
  "pdf_name": "players_handbook.pdf",
  "genre": "FANTASY",
  "races": [
    {
      "name": "Elf",
      "description": "...",
      "traits": ["Darkvision", "Keen Senses"],
      "stat_modifiers": {"DEX": 2},
      "source": {
        "page_number": 23,
        "confidence": "HIGH"
      }
    }
  ],
  "classes": [...],
  "npcs": [...],
  "equipment": [...],
  "extraction_metadata": {
    "extraction_method": "pypdf",
    "genre_confidence": 0.92,
    "pages_processed": 350,
    "extraction_time": 45.3
  }
}
```

### Processing Report
```json
{
  "processing_summary": {
    "total_processed": 50,
    "successful": 45,
    "partial": 3,
    "failed": 2,
    "processing_time_seconds": 1234.5,
    "average_time_per_pdf": 24.7
  },
  "content_extracted": {
    "total_races": 127,
    "total_classes": 89,
    "total_npcs": 456,
    "total_equipment": 892,
    "average_confidence": 0.78
  },
  "quality_metrics": {
    "full_extraction_rate": 0.90,
    "partial_extraction_rate": 0.06,
    "failure_rate": 0.04,
    "ocr_usage_rate": 0.12
  }
}
```

## Error Handling

The system uses the Result/Either pattern for comprehensive error handling:

```python
from returns.result import Success, Failure

# All major operations return Result[T, AppError]
result = analyzer.extract_text_from_pdf(pdf_path)

if isinstance(result, Success):
    text, method, metrics = result.unwrap()
    # Process successful extraction
else:
    error = result.failure()
    print(f"Error: {error.kind.value} - {error.message}")
    # Handle error appropriately
```

### Error Types
- `VALIDATION`: Invalid input or configuration
- `PROCESSING`: PDF processing errors
- `DATABASE`: Cache or file system errors
- `TIMEOUT`: Operation timeout
- `SYSTEM`: Unexpected system errors

## Performance Optimization

### Multiprocessing
- Automatic CPU core detection
- Configurable worker pool size
- Process isolation for stability

### Caching Strategy
- MD5-based cache keys including file modification time
- Pickle serialization for fast I/O
- Automatic cache invalidation on configuration changes

### Memory Management
- Streaming text extraction for large PDFs
- Page-by-page processing to limit memory usage
- Configurable resource limits

## Monitoring and Progress

### Real-time Progress
```
[████████████████████░░░░░░░░░░░░░░░░░░░] 25/50 (50.0%) | ETA: 14:32:15 | Current: rulebook.pdf (45.2MB)
```

### Logging Levels
- `DEBUG`: Detailed extraction information
- `INFO`: Processing status and summaries
- `WARNING`: Recoverable errors and issues
- `ERROR`: Failed extractions and critical errors

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_pdf_analyzer_enhanced.py

# Run specific test class
python -m unittest test_pdf_analyzer_enhanced.TestEnhancedPDFAnalyzer

# Run with coverage
coverage run test_pdf_analyzer_enhanced.py
coverage report
```

## Troubleshooting

### Common Issues

1. **OCR Not Working**
   - Ensure Tesseract is installed: `tesseract --version`
   - Check Poppler installation: `pdftoppm -h`
   - Verify Python packages: `pip show pytesseract pdf2image`

2. **Memory Issues**
   - Reduce `max_workers` in configuration
   - Lower `ocr_dpi` setting
   - Enable `sequential` processing mode

3. **Slow Processing**
   - Disable OCR if not needed: `--no-ocr`
   - Increase worker count for parallel processing
   - Check cache is working (look for .pkl files in cache directory)

4. **Failed Extractions**
   - Check PDF isn't corrupted: `pdfinfo file.pdf`
   - Try different extraction method manually
   - Review logs for specific error messages

## Future Enhancements

- [ ] Integration with MCP server for real-time processing
- [ ] Machine learning-based content classification
- [ ] Support for additional document formats (EPUB, DOCX)
- [ ] Web interface for monitoring and management
- [ ] Database backend for extracted content
- [ ] API endpoints for content queries
- [ ] Automated quality validation
- [ ] Content deduplication across PDFs

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLI / API Interface                │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              EnhancedPDFAnalyzer                     │
│  ┌─────────────────────────────────────────────┐    │
│  │           ProcessingConfig                  │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────┐    │
│  │           Result Pattern Handler            │    │
│  └─────────────────────────────────────────────┘    │
└─────────┬───────────────────────┬───────────────────┘
          │                       │
┌─────────▼──────────┐  ┌────────▼──────────────┐
│  Text Extraction   │  │  Content Extraction   │
│  ┌──────────────┐  │  │  ┌────────────────┐  │
│  │    PyPDF     │  │  │  │Genre Classifier│  │
│  ├──────────────┤  │  │  ├────────────────┤  │
│  │  PDFPlumber  │  │  │  │Content Extractor│ │
│  ├──────────────┤  │  │  ├────────────────┤  │
│  │     OCR      │  │  │  │Pattern Matching│  │
│  └──────────────┘  │  │  └────────────────┘  │
└────────────────────┘  └───────────────────────┘
          │                       │
┌─────────▼───────────────────────▼───────────────────┐
│                 Cache & Storage Layer                │
│  ┌─────────────────────────────────────────────┐    │
│  │            Pickle Cache                     │    │
│  ├─────────────────────────────────────────────┤    │
│  │            JSON Output                      │    │
│  ├─────────────────────────────────────────────┤    │
│  │          Processing State                   │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

## License

This implementation is part of the MDMAI TTRPG Assistant project.

## Contributing

When contributing enhancements:
1. Maintain Result pattern for error handling
2. Add comprehensive tests for new features
3. Update configuration schema if needed
4. Document new command-line options
5. Ensure backwards compatibility with cache format