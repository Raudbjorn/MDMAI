#!/usr/bin/env python3
"""
Command-line interface for PDF content extraction.

Usage:
    python extract_pdfs.py <pdf_directory> --output-dir ./extracted_content
    
Example:
    python extract_pdfs.py ./sample_pdfs --output-dir ./extracted_content
    python extract_pdfs.py $PDF_DIR --output-dir ./output
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.parent
if parent_dir.exists():
    sys.path.insert(0, str(parent_dir))

from src.content_expansion.pdf_analyzer import main

if __name__ == "__main__":
    # Run the main function from pdf_analyzer
    main()