#!/usr/bin/env python3
"""
Command-line interface for PDF content extraction.

Usage:
    python extract_pdfs.py /home/svnbjrn/code/phase12/sample_pdfs --output-dir ./extracted_content
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.content_expansion.pdf_analyzer import main

if __name__ == "__main__":
    # Run the main function from pdf_analyzer
    main()