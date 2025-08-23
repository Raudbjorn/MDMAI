#!/usr/bin/env python
"""Setup script for TTRPG Assistant MCP Server."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="ttrpg-assistant",
    version="0.1.0",
    author="MDMAI Project",
    description="MCP server for TTRPG assistance with rule lookup, campaign management, and AI-powered content generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Raudbjorn/MDMAI",
    packages=find_packages(where=".", include=["src*"]),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=[
        # Core MCP dependencies
        "mcp>=1.0.0",
        "fastmcp>=0.1.0",
        
        # Database
        "chromadb>=0.4.0",
        
        # PDF Processing
        "pypdf2>=3.0.0",
        "pdfplumber>=0.10.0",
        "python-magic>=0.4.27",
        
        # NLP and Embeddings
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "spacy>=3.5.0",
        "textblob>=0.17.1",
        
        # Search
        "rank-bm25>=0.2.2",
        
        # Web Server (for development/testing)
        "uvicorn>=0.23.0",
        "fastapi>=0.100.0",
        
        # Utilities
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "aiofiles>=23.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
        
        # Logging and Monitoring
        "structlog>=23.0.0",
        "rich>=13.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "pytest-timeout>=2.1.0",
            "pytest-xdist>=3.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
            "pylint>=2.17.0",
            "pre-commit>=3.3.0",
            "ipython>=8.14.0",
            "ipdb>=0.13.13",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "pytest-timeout>=2.1.0",
            "pytest-xdist>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ttrpg-server=src.main:main",
            "ttrpg-test=pytest:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Games/Entertainment :: Role-Playing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ttrpg mcp ai assistant rpg",
    project_urls={
        "Bug Reports": "https://github.com/Raudbjorn/MDMAI/issues",
        "Source": "https://github.com/Raudbjorn/MDMAI",
        "Documentation": "https://github.com/Raudbjorn/MDMAI/wiki",
    },
)