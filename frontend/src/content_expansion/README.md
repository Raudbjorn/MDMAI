# TTRPG PDF Content Extraction System

A comprehensive Python system for extracting and classifying content from tabletop role-playing game (TTRPG) PDF rulebooks. The system can process hundreds of PDFs in batch, identify game genres, and extract races, classes, NPCs, equipment, and other game elements with source attribution.

## Features

- **Multi-Genre Support**: Automatically classifies PDFs into 20+ TTRPG genres (Fantasy, Sci-Fi, Cyberpunk, Cosmic Horror, etc.)
- **Intelligent Content Extraction**: Uses pattern matching and NLP to identify:
  - Character races/species
  - Character classes/professions
  - NPCs and monsters
  - Equipment and items
  - Spells and abilities
  - Game mechanics and rules
- **Multiple Extraction Methods**: 
  - PyPDF for standard PDFs
  - PDFPlumber for complex layouts
  - OCR fallback for scanned/image-based PDFs
- **Batch Processing**: Process hundreds of PDFs with parallel processing support
- **Resumable Operations**: Automatically saves progress and can resume interrupted batch jobs
- **Source Attribution**: Tracks which PDF and page each piece of content came from
- **Caching System**: Avoids reprocessing already analyzed PDFs
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## Installation

### Prerequisites

1. Python 3.8 or higher
2. System dependencies for OCR (optional but recommended):

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# MacOS
brew install tesseract poppler

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pypdf>=3.0.0
- pdfplumber>=0.9.0
- pytesseract>=0.3.10 (optional, for OCR)
- pdf2image>=1.16.0 (optional, for OCR)
- Pillow>=10.0.0 (optional, for OCR)

## Usage

### Command Line Interface

#### Process all PDFs in a directory:
```bash
python extract_pdfs.py /path/to/your/pdfs --output-dir ./extracted_content
```

#### Process with specific options:
```bash
python extract_pdfs.py /path/to/pdfs \
    --output-dir ./output \
    --cache-dir ./cache \
    --max-workers 4 \
    --verbose
```

#### Process a single PDF:
```bash
python extract_pdfs.py /path/to/pdfs --single /path/to/specific.pdf
```

#### Resume interrupted processing:
```bash
python extract_pdfs.py /path/to/pdfs --output-dir ./output
# (automatically resumes if previous state exists)
```

#### Start fresh (ignore previous state):
```bash
python extract_pdfs.py /path/to/pdfs --no-resume --clear-cache
```

### Python API

```python
from content_expansion import PDFAnalyzer, TTRPGGenre

# Initialize analyzer
analyzer = PDFAnalyzer(
    output_dir="./extracted_content",
    cache_dir="./cache",
    use_multiprocessing=True,
    max_workers=4
)

# Process single PDF
from pathlib import Path
pdf_path = Path("/path/to/rulebook.pdf")
result = analyzer.analyze_single_pdf(pdf_path)

if result:
    print(f"Genre: {result.genre.name}")
    print(f"Races found: {len(result.races)}")
    print(f"Classes found: {len(result.classes)}")
    
    # Access extracted content
    for race in result.races:
        print(f"- {race.name}: {race.description}")

# Process batch
report = analyzer.process_batch("/path/to/pdf/directory", resume=True)
print(f"Processed {report['processing_summary']['total_processed']} PDFs")
```

### Testing

Run the test suite:
```bash
# Test all components
python test_extraction.py --all

# Test genre classifier only
python test_extraction.py --classifier

# Test single PDF extraction
python test_extraction.py --single /path/to/test.pdf

# Test batch processing (first 5 PDFs)
python test_extraction.py --batch /path/to/pdfs --max-files 5
```

## Data Models

### TTRPGGenre
Enumeration of supported TTRPG genres:
- FANTASY, SCI_FI, CYBERPUNK, COSMIC_HORROR
- POST_APOCALYPTIC, STEAMPUNK, URBAN_FANTASY
- SPACE_OPERA, SUPERHERO, WESTERN, NOIR
- And more...

### ExtendedCharacterRace
- name, genre, description
- traits, abilities, stat_modifiers
- size, speed, languages
- subraces, special_features
- source attribution

### ExtendedCharacterClass
- name, genre, description
- hit_dice, primary_ability
- saves, skills, equipment
- features by level
- subclasses, spell_casting
- source attribution

### ExtendedNPCRole
- name, genre, description
- role_type, challenge_rating
- abilities, stats, skills
- traits, actions, reactions
- equipment, tactics, loot
- source attribution

### ExtendedEquipment
- name, genre, equipment_type
- description, cost, weight
- properties, damage, armor_class
- requirements, special_abilities
- tech_level (for sci-fi/cyberpunk)
- source attribution

## Output Format

Extracted content is saved as JSON files with the following structure:

```json
{
  "pdf_path": "/path/to/file.pdf",
  "pdf_name": "rulebook.pdf",
  "genre": "FANTASY",
  "races": [
    {
      "name": "Elf",
      "genre": "FANTASY",
      "description": "Graceful and long-lived...",
      "traits": ["Darkvision", "Keen Senses"],
      "source": {
        "pdf_name": "rulebook.pdf",
        "page_number": 15,
        "confidence": "HIGH"
      }
    }
  ],
  "classes": [...],
  "npcs": [...],
  "equipment": [...],
  "extraction_metadata": {
    "extraction_method": "pypdf",
    "genre_confidence": 0.85,
    "pages_processed": 250,
    "extraction_timestamp": "2024-01-15T10:30:00"
  }
}
```

## Performance

- **Sequential Processing**: ~1-5 seconds per PDF (depending on size and complexity)
- **Parallel Processing**: Can process 4-8 PDFs simultaneously (CPU dependent)
- **OCR Processing**: ~30-60 seconds per PDF (much slower but handles scanned PDFs)
- **Memory Usage**: ~100-500 MB per worker process
- **Cache Hit**: <0.1 seconds (skips reprocessing)

## Troubleshooting

### Common Issues

1. **"No text extracted from PDF"**
   - PDF might be scanned/image-based
   - Install OCR dependencies (tesseract, poppler)
   - System will automatically fall back to OCR

2. **"Genre classification uncertain"**
   - PDF might be generic or multi-genre
   - Check confidence scores in extraction metadata
   - Manual genre override can be implemented if needed

3. **"Processing interrupted"**
   - Simply rerun the same command
   - System will automatically resume from last checkpoint
   - Use `--no-resume` to start fresh

4. **"Out of memory"**
   - Reduce `--max-workers` parameter
   - Process PDFs in smaller batches
   - Clear cache periodically with `--clear-cache`

### Logging

Detailed logs are saved to `pdf_extraction.log`. Enable verbose mode for debugging:
```bash
python extract_pdfs.py /path/to/pdfs --verbose
```

## Architecture

```
content_expansion/
├── models.py           # Data models and structures
├── genre_classifier.py # Genre detection logic
├── content_extractor.py # Pattern-based extraction
├── pdf_analyzer.py     # Main processing engine
├── extract_pdfs.py     # CLI interface
├── test_extraction.py  # Test suite
└── requirements.txt    # Dependencies
```

## Contributing

To extend the system:

1. **Add new genres**: Update `TTRPGGenre` enum and add keywords to `GenreClassifier`
2. **Add content types**: Update `ContentType` enum and add extraction patterns
3. **Improve extraction**: Add patterns to `ContentExtractor._initialize_patterns()`
4. **Add data models**: Create new dataclasses in `models.py`

## License

This system is part of the MDMAI project and follows the project's licensing terms.

## Contact

For issues, questions, or contributions, please refer to the main MDMAI project documentation.