"""
TTRPG PDF Content Extraction System.

A comprehensive system for extracting and classifying content from
tabletop role-playing game PDF rulebooks.
"""

from .models import (
    TTRPGGenre,
    ContentType,
    ExtractionConfidence,
    SourceAttribution,
    ExtendedCharacterRace,
    ExtendedCharacterClass,
    ExtendedNPCRole,
    ExtendedEquipment,
    ExtractedContent,
    ProcessingState
)

from .genre_classifier import GenreClassifier
from .content_extractor import ContentExtractor
from .pdf_analyzer import PDFAnalyzer

__version__ = "1.0.0"
__author__ = "MDMAI Development Team"

__all__ = [
    # Models
    "TTRPGGenre",
    "ContentType",
    "ExtractionConfidence",
    "SourceAttribution",
    "ExtendedCharacterRace",
    "ExtendedCharacterClass",
    "ExtendedNPCRole",
    "ExtendedEquipment",
    "ExtractedContent",
    "ProcessingState",
    
    # Core Classes
    "GenreClassifier",
    "ContentExtractor",
    "PDFAnalyzer"
]