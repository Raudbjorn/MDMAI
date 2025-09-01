"""
Ebook content analyzer for extracting insights and metadata.

This module provides functionality to analyze ebook content, extract key themes,
generate summaries, and identify important metadata.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class TextStatistics:
    """Statistics about the text content."""
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    average_word_length: float = 0.0
    average_sentence_length: float = 0.0
    unique_words: int = 0
    lexical_diversity: float = 0.0
    
    
@dataclass
class ContentAnalysis:
    """Analysis results for ebook content."""
    statistics: TextStatistics
    key_themes: List[str] = field(default_factory=list)
    frequent_words: List[tuple[str, int]] = field(default_factory=list)
    reading_level: str = ""
    estimated_reading_time: int = 0  # in minutes
    language_detected: Optional[str] = None
    summary: Optional[str] = None
    chapters_analyzed: int = 0
    

class EbookAnalyzer:
    """Analyze ebook content for insights and metadata."""
    
    # Common stop words to filter out
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'shall', 'it',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we',
        'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'whose',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'there', 'here'
    }
    
    # Average reading speed in words per minute
    AVERAGE_READING_SPEED = 250
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the ebook analyzer.
        
        Args:
            cache_dir: Optional directory for caching analysis results
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def analyze_content(self, content: Union[str, List[Dict[str, Any]]]) -> ContentAnalysis:
        """
        Analyze ebook content.
        
        Args:
            content: Either raw text string or list of chapter dictionaries
            
        Returns:
            ContentAnalysis object with results
        """
        # Convert to text if needed
        if isinstance(content, list):
            text = self._chapters_to_text(content)
            chapters_count = len(content)
        else:
            text = content
            chapters_count = 1
            
        # Check cache if available
        if self.cache_dir:
            cached_result = self._get_cached_analysis(text)
            if cached_result:
                return cached_result
                
        # Perform analysis
        statistics = self._calculate_statistics(text)
        themes = self._extract_themes(text)
        frequent_words = self._get_frequent_words(text)
        reading_level = self._estimate_reading_level(statistics)
        reading_time = self._estimate_reading_time(statistics.word_count)
        
        analysis = ContentAnalysis(
            statistics=statistics,
            key_themes=themes,
            frequent_words=frequent_words,
            reading_level=reading_level,
            estimated_reading_time=reading_time,
            chapters_analyzed=chapters_count
        )
        
        # Cache result if enabled
        if self.cache_dir:
            self._cache_analysis(text, analysis)
            
        return analysis
        
    def _chapters_to_text(self, chapters: List[Dict[str, Any]]) -> str:
        """Convert chapter list to single text string."""
        text_parts = []
        for chapter in chapters:
            if isinstance(chapter, dict) and 'content' in chapter:
                text_parts.append(chapter['content'])
        return '\n\n'.join(text_parts)
        
    def _calculate_statistics(self, text: str) -> TextStatistics:
        """Calculate text statistics."""
        # Clean text
        text = text.strip()
        if not text:
            return TextStatistics()
            
        # Word statistics
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        unique_words = len(set(words))
        
        # Sentence statistics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # Paragraph statistics
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Calculate averages
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        return TextStatistics(
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            average_word_length=avg_word_length,
            average_sentence_length=avg_sentence_length,
            unique_words=unique_words,
            lexical_diversity=lexical_diversity
        )
        
    def _extract_themes(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract key themes from text.
        
        Args:
            text: Text to analyze
            top_n: Number of top themes to return
            
        Returns:
            List of key themes
        """
        # Extract noun phrases (simplified approach)
        # In a production system, you'd use NLP libraries like spaCy or NLTK
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out single common words
        themes = []
        for word in words:
            if len(word.split()) > 1 or (
                len(word) > 4 and word.lower() not in self.STOP_WORDS
            ):
                themes.append(word)
                
        # Count frequencies
        theme_counts = Counter(themes)
        
        # Return top themes
        return [theme for theme, _ in theme_counts.most_common(top_n)]
        
    def _get_frequent_words(self, text: str, top_n: int = 20) -> List[tuple[str, int]]:
        """
        Get most frequent meaningful words.
        
        Args:
            text: Text to analyze
            top_n: Number of top words to return
            
        Returns:
            List of (word, frequency) tuples
        """
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stop words and short words
        meaningful_words = [
            word for word in words
            if word not in self.STOP_WORDS and len(word) > 3
        ]
        
        # Count frequencies
        word_counts = Counter(meaningful_words)
        
        return word_counts.most_common(top_n)
        
    def _estimate_reading_level(self, statistics: TextStatistics) -> str:
        """
        Estimate reading level based on text statistics.
        
        Args:
            statistics: Text statistics
            
        Returns:
            Reading level estimate
        """
        # Simplified Flesch-Kincaid Grade Level approximation
        if statistics.sentence_count == 0 or statistics.word_count == 0:
            return "Unknown"
            
        # Approximate syllables (simplified: count vowel groups)
        # In production, use a proper syllable counter
        avg_syllables_per_word = 1.5  # Rough estimate
        
        grade = (
            0.39 * statistics.average_sentence_length +
            11.8 * avg_syllables_per_word - 15.59
        )
        
        if grade < 6:
            return "Elementary"
        elif grade < 9:
            return "Middle School"
        elif grade < 13:
            return "High School"
        elif grade < 16:
            return "College"
        else:
            return "Graduate"
            
    def _estimate_reading_time(self, word_count: int) -> int:
        """
        Estimate reading time in minutes.
        
        Args:
            word_count: Total word count
            
        Returns:
            Estimated reading time in minutes
        """
        return max(1, round(word_count / self.AVERAGE_READING_SPEED))
        
    def generate_summary(self, content: str, max_sentences: int = 5) -> str:
        """
        Generate a simple extractive summary.
        
        Args:
            content: Text to summarize
            max_sentences: Maximum sentences in summary
            
        Returns:
            Summary text
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return ""
            
        # Score sentences based on word frequency
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = Counter(words)
        
        # Remove stop words from frequency count
        for stop_word in self.STOP_WORDS:
            word_freq.pop(stop_word, None)
            
        # Score each sentence
        sentence_scores = []
        for sentence in sentences:
            words_in_sentence = re.findall(r'\b\w+\b', sentence.lower())
            score = sum(
                word_freq.get(word, 0)
                for word in words_in_sentence
                if word not in self.STOP_WORDS
            )
            # Normalize by sentence length
            if len(words_in_sentence) > 0:
                score = score / len(words_in_sentence)
            sentence_scores.append((sentence, score))
            
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        summary_sentences = [s for s, _ in sentence_scores[:max_sentences]]
        
        # Reorder sentences by original position
        summary_sentences = sorted(
            summary_sentences,
            key=lambda s: sentences.index(s) if s in sentences else 0
        )
        
        return '. '.join(summary_sentences) + '.'
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
        
    def _get_cached_analysis(self, text: str) -> Optional[ContentAnalysis]:
        """Get cached analysis if available."""
        if not self.cache_dir:
            return None
            
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct ContentAnalysis
                statistics = TextStatistics(**data['statistics'])
                return ContentAnalysis(
                    statistics=statistics,
                    key_themes=data.get('key_themes', []),
                    frequent_words=[(w, c) for w, c in data.get('frequent_words', [])],
                    reading_level=data.get('reading_level', ''),
                    estimated_reading_time=data.get('estimated_reading_time', 0),
                    language_detected=data.get('language_detected'),
                    summary=data.get('summary'),
                    chapters_analyzed=data.get('chapters_analyzed', 0)
                )
            except Exception as e:
                logger.warning(f"Failed to load cached analysis: {e}")
                
        return None
        
    def _cache_analysis(self, text: str, analysis: ContentAnalysis):
        """Cache analysis results."""
        if not self.cache_dir:
            return
            
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            data = {
                'statistics': {
                    'word_count': analysis.statistics.word_count,
                    'sentence_count': analysis.statistics.sentence_count,
                    'paragraph_count': analysis.statistics.paragraph_count,
                    'average_word_length': analysis.statistics.average_word_length,
                    'average_sentence_length': analysis.statistics.average_sentence_length,
                    'unique_words': analysis.statistics.unique_words,
                    'lexical_diversity': analysis.statistics.lexical_diversity
                },
                'key_themes': analysis.key_themes,
                'frequent_words': analysis.frequent_words,
                'reading_level': analysis.reading_level,
                'estimated_reading_time': analysis.estimated_reading_time,
                'language_detected': analysis.language_detected,
                'summary': analysis.summary,
                'chapters_analyzed': analysis.chapters_analyzed
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to cache analysis: {e}")