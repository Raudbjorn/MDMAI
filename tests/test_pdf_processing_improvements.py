"""Tests for PDF processing improvements."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import hashlib

from src.pdf_processing.pdf_parser import PDFParser, PDFProcessingError
from src.pdf_processing.content_chunker import ContentChunker, ContentChunk
from src.pdf_processing.embedding_generator import EmbeddingGenerator
from src.pdf_processing.adaptive_learning import AdaptiveLearningSystem
from src.pdf_processing.pipeline import PDFProcessingPipeline


class TestPDFParserImprovements:
    """Test improvements to PDF parser error handling and validation."""
    
    def test_pdf_processing_error_custom_exception(self):
        """Test that custom PDFProcessingError is properly defined."""
        error = PDFProcessingError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_file_hash_validation(self):
        """Test file hash calculation with validation."""
        parser = PDFParser()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            parser._calculate_file_hash(Path("/nonexistent/file.pdf"))
    
    def test_file_hash_io_error_handling(self):
        """Test file hash calculation handles IO errors."""
        parser = PDFParser()
        
        # Create a temporary file and make it unreadable
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Make file unreadable (Unix-specific)
            os.chmod(tmp_path, 0o000)
            
            # Should raise PDFProcessingError on permission denied
            with pytest.raises(PDFProcessingError) as exc_info:
                parser._calculate_file_hash(tmp_path)
            assert "Cannot hash file" in str(exc_info.value)
        finally:
            # Clean up
            os.chmod(tmp_path, 0o644)
            os.unlink(tmp_path)
    
    @patch('src.pdf_processing.pdf_parser.PdfReader')
    def test_pdf_extraction_error_handling(self, mock_reader):
        """Test PDF extraction with specific error handling."""
        parser = PDFParser()
        
        # Test permission error
        mock_reader.side_effect = PermissionError("Access denied")
        with pytest.raises(PermissionError):
            parser.extract_text_from_pdf("/fake/path.pdf")
        
        # Test PDF read error
        mock_reader.side_effect = Exception("Corrupted PDF")
        with pytest.raises(PDFProcessingError) as exc_info:
            parser.extract_text_from_pdf("/fake/path.pdf")
        assert "Failed to process" in str(exc_info.value)


class TestContentChunkerImprovements:
    """Test improvements to content chunker."""
    
    def test_regex_constants_defined(self):
        """Test that missing regex constants are now defined."""
        from src.pdf_processing.content_chunker import (
            STAT_BLOCK_REGEX,
            SPELL_REGEX,
            MONSTER_REGEX
        )
        
        assert STAT_BLOCK_REGEX is not None
        assert SPELL_REGEX is not None
        assert MONSTER_REGEX is not None
        
        # Test regex patterns are valid
        import re
        re.compile(STAT_BLOCK_REGEX)
        re.compile(SPELL_REGEX)
        re.compile(MONSTER_REGEX)
    
    def test_parameter_validation(self):
        """Test chunker parameter validation."""
        # Test negative max_chunk_size
        with pytest.raises(ValueError) as exc_info:
            ContentChunker(max_chunk_size=-100)
        assert "max_chunk_size must be positive" in str(exc_info.value)
        
        # Test negative chunk_overlap
        with pytest.raises(ValueError) as exc_info:
            ContentChunker(max_chunk_size=1000, chunk_overlap=-10)
        assert "chunk_overlap cannot be negative" in str(exc_info.value)
        
        # Test overlap >= chunk_size
        with pytest.raises(ValueError) as exc_info:
            ContentChunker(max_chunk_size=100, chunk_overlap=150)
        assert "chunk_overlap must be less than max_chunk_size" in str(exc_info.value)
    
    def test_valid_chunker_initialization(self):
        """Test valid chunker initialization."""
        chunker = ContentChunker(max_chunk_size=2000, chunk_overlap=200)
        assert chunker.max_chunk_size == 2000
        assert chunker.chunk_overlap == 200


class TestEmbeddingGeneratorImprovements:
    """Test improvements to embedding generator."""
    
    @patch('src.pdf_processing.embedding_generator.SentenceTransformer')
    def test_embedding_failure_raises_error(self, mock_transformer):
        """Test that embedding failures raise errors instead of adding zeros."""
        generator = EmbeddingGenerator()
        
        # Mock the model to raise an error during encoding
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("CUDA out of memory")
        generator.model = mock_model
        
        # Create test chunks
        chunks = [
            ContentChunk(
                id="test1",
                content="Test content 1",
                metadata={},
                chunk_type="narrative",
                page_start=1,
                page_end=1
            ),
            ContentChunk(
                id="test2",
                content="Test content 2",
                metadata={},
                chunk_type="narrative",
                page_start=2,
                page_end=2
            )
        ]
        
        # Should raise RuntimeError instead of silently adding zeros
        with pytest.raises(RuntimeError) as exc_info:
            generator.generate_embeddings(chunks)
        assert "Embedding generation failed" in str(exc_info.value)
    
    @patch('src.pdf_processing.embedding_generator.SentenceTransformer')
    def test_text_truncation_fixed(self, mock_transformer):
        """Test that duplicate text truncation logic is fixed."""
        generator = EmbeddingGenerator()
        
        # Mock model with max_seq_length
        mock_model = MagicMock()
        mock_model.max_seq_length = 512
        mock_model.tokenizer = None  # No tokenizer available
        generator.model = mock_model
        
        # Create a chunk with long content
        long_text = "a" * 1000
        chunk = ContentChunk(
            id="test",
            content=long_text,
            metadata={},
            chunk_type="narrative",
            page_start=1,
            page_end=1
        )
        
        # Prepare text for embedding
        prepared = generator._prepare_text_for_embedding(chunk)
        
        # Should be truncated to max_seq_length
        assert len(prepared) <= 512


class TestAdaptiveLearningImprovements:
    """Test improvements to adaptive learning system."""
    
    def test_atomic_file_operations(self):
        """Test atomic file save operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = AdaptiveLearningSystem()
            system.cache_dir = Path(tmpdir)
            
            # Add some test patterns
            system.patterns["test_system"] = {"spell": "test_pattern"}
            system.content_classifiers = {"test": [{"feature": "value"}]}
            system.extraction_metrics["test"] = {"success": 5, "total": 10}
            
            # Save patterns
            system._save_patterns()
            
            # Verify files were created
            assert (system.cache_dir / "patterns.pkl").exists()
            assert (system.cache_dir / "classifiers.pkl").exists()
            assert (system.cache_dir / "metrics.json").exists()
            
            # Verify no temporary files left behind
            temp_files = list(system.cache_dir.glob("tmp*"))
            assert len(temp_files) == 0
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted pickle files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = AdaptiveLearningSystem()
            system.cache_dir = Path(tmpdir)
            
            # Create corrupted pickle files
            patterns_file = system.cache_dir / "patterns.pkl"
            patterns_file.write_text("corrupted data")
            
            classifiers_file = system.cache_dir / "classifiers.pkl"
            classifiers_file.write_text("corrupted data")
            
            # Should handle corruption gracefully
            system._load_cached_patterns()
            
            # System should still be functional with empty data
            assert isinstance(system.patterns, dict)
            assert isinstance(system.content_classifiers, dict)


class TestPipelineImprovements:
    """Test improvements to PDF processing pipeline."""
    
    @patch('src.pdf_processing.pipeline.ChromaDBManager')
    def test_batch_storage_operations(self, mock_db_class):
        """Test batch storage operations for better performance."""
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        
        # Mock database with batch operation support
        mock_db.add_documents = MagicMock()
        
        pipeline = PDFProcessingPipeline()
        pipeline.db = mock_db
        
        # Create test chunks and embeddings
        chunks = [
            ContentChunk(
                id=f"chunk_{i}",
                content=f"Content {i}",
                metadata={"source": "test"},
                chunk_type="narrative",
                page_start=i,
                page_end=i
            )
            for i in range(5)
        ]
        
        embeddings = [[0.1] * 384 for _ in range(5)]
        
        # Store chunks
        count = pipeline._store_chunks(chunks, embeddings, "rulebook")
        
        # Should use batch operation
        mock_db.add_documents.assert_called_once()
        assert count == 5
        
        # Verify batch parameters
        call_args = mock_db.add_documents.call_args
        assert call_args[1]['collection_name'] == 'rulebooks'
        assert len(call_args[1]['documents']) == 5
        assert len(call_args[1]['embeddings']) == 5
        assert len(call_args[1]['ids']) == 5
        assert len(call_args[1]['metadatas']) == 5
    
    @patch('src.pdf_processing.pipeline.ChromaDBManager')
    def test_fallback_to_individual_operations(self, mock_db_class):
        """Test fallback to individual operations when batch not available."""
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        
        # Mock database without batch operation support
        mock_db.add_document = MagicMock()
        del mock_db.add_documents  # Remove batch method
        
        pipeline = PDFProcessingPipeline()
        pipeline.db = mock_db
        
        # Create test chunks and embeddings
        chunks = [
            ContentChunk(
                id=f"chunk_{i}",
                content=f"Content {i}",
                metadata={"source": "test"},
                chunk_type="narrative",
                page_start=i,
                page_end=i
            )
            for i in range(3)
        ]
        
        embeddings = [[0.1] * 384 for _ in range(3)]
        
        # Store chunks
        count = pipeline._store_chunks(chunks, embeddings, "rulebook")
        
        # Should fall back to individual operations
        assert mock_db.add_document.call_count == 3
        assert count == 3
    
    @patch('src.pdf_processing.pipeline.ChromaDBManager')
    def test_batch_storage_error_handling(self, mock_db_class):
        """Test error handling in batch storage."""
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        
        # Mock database that raises error
        mock_db.add_documents = MagicMock(side_effect=Exception("Database error"))
        
        pipeline = PDFProcessingPipeline()
        pipeline.db = mock_db
        
        # Create test chunks and embeddings
        chunks = [
            ContentChunk(
                id="chunk_1",
                content="Content",
                metadata={},
                chunk_type="narrative",
                page_start=1,
                page_end=1
            )
        ]
        
        embeddings = [[0.1] * 384]
        
        # Should raise RuntimeError with context
        with pytest.raises(RuntimeError) as exc_info:
            pipeline._store_chunks(chunks, embeddings, "rulebook")
        assert "Batch storage failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])