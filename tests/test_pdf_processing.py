"""Comprehensive tests for PDF processing modules."""

import pytest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.pdf_processing.pdf_parser import PDFParser
from src.pdf_processing.content_chunker import ContentChunker, ContentChunk
from src.pdf_processing.embedding_generator import EmbeddingGenerator
from src.pdf_processing.adaptive_learning import AdaptiveLearningSystem, PatternTemplate
from src.pdf_processing.pipeline import PDFProcessingPipeline


class TestPDFParser:
    """Test suite for PDFParser class."""
    
    @pytest.fixture
    def parser(self):
        """Create a PDFParser instance."""
        return PDFParser()
    
    def test_initialization(self, parser):
        """Test PDFParser initialization."""
        assert parser is not None
        assert hasattr(parser, 'extract_content')
        assert hasattr(parser, 'extract_tables')
    
    @patch('src.pdf_processing.pdf_parser.PdfReader')
    def test_extract_with_pypdf(self, mock_pdf_reader, parser):
        """Test extraction using PyPDF2."""
        # Mock PDF reader
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test content from page 1"
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        # Test extraction
        result = parser._extract_with_pypdf("test.pdf")
        
        assert len(result) == 1
        assert result[0]["page_num"] == 1
        assert result[0]["text"] == "Test content from page 1"
    
    @patch('src.pdf_processing.pdf_parser.pdfplumber.open')
    def test_extract_with_pdfplumber(self, mock_plumber_open, parser):
        """Test extraction using pdfplumber."""
        # Mock pdfplumber
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test content from plumber"
        mock_page.extract_tables.return_value = [
            [["Header1", "Header2"], ["Value1", "Value2"]]
        ]
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        mock_plumber_open.return_value = mock_pdf
        
        # Test extraction
        result = parser._extract_with_pdfplumber("test.pdf")
        
        assert len(result) == 1
        assert result[0]["page_num"] == 1
        assert result[0]["text"] == "Test content from plumber"
        assert len(result[0]["tables"]) == 1
    
    def test_calculate_file_hash(self, parser):
        """Test file hash calculation."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write("Test content for hashing")
            tmp_path = tmp.name
        
        try:
            # Calculate hash
            file_hash = parser._calculate_file_hash(tmp_path)
            
            # Verify hash format
            assert len(file_hash) == 64  # SHA256 hash length
            assert all(c in '0123456789abcdef' for c in file_hash)
        finally:
            Path(tmp_path).unlink()
    
    def test_table_to_markdown(self, parser):
        """Test table to markdown conversion."""
        table = [
            ["Header 1", "Header 2"],
            ["Value 1", "Value 2"],
            ["Value 3", "Value 4"]
        ]
        
        markdown = parser._table_to_markdown(table)
        
        assert "| Header 1 | Header 2 |" in markdown
        assert "|----------|----------|" in markdown
        assert "| Value 1 | Value 2 |" in markdown
        assert "| Value 3 | Value 4 |" in markdown


class TestContentChunker:
    """Test suite for ContentChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create a ContentChunker instance."""
        return ContentChunker(max_chunk_size=100, chunk_overlap=20)
    
    def test_initialization(self, chunker):
        """Test ContentChunker initialization."""
        assert chunker.max_chunk_size == 100
        assert chunker.chunk_overlap == 20
    
    def test_create_chunks_basic(self, chunker):
        """Test basic chunk creation."""
        content = [{
            "page_num": 1,
            "text": "This is a test. " * 10,  # 160 chars
            "tables": []
        }]
        
        chunks = chunker.create_chunks(
            content=content,
            source_id="test_source",
            rulebook_name="Test Book",
            system="Test System"
        )
        
        assert len(chunks) > 1  # Should be split due to size
        assert all(isinstance(chunk, ContentChunk) for chunk in chunks)
        assert all(chunk.source_id == "test_source" for chunk in chunks)
    
    def test_classify_content_type(self, chunker):
        """Test content type classification."""
        # Test spell detection
        spell_text = "Fireball\n3rd-level evocation\nCasting Time: 1 action"
        assert chunker._classify_content_type(spell_text) == "spell"
        
        # Test monster detection
        monster_text = "Dragon\nHit Points: 200\nArmor Class: 18\nChallenge Rating: 15"
        assert chunker._classify_content_type(monster_text) == "monster"
        
        # Test rule detection
        rule_text = "You can take only one bonus action on your turn"
        assert chunker._classify_content_type(rule_text) == "rule"
        
        # Test table detection
        table_text = "| Level | Proficiency |\n|-------|-------------|"
        assert chunker._classify_content_type(table_text) == "table"
        
        # Default to narrative
        narrative_text = "The ancient castle stood on the hill."
        assert chunker._classify_content_type(narrative_text) == "narrative"
    
    def test_chunk_overlap(self, chunker):
        """Test that chunks have proper overlap."""
        content = [{
            "page_num": 1,
            "text": " ".join([f"Word{i}" for i in range(50)]),
            "tables": []
        }]
        
        chunks = chunker.create_chunks(
            content=content,
            source_id="test",
            rulebook_name="Test",
            system="Test"
        )
        
        if len(chunks) > 1:
            # Check that consecutive chunks have overlap
            for i in range(len(chunks) - 1):
                chunk1_words = chunks[i].content.split()
                chunk2_words = chunks[i + 1].content.split()
                
                # Should have some overlapping words
                overlap = set(chunk1_words[-10:]) & set(chunk2_words[:10])
                assert len(overlap) > 0


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create an EmbeddingGenerator instance with mocked model."""
        with patch('src.pdf_processing.embedding_generator.SentenceTransformer'):
            gen = EmbeddingGenerator(model_name="test-model")
            # Mock the model methods
            gen.model.encode = Mock(return_value=MagicMock(tolist=lambda: [0.1] * 384))
            gen.model.get_sentence_embedding_dimension = Mock(return_value=384)
            return gen
    
    def test_initialization(self):
        """Test EmbeddingGenerator initialization."""
        with patch('src.pdf_processing.embedding_generator.SentenceTransformer') as mock_st:
            gen = EmbeddingGenerator()
            mock_st.assert_called_once()
    
    def test_generate_embeddings(self, generator):
        """Test embedding generation for chunks."""
        chunks = [
            ContentChunk(
                chunk_id="1",
                source_id="test",
                content="Test content 1",
                chunk_type="rule",
                page_num=1,
                chunk_index=0,
                metadata={}
            ),
            ContentChunk(
                chunk_id="2",
                source_id="test",
                content="Test content 2",
                chunk_type="spell",
                page_num=2,
                chunk_index=1,
                metadata={}
            )
        ]
        
        embeddings = generator.generate_embeddings(chunks)
        
        assert len(embeddings) == 2
        assert all(len(emb) == 384 for emb in embeddings.values())
        assert "1" in embeddings
        assert "2" in embeddings
    
    def test_generate_single_embedding(self, generator):
        """Test single embedding generation."""
        text = "Test text for embedding"
        
        embedding = generator.generate_single_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
    
    def test_embedding_error_handling(self):
        """Test that embedding errors are properly raised."""
        with patch('src.pdf_processing.embedding_generator.SentenceTransformer'):
            gen = EmbeddingGenerator()
            gen.model.encode = Mock(side_effect=Exception("Model error"))
            
            with pytest.raises(ValueError) as exc_info:
                gen.generate_single_embedding("test")
            
            assert "Failed to generate embedding" in str(exc_info.value)
    
    def test_calculate_similarity(self, generator):
        """Test similarity calculation between embeddings."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        emb3 = [0.0, 1.0, 0.0]
        
        # Same embeddings should have similarity 1.0
        assert generator.calculate_similarity(emb1, emb2) == pytest.approx(1.0)
        
        # Orthogonal embeddings should have similarity 0.0
        assert generator.calculate_similarity(emb1, emb3) == pytest.approx(0.0)


class TestAdaptiveLearningSystem:
    """Test suite for AdaptiveLearningSystem class."""
    
    @pytest.fixture
    def learning_system(self):
        """Create an AdaptiveLearningSystem instance."""
        return AdaptiveLearningSystem()
    
    def test_initialization(self, learning_system):
        """Test AdaptiveLearningSystem initialization."""
        assert learning_system.patterns == {}
        assert learning_system.learning_metrics == {}
    
    def test_learn_patterns(self, learning_system):
        """Test pattern learning from content."""
        content = [{
            "page_num": 1,
            "text": "Spell Name\n3rd-level evocation\nCasting Time: 1 action",
            "tables": []
        }]
        
        patterns = learning_system.learn_patterns(
            content=content,
            system="D&D 5e",
            rulebook_name="Test Book"
        )
        
        assert "spell_pattern" in patterns
        assert patterns["spell_pattern"]["system"] == "D&D 5e"
    
    def test_apply_patterns(self, learning_system):
        """Test pattern application to new content."""
        # First, learn some patterns
        content = [{
            "page_num": 1,
            "text": "Fireball\n3rd-level evocation",
            "tables": []
        }]
        
        learning_system.learn_patterns(content, "D&D 5e", "PHB")
        
        # Apply patterns to new content
        new_content = [{
            "page_num": 1,
            "text": "Lightning Bolt\n3rd-level evocation",
            "tables": []
        }]
        
        enhanced = learning_system.apply_patterns(new_content, "D&D 5e")
        
        assert enhanced is not None
        assert len(enhanced) == len(new_content)
    
    def test_pattern_template(self):
        """Test PatternTemplate class."""
        template = PatternTemplate(
            pattern_type="spell",
            system="D&D 5e",
            regex_pattern=r"(\w+)\n(\d+).*level",
            extraction_fields=["name", "level"]
        )
        
        assert template.pattern_type == "spell"
        assert template.system == "D&D 5e"
        assert template.usage_count == 0
        assert template.success_rate == 1.0
    
    def test_update_metrics(self, learning_system):
        """Test metrics updating."""
        learning_system.update_metrics(
            system="D&D 5e",
            success=True,
            pattern_type="spell"
        )
        
        assert "D&D 5e" in learning_system.learning_metrics
        assert learning_system.learning_metrics["D&D 5e"]["patterns_learned"] == 1
        assert learning_system.learning_metrics["D&D 5e"]["success_rate"] == 1.0


class TestPDFProcessingPipeline:
    """Test suite for PDFProcessingPipeline class."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a PDFProcessingPipeline instance with mocked components."""
        with patch('src.pdf_processing.pipeline.PDFParser'), \
             patch('src.pdf_processing.pipeline.ContentChunker'), \
             patch('src.pdf_processing.pipeline.EmbeddingGenerator'), \
             patch('src.pdf_processing.pipeline.AdaptiveLearningSystem'), \
             patch('src.pdf_processing.pipeline.get_db_manager'):
            return PDFProcessingPipeline()
    
    @pytest.mark.asyncio
    async def test_process_pdf_success(self, pipeline):
        """Test successful PDF processing."""
        # Mock the components
        pipeline.parser.extract_content = Mock(return_value={
            "pages": [{"page_num": 1, "text": "Test", "tables": []}],
            "metadata": {"title": "Test"},
            "file_hash": "abc123",
            "toc": []
        })
        
        pipeline.chunker.create_chunks = Mock(return_value=[
            ContentChunk(
                chunk_id="1",
                source_id="test",
                content="Test",
                chunk_type="rule",
                page_num=1,
                chunk_index=0,
                metadata={}
            )
        ])
        
        pipeline.embedding_generator.generate_embeddings = Mock(return_value={
            "1": [0.1] * 384
        })
        
        pipeline.db_manager.search = Mock(return_value=[])
        pipeline.db_manager.add_document = Mock()
        
        # Process PDF
        result = await pipeline.process_pdf(
            pdf_path="test.pdf",
            rulebook_name="Test Book",
            system="Test System",
            source_type="rulebook"
        )
        
        assert result["status"] == "success"
        assert result["source_id"] == "test_system_test_book"
        assert result["total_chunks"] == 1
        assert result["total_pages"] == 1
    
    @pytest.mark.asyncio
    async def test_process_pdf_duplicate(self, pipeline):
        """Test duplicate PDF detection."""
        # Mock duplicate detection
        pipeline.parser.extract_content = Mock(return_value={
            "pages": [],
            "metadata": {},
            "file_hash": "abc123",
            "toc": []
        })
        
        pipeline.db_manager.search = Mock(return_value=[
            {"metadata": {"file_hash": "abc123"}}
        ])
        
        # Process PDF
        result = await pipeline.process_pdf(
            pdf_path="test.pdf",
            rulebook_name="Test Book",
            system="Test System"
        )
        
        assert result["status"] == "duplicate"
        assert "already processed" in result["message"]
    
    @pytest.mark.asyncio
    async def test_process_pdf_error(self, pipeline):
        """Test error handling in PDF processing."""
        # Mock an error
        pipeline.parser.extract_content = Mock(side_effect=Exception("Parse error"))
        
        # Process PDF
        result = await pipeline.process_pdf(
            pdf_path="test.pdf",
            rulebook_name="Test Book",
            system="Test System"
        )
        
        assert result["status"] == "error"
        assert "Parse error" in result["error"]
    
    def test_validate_pdf_path(self, pipeline):
        """Test PDF path validation."""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Should not raise for valid PDF
            pipeline._validate_pdf_path(tmp_path)
            
            # Should raise for non-existent file
            with pytest.raises(FileNotFoundError):
                pipeline._validate_pdf_path("nonexistent.pdf")
            
            # Should raise for non-PDF file
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as txt:
                txt_path = txt.name
            
            with pytest.raises(ValueError):
                pipeline._validate_pdf_path(txt_path)
            
            Path(txt_path).unlink()
        finally:
            Path(tmp_path).unlink()


# Integration test
class TestIntegration:
    """Integration tests for the complete PDF processing flow."""
    
    @pytest.mark.asyncio
    @patch('src.pdf_processing.pipeline.get_db_manager')
    async def test_end_to_end_processing(self, mock_get_db):
        """Test complete PDF processing flow."""
        # Mock database
        mock_db = Mock()
        mock_db.search = Mock(return_value=[])
        mock_db.add_document = Mock()
        mock_get_db.return_value = mock_db
        
        # Create pipeline
        pipeline = PDFProcessingPipeline()
        
        # Create a minimal test PDF content
        with patch.object(pipeline.parser, 'extract_content') as mock_extract:
            mock_extract.return_value = {
                "pages": [
                    {
                        "page_num": 1,
                        "text": "Chapter 1: Introduction\nThis is test content.",
                        "tables": []
                    }
                ],
                "metadata": {"title": "Test PDF"},
                "file_hash": "unique_hash_123",
                "toc": [{"title": "Chapter 1", "page": 1}]
            }
            
            # Process the PDF
            result = await pipeline.process_pdf(
                pdf_path="test.pdf",
                rulebook_name="Test Rulebook",
                system="Test System",
                source_type="rulebook",
                enable_adaptive_learning=True
            )
            
            # Verify results
            assert result["status"] == "success"
            assert result["total_pages"] == 1
            assert result["total_chunks"] > 0
            assert mock_db.add_document.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])