"""Tests for Ollama model selection in PDF processing."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.pdf_processing.pipeline import PDFProcessingPipeline
from src.pdf_processing.ollama_provider import OllamaEmbeddingProvider


@pytest.fixture
def test_client():
    """Create test client for API."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_ollama_models():
    """Mock Ollama model list."""
    return [
        {
            "name": "nomic-embed-text",
            "modified_at": "2024-01-15T10:00:00Z",
            "size": 274000000,
            "digest": "abc123",
            "details": {"families": ["bert"]}
        },
        {
            "name": "llama2:latest",
            "modified_at": "2024-01-14T10:00:00Z",
            "size": 3800000000,
            "digest": "def456",
            "details": {"families": ["llama"]}
        },
        {
            "name": "all-minilm",
            "modified_at": "2024-01-13T10:00:00Z",
            "size": 46000000,
            "digest": "ghi789",
            "details": {"families": ["bert"]}
        }
    ]


class TestOllamaModelSelection:
    """Test Ollama model selection functionality."""
    
    def test_pipeline_accepts_model_name(self):
        """Test that pipeline accepts model_name parameter."""
        pipeline = PDFProcessingPipeline(
            enable_parallel=False,
            prompt_for_ollama=False,
            model_name="nomic-embed-text"
        )
        
        assert pipeline.embedding_generator.model_name == "nomic-embed-text"
    
    def test_pipeline_defaults_without_model(self):
        """Test pipeline defaults when no model specified."""
        pipeline = PDFProcessingPipeline(
            enable_parallel=False,
            prompt_for_ollama=False
        )
        
        # Should use default (Sentence Transformers or default Ollama)
        assert pipeline.embedding_generator is not None
    
    @patch('requests.get')
    def test_api_list_models(self, mock_get, test_client, mock_ollama_models):
        """Test API endpoint for listing Ollama models."""
        # Mock Ollama API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": mock_ollama_models}
        mock_get.return_value = mock_response
        
        response = test_client.get("/api/ollama/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 3
        
        # Check model classification
        embedding_models = [m for m in data["models"] if m["type"] == "embedding"]
        assert len(embedding_models) == 2  # nomic-embed-text and all-minilm
    
    @patch('requests.get')
    def test_api_ollama_status(self, mock_get, test_client):
        """Test API endpoint for Ollama service status."""
        # Mock Ollama running
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response
        
        response = test_client.get("/api/ollama/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is True
    
    @patch('requests.get')
    def test_api_ollama_status_offline(self, mock_get, test_client):
        """Test API endpoint when Ollama is offline."""
        # Mock Ollama offline
        mock_get.side_effect = ConnectionError("Connection refused")
        
        response = test_client.get("/api/ollama/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is False
    
    @patch('requests.post')
    @patch('requests.get')
    def test_api_select_model(self, mock_get, mock_post, test_client, mock_ollama_models):
        """Test API endpoint for selecting a model."""
        # Mock Ollama API responses
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": mock_ollama_models}
        mock_get.return_value = mock_get_response
        
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3] * 256]  # batch of 1 embedding vector, 768 dimensions
        }
        mock_post.return_value = mock_post_response
        
        response = test_client.post(
            "/api/ollama/select",
            json={"model_name": "nomic-embed-text"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_name"] == "nomic-embed-text"
        assert data["dimension"] == 768
    
    @patch('requests.get')
    def test_api_select_invalid_model(self, mock_get, test_client, mock_ollama_models):
        """Test selecting a non-existent model."""
        # Mock Ollama API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": mock_ollama_models}
        mock_get.return_value = mock_response
        
        response = test_client.post(
            "/api/ollama/select",
            json={"model_name": "non-existent-model"}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestPDFUploadWithModel:
    """Test PDF upload with model selection."""
    
    @patch('src.pdf_processing.pipeline.PDFProcessingPipeline.process_pdf')
    async def test_pdf_upload_with_model(self, mock_process, test_client):
        """Test PDF upload with specific model."""
        # Mock processing result
        mock_process.return_value = {
            "success": True,
            "chunks_processed": 100,
            "embeddings_generated": 100,
            "processing_time": 5.2
        }
        
        # Create a test PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"PDF content")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = test_client.post(
                    "/api/pdf/upload",
                    files={"file": ("test.pdf", f, "application/pdf")},
                    data={
                        "rulebook_name": "Test Rulebook",
                        "system": "D&D 5e",
                        "model_name": "nomic-embed-text"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "embedding_model" in data
            assert data["embedding_model"] == "nomic-embed-text"
        finally:
            Path(tmp_path).unlink()
    
    @patch('src.pdf_processing.pipeline.PDFProcessingPipeline.process_pdf')
    async def test_pdf_upload_without_model(self, mock_process, test_client):
        """Test PDF upload without specific model (uses default)."""
        # Mock processing result
        mock_process.return_value = {
            "success": True,
            "chunks_processed": 100,
            "embeddings_generated": 100,
            "processing_time": 5.2
        }
        
        # Create a test PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"PDF content")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = test_client.post(
                    "/api/pdf/upload",
                    files={"file": ("test.pdf", f, "application/pdf")},
                    data={
                        "rulebook_name": "Test Rulebook",
                        "system": "D&D 5e"
                        # No model_name specified
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            # Should work with default model
            assert "success" in data or "chunks_processed" in data
        finally:
            Path(tmp_path).unlink()
    
    def test_pdf_upload_invalid_file(self, test_client):
        """Test PDF upload with non-PDF file."""
        # Create a non-PDF file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"Not a PDF")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = test_client.post(
                    "/api/pdf/upload",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={
                        "rulebook_name": "Test Rulebook",
                        "system": "D&D 5e"
                    }
                )
            
            assert response.status_code == 400
            data = response.json()
            assert "must be a PDF" in data["detail"]
        finally:
            Path(tmp_path).unlink()


class TestOllamaProvider:
    """Test OllamaEmbeddingProvider with model selection."""
    
    def test_provider_initialization_with_model(self):
        """Test provider initialization with specific model."""
        provider = OllamaEmbeddingProvider(model_name="nomic-embed-text")
        
        assert provider.model_name == "nomic-embed-text"
        assert provider._model_info["name"] == "nomic-embed-text"
        assert provider._model_info["dimension"] == 768
    
    def test_provider_initialization_default(self):
        """Test provider initialization with default model."""
        provider = OllamaEmbeddingProvider()
        
        assert provider.model_name == "nomic-embed-text"  # Default
    
    def test_provider_unknown_model(self):
        """Test provider with unknown model."""
        provider = OllamaEmbeddingProvider(model_name="custom-model")
        
        assert provider.model_name == "custom-model"
        assert provider._model_info["name"] == "custom-model"
        assert provider._embedding_dimension is None  # Will be detected
    
    @patch('requests.get')
    def test_check_ollama_installed(self, mock_get):
        """Test checking if Ollama is installed."""
        provider = OllamaEmbeddingProvider()
        
        # Mock Ollama running
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        assert provider.check_ollama_installed() is True
    
    @patch('requests.get')
    @patch('subprocess.run')
    def test_check_ollama_installed_not_running(self, mock_run, mock_get):
        """Test checking Ollama when service not running but installed."""
        provider = OllamaEmbeddingProvider()
        
        # Mock Ollama not running
        mock_get.side_effect = ConnectionError()
        
        # Mock ollama command exists
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        assert provider.check_ollama_installed() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])