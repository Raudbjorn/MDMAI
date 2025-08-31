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
        {"name": "nomic-embed-text", "modified_at": "2024-01-15T10:00:00Z", 
         "size": 274000000, "digest": "abc123"},
        {"name": "llama2:latest", "modified_at": "2024-01-14T10:00:00Z", 
         "size": 3800000000, "digest": "def456"},
        {"name": "all-minilm", "modified_at": "2024-01-13T10:00:00Z", 
         "size": 46000000, "digest": "ghi789"}
    ]


class TestOllamaModelSelection:
    """Test Ollama model selection functionality."""
    
    def test_pipeline_accepts_model_name(self):
        """Test that pipeline accepts model_name parameter."""
        pipeline = PDFProcessingPipeline(
            enable_parallel=False, prompt_for_ollama=False, model_name="nomic-embed-text"
        )
        assert pipeline.embedding_generator.model_name == "nomic-embed-text"
    
    def test_pipeline_defaults_without_model(self):
        """Test pipeline defaults when no model specified."""
        pipeline = PDFProcessingPipeline(enable_parallel=False, prompt_for_ollama=False)
        assert pipeline.embedding_generator is not None
    
    @patch('requests.get')
    def test_api_list_models(self, mock_get, test_client, mock_ollama_models):
        """Test API endpoint for listing Ollama models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": mock_ollama_models}
        mock_get.return_value = mock_response
        
        response = test_client.get("/api/ollama/models")
        assert response.status_code == 200
        
        models = response.json()
        assert len(models) == 3
        
        # Check for embedding models
        embedding_models = [m for m in models if m.get("model_type") == "embedding"]
        assert len(embedding_models) >= 1
    
    @patch('requests.get')
    def test_api_ollama_status_online(self, mock_get, test_client):
        """Test API endpoint for Ollama service status when online."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response
        
        response = test_client.get("/api/ollama/status")
        assert response.status_code == 200
        assert response.json()["is_running"] is True
    
    @patch('requests.get')
    def test_api_ollama_status_offline(self, mock_get, test_client):
        """Test API endpoint when Ollama is offline."""
        mock_get.side_effect = ConnectionError("Connection refused")
        response = test_client.get("/api/ollama/status")
        assert response.status_code == 200
        assert response.json()["is_running"] is False
    
    @patch('requests.get')
    def test_api_select_model(self, mock_get, test_client, mock_ollama_models):
        """Test API endpoint for selecting a model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": mock_ollama_models}
        mock_get.return_value = mock_response
        
        response = test_client.post(
            "/api/ollama/select",
            json={"model_name": "nomic-embed-text"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_name"] == "nomic-embed-text"
    
    @patch('requests.get')
    def test_api_select_invalid_model(self, mock_get, test_client, mock_ollama_models):
        """Test selecting a non-existent model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": mock_ollama_models}
        mock_get.return_value = mock_response
        
        response = test_client.post(
            "/api/ollama/select",
            json={"model_name": "non-existent-model"}
        )
        
        assert response.status_code == 404


class TestPDFUploadWithModel:
    """Test PDF upload with model selection."""
    
    def _create_test_pdf(self):
        """Create a temporary PDF file for testing."""
        tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_file.write(b"PDF content")
        tmp_file.close()
        return tmp_file.name
    
    @patch('src.pdf_processing.pipeline.PDFProcessingPipeline.process_pdf')
    async def test_pdf_upload_with_model(self, mock_process, test_client):
        """Test PDF upload with specific model."""
        mock_process.return_value = {
            "success": True, "chunks_processed": 100, 
            "embeddings_generated": 100, "processing_time": 5.2
        }
        
        tmp_path = self._create_test_pdf()
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
            assert data["embedding_model"] == "nomic-embed-text"
        finally:
            Path(tmp_path).unlink()
    
    @patch('src.pdf_processing.pipeline.PDFProcessingPipeline.process_pdf')
    async def test_pdf_upload_without_model(self, mock_process, test_client):
        """Test PDF upload without specific model (uses default)."""
        mock_process.return_value = {
            "success": True, "chunks_processed": 100, 
            "embeddings_generated": 100, "processing_time": 5.2
        }
        
        tmp_path = self._create_test_pdf()
        try:
            with open(tmp_path, 'rb') as f:
                response = test_client.post(
                    "/api/pdf/upload",
                    files={"file": ("test.pdf", f, "application/pdf")},
                    data={"rulebook_name": "Test Rulebook", "system": "D&D 5e"}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "success" in data or "chunks_processed" in data
        finally:
            Path(tmp_path).unlink()
    
    def test_pdf_upload_invalid_file(self, test_client):
        """Test PDF upload with non-PDF file."""
        tmp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        tmp_file.write(b"Not a PDF")
        tmp_file.close()
        
        try:
            with open(tmp_file.name, 'rb') as f:
                response = test_client.post(
                    "/api/pdf/upload",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"rulebook_name": "Test Rulebook", "system": "D&D 5e"}
                )
            
            assert response.status_code == 400
            assert "must be a PDF" in response.json()["detail"]
        finally:
            Path(tmp_file.name).unlink()


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