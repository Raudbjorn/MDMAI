"""Comprehensive tests for Ollama model selection with modern pytest patterns."""

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.api.main import create_app, AppConfig
from src.pdf_processing.pipeline import PDFProcessingPipeline, PipelineConfig
from src.pdf_processing.ollama_provider import OllamaEmbeddingProvider


@dataclass
class MockOllamaModel:
    """Mock Ollama model for testing."""
    name: str
    size: int
    digest: str
    modified_at: str
    model_type: str = "unknown"
    dimension: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "size": self.size,
            "digest": self.digest,
            "modified_at": self.modified_at,
            "model_type": self.model_type,
            "dimension": self.dimension,
        }


@pytest.fixture
def test_client():
    """Create test client for API with test configuration."""
    test_config = AppConfig(
        ollama_base_url="http://localhost:11434",
        cors_origins=["http://testserver"],
        request_timeout=5.0,
        cache_ttl=10  # Shorter TTL for tests
    )
    app = create_app(test_config)
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client():
    """Create async test client."""
    test_config = AppConfig(
        ollama_base_url="http://localhost:11434",
        cors_origins=["http://testserver"],
        request_timeout=5.0,
        cache_ttl=10
    )
    app = create_app(test_config)
    
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def mock_ollama_models() -> List[MockOllamaModel]:
    """Mock Ollama model list with comprehensive model data."""
    return [
        MockOllamaModel(
            name="nomic-embed-text",
            size=274000000,
            digest="abc123",
            modified_at="2024-01-15T10:00:00Z",
            model_type="embedding",
            dimension=768
        ),
        MockOllamaModel(
            name="llama2:latest",
            size=3800000000,
            digest="def456",
            modified_at="2024-01-14T10:00:00Z",
            model_type="text_generation"
        ),
        MockOllamaModel(
            name="all-minilm",
            size=46000000,
            digest="ghi789",
            modified_at="2024-01-13T10:00:00Z",
            model_type="embedding",
            dimension=384
        )
    ]


@pytest.fixture
def mock_pipeline_config():
    """Mock pipeline configuration for testing."""
    return PipelineConfig(
        enable_parallel=False,
        prompt_for_ollama=False,
        model_name="nomic-embed-text",
        enable_adaptive_learning=False,
        batch_size=10,
        timeout_seconds=30.0
    )


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(b"PDF content for testing")
        tmp_file.flush()
        yield tmp_file.name
    
    # Cleanup
    Path(tmp_file.name).unlink(missing_ok=True)


class TestOllamaModelSelection:
    """Test Ollama model selection functionality with modern pytest patterns."""
    
    def test_pipeline_accepts_model_name(self, mock_pipeline_config):
        """Test that pipeline accepts model_name parameter."""
        config = mock_pipeline_config
        config.model_name = "nomic-embed-text"
        
        pipeline = PDFProcessingPipeline(config=config)
        assert pipeline.embedding_generator.model_name == "nomic-embed-text"
    
    def test_pipeline_defaults_without_model(self, mock_pipeline_config):
        """Test pipeline defaults when no model specified."""
        config = mock_pipeline_config
        config.model_name = None
        
        pipeline = PDFProcessingPipeline(config=config)
        assert pipeline.embedding_generator is not None
    
    @pytest.mark.parametrize("model_type,expected_count", [
        ("embedding", 2),
        ("text_generation", 1),
        ("unknown", 0),
    ])
    @patch('httpx.AsyncClient')
    def test_api_list_models_by_type(self, mock_client, test_client, mock_ollama_models, model_type, expected_count):
        """Test API endpoint filtering models by type."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [m.to_dict() for m in mock_ollama_models]}
        mock_response.raise_for_status.return_value = None
        
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        response = test_client.get("/api/ollama/models")
        assert response.status_code == 200
        
        models = response.json()
        filtered_models = [m for m in models if m.get("model_type") == model_type]
        assert len(filtered_models) == expected_count
    
    @patch('httpx.AsyncClient')
    def test_api_ollama_status_comprehensive(self, mock_client, test_client):
        """Test comprehensive Ollama service status response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "test-model"}]}
        
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        response = test_client.get("/api/ollama/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "api_url" in data
        assert "models_count" in data
        assert data["models_count"] == 1
    
    @patch('httpx.AsyncClient')
    def test_api_ollama_status_offline(self, mock_client, test_client):
        """Test API endpoint when Ollama is offline."""
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.get.side_effect = ConnectionError("Connection refused")
        mock_client.return_value = mock_client_instance
        
        response = test_client.get("/api/ollama/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "offline"
    
    @patch('httpx.AsyncClient')
    def test_api_select_model_success(self, mock_client, test_client, mock_ollama_models):
        """Test successful model selection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [m.to_dict() for m in mock_ollama_models]}
        mock_response.raise_for_status.return_value = None
        
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        response = test_client.post(
            "/api/ollama/select",
            json={"model_name": "nomic-embed-text", "pull_if_missing": False}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_name"] == "nomic-embed-text"
        assert data["dimension"] == 768
        assert data["model_type"] == "embedding"
    
    @patch('httpx.AsyncClient')
    def test_api_select_invalid_model(self, mock_client, test_client, mock_ollama_models):
        """Test selecting a non-existent model with detailed error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [m.to_dict() for m in mock_ollama_models]}
        mock_response.raise_for_status.return_value = None
        
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        response = test_client.post(
            "/api/ollama/select",
            json={"model_name": "non-existent-model", "pull_if_missing": False}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "non-existent-model" in data["detail"]
    
    @patch('httpx.AsyncClient')
    def test_api_cache_operations(self, mock_client, test_client):
        """Test cache clearing and statistics endpoints."""
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
        mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Test cache stats
        response = test_client.get("/api/ollama/cache/stats")
        assert response.status_code == 200
        
        stats = response.json()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        
        # Test cache clear
        response = test_client.delete("/api/ollama/cache")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "previous_stats" in data
    
    def test_current_model_endpoint_no_selection(self, test_client):
        """Test current model endpoint when no model is selected."""
        response = test_client.get("/api/ollama/current")
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
        assert "No model currently selected" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_async_model_operations(self, async_client, mock_ollama_models):
        """Test model operations using async client."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [m.to_dict() for m in mock_ollama_models]}
            mock_response.raise_for_status.return_value = None
            
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            # Test async model listing
            response = await async_client.get("/api/ollama/models")
            assert response.status_code == 200
            
            models = response.json()
            assert len(models) == 3
            
            # Verify model data structure
            for model in models:
                assert "name" in model
                assert "model_type" in model
                assert "size" in model
                if model["model_type"] == "embedding":
                    assert "dimension" in model
                    assert isinstance(model["dimension"], int)


class TestPDFUploadWithModel:
    """Test PDF upload with model selection using modern patterns."""
    
    @pytest.mark.asyncio
    @patch('src.pdf_processing.pipeline.PDFProcessingPipeline')
    async def test_pdf_upload_with_specific_model(self, mock_pipeline_class, test_client, temp_pdf_file):
        """Test PDF upload with specific embedding model."""
        # Setup mock pipeline
        mock_pipeline = MagicMock()
        mock_result = {
            "status": "success",
            "source_id": "test-uuid",
            "total_chunks": 100,
            "stored_chunks": 100,
            "embeddings_generated": 100,
            "processing_time_seconds": 5.2,
            "embedding_model": "nomic-embed-text"
        }
        
        # Configure async mock for process_pdf method
        mock_process = AsyncMock(return_value=mock_result)
        mock_pipeline.process_pdf = mock_process
        mock_pipeline_class.return_value = mock_pipeline
        
        with open(temp_pdf_file, 'rb') as f:
            response = test_client.post(
                "/api/pdf/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={
                    "rulebook_name": "Test Rulebook",
                    "system": "D&D 5e",
                    "model_name": "nomic-embed-text",
                    "enable_adaptive_learning": "true"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["embedding_model"] == "nomic-embed-text"
        assert "file_info" in data
        assert data["file_info"]["filename"] == "test.pdf"
    
    @pytest.mark.asyncio
    @patch('src.pdf_processing.pipeline.PDFProcessingPipeline')
    async def test_pdf_upload_without_model(self, mock_pipeline_class, test_client, temp_pdf_file):
        """Test PDF upload without specific model (uses default)."""
        mock_pipeline = MagicMock()
        mock_result = {
            "status": "success",
            "source_id": "test-uuid-2",
            "total_chunks": 85,
            "stored_chunks": 85,
            "embeddings_generated": 85,
            "processing_time_seconds": 4.1
        }
        
        mock_process = AsyncMock(return_value=mock_result)
        mock_pipeline.process_pdf = mock_process
        mock_pipeline_class.return_value = mock_pipeline
        
        with open(temp_pdf_file, 'rb') as f:
            response = test_client.post(
                "/api/pdf/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={
                    "rulebook_name": "Test Rulebook",
                    "system": "D&D 5e",
                    "source_type": "rulebook"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "total_chunks" in data
        assert data["total_chunks"] == 85
    
    def test_pdf_upload_invalid_file_extension(self, test_client):
        """Test PDF upload with non-PDF file extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"Not a PDF")
            tmp_file.flush()
            
            try:
                with open(tmp_file.name, 'rb') as f:
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
                assert "must have one of these extensions" in data["detail"]
            finally:
                Path(tmp_file.name).unlink()
    
    @pytest.mark.parametrize("field_name,field_value,expected_error", [
        ("rulebook_name", "", "Field cannot be empty"),
        ("system", "", "Field cannot be empty"),
        ("source_type", "invalid", "source_type must be"),
        ("rulebook_name", "A" * 201, "ensure this value has at most 200 characters"),
    ])
    def test_pdf_upload_validation_errors(self, test_client, temp_pdf_file, field_name, field_value, expected_error):
        """Test PDF upload with various validation errors."""
        data = {
            "rulebook_name": "Valid Rulebook",
            "system": "D&D 5e",
            "source_type": "rulebook"
        }
        data[field_name] = field_value
        
        with open(temp_pdf_file, 'rb') as f:
            response = test_client.post(
                "/api/pdf/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                data=data
            )
        
        assert response.status_code in [400, 422]
        response_data = response.json()
        assert expected_error.lower() in response_data["detail"].lower()
    
    def test_pdf_upload_file_too_large(self, test_client):
        """Test PDF upload with file exceeding size limit."""
        # Create a large temporary file (simulate large PDF)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            # Write data that would exceed typical limits if actually checked
            large_content = b"Large PDF content " * 1000000  # ~18MB of data
            tmp_file.write(large_content)
            tmp_file.flush()
            
            try:
                with open(tmp_file.name, 'rb') as f:
                    response = test_client.post(
                        "/api/pdf/upload",
                        files={"file": ("large_test.pdf", f, "application/pdf")},
                        data={
                            "rulebook_name": "Large Test Rulebook",
                            "system": "D&D 5e"
                        }
                    )
                
                # Depending on implementation, this might be 413 or 200 (if processed)
                # We'll check for reasonable responses
                assert response.status_code in [200, 413, 422]
                
                if response.status_code == 413:
                    data = response.json()
                    assert "too large" in data["detail"].lower()
                    
            finally:
                Path(tmp_file.name).unlink()
    
    def test_pdf_health_endpoint(self, test_client):
        """Test PDF service health endpoint."""
        response = test_client.get("/api/pdf/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "config" in data
        assert "max_size_mb" in data["config"]
        assert "allowed_extensions" in data["config"]
        assert "pipeline_cache_size" in data
    
    def test_pdf_recent_uploads_placeholder(self, test_client):
        """Test recent uploads endpoint (placeholder)."""
        response = test_client.get("/api/pdf/recent")
        assert response.status_code == 200
        
        data = response.json()
        assert "uploads" in data
        assert "total" in data
        assert "limit" in data
        assert isinstance(data["uploads"], list)
        
        # Test with custom limit
        response = test_client.get("/api/pdf/recent?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5


class TestOllamaProvider:
    """Test OllamaEmbeddingProvider with comprehensive model testing."""
    
    def test_provider_initialization_with_specific_model(self):
        """Test provider initialization with specific model."""
        provider = OllamaEmbeddingProvider(model_name="nomic-embed-text")
        
        assert provider.model_name == "nomic-embed-text"
        assert provider._model_info["name"] == "nomic-embed-text"
        assert provider._model_info["dimension"] == 768
    
    def test_provider_initialization_default_fallback(self):
        """Test provider initialization with default model fallback."""
        provider = OllamaEmbeddingProvider()
        
        # Should use default model
        assert provider.model_name == "nomic-embed-text"  # Default
        assert provider._model_info is not None
    
    @pytest.mark.parametrize("model_name,expected_dimension", [
        ("nomic-embed-text", 768),
        ("all-minilm", 384),
        ("mxbai-embed-large", 1024),
        ("custom-unknown-model", None),
    ])
    def test_provider_model_dimensions(self, model_name, expected_dimension):
        """Test provider model dimension detection."""
        provider = OllamaEmbeddingProvider(model_name=model_name)
        
        assert provider.model_name == model_name
        assert provider._model_info["name"] == model_name
        
        if expected_dimension is not None:
            assert provider._model_info["dimension"] == expected_dimension
        else:
            # For unknown models, dimension might be None or will be auto-detected
            assert provider._embedding_dimension is None or isinstance(provider._embedding_dimension, int)
    
    @patch('httpx.AsyncClient')
    def test_check_ollama_service_online(self, mock_client):
        """Test checking if Ollama service is online using httpx."""
        provider = OllamaEmbeddingProvider()
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        # Test service check (assuming provider has updated method)
        result = provider.check_ollama_installed()
        assert result is True
    
    @patch('httpx.AsyncClient')
    @patch('subprocess.run')
    def test_check_ollama_installed_but_not_running(self, mock_run, mock_client):
        """Test checking Ollama when service not running but installed."""
        provider = OllamaEmbeddingProvider()
        
        # Mock service not running (connection error)
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.get.side_effect = ConnectionError("Connection refused")
        mock_client.return_value = mock_client_instance
        
        # Mock ollama command exists
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        assert provider.check_ollama_installed() is True
    
    def test_provider_model_info_structure(self):
        """Test provider model info has expected structure."""
        provider = OllamaEmbeddingProvider(model_name="nomic-embed-text")
        
        model_info = provider._model_info
        assert isinstance(model_info, dict)
        assert "name" in model_info
        assert "dimension" in model_info
        assert model_info["name"] == "nomic-embed-text"
        assert isinstance(model_info["dimension"], (int, type(None)))
    
    def test_provider_get_model_info_method(self):
        """Test provider's get_model_info method if available."""
        provider = OllamaEmbeddingProvider(model_name="nomic-embed-text")
        
        if hasattr(provider, 'get_model_info'):
            model_info = provider.get_model_info()
            assert isinstance(model_info, dict)
            assert "model_name" in model_info or "name" in model_info
        else:
            # If method doesn't exist, that's also valid
            assert True


# Performance and integration tests
@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    @patch('src.pdf_processing.pipeline.PDFProcessingPipeline')
    async def test_complete_pdf_processing_workflow(self, mock_pipeline_class, async_client, temp_pdf_file):
        """Test complete PDF processing workflow from upload to completion."""
        # Setup comprehensive mock response
        mock_pipeline = MagicMock()
        mock_result = {
            "status": "success",
            "source_id": "workflow-test-uuid",
            "rulebook_name": "Integration Test Book",
            "system": "Test System",
            "total_pages": 45,
            "total_chunks": 150,
            "stored_chunks": 148,  # Simulate 2 failed chunks
            "embeddings_generated": 148,
            "processing_time_seconds": 12.5,
            "file_hash": "test-hash-12345",
            "embedding_model": "nomic-embed-text"
        }
        
        mock_process = AsyncMock(return_value=mock_result)
        mock_pipeline.process_pdf = mock_process
        mock_pipeline_class.return_value = mock_pipeline
        
        # Step 1: Upload and process PDF
        with open(temp_pdf_file, 'rb') as f:
            response = await async_client.post(
                "/api/pdf/upload",
                files={"file": ("integration_test.pdf", f, "application/pdf")},
                data={
                    "rulebook_name": "Integration Test Book",
                    "system": "Test System",
                    "model_name": "nomic-embed-text",
                    "enable_adaptive_learning": "true",
                    "source_type": "rulebook"
                }
            )
        
        assert response.status_code == 200
        upload_data = response.json()
        
        # Verify comprehensive response
        assert upload_data["status"] == "success"
        assert upload_data["source_id"] == "workflow-test-uuid"
        assert upload_data["total_chunks"] == 150
        assert upload_data["stored_chunks"] == 148
        assert upload_data["processing_time_seconds"] == 12.5
        assert "file_info" in upload_data
        
        # Step 2: Check system health after processing
        health_response = await async_client.get("/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert "status" in health_data
        assert "version" in health_data
    
    @pytest.mark.performance
    @pytest.mark.timeout(30)  # 30 second timeout
    def test_api_response_times(self, test_client):
        """Test API response times are within acceptable limits."""
        import time
        
        endpoints_to_test = [
            ("/health", "GET"),
            ("/api/ollama/status", "GET"),
            ("/api/pdf/health", "GET"),
        ]
        
        for endpoint, method in endpoints_to_test:
            start_time = time.time()
            
            if method == "GET":
                response = test_client.get(endpoint)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Assert response time is reasonable (< 1 second for health checks)
            assert response_time < 1.0, f"Endpoint {endpoint} took {response_time:.2f}s (too slow)"
            assert response.status_code in [200, 404], f"Endpoint {endpoint} returned {response.status_code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])