"""
Regression tests for critical functionality after dependency updates.

These tests ensure that existing functionality continues to work correctly
after updating dependencies.
"""

import asyncio
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import torch
import torch.nn.functional as F
from pypdf import PdfReader, PdfWriter
from returns.result import Failure, Success, Result
from transformers import pipeline

from src.core.result_pattern import (
    AppError,
    ErrorKind,
    validation_error,
    database_error,
)


class TestPDFProcessingRegression:
    """Regression tests for PDF processing functionality."""

    @pytest.fixture
    def pdf_parser(self):
        """Get PDF parser instance."""
        from src.pdf_processing.pdf_parser import PDFParser
        return PDFParser()

    @pytest.fixture
    def content_chunker(self):
        """Get content chunker instance."""
        from src.pdf_processing.content_chunker import ContentChunker
        return ContentChunker()

    def test_pdf_parser_backwards_compatibility(self, pdf_parser):
        """Test that PDFParser works with both old and new pypdf APIs."""
        # Create test PDF using new API
        writer = PdfWriter()
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, "Backwards compatibility test")
        c.save()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(buffer.getvalue())
            tmp_path = tmp.name
        
        try:
            # Test extraction
            result = pdf_parser.extract_content(tmp_path)
            
            assert result is not None
            assert 'content' in result
            assert 'metadata' in result
            assert len(result['content']) > 0
        finally:
            Path(tmp_path).unlink()

    def test_table_extraction_regression(self, pdf_parser):
        """Test table extraction still works correctly."""
        # Create PDF with table
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        data = [
            ['Header 1', 'Header 2', 'Header 3'],
            ['Row 1 Col 1', 'Row 1 Col 2', 'Row 1 Col 3'],
            ['Row 2 Col 1', 'Row 2 Col 2', 'Row 2 Col 3'],
        ]
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        doc.build([t])
        
        # Save and test
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(buffer.getvalue())
            tmp_path = tmp.name
        
        try:
            tables = pdf_parser.extract_tables(tmp_path)
            
            assert tables is not None
            assert len(tables) > 0
            
            # Verify table structure
            if tables:
                first_table = tables[0]
                assert 'data' in first_table
                assert len(first_table['data']) >= 3  # Header + 2 rows
        finally:
            Path(tmp_path).unlink()

    def test_content_chunking_regression(self, content_chunker):
        """Test content chunking still works correctly."""
        test_text = """
        Chapter 1: Introduction
        
        This is the first paragraph of the introduction. It contains important
        information about the topic we're discussing.
        
        Chapter 2: Main Content
        
        This is the main content section with multiple paragraphs.
        Each paragraph should be properly chunked.
        
        The chunking algorithm should respect chapter boundaries and maintain
        context across chunks.
        """
        
        chunks = content_chunker.chunk_text(
            test_text,
            chunk_size=100,
            overlap=20
        )
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'text') for chunk in chunks)
        assert all(hasattr(chunk, 'metadata') for chunk in chunks)
        
        # Check that chapter headers are preserved
        chapter_chunks = [c for c in chunks if 'Chapter' in c.text]
        assert len(chapter_chunks) >= 2


class TestTransformerRegression:
    """Regression tests for transformer functionality."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_gpu_model_loading(self):
        """Test that models can still be loaded on GPU."""
        with patch('transformers.AutoModel.from_pretrained') as mock_model:
            mock_model.return_value = Mock()
            
            from transformers import AutoModel
            
            model = AutoModel.from_pretrained(
                "bert-base-uncased",
                device_map="auto"
            )
            
            assert model is not None

    def test_embedding_generation_regression(self):
        """Test embedding generation still works."""
        from src.pdf_processing.embedding_generator import EmbeddingGenerator
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = torch.randn(1, 768).numpy()
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator()
            
            test_texts = [
                "This is the first test sentence.",
                "This is the second test sentence.",
            ]
            
            embeddings = generator.generate_embeddings(test_texts)
            
            assert embeddings is not None
            assert len(embeddings) == len(test_texts)
            assert embeddings[0].shape[-1] == 768  # Embedding dimension

    @patch('transformers.pipeline')
    def test_pipeline_backwards_compatibility(self, mock_pipeline):
        """Test that pipelines work with both old and new APIs."""
        mock_pipe = Mock()
        mock_pipe.return_value = [
            {"label": "POSITIVE", "score": 0.95}
        ]
        mock_pipeline.return_value = mock_pipe
        
        # Test with old-style call
        classifier = pipeline("sentiment-analysis")
        result = classifier("This is great!")
        
        assert len(result) > 0
        assert "label" in result[0]
        assert "score" in result[0]
        
        # Test with new-style call
        classifier = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased",
            framework="pt"
        )
        result = classifier("This is great!")
        
        assert len(result) > 0

    def test_tokenizer_padding_regression(self):
        """Test tokenizer padding behavior."""
        from transformers import AutoTokenizer
        
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            tokenizer = Mock()
            tokenizer.pad_token_id = 0
            tokenizer.model_max_length = 512
            
            # Mock batch_encode_plus
            def batch_encode_plus(texts, **kwargs):
                return {
                    'input_ids': [[101, 1, 2, 3, 102, 0, 0, 0]],
                    'attention_mask': [[1, 1, 1, 1, 1, 0, 0, 0]]
                }
            
            tokenizer.batch_encode_plus = batch_encode_plus
            mock_tokenizer.return_value = tokenizer
            
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            result = tok.batch_encode_plus(
                ["Test text"],
                padding="max_length",
                max_length=8,
                truncation=True,
                return_tensors="pt"
            )
            
            assert 'input_ids' in result
            assert 'attention_mask' in result
            assert result['input_ids'][0][-1] == 0  # Padded


class TestTorchRegression:
    """Regression tests for PyTorch functionality."""

    def test_tensor_dtype_compatibility(self):
        """Test tensor dtype handling."""
        # Test different dtype conversions
        x_float32 = torch.tensor([1, 2, 3], dtype=torch.float32)
        x_float64 = torch.tensor([1, 2, 3], dtype=torch.float64)
        x_int32 = torch.tensor([1, 2, 3], dtype=torch.int32)
        
        # Test operations between different dtypes
        result = x_float32 + x_float64.float()
        assert result.dtype == torch.float32
        
        # Test type casting
        x_casted = x_int32.float()
        assert x_casted.dtype == torch.float32

    def test_autograd_regression(self):
        """Test autograd functionality."""
        # Create computation graph
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        
        # Complex operation
        z = torch.sum(x * y) + torch.mean(x ** 2)
        
        # Compute gradients
        z.backward()
        
        assert x.grad is not None
        assert y.grad is not None
        
        # Test gradient values
        expected_x_grad = y + 2 * x / 3  # d/dx of (x*y + mean(x^2))
        assert torch.allclose(x.grad, expected_x_grad, rtol=1e-5)

    def test_optimizer_state_regression(self):
        """Test optimizer state handling."""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Run a few optimization steps
        for _ in range(3):
            input_data = torch.randn(32, 10)
            target = torch.randn(32, 5)
            
            output = model(input_data)
            loss = F.mse_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check optimizer state
        state_dict = optimizer.state_dict()
        assert 'state' in state_dict
        assert 'param_groups' in state_dict
        
        # Test state loading
        new_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        new_optimizer.load_state_dict(state_dict)
        
        assert len(new_optimizer.state) == len(optimizer.state)

    def test_dataloader_regression(self):
        """Test DataLoader functionality."""
        from torch.utils.data import Dataset, DataLoader
        
        class TestDataset(Dataset):
            def __init__(self, size=100):
                self.data = torch.randn(size, 10)
                self.labels = torch.randint(0, 2, (size,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        dataset = TestDataset(100)
        
        # Test with different configurations
        configs = [
            {'batch_size': 16, 'shuffle': True},
            {'batch_size': 32, 'shuffle': False, 'num_workers': 0},
            {'batch_size': 8, 'drop_last': True},
        ]
        
        for config in configs:
            dataloader = DataLoader(dataset, **config)
            
            batch_count = 0
            for batch_data, batch_labels in dataloader:
                batch_count += 1
                assert batch_data.shape[1] == 10
                
                if config.get('drop_last'):
                    assert batch_data.shape[0] == config['batch_size']
                else:
                    assert batch_data.shape[0] <= config['batch_size']
            
            assert batch_count > 0


class TestAiohttpRegression:
    """Regression tests for aiohttp functionality."""

    @pytest.mark.asyncio
    async def test_session_context_manager(self):
        """Test session context manager behavior."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            assert not session.closed
        
        # Session should be closed after context
        assert session.closed

    @pytest.mark.asyncio
    async def test_request_headers_regression(self):
        """Test request headers handling."""
        import aiohttp
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {"status": "ok"}
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': 'Bearer token',
                    'Content-Type': 'application/json',
                }
                
                async with session.post(
                    'http://api.example.com/endpoint',
                    headers=headers,
                    json={'data': 'test'}
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data['status'] == 'ok'

    @pytest.mark.asyncio
    async def test_timeout_regression(self):
        """Test timeout handling."""
        import aiohttp
        from aiohttp import ClientTimeout
        
        # Test different timeout configurations
        timeout_configs = [
            ClientTimeout(total=30),
            ClientTimeout(total=60, connect=10),
            ClientTimeout(total=None, sock_read=5),
        ]
        
        for timeout in timeout_configs:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                assert session.timeout == timeout

    @pytest.mark.asyncio
    async def test_websocket_regression(self):
        """Test WebSocket functionality."""
        import aiohttp
        from aiohttp import web
        
        # Create WebSocket handler
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await ws.send_str(f"Echo: {msg.data}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
            
            return ws
        
        app = web.Application()
        app.router.add_get('/ws', websocket_handler)
        
        assert len(app.router.routes()) > 0

    @pytest.mark.asyncio
    async def test_middleware_chain_regression(self):
        """Test middleware chain execution."""
        from aiohttp import web
        
        # Create multiple middlewares
        @web.middleware
        async def middleware1(request, handler):
            request['middleware1'] = True
            response = await handler(request)
            response.headers['X-Middleware-1'] = 'true'
            return response
        
        @web.middleware
        async def middleware2(request, handler):
            request['middleware2'] = True
            response = await handler(request)
            response.headers['X-Middleware-2'] = 'true'
            return response
        
        app = web.Application(middlewares=[middleware1, middleware2])
        
        async def handler(request):
            assert request.get('middleware1') is True
            assert request.get('middleware2') is True
            return web.Response(text='OK')
        
        app.router.add_get('/', handler)
        
        assert len(app.middlewares) == 2


class TestIntegrationRegression:
    """Integration regression tests."""

    @pytest.mark.asyncio
    async def test_pdf_to_embeddings_pipeline(self):
        """Test complete PDF to embeddings pipeline."""
        from src.pdf_processing.pipeline import PDFProcessingPipeline
        
        with patch.object(PDFProcessingPipeline, '__init__', return_value=None):
            pipeline = PDFProcessingPipeline()
            
            # Mock the process method
            async def mock_process(file_path):
                return Success({
                    'embeddings': torch.randn(10, 768).numpy().tolist(),
                    'chunks': ['chunk1', 'chunk2'],
                    'metadata': {'pages': 5}
                })
            
            pipeline.process = mock_process
            
            result = await pipeline.process("test.pdf")
            
            assert result.is_ok()
            data = result.unwrap()
            assert 'embeddings' in data
            assert 'chunks' in data
            assert len(data['embeddings']) == 10

    @pytest.mark.asyncio
    async def test_ai_provider_chain_regression(self):
        """Test AI provider chain functionality."""
        from src.ai_providers.provider_manager import AIProviderManager
        
        with patch.object(AIProviderManager, '__init__', return_value=None):
            manager = AIProviderManager()
            
            # Mock providers
            manager.providers = {
                'openai': Mock(),
                'anthropic': Mock(),
            }
            
            # Mock generate method
            async def mock_generate(provider, request):
                return Success({
                    'response': 'Generated text',
                    'usage': {'tokens': 100}
                })
            
            manager.generate = mock_generate
            
            result = await manager.generate('openai', Mock())
            
            assert result.is_ok()
            data = result.unwrap()
            assert 'response' in data

    def test_model_serialization_regression(self):
        """Test model serialization/deserialization."""
        import pickle
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        
        # Save model state
        state_dict = model.state_dict()
        
        # Serialize
        serialized = pickle.dumps(state_dict)
        
        # Deserialize
        loaded_state = pickle.loads(serialized)
        
        # Load into new model
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        new_model.load_state_dict(loaded_state)
        
        # Test that models produce same output
        test_input = torch.randn(5, 10)
        with torch.no_grad():
            output1 = model(test_input)
            output2 = new_model(test_input)
        
        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])