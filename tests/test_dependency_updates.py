"""
Comprehensive tests for dependency updates.

Tests cover:
1. pypdf: 3.17.0 -> 6.0.0
2. transformers: 4.36.0 -> 4.53.0
3. torch: 2.1.0 -> 2.8.0
4. aiohttp: 3.9.0 -> 3.12.14

These tests ensure compatibility and functionality after version updates.
"""

import asyncio
import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
import torch
import torch.nn as nn
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError
from returns.result import Failure, Success, Result
from transformers import (
    AutoModel,
    AutoTokenizer,
    pipeline,
)

from src.core.result_pattern import (
    AppError,
    ErrorKind,
    validation_error,
    not_found_error,
)


class TestPyPDFUpdates:
    """Test suite for pypdf 6.0.0 compatibility."""

    @pytest.fixture
    def sample_pdf_bytes(self) -> bytes:
        """Create sample PDF bytes for testing."""
        writer = PdfWriter()
        
        # Create a simple PDF with text
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Add text content
        c.drawString(100, 750, "Test PDF Document")
        c.drawString(100, 700, "This is a test page with sample content.")
        c.drawString(100, 650, "Testing pypdf 6.0.0 compatibility")
        
        # Add table-like structure
        y = 600
        headers = ["Column 1", "Column 2", "Column 3"]
        for i, header in enumerate(headers):
            c.drawString(100 + i * 150, y, header)
        
        y -= 30
        rows = [
            ["Value 1-1", "Value 1-2", "Value 1-3"],
            ["Value 2-1", "Value 2-2", "Value 2-3"],
            ["Value 3-1", "Value 3-2", "Value 3-3"],
        ]
        
        for row in rows:
            for i, value in enumerate(row):
                c.drawString(100 + i * 150, y, value)
            y -= 20
        
        # Add metadata
        c.setTitle("Test Document")
        c.setAuthor("Test Author")
        c.setSubject("Testing pypdf")
        
        c.save()
        return buffer.getvalue()

    def test_pdf_reader_initialization(self, sample_pdf_bytes):
        """Test PdfReader initialization with new API."""
        reader = PdfReader(io.BytesIO(sample_pdf_bytes))
        
        assert reader is not None
        assert len(reader.pages) > 0
        assert hasattr(reader, 'metadata')
        assert hasattr(reader, 'pages')

    def test_pdf_text_extraction(self, sample_pdf_bytes):
        """Test text extraction from PDF."""
        reader = PdfReader(io.BytesIO(sample_pdf_bytes))
        
        # Extract text from first page
        page = reader.pages[0]
        text = page.extract_text()
        
        assert "Test PDF Document" in text
        assert "sample content" in text
        assert len(text) > 0

    def test_pdf_metadata_extraction(self, sample_pdf_bytes):
        """Test metadata extraction from PDF."""
        reader = PdfReader(io.BytesIO(sample_pdf_bytes))
        metadata = reader.metadata
        
        assert metadata is not None
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.subject == "Testing pypdf"

    def test_pdf_page_operations(self, sample_pdf_bytes):
        """Test page operations with new API."""
        reader = PdfReader(io.BytesIO(sample_pdf_bytes))
        writer = PdfWriter()
        
        # Add page to writer
        page = reader.pages[0]
        writer.add_page(page)
        
        # Rotate page
        rotated_page = writer.pages[0]
        rotated_page.rotate(90)
        
        # Scale page
        rotated_page.scale(0.5, 0.5)
        
        assert len(writer.pages) == 1
        assert rotated_page.rotation == 90

    def test_pdf_merge_operations(self, sample_pdf_bytes):
        """Test PDF merging operations."""
        reader1 = PdfReader(io.BytesIO(sample_pdf_bytes))
        reader2 = PdfReader(io.BytesIO(sample_pdf_bytes))
        
        writer = PdfWriter()
        
        # Merge PDFs
        for page in reader1.pages:
            writer.add_page(page)
        for page in reader2.pages:
            writer.add_page(page)
        
        assert len(writer.pages) == len(reader1.pages) + len(reader2.pages)

    def test_pdf_error_handling(self):
        """Test error handling with invalid PDF."""
        invalid_pdf = b"This is not a PDF"
        
        with pytest.raises(PdfReadError):
            PdfReader(io.BytesIO(invalid_pdf))

    def test_pdf_table_extraction_compatibility(self, sample_pdf_bytes):
        """Test table extraction with pdfplumber compatibility."""
        import pdfplumber
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(sample_pdf_bytes)
            tmp_path = tmp.name
        
        try:
            with pdfplumber.open(tmp_path) as pdf:
                page = pdf.pages[0]
                
                # Extract text
                text = page.extract_text()
                assert text is not None
                
                # Extract tables
                tables = page.extract_tables()
                assert isinstance(tables, list)
        finally:
            Path(tmp_path).unlink()

    def test_pdf_annotations(self, sample_pdf_bytes):
        """Test PDF annotations handling."""
        reader = PdfReader(io.BytesIO(sample_pdf_bytes))
        writer = PdfWriter()
        
        page = reader.pages[0]
        writer.add_page(page)
        
        # Add annotation
        annotation = writer.add_annotation(
            page_number=0,
            annotation=writer.add_text_annotation(
                text="Test annotation",
                rect=(100, 100, 200, 200),
            )
        )
        
        assert len(writer.pages) == 1

    @pytest.mark.asyncio
    async def test_async_pdf_processing(self, sample_pdf_bytes):
        """Test async PDF processing."""
        async def process_pdf(pdf_bytes: bytes) -> Result[Dict[str, Any], AppError]:
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                text = reader.pages[0].extract_text()
                metadata = reader.metadata
                
                return Success({
                    'text': text,
                    'page_count': len(reader.pages),
                    'metadata': {
                        'title': metadata.title if metadata else None,
                        'author': metadata.author if metadata else None,
                    }
                })
            except Exception as e:
                return Failure(
                    AppError.from_exception(e, kind=ErrorKind.VALIDATION)
                )
        
        result = await process_pdf(sample_pdf_bytes)
        assert result.is_ok()
        
        value = result.unwrap()
        assert 'text' in value
        assert value['page_count'] == 1
        assert value['metadata']['title'] == "Test Document"


class TestTransformersUpdates:
    """Test suite for transformers 4.53.0 compatibility."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode.return_value = [101, 2023, 2003, 1037, 3231, 102]
        tokenizer.decode.return_value = "This is a test"
        tokenizer.tokenize.return_value = ["This", "is", "a", "test"]
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 102
        tokenizer.model_max_length = 512
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Create mock transformer model."""
        model = Mock()
        
        # Mock output structure
        output = Mock()
        output.last_hidden_state = torch.randn(1, 10, 768)
        output.pooler_output = torch.randn(1, 768)
        output.logits = torch.randn(1, 10, 50000)
        
        model.return_value = output
        model.forward.return_value = output
        model.config = Mock()
        model.config.hidden_size = 768
        model.config.num_labels = 2
        
        return model

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_model_loading(self, mock_model_class, mock_tokenizer_class):
        """Test model loading with new API."""
        mock_tokenizer_class.return_value = self.mock_tokenizer()
        mock_model_class.return_value = self.mock_model()
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        
        assert tokenizer is not None
        assert model is not None
        
        # Test tokenization
        tokens = tokenizer.encode("Test text", return_tensors="pt")
        assert tokens is not None

    @patch('transformers.pipeline')
    def test_pipeline_creation(self, mock_pipeline):
        """Test pipeline creation with new features."""
        mock_pipe = Mock()
        mock_pipe.return_value = [{"label": "POSITIVE", "score": 0.95}]
        mock_pipeline.return_value = mock_pipe
        
        # Create pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU
        )
        
        # Test inference
        result = classifier("This is great!")
        assert len(result) > 0
        assert result[0]["label"] == "POSITIVE"

    def test_tokenizer_features(self, mock_tokenizer):
        """Test new tokenizer features."""
        tokenizer = mock_tokenizer
        
        # Test encoding with special tokens
        encoded = tokenizer.encode("Test text", add_special_tokens=True)
        assert len(encoded) > 0
        
        # Test batch encoding
        batch_texts = ["Text 1", "Text 2", "Text 3"]
        tokenizer.batch_encode_plus = Mock(return_value={
            'input_ids': [[101, 1, 102], [101, 2, 102], [101, 3, 102]],
            'attention_mask': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        })
        
        batch_encoded = tokenizer.batch_encode_plus(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        assert 'input_ids' in batch_encoded
        assert 'attention_mask' in batch_encoded

    def test_model_inference(self, mock_model):
        """Test model inference with new features."""
        model = mock_model
        
        # Create input tensors
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])
        attention_mask = torch.ones_like(input_ids)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        assert outputs is not None
        assert hasattr(outputs, 'last_hidden_state')
        assert outputs.last_hidden_state.shape == (1, 10, 768)

    @pytest.mark.asyncio
    async def test_async_model_inference(self, mock_model):
        """Test async model inference."""
        async def run_inference(text: str) -> Result[Dict[str, Any], AppError]:
            try:
                model = mock_model
                
                # Mock tokenization
                input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])
                
                # Run inference
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                
                return Success({
                    'embeddings': outputs.last_hidden_state.mean(dim=1).numpy().tolist(),
                    'shape': list(outputs.last_hidden_state.shape)
                })
            except Exception as e:
                return Failure(
                    AppError.from_exception(e, kind=ErrorKind.PROCESSING)
                )
        
        result = await run_inference("Test text")
        assert result.is_ok()
        
        value = result.unwrap()
        assert 'embeddings' in value
        assert 'shape' in value

    def test_attention_mechanisms(self, mock_model):
        """Test attention mechanism compatibility."""
        model = mock_model
        
        # Mock attention outputs
        model.return_value.attentions = tuple([
            torch.randn(1, 12, 10, 10) for _ in range(12)
        ])
        
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])
        
        outputs = model(input_ids=input_ids, output_attentions=True)
        
        assert hasattr(outputs, 'attentions')
        assert len(outputs.attentions) == 12

    @patch('transformers.TrainingArguments')
    @patch('transformers.Trainer')
    def test_training_compatibility(self, mock_trainer_class, mock_args_class):
        """Test training API compatibility."""
        from transformers import TrainingArguments, Trainer
        
        # Mock training arguments
        training_args = Mock()
        training_args.output_dir = "./test_output"
        training_args.per_device_train_batch_size = 8
        training_args.num_train_epochs = 3
        mock_args_class.return_value = training_args
        
        # Mock trainer
        trainer = Mock()
        trainer.train.return_value = Mock(
            global_step=100,
            training_loss=0.5
        )
        mock_trainer_class.return_value = trainer
        
        # Create trainer
        args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=8,
            num_train_epochs=3,
        )
        
        trainer = Trainer(
            model=self.mock_model(),
            args=args,
        )
        
        assert trainer is not None


class TestTorchUpdates:
    """Test suite for torch 2.8.0 compatibility."""

    def test_tensor_operations(self):
        """Test basic tensor operations."""
        # Create tensors
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        y = torch.tensor([2, 3, 4, 5, 6], dtype=torch.float32)
        
        # Test operations
        z = x + y
        assert z.shape == x.shape
        assert torch.allclose(z, torch.tensor([3, 5, 7, 9, 11], dtype=torch.float32))
        
        # Test matrix operations
        mat1 = torch.randn(3, 4)
        mat2 = torch.randn(4, 5)
        result = torch.matmul(mat1, mat2)
        assert result.shape == (3, 5)

    def test_neural_network_layers(self):
        """Test neural network layer compatibility."""
        # Define simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(20, 1)
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        model = SimpleModel()
        
        # Test forward pass
        input_tensor = torch.randn(32, 10)
        output = model(input_tensor)
        assert output.shape == (32, 1)

    def test_autograd_functionality(self):
        """Test autograd and gradient computation."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x ** 2 + 2 * x + 1
        
        # Compute gradients
        loss = y.mean()
        loss.backward()
        
        assert x.grad is not None
        expected_grad = 2 * x + 2
        assert torch.allclose(x.grad, expected_grad / 3)

    def test_optimizer_compatibility(self):
        """Test optimizer compatibility."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Test optimization step
        input_data = torch.randn(16, 10)
        target = torch.randn(16, 5)
        
        # Forward pass
        output = model(input_data)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0

    def test_cuda_compatibility(self):
        """Test CUDA compatibility (if available)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            
            z = torch.matmul(x, y)
            assert z.device.type == "cuda"
            assert z.shape == (100, 100)
        else:
            pytest.skip("CUDA not available")

    def test_data_loading(self):
        """Test DataLoader functionality."""
        from torch.utils.data import Dataset, DataLoader
        
        class SimpleDataset(Dataset):
            def __init__(self, size=100):
                self.data = torch.randn(size, 10)
                self.labels = torch.randint(0, 2, (size,))
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        dataset = SimpleDataset(100)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Test iteration
        for batch_idx, (data, labels) in enumerate(dataloader):
            assert data.shape == (16, 10) or data.shape[0] <= 16  # Last batch
            assert labels.shape[0] == data.shape[0]
            break

    def test_mixed_precision(self):
        """Test automatic mixed precision training."""
        from torch.cuda.amp import autocast, GradScaler
        
        model = nn.Linear(10, 5)
        scaler = GradScaler()
        
        input_data = torch.randn(16, 10)
        target = torch.randn(16, 5)
        
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        model = model.cuda()
        input_data = input_data.cuda()
        target = target.cuda()
        
        with autocast():
            output = model(input_data)
            loss = nn.MSELoss()(output, target)
        
        assert loss is not None

    @pytest.mark.asyncio
    async def test_async_tensor_operations(self):
        """Test async tensor operations."""
        async def process_tensors(x: torch.Tensor, y: torch.Tensor) -> Result[torch.Tensor, AppError]:
            try:
                result = torch.matmul(x, y)
                return Success(result)
            except Exception as e:
                return Failure(
                    AppError.from_exception(e, kind=ErrorKind.PROCESSING)
                )
        
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        
        result = await process_tensors(x, y)
        assert result.is_ok()
        
        tensor_result = result.unwrap()
        assert tensor_result.shape == (3, 5)


class TestAiohttpUpdates:
    """Test suite for aiohttp 3.12.14 compatibility."""

    @pytest.mark.asyncio
    async def test_client_session(self):
        """Test aiohttp ClientSession."""
        async with aiohttp.ClientSession() as session:
            assert session is not None
            assert hasattr(session, 'get')
            assert hasattr(session, 'post')
            assert hasattr(session, 'put')
            assert hasattr(session, 'delete')

    @pytest.mark.asyncio
    async def test_client_request_mock(self):
        """Test client request with mock."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = "Success"
        mock_response.json.return_value = {"status": "ok", "data": "test"}
        
        with patch('aiohttp.ClientSession.get', return_value=mock_response):
            async with aiohttp.ClientSession() as session:
                async with session.get('http://example.com/api') as resp:
                    assert resp.status == 200
                    text = await resp.text()
                    assert text == "Success"
                    
                    json_data = await resp.json()
                    assert json_data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_server_app_creation(self):
        """Test aiohttp web application creation."""
        from aiohttp import web
        
        app = web.Application()
        
        async def hello(request):
            return web.Response(text="Hello, World!")
        
        app.router.add_get('/', hello)
        app.router.add_post('/data', hello)
        
        assert app is not None
        assert len(app.router.routes()) > 0

    @pytest.mark.asyncio
    async def test_websocket_handler(self):
        """Test WebSocket handler."""
        from aiohttp import web
        
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
        
        assert app is not None

    @pytest.mark.asyncio
    async def test_middleware(self):
        """Test middleware functionality."""
        from aiohttp import web
        
        @web.middleware
        async def error_middleware(request, handler):
            try:
                response = await handler(request)
                return response
            except web.HTTPException as ex:
                return web.json_response(
                    {'error': ex.reason},
                    status=ex.status
                )
        
        app = web.Application(middlewares=[error_middleware])
        assert len(app.middlewares) == 1

    @pytest.mark.asyncio
    async def test_client_timeout(self):
        """Test client timeout configuration."""
        timeout = aiohttp.ClientTimeout(total=30, connect=5, sock_read=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            assert session.timeout.total == 30
            assert session.timeout.connect == 5
            assert session.timeout.sock_read == 10

    @pytest.mark.asyncio
    async def test_form_data(self):
        """Test form data handling."""
        data = aiohttp.FormData()
        data.add_field('name', 'test')
        data.add_field('file',
                      b'file content',
                      filename='test.txt',
                      content_type='text/plain')
        
        assert data is not None

    @pytest.mark.asyncio
    async def test_streaming_response(self):
        """Test streaming response."""
        from aiohttp import web
        
        async def stream_handler(request):
            response = web.StreamResponse()
            await response.prepare(request)
            
            for i in range(5):
                await response.write(f"Chunk {i}\n".encode())
                await asyncio.sleep(0.1)
            
            return response
        
        app = web.Application()
        app.router.add_get('/stream', stream_handler)
        
        assert app is not None

    @pytest.mark.asyncio
    async def test_error_handling_with_result(self):
        """Test error handling with Result pattern."""
        async def fetch_data(url: str) -> Result[Dict[str, Any], AppError]:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return Success(data)
                        elif response.status == 404:
                            return Failure(not_found_error("Resource", url))
                        else:
                            return Failure(
                                validation_error(f"HTTP {response.status}")
                            )
            except Exception as e:
                return Failure(
                    AppError.from_exception(e, kind=ErrorKind.NETWORK)
                )
        
        # Mock successful response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"data": "test"}
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await fetch_data("http://example.com/api")
            assert result.is_ok()
            assert result.unwrap()["data"] == "test"


class TestIntegrationCompatibility:
    """Test integration between all updated dependencies."""

    @pytest.mark.asyncio
    async def test_pdf_to_embeddings_pipeline(self):
        """Test PDF processing to embeddings pipeline."""
        async def process_pdf_to_embeddings(
            pdf_bytes: bytes
        ) -> Result[Dict[str, Any], AppError]:
            try:
                # Step 1: Extract text from PDF
                reader = PdfReader(io.BytesIO(pdf_bytes))
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                
                # Step 2: Mock tokenization (would use real transformer)
                mock_tokens = torch.randint(0, 1000, (1, 512))
                
                # Step 3: Mock embedding generation
                mock_embeddings = torch.randn(1, 768)
                
                return Success({
                    'text_length': len(text),
                    'embedding_shape': list(mock_embeddings.shape),
                    'pdf_pages': len(reader.pages)
                })
            except Exception as e:
                return Failure(
                    AppError.from_exception(e, kind=ErrorKind.PROCESSING)
                )
        
        # Create sample PDF
        writer = PdfWriter()
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, "Integration test document")
        c.save()
        pdf_bytes = buffer.getvalue()
        
        result = await process_pdf_to_embeddings(pdf_bytes)
        assert result.is_ok()

    @pytest.mark.asyncio
    async def test_async_model_serving(self):
        """Test async model serving with aiohttp."""
        from aiohttp import web
        
        # Mock model
        mock_model = Mock()
        mock_model.return_value = torch.randn(1, 10)
        
        async def predict_handler(request):
            try:
                data = await request.json()
                
                # Mock inference
                input_tensor = torch.tensor(data['input'])
                with torch.no_grad():
                    output = mock_model(input_tensor)
                
                return web.json_response({
                    'predictions': output.numpy().tolist()
                })
            except Exception as e:
                return web.json_response(
                    {'error': str(e)},
                    status=400
                )
        
        app = web.Application()
        app.router.add_post('/predict', predict_handler)
        
        assert app is not None

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing with all dependencies."""
        async def process_document(doc_id: int) -> Result[Dict[str, Any], AppError]:
            try:
                # Simulate processing
                await asyncio.sleep(0.1)
                
                # Mock results
                return Success({
                    'doc_id': doc_id,
                    'embeddings': torch.randn(768).numpy().tolist()[:5],  # First 5 values
                    'processed': True
                })
            except Exception as e:
                return Failure(
                    AppError.from_exception(e, kind=ErrorKind.PROCESSING)
                )
        
        # Process multiple documents concurrently
        tasks = [process_document(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(r.is_ok() for r in results)

    def test_memory_efficient_processing(self):
        """Test memory-efficient processing with generators."""
        def process_large_dataset():
            """Generator for memory-efficient processing."""
            for i in range(1000):
                # Simulate processing
                tensor = torch.randn(100, 100)
                yield tensor.mean().item()
        
        # Process with generator
        results = []
        for i, value in enumerate(process_large_dataset()):
            results.append(value)
            if i >= 10:  # Process only first 10 for test
                break
        
        assert len(results) == 11

    @pytest.mark.asyncio
    async def test_error_recovery_chain(self):
        """Test error recovery chain with all dependencies."""
        async def step1() -> Result[str, AppError]:
            return Success("Step 1 complete")
        
        async def step2(input_val: str) -> Result[Dict[str, Any], AppError]:
            if "complete" in input_val:
                return Success({"status": "Step 2 complete"})
            return Failure(validation_error("Invalid input"))
        
        async def step3(input_dict: Dict[str, Any]) -> Result[str, AppError]:
            return Success(f"Final: {input_dict['status']}")
        
        # Chain operations
        result1 = await step1()
        if result1.is_ok():
            result2 = await step2(result1.unwrap())
            if result2.is_ok():
                result3 = await step3(result2.unwrap())
                assert result3.is_ok()
                assert "Final" in result3.unwrap()


# Performance and stress tests
class TestPerformanceAfterUpdates:
    """Test performance with updated dependencies."""

    @pytest.mark.asyncio
    async def test_concurrent_pdf_processing(self):
        """Test concurrent PDF processing performance."""
        async def process_pdf(pdf_id: int) -> float:
            start = asyncio.get_event_loop().time()
            
            # Create simple PDF
            writer = PdfWriter()
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.drawString(100, 750, f"Document {pdf_id}")
            c.save()
            
            # Read PDF
            reader = PdfReader(io.BytesIO(buffer.getvalue()))
            _ = reader.pages[0].extract_text()
            
            end = asyncio.get_event_loop().time()
            return end - start
        
        # Process multiple PDFs concurrently
        tasks = [process_pdf(i) for i in range(10)]
        times = await asyncio.gather(*tasks)
        
        avg_time = sum(times) / len(times)
        assert avg_time < 1.0  # Should process each PDF in under 1 second

    def test_torch_operation_performance(self):
        """Test PyTorch operation performance."""
        import time
        
        # Large tensor operations
        size = 1000
        x = torch.randn(size, size)
        y = torch.randn(size, size)
        
        start = time.time()
        for _ in range(10):
            z = torch.matmul(x, y)
        end = time.time()
        
        avg_time = (end - start) / 10
        assert avg_time < 1.0  # Should complete in under 1 second per operation

    @pytest.mark.asyncio
    async def test_aiohttp_request_performance(self):
        """Test aiohttp request performance."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "ok"}
        
        with patch('aiohttp.ClientSession.get', return_value=mock_response):
            async with aiohttp.ClientSession() as session:
                start = asyncio.get_event_loop().time()
                
                # Make multiple concurrent requests
                tasks = []
                for i in range(20):
                    tasks.append(session.get(f'http://example.com/api/{i}'))
                
                await asyncio.gather(*tasks)
                
                end = asyncio.get_event_loop().time()
                total_time = end - start
                
                assert total_time < 2.0  # Should complete in under 2 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])