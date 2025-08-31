# Ollama Embeddings Integration

## Overview

The TTRPG Assistant now supports using local Ollama models for generating embeddings from rulebooks and source PDFs. This provides potentially higher quality embeddings compared to the default Sentence Transformers, while keeping all processing local and private.

## Features

- **Local Processing**: All embeddings are generated locally using Ollama
- **Multiple Model Options**: Choose from various embedding models optimized for different use cases
- **Automatic Download**: Models are automatically downloaded when selected
- **Fallback Support**: Automatically falls back to Sentence Transformers if Ollama is unavailable
- **User-Friendly Prompts**: Interactive selection of embedding models during setup

## Prerequisites

### Installing Ollama

1. **macOS/Linux**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Windows**:
   Download and run the installer from [https://ollama.ai/download](https://ollama.ai/download)

3. **Verify Installation**:
   ```bash
   ollama --version
   ```

## Available Embedding Models

| Model | Dimension | Size | Description |
|-------|-----------|------|-------------|
| `nomic-embed-text` | 768 | 274MB | High-quality general-purpose embeddings (Recommended) |
| `mxbai-embed-large` | 1024 | 669MB | Large model for maximum quality |
| `all-minilm` | 384 | 46MB | Lightweight, fast embeddings |
| `bge-small` | 384 | 133MB | Small but effective embeddings |
| `bge-large` | 1024 | 1.3GB | Large BAAI general embeddings |

## Configuration

### Interactive Setup

When you first run the PDF processing pipeline, you'll be prompted to choose an embedding model:

```
============================================================
OLLAMA EMBEDDING MODEL SELECTION
============================================================

Would you like to use Ollama for potentially higher quality local embeddings?
This requires Ollama to be installed (https://ollama.ai)

Available embedding models:
----------------------------------------
1. nomic-embed-text
   Description: High-quality general-purpose embeddings
   Dimension: 768
   Size: 274MB

2. mxbai-embed-large
   Description: Large embedding model for better quality
   Dimension: 1024
   Size: 669MB

[...]

6. Skip - Use default Sentence Transformers
----------------------------------------

Select option (1-6): 
```

### Environment Variables

You can also configure Ollama embeddings using environment variables:

```bash
# Enable Ollama embeddings
USE_OLLAMA_EMBEDDINGS=true

# Specify the model to use
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Ollama API endpoint (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434
```

Add these to your `.env` file to persist the configuration.

### Programmatic Usage

```python
from src.pdf_processing.embedding_generator import EmbeddingGenerator

# Use Ollama with specific model
generator = EmbeddingGenerator(
    model_name="nomic-embed-text",
    use_ollama=True
)

# Use interactive prompt
generator = EmbeddingGenerator.prompt_and_create()

# Generate embeddings
text = "Roll 3d6 for damage"
embedding = generator.generate_single_embedding(text)
```

## Performance Considerations

### Embedding Generation Speed

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `all-minilm` | Fastest | Good | Quick prototyping, small datasets |
| `nomic-embed-text` | Fast | Excellent | Recommended for most use cases |
| `mxbai-embed-large` | Moderate | Best | When quality is paramount |
| `bge-large` | Slow | Excellent | Large-scale production systems |

### Memory Requirements

- **Minimum**: 2GB RAM for small models
- **Recommended**: 8GB RAM for optimal performance
- **Large Models**: 16GB+ RAM for models like `bge-large`

## Comparison with Sentence Transformers

| Feature | Ollama | Sentence Transformers |
|---------|--------|----------------------|
| Quality | Higher (with appropriate model) | Good |
| Speed | Varies by model | Generally faster |
| Model Selection | Multiple specialized options | Limited options |
| Setup | Requires Ollama installation | Works out-of-box |
| Memory Usage | Higher | Lower |
| Privacy | 100% local | 100% local |

## Troubleshooting

### Ollama Not Found

If you see "Ollama not available", ensure:
1. Ollama is installed: `ollama --version`
2. Ollama service is running: `ollama serve`
3. The service is accessible at `http://localhost:11434`

### Model Download Issues

If model download fails:
1. Check internet connection
2. Ensure sufficient disk space
3. Try manual download: `ollama pull nomic-embed-text`

### Falling Back to Sentence Transformers

The system automatically falls back to Sentence Transformers if:
- Ollama is not installed
- Ollama service cannot be started
- Selected model cannot be downloaded
- Network issues prevent API access

### Performance Issues

If embeddings are slow:
1. Use a smaller model (`all-minilm`)
2. Reduce batch size in settings
3. Ensure Ollama has sufficient resources
4. Check if GPU acceleration is available

## Testing

Run the test script to verify your Ollama setup:

```bash
python test_ollama_embeddings.py
```

This will:
1. Check Ollama installation
2. List available models
3. Download a test model if needed
4. Generate test embeddings
5. Verify integration with the pipeline

## Best Practices

1. **Model Selection**: 
   - Start with `nomic-embed-text` for balanced performance
   - Use `mxbai-embed-large` for critical applications
   - Use `all-minilm` for rapid prototyping

2. **Batch Processing**:
   - Process documents in batches for efficiency
   - Monitor memory usage with large batches

3. **Caching**:
   - Embeddings are cached automatically
   - Clear cache when switching models

4. **Monitoring**:
   - Check logs for embedding generation times
   - Monitor Ollama service health
   - Track embedding quality metrics

## API Reference

### OllamaEmbeddingProvider

```python
class OllamaEmbeddingProvider:
    def __init__(self, model_name: str = "nomic-embed-text", 
                 base_url: str = "http://localhost:11434")
    
    def check_ollama_installed(self) -> bool
    def pull_model(self, model_name: str) -> bool
    def generate_embedding(self, text: str) -> List[float]
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]
```

### EmbeddingGenerator

```python
class EmbeddingGenerator:
    def __init__(self, model_name: Optional[str] = None, 
                 use_ollama: Optional[bool] = None)
    
    @classmethod
    def prompt_and_create(cls) -> 'EmbeddingGenerator'
    
    def generate_embeddings(self, chunks: List[ContentChunk]) -> List[List[float]]
    def generate_single_embedding(self, text: str) -> List[float]
```

## Future Enhancements

- [ ] Support for custom Ollama models
- [ ] GPU acceleration configuration
- [ ] Embedding quality metrics dashboard
- [ ] A/B testing between models
- [ ] Automatic model selection based on content type
- [ ] Distributed embedding generation
- [ ] Fine-tuning support for TTRPG-specific content