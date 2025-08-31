"""Embedding generation module for content chunks."""

from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.logging_config import get_logger
from config.settings import settings
from src.pdf_processing.content_chunker import ContentChunk
from src.pdf_processing.ollama_provider import OllamaEmbeddingProvider

logger = get_logger(__name__)

# Constants
OLLAMA_DEFAULT_MAX_LENGTH = 8192  # Default max sequence length for Ollama models


class EmbeddingGenerator:
    """Generates vector embeddings for content chunks."""

    def __init__(self, model_name: Optional[str] = None, use_ollama: Optional[bool] = None):
        """
        Initialize embedding generator.

        Args:
            model_name: Name of the sentence transformer or Ollama model
            use_ollama: Whether to use Ollama for embeddings (if None, checks env var)
        """
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self.device = None
        self.ollama_provider = None
        
        # Check if Ollama should be used
        if use_ollama is None:
            use_ollama = os.getenv("USE_OLLAMA_EMBEDDINGS", "false").lower() == "true"
        
        self.use_ollama = use_ollama
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model (Sentence Transformer or Ollama)."""
        try:
            if self.use_ollama:
                # Initialize Ollama provider
                self.ollama_provider = OllamaEmbeddingProvider(self.model_name)
                
                # Check if Ollama is available
                if not self.ollama_provider.check_ollama_installed():
                    logger.error("Ollama service is not available. Please start the Ollama service manually before proceeding.")
                    logger.warning("Falling back to Sentence Transformers")
                    self.use_ollama = False
                    self._initialize_sentence_transformer()
                    return
                
                # Check if model needs to be downloaded
                if not self.ollama_provider.is_model_available(self.model_name):
                    logger.info(f"Ollama model {self.model_name} not found locally, downloading...")
                    if not self.ollama_provider.pull_model(self.model_name):
                        logger.error(f"Failed to pull Ollama model {self.model_name}")
                        raise RuntimeError(f"Could not download Ollama model: {self.model_name}")
                
                logger.info(
                    "Ollama embedding model initialized",
                    model=self.model_name,
                    provider="ollama"
                )
            else:
                self._initialize_sentence_transformer()

        except Exception as e:
            logger.error("Failed to initialize embedding model", error=str(e))
            raise
    
    def _initialize_sentence_transformer(self):
        """Initialize the sentence transformer model."""
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Set model to eval mode
        self.model.eval()

        logger.info(
            "Sentence Transformer model initialized",
            model=self.model_name,
            device=self.device,
            embedding_dim=self.model.get_sentence_embedding_dimension(),
        )

    def generate_embeddings(
        self,
        chunks: List[ContentChunk],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of content chunks.

        Args:
            chunks: List of content chunks
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []

        batch_size = batch_size or settings.embedding_batch_size

        # Extract text from chunks
        texts = [self._prepare_text_for_embedding(chunk) for chunk in chunks]

        embeddings = []

        # Process in batches
        total_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings", total=total_batches)

        for i in iterator:
            batch_texts = texts[i : i + batch_size]

            try:
                if self.use_ollama:
                    # Generate embeddings using Ollama (no batch_size param)
                    batch_embeddings = self.ollama_provider.generate_embeddings_batch(
                        batch_texts,
                        normalize=True
                    )
                    embeddings.extend(batch_embeddings)
                else:
                    # Generate embeddings using Sentence Transformers
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,  # Normalize for cosine similarity
                        show_progress_bar=False,
                    )

                    # Convert to list and add to results
                    for embedding in batch_embeddings:
                        embeddings.append(embedding.tolist())

            except Exception as e:
                logger.error("Failed to generate embeddings for batch", error=str(e))
                # Re-raise the exception instead of silently adding zero embeddings
                # Zero embeddings corrupt search quality
                raise RuntimeError(f"Embedding generation failed for batch: {e}") from e

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            if self.use_ollama:
                embedding = self.ollama_provider.generate_embedding(text)
                # Normalize if needed
                embedding_np = np.array(embedding)
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    embedding = (embedding_np / norm).tolist()
                return embedding
            else:
                embedding = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return embedding.tolist()

        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            # Raise exception instead of returning zero embeddings
            # Zero embeddings would corrupt search quality
            raise ValueError(f"Failed to generate embedding for text: {str(e)}")

    def _prepare_text_for_embedding(self, chunk: ContentChunk) -> str:
        """
        Prepare chunk text for embedding generation.

        Args:
            chunk: Content chunk

        Returns:
            Prepared text
        """
        # Combine content with metadata for richer embeddings
        text_parts = []

        # Add section context if available
        if chunk.section:
            text_parts.append(f"Section: {chunk.section}")
        if chunk.subsection:
            text_parts.append(f"Subsection: {chunk.subsection}")

        # Add content type
        text_parts.append(f"Type: {chunk.chunk_type}")

        # Add main content
        text_parts.append(chunk.content)

        # Combine with newlines
        prepared_text = "\n".join(text_parts)

        # Truncate if too long using model's actual max sequence length
        if self.use_ollama:
            # Ollama models typically handle longer contexts
            max_length = OLLAMA_DEFAULT_MAX_LENGTH
            if len(prepared_text) > max_length:
                prepared_text = prepared_text[:max_length]
        else:
            max_length = getattr(self.model, "max_seq_length", 512)

            if hasattr(self.model, "tokenizer"):
                # Tokenize, truncate, and decode back to text
                tokens = self.model.tokenizer.encode(
                    prepared_text, max_length=max_length, truncation=True, add_special_tokens=True
                )
                prepared_text = self.model.tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                # Fallback: truncate by character count if tokenizer is unavailable
                if len(prepared_text) > max_length:
                    prepared_text = prepared_text[:max_length]

        return prepared_text

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Ensure in [0, 1] range
        return max(0.0, min(1.0, similarity))

    def find_similar_chunks(
        self,
        query_embedding: List[float],
        chunk_embeddings: List[List[float]],
        chunks: List[ContentChunk],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[ContentChunk, float]]:
        """
        Find chunks similar to a query embedding.

        Args:
            query_embedding: Query embedding vector
            chunk_embeddings: List of chunk embeddings
            chunks: List of content chunks
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not chunk_embeddings or not chunks:
            return []

        # Calculate similarities
        similarities = []
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = self.calculate_similarity(query_embedding, chunk_embedding)
            if similarity >= threshold:
                similarities.append((chunks[i], similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        return similarities[:top_k]

    def update_chunk_embeddings(
        self, chunks: List[ContentChunk], embeddings: List[List[float]]
    ) -> List[ContentChunk]:
        """
        Update chunks with their embeddings.

        Args:
            chunks: List of content chunks
            embeddings: List of embedding vectors

        Returns:
            Updated chunks with embeddings
        """
        if len(chunks) != len(embeddings):
            logger.warning(
                "Chunk count mismatch",
                chunks=len(chunks),
                embeddings=len(embeddings),
            )
            return chunks

        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata["embedding"] = embedding

        return chunks

    def validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """
        Validate embedding quality.

        Args:
            embeddings: List of embedding vectors

        Returns:
            True if embeddings are valid
        """
        if not embeddings:
            return False

        if self.use_ollama:
            expected_dim = self.ollama_provider.get_embedding_dimension()
            if expected_dim is None:
                # Dimension will be set after first embedding
                expected_dim = len(embeddings[0]) if embeddings else 0
        else:
            expected_dim = self.model.get_sentence_embedding_dimension()

        for i, embedding in enumerate(embeddings):
            # Check dimension
            if len(embedding) != expected_dim:
                logger.error(
                    "Invalid embedding dimension",
                    index=i,
                    expected=expected_dim,
                    actual=len(embedding),
                )
                return False

            # Check for NaN or Inf values
            if any(np.isnan(val) or np.isinf(val) for val in embedding):
                logger.error("Invalid embedding values", index=i)
                return False

            # Check if all zeros (failed embedding)
            if all(val == 0.0 for val in embedding):
                logger.warning("Zero embedding detected", index=i)

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.

        Returns:
            Model information dictionary
        """
        if self.use_ollama:
            return self.ollama_provider.get_model_info()
        else:
            return {
                "provider": "sentence_transformers",
                "model_name": self.model_name,
                "device": self.device,
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "max_sequence_length": self.model.max_seq_length,
            }
    
    @classmethod
    def prompt_and_create(cls) -> 'EmbeddingGenerator':
        """
        Prompt user for embedding model choice and create generator.
        
        Returns:
            Configured EmbeddingGenerator instance
        """
        # Check if already configured via environment
        if os.getenv("USE_OLLAMA_EMBEDDINGS"):
            use_ollama = os.getenv("USE_OLLAMA_EMBEDDINGS", "false").lower() == "true"
            model_name = os.getenv("OLLAMA_EMBEDDING_MODEL") if use_ollama else None
            return cls(model_name=model_name, use_ollama=use_ollama)
        
        # Prompt for Ollama model selection
        selected_model = OllamaEmbeddingProvider.prompt_for_model_selection()
        
        if selected_model:
            # User selected an Ollama model
            return cls(model_name=selected_model, use_ollama=True)
        else:
            # User chose default Sentence Transformers
            return cls(use_ollama=False)
