"""Embedding generation module for content chunks."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from config.logging_config import get_logger
from config.settings import settings
from src.pdf_processing.content_chunker import ContentChunk

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generates vector embeddings for content chunks."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            # Check for GPU availability
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Set model to eval mode
            self.model.eval()
            
            logger.info(
                f"Embedding model initialized",
                model=self.model_name,
                device=self.device,
                embedding_dim=self.model.get_sentence_embedding_dimension(),
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model", error=str(e))
            raise
    
    def generate_embeddings(
        self,
        chunks: List[ContentChunk],
        batch_size: Optional[int] = None,
        show_progress: bool = True
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
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Generate embeddings for batch
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
                logger.error(f"Failed to generate embeddings for batch", error=str(e))
                # Add zero embeddings for failed batch
                embedding_dim = self.model.get_sentence_embedding_dimension()
                for _ in range(len(batch_texts)):
                    embeddings.append([0.0] * embedding_dim)
        
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
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding", error=str(e))
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
        
        # Truncate if too long (most models have max token limits)
        max_length = 512  # Conservative limit for most models
        if len(prepared_text) > max_length:
            prepared_text = prepared_text[:max_length] + "..."
        
        return prepared_text
    
    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
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
        threshold: float = 0.5
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
        self,
        chunks: List[ContentChunk],
        embeddings: List[List[float]]
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
                f"Chunk count mismatch",
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
        
        expected_dim = self.model.get_sentence_embedding_dimension()
        
        for i, embedding in enumerate(embeddings):
            # Check dimension
            if len(embedding) != expected_dim:
                logger.error(
                    f"Invalid embedding dimension",
                    index=i,
                    expected=expected_dim,
                    actual=len(embedding),
                )
                return False
            
            # Check for NaN or Inf values
            if any(np.isnan(val) or np.isinf(val) for val in embedding):
                logger.error(f"Invalid embedding values", index=i)
                return False
            
            # Check if all zeros (failed embedding)
            if all(val == 0.0 for val in embedding):
                logger.warning(f"Zero embedding detected", index=i)
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_sequence_length": self.model.max_seq_length,
        }