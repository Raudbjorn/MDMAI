"""Ollama embedding provider for local model support."""

import json
import subprocess
from typing import List, Optional, Dict, Any
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.logging_config import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_EMBEDDING_DIMENSION = 768  # Default dimension for unknown models


class OllamaEmbeddingProvider:
    """Provider for Ollama-based local embeddings."""
    
    # Popular embedding models for Ollama
    EMBEDDING_MODELS = {
        "nomic-embed-text": {
            "name": "nomic-embed-text",
            "dimension": 768,
            "description": "High-quality general-purpose embeddings",
            "size": "274MB"
        },
        "mxbai-embed-large": {
            "name": "mxbai-embed-large", 
            "dimension": 1024,
            "description": "Large embedding model for better quality",
            "size": "669MB"
        },
        "all-minilm": {
            "name": "all-minilm",
            "dimension": 384,
            "description": "Lightweight, fast embeddings",
            "size": "46MB"
        },
        "bge-small": {
            "name": "bge-small",
            "dimension": 384,
            "description": "Small but effective embeddings",
            "size": "133MB"
        },
        "bge-large": {
            "name": "bge-large",
            "dimension": 1024,
            "description": "Large BAAI general embeddings",
            "size": "1.3GB"
        }
    }
    
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama embedding provider.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self._model_info = self.EMBEDDING_MODELS.get(model_name, {
            "name": model_name,
            "dimension": None,  # Will be detected after first embedding
            "description": "Custom Ollama model"
        })
        self._embedding_dimension = self._model_info.get("dimension")
        
    def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed and running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except (requests.RequestException, ConnectionError):
            logger.warning("Ollama service not running. Checking if installed...")
            try:
                # Use direct list for static commands
                result = subprocess.run(
                    ["ollama", "--version"], 
                    capture_output=True, 
                    text=True,
                    timeout=5,  # Add timeout
                    check=False  # Don't raise on non-zero exit
                )
                return result.returncode == 0
            except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                logger.error(f"Failed to check Ollama installation: {e}")
                return False
    
    def start_ollama_service(self) -> bool:
        """Attempt to start Ollama service."""
        try:
            # Use direct list for static commands
            # Start process with proper isolation
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Isolate process group
            )
            
            # Poll for service to be ready (improved from fixed sleep)
            import time
            for _ in range(10):  # Poll for up to 5 seconds
                if self.check_ollama_installed():
                    return True
                time.sleep(0.5)
            
            # Check if process failed to start
            if process.poll() is not None:
                logger.error(f"Ollama service exited with code: {process.returncode}")
            
            return False
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Failed to start Ollama service: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """List locally available Ollama models."""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available locally."""
        available_models = self.list_available_models()
        # Check for exact match or base model name (without tag)
        base_name = model_name.split(":")[0]
        return any(base_name in model for model in available_models)
    
    def pull_model(self, model_name: str, show_progress: bool = True) -> bool:
        """
        Download an Ollama model.
        
        Args:
            model_name: Name of the model to download
            show_progress: Whether to show download progress
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Pulling Ollama model: {model_name}")
            
            # Validate model name with proper regex to prevent injection and path traversal
            import re
            if not model_name or not re.match(r'^[a-zA-Z0-9_-]+(?::[a-zA-Z0-9_-]+)?$', model_name):
                logger.error(f"Invalid model name: {model_name}")
                return False
            
            # Use streaming to show progress with timeout for initial connection
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model_name},
                stream=True,
                timeout=(30, 1800)  # 30s connect timeout, 30min read timeout for download
            )
            
            if show_progress:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if "pulling" in status.lower():
                            logger.info(f"Model download progress: {status}")
                        elif "error" in status.lower():
                            logger.error(f"Error pulling model: {status}")
                            return False
            
            # Verify model was pulled
            return self.is_model_available(model_name)
            
        except Exception as e:
            logger.error(f"Failed to pull Ollama model: {e}")
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=30  # Add proper timeout
            )
            
            if response.status_code == 200:
                embedding = response.json()["embedding"]
                
                # Update dimension if not set
                if self._embedding_dimension is None:
                    self._embedding_dimension = len(embedding)
                    
                return embedding
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to generate Ollama embedding: {e}")
            raise
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        normalize: bool = True,
        max_workers: int = 4
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with parallel processing.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = [None] * len(texts)
        
        # Process in parallel for better performance
        with ThreadPoolExecutor(max_workers=min(max_workers, len(texts))) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.generate_embedding, text): i 
                for i, text in enumerate(texts)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embedding = future.result(timeout=60)  # Per-embedding timeout
                    
                    if normalize:
                        # Normalize embedding for cosine similarity
                        embedding_np = np.array(embedding)
                        norm = np.linalg.norm(embedding_np)
                        if norm > 0:
                            embedding = (embedding_np / norm).tolist()
                    
                    embeddings[index] = embedding
                    
                except Exception as e:
                    logger.error(f"Failed to generate embedding for text {index}: {e}")
                    # Return empty embedding on failure to maintain alignment
                    embeddings[index] = [0.0] * (self._embedding_dimension or DEFAULT_EMBEDDING_DIMENSION)
        
        return embeddings
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings produced by this model."""
        return self._embedding_dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        info = {
            "provider": "ollama",
            "model_name": self.model_name,
            "dimension": self._embedding_dimension,
            "base_url": self.base_url,
            "is_available": self.is_model_available(self.model_name)
        }
        
        if self.model_name in self.EMBEDDING_MODELS:
            info.update(self.EMBEDDING_MODELS[self.model_name])
        
        return info

    @classmethod
    def prompt_for_model_selection(cls) -> Optional[str]:
        """
        Interactive prompt for user to select an Ollama embedding model.
        
        Returns:
            Selected model name or None if user declines
        """
        print("\n" + "="*60)
        print("OLLAMA EMBEDDING MODEL SELECTION")
        print("="*60)
        print("\nWould you like to use Ollama for potentially higher quality local embeddings?")
        print("This requires Ollama to be installed (https://ollama.ai)")
        print("\nAvailable embedding models:")
        print("-" * 40)
        
        models = list(cls.EMBEDDING_MODELS.items())
        for i, (key, info) in enumerate(models, 1):
            print(f"{i}. {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Dimension: {info['dimension']}")
            print(f"   Size: {info.get('size', 'Unknown')}")
            print()
        
        print(f"{len(models) + 1}. Skip - Use default Sentence Transformers")
        print("-" * 40)
        
        while True:
            try:
                choice = input(f"\nSelect option (1-{len(models) + 1}): ").strip()
                choice_num = int(choice)
                
                if choice_num == len(models) + 1:
                    print("Using default Sentence Transformers model.")
                    return None
                    
                if 1 <= choice_num <= len(models):
                    selected_model = models[choice_num - 1][1]["name"]
                    print(f"\nSelected: {selected_model}")
                    return selected_model
                else:
                    print(f"Please enter a number between 1 and {len(models) + 1}")
                    
            except (ValueError, KeyboardInterrupt):
                print("\nUsing default Sentence Transformers model.")
                return None