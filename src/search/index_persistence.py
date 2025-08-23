"""BM25 index persistence for improved performance."""

import hashlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class IndexPersistence:
    """Manages persistence of BM25 indices to disk."""

    def __init__(self, index_dir: Optional[Path] = None):
        """
        Initialize index persistence manager.

        Args:
            index_dir: Directory to store indices
        """
        self.index_dir = index_dir or settings.cache_dir / "indices"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.index_dir / "index_metadata.json"
        self.metadata = self._load_metadata()

    def save_index(
        self,
        collection_name: str,
        bm25_index: BM25Okapi,
        documents: List[Dict[str, Any]],
        tokenized_docs: List[List[str]],
    ) -> bool:
        """
        Save BM25 index to disk.

        Args:
            collection_name: Name of the collection
            bm25_index: BM25 index object
            documents: Original documents
            tokenized_docs: Tokenized documents used for index

        Returns:
            Success status
        """
        try:
            # Generate index filename
            index_file = self.index_dir / f"{collection_name}_bm25.pkl"
            docs_file = self.index_dir / f"{collection_name}_docs.pkl"
            tokens_file = self.index_dir / f"{collection_name}_tokens.pkl"

            # Save BM25 index
            with open(index_file, "wb") as f:
                pickle.dump(bm25_index, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save documents
            with open(docs_file, "wb") as f:
                pickle.dump(documents, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save tokenized documents
            with open(tokens_file, "wb") as f:
                pickle.dump(tokenized_docs, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Update metadata
            self.metadata[collection_name] = {
                "index_file": str(index_file),
                "docs_file": str(docs_file),
                "tokens_file": str(tokens_file),
                "document_count": len(documents),
                "created_at": datetime.utcnow().isoformat(),
                "checksum": self._calculate_checksum(documents),
            }
            self._save_metadata()

            logger.info(f"Saved BM25 index for {collection_name}", document_count=len(documents))
            return True

        except Exception as e:
            logger.error(f"Failed to save index for {collection_name}: {str(e)}")
            return False

    def load_index(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Load BM25 index from disk.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with index, documents, and tokens, or None if not found
        """
        try:
            # Check if index exists in metadata
            if collection_name not in self.metadata:
                logger.debug(f"No saved index found for {collection_name}")
                return None

            meta = self.metadata[collection_name]

            # Load BM25 index
            with open(meta["index_file"], "rb") as f:
                bm25_index = pickle.load(f)

            # Load documents
            with open(meta["docs_file"], "rb") as f:
                documents = pickle.load(f)

            # Load tokenized documents
            with open(meta["tokens_file"], "rb") as f:
                tokenized_docs = pickle.load(f)

            logger.info(f"Loaded BM25 index for {collection_name}", document_count=len(documents))

            return {
                "index": bm25_index,
                "documents": documents,
                "tokenized_docs": tokenized_docs,
                "metadata": meta,
            }

        except Exception as e:
            logger.error(f"Failed to load index for {collection_name}: {str(e)}")
            # Remove corrupted index from metadata
            if collection_name in self.metadata:
                del self.metadata[collection_name]
                self._save_metadata()
            return None

    def is_index_valid(self, collection_name: str, current_documents: List[Dict[str, Any]]) -> bool:
        """
        Check if saved index is still valid for current documents.

        Args:
            collection_name: Name of the collection
            current_documents: Current documents in collection

        Returns:
            True if index is valid, False otherwise
        """
        if collection_name not in self.metadata:
            return False

        meta = self.metadata[collection_name]

        # Check document count
        if meta["document_count"] != len(current_documents):
            logger.debug(f"Document count mismatch for {collection_name}")
            return False

        # Check checksum
        current_checksum = self._calculate_checksum(current_documents)
        if meta.get("checksum") != current_checksum:
            logger.debug(f"Checksum mismatch for {collection_name}")
            return False

        # Check if files exist
        for file_key in ["index_file", "docs_file", "tokens_file"]:
            if not os.path.exists(meta[file_key]):
                logger.debug(f"Missing file for {collection_name}: {meta[file_key]}")
                return False

        return True

    def delete_index(self, collection_name: str) -> bool:
        """
        Delete saved index for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Success status
        """
        try:
            if collection_name in self.metadata:
                meta = self.metadata[collection_name]

                # Delete files
                for file_key in ["index_file", "docs_file", "tokens_file"]:
                    file_path = meta.get(file_key)
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)

                # Remove from metadata
                del self.metadata[collection_name]
                self._save_metadata()

                logger.info(f"Deleted index for {collection_name}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete index for {collection_name}: {str(e)}")
            return False

    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about all saved indices.

        Returns:
            Dictionary with index information
        """
        info = {}
        for collection_name, meta in self.metadata.items():
            # Calculate file sizes
            total_size = 0
            for file_key in ["index_file", "docs_file", "tokens_file"]:
                file_path = meta.get(file_key)
                if file_path and os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)

            info[collection_name] = {
                "document_count": meta["document_count"],
                "created_at": meta["created_at"],
                "size_mb": total_size / (1024 * 1024),
            }

        return info

    def cleanup_old_indices(self, max_age_days: int = 7) -> int:
        """
        Clean up indices older than specified days.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of indices cleaned up
        """
        cleaned = 0
        current_time = datetime.utcnow()

        for collection_name in list(self.metadata.keys()):
            meta = self.metadata[collection_name]
            created_at = datetime.fromisoformat(meta["created_at"])
            age_days = (current_time - created_at).days

            if age_days > max_age_days:
                if self.delete_index(collection_name):
                    cleaned += 1
                    logger.info(f"Cleaned up old index: {collection_name} (age: {age_days} days)")

        return cleaned

    def _calculate_checksum(self, documents: List[Dict[str, Any]]) -> str:
        """Calculate checksum for documents."""
        # Create a hash of document contents
        hasher = hashlib.md5()
        for doc in documents:
            content = doc.get("content", "")
            hasher.update(content.encode("utf-8"))
        return hasher.hexdigest()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {str(e)}")
        return {}

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
