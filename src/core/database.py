"""ChromaDB database integration for TTRPG Assistant."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ChromaDBManager:
    """Manages ChromaDB collections and operations."""
    
    def __init__(self):
        """Initialize ChromaDB client and collections."""
        self.client = None
        self.embedding_function = None
        self.collections = {}
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client with persistent storage."""
        try:
            # Configure ChromaDB settings
            chroma_settings = ChromaSettings(
                chroma_db_impl=settings.chroma_db_impl,
                persist_directory=str(settings.chroma_db_path),
                anonymized_telemetry=False,
            )
            
            # Create client
            self.client = chromadb.Client(chroma_settings)
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
            
            # Initialize collections
            self._initialize_collections()
            
            logger.info(
                "ChromaDB initialized",
                db_path=str(settings.chroma_db_path),
                embedding_model=settings.embedding_model,
            )
            
        except Exception as e:
            logger.error("Failed to initialize ChromaDB", error=str(e))
            raise
    
    def _initialize_collections(self) -> None:
        """Initialize or get existing collections."""
        collection_names = [
            "rulebooks",
            "flavor_sources",
            "campaigns",
            "sessions",
            "personalities",
        ]
        
        for name in collection_names:
            collection_name = f"{settings.chroma_collection_prefix}{name}"
            try:
                # Try to get existing collection
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                )
                logger.debug(f"Retrieved existing collection: {collection_name}")
            except ValueError:
                # Create new collection if it doesn't exist
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": f"Collection for {name}"},
                )
                logger.info(f"Created new collection: {collection_name}")
            
            self.collections[name] = collection
    
    def add_document(
        self,
        collection_name: str,
        document_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> None:
        """
        Add a document to a collection.
        
        Args:
            collection_name: Name of the collection
            document_id: Unique identifier for the document
            content: Document content
            metadata: Document metadata
            embedding: Optional pre-computed embedding
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            if embedding:
                collection.add(
                    ids=[document_id],
                    documents=[content],
                    metadatas=[metadata],
                    embeddings=[embedding],
                )
            else:
                collection.add(
                    ids=[document_id],
                    documents=[content],
                    metadatas=[metadata],
                )
            
            logger.debug(
                "Document added",
                collection=collection_name,
                document_id=document_id,
            )
            
        except Exception as e:
            logger.error(
                "Failed to add document",
                collection=collection_name,
                document_id=document_id,
                error=str(e),
            )
            raise
    
    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Search query
            n_results: Number of results to return
            metadata_filter: Optional metadata filter
        
        Returns:
            List of search results with content and metadata
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            # Build where clause from metadata filter
            where_clause = metadata_filter if metadata_filter else None
            
            # Perform search
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
            )
            
            # Format results
            formatted_results = []
            if results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None,
                    })
            
            logger.debug(
                "Search completed",
                collection=collection_name,
                query=query[:50],
                n_results=len(formatted_results),
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(
                "Search failed",
                collection=collection_name,
                query=query[:50],
                error=str(e),
            )
            raise
    
    def update_document(
        self,
        collection_name: str,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update a document in a collection.
        
        Args:
            collection_name: Name of the collection
            document_id: Document identifier
            content: Updated content (optional)
            metadata: Updated metadata (optional)
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            update_args = {"ids": [document_id]}
            
            if content is not None:
                update_args["documents"] = [content]
            
            if metadata is not None:
                update_args["metadatas"] = [metadata]
            
            collection.update(**update_args)
            
            logger.debug(
                "Document updated",
                collection=collection_name,
                document_id=document_id,
            )
            
        except Exception as e:
            logger.error(
                "Failed to update document",
                collection=collection_name,
                document_id=document_id,
                error=str(e),
            )
            raise
    
    def delete_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> None:
        """
        Delete a document from a collection.
        
        Args:
            collection_name: Name of the collection
            document_id: Document identifier
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            collection.delete(ids=[document_id])
            
            logger.debug(
                "Document deleted",
                collection=collection_name,
                document_id=document_id,
            )
            
        except Exception as e:
            logger.error(
                "Failed to delete document",
                collection=collection_name,
                document_id=document_id,
                error=str(e),
            )
            raise
    
    def get_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        
        Args:
            collection_name: Name of the collection
            document_id: Document identifier
        
        Returns:
            Document data or None if not found
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            result = collection.get(ids=[document_id])
            
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "content": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
            
            return None
            
        except Exception as e:
            logger.error(
                "Failed to get document",
                collection=collection_name,
                document_id=document_id,
                error=str(e),
            )
            raise
    
    def list_documents(
        self,
        collection_name: str,
        limit: int = 100,
        offset: int = 0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List documents in a collection.
        
        Args:
            collection_name: Name of the collection
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            metadata_filter: Optional metadata filter
        
        Returns:
            List of documents
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            where_clause = metadata_filter if metadata_filter else None
            
            result = collection.get(
                limit=limit,
                offset=offset,
                where=where_clause,
            )
            
            documents = []
            for i in range(len(result["ids"])):
                documents.append({
                    "id": result["ids"][i],
                    "content": result["documents"][i] if result["documents"] else None,
                    "metadata": result["metadatas"][i] if result["metadatas"] else None,
                })
            
            return documents
            
        except Exception as e:
            logger.error(
                "Failed to list documents",
                collection=collection_name,
                error=str(e),
            )
            raise
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            Collection statistics
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection = self.collections[collection_name]
        
        try:
            # Get collection count
            count = collection.count()
            
            return {
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata,
            }
            
        except Exception as e:
            logger.error(
                "Failed to get collection stats",
                collection=collection_name,
                error=str(e),
            )
            raise
    
    def persist(self) -> None:
        """Persist the database to disk."""
        try:
            if hasattr(self.client, "persist"):
                self.client.persist()
                logger.debug("Database persisted to disk")
        except Exception as e:
            logger.error("Failed to persist database", error=str(e))
            raise