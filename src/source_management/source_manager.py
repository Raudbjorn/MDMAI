"""Main source management system coordinating all components."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .flavor_integrator import FlavorIntegrator
from .models import (
    ContentCategory,
    FlavorSource,
    ProcessingStatus,
    Source,
    SourceMetadata,
    SourceType,
)
from .source_organizer import SourceOrganizer
from .source_validator import SourceValidator

logger = logging.getLogger(__name__)

# Quality level ordering for comparisons
QUALITY_LEVEL_ORDER = {"excellent": 4, "good": 3, "fair": 2, "poor": 1, "unvalidated": 0}


class SourceManager:
    """Manage sources throughout their lifecycle."""

    def __init__(self, db_manager, pdf_pipeline=None):
        """
        Initialize the source manager.

        Args:
            db_manager: Database manager instance
            pdf_pipeline: Optional PDF processing pipeline
        """
        self.db = db_manager
        self.pdf_pipeline = pdf_pipeline

        # Initialize components
        self.validator = SourceValidator()
        self.organizer = SourceOrganizer()
        self.flavor_integrator = FlavorIntegrator()

        # Cache for quick access
        self.source_cache = {}
        self.metadata_cache = {}

        logger.info("Source manager initialized")

    async def add_source(
        self,
        file_path: str,
        title: str,
        system: str,
        source_type: str = "rulebook",
        metadata: Optional[Dict[str, Any]] = None,
        auto_process: bool = True,
    ) -> Dict[str, Any]:
        """
        Add a new source with comprehensive validation and processing.

        Args:
            file_path: Path to the source file
            title: Title of the source
            system: Game system
            source_type: Type of source
            metadata: Additional metadata
            auto_process: Whether to auto-process the source

        Returns:
            Result dictionary with source info
        """
        try:
            # Validate file
            file_validation = self.validator.validate_source_file(file_path)
            if not file_validation["valid"]:
                return {
                    "success": False,
                    "errors": file_validation["errors"],
                    "warnings": file_validation["warnings"],
                }

            # Check for duplicates
            file_hash = file_validation["file_info"]["hash"]
            existing = await self._check_duplicate(file_hash)
            if existing:
                return {
                    "success": False,
                    "error": "Duplicate source detected",
                    "existing_source": existing,
                }

            # Create source object
            source = Source()

            # Validate and convert source type
            try:
                source_type_enum = SourceType(source_type.lower())
            except ValueError:
                # Provide helpful error message with valid options
                valid_types = [st.value for st in SourceType]
                return {
                    "success": False,
                    "error": f"Invalid source type '{source_type}'. Valid types are: {', '.join(valid_types)}",
                }

            source.metadata = SourceMetadata(
                title=title,
                system=system,
                source_type=source_type_enum,
                file_path=file_path,
                file_hash=file_hash,
                file_size=file_validation["file_info"]["size_mb"],
            )

            # Add additional metadata
            if metadata:
                for key, value in metadata.items():
                    if hasattr(source.metadata, key):
                        setattr(source.metadata, key, value)

            # Validate metadata
            is_valid, errors, warnings = self.validator.validate_metadata(source.metadata)
            if not is_valid:
                return {"success": False, "errors": errors, "warnings": warnings}

            # Process if requested
            if auto_process and self.pdf_pipeline:
                source.status = ProcessingStatus.PROCESSING
                processing_result = await self._process_source(source)

                if not processing_result["success"]:
                    source.status = ProcessingStatus.FAILED
                    source.errors = processing_result.get("errors", [])
                else:
                    source.status = ProcessingStatus.COMPLETED
                    source.chunk_count = processing_result.get("chunk_count", 0)
                    source.content_chunks = processing_result.get("chunks", [])

            # Validate content quality
            source.quality = self.validator.validate_content(source)

            # Auto-categorize
            categories = self.organizer.categorize_source(source)
            source.categories = categories

            # Process as flavor if appropriate
            if source.metadata.source_type in [
                SourceType.FLAVOR,
                SourceType.ADVENTURE,
                SourceType.SETTING,
            ]:
                flavor_source = self.flavor_integrator.process_flavor_source(source)
                source = flavor_source

            # Store in database
            await self._store_source(source)

            # Update cache
            self.source_cache[source.id] = source
            self.metadata_cache[source.id] = source.metadata

            logger.info(f"Source added successfully: {source.id} - {title}")

            return {
                "success": True,
                "source_id": source.id,
                "title": title,
                "quality": source.quality.level.value,
                "categories": [cat.name for cat in categories],
                "chunk_count": source.chunk_count,
                "warnings": warnings,
            }

        except Exception as e:
            logger.error(f"Failed to add source: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_source(self, source_id: str, include_content: bool = False) -> Optional[Source]:
        """
        Get a source by ID.

        Args:
            source_id: Source ID
            include_content: Whether to include content chunks

        Returns:
            Source object or None
        """
        # Check cache first
        if source_id in self.source_cache:
            source = self.source_cache[source_id]
            if not include_content:
                # Return without content chunks for efficiency
                source_copy = Source.from_dict(source.to_dict())
                source_copy.content_chunks = []
                return source_copy
            return source

        # Load from database
        try:
            result = await self.db.get_document(collection_name="sources", document_id=source_id)

            if result:
                metadata = result.get("metadata", {})
                source_data = json.loads(metadata.get("source_data", "{}"))
                source = Source.from_dict(source_data)

                # Update access tracking
                source.last_accessed = datetime.utcnow()
                source.access_count += 1

                # Cache it
                self.source_cache[source_id] = source

                return source

        except Exception as e:
            logger.error(f"Failed to get source {source_id}: {str(e)}")

        return None

    async def list_sources(
        self,
        system: Optional[str] = None,
        source_type: Optional[str] = None,
        category: Optional[str] = None,
        quality_min: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List sources with advanced filtering.

        Args:
            system: Filter by game system
            source_type: Filter by source type
            category: Filter by category
            quality_min: Minimum quality level
            limit: Maximum results

        Returns:
            List of source summaries
        """
        try:
            # Build query
            where_clause = {}
            if system:
                where_clause["system"] = system
            if source_type:
                where_clause["source_type"] = source_type

            # Query database
            results = await self.db.query_collection(
                collection_name="sources", where=where_clause, limit=limit
            )

            sources = []
            for result in results.get("documents", []):
                metadata = result.get("metadata", {})
                source_data = json.loads(metadata.get("source_data", "{}"))

                # Apply additional filters
                if category:
                    categories = source_data.get("categories", [])
                    if not any(cat.get("name") == category for cat in categories):
                        continue

                if quality_min:
                    quality = source_data.get("quality", {})
                    quality_level = quality.get("level", "unvalidated")
                    if QUALITY_LEVEL_ORDER.get(quality_level, 0) < QUALITY_LEVEL_ORDER.get(
                        quality_min, 0
                    ):
                        continue

                sources.append(
                    {
                        "id": source_data.get("id"),
                        "title": source_data.get("metadata", {}).get("title"),
                        "system": source_data.get("metadata", {}).get("system"),
                        "type": source_data.get("metadata", {}).get("source_type"),
                        "quality": source_data.get("quality", {}).get("level"),
                        "categories": [
                            cat.get("name") for cat in source_data.get("categories", [])
                        ],
                        "chunk_count": source_data.get("chunk_count", 0),
                        "created_at": source_data.get("created_at"),
                    }
                )

            return sources

        except Exception as e:
            logger.error(f"Failed to list sources: {str(e)}")
            return []

    async def update_source(self, source_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update source metadata or properties.

        Args:
            source_id: Source ID
            updates: Updates to apply

        Returns:
            Result dictionary
        """
        try:
            source = await self.get_source(source_id, include_content=True)
            if not source:
                return {"success": False, "error": "Source not found"}

            # Apply updates
            for key, value in updates.items():
                if key == "metadata":
                    # Update metadata fields
                    for meta_key, meta_value in value.items():
                        if hasattr(source.metadata, meta_key):
                            setattr(source.metadata, meta_key, meta_value)
                elif key == "categories":
                    # Re-categorize
                    source.categories = self.organizer.categorize_source(
                        source, auto_categorize=value
                    )
                elif hasattr(source, key):
                    setattr(source, key, value)

            # Update timestamp
            source.updated_at = datetime.utcnow()

            # Re-validate if metadata changed
            if "metadata" in updates:
                is_valid, errors, warnings = self.validator.validate_metadata(source.metadata)
                if not is_valid:
                    return {"success": False, "errors": errors, "warnings": warnings}

            # Store updates
            await self._store_source(source)

            # Update cache
            self.source_cache[source_id] = source

            return {
                "success": True,
                "message": "Source updated successfully",
                "source_id": source_id,
            }

        except Exception as e:
            logger.error(f"Failed to update source: {str(e)}")
            return {"success": False, "error": str(e)}

    async def delete_source(self, source_id: str, remove_chunks: bool = True) -> Dict[str, Any]:
        """
        Delete a source.

        Args:
            source_id: Source ID
            remove_chunks: Whether to remove content chunks

        Returns:
            Result dictionary
        """
        try:
            # Remove from database
            if remove_chunks:
                # Remove all chunks
                await self.db.delete_documents(
                    collection_name="rulebooks", where={"source_id": source_id}
                )
                await self.db.delete_documents(
                    collection_name="flavor_sources", where={"source_id": source_id}
                )

            # Remove source record
            await self.db.delete_document(collection_name="sources", document_id=source_id)

            # Clear from cache
            if source_id in self.source_cache:
                del self.source_cache[source_id]
            if source_id in self.metadata_cache:
                del self.metadata_cache[source_id]

            logger.info(f"Source deleted: {source_id}")

            return {
                "success": True,
                "message": "Source deleted successfully",
                "source_id": source_id,
            }

        except Exception as e:
            logger.error(f"Failed to delete source: {str(e)}")
            return {"success": False, "error": str(e)}

    async def detect_relationships(
        self, source_id: str, rescan: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect relationships for a source.

        Args:
            source_id: Source ID
            rescan: Whether to rescan even if cached

        Returns:
            List of relationships
        """
        try:
            source = await self.get_source(source_id)
            if not source:
                return []

            # Check cache unless rescanning
            if not rescan and source.relationships:
                return [rel.to_dict() for rel in source.relationships]

            # Efficiently get all sources in a single query
            existing_sources = await self._get_all_sources_bulk(exclude_id=source_id)

            # Detect relationships
            relationships = self.organizer.detect_relationships(source, existing_sources)
            source.relationships = relationships

            # Store updated source
            await self._store_source(source)

            return [rel.to_dict() for rel in relationships]

        except Exception as e:
            logger.error(f"Failed to detect relationships: {str(e)}")
            return []

    async def get_flavor_sources(
        self, system: Optional[str] = None, canonical_only: bool = False
    ) -> List[FlavorSource]:
        """
        Get available flavor sources.

        Args:
            system: Filter by system
            canonical_only: Only return canonical sources

        Returns:
            List of flavor sources
        """
        try:
            where_clause = {"source_type": "flavor"}
            if system:
                where_clause["system"] = system

            results = await self.db.query_collection(
                collection_name="sources", where=where_clause, limit=100
            )

            flavor_sources = []
            for result in results.get("documents", []):
                metadata = result.get("metadata", {})
                source_data = json.loads(metadata.get("source_data", "{}"))

                # Convert to FlavorSource
                flavor = FlavorSource.from_dict(source_data)

                if canonical_only and not flavor.canonical:
                    continue

                flavor_sources.append(flavor)

            return flavor_sources

        except Exception as e:
            logger.error(f"Failed to get flavor sources: {str(e)}")
            return []

    async def blend_flavor_sources(
        self, source_ids: List[str], weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Blend multiple flavor sources.

        Args:
            source_ids: IDs of sources to blend
            weights: Optional weights

        Returns:
            Blended flavor data
        """
        try:
            sources = []
            for sid in source_ids:
                source = await self.get_source(sid)
                if source and isinstance(source, FlavorSource):
                    sources.append(source)

            if not sources:
                return {"success": False, "error": "No valid flavor sources found"}

            blended = self.flavor_integrator.blend_flavor_sources(sources, weights)

            return {"success": True, "blended_data": blended, "source_count": len(sources)}

        except Exception as e:
            logger.error(f"Failed to blend sources: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _get_all_sources_bulk(
        self, exclude_id: Optional[str] = None, limit: int = 1000
    ) -> List[Source]:
        """
        Efficiently retrieve all sources in bulk.

        Args:
            exclude_id: Optional ID to exclude
            limit: Maximum number of sources

        Returns:
            List of Source objects
        """
        try:
            # Get all source documents in a single query
            results = await self.db.query_collection(collection_name="sources", limit=limit)

            sources = []
            for result in results.get("documents", []):
                metadata = result.get("metadata", {})
                source_data = json.loads(metadata.get("source_data", "{}"))

                # Skip if this is the excluded ID
                if exclude_id and source_data.get("id") == exclude_id:
                    continue

                # Create Source object from data
                source = Source.from_dict(source_data)
                sources.append(source)

            return sources

        except Exception as e:
            logger.error(f"Failed to bulk retrieve sources: {str(e)}")
            return []

    async def _check_duplicate(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Check if a source with this hash already exists."""
        try:
            results = await self.db.query_collection(
                collection_name="sources", where={"file_hash": file_hash}, limit=1
            )

            if results.get("documents"):
                doc = results["documents"][0]
                metadata = doc.get("metadata", {})
                return {
                    "source_id": metadata.get("source_id"),
                    "title": metadata.get("title"),
                    "system": metadata.get("system"),
                }

        except Exception as e:
            logger.error(f"Duplicate check failed: {str(e)}")

        return None

    async def _process_source(self, source: Source) -> Dict[str, Any]:
        """Process source through PDF pipeline."""
        if not self.pdf_pipeline:
            return {"success": False, "error": "PDF pipeline not available"}

        try:
            result = await self.pdf_pipeline.process_pdf(
                pdf_path=source.metadata.file_path,
                rulebook_name=source.metadata.title,
                system=source.metadata.system,
                source_type=source.metadata.source_type.value,
            )

            return {
                "success": result.get("status") == "success",
                "chunk_count": result.get("total_chunks", 0),
                "chunks": result.get("chunks", []),
                "errors": result.get("errors", []),
            }

        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _store_source(self, source: Source):
        """Store source in database."""
        try:
            source_dict = source.to_dict()

            await self.db.add_document(
                collection_name="sources",
                document_id=source.id,
                content=f"Source: {source.metadata.title} ({source.metadata.system})",
                metadata={
                    "source_id": source.id,
                    "title": source.metadata.title,
                    "system": source.metadata.system,
                    "source_type": source.metadata.source_type.value,
                    "file_hash": source.metadata.file_hash,
                    "quality_level": source.quality.level.value,
                    "source_data": json.dumps(source_dict),
                },
            )

        except Exception as e:
            logger.error(f"Failed to store source: {str(e)}")
            raise
