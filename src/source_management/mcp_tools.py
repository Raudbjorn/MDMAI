"""MCP tools for source management."""

import logging
from typing import Any, Dict, List, Optional

from .models import ContentCategory, QualityLevel, SourceType
from .source_manager import SourceManager

logger = logging.getLogger(__name__)

# Module-level instances
_source_manager: Optional[SourceManager] = None


def initialize_source_tools(db_manager, pdf_pipeline=None):
    """
    Initialize source management tools.

    Args:
        db_manager: Database manager instance
        pdf_pipeline: Optional PDF processing pipeline
    """
    global _source_manager

    _source_manager = SourceManager(db_manager, pdf_pipeline)
    logger.info("Source management tools initialized")


def register_source_tools(mcp_server):
    """
    Register enhanced source management tools with the MCP server.

    Args:
        mcp_server: The FastMCP server instance
    """

    @mcp_server.tool()
    async def add_enhanced_source(
        pdf_path: str,
        title: str,
        system: str,
        source_type: str = "rulebook",
        author: Optional[str] = None,
        publisher: Optional[str] = None,
        publication_date: Optional[str] = None,
        edition: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        auto_process: bool = True,
    ) -> Dict[str, Any]:
        """
        Add a source with comprehensive validation and metadata.

        Args:
            pdf_path: Path to the PDF file
            title: Title of the source
            system: Game system (e.g., "D&D 5e")
            source_type: Type of source (rulebook, flavor, supplement, etc.)
            author: Author(s) of the source
            publisher: Publisher name
            publication_date: Publication date
            edition: Edition or version
            description: Description of the source
            tags: Tags for categorization
            auto_process: Whether to automatically process the PDF
            validate_quality: Whether to validate content quality

        Returns:
            Result with source ID, quality assessment, and categories
        """
        try:
            if not _source_manager:
                return {"success": False, "error": "Source manager not initialized"}

            # Prepare metadata
            metadata = {
                "author": author,
                "publisher": publisher,
                "publication_date": publication_date,
                "edition": edition,
                "description": description,
                "tags": tags or [],
            }

            # Add source
            result = await _source_manager.add_source(
                file_path=pdf_path,
                title=title,
                system=system,
                source_type=source_type,
                metadata=metadata,
                auto_process=auto_process,
            )

            return result

        except Exception as e:
            logger.error(f"Failed to add enhanced source: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp_server.tool()
    async def list_enhanced_sources(
        system: Optional[str] = None,
        source_type: Optional[str] = None,
        category: Optional[str] = None,
        quality_min: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        List sources with advanced filtering options.

        Args:
            system: Filter by game system
            source_type: Filter by source type
            category: Filter by content category
            quality_min: Minimum quality level (excellent, good, fair)
            tags: Filter by tags
            limit: Maximum number of results

        Returns:
            List of sources with metadata and quality information
        """
        try:
            if not _source_manager:
                return {"success": False, "error": "Source manager not initialized"}

            sources = await _source_manager.list_sources(
                system=system,
                source_type=source_type,
                category=category,
                quality_min=quality_min,
                limit=limit,
            )

            # Filter by tags if specified
            if tags:
                sources = [s for s in sources if any(tag in s.get("tags", []) for tag in tags)]

            return {"success": True, "sources": sources, "total": len(sources)}

        except Exception as e:
            logger.error(f"Failed to list sources: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp_server.tool()
    async def get_source_details(
        source_id: str, include_content: bool = False, include_relationships: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed information about a source.

        Args:
            source_id: ID of the source
            include_content: Whether to include content chunks
            include_relationships: Whether to include relationships

        Returns:
            Detailed source information
        """
        try:
            if not _source_manager:
                return {"success": False, "error": "Source manager not initialized"}

            source = await _source_manager.get_source(source_id, include_content=include_content)

            if not source:
                return {"success": False, "error": "Source not found"}

            result = {"success": True, "source": source.to_dict()}

            if include_relationships:
                relationships = await _source_manager.detect_relationships(source_id)
                result["relationships"] = relationships

            return result

        except Exception as e:
            logger.error(f"Failed to get source details: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp_server.tool()
    async def update_source_metadata(
        source_id: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        publisher: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        recategorize: bool = False,
    ) -> Dict[str, Any]:
        """
        Update source metadata and optionally recategorize.

        Args:
            source_id: ID of the source to update
            title: New title
            author: New author
            publisher: New publisher
            description: New description
            tags: New tags
            recategorize: Whether to recategorize the source

        Returns:
            Update result
        """
        try:
            if not _source_manager:
                return {"success": False, "error": "Source manager not initialized"}

            updates = {"metadata": {}}

            if title:
                updates["metadata"]["title"] = title
            if author:
                updates["metadata"]["author"] = author
            if publisher:
                updates["metadata"]["publisher"] = publisher
            if description:
                updates["metadata"]["description"] = description
            if tags:
                updates["metadata"]["tags"] = tags

            if recategorize:
                updates["categories"] = True

            result = await _source_manager.update_source(source_id, updates)

            return result

        except Exception as e:
            logger.error(f"Failed to update source: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp_server.tool()
    async def delete_source(source_id: str, remove_content: bool = True) -> Dict[str, Any]:
        """
        Delete a source from the system.

        Args:
            source_id: ID of the source to delete
            remove_content: Whether to remove content chunks

        Returns:
            Deletion result
        """
        try:
            if not _source_manager:
                return {"success": False, "error": "Source manager not initialized"}

            result = await _source_manager.delete_source(source_id, remove_chunks=remove_content)

            return result

        except Exception as e:
            logger.error(f"Failed to delete source: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp_server.tool()
    async def detect_source_relationships(source_id: str, rescan: bool = False) -> Dict[str, Any]:
        """
        Detect relationships between sources.

        Args:
            source_id: ID of the source to analyze
            rescan: Whether to force rescan even if cached

        Returns:
            Detected relationships
        """
        try:
            if not _source_manager:
                return {"success": False, "error": "Source manager not initialized"}

            relationships = await _source_manager.detect_relationships(source_id, rescan=rescan)

            return {"success": True, "relationships": relationships, "total": len(relationships)}

        except Exception as e:
            logger.error(f"Failed to detect relationships: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp_server.tool()
    async def get_flavor_sources(
        system: Optional[str] = None, canonical_only: bool = False
    ) -> Dict[str, Any]:
        """
        Get available flavor sources for narrative generation.

        Args:
            system: Filter by game system
            canonical_only: Only return canonical sources

        Returns:
            List of flavor sources with narrative elements
        """
        try:
            if not _source_manager:
                return {"success": False, "error": "Source manager not initialized"}

            flavor_sources = await _source_manager.get_flavor_sources(
                system=system, canonical_only=canonical_only
            )

            return {
                "success": True,
                "flavor_sources": [fs.to_dict() for fs in flavor_sources],
                "total": len(flavor_sources),
            }

        except Exception as e:
            logger.error(f"Failed to get flavor sources: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp_server.tool()
    async def blend_flavor_sources(
        source_ids: List[str],
        weights: Optional[Dict[str, float]] = None,
        conflict_resolution: str = "priority",
    ) -> Dict[str, Any]:
        """
        Blend multiple flavor sources for narrative generation.

        Args:
            source_ids: IDs of sources to blend
            weights: Optional weights for each source
            conflict_resolution: How to resolve conflicts (priority, blend, random)

        Returns:
            Blended narrative elements
        """
        try:
            if not _source_manager:
                return {"success": False, "error": "Source manager not initialized"}

            result = await _source_manager.blend_flavor_sources(source_ids, weights)

            return result

        except Exception as e:
            logger.error(f"Failed to blend flavor sources: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp_server.tool()
    async def organize_sources_by_system() -> Dict[str, Any]:
        """
        Organize all sources by game system.

        Returns:
            Sources organized by system
        """
        try:
            if not _source_manager:
                return {"success": False, "error": "Source manager not initialized"}

            # Get all sources
            all_sources = await _source_manager.list_sources(limit=1000)

            # Organize by system
            organized = {}
            for source in all_sources:
                system = source.get("system", "Unknown")
                if system not in organized:
                    organized[system] = []
                organized[system].append(source)

            return {
                "success": True,
                "systems": organized,
                "total_systems": len(organized),
                "total_sources": len(all_sources),
            }

        except Exception as e:
            logger.error(f"Failed to organize sources: {str(e)}")
            return {"success": False, "error": str(e)}

    @mcp_server.tool()
    async def validate_source_quality(source_id: str) -> Dict[str, Any]:
        """
        Validate and assess the quality of a source.

        Args:
            source_id: ID of the source to validate

        Returns:
            Quality assessment with issues and recommendations
        """
        try:
            if not _source_manager:
                return {"success": False, "error": "Source manager not initialized"}

            source = await _source_manager.get_source(source_id, include_content=True)

            if not source:
                return {"success": False, "error": "Source not found"}

            # Revalidate quality
            from .source_validator import SourceValidator

            validator = SourceValidator()
            quality = validator.validate_content(source)

            # Update source quality
            source.quality = quality
            await _source_manager.update_source(source_id, {"quality": quality})

            return {
                "success": True,
                "quality_level": quality.level.value,
                "overall_score": quality.overall_score,
                "text_quality": quality.text_quality,
                "structure_quality": quality.structure_quality,
                "metadata_completeness": quality.metadata_completeness,
                "content_coverage": quality.content_coverage,
                "issues": quality.issues,
                "warnings": quality.warnings,
                "statistics": {
                    "total_pages": quality.total_pages,
                    "extracted_pages": quality.extracted_pages,
                    "total_chunks": quality.total_chunks,
                    "valid_chunks": quality.valid_chunks,
                    "avg_chunk_length": quality.avg_chunk_length,
                },
            }

        except Exception as e:
            logger.error(f"Failed to validate source quality: {str(e)}")
            return {"success": False, "error": str(e)}

    logger.info("Enhanced source management tools registered with MCP server")
