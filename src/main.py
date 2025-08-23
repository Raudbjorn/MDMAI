"""Main entry point for TTRPG Assistant MCP Server."""

import atexit
import asyncio
import json
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from config.logging_config import setup_logging, get_logger
from config.settings import settings
from src.core.database import ChromaDBManager
from src.pdf_processing.pipeline import PDFProcessingPipeline
from src.search.search_service import SearchService
from src.personality.personality_manager import PersonalityManager
from src.personality.response_generator import ResponseGenerator
from src.campaign.campaign_manager import CampaignManager
from src.campaign.rulebook_linker import RulebookLinker
from src.campaign import (
    initialize_campaign_tools,
    register_campaign_tools
)
from src.session import (
    SessionManager,
    initialize_session_tools,
    register_session_tools
)
from src.character_generation import (
    initialize_character_tools,
    register_character_tools
)
from src.source_management import (
    initialize_source_tools,
    register_source_tools
)
from src.performance import (
    GlobalCacheManager,
    initialize_performance_tools,
    register_performance_tools
)
from src.performance.parallel_mcp_tools import register_parallel_tools

# Set up logging
setup_logging(level=settings.log_level, log_file=settings.log_file)
logger = get_logger(__name__)

# Initialize FastMCP server
mcp = FastMCP("TTRPG Assistant")

# Database manager will be initialized in main()
db: Optional[ChromaDBManager] = None

# Initialize PDF processing pipeline
pdf_pipeline = PDFProcessingPipeline()

# Initialize search service
search_service = SearchService()

# Initialize personality system
personality_manager = PersonalityManager()
response_generator = ResponseGenerator(personality_manager)

# Campaign management components (initialized in main())
campaign_manager: Optional[CampaignManager] = None
rulebook_linker: Optional[RulebookLinker] = None

# Session management components (initialized in main())
session_manager: Optional[SessionManager] = None

# Performance management components (initialized in main())
cache_manager: Optional[GlobalCacheManager] = None


@mcp.tool()
async def search(
    query: str,
    rulebook: Optional[str] = None,
    source_type: Optional[str] = None,
    content_type: Optional[str] = None,
    max_results: int = 5,
    use_hybrid: bool = True,
    explain_results: bool = False,
) -> Dict[str, Any]:
    """
    Search across TTRPG content with semantic and keyword matching.
    
    Args:
        query: Search query string
        rulebook: Optional specific rulebook to search
        source_type: Filter by source type ('rulebook' or 'flavor')
        content_type: Filter by content type ('rule', 'spell', 'monster', etc.)
        max_results: Maximum number of results to return
        use_hybrid: Whether to use hybrid search (semantic + keyword)
        explain_results: Whether to include explanations for why results matched
    
    Returns:
        Dictionary containing search results with content, sources, and relevance scores
    """
    # Check database initialization
    if db is None:
        return {
            "status": "error",
            "error": "Database not initialized",
            "query": query,
            "results": [],
        }
    
    # Validate parameters
    if max_results < 1 or max_results > 100:
        return {
            "status": "error",
            "error": "max_results must be between 1 and 100",
            "query": query,
            "results": [],
        }
    
    try:
        logger.info(
            "Search request",
            query=query[:100],
            rulebook=rulebook,
            source_type=source_type,
            max_results=max_results,
        )
        
        # Use the search service
        result = await search_service.search(
            query=query,
            rulebook=rulebook,
            source_type=source_type,
            content_type=content_type,
            max_results=max_results,
            use_hybrid=use_hybrid,
            explain_results=explain_results,
        )
        
        return {
            "status": "success",
            "query": result["query"],
            "processed_query": result["processed_query"],
            "results": result["results"],
            "total_results": result["total_results"],
            "search_time": result["search_time"],
            "suggestions": result["suggestions"],
            "filters_applied": result["filters_applied"],
        }
        
    except Exception as e:
        logger.error("Search failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "query": query,
            "results": [],
        }


@mcp.tool()
async def add_source(
    pdf_path: str,
    rulebook_name: str,
    system: str,
    source_type: str = "rulebook",
) -> Dict[str, str]:
    """
    Add a new PDF source to the knowledge base.
    
    Args:
        pdf_path: Path to the PDF file
        rulebook_name: Name of the rulebook
        system: Game system (e.g., "D&D 5e", "Pathfinder")
        source_type: Type of source ('rulebook' or 'flavor')
    
    Returns:
        Dictionary with status and source ID
    """
    # Check database initialization
    if db is None:
        return {
            "status": "error",
            "error": "Database not initialized",
        }
    
    try:
        logger.info(
            "Adding source",
            pdf_path=pdf_path,
            rulebook_name=rulebook_name,
            system=system,
            source_type=source_type,
        )
        
        # Process PDF through the pipeline
        result = await pdf_pipeline.process_pdf(
            pdf_path=pdf_path,
            rulebook_name=rulebook_name,
            system=system,
            source_type=source_type,
            enable_adaptive_learning=settings.enable_adaptive_learning,
        )
        
        if result["status"] == "success":
            return {
                "status": "success",
                "message": f"Source '{rulebook_name}' added successfully",
                "source_id": result["source_id"],
                "chunks_created": result["total_chunks"],
                "pages_processed": result["total_pages"],
            }
        elif result["status"] == "duplicate":
            return {
                "status": "duplicate",
                "message": result["message"],
                "file_hash": result["file_hash"],
            }
        else:
            return {
                "status": "error",
                "error": result.get("error", "Unknown error"),
            }
        
    except Exception as e:
        logger.error("Failed to add source", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def list_sources(
    system: Optional[str] = None,
    source_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List available sources in the system.
    
    Args:
        system: Filter by game system
        source_type: Filter by source type ('rulebook' or 'flavor')
    
    Returns:
        List of available sources with metadata
    """
    # Check database initialization
    if db is None:
        logger.error("Database not initialized")
        return []
    
    try:
        logger.info("Listing sources", system=system, source_type=source_type)
        
        # Determine which collection to query
        collection_name = "flavor_sources" if source_type == "flavor" else "rulebooks"
        
        # Build metadata filter
        metadata_filter = {}
        if system:
            metadata_filter["system"] = system
        
        # Get unique sources from the collection (limit to reasonable number)
        documents = db.list_documents(
            collection_name=collection_name,
            limit=100,  # Limit to 100 to prevent memory issues
            metadata_filter=metadata_filter if metadata_filter else None,
        )
        
        # Extract unique sources
        sources = {}
        for doc in documents:
            if doc["metadata"]:
                source_id = doc["metadata"].get("source_id", "unknown")
                if source_id not in sources:
                    sources[source_id] = {
                        "source_id": source_id,
                        "name": doc["metadata"].get("rulebook_name", "Unknown"),
                        "system": doc["metadata"].get("system", "Unknown"),
                        "type": source_type or "rulebook",
                        "document_count": 1,
                    }
                else:
                    sources[source_id]["document_count"] += 1
        
        return list(sources.values())
        
    except Exception as e:
        logger.error("Failed to list sources", error=str(e))
        return []


# NOTE: Campaign tools are now provided by the enhanced campaign management system
# See src/campaign/mcp_tools.py for the implementations
# Features include:
# - Full CRUD operations with versioning
# - Campaign-rulebook linking
# - Character, NPC, Location, and Plot Point management
# - Version history and rollback capabilities


@mcp.tool()
async def search_analytics() -> Dict[str, Any]:
    """
    Get search analytics and statistics.
    
    Returns:
        Search analytics including popular queries and performance metrics
    """
    try:
        analytics = search_service.get_search_analytics()
        return {
            "status": "success",
            **analytics,
        }
    except Exception as e:
        logger.error("Failed to get search analytics", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def clear_search_cache() -> Dict[str, str]:
    """
    Clear the search result cache.
    
    Returns:
        Status message
    """
    try:
        search_service.clear_cache()
        return {
            "status": "success",
            "message": "Search cache cleared successfully",
        }
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def update_search_indices() -> Dict[str, str]:
    """
    Update search indices for all collections.
    
    Returns:
        Status message
    """
    try:
        search_service.update_indices()
        return {
            "status": "success",
            "message": "Search indices updated successfully",
        }
    except Exception as e:
        logger.error("Failed to update indices", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def server_info() -> Dict[str, Any]:
    """
    Get information about the TTRPG Assistant server.
    
    Returns:
        Server information and statistics
    """
    try:
        stats = {}
        if db is not None:
            for collection_name in db.collections.keys():
                stats[collection_name] = db.get_collection_stats(collection_name)
        
        return {
            "name": settings.app_name,
            "version": "0.1.0",
            "status": "running",
            "settings": {
                "embedding_model": settings.embedding_model,
                "hybrid_search": settings.enable_hybrid_search,
                "adaptive_learning": settings.enable_adaptive_learning,
            },
            "collections": stats,
        }
        
    except Exception as e:
        logger.error("Failed to get server info", error=str(e))
        return {
            "name": settings.app_name,
            "version": "0.1.0",
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def create_personality_profile(
    name: str,
    system: str,
    source_text: Optional[str] = None,
    base_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new personality profile for response generation.
    
    Args:
        name: Profile name
        system: Game system
        source_text: Optional text to extract personality from
        base_profile: Optional base profile name to build upon
    
    Returns:
        Dictionary with profile information
    """
    try:
        # Get base profile if specified
        base = None
        if base_profile:
            base = personality_manager.get_profile_by_name(base_profile, system)
            if base:
                base_profile = base.profile_id
        
        # Create profile
        profile = personality_manager.create_profile(
            name=name,
            system=system,
            source_text=source_text,
            base_profile=base_profile,
        )
        
        return {
            "status": "success",
            "profile_id": profile.profile_id,
            "name": profile.name,
            "system": profile.system,
            "characteristics": profile.characteristics,
            "tone": profile.tone.get("dominant", "neutral"),
            "message": f"Personality profile '{name}' created successfully",
        }
        
    except Exception as e:
        logger.error("Failed to create personality profile", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def list_personality_profiles(
    system: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List available personality profiles.
    
    Args:
        system: Optional game system filter
    
    Returns:
        List of personality profiles
    """
    try:
        profiles = personality_manager.list_profiles(system)
        
        profile_list = []
        for profile in profiles:
            profile_list.append({
                "profile_id": profile.profile_id,
                "name": profile.name,
                "system": profile.system,
                "characteristics": profile.characteristics,
                "tone": profile.tone.get("dominant", "neutral"),
                "usage_count": profile.usage_count,
                "is_default": profile.custom_traits.get("is_default", False),
            })
        
        return {
            "status": "success",
            "profiles": profile_list,
            "total": len(profile_list),
            "active_profile": personality_manager.active_profile.name if personality_manager.active_profile else None,
        }
        
    except Exception as e:
        logger.error("Failed to list personality profiles", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def set_active_personality(
    profile_name: str,
    system: Optional[str] = None,
) -> Dict[str, str]:
    """
    Set the active personality profile for responses.
    
    Args:
        profile_name: Name of the profile to activate
        system: Optional system to search within
    
    Returns:
        Status dictionary
    """
    try:
        # Find profile by name
        profile = personality_manager.get_profile_by_name(profile_name, system)
        
        if not profile:
            return {
                "status": "error",
                "error": f"Profile '{profile_name}' not found",
            }
        
        # Set as active
        success = personality_manager.set_active_profile(profile.profile_id)
        
        if success:
            return {
                "status": "success",
                "profile_id": profile.profile_id,
                "name": profile.name,
                "message": f"Active personality set to '{profile_name}'",
            }
        else:
            return {
                "status": "error",
                "error": "Failed to set active profile",
            }
        
    except Exception as e:
        logger.error("Failed to set active personality", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def apply_personality(
    content: str,
    profile_name: Optional[str] = None,
    apply_to_search: bool = False,
) -> Dict[str, Any]:
    """
    Apply personality to content or enable for search results.
    
    Args:
        content: Text content to transform
        profile_name: Optional specific profile to use (uses active if not specified)
        apply_to_search: Whether to apply personality to search results
    
    Returns:
        Transformed content or status
    """
    try:
        # Get profile
        profile = None
        if profile_name:
            profile = personality_manager.get_profile_by_name(profile_name)
            if not profile:
                return {
                    "status": "error",
                    "error": f"Profile '{profile_name}' not found",
                }
        
        # Apply personality to content
        transformed = response_generator.generate_response(content, profile)
        
        return {
            "status": "success",
            "original": content,
            "transformed": transformed,
            "profile_used": profile.name if profile else (
                personality_manager.active_profile.name if personality_manager.active_profile else "None"
            ),
            "apply_to_search": apply_to_search,
        }
        
    except Exception as e:
        logger.error("Failed to apply personality", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


def main():
    """Main entry point for the MCP server."""
    global db, cache_manager
    
    try:
        # Create necessary directories
        settings.create_directories()
        
        # Initialize database manager
        db = ChromaDBManager()
        
        # Initialize performance/cache management system
        cache_manager = GlobalCacheManager()
        initialize_performance_tools(
            cache_manager,
            cache_manager.invalidator,
            cache_manager.config,
            db  # Pass database for optimizer and monitor
        )
        
        # Register cleanup handler
        def cleanup_resources():
            # Cleanup cache manager
            if cache_manager:
                try:
                    cache_manager.shutdown()
                except Exception:
                    pass  # Already logged
            
            # Cleanup database optimizer and monitor
            if db and hasattr(db, 'cleanup'):
                try:
                    asyncio.run(db.cleanup())
                except Exception as e:
                    logger.error("Error during database cleanup on exit", error=str(e))
        
        atexit.register(cleanup_resources)
        
        # Register performance tools with MCP server
        register_performance_tools(mcp)
        
        # Register parallel processing tools
        initialize_parallel_tools()
        register_parallel_tools(mcp)
        
        # Initialize campaign management system
        global campaign_manager, rulebook_linker
        campaign_manager = CampaignManager(db)
        rulebook_linker = RulebookLinker(db)
        initialize_campaign_tools(db, campaign_manager, rulebook_linker)
        
        # Register enhanced campaign tools with MCP server
        register_campaign_tools(mcp)
        
        # Initialize session management system
        global session_manager
        session_manager = SessionManager(db)
        initialize_session_tools(session_manager, campaign_manager)
        
        # Register session tools with MCP server
        register_session_tools(mcp)
        
        # Initialize character generation system
        initialize_character_tools(db, personality_manager)
        
        # Register character generation tools with MCP server
        register_character_tools(mcp)
        
        # Initialize enhanced source management system
        initialize_source_tools(db, pdf_pipeline)
        
        # Register enhanced source management tools with MCP server
        register_source_tools(mcp)
        
        logger.info(
            "Starting TTRPG Assistant MCP Server",
            version="0.1.0",
            stdio_mode=settings.mcp_stdio_mode,
        )
        
        if settings.mcp_stdio_mode:
            # Run in stdio mode for MCP
            mcp.run(transport="stdio")
        else:
            # Run as HTTP server for testing
            import uvicorn
            uvicorn.run(
                mcp.get_app(),
                host="127.0.0.1",
                port=8000,
                log_level=settings.log_level.lower(),
            )
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        if cache_manager:
            cache_manager.shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error("Server failed to start", error=str(e))
        if cache_manager:
            cache_manager.shutdown()
        sys.exit(1)
    finally:
        # Ensure cache manager is shut down
        if cache_manager:
            try:
                cache_manager.shutdown()
            except Exception:
                pass  # Already logged in shutdown method


if __name__ == "__main__":
    main()