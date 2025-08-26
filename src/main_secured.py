"""Enhanced main entry point with security integration for TTRPG Assistant MCP Server."""

import asyncio
import atexit
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from config.logging_config import get_logger, setup_logging
from config.settings import settings
from src.campaign import initialize_campaign_tools, register_campaign_tools
from src.campaign.campaign_manager import CampaignManager
from src.campaign.rulebook_linker import RulebookLinker
from src.character_generation import initialize_character_tools, register_character_tools
from src.core.database import ChromaDBManager
from src.pdf_processing.pipeline import PDFProcessingPipeline
from src.performance import (
    GlobalCacheManager,
    initialize_performance_tools,
    register_performance_tools,
)
from src.performance.parallel_mcp_tools import register_parallel_tools
from src.personality.personality_manager import PersonalityManager
from src.personality.response_generator import ResponseGenerator
from src.search.search_service import SearchService
from src.security import (
    CampaignParameters,
    FilePathParameters,
    OperationType,
    Permission,
    ResourceType,
    SearchParameters,
    SecurityConfig,
    SecurityEventType,
    initialize_security,
    secure_mcp_tool,
)
from src.session import SessionManager, initialize_session_tools, register_session_tools
from src.source_management import initialize_source_tools, register_source_tools
from src.utils.user_interaction import register_user_interaction_tools

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

# Security manager (initialized in main())
security_manager = None


@mcp.tool()
@secure_mcp_tool(
    permission=Permission.SEARCH_BASIC,
    operation_type=OperationType.SEARCH_BASIC,
    audit_event=SecurityEventType.DATA_ACCESS,
)
async def search(
    query: str,
    rulebook: Optional[str] = None,
    source_type: Optional[str] = None,
    content_type: Optional[str] = None,
    max_results: int = 5,
    use_hybrid: bool = True,
    explain_results: bool = False,
    **kwargs,  # For security context
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
    global security_manager

    # Check database initialization
    if db is None:
        return {
            "status": "error",
            "error": "Database not initialized",
            "query": query,
            "results": [],
        }

    # Validate parameters using security module
    if security_manager:
        validation = security_manager.validate_search_params(
            query=query,
            rulebook=rulebook,
            source_type=source_type,
            content_type=content_type,
            max_results=max_results,
            use_hybrid=use_hybrid,
            explain_results=explain_results,
        )
        if not validation.is_valid:
            return {
                "status": "error",
                "error": "Invalid parameters",
                "validation_errors": validation.errors,
                "query": query,
                "results": [],
            }

        # Use validated parameters
        params = validation.value or {}
        query = params.get("query", query)
        max_results = params.get("max_results", max_results)

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
@secure_mcp_tool(
    permission=Permission.SOURCE_ADD,
    operation_type=OperationType.SOURCE_ADD,
    audit_event=SecurityEventType.SOURCE_ADDED,
)
async def add_source(
    pdf_path: str,
    rulebook_name: str,
    system: str,
    source_type: str = "rulebook",
    user_confirmed_large_file: bool = False,
    **kwargs,  # For security context
) -> Dict[str, Any]:
    """
    Add a new PDF source to the knowledge base.

    Args:
        pdf_path: Path to the PDF file
        rulebook_name: Name of the rulebook
        system: Game system (e.g., "D&D 5e", "Pathfinder")
        source_type: Type of source ('rulebook' or 'flavor')
        user_confirmed_large_file: Whether user has confirmed processing a large file

    Returns:
        Dictionary with status and source ID, or confirmation request for large files
    """
    global security_manager

    # Check database initialization
    if db is None:
        return {
            "status": "error",
            "error": "Database not initialized",
        }

    # Validate file path using security module
    if security_manager:
        path_validation = security_manager.validate_file_path(
            pdf_path,
            must_exist=True,
            allowed_extensions=["pdf"],
        )
        if not path_validation.is_valid:
            return {
                "status": "error",
                "error": "Invalid file path",
                "validation_errors": path_validation.errors,
            }

        # Use validated path
        pdf_path = path_validation.value

        # Validate campaign parameters
        campaign_validation = security_manager.validate_campaign_params(
            name=rulebook_name,
            system=system,
            description="",
            setting="",
        )
        if not campaign_validation.is_valid:
            return {
                "status": "error",
                "error": "Invalid parameters",
                "validation_errors": campaign_validation.errors,
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
            user_confirmed=user_confirmed_large_file,
        )

        # Check if confirmation is required
        if result.get("requires_confirmation"):
            return {
                "status": "confirmation_required",
                "message": result.get("message"),
                "confirmation_message": result.get("confirmation_message"),
                "file_info": result.get("file_info"),
                "instruction": "Please confirm with the user and call this tool again with user_confirmed_large_file=True",
            }

        if result.get("status") == "success" or result.get("success"):
            # Log successful source addition
            if security_manager:
                security_manager.audit_trail.log_event(
                    SecurityEventType.SOURCE_ADDED,
                    SecuritySeverity.INFO,
                    f"Source '{rulebook_name}' added successfully",
                    resource_id=result["source_id"],
                    resource_type="source",
                    details={
                        "system": system,
                        "source_type": source_type,
                        "chunks": result["total_chunks"],
                        "pages": result["total_pages"],
                    },
                )

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
@secure_mcp_tool(
    permission=Permission.SOURCE_READ,
    operation_type=OperationType.SOURCE_READ,
)
async def list_sources(
    system: Optional[str] = None,
    source_type: Optional[str] = None,
    **kwargs,  # For security context
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


@mcp.tool()
@secure_mcp_tool(
    permission=Permission.SEARCH_ANALYTICS,
    operation_type=OperationType.SEARCH_ANALYTICS,
)
async def search_analytics(**kwargs) -> Dict[str, Any]:
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
@secure_mcp_tool(
    permission=Permission.CACHE_CLEAR,
    operation_type=OperationType.CACHE_CLEAR,
    audit_event=SecurityEventType.CACHE_CLEARED,
)
async def clear_search_cache(**kwargs) -> Dict[str, str]:
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
@secure_mcp_tool(
    permission=Permission.SYSTEM_CONFIG,
    operation_type=OperationType.INDEX_UPDATE,
    audit_event=SecurityEventType.INDEX_UPDATED,
)
async def update_search_indices(**kwargs) -> Dict[str, str]:
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
@secure_mcp_tool(
    permission=Permission.SYSTEM_MONITOR,
    operation_type=None,  # No rate limit for info
)
async def server_info(**kwargs) -> Dict[str, Any]:
    """
    Get information about the TTRPG Assistant server.

    Returns:
        Server information and statistics
    """
    global security_manager

    try:
        stats = {}
        if db is not None:
            for collection_name in db.collections.keys():
                stats[collection_name] = db.get_collection_stats(collection_name)

        server_data = {
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

        # Add security status if available
        if security_manager:
            server_data["security"] = {
                "authentication_enabled": security_manager.config.enable_authentication,
                "rate_limiting_enabled": security_manager.config.enable_rate_limiting,
                "audit_enabled": security_manager.config.enable_audit,
                "input_validation_enabled": security_manager.config.enable_input_validation,
            }

        return server_data

    except Exception as e:
        logger.error("Failed to get server info", error=str(e))
        return {
            "name": settings.app_name,
            "version": "0.1.0",
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
@secure_mcp_tool(
    permission=Permission.PERSONALITY_CREATE,
    operation_type=OperationType.PERSONALITY_CREATE,
)
async def create_personality_profile(
    name: str,
    system: str,
    source_text: Optional[str] = None,
    base_profile: Optional[str] = None,
    **kwargs,
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
    global security_manager

    # Validate inputs if security manager available
    if security_manager:
        name_validation = security_manager.validate_input(name, "general", max_length=200)
        if not name_validation.is_valid:
            return {
                "status": "error",
                "error": "Invalid profile name",
                "validation_errors": name_validation.errors,
            }

        system_validation = security_manager.validate_input(
            system, "general", max_length=100
        )
        if not system_validation.is_valid:
            return {
                "status": "error",
                "error": "Invalid system name",
                "validation_errors": system_validation.errors,
            }

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
async def security_status(**kwargs) -> Dict[str, Any]:
    """
    Get security status and report.

    Returns:
        Security status and recent events
    """
    global security_manager

    if not security_manager:
        return {
            "status": "error",
            "error": "Security manager not initialized",
        }

    try:
        # Generate security report
        report = security_manager.get_security_report()

        # Get recent security events
        recent_events = security_manager.audit_trail.get_recent_events(limit=10)

        return {
            "status": "success",
            "security_enabled": {
                "authentication": security_manager.config.enable_authentication,
                "rate_limiting": security_manager.config.enable_rate_limiting,
                "audit": security_manager.config.enable_audit,
                "input_validation": security_manager.config.enable_input_validation,
            },
            "report": {
                "total_events": report.total_events,
                "events_by_type": report.events_by_type,
                "events_by_severity": report.events_by_severity,
                "suspicious_activities": len(report.suspicious_activities),
                "security_violations": len(report.security_violations),
                "recommendations": report.recommendations,
            },
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.event_type.value if hasattr(event.event_type, 'value') else event.event_type,
                    "severity": event.severity.value if hasattr(event.severity, 'value') else event.severity,
                    "message": event.message,
                    "user_id": event.user_id,
                    "ip_address": event.ip_address,
                }
                for event in recent_events
            ],
        }

    except Exception as e:
        logger.error("Failed to get security status", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


def main():
    """Main entry point for the MCP server with security integration."""
    global db, cache_manager, security_manager, campaign_manager, rulebook_linker, session_manager

    try:
        # Create necessary directories
        settings.create_directories()

        # Initialize security manager
        security_config = SecurityConfig(
            enable_authentication=getattr(settings, "enable_authentication", False),
            enable_rate_limiting=getattr(settings, "enable_rate_limiting", True),
            enable_audit=getattr(settings, "enable_audit", True),
            enable_input_validation=getattr(settings, "enable_input_validation", True),
            session_timeout_minutes=getattr(settings, "session_timeout_minutes", 60),
            audit_retention_days=getattr(settings, "audit_retention_days", 90),
            allowed_directories=[
                Path(settings.chroma_db_path),
                Path(settings.cache_dir),
                Path("/tmp"),
            ],
        )
        security_manager = initialize_security(security_config)

        logger.info(
            "Security manager initialized",
            auth=security_config.enable_authentication,
            rate_limiting=security_config.enable_rate_limiting,
            audit=security_config.enable_audit,
        )

        # Initialize database manager
        db = ChromaDBManager()

        # Initialize performance/cache management system
        cache_manager = GlobalCacheManager()
        initialize_performance_tools(
            cache_manager,
            cache_manager.invalidator,
            cache_manager.config,
            db,  # Pass database for optimizer and monitor
        )

        # Register cleanup handler
        def cleanup_resources():
            # Cleanup security manager
            if security_manager:
                try:
                    security_manager.perform_security_maintenance()
                except Exception as e:
                    logger.error("Error during security cleanup", error=str(e))

            # Cleanup cache manager
            if cache_manager:
                try:
                    cache_manager.shutdown()
                except Exception:
                    pass  # Already logged

            # Cleanup database optimizer and monitor
            if db and hasattr(db, "cleanup"):
                try:
                    asyncio.run(db.cleanup())
                except Exception as e:
                    logger.error("Error during database cleanup on exit", error=str(e))

        atexit.register(cleanup_resources)

        # Schedule periodic security maintenance
        async def security_maintenance_task():
            """Periodic security maintenance task."""
            while True:
                await asyncio.sleep(3600)  # Run every hour
                try:
                    counts = security_manager.perform_security_maintenance()
                    logger.info("Security maintenance completed", counts=counts)
                except Exception as e:
                    logger.error("Security maintenance failed", error=str(e))

        # Start maintenance task in background
        asyncio.create_task(security_maintenance_task())

        # Register performance tools with MCP server
        register_performance_tools(mcp)

        # Register parallel processing tools
        # initialize_parallel_tools()
        register_parallel_tools(mcp)

        # Initialize campaign management system
        campaign_manager = CampaignManager(db)
        rulebook_linker = RulebookLinker(db)
        initialize_campaign_tools(db, campaign_manager, rulebook_linker)

        # Register enhanced campaign tools with MCP server
        register_campaign_tools(mcp)

        # Initialize session management system
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

        # Register user interaction tools
        register_user_interaction_tools(mcp)

        logger.info(
            "Starting TTRPG Assistant MCP Server with Security",
            version="0.1.0",
            stdio_mode=settings.mcp_stdio_mode,
            security_features={
                "authentication": security_config.enable_authentication,
                "rate_limiting": security_config.enable_rate_limiting,
                "audit": security_config.enable_audit,
                "input_validation": security_config.enable_input_validation,
            },
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
        if security_manager:
            security_manager.perform_security_maintenance()
        if cache_manager:
            cache_manager.shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error("Server failed to start", error=str(e))
        if cache_manager:
            cache_manager.shutdown()
        sys.exit(1)
    finally:
        # Ensure cleanup
        if security_manager:
            try:
                security_manager.perform_security_maintenance()
            except Exception:
                pass
        if cache_manager:
            try:
                cache_manager.shutdown()
            except Exception:
                pass  # Already logged in shutdown method


if __name__ == "__main__":
    main()