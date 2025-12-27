"""Main entry point for TTRPG Assistant MCP Server."""

import asyncio
import atexit
import sys
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
    FilePathParameters,
    OperationType,
    Permission,
    ResourceType,
    SearchParameters,
    SecurityConfig,
    SecurityEventType,
    SecurityManager,
    initialize_security,
    secure_mcp_tool,
)
from src.session import SessionManager, initialize_session_tools, register_session_tools
from src.source_management import initialize_source_tools, register_source_tools
from src.utils.user_interaction import register_user_interaction_tools

from src.voice_synthesis import (
    VoiceManager,
    VoiceProviderConfig,
    VoiceProviderType,
    initialize_voice_tools,
    register_voice_tools,
)

# Set up logging
setup_logging(level=settings.log_level, log_file=settings.log_file)
logger = get_logger(__name__)

# Initialize FastMCP server
mcp = FastMCP("TTRPG Assistant")

# Database manager will be initialized in main()
db: Optional[ChromaDBManager] = None

# Initialize PDF processing pipeline
# Disable interactive prompts in non-interactive environments (tests, CI/CD)
is_interactive = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False
pdf_pipeline = PDFProcessingPipeline(prompt_for_ollama=is_interactive)

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

# Security management components (initialized in main())
security_manager: Optional[SecurityManager] = None

# Voice management components (initialized in main())
voice_manager: Optional[VoiceManager] = None


@mcp.tool()
@secure_mcp_tool(
    permission=Permission.SEARCH_BASIC,
    operation_type=OperationType.SEARCH_BASIC,
    validate_params={"query": SearchParameters},
    resource_type=ResourceType.SOURCE,
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
@secure_mcp_tool(
    permission=Permission.SOURCE_ADD,
    operation_type=OperationType.SOURCE_ADD,
    validate_params={"pdf_path": FilePathParameters},
    resource_type=ResourceType.SOURCE,
    audit_event=SecurityEventType.CAMPAIGN_CREATED,
)
async def add_source(
    pdf_path: str,
    rulebook_name: str,
    system: str,
    source_type: str = "rulebook",
    user_confirmed_large_file: bool = False,
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
    resource_type=ResourceType.SOURCE,
    audit_event=SecurityEventType.DATA_ACCESS,
)
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
@secure_mcp_tool(
    permission=Permission.SEARCH_ANALYTICS,
    operation_type=OperationType.SEARCH_ANALYTICS,
    resource_type=ResourceType.SEARCH,
    audit_event=SecurityEventType.DATA_ACCESS,
)
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
@secure_mcp_tool(
    permission=Permission.CACHE_CLEAR,
    operation_type=OperationType.CAMPAIGN_WRITE,
    resource_type=ResourceType.CACHE,
    audit_event=SecurityEventType.SECURITY_CONFIG_CHANGED,
)
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
@secure_mcp_tool(
    permission=Permission.SYSTEM_ADMIN,
    operation_type=OperationType.CAMPAIGN_WRITE,
    resource_type=ResourceType.SYSTEM,
    audit_event=SecurityEventType.SECURITY_CONFIG_CHANGED,
)
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
@secure_mcp_tool(
    permission=Permission.SYSTEM_MONITOR,
    operation_type=OperationType.CAMPAIGN_READ,
    resource_type=ResourceType.SYSTEM,
    audit_event=SecurityEventType.DATA_ACCESS,
)
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
async def roll_dice(notation: str) -> Dict[str, Any]:
    """
    Roll dice using standard RPG notation (e.g., '3d6+2', '1d20', '2d10+5').

    Args:
        notation: Dice expression like '3d6+2' or '1d20'

    Returns:
        Dictionary with total, individual rolls, and breakdown
    """
    import random
    import re

    pattern = r"(\d+)d(\d+)([+-]\d+)?"
    match = re.match(pattern, notation.strip())

    if not match:
        return {
            "error": "Invalid dice notation. Use format like '3d6+2' or '1d20'",
            "notation": notation,
        }

    num_dice = int(match.group(1))
    num_sides = int(match.group(2))
    modifier = int(match.group(3) or 0)

    # Validate to prevent DoS
    if num_dice > 1000 or num_sides > 1000:
        return {"error": "Maximum 1000 dice or 1000 sides", "notation": notation}
    if num_dice < 1 or num_sides < 1:
        return {"error": "Dice count and sides must be at least 1", "notation": notation}

    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    total = sum(rolls) + modifier

    breakdown_parts = [str(r) for r in rolls]
    if modifier != 0:
        breakdown_parts.append(f"{modifier:+d}")
    breakdown = " + ".join(breakdown_parts).replace("+ -", "- ")

    return {
        "notation": notation,
        "total": total,
        "rolls": rolls,
        "modifier": modifier,
        "breakdown": f"{breakdown} = {total}",
        "dice": {"count": num_dice, "sides": num_sides},
    }


@mcp.tool()
@secure_mcp_tool(
    permission=Permission.PERSONALITY_CREATE,
    operation_type=OperationType.CHARACTER_GENERATE,
    resource_type=ResourceType.SYSTEM,
    audit_event=SecurityEventType.CAMPAIGN_CREATED,
)
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
@secure_mcp_tool(
    permission=Permission.SYSTEM_MONITOR,
    operation_type=OperationType.CAMPAIGN_READ,
    resource_type=ResourceType.SYSTEM,
    audit_event=SecurityEventType.DATA_ACCESS,
)
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
            profile_list.append(
                {
                    "profile_id": profile.profile_id,
                    "name": profile.name,
                    "system": profile.system,
                    "characteristics": profile.characteristics,
                    "tone": profile.tone.get("dominant", "neutral"),
                    "usage_count": profile.usage_count,
                    "is_default": profile.custom_traits.get("is_default", False),
                }
            )

        return {
            "status": "success",
            "profiles": profile_list,
            "total": len(profile_list),
            "active_profile": (
                personality_manager.active_profile.name
                if personality_manager.active_profile
                else None
            ),
        }

    except Exception as e:
        logger.error("Failed to list personality profiles", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
@secure_mcp_tool(
    permission=Permission.PERSONALITY_UPDATE,
    operation_type=OperationType.CHARACTER_UPDATE,
    resource_type=ResourceType.SYSTEM,
    audit_event=SecurityEventType.SECURITY_CONFIG_CHANGED,
)
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
@secure_mcp_tool(
    permission=Permission.SYSTEM_MONITOR,
    operation_type=OperationType.CAMPAIGN_READ,
    resource_type=ResourceType.PERSONALITY,
    audit_event=SecurityEventType.DATA_ACCESS,
)
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
            "profile_used": (
                profile.name
                if profile
                else (
                    personality_manager.active_profile.name
                    if personality_manager.active_profile
                    else "None"
                )
            ),
            "apply_to_search": apply_to_search,
        }

    except Exception as e:
        logger.error("Failed to apply personality", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
@secure_mcp_tool(
    permission=Permission.SYSTEM_MONITOR,
    operation_type=OperationType.CAMPAIGN_READ,
    resource_type=ResourceType.SYSTEM,
    audit_event=SecurityEventType.DATA_ACCESS,
)
async def security_status() -> Dict[str, Any]:
    """
    Get security status and statistics.

    Returns:
        Security status including active sessions, rate limits, and audit statistics
    """
    if security_manager is None:
        return {
            "status": "error",
            "error": "Security not initialized",
            "security_enabled": False,
        }

    try:
        # Get security report
        report = security_manager.get_security_report()

        # Get current rate limit status for common operations
        rate_limits = {}
        for op_type in [OperationType.SEARCH_BASIC, OperationType.SOURCE_ADD, OperationType.CHARACTER_UPDATE]:
            status = security_manager.rate_limiter.check_rate_limit("default", op_type, consume=False)
            rate_limits[op_type.value] = {
                "allowed": status.allowed,
                "remaining": status.remaining,
                "reset_in": status.reset_in,
            }

        # Get session statistics
        session_stats = {
            "active_sessions": len(security_manager.access_control.sessions),
            "active_users": len([u for u in security_manager.access_control.users.values() if u.is_active]),
        }

        return {
            "status": "success",
            "security_enabled": True,
            "configuration": {
                "authentication_enabled": security_manager.config.enable_authentication,
                "rate_limiting_enabled": security_manager.config.enable_rate_limiting,
                "audit_enabled": security_manager.config.enable_audit,
                "input_validation_enabled": security_manager.config.enable_input_validation,
            },
            "sessions": session_stats,
            "rate_limits": rate_limits,
            "audit_summary": {
                "total_events": report.total_events,
                "security_events": report.security_events,
                "failed_auth_attempts": report.failed_auth_attempts,
                "permission_violations": report.permission_violations,
                "injection_attempts": report.injection_attempts,
                "rate_limit_violations": report.rate_limit_violations,
            },
            "last_maintenance": datetime.now().isoformat() if security_manager else None,
        }
    except Exception as e:
        logger.error("Failed to get security status", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "security_enabled": security_manager is not None,
        }


@mcp.tool()
@secure_mcp_tool(
    permission=Permission.SYSTEM_ADMIN,
    operation_type=OperationType.CAMPAIGN_WRITE,
    resource_type=ResourceType.SYSTEM,
    audit_event=SecurityEventType.SECURITY_CONFIG_CHANGED,
)
async def security_maintenance() -> Dict[str, Any]:
    """
    Perform security maintenance tasks.

    Returns:
        Results of maintenance operations
    """
    if security_manager is None:
        return {
            "status": "error",
            "error": "Security not initialized",
        }

    try:
        results = security_manager.perform_security_maintenance()

        return {
            "status": "success",
            "maintenance_results": results,
            "timestamp": datetime.now().isoformat(),
            "message": "Security maintenance completed successfully",
        }
    except Exception as e:
        logger.error("Failed to perform security maintenance", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


def main():
    """Main entry point for the MCP server."""
    global db, cache_manager, security_manager

    try:
        # Create necessary directories
        settings.create_directories()

        # Initialize database manager
        db = ChromaDBManager()

        # Initialize security management system
        # Security can be configured via settings or environment variables
        security_config = SecurityConfig(
            enable_authentication=settings.enable_authentication,
            enable_rate_limiting=settings.enable_rate_limiting,
            enable_audit=settings.enable_audit,
            enable_input_validation=settings.enable_input_validation,
            session_timeout_minutes=settings.session_timeout_minutes,
            audit_retention_days=settings.audit_retention_days,
            allowed_directories=[
                Path(settings.chroma_db_path),
                Path(settings.cache_dir),
                Path("/tmp"),
            ],
        )
        security_manager = initialize_security(security_config)

        logger.info(
            "Security system initialized",
            authentication=security_config.enable_authentication,
            rate_limiting=security_config.enable_rate_limiting,
            audit=security_config.enable_audit,
            input_validation=security_config.enable_input_validation,
        )

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
                    # Perform final security maintenance
                    security_manager.perform_security_maintenance()
                    logger.info("Security cleanup completed")
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

        # Register performance tools with MCP server
        register_performance_tools(mcp)

        # Register parallel processing tools
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

        # Register user interaction tools
        register_user_interaction_tools(mcp)

        # Initialize voice synthesis system
        global voice_manager

        # Configure voice providers
        voice_configs = []

        # ElevenLabs configuration
        if settings.elevenlabs_api_key:
            voice_configs.append(VoiceProviderConfig(
                provider_type=VoiceProviderType.ELEVENLABS,
                api_key=settings.elevenlabs_api_key,
                priority=10 if settings.voice_provider.lower() == "elevenlabs" else 20
            ))

        # Fish Audio configuration
        if settings.fish_audio_api_key:
            voice_configs.append(VoiceProviderConfig(
                provider_type=VoiceProviderType.FISH_AUDIO,
                api_key=settings.fish_audio_api_key,
                priority=10 if settings.voice_provider.lower() == "fish_audio" else 20
            ))

        # Ollama TTS configuration
        voice_configs.append(VoiceProviderConfig(
            provider_type=VoiceProviderType.OLLAMA_TTS,
            base_url=settings.ollama_tts_url,
            priority=10 if settings.voice_provider.lower() == "ollama_tts" else 30
        ))

        # Initialize manager
        voice_manager = VoiceManager(
            provider_configs=voice_configs,
            cache_dir=settings.cache_dir / "audio",
            prefer_local=True
        )

        # Needs to be awaited, but main is sync appearing wrapper around async logic?
        # Actually main() calls mcp.run() which handles async.
        # FastMCP doesn't have a clear startup hook for async initialization in this structure.
        # We'll initialize lazily in the tools or need a startup event.
        # Looking at existing code, database initialization happens effectively synchronously relative to tools or passed in.
        # But VoiceManager.initialize() is async.
        # We will wrap it in a startup event if FastMCP supports it, or initialize in the first tool call.
        # Existing code uses @mcp.on_startup? No, FastMCP usually has lifespan.

        # Let's check how other async managers are initialized.
        # SearchService seems to be initialized synchronously in constructor?
        # Database uses Chroma which might be sync or async.

        # We can use startup context if available, or just initialize first time.
        # VoiceManager handles initialized state check, so we can call it in tools.

        initialize_voice_tools(voice_manager)
        register_voice_tools(mcp)

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
        if security_manager:
            try:
                security_manager.perform_security_maintenance()
            except Exception:
                pass
        if cache_manager:
            cache_manager.shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error("Server failed to start", error=str(e))
        if security_manager:
            try:
                security_manager.perform_security_maintenance()
            except Exception:
                pass
        if cache_manager:
            cache_manager.shutdown()
        sys.exit(1)
    finally:
        # Ensure security and cache managers are shut down
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
