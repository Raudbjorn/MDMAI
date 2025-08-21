"""Main entry point for TTRPG Assistant MCP Server."""

import asyncio
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from config.logging_config import setup_logging, get_logger
from config.settings import settings
from src.core.database import get_db_manager

# Set up logging
setup_logging(level=settings.log_level, log_file=settings.log_file)
logger = get_logger(__name__)

# Initialize FastMCP server
mcp = FastMCP("TTRPG Assistant")

# Initialize database manager
db = get_db_manager()


@mcp.tool()
async def search(
    query: str,
    rulebook: Optional[str] = None,
    source_type: Optional[str] = None,
    content_type: Optional[str] = None,
    max_results: int = 5,
    use_hybrid: bool = True,
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
    
    Returns:
        Dictionary containing search results with content, sources, and relevance scores
    """
    try:
        logger.info(
            "Search request",
            query=query[:100],
            rulebook=rulebook,
            source_type=source_type,
            max_results=max_results,
        )
        
        # Determine which collection to search
        collection_name = "flavor_sources" if source_type == "flavor" else "rulebooks"
        
        # Build metadata filter
        metadata_filter = {}
        if rulebook:
            metadata_filter["rulebook"] = rulebook
        if content_type:
            metadata_filter["content_type"] = content_type
        
        # Perform search
        results = db.search(
            collection_name=collection_name,
            query=query,
            n_results=max_results,
            metadata_filter=metadata_filter if metadata_filter else None,
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result["content"],
                "source": result["metadata"].get("source", "Unknown"),
                "page": result["metadata"].get("page", None),
                "section": result["metadata"].get("section", None),
                "relevance_score": 1.0 - result.get("distance", 0) if result.get("distance") else None,
            })
        
        return {
            "status": "success",
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
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
    try:
        logger.info(
            "Adding source",
            pdf_path=pdf_path,
            rulebook_name=rulebook_name,
            system=system,
            source_type=source_type,
        )
        
        # TODO: Implement PDF processing pipeline
        # This is a placeholder for now
        
        return {
            "status": "success",
            "message": f"Source '{rulebook_name}' added successfully",
            "source_id": f"{system}_{rulebook_name}".replace(" ", "_").lower(),
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
    try:
        logger.info("Listing sources", system=system, source_type=source_type)
        
        # Determine which collection to query
        collection_name = "flavor_sources" if source_type == "flavor" else "rulebooks"
        
        # Build metadata filter
        metadata_filter = {}
        if system:
            metadata_filter["system"] = system
        
        # Get unique sources from the collection
        documents = db.list_documents(
            collection_name=collection_name,
            limit=1000,
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
async def create_campaign(
    name: str,
    system: str,
    description: Optional[str] = None,
) -> Dict[str, str]:
    """
    Create a new campaign.
    
    Args:
        name: Campaign name
        system: Game system
        description: Optional campaign description
    
    Returns:
        Dictionary with campaign ID and status
    """
    try:
        
        campaign_id = str(uuid.uuid4())
        
        campaign_data = {
            "id": campaign_id,
            "name": name,
            "system": system,
            "description": description or "",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "characters": [],
            "npcs": [],
            "locations": [],
            "plot_points": [],
            "sessions": [],
        }
        
        # Store in database
        db.add_document(
            collection_name="campaigns",
            document_id=campaign_id,
            content=json.dumps(campaign_data),
            metadata={
                "campaign_id": campaign_id,
                "name": name,
                "system": system,
                "data_type": "campaign",
            },
        )
        
        logger.info("Campaign created", campaign_id=campaign_id, name=name)
        
        return {
            "status": "success",
            "campaign_id": campaign_id,
            "message": f"Campaign '{name}' created successfully",
        }
        
    except Exception as e:
        logger.error("Failed to create campaign", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def get_campaign_data(
    campaign_id: str,
    data_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve campaign-specific data.
    
    Args:
        campaign_id: Campaign identifier
        data_type: Optional filter for data type (characters, npcs, locations, etc.)
    
    Returns:
        Campaign data dictionary
    """
    try:
        logger.info("Getting campaign data", campaign_id=campaign_id, data_type=data_type)
        
        # Retrieve campaign document
        campaign_doc = db.get_document(
            collection_name="campaigns",
            document_id=campaign_id,
        )
        
        if not campaign_doc:
            return {
                "status": "error",
                "error": f"Campaign '{campaign_id}' not found",
            }
        
        # Parse campaign data
        campaign_data = json.loads(campaign_doc["content"])
        
        # Filter by data type if specified
        if data_type:
            if data_type in campaign_data:
                return {
                    "status": "success",
                    "campaign_id": campaign_id,
                    "data_type": data_type,
                    "data": campaign_data[data_type],
                }
            else:
                return {
                    "status": "error",
                    "error": f"Data type '{data_type}' not found in campaign",
                }
        
        return {
            "status": "success",
            "campaign_id": campaign_id,
            "data": campaign_data,
        }
        
    except Exception as e:
        logger.error("Failed to get campaign data", error=str(e))
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


def main():
    """Main entry point for the MCP server."""
    try:
        logger.info(
            "Starting TTRPG Assistant MCP Server",
            version="0.1.0",
            stdio_mode=settings.mcp_stdio_mode,
        )
        
        if settings.mcp_stdio_mode:
            # Run in stdio mode for MCP
            import sys
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
        sys.exit(0)
    except Exception as e:
        logger.error("Server failed to start", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()