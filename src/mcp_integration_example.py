"""
MCP Tools Integration with Result Pattern using returns library.

This module demonstrates how to integrate the Result pattern
with MCP tools for consistent error handling across the application.
"""

import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from returns.result import Failure, Result, Success

from src.core.result_pattern import (
    AppError,
    ErrorKind,
    validation_error,
)

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("MDMAI-TTRPG")


def format_mcp_response(result: Result[Any, AppError]) -> Dict[str, Any]:
    """
    Format a Result for MCP tool response.
    
    This ensures consistent response format across all MCP tools:
    - Success cases include data and success flag
    - Failure cases include error details and recovery hints
    
    Args:
        result: Result to format
    
    Returns:
        Dictionary suitable for MCP response
    """
    if isinstance(result, Success):
        return {
            "success": True,
            "data": result.unwrap(),
        }
    
    error = result.failure()
    response = {
        "success": False,
        "error": {
            "code": error.kind.value,
            "message": error.message,
            "details": error.details or {},
        },
    }
    
    # Add recovery hints based on error kind
    if error.kind == ErrorKind.VALIDATION:
        response["error"]["hint"] = "Please check the input parameters and try again."
    elif error.kind == ErrorKind.NOT_FOUND:
        response["error"]["hint"] = "The requested resource does not exist."
    elif error.kind == ErrorKind.DATABASE:
        response["error"]["hint"] = "There was a database issue. Please try again later."
        response["error"]["recoverable"] = error.recoverable
    elif error.kind == ErrorKind.PERMISSION:
        response["error"]["hint"] = "You don't have permission for this operation."
    elif error.kind == ErrorKind.RATE_LIMIT:
        response["error"]["hint"] = "Too many requests. Please wait before trying again."
    
    return response


@mcp.tool()
async def search_rulebook(
    query: str,
    rulebook: Optional[str] = None,
    source_type: Optional[str] = None,
    max_results: int = 5,
    use_hybrid: bool = True,
) -> Dict[str, Any]:
    """
    Search TTRPG rulebooks with Result pattern error handling.
    
    This tool demonstrates:
    - Input validation with specific error messages
    - Service integration with Result pattern
    - Consistent error response format
    
    Args:
        query: Search query string
        rulebook: Optional specific rulebook to search
        source_type: Optional source type filter
        max_results: Maximum number of results (1-20)
        use_hybrid: Use hybrid search combining vector and keyword
    
    Returns:
        MCP-formatted response with search results or error
    """
    # Validate inputs using Result pattern
    validation_result = validate_search_params(query, max_results)
    if isinstance(validation_result, Failure):
        return format_mcp_response(validation_result)
    
    # Perform search (assuming search_service exists)
    # In real implementation, this would call the actual search service
    try:
        # Simulated search logic
        results = {
            "query": query,
            "results": [
                {
                    "id": "1",
                    "title": "Player's Handbook",
                    "content": f"Result matching '{query}'",
                    "score": 0.95,
                },
            ],
            "total_results": 1,
            "search_type": "hybrid" if use_hybrid else "vector",
            "metadata": {
                "rulebook": rulebook,
                "source_type": source_type,
            },
        }
        
        return format_mcp_response(Success(results))
    except Exception as e:
        logger.error(f"Search failed: {e}")
        error = AppError(
            kind=ErrorKind.SYSTEM,
            message=f"Search operation failed: {str(e)}",
            details={"query": query},
            recoverable=True,
        )
        return format_mcp_response(Failure(error))


def validate_search_params(
    query: str, max_results: int
) -> Result[None, AppError]:
    """
    Validate search parameters.
    
    Args:
        query: Search query
        max_results: Maximum results
    
    Returns:
        Success(None) if valid, Failure(AppError) otherwise
    """
    if not query or not query.strip():
        return Failure(
            validation_error(
                "Search query cannot be empty",
                field="query",
            )
        )
    
    if len(query) < 2:
        return Failure(
            validation_error(
                "Search query must be at least 2 characters",
                field="query",
                min_length=2,
                actual_length=len(query),
            )
        )
    
    if max_results < 1 or max_results > 20:
        return Failure(
            validation_error(
                "max_results must be between 1 and 20",
                field="max_results",
                min=1,
                max=20,
                actual=max_results,
            )
        )
    
    return Success(None)


@mcp.tool()
async def create_campaign(
    name: str,
    system: str,
    description: Optional[str] = None,
    player_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a new campaign with comprehensive validation.
    
    This tool demonstrates:
    - Complex input validation
    - Database operation with Result pattern
    - Rich error responses with recovery hints
    
    Args:
        name: Campaign name (required, 1-100 characters)
        system: Game system (e.g., "D&D 5e", "Pathfinder")
        description: Optional campaign description
        player_count: Optional expected number of players
    
    Returns:
        MCP-formatted response with campaign data or error
    """
    # Validate additional parameters
    if player_count is not None and (player_count < 1 or player_count > 20):
        error = validation_error(
            "Player count must be between 1 and 20",
            field="player_count",
            min=1,
            max=20,
            actual=player_count,
        )
        return format_mcp_response(Failure(error))
    
    # In real implementation, this would use actual CampaignManager
    # For demonstration, we'll simulate the operation
    try:
        campaign_data = {
            "id": "campaign-123",
            "name": name,
            "system": system,
            "description": description,
            "player_count": player_count,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }
        
        response_data = {
            "message": f"Campaign '{name}' created successfully",
            "campaign": campaign_data,
        }
        
        return format_mcp_response(Success(response_data))
    except Exception as e:
        logger.error(f"Failed to create campaign: {e}")
        error = AppError(
            kind=ErrorKind.DATABASE,
            message=f"Failed to create campaign: {str(e)}",
            details={"name": name, "system": system},
            recoverable=True,
        )
        return format_mcp_response(Failure(error))


@mcp.tool()
async def get_campaign_data(
    campaign_id: str,
    include_characters: bool = False,
    include_sessions: bool = False,
    include_npcs: bool = False,
) -> Dict[str, Any]:
    """
    Retrieve campaign data with optional related information.
    
    This tool demonstrates:
    - Conditional data fetching
    - Graceful degradation with partial failures
    - Detailed error context
    
    Args:
        campaign_id: Unique campaign identifier
        include_characters: Include player characters
        include_sessions: Include session history
        include_npcs: Include NPCs
    
    Returns:
        MCP-formatted response with campaign data or error
    """
    if not campaign_id or not campaign_id.strip():
        error = validation_error(
            "Campaign ID cannot be empty",
            field="campaign_id",
        )
        return format_mcp_response(Failure(error))
    
    # Simulate fetching campaign data
    try:
        campaign_data = {
            "id": campaign_id,
            "name": "Example Campaign",
            "system": "D&D 5e",
            "description": "An epic adventure",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-15T00:00:00Z",
        }
        
        # Add optional related data
        if include_characters:
            campaign_data["characters"] = [
                {"id": "char-1", "name": "Aragorn", "class": "Ranger"},
                {"id": "char-2", "name": "Gandalf", "class": "Wizard"},
            ]
        
        if include_sessions:
            campaign_data["sessions"] = [
                {"id": "session-1", "date": "2024-01-05", "title": "The Beginning"},
                {"id": "session-2", "date": "2024-01-12", "title": "Into the Mines"},
            ]
        
        if include_npcs:
            campaign_data["npcs"] = [
                {"id": "npc-1", "name": "Elrond", "role": "Quest Giver"},
                {"id": "npc-2", "name": "Sauron", "role": "Main Antagonist"},
            ]
        
        return format_mcp_response(Success(campaign_data))
    except Exception as e:
        logger.error(f"Failed to fetch campaign data: {e}")
        error = AppError(
            kind=ErrorKind.DATABASE,
            message=f"Failed to retrieve campaign data: {str(e)}",
            details={"campaign_id": campaign_id},
            recoverable=True,
        )
        return format_mcp_response(Failure(error))


@mcp.tool()
async def add_character_to_campaign(
    campaign_id: str,
    character_name: str,
    character_class: str,
    level: int = 1,
    background: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Add a character to an existing campaign.
    
    This tool demonstrates:
    - Nested validation logic
    - Transaction-like operations
    - Detailed success responses
    
    Args:
        campaign_id: Campaign to add character to
        character_name: Character's name
        character_class: Character's class
        level: Character level (1-20)
        background: Optional character background
        attributes: Optional additional attributes
    
    Returns:
        MCP-formatted response with character data or error
    """
    # Validate inputs
    validation_errors = []
    
    if not campaign_id:
        validation_errors.append("Campaign ID is required")
    
    if not character_name or not character_name.strip():
        validation_errors.append("Character name cannot be empty")
    
    if not character_class:
        validation_errors.append("Character class is required")
    
    if level < 1 or level > 20:
        validation_errors.append("Character level must be between 1 and 20")
    
    if validation_errors:
        error = validation_error(
            "Multiple validation errors occurred",
            errors=validation_errors,
        )
        return format_mcp_response(Failure(error))
    
    try:
        # Simulate adding character
        character_data = {
            "id": "char-new-123",
            "campaign_id": campaign_id,
            "name": character_name,
            "class": character_class,
            "level": level,
            "background": background,
            "attributes": attributes or {},
            "created_at": "2024-01-20T00:00:00Z",
        }
        
        response_data = {
            "message": f"Character '{character_name}' added to campaign",
            "character": character_data,
        }
        
        return format_mcp_response(Success(response_data))
    except Exception as e:
        logger.error(f"Failed to add character: {e}")
        error = AppError(
            kind=ErrorKind.DATABASE,
            message=f"Failed to add character: {str(e)}",
            details={
                "campaign_id": campaign_id,
                "character_name": character_name,
            },
            recoverable=True,
        )
        return format_mcp_response(Failure(error))


@mcp.tool()
async def batch_operation_example(
    operations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Example of batch operations with Result pattern.
    
    This demonstrates how to handle multiple operations where
    some might succeed and others fail, providing detailed
    feedback for each operation.
    
    Args:
        operations: List of operations to perform
    
    Returns:
        MCP-formatted response with results for each operation
    """
    if not operations:
        error = validation_error("No operations provided", field="operations")
        return format_mcp_response(Failure(error))
    
    results = []
    successes = 0
    failures = 0
    
    for i, operation in enumerate(operations):
        try:
            # Validate operation
            if "type" not in operation:
                result = {
                    "index": i,
                    "success": False,
                    "error": "Operation type is required",
                }
                failures += 1
            else:
                # Simulate processing operation
                result = {
                    "index": i,
                    "success": True,
                    "data": f"Processed {operation['type']}",
                }
                successes += 1
            
            results.append(result)
        except Exception as e:
            result = {
                "index": i,
                "success": False,
                "error": str(e),
            }
            failures += 1
            results.append(result)
    
    response_data = {
        "total": len(operations),
        "successes": successes,
        "failures": failures,
        "results": results,
    }
    
    # If all operations failed, return as failure
    if failures == len(operations):
        error = AppError(
            kind=ErrorKind.PROCESSING,
            message="All batch operations failed",
            details=response_data,
            recoverable=True,
        )
        return format_mcp_response(Failure(error))
    
    # Otherwise, return as success with mixed results
    return format_mcp_response(Success(response_data))


# Health check tool demonstrating simple Result usage
@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """
    Simple health check tool using Result pattern.
    
    Returns:
        MCP-formatted response with health status
    """
    try:
        health_data = {
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "database": "connected",
                "search": "ready",
                "mcp": "active",
            },
        }
        return format_mcp_response(Success(health_data))
    except Exception as e:
        error = AppError(
            kind=ErrorKind.SYSTEM,
            message=f"Health check failed: {str(e)}",
            recoverable=False,
        )
        return format_mcp_response(Failure(error))


if __name__ == "__main__":
    # Example of running the MCP server
    import asyncio
    
    async def main():
        """Run the MCP server."""
        logger.info("Starting MDMAI TTRPG MCP server with Result pattern...")
        await mcp.run()
    
    asyncio.run(main())