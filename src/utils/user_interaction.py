"""User interaction utilities for MCP tools."""

from typing import Dict, Any, Optional
import asyncio

from config.logging_config import get_logger

logger = get_logger(__name__)


class UserInteraction:
    """Handles user interaction through MCP tools."""
    
    @staticmethod
    async def confirm_large_file_processing(
        file_name: str,
        file_size: str,
        estimated_time: str,
        warning_level: str = "warning"
    ) -> Dict[str, Any]:
        """
        MCP tool response for large file confirmation.
        
        This returns a structured response that the LLM can use
        to ask the user for confirmation.
        
        Args:
            file_name: Name of the file
            file_size: Formatted file size
            estimated_time: Estimated processing time
            warning_level: Warning level (info, warning, critical)
            
        Returns:
            Response for MCP tool
        """
        icon = "ℹ️" if warning_level == "info" else "⚠️" if warning_level == "warning" else "🔴"
        
        response = {
            "requires_confirmation": True,
            "confirmation_type": "large_file_processing",
            "file_info": {
                "name": file_name,
                "size": file_size,
                "estimated_time": estimated_time,
                "warning_level": warning_level
            },
            "message": f"{icon} The file '{file_name}' is {file_size}. Processing will take approximately {estimated_time}.",
            "prompt": "Would you like to proceed with processing this file?",
            "options": ["yes", "no"],
            "recommendations": []
        }
        
        if warning_level == "critical":
            response["recommendations"] = [
                "Consider splitting the file into smaller parts",
                "Process during off-peak hours",
                "Ensure sufficient system resources are available"
            ]
        elif warning_level == "warning":
            response["recommendations"] = [
                "Processing may take significant time",
                "System performance may be impacted"
            ]
        
        return response


def register_user_interaction_tools(mcp_server):
    """
    Register user interaction tools with the MCP server.
    
    Args:
        mcp_server: The FastMCP server instance
    """
    
    @mcp_server.tool()
    async def confirm_large_file(
        file_path: str,
        proceed_with_processing: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Handle large file confirmation flow.
        
        If proceed_with_processing is None, returns information for user confirmation.
        If proceed_with_processing is True/False, records the user's decision.
        
        Args:
            file_path: Path to the file being processed
            proceed_with_processing: User's decision (None for initial check)
            
        Returns:
            Confirmation request or processing decision
        """
        from pathlib import Path
        from src.utils.file_size_handler import FileSizeHandler
        
        try:
            path = Path(file_path)
            file_info = FileSizeHandler.get_file_info(path)
            
            if proceed_with_processing is None:
                # Initial check - return info for confirmation
                if file_info["requires_confirmation"]:
                    return await UserInteraction.confirm_large_file_processing(
                        file_name=file_info["name"],
                        file_size=file_info["size_formatted"],
                        estimated_time=file_info["estimated_time"],
                        warning_level=file_info["warning_level"]
                    )
                else:
                    return {
                        "requires_confirmation": False,
                        "message": f"File '{file_info['name']}' ({file_info['size_formatted']}) is within normal limits.",
                        "can_proceed": True
                    }
            else:
                # User has made a decision
                if proceed_with_processing:
                    logger.info(f"User approved processing of {file_info['name']} ({file_info['size_formatted']})")
                    return {
                        "success": True,
                        "message": f"Processing approved for '{file_info['name']}'",
                        "processing_recommendations": FileSizeHandler.get_processing_recommendations(file_info)
                    }
                else:
                    logger.info(f"User declined processing of {file_info['name']} ({file_info['size_formatted']})")
                    return {
                        "success": False,
                        "message": f"Processing cancelled for '{file_info['name']}'",
                        "reason": "User declined due to file size"
                    }
                    
        except FileNotFoundError as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "FileNotFoundError"
            }
        except Exception as e:
            logger.error(f"Error in confirm_large_file: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to check file: {str(e)}",
                "error_type": type(e).__name__
            }
    
    @mcp_server.tool()
    async def get_file_processing_status(
        file_path: str
    ) -> Dict[str, Any]:
        """
        Get detailed processing recommendations for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processing status and recommendations
        """
        from pathlib import Path
        from src.utils.file_size_handler import FileSizeHandler
        
        try:
            path = Path(file_path)
            file_info = FileSizeHandler.get_file_info(path)
            recommendations = FileSizeHandler.get_processing_recommendations(file_info)
            
            return {
                "success": True,
                "file_info": file_info,
                "processing_recommendations": recommendations,
                "message": f"File analysis complete for '{file_info['name']}'"
            }
            
        except Exception as e:
            logger.error(f"Error getting file status: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }