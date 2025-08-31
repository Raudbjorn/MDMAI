"""
User-friendly error message templates for MDMAI TTRPG Assistant.

This module provides templates and formatters for creating clear, actionable
error messages for MCP tool responses.
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorMessageCategory(Enum):
    """Categories for error message templates."""

    SYSTEM = "system"
    DATABASE = "database"
    NETWORK = "network"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMIT = "rate_limit"
    SERVICE = "service"
    MCP = "mcp"


class ErrorMessageTemplate:
    """Template for generating user-friendly error messages."""

    # Message templates by category
    TEMPLATES: Dict[ErrorMessageCategory, Dict[str, str]] = {
        ErrorMessageCategory.SYSTEM: {
            "default": "A system error occurred. Please try again later.",
            "maintenance": "The system is currently under maintenance. Please try again in {retry_after} minutes.",
            "overload": "The system is experiencing high load. Your request has been queued.",
            "critical": "A critical system error occurred. Our team has been notified.",
        },
        ErrorMessageCategory.DATABASE: {
            "default": "Unable to access game data. Please try again.",
            "connection": "Cannot connect to the game database. Please check your connection.",
            "timeout": "Database operation timed out. Please try again with fewer items.",
            "integrity": "Data integrity error detected. Please refresh and try again.",
            "migration": "Database is being updated. Some features may be temporarily unavailable.",
        },
        ErrorMessageCategory.NETWORK: {
            "default": "Network error occurred. Please check your connection.",
            "timeout": "Request timed out. Please try again.",
            "dns": "Cannot resolve server address. Please check your network settings.",
            "connection_refused": "Connection refused. The service may be temporarily unavailable.",
            "ssl": "Secure connection failed. Please update your client.",
        },
        ErrorMessageCategory.VALIDATION: {
            "default": "Invalid input provided. Please check your data.",
            "required_field": "Required field '{field}' is missing.",
            "invalid_format": "Invalid format for '{field}'. Expected: {expected_format}",
            "out_of_range": "Value for '{field}' is out of range. Must be between {min} and {max}.",
            "invalid_type": "Invalid type for '{field}'. Expected {expected_type}, got {actual_type}.",
            "character_limit": "'{field}' exceeds maximum length of {max_length} characters.",
        },
        ErrorMessageCategory.AUTHENTICATION: {
            "default": "Authentication failed. Please log in again.",
            "invalid_credentials": "Invalid username or password.",
            "token_expired": "Your session has expired. Please log in again.",
            "token_invalid": "Invalid authentication token. Please log in again.",
            "mfa_required": "Multi-factor authentication required.",
        },
        ErrorMessageCategory.PERMISSION: {
            "default": "You don't have permission to perform this action.",
            "insufficient_role": "This action requires {required_role} role or higher.",
            "campaign_access": "You don't have access to this campaign.",
            "character_access": "You don't have permission to modify this character.",
            "dm_only": "Only the Dungeon Master can perform this action.",
        },
        ErrorMessageCategory.NOT_FOUND: {
            "default": "The requested resource was not found.",
            "character": "Character '{name}' not found.",
            "campaign": "Campaign '{name}' not found.",
            "item": "Item '{name}' not found in inventory.",
            "spell": "Spell '{name}' not found in spellbook.",
            "rule": "Rule '{rule}' not found in rulebook.",
        },
        ErrorMessageCategory.CONFLICT: {
            "default": "A conflict occurred with your request.",
            "duplicate": "A {resource} with name '{name}' already exists.",
            "version": "The {resource} was modified by another user. Please refresh and try again.",
            "state": "Cannot perform this action in current state: {current_state}.",
            "combat": "Cannot modify character while in combat.",
        },
        ErrorMessageCategory.RATE_LIMIT: {
            "default": "Too many requests. Please slow down.",
            "api": "API rate limit exceeded. Please wait {retry_after} seconds.",
            "search": "Too many searches. Please wait before searching again.",
            "dice": "Too many dice rolls. Maximum {max_rolls} rolls per minute.",
        },
        ErrorMessageCategory.SERVICE: {
            "default": "External service error. Please try again later.",
            "unavailable": "The {service} service is currently unavailable.",
            "degraded": "The {service} service is running in degraded mode. Some features may be limited.",
            "maintenance": "The {service} service is under maintenance until {end_time}.",
            "timeout": "Request to {service} service timed out.",
        },
        ErrorMessageCategory.MCP: {
            "default": "MCP protocol error occurred.",
            "tool_not_found": "MCP tool '{tool}' not found.",
            "invalid_params": "Invalid parameters for MCP tool '{tool}'.",
            "execution_failed": "Failed to execute MCP tool '{tool}': {error}",
            "response_format": "Invalid response format from MCP tool '{tool}'.",
        },
    }

    # Suggestions for common errors
    SUGGESTIONS: Dict[ErrorMessageCategory, List[str]] = {
        ErrorMessageCategory.SYSTEM: [
            "Try refreshing the page",
            "Clear your browser cache",
            "Contact support if the problem persists",
        ],
        ErrorMessageCategory.DATABASE: [
            "Try again with fewer items",
            "Refresh your data",
            "Check for system status updates",
        ],
        ErrorMessageCategory.NETWORK: [
            "Check your internet connection",
            "Try disabling VPN if enabled",
            "Check firewall settings",
        ],
        ErrorMessageCategory.VALIDATION: [
            "Review the input requirements",
            "Check for required fields",
            "Ensure data formats match examples",
        ],
        ErrorMessageCategory.AUTHENTICATION: [
            "Try logging in again",
            "Reset your password if forgotten",
            "Check if your account is active",
        ],
        ErrorMessageCategory.PERMISSION: [
            "Contact your campaign administrator",
            "Request necessary permissions",
            "Check your role assignments",
        ],
        ErrorMessageCategory.NOT_FOUND: [
            "Verify the name or ID",
            "Check if the resource was deleted",
            "Try searching with different criteria",
        ],
        ErrorMessageCategory.CONFLICT: [
            "Refresh your data and try again",
            "Choose a different name",
            "Resolve conflicts before proceeding",
        ],
        ErrorMessageCategory.RATE_LIMIT: [
            "Wait before making more requests",
            "Reduce request frequency",
            "Consider batching operations",
        ],
        ErrorMessageCategory.SERVICE: [
            "Try again in a few minutes",
            "Use alternative features if available",
            "Check service status page",
        ],
        ErrorMessageCategory.MCP: [
            "Verify tool parameters",
            "Check MCP server connection",
            "Review tool documentation",
        ],
    }

    @classmethod
    def get_message(
        cls,
        category: ErrorMessageCategory,
        template_key: str = "default",
        **kwargs: Any,
    ) -> str:
        """
        Get formatted error message from template.

        Args:
            category: Error category
            template_key: Template key within category
            **kwargs: Template format arguments

        Returns:
            Formatted error message
        """
        templates = cls.TEMPLATES.get(category, {})
        template = templates.get(template_key, templates.get("default", "An error occurred"))
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            # Return template with placeholders if formatting fails
            return template

    @classmethod
    def get_suggestions(cls, category: ErrorMessageCategory) -> List[str]:
        """
        Get suggestions for error category.

        Args:
            category: Error category

        Returns:
            List of suggestions
        """
        return cls.SUGGESTIONS.get(category, ["Please try again later"])


class MCPErrorFormatter:
    """Formatter for MCP tool error responses."""

    @staticmethod
    def format_error(
        error: Exception,
        tool_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format error for MCP tool response.

        Args:
            error: The exception
            tool_name: MCP tool name
            context: Additional context

        Returns:
            Formatted error response
        """
        from ..core.error_handling import BaseError
        
        # Determine error details
        if isinstance(error, BaseError):
            error_code = error.error_code
            message = error.message
            category = error.category.name.lower()
            recoverable = error.recoverable
            retry_after = error.retry_after
            error_context = error.context
        else:
            error_code = "UNKNOWN"
            message = str(error)
            category = "system"
            recoverable = False
            retry_after = None
            error_context = {}
        
        # Get user-friendly message
        try:
            error_category = ErrorMessageCategory(category)
        except ValueError:
            error_category = ErrorMessageCategory.SYSTEM
        
        user_message = ErrorMessageTemplate.get_message(error_category)
        suggestions = ErrorMessageTemplate.get_suggestions(error_category)
        
        # Build response
        response = {
            "error": {
                "code": error_code,
                "message": user_message,
                "details": message,
                "category": category,
                "recoverable": recoverable,
                "suggestions": suggestions,
            }
        }
        
        if tool_name:
            response["error"]["tool"] = tool_name
        
        if retry_after:
            response["error"]["retry_after"] = retry_after
        
        if context:
            response["error"]["context"] = context
        
        if error_context:
            response["error"]["error_context"] = error_context
        
        return response

    @staticmethod
    def format_validation_errors(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format validation errors for MCP response.

        Args:
            errors: List of validation errors

        Returns:
            Formatted validation error response
        """
        formatted_errors = []
        
        for error in errors:
            field = error.get("field", "unknown")
            issue = error.get("issue", "invalid")
            value = error.get("value")
            
            # Get appropriate message template
            if issue == "required":
                message = ErrorMessageTemplate.get_message(
                    ErrorMessageCategory.VALIDATION,
                    "required_field",
                    field=field,
                )
            elif issue == "format":
                message = ErrorMessageTemplate.get_message(
                    ErrorMessageCategory.VALIDATION,
                    "invalid_format",
                    field=field,
                    expected_format=error.get("expected", "valid format"),
                )
            elif issue == "range":
                message = ErrorMessageTemplate.get_message(
                    ErrorMessageCategory.VALIDATION,
                    "out_of_range",
                    field=field,
                    min=error.get("min", "minimum"),
                    max=error.get("max", "maximum"),
                )
            else:
                message = f"Validation failed for field '{field}': {issue}"
            
            formatted_errors.append({
                "field": field,
                "message": message,
                "value": value,
            })
        
        return {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Input validation failed",
                "category": "validation",
                "recoverable": True,
                "validation_errors": formatted_errors,
                "suggestions": ErrorMessageTemplate.get_suggestions(
                    ErrorMessageCategory.VALIDATION
                ),
            }
        }

    @staticmethod
    def format_success_with_warnings(
        result: Any,
        warnings: List[str],
    ) -> Dict[str, Any]:
        """
        Format successful response with warnings.

        Args:
            result: Success result
            warnings: List of warning messages

        Returns:
            Formatted response with warnings
        """
        return {
            "success": True,
            "result": result,
            "warnings": warnings,
        }


class ErrorMessageBuilder:
    """Builder for constructing detailed error messages."""

    def __init__(self) -> None:
        """Initialize error message builder."""
        self.message: str = ""
        self.details: List[str] = []
        self.suggestions: List[str] = []
        self.context: Dict[str, Any] = {}
        self.error_code: Optional[str] = None
        self.category: Optional[ErrorMessageCategory] = None

    def with_message(self, message: str) -> "ErrorMessageBuilder":
        """Set main error message."""
        self.message = message
        return self

    def with_template(
        self,
        category: ErrorMessageCategory,
        template_key: str = "default",
        **kwargs: Any,
    ) -> "ErrorMessageBuilder":
        """Set message from template."""
        self.category = category
        self.message = ErrorMessageTemplate.get_message(category, template_key, **kwargs)
        self.suggestions = ErrorMessageTemplate.get_suggestions(category)
        return self

    def add_detail(self, detail: str) -> "ErrorMessageBuilder":
        """Add detail to error message."""
        self.details.append(detail)
        return self

    def add_suggestion(self, suggestion: str) -> "ErrorMessageBuilder":
        """Add suggestion for resolution."""
        self.suggestions.append(suggestion)
        return self

    def with_context(self, **context: Any) -> "ErrorMessageBuilder":
        """Add context information."""
        self.context.update(context)
        return self

    def with_code(self, code: str) -> "ErrorMessageBuilder":
        """Set error code."""
        self.error_code = code
        return self

    def build(self) -> Dict[str, Any]:
        """Build final error message structure."""
        result = {
            "message": self.message,
        }
        
        if self.error_code:
            result["code"] = self.error_code
        
        if self.category:
            result["category"] = self.category.value
        
        if self.details:
            result["details"] = self.details
        
        if self.suggestions:
            result["suggestions"] = self.suggestions
        
        if self.context:
            result["context"] = self.context
        
        return result

    def build_json(self) -> str:
        """Build error message as JSON string."""
        return json.dumps(self.build(), indent=2)


def create_user_friendly_error(
    error: Exception,
    operation: Optional[str] = None,
    resource: Optional[str] = None,
) -> str:
    """
    Create user-friendly error message from exception.

    Args:
        error: The exception
        operation: Operation being performed
        resource: Resource being accessed

    Returns:
        User-friendly error message
    """
    builder = ErrorMessageBuilder()
    
    # Import error classes for isinstance checks
    from ..core.error_handling import (
        BaseError
    )
    
    # Use isinstance checks for proper error categorization
    if isinstance(error, BaseError):
        # Use the category from the BaseError
        try:
            category_name = error.category.name.lower()
            error_category = ErrorMessageCategory(category_name)
        except (AttributeError, ValueError):
            error_category = ErrorMessageCategory.SYSTEM
            
        builder.with_template(error_category)
        if hasattr(error, 'context') and error.context:
            builder.with_context(**error.context)
    # Check standard library exceptions
    elif isinstance(error, FileNotFoundError):
        builder.with_template(
            ErrorMessageCategory.NOT_FOUND,
            resource or "default",
            name=resource or "resource",
        )
    elif isinstance(error, PermissionError):
        builder.with_template(ErrorMessageCategory.PERMISSION)
    elif isinstance(error, (ValueError, TypeError)):
        builder.with_template(ErrorMessageCategory.VALIDATION)
    elif isinstance(error, (ConnectionError, TimeoutError)):
        builder.with_template(ErrorMessageCategory.NETWORK)
    elif isinstance(error, OSError):
        # OSError can be many things, check errno if available
        if hasattr(error, 'errno'):
            import errno
            if error.errno in (errno.EACCES, errno.EPERM):
                builder.with_template(ErrorMessageCategory.PERMISSION)
            elif error.errno == errno.ENOENT:
                builder.with_template(
                    ErrorMessageCategory.NOT_FOUND,
                    resource or "default",
                    name=resource or "resource",
                )
            elif error.errno in (errno.ECONNREFUSED, errno.ENETUNREACH):
                builder.with_template(ErrorMessageCategory.NETWORK)
            else:
                builder.with_template(ErrorMessageCategory.SYSTEM)
        else:
            builder.with_template(ErrorMessageCategory.SYSTEM)
    else:
        # Fall back to string matching only for unknown exception types
        error_type = type(error).__name__
        
        if "NotFound" in error_type or "404" in str(error):
            builder.with_template(
                ErrorMessageCategory.NOT_FOUND,
                resource or "default",
                name=resource or "resource",
            )
        elif "Permission" in error_type or "Authorization" in error_type or "403" in str(error):
            builder.with_template(ErrorMessageCategory.PERMISSION)
        elif "Validation" in error_type or "Invalid" in error_type:
            builder.with_template(ErrorMessageCategory.VALIDATION)
        elif "Network" in error_type or "Connection" in error_type:
            builder.with_template(ErrorMessageCategory.NETWORK)
        elif "Database" in error_type or "SQL" in error_type:
            builder.with_template(ErrorMessageCategory.DATABASE)
        else:
            builder.with_template(ErrorMessageCategory.SYSTEM)
    
    if operation:
        builder.add_detail(f"Failed during: {operation}")
    
    if hasattr(error, "__cause__") and error.__cause__:
        builder.add_detail(f"Caused by: {str(error.__cause__)}")
    
    return builder.build_json()