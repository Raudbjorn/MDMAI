"""Custom exception classes for the TTRPG Assistant MCP Server."""

from typing import Optional, Dict, Any


class TTRPGException(Exception):
    """Base exception for all TTRPG Assistant errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Database Exceptions
class DatabaseException(TTRPGException):
    """Base exception for database-related errors."""
    pass


class ConnectionError(DatabaseException):
    """Raised when database connection fails."""
    pass


class CollectionNotFoundError(DatabaseException):
    """Raised when a collection doesn't exist."""
    pass


class DocumentNotFoundError(DatabaseException):
    """Raised when a document is not found."""
    pass


class DuplicateDocumentError(DatabaseException):
    """Raised when attempting to create a duplicate document."""
    pass


class QueryExecutionError(DatabaseException):
    """Raised when a database query fails."""
    pass


# PDF Processing Exceptions
class PDFProcessingException(TTRPGException):
    """Base exception for PDF processing errors."""
    pass


class PDFReadError(PDFProcessingException):
    """Raised when PDF cannot be read."""
    pass


class PDFParseError(PDFProcessingException):
    """Raised when PDF parsing fails."""
    pass


class InvalidPDFError(PDFProcessingException):
    """Raised when PDF is invalid or corrupted."""
    pass


class PDFSizeError(PDFProcessingException):
    """Raised when PDF exceeds size limits."""
    pass


class ChunkingError(PDFProcessingException):
    """Raised when content chunking fails."""
    pass


class EmbeddingGenerationError(PDFProcessingException):
    """Raised when embedding generation fails."""
    pass


# Search Exceptions
class SearchException(TTRPGException):
    """Base exception for search-related errors."""
    pass


class InvalidQueryError(SearchException):
    """Raised when search query is invalid."""
    pass


class SearchTimeoutError(SearchException):
    """Raised when search times out."""
    pass


class IndexNotFoundError(SearchException):
    """Raised when search index is missing."""
    pass


class SearchServiceUnavailableError(SearchException):
    """Raised when search service is unavailable."""
    pass


# Campaign Management Exceptions
class CampaignException(TTRPGException):
    """Base exception for campaign-related errors."""
    pass


class CampaignNotFoundError(CampaignException):
    """Raised when campaign doesn't exist."""
    pass


class InvalidCampaignDataError(CampaignException):
    """Raised when campaign data is invalid."""
    pass


class CampaignVersionError(CampaignException):
    """Raised when campaign version operations fail."""
    pass


class CampaignLockError(CampaignException):
    """Raised when campaign is locked by another operation."""
    pass


# Session Management Exceptions
class SessionException(TTRPGException):
    """Base exception for session-related errors."""
    pass


class SessionNotFoundError(SessionException):
    """Raised when session doesn't exist."""
    pass


class InvalidSessionStateError(SessionException):
    """Raised when session is in invalid state for operation."""
    pass


class SessionAlreadyActiveError(SessionException):
    """Raised when trying to start a session when one is already active."""
    pass


# Character Generation Exceptions
class CharacterGenerationException(TTRPGException):
    """Base exception for character generation errors."""
    pass


class InvalidSystemError(CharacterGenerationException):
    """Raised when game system is not supported."""
    pass


class InvalidClassError(CharacterGenerationException):
    """Raised when character class is invalid."""
    pass


class InvalidLevelError(CharacterGenerationException):
    """Raised when character level is invalid."""
    pass


class GenerationFailedError(CharacterGenerationException):
    """Raised when character generation fails."""
    pass


# Personality System Exceptions
class PersonalityException(TTRPGException):
    """Base exception for personality system errors."""
    pass


class PersonalityNotFoundError(PersonalityException):
    """Raised when personality profile doesn't exist."""
    pass


class PersonalityExtractionError(PersonalityException):
    """Raised when personality extraction fails."""
    pass


# Source Management Exceptions
class SourceException(TTRPGException):
    """Base exception for source management errors."""
    pass


class SourceNotFoundError(SourceException):
    """Raised when source doesn't exist."""
    pass


class DuplicateSourceError(SourceException):
    """Raised when source already exists."""
    pass


class InvalidSourceTypeError(SourceException):
    """Raised when source type is invalid."""
    pass


class SourceValidationError(SourceException):
    """Raised when source validation fails."""
    pass


# Performance Exceptions
class PerformanceException(TTRPGException):
    """Base exception for performance-related errors."""
    pass


class ResourceLimitExceededError(PerformanceException):
    """Raised when resource limits are exceeded."""
    pass


class TaskExecutionError(PerformanceException):
    """Raised when task execution fails."""
    pass


class ProcessorShutdownError(PerformanceException):
    """Raised when processor is shut down."""
    pass


class CacheError(PerformanceException):
    """Raised when cache operations fail."""
    pass


# MCP Tool Exceptions
class MCPToolException(TTRPGException):
    """Base exception for MCP tool errors."""
    pass


class InvalidParameterError(MCPToolException):
    """Raised when MCP tool receives invalid parameters."""
    pass


class ToolExecutionError(MCPToolException):
    """Raised when MCP tool execution fails."""
    pass


class ToolNotFoundError(MCPToolException):
    """Raised when MCP tool doesn't exist."""
    pass


# Validation Exceptions (already in security.py but included for completeness)
class ValidationException(TTRPGException):
    """Base exception for validation errors."""
    pass


class InputValidationError(ValidationException):
    """Raised when input validation fails."""
    pass


class PathValidationError(ValidationException):
    """Raised when path validation fails."""
    pass


class MetadataValidationError(ValidationException):
    """Raised when metadata validation fails."""
    pass


# Utility functions for exception handling
def format_exception_response(exception: TTRPGException) -> Dict[str, Any]:
    """
    Format an exception into a standard response dictionary.
    
    Args:
        exception: The exception to format
        
    Returns:
        Formatted error response
    """
    return {
        "success": False,
        "error": str(exception),
        "error_type": exception.__class__.__name__,
        "details": exception.details if hasattr(exception, 'details') else {}
    }


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an error is retryable.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the error is retryable
    """
    retryable_types = (
        ConnectionError,
        SearchTimeoutError,
        SearchServiceUnavailableError,
        CampaignLockError,
        ResourceLimitExceededError,
    )
    
    return isinstance(exception, retryable_types)


def get_error_severity(exception: Exception) -> str:
    """
    Get the severity level of an error.
    
    Args:
        exception: The exception to check
        
    Returns:
        Severity level (CRITICAL, ERROR, WARNING)
    """
    critical_types = (
        ConnectionError,
        ProcessorShutdownError,
        InvalidPDFError,
    )
    
    warning_types = (
        CacheError,
        PersonalityNotFoundError,
    )
    
    if isinstance(exception, critical_types):
        return "CRITICAL"
    elif isinstance(exception, warning_types):
        return "WARNING"
    else:
        return "ERROR"