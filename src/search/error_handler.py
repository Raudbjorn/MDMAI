"""Error handling for the search system."""

from typing import Any, Dict, Optional, Type
from functools import wraps
import traceback
import asyncio

from config.logging_config import get_logger

logger = get_logger(__name__)


class SearchError(Exception):
    """Base exception for search-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class QueryProcessingError(SearchError):
    """Error in query processing."""
    pass


class EmbeddingGenerationError(SearchError):
    """Error generating embeddings."""
    pass


class DatabaseError(SearchError):
    """Database operation error."""
    pass


class CacheError(SearchError):
    """Cache operation error."""
    pass


class IndexError(SearchError):
    """Search index error."""
    pass


def handle_search_errors(fallback_result=None):
    """
    Decorator for handling errors in search operations.
    
    Args:
        fallback_result: Result to return on error
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except SearchError as e:
                logger.error(f"Search error in {func.__name__}: {str(e)}", details=e.details)
                if fallback_result is not None:
                    return fallback_result
                return _create_error_response(str(e), "search_error")
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}", 
                           traceback=traceback.format_exc())
                if fallback_result is not None:
                    return fallback_result
                return _create_error_response(
                    "An unexpected error occurred during search",
                    "unexpected_error"
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except SearchError as e:
                logger.error(f"Search error in {func.__name__}: {str(e)}", details=e.details)
                if fallback_result is not None:
                    return fallback_result
                return _create_error_response(str(e), "search_error")
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}", 
                           traceback=traceback.format_exc())
                if fallback_result is not None:
                    return fallback_result
                return _create_error_response(
                    "An unexpected error occurred during search",
                    "unexpected_error"
                )
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def _create_error_response(message: str, error_type: str) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        error_type: Type of error
        
    Returns:
        Error response dictionary
    """
    return {
        "error": True,
        "error_type": error_type,
        "message": message,
        "results": [],
        "total_results": 0,
        "suggestions": [
            "Try simplifying your query",
            "Check spelling and try again",
            "Use different search terms",
        ],
    }


class ErrorRecovery:
    """Handles error recovery strategies."""
    
    @staticmethod
    def with_retry(func, max_retries: int = 3, delay: float = 1.0):
        """
        Retry a function on failure.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            delay: Delay between retries in seconds
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        await asyncio.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_error
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        import time
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_error
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    @staticmethod
    def graceful_degradation(primary_func, fallback_func):
        """
        Use fallback function if primary fails.
        
        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function on failure
        """
        @wraps(primary_func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function failed: {str(e)}. Using fallback.")
                return await fallback_func(*args, **kwargs)
        
        @wraps(primary_func)
        def sync_wrapper(*args, **kwargs):
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function failed: {str(e)}. Using fallback.")
                return fallback_func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(primary_func):
            return async_wrapper
        else:
            return sync_wrapper


class SearchValidator:
    """Validates search inputs and outputs."""
    
    @staticmethod
    def validate_query(query: str) -> str:
        """
        Validate and sanitize search query.
        
        Args:
            query: Search query
            
        Returns:
            Validated query
            
        Raises:
            QueryProcessingError: If query is invalid
        """
        if not query or not query.strip():
            raise QueryProcessingError("Query cannot be empty")
        
        # Remove dangerous characters
        sanitized = query.strip()
        
        # Check length
        if len(sanitized) > 500:
            raise QueryProcessingError("Query too long (max 500 characters)")
        
        if len(sanitized) < 2:
            raise QueryProcessingError("Query too short (min 2 characters)")
        
        return sanitized
    
    @staticmethod
    def validate_results(results: list) -> list:
        """
        Validate search results.
        
        Args:
            results: Search results
            
        Returns:
            Validated results
        """
        if not results:
            return []
        
        validated = []
        for result in results:
            try:
                # Ensure required fields exist
                if not isinstance(result, dict):
                    logger.warning(f"Invalid result type: {type(result)}")
                    continue
                
                if "content" not in result:
                    logger.warning("Result missing content field")
                    continue
                
                # Ensure content is not empty
                if not result["content"] or not str(result["content"]).strip():
                    logger.warning("Result has empty content")
                    continue
                
                validated.append(result)
                
            except Exception as e:
                logger.warning(f"Error validating result: {str(e)}")
                continue
        
        return validated