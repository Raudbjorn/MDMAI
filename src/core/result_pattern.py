"""
Result pattern implementation using the returns library for MDMAI.

This module provides:
- Standardized error handling using Result/Either pattern
- Custom AppError type that works with returns library
- Decorators for automatic Result wrapping
- Helper functions for Result manipulation
"""

import asyncio
import functools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

from returns.result import Failure, Result, Success

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
E = TypeVar("E")
F = TypeVar("F")


class ErrorKind(Enum):
    """Categories of errors in the system."""

    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    DATABASE = "database"
    NETWORK = "network"
    PERMISSION = "permission"
    PARSING = "parsing"
    PROCESSING = "processing"
    RATE_LIMIT = "rate_limit"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    MCP_PROTOCOL = "mcp_protocol"
    SYSTEM = "system"


@dataclass(frozen=True)
class AppError:
    """
    Application-wide error type that works with returns library.
    
    This error type is designed to be informative and serializable,
    providing all necessary context for debugging and user feedback.
    """

    kind: ErrorKind
    message: str
    details: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    recoverable: bool = True

    def __str__(self) -> str:
        """String representation of the error."""
        base = f"[{self.kind.value}] {self.message}"
        if self.source:
            base = f"{base} (from {self.source})"
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": self.kind.value,
            "message": self.message,
            "details": self.details or {},
            "source": self.source,
            "recoverable": self.recoverable,
        }

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        kind: Optional[ErrorKind] = None,
        source: Optional[str] = None,
    ) -> "AppError":
        """Create AppError from a regular exception."""
        return cls(
            kind=kind or ErrorKind.SYSTEM,
            message=str(exception),
            details={"exception_type": type(exception).__name__},
            source=source,
            recoverable=False,
        )


# Convenience error constructors
def validation_error(msg: str, field: Optional[str] = None, **details: Any) -> AppError:
    """Create a validation error."""
    error_details = {"field": field} if field else {}
    error_details.update(details)
    return AppError(ErrorKind.VALIDATION, msg, error_details, recoverable=False)


def not_found_error(resource: str, identifier: str, **details: Any) -> AppError:
    """Create a not found error."""
    error_details = {"resource": resource, "id": identifier}
    error_details.update(details)
    return AppError(
        ErrorKind.NOT_FOUND,
        f"{resource} not found: {identifier}",
        error_details,
        recoverable=False,
    )


def database_error(msg: str, operation: Optional[str] = None, **details: Any) -> AppError:
    """Create a database error."""
    error_details = {"operation": operation} if operation else {}
    error_details.update(details)
    return AppError(ErrorKind.DATABASE, msg, error_details)


def permission_error(action: str, resource: str, **details: Any) -> AppError:
    """Create a permission denied error."""
    error_details = {"action": action, "resource": resource}
    error_details.update(details)
    return AppError(
        ErrorKind.PERMISSION,
        f"Permission denied for {action} on {resource}",
        error_details,
        recoverable=False,
    )


def network_error(msg: str, endpoint: Optional[str] = None, **details: Any) -> AppError:
    """Create a network error."""
    error_details = {"endpoint": endpoint} if endpoint else {}
    error_details.update(details)
    return AppError(ErrorKind.NETWORK, msg, error_details)


def mcp_protocol_error(msg: str, tool_name: Optional[str] = None, **details: Any) -> AppError:
    """Create an MCP protocol error."""
    error_details = {"tool_name": tool_name} if tool_name else {}
    error_details.update(details)
    return AppError(ErrorKind.MCP_PROTOCOL, msg, error_details)


# Decorator for automatic Result wrapping
def with_result(
    error_constructor: Optional[Callable[[str], AppError]] = None,
    error_kind: ErrorKind = ErrorKind.SYSTEM,
    source: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator that wraps function exceptions in Result[T, AppError].
    
    This decorator automatically catches exceptions and converts them to
    AppError wrapped in Failure, making error handling explicit and type-safe.
    
    Args:
        error_constructor: Optional function to create AppError from exception message
        error_kind: Default ErrorKind if no constructor provided
        source: Optional source identifier for error tracking
    
    Returns:
        Decorated function that returns Result[T, AppError]
    
    Example:
        @with_result(error_kind=ErrorKind.DATABASE)
        def fetch_user(user_id: str) -> User:
            # This can raise exceptions
            return db.get_user(user_id)
        
        # Now returns Result[User, AppError]
        result = fetch_user("123")
        if isinstance(result, Success):
            user = result.unwrap()
        else:
            error = result.failure()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Result[Any, AppError]:
            try:
                result = func(*args, **kwargs)
                # If function already returns Result, pass it through
                if isinstance(result, (Success, Failure)):
                    return result
                return Success(result)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    extra={"function": func.__name__, "source": source},
                    exc_info=True,
                )
                
                if error_constructor:
                    error = error_constructor(str(e))
                else:
                    error = AppError.from_exception(e, kind=error_kind, source=source or func.__name__)
                
                return Failure(error)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Result[Any, AppError]:
            try:
                result = await func(*args, **kwargs)
                # If function already returns Result, pass it through
                if isinstance(result, (Success, Failure)):
                    return result
                return Success(result)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    extra={"function": func.__name__, "source": source},
                    exc_info=True,
                )
                
                if error_constructor:
                    error = error_constructor(str(e))
                else:
                    error = AppError.from_exception(e, kind=error_kind, source=source or func.__name__)
                
                return Failure(error)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def collect_results(results: List[Result[T, E]]) -> Result[List[T], E]:
    """
    Collect a list of Results into a Result of list.
    
    If any Result is a Failure, returns the first Failure.
    Otherwise, returns Success with list of all values.
    
    Args:
        results: List of Result objects to collect
    
    Returns:
        Result containing either list of values or first error
    """
    values = []
    for result in results:
        if isinstance(result, Failure):
            return result
        values.append(result.unwrap())
    return Success(values)


async def collect_async_results(
    results: List[asyncio.Task[Result[T, E]]]
) -> Result[List[T], E]:
    """
    Collect async Results into a Result of list.
    
    Waits for all tasks to complete, then collects results.
    
    Args:
        results: List of async tasks returning Results
    
    Returns:
        Result containing either list of values or first error
    """
    awaited = await asyncio.gather(*results)
    return collect_results(awaited)


def map_error(result: Result[T, E], mapper: Callable[[E], F]) -> Result[T, F]:
    """
    Map the error value of a Result.
    
    Args:
        result: Result to map error for
        mapper: Function to transform error
    
    Returns:
        Result with mapped error type
    """
    if isinstance(result, Success):
        return result
    return Failure(mapper(result.failure()))


def flat_map_async(
    result: Result[T, E], 
    mapper: Callable[[T], Coroutine[Any, Any, Result[F, E]]]
) -> Coroutine[Any, Any, Result[F, E]]:
    """
    Async flat_map operation for Result.
    
    Args:
        result: Result to flat map
        mapper: Async function that returns a Result
    
    Returns:
        Coroutine that yields mapped Result
    """
    async def _flat_map() -> Result[F, E]:
        if isinstance(result, Success):
            return await mapper(result.unwrap())
        return result
    
    return _flat_map()


def unwrap_or_raise(result: Result[T, AppError]) -> T:
    """
    Unwrap a Result or raise the AppError as an exception.
    
    Useful for bridging Result-based code with exception-based code.
    
    Args:
        result: Result to unwrap
    
    Returns:
        The success value
    
    Raises:
        RuntimeError: With the AppError message if Result is Failure
    """
    if isinstance(result, Success):
        return result.unwrap()
    
    error = result.failure()
    raise RuntimeError(str(error))


# Example usage functions showing patterns
def result_to_response(result: Result[T, AppError]) -> Dict[str, Any]:
    """
    Convert a Result to an API response dictionary.
    
    This is useful for FastAPI endpoints that need to return
    consistent response formats.
    
    Args:
        result: Result to convert
    
    Returns:
        Dictionary suitable for JSON response
    """
    if isinstance(result, Success):
        return {
            "success": True,
            "data": result.unwrap(),
        }
    
    error = result.failure()
    return {
        "success": False,
        "error": error.to_dict(),
    }


def chain_results(
    *operations: Callable[[Any], Result[Any, AppError]]
) -> Callable[[Any], Result[Any, AppError]]:
    """
    Chain multiple Result-returning operations.
    
    Each operation receives the unwrapped value from the previous operation.
    If any operation fails, the chain short-circuits and returns the failure.
    
    Args:
        *operations: Functions that return Results
    
    Returns:
        Function that chains all operations
    
    Example:
        process = chain_results(
            validate_input,
            transform_data,
            save_to_database
        )
        result = process(input_data)
    """
    def chained(initial_value: Any) -> Result[Any, AppError]:
        result = Success(initial_value)
        for operation in operations:
            if isinstance(result, Failure):
                return result
            result = result.bind(operation)
        return result
    
    return chained


# Async-specific helpers
class AsyncResult:
    """Helper class for working with async Results."""
    
    @staticmethod
    def from_coroutine(
        coro: Coroutine[Any, Any, T],
        error_kind: ErrorKind = ErrorKind.SYSTEM,
    ) -> Coroutine[Any, Any, Result[T, AppError]]:
        """
        Convert a coroutine to one that returns a Result.
        
        Args:
            coro: Coroutine to wrap
            error_kind: ErrorKind for exceptions
        
        Returns:
            Coroutine that returns Result
        """
        async def wrapped() -> Result[T, AppError]:
            try:
                value = await coro
                return Success(value)
            except Exception as e:
                error = AppError.from_exception(e, kind=error_kind)
                return Failure(error)
        
        return wrapped()
    
    @staticmethod
    async def sequence(
        results: List[Result[T, E]]
    ) -> Result[List[T], E]:
        """
        Sequence a list of Results into a Result of list.
        
        Similar to collect_results but async-aware.
        
        Args:
            results: List of Results to sequence
        
        Returns:
            Result containing list or first error
        """
        return collect_results(results)


# Type aliases for common Result types
UserResult = Result[Dict[str, Any], AppError]
QueryResult = Result[List[Dict[str, Any]], AppError]
BoolResult = Result[bool, AppError]
StringResult = Result[str, AppError]
VoidResult = Result[None, AppError]