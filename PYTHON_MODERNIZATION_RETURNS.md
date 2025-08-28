# Python Modernization with Result Pattern (Using returns Library)

## Overview

This document outlines the Python modernization strategy for the TTRPG Assistant MCP Server, focusing on using the `returns` library (v0.22.0) for implementing the Result/Either pattern consistently throughout the codebase.

## 1. Core Dependencies

```txt
# Result Pattern & Error Handling
returns==0.22.0         # Result/Either pattern implementation

# Modern Python Dependencies
httpx==0.26.0          # Async HTTP client (replaces requests)
tenacity==8.2.0        # Retry logic with exponential backoff
structlog==24.1.0      # Structured logging
pypdf==3.17.0          # PDF processing (replaces PyPDF2)
pdfplumber==0.10.3     # Advanced PDF parsing
```

## 2. Result Pattern Implementation with returns Library

### Basic Setup

```python
# src/core/result_pattern.py
from typing import TypeVar, Optional, Callable, Any, List, Dict
from dataclasses import dataclass
from enum import Enum
import functools
import asyncio
import logging

# Import from returns library
from returns.result import Result, Success, Failure
from returns.pipeline import is_successful, flow
from returns.pointfree import bind, map_
from returns.converters import maybe_to_result
from returns.functions import raise_exception

logger = logging.getLogger(__name__)

T = TypeVar('T')
E = TypeVar('E')


class ErrorKind(Enum):
    """Categorizes different types of errors."""
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    DATABASE = "database"
    PERMISSION = "permission"
    NETWORK = "network"
    PARSING = "parsing"
    RATE_LIMIT = "rate_limit"
    INTERNAL = "internal"


@dataclass(frozen=True)
class AppError:
    """Application-specific error type with rich information."""
    kind: ErrorKind
    message: str
    details: Optional[Any] = None
    recovery_hint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "kind": self.kind.value,
            "message": self.message,
            "details": self.details,
            "recovery_hint": self.recovery_hint
        }
```

### Error Constructors

```python
# Convenience constructors for common errors
def validation_error(message: str, field: Optional[str] = None) -> AppError:
    """Create a validation error."""
    return AppError(
        kind=ErrorKind.VALIDATION,
        message=message,
        details={"field": field} if field else None,
        recovery_hint="Please check the input data and try again"
    )


def not_found_error(resource: str, identifier: str) -> AppError:
    """Create a not found error."""
    return AppError(
        kind=ErrorKind.NOT_FOUND,
        message=f"{resource} with id '{identifier}' not found",
        details={"resource": resource, "id": identifier},
        recovery_hint=f"Ensure the {resource.lower()} exists"
    )


def database_error(message: str, operation: Optional[str] = None) -> AppError:
    """Create a database error."""
    return AppError(
        kind=ErrorKind.DATABASE,
        message=message,
        details={"operation": operation} if operation else None,
        recovery_hint="Check database connection and retry"
    )
```

### Fixed Decorator for Exception Handling

```python
def with_result(
    error_kind: Optional[ErrorKind] = None,
    error_constructor: Optional[Callable[[str], AppError]] = None
):
    """
    Decorator to wrap function exceptions in Result.
    
    Args:
        error_kind: ErrorKind to use for AppError construction
        error_constructor: Custom error constructor function
    
    Example:
        @with_result(error_kind=ErrorKind.DATABASE)
        async def fetch_data(id: str) -> Dict:
            # Exceptions will be caught and wrapped as AppError
            return await db.query(id)
            
        @with_result(error_constructor=lambda msg: database_error(msg, "fetch"))
        def process_data(data: str) -> str:
            # Custom error constructor
            return transform(data)
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Result[Any, AppError]:
            try:
                result = await func(*args, **kwargs)
                # If already a Result, return as-is
                if isinstance(result, (Success, Failure)):
                    return result
                return Success(result)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                
                if error_constructor:
                    # Use custom constructor
                    return Failure(error_constructor(str(e)))
                elif error_kind:
                    # Use specified error kind
                    return Failure(AppError(
                        kind=error_kind,
                        message=f"Error in {func.__name__}: {str(e)}"
                    ))
                else:
                    # Default to internal error
                    return Failure(AppError(
                        kind=ErrorKind.INTERNAL,
                        message=str(e)
                    ))

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Result[Any, AppError]:
            try:
                result = func(*args, **kwargs)
                if isinstance(result, (Success, Failure)):
                    return result
                return Success(result)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                
                if error_constructor:
                    return Failure(error_constructor(str(e)))
                elif error_kind:
                    return Failure(AppError(
                        kind=error_kind,
                        message=f"Error in {func.__name__}: {str(e)}"
                    ))
                else:
                    return Failure(AppError(
                        kind=ErrorKind.INTERNAL,
                        message=str(e)
                    ))

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

## 3. Service Implementation with returns

### Campaign Manager Example

```python
# src/campaign/campaign_manager.py
from typing import Dict, List, Optional
from returns.result import Result, Success, Failure
from returns.pipeline import flow, is_successful
from returns.pointfree import bind, map_
from returns.converters import flatten
from returns.iterables import Fold

from src.core.result_pattern import (
    AppError, ErrorKind, with_result,
    validation_error, not_found_error, database_error
)
from src.models.campaign import Campaign, Character, Location


class CampaignManager:
    """Manages campaign operations with Result pattern."""
    
    def __init__(self, db_client):
        self.db = db_client
        self.collection_name = "campaigns"
    
    async def create_campaign(
        self,
        name: str,
        system: str,
        description: Optional[str] = None
    ) -> Result[Campaign, AppError]:
        """
        Create a new campaign with validation.
        Uses returns library for Result handling.
        """
        # Validation using Result chaining
        validation_result = self._validate_campaign_data(name, system)
        if not is_successful(validation_result):
            return validation_result
        
        # Create campaign using flow pipeline
        return await flow(
            Success(Campaign(name=name, system=system, description=description)),
            bind(self._check_duplicate_name),
            bind(self._store_campaign),
            map_(self._enrich_campaign_data)
        )
    
    def _validate_campaign_data(
        self,
        name: str,
        system: str
    ) -> Result[None, AppError]:
        """Validate campaign input data."""
        if not name or len(name) < 3:
            return Failure(validation_error(
                "Campaign name must be at least 3 characters",
                field="name"
            ))
        
        if system not in self._get_supported_systems():
            return Failure(validation_error(
                f"System '{system}' is not supported",
                field="system"
            ))
        
        return Success(None)
    
    @with_result(error_kind=ErrorKind.DATABASE)
    async def _check_duplicate_name(
        self,
        campaign: Campaign
    ) -> Campaign:
        """Check for duplicate campaign names."""
        existing = await self.db.query(
            collection=self.collection_name,
            filter={"name": campaign.name}
        )
        
        if existing:
            raise ValueError(f"Campaign '{campaign.name}' already exists")
        
        return campaign
    
    @with_result(error_constructor=lambda msg: database_error(msg, "store"))
    async def _store_campaign(self, campaign: Campaign) -> Campaign:
        """Store campaign in database."""
        await self.db.add_document(
            collection=self.collection_name,
            document_id=campaign.id,
            content=campaign.to_dict()
        )
        return campaign
    
    async def get_campaign(
        self,
        campaign_id: str
    ) -> Result[Campaign, AppError]:
        """
        Retrieve a campaign by ID.
        """
        # Use the with_result decorator for automatic error handling
        @with_result(error_kind=ErrorKind.DATABASE)
        async def fetch():
            result = await self.db.get_document(
                collection=self.collection_name,
                document_id=campaign_id
            )
            if not result:
                raise ValueError(f"Campaign not found: {campaign_id}")
            return Campaign.from_dict(result)
        
        result = await fetch()
        
        # Transform not found errors
        if not is_successful(result):
            error = result.failure()
            if "not found" in error.message.lower():
                return Failure(not_found_error("Campaign", campaign_id))
        
        return result
    
    async def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any]
    ) -> Result[Campaign, AppError]:
        """Update campaign with validation."""
        # Chain operations using returns pipeline
        return await flow(
            self.get_campaign(campaign_id),
            bind(lambda c: self._validate_updates(c, updates)),
            bind(self._apply_updates),
            bind(self._save_campaign)
        )
    
    def _get_supported_systems(self) -> List[str]:
        """Get list of supported game systems."""
        return ["D&D 5e", "Pathfinder 2e", "Call of Cthulhu", "Blades in the Dark"]
```

## 4. Batch Operations with returns

```python
from returns.iterables import Fold
from returns.converters import flatten


class BatchProcessor:
    """Handle batch operations with Result pattern."""
    
    async def process_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> Result[List[Any], AppError]:
        """
        Process multiple items, collecting all results.
        """
        # Process each item
        results = [await self.process_item(item) for item in items]
        
        # Collect all results - fails on first error
        return Fold.collect(results, Success(()))
    
    async def process_item(
        self,
        item: Dict[str, Any]
    ) -> Result[Any, AppError]:
        """Process a single item."""
        return await flow(
            Success(item),
            bind(self.validate_item),
            bind(self.transform_item),
            bind(self.save_item)
        )
    
    @with_result(error_kind=ErrorKind.VALIDATION)
    def validate_item(self, item: Dict) -> Dict:
        """Validate item data."""
        if "id" not in item:
            raise ValueError("Item missing required 'id' field")
        return item
```

## 5. MCP Integration

```python
# src/mcp_tools.py
from typing import Dict, Any
from returns.result import Result, is_successful
from mcp.server.fastmcp import FastMCP

from src.campaign.campaign_manager import CampaignManager
from src.core.result_pattern import AppError


def format_mcp_response(result: Result[Any, AppError]) -> Dict[str, Any]:
    """Convert Result to MCP tool response format."""
    if is_successful(result):
        return {
            "success": True,
            "data": result.unwrap()
        }
    else:
        error = result.failure()
        return {
            "success": False,
            "error": error.to_dict()
        }


def register_campaign_tools(mcp: FastMCP, manager: CampaignManager):
    """Register campaign tools with MCP server."""
    
    @mcp.tool()
    async def create_campaign(
        name: str,
        system: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new campaign."""
        result = await manager.create_campaign(name, system, description)
        return format_mcp_response(result)
    
    @mcp.tool()
    async def get_campaign(campaign_id: str) -> Dict[str, Any]:
        """Retrieve campaign data."""
        result = await manager.get_campaign(campaign_id)
        return format_mcp_response(result)
```

## 6. Testing with returns

```python
# tests/test_campaign_manager.py
import pytest
from returns.result import is_successful, Success, Failure

from src.campaign.campaign_manager import CampaignManager
from src.core.result_pattern import ErrorKind


@pytest.mark.asyncio
async def test_create_campaign_success(mock_db):
    """Test successful campaign creation."""
    manager = CampaignManager(mock_db)
    
    result = await manager.create_campaign(
        name="Dragon's Crown",
        system="D&D 5e",
        description="Epic adventure"
    )
    
    assert is_successful(result)
    campaign = result.unwrap()
    assert campaign.name == "Dragon's Crown"
    assert campaign.system == "D&D 5e"


@pytest.mark.asyncio
async def test_create_campaign_validation_error(mock_db):
    """Test campaign creation with invalid data."""
    manager = CampaignManager(mock_db)
    
    result = await manager.create_campaign(
        name="",  # Invalid: empty name
        system="D&D 5e"
    )
    
    assert not is_successful(result)
    error = result.failure()
    assert error.kind == ErrorKind.VALIDATION
    assert "at least 3 characters" in error.message


@pytest.mark.asyncio
async def test_campaign_not_found(mock_db):
    """Test retrieving non-existent campaign."""
    manager = CampaignManager(mock_db)
    mock_db.get_document.return_value = None
    
    result = await manager.get_campaign("non-existent-id")
    
    assert not is_successful(result)
    error = result.failure()
    assert error.kind == ErrorKind.NOT_FOUND
    assert "Campaign" in error.message
```

## 7. Benefits of Using returns Library

1. **Battle-tested**: Well-maintained library with extensive documentation
2. **Rich utilities**: Provides flow, bind, map, fold, and many other combinators
3. **Type safety**: Full mypy support with proper generics
4. **Async support**: Works seamlessly with async/await
5. **Composability**: Easy to chain operations and handle errors
6. **Consistent API**: Same patterns work for sync and async code

## 8. Migration Strategy

### Phase 1: Core Infrastructure
1. Install returns library: `pip install returns==0.22.0`
2. Create `src/core/result_pattern.py` with AppError and helpers
3. Update type hints in interfaces

### Phase 2: Service Layer
1. Update service methods to return `Result[T, AppError]`
2. Add `@with_result` decorator to existing functions
3. Replace try/except blocks with Result handling

### Phase 3: MCP Tools
1. Update tool functions to handle Results
2. Create consistent response format
3. Add error details to responses

### Phase 4: Testing
1. Update tests to check both success and failure paths
2. Add property-based tests for Result combinators
3. Ensure 100% coverage of error paths

## 9. Common Patterns

### Chaining Operations
```python
from returns.pipeline import flow
from returns.pointfree import bind, map_

result = await flow(
    initial_value,
    bind(validate),
    bind(transform),
    bind(save),
    map_(format_response)
)
```

### Collecting Multiple Results
```python
from returns.iterables import Fold

results = [process(item) for item in items]
all_results = Fold.collect(results, Success(()))
```

### Early Returns
```python
async def complex_operation(data: str) -> Result[str, AppError]:
    # Validate first
    validation = validate_input(data)
    if not is_successful(validation):
        return validation
    
    # Process if valid
    return await process_data(data)
```

This approach using the `returns` library provides a consistent, type-safe, and composable way to handle errors throughout the TTRPG Assistant codebase.