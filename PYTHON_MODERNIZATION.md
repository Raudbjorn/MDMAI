# Python Backend Modernization with Result Pattern

## Overview

This document outlines the modernization of Python dependencies and the adoption of the error-as-values pattern using a Result type, replacing exception-based error handling throughout the MCP server backend.

## Core Dependencies Updates

### Current → Modern Dependencies

```toml
# pyproject.toml
[project]
name = "mdmai-mcp-server"
version = "2.0.0"
requires-python = ">=3.11"

[dependencies]
# Core MCP
mcp = "^0.5.0"
fastmcp = "^0.2.0"

# Web Framework (Bridge)
fastapi = "^0.104.0"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
websockets = "^12.0"

# Database
chromadb = "^0.4.18"
sqlalchemy = "^2.0.23"
alembic = "^1.12.1"

# PDF Processing
pypdf = "^3.17.0"  # Replacing PyPDF2
pdfplumber = "^0.10.3"
pillow = "^10.1.0"

# NLP/ML
sentence-transformers = "^2.2.2"
spacy = "^3.7.2"
tiktoken = "^0.5.1"

# Search
rank-bm25 = "^0.2.2"
whoosh = "^2.7.4"

# Result Pattern & Validation
result = "^0.9.0"  # Or custom implementation
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"

# Async & Concurrency
anyio = "^4.0.0"
trio = "^0.23.0"

# Caching
redis = "^5.0.1"
diskcache = "^5.6.3"

# Monitoring
prometheus-client = "^0.19.0"
opentelemetry-api = "^1.21.0"
opentelemetry-sdk = "^1.21.0"

# Testing
pytest = "^7.4.3"
pytest-asyncio = "^0.21.1"
hypothesis = "^6.91.0"
pytest-cov = "^4.1.0"

# Development
black = "^23.11.0"
ruff = "^0.1.6"
mypy = "^1.7.1"
pre-commit = "^3.5.0"
```

## Result Pattern Implementation

### 1. Core Result Type

```python
# src/core/result.py
from typing import TypeVar, Generic, Union, Optional, Callable, Awaitable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import functools
import asyncio
import logging

T = TypeVar('T')
E = TypeVar('E')
U = TypeVar('U')

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Ok(Generic[T]):
    """Represents a successful result."""
    value: T
    
    def is_ok(self) -> bool:
        return True
    
    def is_err(self) -> bool:
        return False
    
    def unwrap(self) -> T:
        """Get the value, safe since we know it's Ok."""
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Get the value."""
        return self.value
    
    def map(self, fn: Callable[[T], U]) -> 'Result[U, E]':
        """Transform the value if Ok."""
        return Ok(fn(self.value))
    
    def flat_map(self, fn: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        """Chain operations that return Results."""
        return fn(self.value)
    
    def map_err(self, fn: Callable[[E], E]) -> 'Result[T, E]':
        """Transform error if Err (no-op for Ok)."""
        return self

@dataclass(frozen=True)
class Err(Generic[E]):
    """Represents an error result."""
    error: E
    
    def is_ok(self) -> bool:
        return False
    
    def is_err(self) -> bool:
        return True
    
    def unwrap(self) -> None:
        """Raise an exception with the error."""
        raise ValueError(f"Called unwrap on Err: {self.error}")
    
    def unwrap_or(self, default: T) -> T:
        """Return the default value."""
        return default
    
    def map(self, fn: Callable[[T], U]) -> 'Result[U, E]':
        """Transform value if Ok (no-op for Err)."""
        return self
    
    def flat_map(self, fn: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        """Chain operations (no-op for Err)."""
        return self
    
    def map_err(self, fn: Callable[[E], E]) -> 'Result[T, E]':
        """Transform the error."""
        return Err(fn(self.error))

Result = Union[Ok[T], Err[E]]

# Helper functions
def ok(value: T) -> Result[T, E]:
    """Create a successful result."""
    return Ok(value)

def err(error: E) -> Result[T, E]:
    """Create an error result."""
    return Err(error)

def collect_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """Collect a list of Results into a Result of list."""
    values = []
    for result in results:
        if isinstance(result, Err):
            return result
        values.append(result.value)
    return Ok(values)

async def collect_async_results(
    results: list[Awaitable[Result[T, E]]]
) -> Result[list[T], E]:
    """Collect async Results into a Result of list."""
    awaited = await asyncio.gather(*results)
    return collect_results(awaited)

# Decorator for automatic Result wrapping
def with_result(error_type: type[E] = str):
    """Decorator to wrap function exceptions in Result."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Result:
            try:
                result = await func(*args, **kwargs)
                return Ok(result) if not isinstance(result, (Ok, Err)) else result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                if isinstance(error_type, type) and error_type == str:
                    return Err(str(e))
                return Err(error_type(str(e)) if callable(error_type) else e)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Result:
            try:
                result = func(*args, **kwargs)
                return Ok(result) if not isinstance(result, (Ok, Err)) else result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                if isinstance(error_type, type) and error_type == str:
                    return Err(str(e))
                return Err(error_type(str(e)) if callable(error_type) else e)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

### 2. Domain Error Types

```python
# src/core/errors.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

class ErrorKind(Enum):
    """Categories of errors in the system."""
    DATABASE = "database"
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    PERMISSION = "permission"
    NETWORK = "network"
    PARSING = "parsing"
    PROCESSING = "processing"
    RATE_LIMIT = "rate_limit"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"

@dataclass(frozen=True)
class AppError:
    """Application-wide error type."""
    kind: ErrorKind
    message: str
    details: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    
    def __str__(self) -> str:
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
            "source": self.source
        }

# Convenience constructors
def database_error(msg: str, **details) -> AppError:
    return AppError(ErrorKind.DATABASE, msg, details)

def validation_error(msg: str, **details) -> AppError:
    return AppError(ErrorKind.VALIDATION, msg, details)

def not_found(resource: str, id: str) -> AppError:
    return AppError(
        ErrorKind.NOT_FOUND,
        f"{resource} not found",
        {"resource": resource, "id": id}
    )

def permission_denied(action: str, resource: str) -> AppError:
    return AppError(
        ErrorKind.PERMISSION,
        f"Permission denied for {action} on {resource}",
        {"action": action, "resource": resource}
    )
```

### 3. Updated Campaign Manager with Result Pattern

```python
# src/campaign/campaign_manager.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import logging

from src.core.result import Result, Ok, Err, with_result
from src.core.errors import AppError, not_found, database_error, validation_error
from src.core.database import ChromaDBManager
from src.campaign.models import Campaign, Character, NPC, Location

logger = logging.getLogger(__name__)

class CampaignManager:
    """Manages campaign data with Result pattern for error handling."""
    
    def __init__(self, db: ChromaDBManager):
        self.db = db
        self.collection_name = "campaigns"
    
    async def create_campaign(
        self,
        name: str,
        system: str,
        description: Optional[str] = None
    ) -> Result[Campaign, AppError]:
        """Create a new campaign with validation."""
        # Validate inputs
        if not name or not name.strip():
            return Err(validation_error("Campaign name cannot be empty"))
        
        if not system or not system.strip():
            return Err(validation_error("System cannot be empty"))
        
        # Create campaign
        campaign = Campaign(
            id=str(uuid.uuid4()),
            name=name.strip(),
            system=system.strip(),
            description=description,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store in database
        storage_result = await self._store_campaign(campaign)
        if isinstance(storage_result, Err):
            return storage_result
        
        logger.info(f"Created campaign: {campaign.id}")
        return Ok(campaign)
    
    async def get_campaign(
        self,
        campaign_id: str,
        include_related: bool = False
    ) -> Result[Dict[str, Any], AppError]:
        """Retrieve campaign data with optional related data."""
        # Validate campaign ID
        if not campaign_id:
            return Err(validation_error("Campaign ID cannot be empty"))
        
        # Fetch from database
        fetch_result = await self._fetch_campaign(campaign_id)
        if isinstance(fetch_result, Err):
            return fetch_result
        
        campaign_data = fetch_result.value
        
        # Include related data if requested
        if include_related:
            related_result = await self._fetch_related_data(campaign_id)
            if isinstance(related_result, Ok):
                campaign_data.update(related_result.value)
        
        return Ok(campaign_data)
    
    async def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any]
    ) -> Result[Campaign, AppError]:
        """Update campaign with validation."""
        # Get existing campaign
        existing = await self.get_campaign(campaign_id)
        if isinstance(existing, Err):
            return existing
        
        # Validate updates
        if 'name' in updates and not updates['name']:
            return Err(validation_error("Campaign name cannot be empty"))
        
        # Apply updates
        campaign_dict = existing.value
        campaign_dict.update(updates)
        campaign_dict['updated_at'] = datetime.utcnow().isoformat()
        
        # Create updated campaign object
        try:
            campaign = Campaign.from_dict(campaign_dict)
        except Exception as e:
            return Err(validation_error(f"Invalid campaign data: {str(e)}"))
        
        # Store updates
        storage_result = await self._store_campaign(campaign)
        if isinstance(storage_result, Err):
            return storage_result
        
        return Ok(campaign)
    
    async def add_character(
        self,
        campaign_id: str,
        character_data: Dict[str, Any]
    ) -> Result[Character, AppError]:
        """Add a character to campaign with validation."""
        # Verify campaign exists
        campaign_result = await self.get_campaign(campaign_id)
        if isinstance(campaign_result, Err):
            return campaign_result
        
        # Validate character data
        if not character_data.get('name'):
            return Err(validation_error("Character name is required"))
        
        # Create character
        character = Character(
            id=str(uuid.uuid4()),
            campaign_id=campaign_id,
            **character_data
        )
        
        # Store character
        storage_result = await self._store_character(character)
        if isinstance(storage_result, Err):
            return storage_result
        
        return Ok(character)
    
    @with_result(AppError)
    async def _store_campaign(self, campaign: Campaign) -> None:
        """Store campaign in database (internal method)."""
        await self.db.add_document(
            collection=self.collection_name,
            document_id=campaign.id,
            content=campaign.to_dict(),
            metadata={
                "type": "campaign",
                "system": campaign.system,
                "created_at": campaign.created_at.isoformat(),
                "updated_at": campaign.updated_at.isoformat()
            }
        )
    
    async def _fetch_campaign(self, campaign_id: str) -> Result[Dict[str, Any], AppError]:
        """Fetch campaign from database."""
        try:
            results = await self.db.get_document(
                collection=self.collection_name,
                document_id=campaign_id
            )
            
            if not results:
                return Err(not_found("Campaign", campaign_id))
            
            return Ok(results[0])
        except Exception as e:
            logger.error(f"Database error fetching campaign: {e}")
            return Err(database_error(f"Failed to fetch campaign: {str(e)}"))
    
    async def _fetch_related_data(
        self, 
        campaign_id: str
    ) -> Result[Dict[str, Any], AppError]:
        """Fetch related campaign data."""
        related_data = {
            "characters": [],
            "npcs": [],
            "locations": [],
            "sessions": []
        }
        
        try:
            # Fetch characters
            characters = await self.db.query_by_metadata(
                collection=self.collection_name,
                filters={"campaign_id": campaign_id, "type": "character"}
            )
            related_data["characters"] = characters
            
            # Fetch NPCs
            npcs = await self.db.query_by_metadata(
                collection=self.collection_name,
                filters={"campaign_id": campaign_id, "type": "npc"}
            )
            related_data["npcs"] = npcs
            
            # Fetch locations
            locations = await self.db.query_by_metadata(
                collection=self.collection_name,
                filters={"campaign_id": campaign_id, "type": "location"}
            )
            related_data["locations"] = locations
            
            return Ok(related_data)
        except Exception as e:
            logger.warning(f"Error fetching related data: {e}")
            # Propagate the error instead of returning partial data as a success.
            # The caller can decide how to handle the partial data if needed.
            return Err(database_error(f"Failed to fetch related data: {str(e)}", partial_data=related_data))

    
    async def _store_character(self, character: Character) -> Result[None, AppError]:
        """Store character in database."""
        try:
            await self.db.add_document(
                collection=self.collection_name,
                document_id=character.id,
                content=character.to_dict(),
                metadata={
                    "type": "character",
                    "campaign_id": character.campaign_id,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to store character: {e}")
            return Err(database_error(f"Failed to store character: {str(e)}"))
```

### 4. Search Service with Result Pattern

```python
# src/search/search_service.py
from typing import List, Dict, Any, Optional
import asyncio
import logging

from src.core.result import Result, Ok, Err, collect_async_results
from src.core.errors import AppError, validation_error, database_error
from src.search.hybrid_search import HybridSearch
from src.search.query_processor import QueryProcessor
from src.search.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class SearchService:
    """Search service with Result-based error handling."""
    
    def __init__(
        self,
        hybrid_search: HybridSearch,
        query_processor: QueryProcessor,
        cache_manager: CacheManager
    ):
        self.hybrid_search = hybrid_search
        self.query_processor = query_processor
        self.cache_manager = cache_manager
    
    async def search(
        self,
        query: str,
        rulebook: Optional[str] = None,
        source_type: Optional[str] = None,
        max_results: int = 5,
        use_hybrid: bool = True
    ) -> Result[Dict[str, Any], AppError]:
        """Perform search with comprehensive error handling."""
        # Validate inputs
        if not query or not query.strip():
            return Err(validation_error("Search query cannot be empty"))
        
        if max_results < 1 or max_results > 100:
            return Err(validation_error("max_results must be between 1 and 100"))
        
        # Check cache
        cache_key = self._generate_cache_key(query, rulebook, source_type, max_results)
        cached = await self.cache_manager.get(cache_key)
        if isinstance(cached, Ok):
            logger.debug(f"Cache hit for query: {query}")
            return cached
        
        # Process query
        processed_query = await self.query_processor.process(query)
        if isinstance(processed_query, Err):
            return processed_query
        
        # Perform search
        if use_hybrid:
            search_result = await self._hybrid_search(
                processed_query.value,
                rulebook,
                source_type,
                max_results
            )
        else:
            search_result = await self._vector_search(
                processed_query.value,
                rulebook,
                source_type,
                max_results
            )
        
        if isinstance(search_result, Err):
            return search_result
        
        # Cache result
        await self.cache_manager.set(cache_key, search_result.value)
        
        return search_result
    
    async def _hybrid_search(
        self,
        query: str,
        rulebook: Optional[str],
        source_type: Optional[str],
        max_results: int
    ) -> Result[Dict[str, Any], AppError]:
        """Perform hybrid search combining vector and keyword search."""
        # Run searches in parallel
        results = await collect_async_results([
            self._vector_search(query, rulebook, source_type, max_results * 2),
            self._keyword_search(query, rulebook, source_type, max_results * 2)
        ])
        
        if isinstance(results, Err):
            return results
        
        vector_results, keyword_results = results.value
        
        # Merge and rerank results
        merged = self._merge_results(vector_results, keyword_results)
        reranked = self._rerank_results(merged, query)
        
        # Format response
        return Ok({
            "query": query,
            "results": reranked[:max_results],
            "total_results": len(reranked),
            "search_type": "hybrid",
            "metadata": {
                "rulebook": rulebook,
                "source_type": source_type,
                "vector_count": len(vector_results),
                "keyword_count": len(keyword_results)
            }
        })
    
    async def _vector_search(
        self,
        query: str,
        rulebook: Optional[str],
        source_type: Optional[str],
        max_results: int
    ) -> Result[List[Dict[str, Any]], AppError]:
        """Perform vector similarity search."""
        try:
            results = await self.hybrid_search.vector_search(
                query=query,
                filters={
                    "rulebook": rulebook,
                    "source_type": source_type
                } if rulebook or source_type else None,
                top_k=max_results
            )
            return Ok(results)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return Err(database_error(f"Vector search failed: {str(e)}"))
    
    async def _keyword_search(
        self,
        query: str,
        rulebook: Optional[str],
        source_type: Optional[str],
        max_results: int
    ) -> Result[List[Dict[str, Any]], AppError]:
        """Perform keyword-based search."""
        try:
            results = await self.hybrid_search.keyword_search(
                query=query,
                filters={
                    "rulebook": rulebook,
                    "source_type": source_type
                } if rulebook or source_type else None,
                top_k=max_results
            )
            return Ok(results)
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return Err(database_error(f"Keyword search failed: {str(e)}"))
    
    def _merge_results(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict]
    ) -> List[Dict]:
        """Merge vector and keyword search results."""
        seen = set()
        merged = []
        
        # Add vector results with higher weight
        for result in vector_results:
            result_id = result.get('id')
            if result_id not in seen:
                seen.add(result_id)
                result['score_weight'] = 0.7
                merged.append(result)
        
        # Add keyword results
        for result in keyword_results:
            result_id = result.get('id')
            if result_id not in seen:
                seen.add(result_id)
                result['score_weight'] = 0.3
                merged.append(result)
            else:
                # Boost score if in both results
                for m in merged:
                    if m.get('id') == result_id:
                        m['score_weight'] = 1.0
                        break
        
        return merged
    
    def _rerank_results(
        self,
        results: List[Dict],
        query: str
    ) -> List[Dict]:
        """Rerank results based on relevance."""
        # Simple reranking by weighted score
        for result in results:
            base_score = result.get('score', 0.5)
            weight = result.get('score_weight', 1.0)
            result['final_score'] = base_score * weight
        
        return sorted(results, key=lambda x: x['final_score'], reverse=True)
    
    def _generate_cache_key(
        self,
        query: str,
        rulebook: Optional[str],
        source_type: Optional[str],
        max_results: int
    ) -> str:
        """Generate cache key for search parameters."""
        parts = [
            f"search:{query}",
            f"book:{rulebook or 'all'}",
            f"type:{source_type or 'all'}",
            f"limit:{max_results}"
        ]
        return ":".join(parts)
```

### 5. MCP Tools with Result Pattern

```python
# src/main.py
from mcp.server.fastmcp import FastMCP
import logging
import asyncio
from typing import Dict, Any

from src.core.database import ChromaDBManager
from src.core.result import Result
from src.campaign.campaign_manager import CampaignManager
from src.search.search_service import SearchService

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("TTRPG")

# Initialize services
db = ChromaDBManager()
campaign_manager = CampaignManager(db)
search_service = SearchService(...)

@mcp.tool()
async def search(
    query: str,
    rulebook: str = None,
    source_type: str = None,
    max_results: int = 5,
    use_hybrid: bool = True
) -> Dict[str, Any]:
    """Search with Result pattern error handling."""
    result = await search_service.search(
        query=query,
        rulebook=rulebook,
        source_type=source_type,
        max_results=max_results,
        use_hybrid=use_hybrid
    )
    
    if result.is_ok():
        return {
            "success": True,
            **result.unwrap()
        }
    else:
        error = result.error
        return {
            "success": False,
            "error": str(error),
            "details": error.to_dict() if hasattr(error, 'to_dict') else {}
        }

@mcp.tool()
async def create_campaign(
    name: str,
    system: str,
    description: str = None
) -> Dict[str, Any]:
    """Create campaign with proper error handling."""
    result = await campaign_manager.create_campaign(
        name=name,
        system=system,
        description=description
    )
    
    if result.is_ok():
        campaign = result.unwrap()
        return {
            "success": True,
            "message": f"Campaign '{campaign.name}' created successfully",
            "id": campaign.id,
            "data": campaign.to_dict()
        }
    else:
        error = result.error
        return {
            "success": False,
            "error": str(error),
            "details": error.to_dict() if hasattr(error, 'to_dict') else {}
        }

@mcp.tool()
async def get_campaign_data(
    campaign_id: str,
    include_related: bool = False
) -> Dict[str, Any]:
    """Get campaign data with Result pattern."""
    result = await campaign_manager.get_campaign(
        campaign_id=campaign_id,
        include_related=include_related
    )
    
    if result.is_ok():
        return {
            "success": True,
            "data": result.unwrap(),
            "campaign_id": campaign_id
        }
    else:
        error = result.error
        return {
            "success": False,
            "error": str(error),
            "campaign_id": campaign_id,
            "details": error.to_dict() if hasattr(error, 'to_dict') else {}
        }

if __name__ == "__main__":
    # Run the MCP server
    asyncio.run(mcp.run())
```

## Testing with Result Pattern

### Unit Testing

```python
# tests/test_campaign_manager.py
import pytest
from unittest.mock import Mock, AsyncMock

from src.campaign.campaign_manager import CampaignManager
from src.core.result import Ok, Err
from src.core.errors import validation_error, not_found

@pytest.mark.asyncio
async def test_create_campaign_success():
    """Test successful campaign creation."""
    mock_db = Mock()
    mock_db.add_document = AsyncMock(return_value=None)
    
    manager = CampaignManager(mock_db)
    result = await manager.create_campaign(
        name="Test Campaign",
        system="D&D 5e",
        description="A test campaign"
    )
    
    assert result.is_ok()
    campaign = result.unwrap()
    assert campaign.name == "Test Campaign"
    assert campaign.system == "D&D 5e"
    mock_db.add_document.assert_called_once()

@pytest.mark.asyncio
async def test_create_campaign_validation_error():
    """Test campaign creation with invalid data."""
    mock_db = Mock()
    manager = CampaignManager(mock_db)
    
    # Test empty name
    result = await manager.create_campaign(
        name="",
        system="D&D 5e"
    )
    
    assert result.is_err()
    assert result.error.kind.value == "validation"
    assert "name cannot be empty" in str(result.error)

@pytest.mark.asyncio
async def test_get_campaign_not_found():
    """Test getting non-existent campaign."""
    mock_db = Mock()
    mock_db.get_document = AsyncMock(return_value=[])
    
    manager = CampaignManager(mock_db)
    result = await manager.get_campaign("non-existent-id")
    
    assert result.is_err()
    assert result.error.kind.value == "not_found"

@pytest.mark.asyncio
async def test_result_chaining():
    """Test Result chaining operations."""
    mock_db = Mock()
    mock_db.get_document = AsyncMock(return_value=[{
        "id": "test-id",
        "name": "Test Campaign",
        "system": "D&D 5e"
    }])
    
    manager = CampaignManager(mock_db)
    result = await manager.get_campaign("test-id")
    
    # Chain operations on Result
    name_result = result.map(lambda data: data.get("name"))
    
    assert name_result.is_ok()
    assert name_result.unwrap() == "Test Campaign"
```

## Migration Strategy

### Phase 1: Core Infrastructure (Week 1)
1. Implement Result type and error types
2. Update core database layer with Result pattern
3. Create helper utilities and decorators

### Phase 2: Service Layer (Week 2)
1. Update CampaignManager with Result pattern
2. Migrate SearchService to Result pattern
3. Update SessionManager and CharacterGenerator

### Phase 3: MCP Integration (Week 3)
1. Update all MCP tool handlers
2. Ensure consistent error responses
3. Add Result-aware middleware

### Phase 4: Testing & Documentation (Week 4)
1. Write comprehensive tests for Result pattern
2. Update API documentation
3. Create migration guide for developers

## Benefits of Result Pattern

### 1. Explicit Error Handling
- No hidden exceptions
- All errors are part of the type signature
- Compiler/type checker ensures error handling

### 2. Composability
- Chain operations with map/flat_map
- Collect multiple Results easily
- Build complex pipelines safely

### 3. Better Testing
- Test success and error paths explicitly
- No need for exception assertions
- Clearer test intent

### 4. Performance
- No exception overhead
- Predictable control flow
- Better for async operations

### 5. Documentation
- Errors are self-documenting in function signatures
- Clear success/failure paths
- Easier to understand API contracts

## Conclusion

The migration to Result pattern with modern Python dependencies provides:
- More reliable error handling
- Better type safety
- Improved testability
- Clearer API contracts
- Enhanced performance

This approach aligns perfectly with the SvelteKit frontend migration, providing consistent error handling patterns across the full stack.