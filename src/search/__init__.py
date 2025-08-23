"""Search system components for TTRPG MCP server."""

from src.search.cache_manager import LRUCache, SearchCacheManager
from src.search.error_handler import (
    CacheError,
    DatabaseError,
    EmbeddingGenerationError,
    ErrorRecovery,
)
from src.search.error_handler import IndexError as SearchIndexError
from src.search.error_handler import (
    QueryProcessingError,
    SearchError,
    SearchValidator,
    handle_search_errors,
)
from src.search.hybrid_search import HybridSearchEngine, SearchResult
from src.search.index_persistence import IndexPersistence
from src.search.query_clarification import QueryClarificationService
from src.search.query_completion import (
    QueryCompletionEngine,
    QueryCompletionService,
    QueryPatternMatcher,
)
from src.search.query_processor import QueryProcessor
from src.search.search_analytics import SearchAnalytics, SearchMetrics
from src.search.search_service import SearchService

__all__ = [
    # Main service
    "SearchService",
    # Core components
    "HybridSearchEngine",
    "SearchResult",
    "QueryProcessor",
    "SearchCacheManager",
    "LRUCache",
    "IndexPersistence",
    # New Phase 3.5 components
    "QueryClarificationService",
    "SearchAnalytics",
    "SearchMetrics",
    "QueryCompletionEngine",
    "QueryCompletionService",
    "QueryPatternMatcher",
    # Error handling
    "SearchError",
    "QueryProcessingError",
    "EmbeddingGenerationError",
    "DatabaseError",
    "CacheError",
    "SearchIndexError",
    "handle_search_errors",
    "SearchValidator",
    "ErrorRecovery",
]
