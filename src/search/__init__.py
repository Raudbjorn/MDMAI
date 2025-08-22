"""Search system components for TTRPG MCP server."""

from src.search.search_service import SearchService
from src.search.hybrid_search import HybridSearchEngine, SearchResult
from src.search.query_processor import QueryProcessor
from src.search.cache_manager import SearchCacheManager, LRUCache
from src.search.error_handler import (
    SearchError,
    QueryProcessingError,
    EmbeddingGenerationError,
    DatabaseError,
    CacheError,
    IndexError as SearchIndexError,
    handle_search_errors,
    SearchValidator,
    ErrorRecovery
)
from src.search.index_persistence import SearchIndexPersistence
from src.search.query_clarification import QueryClarificationService
from src.search.search_analytics import SearchAnalytics, SearchMetrics
from src.search.query_completion import (
    QueryCompletionEngine,
    QueryCompletionService,
    QueryPatternMatcher
)

__all__ = [
    # Main service
    "SearchService",
    
    # Core components
    "HybridSearchEngine",
    "SearchResult",
    "QueryProcessor",
    "SearchCacheManager",
    "LRUCache",
    "SearchIndexPersistence",
    
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