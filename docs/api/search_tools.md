# Search Tools API Documentation

## Overview

The search tools provide comprehensive search capabilities across TTRPG content with both semantic and keyword matching.

## Tools

### `search`

Performs hybrid search across TTRPG content combining semantic similarity and keyword matching.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | The search query |
| `rulebook` | string | No | null | Specific rulebook to search in |
| `source_type` | string | No | null | Filter by source type: "rulebook" or "flavor" |
| `content_type` | string | No | null | Filter by content type: "rule", "spell", "monster", etc. |
| `max_results` | integer | No | 5 | Maximum number of results to return |
| `use_hybrid` | boolean | No | true | Whether to use hybrid search (semantic + keyword) |

#### Response

```json
{
  "success": true,
  "query": "fireball spell damage",
  "results": [
    {
      "content": "Fireball: A bright streak flashes from your pointing finger...",
      "source": "Player's Handbook",
      "page": 241,
      "section": "Spells",
      "relevance_score": 0.95,
      "metadata": {
        "spell_level": 3,
        "school": "evocation",
        "content_type": "spell"
      }
    }
  ],
  "total_results": 12,
  "search_metadata": {
    "search_type": "hybrid",
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "processing_time_ms": 45
  }
}
```

#### Examples

##### Basic Search
```python
result = await mcp.search(
    query="how does initiative work",
    max_results=3
)
```

##### Filtered Search
```python
result = await mcp.search(
    query="dragon",
    content_type="monster",
    rulebook="Monster Manual",
    max_results=10
)
```

##### Keyword-Only Search
```python
result = await mcp.search(
    query="AC 15",
    use_hybrid=False,
    max_results=5
)
```

### Advanced Search Features

#### Query Expansion

The search system automatically expands queries using:
- Synonym detection
- Related term inclusion
- Semantic similarity expansion

#### Result Ranking

Results are ranked based on:
1. Semantic similarity score (0.0-1.0)
2. Keyword match score (BM25)
3. Source reliability weight
4. Content type relevance

#### Caching

- Frequent queries are cached with LRU eviction
- Cache TTL: 15 minutes
- Cache size: 1000 queries

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `InvalidQueryError` | Empty or invalid query | Provide a non-empty query string |
| `SourceNotFoundError` | Specified rulebook doesn't exist | Check available sources with `list_sources()` |
| `SearchTimeoutError` | Search took too long | Reduce max_results or simplify query |

### Error Response Example

```json
{
  "success": false,
  "error": "Source not found: 'Invalid Rulebook'",
  "details": {
    "available_sources": ["Player's Handbook", "Monster Manual", "Dungeon Master's Guide"],
    "suggestion": "Did you mean 'Player's Handbook'?"
  }
}
```

## Performance Metrics

- Average response time: < 100ms
- 95th percentile: < 200ms
- Maximum results: 100
- Concurrent searches: Unlimited (local)

## Best Practices

1. **Use specific queries**: More specific queries yield better results
2. **Leverage filters**: Use source_type and content_type to narrow results
3. **Optimize max_results**: Request only as many results as needed
4. **Cache considerations**: Repeated queries benefit from caching
5. **Hybrid search**: Use hybrid search for best accuracy

## Related Tools

- [`list_sources`](./source_management.md#list_sources) - List available sources
- [`add_source`](./source_management.md#add_source) - Add new content sources