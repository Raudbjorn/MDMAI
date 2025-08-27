# Phase 17: Context Management System

## Overview

The Context Management System is a comprehensive solution for managing conversation contexts, session states, and collaborative environments in the TTRPG Assistant. It provides enterprise-grade features including real-time synchronization, multi-provider translation, versioning with delta compression, and distributed state management.

## Architecture

### Core Components

#### 1. **Context Persistence Layer** (`persistence.py`)
- PostgreSQL-based storage with partitioning support
- Connection pooling with async operations
- Optimized indexing and materialized views
- Automatic cleanup and retention policies
- Performance tracking and query optimization

**Features:**
- Table partitioning by context type
- GIN indexes for JSON queries
- Full-text search support
- Materialized views for statistics
- Configurable retention policies

#### 2. **State Synchronization** (`synchronization.py`)
- Redis-based event bus for distributed updates
- Real-time WebSocket synchronization
- Optimistic locking implementation
- 5 conflict resolution strategies
- Cache coherence protocol

**Conflict Resolution Strategies:**
- `LATEST_WINS`: Most recent update wins
- `MANUAL_MERGE`: Requires user intervention
- `PROVIDER_PRIORITY`: Provider-specific rules
- `USER_CHOICE`: User selects resolution
- `AUTOMATIC_MERGE`: Smart merge of non-conflicting changes

#### 3. **Context Translation** (`translation.py`)
- Provider-specific adapters (Anthropic, OpenAI, Google)
- Seamless context migration between providers
- Format validation and integrity checking
- Fallback strategies for failures
- Round-trip conversion support

**Supported Providers:**
- Anthropic Claude (100k token window)
- OpenAI GPT-4 (128k token window)
- Google Gemini (1M token window)

#### 4. **Serialization & Compression** (`serialization.py`)
- Multiple serialization formats (JSON, MessagePack, Pickle)
- 4 compression algorithms (GZIP, LZ4, Zstandard, Brotli)
- Automatic algorithm selection
- Async operations with thread pooling
- Batch processing support

#### 5. **Validation System** (`validation.py`)
- JSON schema validation
- Semantic consistency checks
- Data integrity verification
- Auto-correction capabilities
- Custom validation rules

#### 6. **Versioning System** (`versioning.py`)
- Delta compression for efficient storage
- Branch support for parallel development
- Smart storage strategies (incremental, full snapshot, hybrid)
- Version history with rollback
- Automatic cleanup of old versions

## Data Models

### Context Types

#### Base Context
```python
Context(
    context_id: str,
    context_type: ContextType,
    title: str,
    description: str,
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    state: ContextState,
    owner_id: Optional[str],
    collaborators: List[str],
    permissions: Dict[str, List[str]]
)
```

#### Conversation Context
- Manages conversation threads
- Message history with roles
- Turn tracking
- Provider-specific settings
- Model parameters

#### Session Context
- User session management
- Active conversations tracking
- User preferences
- UI state persistence
- Activity metrics

#### Collaborative Context
- Multi-user sessions
- Real-time synchronization
- Locking mechanisms
- Participant management
- Conflict resolution

## Usage Examples

### Basic Context Creation

```python
from src.context.context_manager import ContextManager
from src.context.models import ConversationContext

# Initialize context manager
context_manager = ContextManager(
    database_url="postgresql://localhost/ttrpg",
    redis_url="redis://localhost:6379/0"
)
await context_manager.initialize()

# Create a conversation context
context = ConversationContext(
    title="D&D Campaign Session 1",
    messages=[
        {"role": "system", "content": "You are a helpful D&D game master."},
        {"role": "user", "content": "I want to create a new character."}
    ],
    primary_provider="anthropic",
    model_parameters={"temperature": 0.7}
)

context_id = await context_manager.create_context(
    context, 
    user_id="player123"
)
```

### Real-time Collaboration

```python
from src.context.models import CollaborativeContext

# Create collaborative context
collab_context = CollaborativeContext(
    title="Party Planning Session",
    room_id="room_abc123",
    active_participants=["player1", "player2", "dm"]
)

# Join session
await context_manager.join_collaborative_session(
    context_id, 
    user_id="player3"
)

# Acquire lock for editing
await context_manager.acquire_context_lock(
    context_id, 
    user_id="player1"
)

# Update shared state
await context_manager.update_context(
    context_id,
    {"shared_state": {"quest": "Dragon's Lair", "level": 5}},
    user_id="player1",
    sync=True  # Real-time sync
)
```

### Provider Migration

```python
from src.ai_providers.models import ProviderType

# Migrate context between providers
await context_manager.migrate_context(
    context_id,
    from_provider=ProviderType.OPENAI,
    to_provider=ProviderType.ANTHROPIC,
    options={"preserve_history": True}
)
```

### Version Management

```python
# Create a version checkpoint
version = await context_manager.create_version(
    context_id,
    user_id="dm",
    message="Before major plot twist",
    tags=["checkpoint", "pre-battle"]
)

# Rollback to previous version
await context_manager.rollback_to_version(
    context_id,
    version_id=version.version_id,
    user_id="dm"
)

# Get version history
history = await context_manager.get_version_history(
    context_id,
    limit=10
)
```

### Context Validation

```python
from src.context.validation import ContextValidator

validator = ContextValidator(enable_auto_correction=True)

# Validate context
result = await validator.validate_context(context)

if not result.is_valid:
    for issue in result.issues:
        print(f"{issue.severity}: {issue.message}")
        if issue.suggested_fix:
            print(f"  Fix: {issue.suggested_fix}")

# Apply auto-corrections if available
if result.corrected_data:
    context = result.corrected_data
```

## Configuration

### Environment Variables

```bash
# Database Configuration
CONTEXT_DATABASE_URL=postgresql://user:pass@localhost:5432/ttrpg_context
CONTEXT_REDIS_URL=redis://localhost:6379/0

# Feature Flags
CONTEXT_ENABLE_REAL_TIME_SYNC=true
CONTEXT_ENABLE_COMPRESSION=true
CONTEXT_ENABLE_VALIDATION=true
CONTEXT_ENABLE_VERSIONING=true
CONTEXT_ENABLE_TRANSLATION=true

# Performance Settings
CONTEXT_MAX_SIZE_MB=10
CONTEXT_MAX_VERSIONS=100
CONTEXT_CACHE_TTL=3600
CONTEXT_SYNC_TIMEOUT=30

# Storage Settings
CONTEXT_COMPRESSION_ALGORITHM=zstd
CONTEXT_SERIALIZATION_FORMAT=msgpack

# Collaborative Features
CONTEXT_MAX_PARTICIPANTS=50
CONTEXT_LOCK_TIMEOUT=300

# Cleanup Settings
CONTEXT_AUTO_CLEANUP=true
CONTEXT_AUTO_ARCHIVE_DAYS=90
CONTEXT_CLEANUP_INTERVAL=24
```

### Database Schema

```sql
-- Main contexts table with partitioning
CREATE TABLE contexts (
    context_id UUID PRIMARY KEY,
    context_type VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB NOT NULL,
    state VARCHAR(20) NOT NULL,
    owner_id TEXT,
    collaborators TEXT[],
    permissions JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    last_modified TIMESTAMPTZ NOT NULL,
    last_accessed TIMESTAMPTZ NOT NULL
) PARTITION BY LIST (context_type);

-- Create partitions for each context type
CREATE TABLE contexts_conversation PARTITION OF contexts
    FOR VALUES IN ('conversation');
CREATE TABLE contexts_session PARTITION OF contexts
    FOR VALUES IN ('session');
CREATE TABLE contexts_collaborative PARTITION OF contexts
    FOR VALUES IN ('collaborative');

-- Indexes for optimal performance
CREATE INDEX idx_contexts_type_state ON contexts(context_type, state);
CREATE INDEX idx_contexts_owner ON contexts(owner_id);
CREATE INDEX idx_contexts_metadata ON contexts USING GIN(metadata);
CREATE INDEX idx_contexts_data ON contexts USING GIN(data);
CREATE INDEX idx_contexts_text_search ON contexts 
    USING GIN(to_tsvector('english', title || ' ' || description));
```

## Performance Optimization

### Caching Strategy
- In-memory cache for frequently accessed contexts
- Redis cache for distributed caching
- Translation cache for provider conversions
- Version cache for recent versions

### Compression Benefits
- 60-80% reduction in storage size with Zstandard
- Automatic compression for large contexts
- Smart algorithm selection based on data type
- Async compression to avoid blocking

### Query Optimization
- Prepared statements for common queries
- Connection pooling with overflow handling
- Batch operations for bulk updates
- Materialized views for statistics

## Integration Points

### Bridge Integration
- Automatic session context creation
- Message tracking and history
- MCP protocol translation
- WebSocket event handling

### AI Provider Integration
- Request/response hooks
- Context synchronization
- Provider-specific optimizations
- Fallback handling

### Security Integration
- Access control enforcement
- Permission validation
- Audit logging
- Rate limiting support

## Monitoring & Metrics

### Performance Metrics
```python
stats = await context_manager.get_performance_stats()

# Output includes:
# - Query execution times
# - Cache hit rates
# - Compression ratios
# - Synchronization latencies
# - Version creation rates
```

### Health Checks
```python
health = await context_manager.health_check()

# Checks:
# - Database connectivity
# - Redis connectivity
# - Pool utilization
# - Storage usage
# - Active sync operations
```

## Error Handling

### Automatic Recovery
- Connection retry with exponential backoff
- Fallback to cache on database errors
- Graceful degradation of features
- Transaction rollback on failures

### Conflict Resolution
- Automatic merge for non-conflicting changes
- User notification for manual resolution
- Version branching for parallel work
- Audit trail of resolutions

## Best Practices

### Context Lifecycle
1. Create context with appropriate type
2. Validate before major operations
3. Version important checkpoints
4. Archive inactive contexts
5. Clean up expired contexts

### Collaboration Guidelines
1. Use locks for exclusive edits
2. Implement optimistic concurrency
3. Handle conflicts gracefully
4. Sync frequently for real-time updates
5. Monitor participant activity

### Performance Tips
1. Enable compression for large contexts
2. Use batch operations when possible
3. Implement pagination for queries
4. Cache frequently accessed data
5. Monitor and optimize slow queries

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
pytest tests/test_context_management_complete.py -v

# Integration tests
pytest tests/test_context_integration.py -v

# Performance tests
pytest tests/test_context_performance.py -v

# Load tests
locust -f tests/load_test_context.py
```

## Troubleshooting

### Common Issues

#### High Memory Usage
- Check compression settings
- Review cache size limits
- Monitor connection pool size
- Analyze query patterns

#### Slow Synchronization
- Verify Redis connectivity
- Check network latency
- Review event bus configuration
- Optimize message size

#### Version Conflicts
- Review conflict resolution strategy
- Check lock timeouts
- Analyze concurrent access patterns
- Implement retry logic

## Future Enhancements

### Planned Features
- GraphQL API for context queries
- Machine learning-based conflict resolution
- Distributed caching with Redis Cluster
- Context templates and presets
- Advanced analytics and insights

### Optimization Opportunities
- Columnar storage for analytics
- GPU-accelerated compression
- Predictive caching
- Query plan optimization
- Horizontal scaling support

## Support

For issues or questions:
- Check the [troubleshooting guide](#troubleshooting)
- Review [integration examples](#integration-points)
- Contact the development team
- Submit issues to the project repository

## License

Copyright (c) 2024 TTRPG Assistant Project. All rights reserved.