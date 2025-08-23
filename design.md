# TTRPG Assistant MCP Server - Design Document

## System Architecture

### Overview
The TTRPG Assistant is built as a Model Context Protocol (MCP) server that operates locally via stdin/stdout communication. It provides a comprehensive assistant for Dungeon Masters and Game Runners with fast access to rules, campaign data, and AI-powered content generation.

### Implementation Phases
The project follows a phased approach to ensure incremental delivery and testing:

1. **Phase 1**: Core Infrastructure - Database setup, basic MCP server
2. **Phase 2**: PDF Processing - Content extraction, chunking, indexing
3. **Phase 3**: Search System - Hybrid search, query processing, caching
4. **Phase 4**: Personality System - Style extraction and application
5. **Phase 5**: Campaign Management - CRUD operations, versioning, linking
6. **Phase 6**: Session Management - Game session tracking, initiative
7. **Phase 7**: Character Generation - PC/NPC generation with personalities
8. **Phase 8**: Testing & Optimization - Performance tuning, test coverage

Each phase builds on the previous ones, with clear interfaces between components.

### Core Components

#### 1. MCP Server Layer
- **Framework**: FastMCP for Python
- **Communication**: stdin/stdout (local operations)
- **Protocol**: Model Context Protocol standard
- **Tools**: Exposed as decorated async functions

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("TTRPG")
```

**IMPORTANT: MCP Server Singleton Pattern**
- There must be ONLY ONE FastMCP instance in the entire application
- Initialize it in main.py and pass it to modules that need to register tools
- DO NOT create separate FastMCP instances in module files
- Use a registration pattern for module tools:

```python
# In module files (e.g., campaign/mcp_tools.py)
def register_campaign_tools(mcp_server):
    """Register tools with the main MCP server."""
    mcp_server.tool()(create_campaign)
    mcp_server.tool()(get_campaign_data)
    # ... register other tools

# In main.py
from src.campaign import register_campaign_tools
mcp = FastMCP("TTRPG")
register_campaign_tools(mcp)
```

#### 2. Database Layer
- **Primary Database**: ChromaDB (embedded vector database)
- **Storage Strategy**: 
  - Vector embeddings for semantic search
  - Metadata storage for structured queries
  - Document chunking with overlap for context preservation
- **Collections**:
  - `rulebooks`: Game system rules and mechanics
  - `flavor_sources`: Novels, lore, and narrative content
  - `campaigns`: Active campaign data
  - `sessions`: Game session tracking
  - `personalities`: System-specific personality profiles

#### 3. PDF Processing Pipeline
- **Extraction**: PyPDF2 or pdfplumber for text extraction
- **Structure Preservation**:
  - Table detection and formatting
  - Section hierarchy maintenance
  - Index/glossary parsing for metadata
- **Adaptive Learning**:
  - Pattern recognition for content types
  - Template caching for similar documents
  - Progressive improvement through usage

#### 4. Search Engine
- **Hybrid Search Architecture**:
  - Vector similarity search (semantic)
  - BM25 keyword search (exact matching)
  - Weighted combination of results
- **Query Processing**:
  - Query expansion and suggestions
  - Spell correction
  - Context-aware filtering

#### 5. Personality Engine
- **Profile Extraction**:
  - NLP analysis of source material
  - Tone and style classification
  - Vocabulary extraction
- **Profile Application**:
  - Response templating
  - Vocabulary injection
  - Style consistency enforcement

## Data Models

### Source Document
```python
class SourceDocument:
    id: str
    title: str
    system: str  # e.g., "D&D 5e", "Call of Cthulhu"
    source_type: str  # "rulebook" or "flavor"
    content_chunks: List[ContentChunk]
    metadata: Dict
    personality_profile: PersonalityProfile
```

### Content Chunk
```python
class ContentChunk:
    id: str
    source_id: str
    content: str
    page_number: int
    section: str
    chunk_type: str  # "rule", "table", "narrative", etc.
    embedding: List[float]
    metadata: Dict
```

### Campaign Data
```python
class Campaign:
    id: str
    name: str
    system: str
    characters: List[Character]
    npcs: List[NPC]
    locations: List[Location]
    plot_points: List[PlotPoint]
    sessions: List[Session]
    created_at: datetime
    updated_at: datetime
```

### Session Data
```python
class Session:
    id: str
    campaign_id: str
    date: datetime
    notes: List[str]
    initiative_order: List[InitiativeEntry]
    monsters: List[Monster]
    status: str  # "planned", "active", "completed"
```

### Personality Profile
```python
class PersonalityProfile:
    system: str
    tone: str  # "authoritative", "mysterious", "scholarly"
    perspective: str  # "omniscient", "first-person", "instructional"
    style_descriptors: List[str]
    common_phrases: List[str]
    vocabulary: Dict[str, float]  # term -> frequency
```

## MCP Tool Interfaces

### Search Tool
```python
@mcp.tool()
async def search(
    query: str, 
    rulebook: str = None, 
    source_type: str = None,  # "rulebook" or "flavor"
    content_type: str = None,  # "rule", "spell", "monster", etc.
    max_results: int = 5,
    use_hybrid: bool = True
) -> Dict[str, Any]:
    """
    Search across TTRPG content with semantic and keyword matching
    """
```

### Source Management
```python
@mcp.tool()
async def add_source(
    pdf_path: str,
    rulebook_name: str,
    system: str,
    source_type: str = "rulebook"
) -> Dict[str, str]:
    """
    Add a new PDF source to the knowledge base
    """

@mcp.tool()
async def list_sources(
    system: str = None,
    source_type: str = None
) -> List[Dict[str, Any]]:
    """
    List available sources in the system
    """
```

### Campaign Management
```python
@mcp.tool()
async def create_campaign(
    name: str,
    system: str,
    description: str = None
) -> Dict[str, str]:
    """
    Create a new campaign
    """

@mcp.tool()
async def get_campaign_data(
    campaign_id: str,
    data_type: str = None  # "characters", "npcs", "locations", etc.
) -> Dict[str, Any]:
    """
    Retrieve campaign-specific data
    """

@mcp.tool()
async def update_campaign_data(
    campaign_id: str,
    data_type: str,
    data: Dict[str, Any]
) -> Dict[str, str]:
    """
    Update campaign information
    """
```

### Session Management
```python
@mcp.tool()
async def start_session(
    campaign_id: str,
    session_name: str = None
) -> Dict[str, str]:
    """
    Start a new game session
    """

@mcp.tool()
async def add_session_note(
    session_id: str,
    note: str
) -> Dict[str, str]:
    """
    Add a note to the current session
    """

@mcp.tool()
async def set_initiative(
    session_id: str,
    initiative_order: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    Set the initiative order for combat
    """

@mcp.tool()
async def update_monster_hp(
    session_id: str,
    monster_id: str,
    new_hp: int
) -> Dict[str, str]:
    """
    Update a monster's hit points
    """
```

### Character Generation
```python
@mcp.tool()
async def generate_character(
    system: str,
    level: int = 1,
    class_type: str = None,
    backstory_hints: str = None
) -> Dict[str, Any]:
    """
    Generate a player character with stats and backstory
    """

@mcp.tool()
async def generate_npc(
    system: str,
    role: str,  # "merchant", "guard", "noble", etc.
    level: int = None,
    personality_traits: List[str] = None
) -> Dict[str, Any]:
    """
    Generate an NPC with appropriate stats and personality
    """
```

### Personality Management
```python
@mcp.tool()
async def get_system_personality(
    system: str
) -> Dict[str, Any]:
    """
    Get the personality profile for a game system
    """

@mcp.tool()
async def set_active_personality(
    system: str
) -> Dict[str, str]:
    """
    Set the active personality for responses
    """
```

## Processing Pipelines

### PDF Import Pipeline
1. **Document Loading**: Load PDF and extract raw text
2. **Structure Analysis**: Identify sections, tables, indices
3. **Content Chunking**: Split into semantic chunks with overlap
4. **Metadata Extraction**: Extract page numbers, sections, content types
5. **Personality Analysis**: Analyze tone, style, vocabulary
6. **Embedding Generation**: Create vector embeddings for chunks
7. **Storage**: Store in ChromaDB with metadata
8. **Pattern Learning**: Update adaptive learning cache

### Search Pipeline
1. **Query Processing**: Clean and expand query
2. **Hybrid Search**:
   - Vector similarity search
   - Keyword search with BM25
   - Result merging with weights
3. **Reranking**: Apply relevance scoring
4. **Context Enhancement**: Add surrounding context
5. **Personality Application**: Apply system personality to response
6. **Result Formatting**: Structure results with citations

### Campaign Update Pipeline
1. **Validation**: Verify campaign exists and data is valid
2. **Version Creation**: Create snapshot of current state
3. **Update Application**: Apply changes to data
4. **Cross-Reference**: Update links to rulebook content
5. **Index Update**: Update search indices
6. **Confirmation**: Return success with version info

## Implementation Guidelines

### Project Structure
```
src/
├── main.py                 # Single MCP server instance, tool registration
├── core/
│   ├── __init__.py
│   └── database.py         # ChromaDB manager singleton
├── campaign/
│   ├── __init__.py        # Module exports
│   ├── models.py          # Data models (Campaign, Character, etc.)
│   ├── campaign_manager.py # Business logic
│   ├── rulebook_linker.py # Cross-referencing logic
│   └── mcp_tools.py       # Tool definitions (NO MCP instance)
├── search/
│   ├── __init__.py
│   ├── search_service.py  # Main search logic
│   ├── hybrid_search.py   # Search implementation
│   ├── query_processor.py # Query handling
│   ├── cache_manager.py   # Caching logic
│   └── error_handler.py   # Error handling utilities
├── personality/
│   ├── __init__.py
│   ├── personality_manager.py
│   ├── personality_extractor.py
│   └── response_generator.py
├── pdf_processing/
│   ├── __init__.py
│   ├── pipeline.py        # Processing pipeline
│   ├── pdf_parser.py      # PDF extraction
│   ├── content_chunker.py # Chunking logic
│   └── adaptive_learning.py
└── session/              # Future: Session management
    └── __init__.py
```

### Module Structure Best Practices

#### 1. Separation of Concerns
- **Models** (`models.py`): Data structures and validation
- **Managers** (`*_manager.py`): Business logic and operations
- **MCP Tools** (`mcp_tools.py`): Tool definitions and registration
- **Utilities** (`*_linker.py`, `*_helper.py`): Supporting functionality

#### 2. Dependency Injection
- Initialize dependencies in main.py
- Pass dependencies to modules via initialization functions
- Avoid global imports of stateful objects

```python
# Good: Dependency injection
def initialize_campaign_tools(db, campaign_manager, linker):
    global _db, _campaign_manager, _linker
    _db = db
    _campaign_manager = campaign_manager
    _linker = linker

# Bad: Direct import
from src.core.database import db  # Creates coupling
```

#### 3. Async/Await Consistency
- All MCP tools MUST be async functions
- Use `@handle_search_errors()` decorator for error handling
- Maintain async chain through the call stack

#### 4. Error Handling Pattern
```python
from src.search.error_handler import handle_search_errors, DatabaseError

@handle_search_errors()
async def campaign_operation():
    try:
        # Operation logic
        return {"success": True, "data": result}
    except SpecificError as e:
        logger.error(f"Operation failed: {str(e)}")
        raise DatabaseError(f"Operation failed: {str(e)}")
```

#### 5. Standardized Return Patterns
All MCP tools should follow consistent return patterns:

**Success Response:**
```python
{
    "success": True,
    "message": "Operation completed successfully",
    "data": {...},  # Optional: returned data
    "id": "...",    # Optional: created/modified entity ID
}
```

**Error Response:**
```python
{
    "success": False,
    "error": "Descriptive error message",
    "details": {...},  # Optional: additional error context
}
```

**Search Response:**
```python
{
    "success": True,
    "query": "original query",
    "results": [...],
    "total_results": 42,
    "metadata": {...},  # Optional: search metadata
}
```

#### 6. Data Model Patterns
- Use `@dataclass` for models with `to_dict()` and `from_dict()` methods
- Handle datetime serialization consistently
- Provide default values using `field(default_factory=...)`

```python
@dataclass
class Model:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Model':
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
```

## Technical Implementation Details

### Database Schema (ChromaDB Collections)

#### Rulebooks Collection
- **Documents**: Content chunks from rulebooks
- **Metadata**: 
  - `source_id`: Unique identifier for source
  - `page`: Page number
  - `section`: Section name
  - `content_type`: Type of content
  - `system`: Game system
- **Embeddings**: Vector representations of content

#### Campaigns Collection
- **Documents**: Campaign data serialized as JSON
- **Metadata**:
  - `campaign_id`: Unique identifier
  - `data_type`: Type of data stored
  - `version`: Version number
  - `created_at`: Timestamp
  - `updated_at`: Timestamp

### Adaptive Learning System
- **Pattern Cache**: Store learned patterns for content types
- **Classification Models**: Simple ML models for content classification
- **Performance Metrics**: Track accuracy and improve over time
- **Template Library**: Reusable parsing templates

### Error Handling
- **Graceful Degradation**: Fallback to simpler search if advanced fails
- **User Feedback**: Clear error messages with suggestions
- **Retry Logic**: Automatic retry for transient failures
- **Logging**: Comprehensive logging for debugging

### Performance Optimizations
- **Caching**: LRU cache for frequent queries
- **Batch Processing**: Process multiple chunks in parallel
- **Index Optimization**: Periodic index optimization
- **Connection Pooling**: Reuse database connections

## Security Considerations
- **Input Validation**: Sanitize all user inputs
- **File Access**: Restrict PDF access to allowed directories
- **Data Isolation**: Separate campaign data by user/campaign
- **Rate Limiting**: Prevent abuse of resource-intensive operations

## Testing Guidelines

### Unit Testing
- Test each module independently with mocked dependencies
- Use `pytest` and `pytest-asyncio` for async tests
- Maintain test coverage above 80%

### Integration Testing
- Test MCP tool registration and execution
- Verify database operations with test fixtures
- Test error handling and edge cases

### Example Test Structure
```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_db():
    db = Mock(spec=ChromaDBManager)
    db.add_document = AsyncMock()
    return db

@pytest.mark.asyncio
async def test_create_campaign(mock_db):
    manager = CampaignManager(mock_db)
    result = await manager.create_campaign("Test", "D&D 5e")
    assert result["success"] == True
    mock_db.add_document.assert_called_once()
```

## Common Pitfalls to Avoid

1. **Multiple MCP Instances**: Always use a single FastMCP instance
2. **Synchronous MCP Tools**: All tools must be async
3. **Missing Error Handling**: Use decorators consistently
4. **Direct Database Access**: Always go through managers
5. **Hardcoded Paths**: Use configuration settings
6. **Circular Imports**: Use dependency injection
7. **Unhandled Datetime Serialization**: Always convert to ISO format
8. **Missing Type Hints**: Use type hints for all functions
9. **Inconsistent Return Formats**: Standardize on success/error patterns
10. **Memory Leaks**: Implement proper cache eviction

## Development Workflow

### Adding New Features
1. Update requirements.md if needed
2. Design data models first
3. Implement business logic in managers
4. Create MCP tool wrappers
5. Register tools in main.py
6. Write tests
7. Update documentation

### Code Review Checklist
- [ ] Single MCP instance pattern followed
- [ ] All MCP tools are async
- [ ] Error handling decorators used
- [ ] Type hints present
- [ ] Tests included
- [ ] Documentation updated
- [ ] No hardcoded values
- [ ] Dependency injection used

## Future Enhancements
- **Multi-language Support**: Support for non-English rulebooks
- **Image Processing**: Extract and index images from PDFs
- **Voice Integration**: Voice commands for hands-free operation
- **Web Interface**: Optional web UI for visual management (See Web UI Integration Architecture below)
- **Cloud Sync**: Optional cloud backup and sync
- **Plugin System**: Extensible architecture for custom tools
- **Real-time Collaboration**: Multiple DMs sharing a campaign (See Collaborative Features below)
- **AI-powered Content Generation**: Generate NPCs, quests, and encounters
- **Mobile Companion App**: Access campaign data on mobile devices

## Web UI Integration Architecture

### Overview
The Web UI Integration enables users to connect their own AI provider accounts (Anthropic, OpenAI, Google Gemini) to access the TTRPG Assistant's MCP tools through a web interface, eliminating the need for custom clients like Claude Desktop while maintaining the robustness of local stdio operations.

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Web UI       │────│  Orchestrator    │────│ MCP Server      │
│   (React)       │SSE │   (FastAPI)      │stdio│  (Existing)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────┴────────┐              │
         │              │                 │              │
    ┌────▼────┐    ┌────▼────┐      ┌─────▼─────┐       │
    │ Auth    │    │ AI      │      │ Session   │       │
    │ Service │    │Providers│      │ Manager   │       │
    └─────────┘    └─────────┘      └───────────┘       │
                        │                               │
               ┌────────┼────────┐                      │
               │        │        │                      │
          ┌────▼─┐ ┌────▼─┐ ┌────▼─┐                   │
          │Claude│ │OpenAI│ │Gemini│                   │
          └──────┘ └──────┘ └──────┘                   │
```

### Core Components

#### 1. MCP Bridge Service
A new component that bridges HTTP/SSE requests to the stdio MCP server, managing session state and request routing.

```python
        self.mcp_processes = {}  # Dict of session_id -> process
        self.sessions = {}  # Track user sessions
        self.request_queue = asyncio.Queue()
        
    async def start_mcp_server(self, session_id: str):
        """Start a dedicated MCP server process for a session."""
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "src.main",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "MCP_STDIO_MODE": "true"}
        )
        return process
```

#### 2. Transport Configuration
Maintains stdio as primary transport while adding bridge capability:

```python
class TransportConfig:
    STDIO_ONLY = "stdio"          # Current mode
    HTTP_BRIDGE = "http_bridge"   # New bridge mode
    HYBRID = "hybrid"              # Both modes simultaneously
```

#### 3. Unified AI Provider Interface
Abstraction layer for different AI providers:

```python
from abc import ABC, abstractmethod

class AIProvider(ABC):
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[AIMessage], 
        tools: List[Dict],
        stream: bool = False
    ) -> AsyncIterator[StreamChunk]:
        pass
    
    @abstractmethod
    async def validate_credentials(self, api_key: str) -> bool:
        pass

class UnifiedAIClient:
    def __init__(self):
        self.providers = {
            ProviderType.ANTHROPIC: AnthropicProvider(),
            ProviderType.OPENAI: OpenAIProvider(), 
            ProviderType.GEMINI: GeminiProvider()
        }
```

### Request/Response Flow

1. **User Query**: Web UI sends query with context to orchestrator
2. **Tool Discovery**: Orchestrator gets available tools from MCP server
3. **AI Processing**: Query forwarded to selected AI provider with tools
4. **Tool Execution**: AI requests tool execution, routed through MCP bridge
5. **Response Streaming**: Results streamed back to UI via SSE

### Security Architecture

#### Authentication & Authorization
- **Multi-provider Authentication**: Support for API keys, OAuth, JWT
- **Session Isolation**: Each user gets isolated MCP process
- **Tool Permissions**: Granular control over tool access per user
- **Rate Limiting**: Per-user and per-tool rate limits

```python
class AuthenticationManager:
    def __init__(self):
        self.providers = {
            "api_key": APIKeyAuthProvider(),
            "oauth": OAuthProvider(),
            "jwt": JWTProvider()
        }
        
    async def authenticate_request(self, request: Request) -> User:
        auth_header = request.headers.get("Authorization")
        provider = self.get_provider(auth_header)
        user = await provider.validate(auth_header)
        
        if not self.check_mcp_permissions(user):
            raise PermissionError("Insufficient MCP permissions")
            
        return user
```

### Context Management Architecture

#### Conversation Context Handling
Manages context across distributed components with consistency and performance:

```python
@dataclass
class ConversationContext:
    session_id: str
    campaign_id: str
    current_scene: Optional[str] = None
    active_characters: List[str] = field(default_factory=list)
    recent_actions: List[Dict] = field(default_factory=list)
    tool_results_cache: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[AIMessage] = field(default_factory=list)
```

#### State Synchronization Strategy
- **Event-driven Updates**: Use event bus for component communication
- **Optimistic Locking**: Prevent conflicting updates with version tracking
- **Cache Coherence**: Multi-layer caching with automatic invalidation
- **Conflict Resolution**: Last-write-wins with audit trail

### Collaborative Features

#### Real-time Multi-user Sessions
```python
class SessionRoom:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.connections: Set[WebSocket] = set()
        self.participants: Dict[str, Dict] = {}
        
    async def broadcast(self, message: Dict, exclude: WebSocket = None):
        for connection in self.connections:
            if connection != exclude:
                await connection.send_text(json.dumps(message))
```

### Performance Optimization

#### Intelligent Caching System
- **Response Caching**: Cache AI responses with context hashing
- **Tool Result Caching**: Reuse tool results within TTL window
- **Predictive Prefetching**: Preload likely next data based on patterns
- **Compression**: Context compression for storage efficiency

#### Load Balancing
- **MCP Server Pool**: Multiple MCP processes for horizontal scaling
- **Cost-optimized Routing**: Route to cheapest appropriate AI provider
- **Circuit Breakers**: Automatic failover for provider failures

### Frontend Architecture

#### Technology Stack
- **Framework**: React 18+ with TypeScript
- **State Management**: Zustand for lightweight real-time updates
- **UI Components**: Shadcn/ui + Tailwind CSS
- **Real-time**: Socket.io client for WebSocket communication
- **Build Tool**: Vite for fast development

#### Key UI Components
- **Campaign Dashboard**: Overview of active campaigns and sessions
- **Tool Result Visualizer**: Rich visualization for different tool outputs
- **Collaborative Canvas**: Shared workspace for maps and notes
- **AI Provider Selector**: Dynamic provider switching interface

### Implementation Phases

#### Phase 1: Bridge Foundation (Weeks 1-2)
- Create FastAPI bridge service
- Implement stdio subprocess management
- Add SSE transport for real-time updates
- Basic request/response routing

#### Phase 2: AI Provider Integration (Weeks 2-3)
- Implement provider abstraction layer
- Add support for Anthropic, OpenAI, Google
- Tool format conversion utilities
- Response streaming handlers

#### Phase 3: Security & Auth (Weeks 3-4)
- Authentication mechanisms
- Session isolation implementation
- Permission system setup
- Rate limiting and quotas

#### Phase 4: Context Management (Weeks 4-5)
- Context persistence layer
- State synchronization system
- Cache implementation
- Conflict resolution

#### Phase 5: UI Development (Weeks 5-7)
- React frontend setup
- Component development
- Real-time features
- Tool visualization

#### Phase 6: Testing & Optimization (Weeks 7-8)
- Load testing
- Performance optimization
- Security auditing
- Documentation

### Configuration

```yaml
# config/bridge.yaml
bridge:
  enabled: true
  host: "0.0.0.0"
  port: 8080
  
  transport:
    type: "sse"  # or "websocket"
    heartbeat_interval: 30
    
  security:
    auth_required: true
    providers:
      - api_key
      - oauth
    rate_limits:
      per_minute: 60
      per_hour: 1000
      
  ai_providers:
    anthropic:
      enabled: true
      api_key: ${ANTHROPIC_API_KEY}
    openai:
      enabled: true
      api_key: ${OPENAI_API_KEY}
    gemini:
      enabled: true
      api_key: ${GEMINI_API_KEY}
      
  mcp:
    session_ttl:
      # Default TTL for sessions (in seconds). Adjust per session type as needed.
      default: 21600  # 6 hours
      one_shot: 14400  # 4 hours
      campaign: 28800  # 8 hours
    resource_limits:
      memory_mb: 512
      cpu_percent: 50
```

### Monitoring & Observability

#### Key Metrics
- **Context Retrieval**: < 50ms for 95th percentile
- **Cache Hit Rate**: > 90% for frequently accessed data
- **Synchronization Latency**: < 100ms cross-component updates
- **Memory Efficiency**: < 2GB context storage per 1000 sessions
- **Error Recovery Time**: < 5 seconds for component failures

#### Monitoring Stack
- **Metrics**: Prometheus + Grafana
- **Tracing**: OpenTelemetry
- **Logging**: Structured logging with correlation IDs
- **Alerting**: PagerDuty integration for critical issues