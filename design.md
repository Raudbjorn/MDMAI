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
9. **Phase 9-13**: Advanced Features - Tools, monitoring, security, deployment
10. **Phase 14**: Bridge Service - MCP to web integration
11. **Phase 15**: AI Provider Integration - Multi-provider support
12. **Phase 16**: Security & Authentication - OAuth2, JWT, sandboxing
13. **Phase 17**: Context Management - Distributed state management
14. **Phase 18**: Frontend Development - SvelteKit responsive web application
15. **Phase 19**: Integration Testing - End-to-end testing
16. **Phase 20**: Performance Optimization - Final optimizations
17. **Phase 21**: Mobile Support - Integrated into Phase 18 (responsive web)
18. **Phase 22**: Testing & Documentation - Comprehensive test suite and docs
19. **Phase 23**: Desktop Application - Native desktop app using Tauri

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
- **Extraction**: pypdf or pdfplumber for text extraction (modern, maintained packages)
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
- **Progressive Web App**: Installable web app with offline capabilities

## Web UI Integration Architecture

### Overview
The Web UI Integration enables users to connect their own AI provider accounts (Anthropic, OpenAI, Google Gemini) to access the TTRPG Assistant's MCP tools through a web interface, eliminating the need for custom clients like Claude Desktop while maintaining the robustness of local stdio operations.

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Web UI       │────│  Bridge Service  │────│ MCP Server      │
│  (SvelteKit)    │ WS │   (FastAPI)      │stdio│  (FastMCP)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────┴────────┐              │
         │              │                 │              │
    ┌────▼────┐    ┌────▼────┐      ┌─────▼─────┐       │
    │  SSR    │    │ AI      │      │ Session   │       │
    │ Routes  │    │Providers│      │ Manager   │       │
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
A new component that bridges WebSocket requests to the stdio MCP server, managing session state and request routing.

```python
        self.mcp_processes = {}  # Dict of session_id -> process
        self.sessions = {}  # Track user sessions
        self.request_queue = asyncio.Queue()
        
    async def start_mcp_server(self, session_id: str):
        )
        self.sessions[session_id] = process
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
5. **Response Streaming**: Results streamed back to UI via WebSocket

### Security Architecture

#### Authentication & Authorization
- **Multi-provider Authentication**: Support for API keys, OAuth, JWT
- **Session Isolation**: Each user gets isolated MCP process
- **Tool Permissions**: Granular control over tool access per user
- **Rate Limiting**: Per-user and per-tool rate limits
- **Result Pattern**: Error-as-values for secure error handling
- **SSR Security**: Server-side validation and sanitization

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
- **Conflict Resolution**: Hybrid approach based on data type:
  - **Operational Transformation (OT)**: For text-based content (notes, descriptions)
  - **CRDTs (Conflict-free Replicated Data Types)**: For collaborative canvas and shared maps
  - **Last-write-wins with audit trail**: For simple property updates (HP, status flags)
  - **Three-way merge**: For complex campaign data with user-prompted resolution

### Collaborative Features

#### Real-time Multi-user Sessions with CRDT Support
```python
class SessionRoom:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.connections: Set[WebSocket] = set()
        self.participants: Dict[str, Dict] = {}
        # CRDT state for collaborative features
        self.crdt_state = {
            'canvas': YjsDocument(),  # For collaborative canvas
            'notes': ShareableString(),  # For shared notes
            'map_positions': LWWMap()  # Last-write-wins map for positions
        }
        
    async def broadcast(self, message: Dict, exclude: WebSocket = None):
        for connection in self.connections:
            if connection != exclude:
                await connection.send_text(json.dumps(message))
    
    def apply_operation(self, op_type: str, operation: Any):
        """Apply CRDT operation based on type"""
        if op_type == 'canvas':
            self.crdt_state['canvas'].apply_update(operation)
        elif op_type == 'text':
            self.crdt_state['notes'].apply_op(operation)
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
- **Framework**: SvelteKit 2.x with TypeScript
- **State Management**: Built-in Svelte stores for reactive state
- **UI Components**: Tailwind CSS + DaisyUI/Skeleton UI
- **Real-time**: WebSocket API (native) + Server-Sent Events (SSE)
- **Build Tool**: Vite (integrated with SvelteKit)

#### Key UI Components
- **Campaign Dashboard**: SSR-optimized overview of active campaigns
- **Tool Result Visualizer**: Progressive enhancement for tool outputs
- **Collaborative Canvas**: Real-time shared workspace with CRDT support
- **AI Provider Selector**: Server-side provider routing with client fallback
- **Responsive Design**: Mobile-first approach (no separate mobile app)

### Implementation Phases

#### Phase 1: Bridge Foundation (Weeks 1-2)
- Create FastAPI bridge service
- Implement stdio subprocess management
- Add WebSocket transport for bidirectional real-time communication
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
- SvelteKit application setup with TypeScript
- SSR/CSR hybrid components for optimal performance
- WebSocket/SSE integration for real-time updates
- Progressive enhancement with form actions
- Responsive design system (mobile-first, no separate app)

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
    type: "websocket"  # Standardized on WebSocket for bidirectional communication
    heartbeat_interval: 30
    reconnect_timeout: 5000  # Auto-reconnect after 5 seconds
    
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

## Technology Stack Updates (2024)

### Frontend Migration: React → SvelteKit

#### Architecture Changes
- **Framework**: SvelteKit with TypeScript for full-stack web application
- **Routing**: File-based routing with +page.svelte and +layout.svelte
- **State Management**: Native Svelte stores replacing Redux/Zustand
- **Styling**: TailwindCSS with mobile-first responsive design
- **Build**: Vite with @sveltejs/vite-plugin-svelte

#### SvelteKit Project Structure
```
frontend/
├── src/
│   ├── routes/              # File-based routing
│   │   ├── +layout.svelte   # Root layout
│   │   ├── +page.svelte     # Home page
│   │   ├── api/             # API endpoints
│   │   │   └── mcp/         # MCP bridge endpoints
│   │   ├── campaigns/       # Campaign management
│   │   └── session/         # Game sessions
│   ├── lib/                 # Shared code
│   │   ├── components/      # Reusable components
│   │   ├── stores/          # Global state stores
│   │   ├── mcp/            # MCP client
│   │   └── utils/          # Utilities
│   └── app.html            # App template
├── static/                 # Static assets
└── vite.config.js         # Build configuration
```

#### MCP Integration Pattern
```typescript
// Server-side MCP communication
// src/routes/api/mcp/+server.ts
export async function POST({ request }) {
    const { tool, params } = await request.json();
    const result = await mcpBridge.executeTool(tool, params);
    return json(result);
}

// Client-side store
// src/lib/stores/mcp.ts
export const mcpStore = writable<MCPState>({
    connected: false,
    tools: [],
    resources: []
});
```

### Python Modernization

#### Core Dependencies Update
```txt
# Core Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database & Storage
chromadb==0.4.22
sqlalchemy==2.0.25
alembic==1.13.0

# PDF Processing
pypdf==3.17.0           # Replacing PyPDF2
pdfplumber==0.10.3
pikepdf==8.10.0         # For robust PDF handling

# AI/ML
sentence-transformers==2.3.0
instructor==0.5.0       # Structured LLM outputs

# HTTP & Async
httpx==0.26.0          # Replacing requests
tenacity==8.2.0        # Retry logic

# Error Handling (New)
returns==0.22.0        # Result/Either pattern

# Utilities
structlog==24.1.0      # Structured logging
python-multipart==0.0.6

# Development Tools
pytest==7.4.0
pytest-asyncio==0.21.0
mypy==1.7.0
ruff==0.1.0            # Fast Python linter
```

#### Error-as-Values Pattern
```python
# New pattern throughout codebase using returns library
from returns.result import Result, Success, Failure
from src.core.result_pattern import AppError, ErrorKind, with_result

@with_result(error_kind=ErrorKind.PARSING)
async def process_document(pdf_path: str) -> Result[Document, AppError]:
    """Process with Result pattern instead of exceptions."""
    content = await extract_pdf_content(pdf_path)
    if not content:
        return Failure(AppError(
            kind=ErrorKind.PARSING,
            message="Empty document",
            recovery_hint="Check if the PDF contains extractable text"
        ))
    
    chunks = await chunk_content(content)
    embeddings = await generate_embeddings(chunks)
    
    return Success(Document(
        content=content,
        chunks=chunks,
        embeddings=embeddings
    ))

# Usage
result = await process_document("rulebook.pdf")
match result:
    case Success(document):
        await store_document(document)
    case Failure(error):
        logger.error(f"Processing failed: {error}")
```

### Web-First Approach

#### Responsive Design Strategy
- **Mobile-First**: Design for mobile, enhance for desktop
- **Progressive Enhancement**: Core functionality works without JS
- **Single Codebase**: No separate mobile app (removed Phase 21)
- **Touch Optimized**: Touch gestures and mobile interactions
- **Offline Support**: Service workers for offline functionality

#### Key Benefits
1. **Simplified Maintenance**: One codebase for all platforms
2. **Faster Development**: No React Native complexity
3. **Better Performance**: SvelteKit's optimizations
4. **Consistent Experience**: Same features across devices
5. **Lower Cost**: No app store fees or native development

### Integration Architecture

#### Bridge Service Updates
```python
# Updated for SvelteKit SSR
class MCPBridge:
    async def handle_svelte_request(self, request: dict) -> Result[dict, BridgeError]:
        """Handle SvelteKit server-side requests."""
        # Validate request
        validation = self.validate_request(request)
        if isinstance(validation, Failure):
            return validation
        
        # Execute MCP tool
        result = await self.execute_tool(
            request["tool"],
            request["params"]
        )
        
        # Transform for SvelteKit
        return self.transform_for_svelte(result)
```

#### WebSocket Manager
```typescript
// SvelteKit WebSocket handling
// src/lib/mcp/websocket.ts
export class MCPWebSocket {
    private ws: WebSocket;
    private reconnectAttempts = 0;
    
    connect() {
        this.ws = new WebSocket('/api/mcp/ws');
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            mcpStore.update(state => ({
                ...state,
                ...data
            }));
        };
        
        this.ws.onerror = () => {
            this.reconnect();
        };
    }
    
    private reconnect() {
        if (this.reconnectAttempts < 5) {
            setTimeout(() => this.connect(), 
                Math.pow(2, this.reconnectAttempts) * 1000);
            this.reconnectAttempts++;
        }
    }
}
```

### Migration Timeline

#### Phase 1: Documentation & Planning (Week 1)
- Update all documentation for new stack
- Create migration guides
- Set up development environment

#### Phase 2: Backend Updates (Weeks 2-3)
- Update Python dependencies
- Implement Result pattern
- Add SvelteKit-compatible endpoints

#### Phase 3: Frontend Development (Weeks 4-6)
- Initialize SvelteKit project
- Port components to Svelte
- Implement MCP integration

#### Phase 4: Integration & Testing (Weeks 7-8)
- Connect frontend to backend
- End-to-end testing
- Performance optimization

### Performance Targets

#### Frontend Metrics
- **First Contentful Paint**: < 1.5s
- **Time to Interactive**: < 3.5s
- **Bundle Size**: < 200KB (JS)
- **Lighthouse Score**: > 90

#### Backend Metrics
- **API Response Time**: < 100ms (p95)
- **PDF Processing**: < 5s per document
- **Search Latency**: < 50ms
- **WebSocket Latency**: < 10ms

## Phase 23: Desktop Application Architecture

### Overview
The desktop application provides a native experience for running the TTRPG Assistant MCP Server locally with an integrated user interface. Built using Tauri, it combines the efficiency of a Rust backend with the existing SvelteKit frontend, providing a lightweight, secure, and performant desktop experience.

### Technology Stack

#### Core Framework: Tauri
- **Why Tauri**: 
  - Small bundle size (3-10MB base vs 50-150MB for Electron)
  - 90% less memory usage than Electron
  - Native performance with Rust backend
  - Direct reuse of existing SvelteKit frontend
  - Built-in security with sandboxed WebView
  - Cross-platform support (Windows, macOS, Linux)

#### Architecture Components
```
┌─────────────────────────────────────────────┐
│            Tauri Application                 │
├─────────────────────────────────────────────┤
│  ┌────────────────────────────────────┐     │
│  │    SvelteKit Frontend (WebView)    │     │
│  └────────────────────────────────────┘     │
│                    ↕ IPC                     │
│  ┌────────────────────────────────────┐     │
│  │      Rust Backend (Tauri)          │     │
│  │  - Process Management               │     │
│  │  - File System Access               │     │
│  │  - System Tray Integration          │     │
│  └────────────────────────────────────┘     │
│                    ↕ stdio                   │
│  ┌────────────────────────────────────┐     │
│  │    Python MCP Server (subprocess)   │     │
│  │  - FastMCP Server                   │     │
│  │  - ChromaDB/SQLite                  │     │
│  │  - All existing functionality       │     │
│  └────────────────────────────────────┘     │
└─────────────────────────────────────────────┘
```

### Communication Architecture

#### Stdio-Based IPC (Simple and Robust)
- **Frontend ↔ Rust**: Tauri's built-in IPC using JSON serialization
- **Rust ↔ Python**: JSON-RPC 2.0 over stdin/stdout (native MCP protocol)
- **Protocol**: Line-delimited JSON-RPC messages
- **Security**: Process sandboxing, command allowlisting, CSP enforcement

#### Why Stdio Over WebSocket
After careful evaluation, stdio communication provides:
- **Zero Python changes** - MCP server already supports stdio perfectly
- **Process isolation** - Complete memory separation between components
- **Simpler deployment** - No network services or ports to manage
- **Better error recovery** - Tauri can restart crashed processes cleanly
- **Fewer dependencies** - No FastAPI, uvicorn, or WebSocket libraries needed

#### Stdio Bridge Implementation
```rust
// Rust manages Python process lifecycle
struct MCPBridge {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
}

// Bridge translates between Tauri IPC and stdio
async fn mcp_call(method: String, params: Value) -> Result<Value> {
    // Send JSON-RPC request to Python stdin
    // Read JSON-RPC response from Python stdout
}
```

#### Data Flow
1. User interaction in SvelteKit UI
2. IPC command to Rust backend via Tauri
3. Rust sends JSON-RPC to Python process via stdin
4. Python MCP server processes request
5. Response sent back via stdout
6. Rust parses response and relays to frontend
7. UI updates with results or error state

#### Process Management
- **Automatic Restart**: Rust monitors and restarts Python if it crashes
- **Health Monitoring**: Regular heartbeat commands to verify process health
- **Resource Limits**: CPU and memory limits enforced by OS
- **Clean Shutdown**: Graceful termination on app close

### Python Packaging Strategy

#### PyOxidizer Integration
- **Single Executable**: Bundle Python runtime + dependencies
- **Size**: ~50MB for complete Python environment
- **Startup Time**: ~700ms (vs 2-3s for traditional Python)
- **Dependencies**: All included (ChromaDB, sentence-transformers, etc.)

#### Distribution Structure
```
app/
├── tauri-app.exe/app/dmg    # Platform-specific Tauri executable
├── resources/
│   ├── python/               # Embedded Python executable
│   │   └── mcp-server        # PyOxidizer bundle
│   ├── data/                 # User data directory
│   │   ├── chromadb/         # Vector database
│   │   ├── sqlite/           # Structured data
│   │   └── config/           # User configuration
│   └── assets/               # Static assets
```

### Platform-Specific Implementation

#### Windows
- **Installer**: MSI and NSIS installers
- **WebView**: WebView2 (Chromium-based)
- **Python**: Embedded via PyOxidizer
- **Auto-start**: Windows service option
- **Updates**: Built-in auto-updater

#### macOS
- **Distribution**: DMG and .app bundle
- **WebView**: WKWebView (Safari-based)
- **Code Signing**: Required for distribution
- **Python**: Universal binary for Intel/Apple Silicon
- **Updates**: Sparkle framework integration

#### Linux
- **Packages**: AppImage, .deb, .rpm
- **WebView**: WebKitGTK
- **Python**: System Python or embedded
- **Desktop Integration**: .desktop file
- **Updates**: AppImageUpdate or package manager

### Security Considerations

#### Process Isolation
- Python runs as separate subprocess
- Limited file system access
- Network restrictions configurable
- No direct system calls from frontend

#### Data Protection
- Local storage encryption for sensitive data
- Secure credential storage using OS keychain
- Session tokens with expiration
- Audit logging for security events

### Performance Optimizations

#### Startup Performance
- Lazy loading of Python modules
- Pre-compiled Python bytecode
- Background initialization of ChromaDB
- Progressive UI loading

#### Runtime Performance
- Process pooling for parallel operations
- Efficient IPC with message batching
- Memory-mapped file sharing for large data
- Native file dialogs and system integration

### Development Workflow

#### Build Process
```bash
# Development
npm run tauri dev

# Production Build
npm run tauri build

# Platform-specific builds
npm run tauri build -- --target x86_64-pc-windows-msvc
npm run tauri build -- --target x86_64-apple-darwin
npm run tauri build -- --target x86_64-unknown-linux-gnu
```

#### Configuration
```toml
# tauri.conf.json
{
  "build": {
    "beforeBuildCommand": "npm run build",
    "beforeDevCommand": "npm run dev",
    "devPath": "http://localhost:5173",
    "distDir": "../build"
  },
  "package": {
    "productName": "TTRPG Assistant",
    "version": "1.0.0"
  },
  "tauri": {
    "allowlist": {
      "shell": {
        "execute": true,
        "scope": [{
          "name": "run-mcp-server",
          "cmd": "python",
          "args": ["src/main.py"]
        }]
      }
    }
  }
}
```

### Features Comparison

| Feature | Web Version | Desktop Version |
|---------|------------|-----------------|
| MCP Server | Remote | Local (embedded) |
| File Access | Limited | Full (sandboxed) |
| Offline Mode | Service Worker | Native |
| PDF Processing | Server-side | Local |
| Performance | Network-dependent | Native speed |
| Updates | Automatic | Auto-updater |
| Installation | None | One-time |
| System Integration | Limited | Full |

### Migration Path

#### From Web to Desktop
1. User exports data from web version
2. Desktop app imports existing campaigns/settings
3. Automatic schema migration if needed
4. Sync option for hybrid usage

#### Shared Codebase Strategy
- Frontend: 95% code reuse (SvelteKit)
- Backend: 100% code reuse (Python MCP)
- Desktop-specific: ~5% (Tauri commands)
- Maintenance: Single codebase for core logic

### Desktop-Specific Features

#### System Tray Integration
- Quick access to common functions
- Background operation mode
- Resource usage monitoring
- Session status indicators
- Dynamic icon states (active, syncing, error)

#### Native File Handling
- Drag-and-drop PDF import
- OS file associations (.ttrpg files)
- Native file dialogs
- Recent files menu

#### Offline Capabilities
- Full functionality without internet
- Local AI model support (optional)
- Offline documentation
- Local backup/restore

### Visual Assets and Polish

#### Required Icon Files
```
frontend/src-tauri/icons/
├── icon.ico              # Windows icon (multi-size)
├── 32x32.png            # Tray icon, taskbar
├── 128x128.png          # Medium size
├── 128x128@2x.png       # High DPI (256x256)
├── icon.png             # Source (512x512+)
└── tray/
    ├── icon.ico         # Default tray
    ├── icon-active.ico  # Connected state
    ├── icon-error.ico   # Error state
    └── icon-syncing.ico # Loading state
```

#### Windows Installer Graphics
```
frontend/src-tauri/installer/
├── header.bmp           # 150x57px NSIS header
├── welcome.bmp          # 164x314px welcome page
└── icon.ico            # Installer executable icon

frontend/src-tauri/wix/
├── banner.bmp          # 493x58px MSI banner
├── dialog.bmp          # 493x312px background
└── license.rtf         # License agreement
```

#### Custom Window Styling
- Modern titlebar with Windows 11 integration
- Smooth animations and transitions
- High DPI support via manifest
- Custom scrollbars matching OS theme
- Native Windows fonts (Inter + Iosevka for code)

### Performance Targets

#### Desktop Metrics
- **Application Size**: < 65MB total
- **Startup Time**: < 2 seconds
- **Memory Usage**: < 150MB idle
- **CPU Usage**: < 5% idle
- **IPC Latency**: < 5ms