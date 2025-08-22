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
- **Web Interface**: Optional web UI for visual management
- **Cloud Sync**: Optional cloud backup and sync
- **Plugin System**: Extensible architecture for custom tools
- **Real-time Collaboration**: Multiple DMs sharing a campaign
- **AI-powered Content Generation**: Generate NPCs, quests, and encounters
- **Mobile Companion App**: Access campaign data on mobile devices