# TTRPG Assistant MCP Server - Project Context

## Project Overview
This is a Model Context Protocol (MCP) server for assisting with Tabletop Role-Playing Games (TTRPGs). It serves as a comprehensive "side-car" assistant for Dungeon Masters and Game Runners, providing quick access to rules, spells, monsters, and campaign information during gameplay.

## Key Features
- **PDF Processing**: Extracts and indexes content from TTRPG rulebooks and source materials
- **Hybrid Search**: Combines semantic and keyword search for finding rules and content
- **Campaign Management**: Stores and retrieves campaign-specific data (NPCs, locations, plot points)
- **Session Tracking**: Manages initiative, monster health, and session notes
- **Personality System**: Adapts responses to match the tone and style of different game systems
- **Character/NPC Generation**: Creates characters with appropriate stats and backstories
- **Adaptive Learning**: Improves PDF processing accuracy over time

## Technology Stack (Updated 2024)

### Backend
- **Language**: Python 3.11+
- **MCP Framework**: FastMCP
- **Web Framework**: FastAPI (for bridge service)
- **Database**: ChromaDB (embedded vector database)
- **Communication**: stdin/stdout (MCP) + WebSocket/SSE (web)
- **PDF Processing**: pypdf/pdfplumber (modern, maintained packages)
- **Search**: Hybrid approach with vector embeddings and BM25
- **Error Handling**: Result/Either pattern (returns library)

### Frontend
- **Framework**: SvelteKit (replacing React)
- **Language**: TypeScript
- **Styling**: TailwindCSS (responsive, mobile-first)
- **State Management**: Native Svelte stores
- **Build Tool**: Vite
- **Real-time**: WebSockets and Server-Sent Events

### Key Patterns
- **Error as Values**: Using Result types instead of exceptions
- **Responsive Web**: Single codebase for all devices (no separate mobile app)
- **SSR**: Server-side rendering for optimal performance
- **Type Safety**: End-to-end type safety with TypeScript and Python type hints

## Project Structure
```
/home/svnbjrn/MDMAI/
├── requirements.md     # Detailed functional and non-functional requirements
├── design.md          # Technical design and architecture
├── tasks.md           # Implementation tasks with requirement mappings
└── CLAUDE.md          # This file - project context and guidelines
```

## Development Guidelines

### IMPORTANT: Use of Specialized Sub-Agents
When working on this project, ALWAYS proactively use these specialized sub-agents:
- **mcp-protocol-expert**: For all MCP protocol implementation, tool design, and FastMCP integration
- **llm-architect**: For designing AI provider integrations, model selection, and LLM system architecture
- **context-manager**: For context persistence, session management, and distributed state handling
- **python-pro**: For ALL Python development - optimization, advanced patterns, testing, and refactoring

These agents should be used proactively without waiting for explicit requests. They provide specialized expertise critical for this project's success.

### Code Style
- Use Python type hints for all functions
- Follow PEP 8 style guidelines
- Use async/await for all MCP tools
- Implement comprehensive error handling
- Add logging for debugging and monitoring

### MCP Tool Pattern
```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("TTRPG")

@mcp.tool()
async def tool_name(param: type) -> Dict[str, Any]:
    """Tool documentation"""
    # Implementation
```

### Database Collections
- `rulebooks`: Game system rules and mechanics
- `flavor_sources`: Novels and narrative content
- `campaigns`: Campaign data and metadata
- `sessions`: Active game session tracking
- `personalities`: System-specific personality profiles

### Search Implementation
- Always use hybrid search (semantic + keyword) by default
- Return results with source citations and page numbers
- Implement query expansion for better matches
- Cache frequent queries for performance

### Personality System
Each game system has a unique personality:
- **D&D 5e**: Wise Sage (authoritative, academic)
- **Blades in the Dark**: Shadowy Informant (mysterious, Victorian)
- **Delta Green**: Classified Handler (formal, military)
- **Call of Cthulhu**: Antiquarian Scholar (scholarly, ominous)

### Testing Strategy
- Unit tests for all core functions
- Integration tests for MCP tools
- Performance tests for search operations
- End-to-end tests for complete workflows

## Current Implementation Status
- [x] Requirements documented
- [x] Technical design completed
- [x] Task breakdown with requirement mappings
- [ ] Core infrastructure setup
- [ ] PDF processing pipeline
- [ ] Search engine implementation
- [ ] Campaign management system
- [ ] Session tracking features
- [ ] Character generation tools
- [ ] Personality system
- [ ] Testing suite
- [ ] Documentation

## Development Priorities
1. **Phase 1**: Core infrastructure and basic PDF processing
2. **Phase 2**: Search functionality and MCP tools
3. **Phase 3**: Campaign and session management
4. **Phase 4**: Personality system and generation tools
5. **Phase 5**: Performance optimization and testing

## Important Notes
- Focus on local operations via stdin/stdout
- Use ChromaDB for all vector storage needs
- Preserve table structure when processing PDFs
- Implement adaptive learning for better PDF parsing
- Ensure all searches include both semantic and keyword matching
- Maintain clear separation between rulebook and flavor sources
- Version all campaign data for rollback capability

## Common Commands
```bash
# Backend Setup
pip install -r requirements.txt  # Install Python dependencies
python src/main.py  # Run the MCP server

# Frontend Setup
cd frontend
npm install  # Install Node dependencies
npm run dev  # Start SvelteKit dev server
npm run build  # Build for production
npm run preview  # Preview production build

# Testing
pytest tests/  # Run Python tests
cd frontend && npm test  # Run frontend tests

# Code Quality
ruff check .  # Lint Python code (replaces flake8, black, isort)
ruff format .  # Format Python code
mypy src/  # Type check Python
cd frontend && npm run check  # Type check TypeScript

# Development
pre-commit install  # Set up git hooks
docker-compose up  # Run with Docker
```

## Next Steps
1. Set up the project structure and dependencies
2. Implement the core MCP server with FastMCP
3. Create the ChromaDB integration layer
4. Build the PDF processing pipeline
5. Implement the search functionality
6. Add campaign management tools
7. Develop the personality system
8. Create comprehensive tests
9. Write user documentation