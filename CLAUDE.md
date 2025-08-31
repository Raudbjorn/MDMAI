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

**ALWAYS use these specialized sub-agents proactively for this project. Do not wait for explicit requests - use them automatically when working on relevant areas:**

#### Required Sub-Agents by Task Area:

1. **mcp-protocol-expert** - Use for:
   - All MCP protocol implementation and tool design
   - FastMCP integration and bridge service development
   - WebSocket-to-MCP protocol translation
   - Defining new MCP tools or modifying existing ones
   - MCP message formatting and session management

2. **llm-architect** - Use for:
   - AI provider integration design (Anthropic, OpenAI, Google)
   - Model selection and optimization strategies
   - Cost optimization and provider failover mechanisms
   - Prompt engineering for TTRPG-specific tasks
   - RAG system design for rulebook searches

3. **context-manager** - Use for:
   - State synchronization architecture
   - ChromaDB collection design and optimization
   - Session state persistence and recovery
   - Distributed state management for multiplayer
   - Cache invalidation strategies

4. **python-pro** - Use for:
   - ALL Python code implementation
   - Performance optimization and profiling
   - Advanced patterns (async/await, decorators, metaclasses)
   - Testing strategies and pytest fixtures
   - Refactoring existing Python code

5. **svelte-ttrpg-developer** - Use for:
   - ALL Svelte/SvelteKit component development
   - State management with Svelte 5 runes
   - Real-time UI components for TTRPG features
   - TypeScript implementation in frontend
   - Accessibility and responsive design

#### Example Usage Pattern:
```
User: "Implement the dice rolling MCP tool"
Assistant: [Automatically uses mcp-protocol-expert for MCP tool design]
         [Automatically uses python-pro for Python implementation]
         [Automatically uses svelte-ttrpg-developer for frontend component]
```

These agents provide specialized expertise that is **critical for project success**. Using them ensures best practices, optimal performance, and production-ready code.

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
- [x] Frontend infrastructure (SvelteKit, components, stores)
- [x] Real-time WebSocket/SSE implementation
- [x] Provider management system (AI providers)
- [x] Collaborative tools (dice, notes, maps, initiative)
- [x] Performance optimization framework
- [x] Offline support with Service Workers
- [x] Authentication & security design
- [x] Spike research for all unknowns
- [x] Desktop application (Phase 23) - Tauri + SvelteKit with stdio
- [ ] Core MCP server implementation
- [ ] PDF processing pipeline
- [ ] Search engine implementation
- [ ] Backend campaign management
- [ ] MCP tools implementation
- [ ] Third-party integrations
- [ ] Production deployment

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

## GUI Development Guidelines

### Shared Components Between Webapp and Desktop

When adding new UI features that need to work in both the webapp and desktop app:

1. **Create shared components in `/frontend/src/lib/components/`**
   - Use environment detection: `if (browser && window.__TAURI__)` for desktop-specific code
   - Abstract API calls through a client layer that can switch between direct HTTP and Tauri IPC

2. **Use Svelte 5 Runes for State Management**
   ```typescript
   // Use $state for reactive state
   private state = $state<StateType>({...});
   
   // Use $derived for computed values
   computedValue = $derived(() => this.state.value * 2);
   
   // Use $effect for side effects
   $effect(() => { /* reaction to state changes */ });
   ```

3. **Follow Error-as-Values Pattern**
   ```typescript
   type Result<T> = { ok: true; data: T } | { ok: false; error: string };
   ```

4. **Component Structure**
   - Keep components focused and single-purpose
   - Create separate components for complex UI elements
   - Use TypeScript for all component props and events
   - Make components accessible with proper ARIA labels

5. **Model Selection Pattern (Ollama Example)**
   - List only what's already installed (don't manage installations in-app)
   - Provide clear fallback options (e.g., Sentence Transformers)
   - Show service status clearly
   - Integrate selection at point of use (e.g., PDF upload)

### Adding Provider Support

When adding new AI provider support (like Ollama):

1. **Update Types** (`/frontend/src/lib/types/providers.ts`)
   - Add to `ProviderType` enum
   - Create specific types file if needed

2. **Create API Client** (`/frontend/src/lib/api/[provider]-client.ts`)
   - Implement standard interface methods
   - Handle both webapp and desktop environments

3. **Create Store** (`/frontend/src/lib/stores/[provider].svelte.ts`)
   - Use Svelte 5 runes
   - Include caching and error handling
   - Persist user preferences to localStorage

4. **Create UI Components**
   - Selector component for choosing models/options
   - Status indicator for service availability
   - Integration with existing workflows

### Testing Guidelines

1. **Backend Tests** (`/tests/`)
   - Test API endpoints with mocked services
   - Test model selection and fallbacks
   - Test error conditions

2. **Frontend Tests** (`/frontend/src/lib/components/**/*.test.ts`)
   - Use Vitest and Testing Library
   - Mock API calls
   - Test user interactions
   - Test error states and loading states

## Common Commands
```bash
# Backend Setup
pip install -r requirements.txt  # Install Python dependencies
python src/main.py  # Run the MCP server
python run_api.py  # Run the FastAPI server

# Frontend Setup
cd frontend
npm install  # Install Node dependencies
npm run dev  # Start SvelteKit dev server
npm run build  # Build for production
npm run preview  # Preview production build

# Desktop App
cd desktop/frontend
npm install
npm run tauri:dev  # Run desktop app in dev mode
npm run tauri:build  # Build desktop app

# Testing
pytest tests/  # Run Python tests
pytest tests/test_ollama_model_selection.py  # Run specific test
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

## Documentation Structure

### Core Documentation
- `requirements.md` - Functional and non-functional requirements
- `design.md` - Technical architecture and design decisions
- `tasks.md` - Implementation tasks with requirement mappings
- `CLAUDE.md` - Project context and development guidelines

### Frontend Documentation (`/frontend/docs/`)

#### Requirements & Design
- `requirements/TASK_18_3_REQUIREMENTS.md` - Real-time features requirements
- `design/TASK_18_3_DESIGN.md` - Component architecture for real-time
- `planning/TASK_18_3_BREAKDOWN.md` - Story points breakdown (76 points)

#### Spike Research (`/frontend/docs/spikes/`)
- `SPIKE_1_WEBSOCKET.md` - WebSocket-to-MCP bridge architecture
- `SPIKE_2_STATE_SYNC.md` - Hybrid Event Sourcing + CRDT strategy
- `SPIKE_3_AUTH_SECURITY.md` - JWT auth, encryption, security policies
- `SPIKE_4_PERFORMANCE.md` - Performance targets and optimization
- `SPIKE_5_OFFLINE.md` - Service Workers and offline capability
- `SPIKE_6_INTEGRATIONS.md` - D&D Beyond, Roll20, Discord, etc.
- `SPIKE_7_DATABASE.md` - ChromaDB, SQLite, Redis architecture
- `SPIKE_8_ERROR_RECOVERY.md` - Circuit breakers, resilience patterns
- `SPIKE_9_TESTING.md` - Test pyramid, coverage targets
- `SPIKE_10_BROWSER_SUPPORT.md` - Progressive enhancement strategy
- `SPIKE_SUMMARY.md` - Executive summary of all spikes

#### API Documentation
- `REALTIME_API.md` - WebSocket/SSE API specifications
- `REALTIME_MIGRATION_PLAN.md` - Migration strategy for real-time
- `REALTIME_SECURITY_CHECKLIST.md` - Security audit checklist

### MCP Tools Documentation
- `docs/MCP_TOOLS_SPEC.md` - Complete specification of 30+ MCP tools

### Implementation Guides
- `frontend/src/lib/components/providers/README.md` - Provider system guide

## Key Architecture Decisions

### Real-time Communication
- **WebSocket Bridge**: Enhanced bridge service at `/src/bridge/bridge_server.py`
- **State Sync**: Hybrid Event Sourcing + CRDT for conflict resolution
- **Fallback**: SSE and polling for older browsers

### Authentication & Security
- **Dual Mode**: Local (stdin/stdout) and Web (JWT)
- **Roles**: GM, Player, Spectator with granular permissions
- **Encryption**: AES-256 for sensitive data, TLS for transport

### Data Storage
- **Vector DB**: ChromaDB for semantic search and embeddings
- **Structured**: SQLite for campaigns, sessions, characters
- **Cache**: Redis for performance, multi-level caching
- **Offline**: IndexedDB for client-side persistence

### Performance Targets
- **Search**: P50 <50ms, P95 <200ms latency
- **WebSocket**: 5000 msg/sec capacity
- **Concurrent Users**: 20 per session, 2000 total
- **PDF Processing**: 2-30 seconds depending on size

### Testing Strategy
- **Coverage**: 80% overall, 95% critical paths
- **Pyramid**: 70% unit, 25% integration, 5% E2E
- **Tools**: Vitest, Pytest, Playwright, Locust

## MCP Tools Overview

### Core Tools (Enhanced)
- `search_rules` - Hybrid search with campaign context
- `roll_dice` - Advanced modifiers, advantage/disadvantage
- `get_monster` - Party scaling, tactical suggestions
- `manage_initiative` - Conditions, delays, ready actions
- `take_notes` - AI categorization, entity linking

### Combat Automation
- `track_damage` - HP tracking with resistances
- `apply_condition` - Automatic effect tracking
- `generate_random_table` - Custom tables

### Campaign Management
- `manage_campaign_timeline` - In-game time tracking
- `track_campaign_resources` - Party inventory
- `manage_quest_tracker` - Quest objectives

### Generation Tools
- `generate_npc` - Complete NPCs with relationships
- `generate_location` - Maps, NPCs, loot
- `generate_plot_hook` - Context-aware story seeds

### Real-time Collaboration
- `broadcast_to_players` - Selective info sharing
- `sync_game_state` - Automatic synchronization
- `request_player_action` - Interactive prompts

## Phase 23: Desktop Application (Completed)

### Architecture Decision: Stdio over WebSocket
After extensive review, we chose **stdio communication** for the desktop app:
- **Zero Python changes required** - FastMCP already supports stdio natively
- **Simpler architecture** - Direct process communication without network layer
- **Better security** - No exposed ports, process isolation by default
- **Lower latency** - Direct IPC without TCP/HTTP overhead (<1ms vs 5-10ms)

### Desktop Application Stack
- **Framework**: Tauri (Rust) - 90% less memory than Electron
- **Frontend**: SvelteKit with static adapter (SPA mode)
- **Communication**: JSON-RPC 2.0 over stdin/stdout
- **Process Management**: Tauri sidecar API for Python subprocess
- **Bundle Size**: 12-18MB (96% smaller than Electron)

### Key Implementation Files
- **Rust Bridge**: `/desktop/frontend/src-tauri/src/mcp_bridge.rs`
  - Channel-based stdin communication using tokio::sync::mpsc
  - Background task owns CommandChild for proper lifecycle
  - Health checks and automatic reconnection
- **TypeScript Client**: `/desktop/frontend/src/lib/mcp-robust-client.ts`
  - Exponential backoff with jitter (3 attempts, 1s/2s/4s...)
  - LRU cache with TTL for performance
  - Graceful degradation with fallback values
- **Configuration**: CSP hardened, no unsafe-inline scripts

### Desktop Features
- System tray support
- Native file dialogs
- Offline functionality
- Auto-updater support
- Cross-platform (Windows, macOS, Linux)

## Next Steps
1. **Immediate**: Complete core MCP server implementation
2. **Week 1-2**: PDF processing pipeline
3. **Week 3-4**: Search engine with hybrid approach
4. **Week 5-6**: Campaign management backend
5. **Week 7-8**: MCP tools implementation
6. **Week 9-10**: Production deployment

## Quick Reference

### Running the Project
```bash
# Backend MCP Server
python src/main.py

# Bridge Server (WebSocket/SSE)
python src/bridge/bridge_server.py

# Frontend Development
cd frontend && npm run dev

# Run All Services
docker-compose up
```

### Testing
```bash
# Backend Tests
pytest tests/ --cov=src

# Frontend Tests
cd frontend && npm test

# E2E Tests
cd frontend && npx playwright test

# Load Tests
locust -f tests/locustfile.py
```

### Key Files
- WebSocket Client: `/frontend/src/lib/realtime/websocket-client.ts`
- SSE Client: `/frontend/src/lib/realtime/sse-client.ts`
- Collaboration Store: `/frontend/src/lib/stores/collaboration.svelte.ts`
- Provider Store: `/frontend/src/lib/stores/providers.svelte.ts`
- Bridge Server: `/src/bridge/bridge_server.py`
- **Desktop MCP Bridge**: `/desktop/frontend/src-tauri/src/mcp_bridge.rs`
- **Robust MCP Client**: `/desktop/frontend/src/lib/mcp-robust-client.ts`
- **Desktop Build Script**: `/desktop/build_installer.py`