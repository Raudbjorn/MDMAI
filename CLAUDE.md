# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TTRPG Assistant MCP Server - A Model Context Protocol server for tabletop RPG assistance. Provides AI-powered rules search, campaign management, and real-time collaboration for Dungeon Masters.

## Build & Development Commands

All commands use the unified build script that auto-detects package managers (uv/poetry/pip for Python, pnpm/yarn/npm for Node.js).

```bash
# Setup (one-time)
./build.sh setup

# Development servers
./build.sh dev-backend     # Python MCP server (stdio mode)
./build.sh dev-webapp      # SvelteKit web app (http://localhost:5173)
./build.sh dev-desktop     # Tauri desktop app with hot reload

# Build
./build.sh build           # Build all components
./build.sh build python    # Python backend only
./build.sh build js        # SvelteKit frontend only
./build.sh build rust      # Tauri/Rust only

# Quality
./build.sh test            # Run all tests
./build.sh test python     # Python tests only (pytest)
./build.sh lint            # Lint all code
./build.sh format          # Format all code

# Run single Python test
pytest tests/path/to/test.py::test_function -v

# Frontend tests
cd frontend && npm test
cd frontend && npx playwright test  # E2E tests
```

## Architecture

### Three-Platform System
1. **MCP Server** (`src/`): Python FastMCP server communicating via stdin/stdout (JSON-RPC 2.0)
2. **Web App** (`frontend/`): SvelteKit with WebSocket bridge to MCP server
3. **Desktop App** (`desktop/frontend/`): Tauri (Rust) + SvelteKit, direct stdio to Python subprocess

### Communication Patterns
- **Desktop**: Tauri spawns Python as sidecar → JSON-RPC over stdin/stdout → <1ms latency
- **Web**: Browser → WebSocket → Bridge Server (`src/bridge/bridge_server.py`) → MCP Server
- **MCP Tools**: Defined in `src/tools/`, registered via `@mcp.tool()` decorator

### Key Directories
```
src/
├── main.py                    # MCP server entry point
├── tools/                     # MCP tool implementations
├── bridge/bridge_server.py    # WebSocket-to-MCP bridge (FastAPI)
├── model_selection/           # AI model selection and A/B testing
├── ai_providers/              # Multi-provider AI integration (Anthropic, OpenAI, Google)
└── search/                    # Hybrid vector + keyword search

frontend/src/lib/
├── stores/                    # Svelte 5 rune-based state management
├── components/                # Shared UI components
├── realtime/                  # WebSocket/SSE clients
└── api/                       # API clients for each provider

desktop/frontend/
├── src-tauri/src/mcp_bridge.rs  # Rust↔Python stdio communication
└── src/lib/mcp-robust-client.ts # TypeScript MCP client with retry logic
```

## Code Patterns

### Python (Backend)
```python
# MCP Tool pattern
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("TTRPG")

@mcp.tool()
async def tool_name(param: str) -> Dict[str, Any]:
    """Tool documentation for LLM."""
    # Implementation with async/await
```

### TypeScript/Svelte (Frontend)
```typescript
// Error-as-values pattern (used throughout)
type Result<T> = { ok: true; data: T } | { ok: false; error: string };

// Svelte 5 runes for state
let state = $state<StateType>({});
const computed = $derived(state.value * 2);
$effect(() => { /* side effects */ });

// Desktop detection
if (browser && window.__TAURI__) {
    // Desktop-specific code (use Tauri IPC)
} else {
    // Web-specific code (use fetch/WebSocket)
}
```

### Database
- **ChromaDB**: Vector embeddings for semantic search
- **SQLite**: Structured data (campaigns, sessions, characters)
- Collections: `rulebooks`, `campaigns`, `sessions`, `personalities`

## Key Constraints

- **Error Handling**: Use Result types, not exceptions. Never silently fail.
- **Async**: All MCP tools must be async. Use `async/await` throughout.
- **Type Safety**: Python type hints required. TypeScript strict mode enabled.
- **Hybrid Search**: Always combine semantic (vector) + keyword (BM25) search.
- **stdio Communication**: Desktop app uses stdin/stdout, not WebSocket.

## AI Provider Integration

Located in `src/ai_providers/` and `src/model_selection/`:
- Multi-provider support: Anthropic, OpenAI, Google, Ollama (local)
- Automatic failover and cost optimization
- A/B testing framework for model comparison
- Context-aware model selection based on task type

## Configuration

Config files in `src/model_selection/config/`:
- `decision_tree_config.yaml`: Model selection rules
- `preference_learner_config.yaml`: User preference learning

Environment variables in `.env`:
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
- `DATABASE_PATH`, `VECTOR_DB_PATH`
- `MCP_STDIO_MODE=true` (recommended for desktop)
