# MDMAI Desktop Application Setup Guide

## Phase 23.1: Tauri Development Environment Setup

This document describes the setup and configuration of the Tauri-based desktop application for MDMAI.

## Overview

The desktop application uses:
- **Tauri v2.1** - Rust-based desktop framework
- **SvelteKit** - Frontend framework (existing)
- **Python MCP Server** - Backend service running via stdio
- **Stdio Communication** - Direct process communication (not WebSocket)

## Architecture

```
┌─────────────────────────────────────────┐
│         Tauri Application Window         │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │    SvelteKit Frontend (WebView)    │  │
│  │                                    │  │
│  │  - TypeScript MCP Client           │  │
│  │  - Error-as-values pattern         │  │
│  │  - Responsive UI with Tailwind     │  │
│  └──────────────┬─────────────────────┘  │
│                 │ Tauri IPC              │
│  ┌──────────────▼─────────────────────┐  │
│  │     Rust Backend (src-tauri)       │  │
│  │                                    │  │
│  │  - Process Management              │  │
│  │  - Stdio Bridge (mcp_bridge.rs)    │  │
│  │  - JSON-RPC 2.0 Protocol           │  │
│  └──────────────┬─────────────────────┘  │
│                 │ stdio                  │
│  ┌──────────────▼─────────────────────┐  │
│  │   Python MCP Server (Sidecar)      │  │
│  │                                    │  │
│  │  - FastMCP Framework               │  │
│  │  - ChromaDB Vector Store           │  │
│  │  - TTRPG Tools & Services          │  │
│  └────────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Directory Structure

```
MDMAI/
├── desktop/
│   ├── frontend/               # Desktop-specific frontend
│   │   ├── src/                # SvelteKit source
│   │   │   ├── lib/
│   │   │   │   ├── mcp-stdio-bridge.ts    # TypeScript MCP client
│   │   │   │   └── mcp-robust-client.ts   # Robust client with reconnection
│   │   │   └── routes/         # SvelteKit routes
│   │   ├── src-tauri/          # Tauri backend
│   │   │   ├── src/
│   │   │   │   ├── main.rs     # Tauri entry point
│   │   │   │   └── mcp_bridge.rs  # MCP stdio bridge
│   │   │   ├── binaries/       # Platform-specific executables
│   │   │   │   └── mcp-server-x86_64-unknown-linux-gnu
│   │   │   ├── Cargo.toml      # Rust dependencies
│   │   │   └── tauri.conf.json # Tauri configuration
│   │   └── package.json        # Node dependencies
│   ├── backend/                # Python backend build scripts
│   │   ├── mcp_stdio_wrapper.py
│   │   ├── pyinstaller.spec
│   │   └── build.sh
│   └── tools/                  # Build tools
└── src/                        # Main Python MCP server source

```

## Prerequisites

### System Requirements

1. **Rust & Cargo** (v1.70+)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Node.js** (v18+) and npm
   ```bash
   # Install via nvm or system package manager
   ```

3. **Python** (v3.11+)
   ```bash
   python3 --version  # Should be 3.11 or higher
   ```

4. **System Dependencies** (Linux)
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install libwebkit2gtk-4.0-dev \
       build-essential \
       curl \
       wget \
       libssl-dev \
       libgtk-3-dev \
       libayatana-appindicator3-dev \
       librsvg2-dev

   # Fedora
   sudo dnf install webkit2gtk4.0-devel \
       openssl-devel \
       gtk3-devel \
       libappindicator-gtk3-devel \
       librsvg2-devel
   ```

   **Note**: On WSL2, most dependencies are handled by Windows, so the Linux-specific dependencies may not be needed.

## Setup Instructions

### 1. Install Dependencies

```bash
# Navigate to desktop frontend directory
cd desktop/frontend

# Install npm dependencies (includes Tauri CLI)
npm install
```

### 2. Build Frontend

```bash
# Build the SvelteKit frontend
npm run build
```

### 3. Configure Python Sidecar

The Python MCP server runs as a sidecar process. For development, we use a Python script directly. For production, it should be compiled with PyInstaller.

**Development Setup:**
- The sidecar script is at: `desktop/frontend/src-tauri/binaries/mcp-server-x86_64-unknown-linux-gnu`
- It's configured to run the Python MCP server in stdio mode
- No compilation needed for development

**Production Setup:**
```bash
# Build Python executable (optional for development)
cd desktop/backend
./build.sh
```

### 4. Run Development Server

```bash
cd desktop/frontend

# Run Tauri in development mode
npm run tauri:dev
```

This will:
1. Start the SvelteKit dev server
2. Launch the Tauri application window
3. Automatically start the Python MCP server as a sidecar process
4. Enable hot-reload for frontend changes

### 5. Build for Production

```bash
cd desktop/frontend

# Build for production
npm run tauri:build
```

This creates platform-specific installers in `src-tauri/target/release/bundle/`

## Configuration

### Tauri Configuration (`tauri.conf.json`)

Key settings:
- **productName**: "TTRPG Assistant"
- **identifier**: "com.ttrpg.assistant"
- **externalBin**: Points to the MCP server sidecar
- **CSP**: Configured for security with Tauri IPC allowed

### MCP Bridge Configuration

The Rust backend (`mcp_bridge.rs`) handles:
- Process spawning and management
- Stdio communication with Python
- JSON-RPC 2.0 protocol handling
- Automatic reconnection on failure

### TypeScript Client Configuration

The TypeScript client (`mcp-stdio-bridge.ts`) provides:
- Error-as-values pattern for all operations
- Automatic reconnection logic
- Health monitoring
- Type-safe method calls

## Development Workflow

### Frontend Development

1. Make changes to SvelteKit components in `desktop/frontend/src/`
2. Changes hot-reload automatically in dev mode
3. Use TypeScript for type safety
4. Follow error-as-values pattern:
   ```typescript
   const result = await mcpBridge.search(query);
   if (result.ok) {
       // Handle success
       console.log(result.data);
   } else {
       // Handle error
       console.error(result.error);
   }
   ```

### Backend Development

1. Python MCP server changes in `src/`
2. Restart Tauri dev server to reload Python changes
3. Use stdio mode for communication (set via `MCP_STDIO_MODE=true`)

### Rust Backend Development

1. Changes to `src-tauri/src/` files
2. Tauri automatically rebuilds Rust code
3. Window reloads after Rust compilation

## Testing

### Unit Tests

```bash
# Frontend tests
cd desktop/frontend
npm test

# Python tests
cd ../..
pytest tests/

# Rust tests
cd desktop/frontend/src-tauri
cargo test
```

### Integration Testing

Test the stdio communication:
```bash
# Run the test script
python3 desktop/test_mcp_stdio.py
```

### Manual Testing Checklist

- [ ] Application launches without errors
- [ ] MCP server connects (green status indicator)
- [ ] Search functionality works
- [ ] Error handling displays user-friendly messages
- [ ] Reconnection works after MCP server restart
- [ ] Window controls (minimize, maximize, close) work
- [ ] Application closes cleanly

## Troubleshooting

### Common Issues

1. **"MCP server not found" error**
   - Check Python path in sidecar script
   - Ensure Python dependencies are installed
   - Verify `MCP_STDIO_MODE` environment variable is set

2. **Build fails with "frontendDist not found"**
   - Run `npm run build` first
   - Check that `build/` directory exists

3. **Rust compilation errors**
   - Update Rust: `rustup update`
   - Clean build: `cargo clean`
   - Rebuild: `cargo build`

4. **Python import errors**
   - Check Python version (3.11+)
   - Install requirements: `pip install -r requirements.txt`
   - Verify PYTHONPATH includes project directories

### Debug Mode

Enable debug logging:
```bash
# Set environment variables
export RUST_LOG=debug
export MCP_DEBUG=true

# Run with debug output
npm run tauri:dev
```

## Next Steps

### Immediate Tasks
- [x] Set up Tauri development environment
- [x] Configure stdio communication
- [x] Integrate with existing SvelteKit frontend
- [ ] Implement comprehensive error handling
- [ ] Add automated tests for IPC communication
- [ ] Create CI/CD pipeline for releases

### Future Enhancements
- [ ] Code signing for distribution
- [ ] Auto-updater configuration
- [ ] Performance profiling and optimization
- [ ] Accessibility improvements
- [ ] Internationalization support

## Resources

- [Tauri Documentation](https://tauri.app/v2/guides/)
- [SvelteKit Documentation](https://kit.svelte.dev/docs)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)

## Summary

The Tauri development environment is now configured with:
- ✅ Rust and Cargo installed
- ✅ Tauri CLI tools set up
- ✅ Development environment configured
- ✅ Tauri project structure created
- ✅ Integration with existing SvelteKit frontend
- ✅ Stdio communication with Python MCP server
- ✅ TypeScript client with error-as-values pattern
- ✅ Development and production build scripts

The desktop application is ready for development and testing.