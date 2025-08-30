# TTRPG Assistant Desktop Application

A native desktop application for the TTRPG Assistant MCP Server, built with Tauri for minimal footprint and maximum performance.

## Architecture

The desktop application uses a multi-layer architecture:

```
┌─────────────────────────────────────────────┐
│            Tauri Application                 │
├─────────────────────────────────────────────┤
│  SvelteKit Frontend (WebView)                │
│              ↕ IPC                           │
│  Rust Backend (Process Management)           │
│              ↕ stdio                         │
│  Python MCP Server (subprocess)              │
└─────────────────────────────────────────────┘
```

### Key Components

- **Frontend**: SvelteKit application running in native WebView
- **Tauri Backend**: Rust-based process management and stdio bridge
- **Python MCP Server**: Subprocess using native MCP stdio protocol
- **Communication**: JSON-RPC 2.0 over stdin/stdout (native MCP protocol)

## Features

### Native Integration
- System tray support with status indicators
- Native file dialogs and drag-and-drop
- OS notifications
- File associations (.ttrpg files)
- Auto-start on system boot (optional)

### Performance
- < 70MB total application size
- < 2 second startup time
- < 150MB RAM usage when idle
- < 5ms IPC latency
- Full offline functionality

### Security
- Sandboxed process execution
- Encrypted credential storage
- CSP enforcement in WebView
- Minimal permission requirements

## Development Setup

### Prerequisites

1. **Node.js** (v18+): https://nodejs.org/
2. **Rust**: https://rustup.rs/
3. **Python** (3.11+): https://www.python.org/
4. **Tauri CLI**: 
   ```bash
   npm install -g @tauri-apps/cli
   ```

### Quick Start

1. **Clone and navigate to desktop directory**:
   ```bash
   cd desktop
   ```

2. **Install Python dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

3. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Generate icons** (optional):
   ```bash
   python tools/generate_icons.py --create-placeholder
   ```

5. **Run in development mode**:
   ```bash
   # Start Python backend
   cd backend
   python websocket_adapter.py &
   
   # Start Tauri frontend
   cd ../frontend
   npm run tauri dev
   ```

## Building for Production

### Automated Build

Use the build script for one-command builds:

```bash
python build_installer.py
```

This will:
1. Build the Python backend into a standalone executable
2. Build the frontend assets
3. Package everything with Tauri
4. Create platform-specific installers

### Manual Build Steps

1. **Build Python backend**:
   ```bash
   cd backend
   pyinstaller pyinstaller.spec  # or use PyOxidizer
   ```

2. **Build frontend**:
   ```bash
   cd frontend
   npm run build
   ```

3. **Build Tauri app**:
   ```bash
   npm run tauri build
   ```

### Platform-Specific Builds

#### Windows
```bash
npm run tauri build -- --target x86_64-pc-windows-msvc
```
Creates: `.msi` and `.exe` installers

#### macOS
```bash
npm run tauri build -- --target x86_64-apple-darwin
```
Creates: `.dmg` and `.app` bundles

#### Linux
```bash
npm run tauri build -- --target x86_64-unknown-linux-gnu
```
Creates: `.AppImage`, `.deb`, and `.rpm` packages

## Project Structure

```
desktop/
├── backend/                 # Python MCP server with WebSocket adapter
│   ├── websocket_adapter.py # WebSocket server wrapping MCP
│   ├── requirements.txt     # Python dependencies
│   └── pyinstaller.spec     # PyInstaller configuration
│
├── frontend/                # SvelteKit application
│   ├── src/
│   │   ├── lib/
│   │   │   └── mcp-client.ts  # WebSocket client for MCP
│   │   └── routes/            # SvelteKit pages
│   │
│   ├── src-tauri/           # Tauri backend
│   │   ├── src/
│   │   │   └── main.rs      # Rust backend code
│   │   ├── tauri.conf.json  # Tauri configuration
│   │   ├── icons/           # Application icons
│   │   ├── installer/       # NSIS installer assets
│   │   └── wix/            # WiX installer assets
│   │
│   └── static/
│       └── fonts/          # Custom fonts (Inter, Iosevka)
│
├── tools/                  # Build and development tools
│   └── generate_icons.py   # Icon generation script
│
└── build_installer.py      # Main build script
```

## Configuration

### Tauri Configuration

The main configuration is in `frontend/src-tauri/tauri.conf.json`:

```json
{
  "package": {
    "productName": "TTRPG Assistant",
    "version": "1.0.0"
  },
  "tauri": {
    "windows": [{
      "title": "TTRPG Assistant",
      "width": 1200,
      "height": 800
    }]
  }
}
```

### Python Backend Configuration

Configure the WebSocket server in `backend/config.json`:

```json
{
  "host": "127.0.0.1",
  "port": 8765,
  "cors_origins": ["tauri://localhost"],
  "max_connections": 10
}
```

## Visual Assets

### Required Icons

Generate all required icons from a single source image:

```bash
python tools/generate_icons.py assets/logo-source.png
```

This creates:
- Application icons (16x16 to 512x512)
- System tray icons (with states)
- Windows Store tiles
- Installer graphics

### Icon Structure

```
frontend/src-tauri/icons/
├── icon.ico              # Windows icon
├── icon.icns            # macOS icon
├── icon.png             # Linux/source icon
├── 32x32.png           # Small icons
├── 128x128.png         # Medium icons
├── 128x128@2x.png      # High DPI
└── tray/               # System tray icons
    ├── icon.ico
    ├── icon-active.ico
    ├── icon-error.ico
    └── icon-syncing.ico
```

## Troubleshooting

### Common Issues

1. **WebSocket connection fails**:
   - Ensure Python backend is running on port 8765
   - Check firewall settings
   - Verify CORS configuration

2. **High memory usage**:
   - Check for memory leaks in Python backend
   - Ensure ChromaDB is properly configured
   - Monitor WebSocket connections

3. **Slow startup**:
   - Pre-compile Python bytecode
   - Use PyOxidizer for faster Python startup
   - Lazy-load heavy modules

### Debug Mode

Run in debug mode for detailed logging:

```bash
# Set environment variable
export RUST_LOG=debug

# Run with debug build
npm run tauri dev
```

### Logs Location

- **Windows**: `%APPDATA%\ttrpg-assistant\logs\`
- **macOS**: `~/Library/Application Support/ttrpg-assistant/logs/`
- **Linux**: `~/.config/ttrpg-assistant/logs/`

## Distribution

### Code Signing

#### Windows
1. Obtain a code signing certificate
2. Configure in `tauri.conf.json`:
   ```json
   "windows": {
     "certificateThumbprint": "YOUR_CERT_THUMBPRINT"
   }
   ```

#### macOS
1. Enroll in Apple Developer Program
2. Create signing certificates
3. Configure in build process

### Auto-Updates

Configure auto-updates in `tauri.conf.json`:

```json
"updater": {
  "active": true,
  "endpoints": [
    "https://github.com/yourusername/ttrpg-assistant/releases/latest/download/latest.json"
  ]
}
```

## Contributing

See the main project [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

This desktop application is part of the TTRPG Assistant project. See [LICENSE](../LICENSE) for details.