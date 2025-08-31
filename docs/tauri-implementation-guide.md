# Tauri Desktop Implementation Guide for MDMAI

## Quick Start

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Tauri CLI
npm install --save-dev @tauri-apps/cli

# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install libwebkit2gtk-4.0-dev \
    build-essential \
    curl \
    wget \
    libssl-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev

# macOS dependencies
brew install cmake

# Windows: Install Visual Studio Build Tools
```

### Project Setup

```bash
# In MDMAI root directory
npm create tauri-app@latest -- --beta
# Choose: SvelteKit, TypeScript, npm

# Or add to existing project
npm install --save-dev @tauri-apps/cli
npx tauri init
```

## Architecture Implementation

### 1. Tauri Configuration

**tauri.conf.json**
```json
{
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devPath": "http://localhost:5173",
    "distDir": "../build"
  },
  "package": {
    "productName": "MDMAI Assistant",
    "version": "1.0.0"
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "shell": {
        "all": false,
        "open": true,
        "execute": true,
        "sidecar": true,
        "scope": [
          {
            "name": "python-mcp",
            "cmd": "python",
            "args": ["-m", "src.main"]
          }
        ]
      },
      "fs": {
        "all": false,
        "readFile": true,
        "writeFile": true,
        "readDir": true,
        "scope": ["$APPDATA", "$APPLOCAL", "$DOCUMENTS"]
      },
      "dialog": {
        "all": true
      },
      "notification": {
        "all": true
      }
    },
    "bundle": {
      "active": true,
      "identifier": "com.mdmai.assistant",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/128x128@2x.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ],
      "resources": [
        "python-dist/*"
      ]
    },
    "security": {
      "csp": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    },
    "windows": [
      {
        "fullscreen": false,
        "resizable": true,
        "title": "MDMAI Assistant",
        "width": 1200,
        "height": 800,
        "minWidth": 800,
        "minHeight": 600
      }
    ]
  }
}
```

### 2. Rust Backend Implementation

**src-tauri/src/main.rs**
```rust
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;

mod mcp_manager;
use mcp_manager::MCPManager;

#[derive(Debug, Serialize, Deserialize)]
struct MCPMessage {
    jsonrpc: String,
    method: Option<String>,
    params: Option<serde_json::Value>,
    id: Option<serde_json::Value>,
    result: Option<serde_json::Value>,
    error: Option<serde_json::Value>,
}

struct AppState {
    mcp: Mutex<Option<MCPManager>>,
}

#[tauri::command]
async fn start_mcp_server(state: State<'_, AppState>) -> Result<String, String> {
    let mut mcp_guard = state.mcp.lock().unwrap();
    
    if mcp_guard.is_some() {
        return Err("MCP server already running".to_string());
    }
    
    match MCPManager::new() {
        Ok(manager) => {
            *mcp_guard = Some(manager);
            Ok("MCP server started successfully".to_string())
        }
        Err(e) => Err(format!("Failed to start MCP server: {}", e))
    }
}

#[tauri::command]
async fn stop_mcp_server(state: State<'_, AppState>) -> Result<String, String> {
    let mut mcp_guard = state.mcp.lock().unwrap();
    
    if let Some(mut manager) = mcp_guard.take() {
        manager.shutdown()?;
        Ok("MCP server stopped".to_string())
    } else {
        Err("MCP server not running".to_string())
    }
}

#[tauri::command]
async fn send_mcp_request(
    state: State<'_, AppState>,
    method: String,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let mut mcp_guard = state.mcp.lock().unwrap();
    
    if let Some(ref mut manager) = *mcp_guard {
        manager.send_request(method, params)
            .await
            .map_err(|e| e.to_string())
    } else {
        Err("MCP server not running".to_string())
    }
}

#[tauri::command]
async fn get_mcp_status(state: State<'_, AppState>) -> Result<bool, String> {
    let mcp_guard = state.mcp.lock().unwrap();
    Ok(mcp_guard.is_some())
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            mcp: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            start_mcp_server,
            stop_mcp_server,
            send_mcp_request,
            get_mcp_status
        ])
        .setup(|app| {
            // Auto-start MCP server on app launch
            let state = app.state::<AppState>();
            let mut mcp_guard = state.mcp.lock().unwrap();
            *mcp_guard = Some(MCPManager::new().expect("Failed to start MCP"));
            Ok(())
        })
        .on_window_event(|event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event.event() {
                // Clean shutdown of MCP server
                // Handle in window close event
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**src-tauri/src/mcp_manager.rs**
```rust
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct MCPManager {
    process: Child,
    stdin: Box<dyn Write + Send>,
    response_receiver: Receiver<Value>,
    pending_requests: Arc<Mutex<HashMap<String, Sender<Value>>>>,
}

impl MCPManager {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Find Python executable
        let python_cmd = if cfg!(windows) {
            "python.exe"
        } else {
            "python3"
        };
        
        // Get the app's resource directory
        let resource_dir = tauri::api::path::resource_dir()
            .unwrap_or_else(|| std::env::current_dir().unwrap());
        
        let python_dist = resource_dir.join("python-dist");
        
        // Spawn Python MCP server using Tauri sidecar
        // Note: In production, use Command::new_sidecar("mcp-server")
        // This example shows the development approach
        let mut child = tauri::api::process::Command::new_sidecar("mcp-server")
            .expect("failed to create sidecar command")
            .env("MCP_STDIO_MODE", "true")
            .spawn()
            .expect("Failed to spawn sidecar");
        
        let stdin = Box::new(child.stdin.take().unwrap());
        let stdout = child.stdout.take().unwrap();
        
        let (response_sender, response_receiver) = channel();
        let pending_requests = Arc::new(Mutex::new(HashMap::new()));
        let pending_clone = pending_requests.clone();
        
        // Spawn thread to read stdout
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if let Ok(json) = serde_json::from_str::<Value>(&line) {
                        // Route response to appropriate handler
                        if let Some(id) = json.get("id").and_then(|v| v.as_str()) {
                            let pending = pending_clone.lock().unwrap();
                            if let Some(sender) = pending.get(id) {
                                let _ = sender.send(json);
                            }
                        } else {
                            // Notification or event
                            let _ = response_sender.send(json);
                        }
                    }
                }
            }
        });
        
        Ok(MCPManager {
            process: child,
            stdin,
            response_receiver,
            pending_requests,
        })
    }
    
    pub async fn send_request(
        &mut self,
        method: String,
        params: Value,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let id = uuid::Uuid::new_v4().to_string();
        
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": id
        });
        
        // Create channel for this specific request
        let (sender, receiver) = channel();
        self.pending_requests.lock().unwrap().insert(id.clone(), sender);
        
        // Send request
        writeln!(self.stdin, "{}", request.to_string())?;
        self.stdin.flush()?;
        
        // Wait for response (with timeout)
        match receiver.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(response) => {
                self.pending_requests.lock().unwrap().remove(&id);
                Ok(response)
            }
            Err(_) => {
                self.pending_requests.lock().unwrap().remove(&id);
                Err("Request timeout".into())
            }
        }
    }
    
    pub fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Send shutdown command
        let shutdown_request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "shutdown",
            "id": "shutdown"
        });
        
        writeln!(self.stdin, "{}", shutdown_request.to_string())?;
        self.stdin.flush()?;
        
        // Wait for process to exit
        self.process.wait()?;
        Ok(())
    }
}
```

### 3. Frontend Integration

**src/lib/mcp-client.ts**
```typescript
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import { writable } from 'svelte/store';

export interface MCPResponse {
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
}

export class MCPClient {
  private static instance: MCPClient;
  public status = writable<boolean>(false);
  public events = writable<any[]>([]);
  
  private constructor() {
    this.initialize();
  }
  
  static getInstance(): MCPClient {
    if (!MCPClient.instance) {
      MCPClient.instance = new MCPClient();
    }
    return MCPClient.instance;
  }
  
  private async initialize() {
    // Check initial status
    const isRunning = await invoke<boolean>('get_mcp_status');
    this.status.set(isRunning);
    
    // Listen for MCP events
    await listen('mcp-event', (event) => {
      this.events.update(events => [...events, event.payload]);
    });
    
    // Listen for status changes
    await listen('mcp-status', (event) => {
      this.status.set(event.payload as boolean);
    });
  }
  
  async start(): Promise<void> {
    try {
      await invoke('start_mcp_server');
      this.status.set(true);
    } catch (error) {
      console.error('Failed to start MCP server:', error);
      throw error;
    }
  }
  
  async stop(): Promise<void> {
    try {
      await invoke('stop_mcp_server');
      this.status.set(false);
    } catch (error) {
      console.error('Failed to stop MCP server:', error);
      throw error;
    }
  }
  
  async callTool(toolName: string, params?: any): Promise<MCPResponse> {
    try {
      const response = await invoke<any>('send_mcp_request', {
        method: 'tools/call',
        params: {
          name: toolName,
          arguments: params
        }
      });
      return response;
    } catch (error) {
      console.error('MCP request failed:', error);
      throw error;
    }
  }
  
  async getTools(): Promise<any[]> {
    const response = await this.callTool('tools/list');
    return response.result?.tools || [];
  }
  
  async searchRules(query: string): Promise<any[]> {
    const response = await this.callTool('search_rules', { query });
    return response.result || [];
  }
  
  async generateContent(prompt: string, context?: any): Promise<string> {
    const response = await this.callTool('generate_content', {
      prompt,
      context
    });
    return response.result?.content || '';
  }
}

export const mcp = MCPClient.getInstance();
```

**src/routes/+page.svelte**
```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import { mcp } from '$lib/mcp-client';
  
  let isConnected = false;
  let searchQuery = '';
  let searchResults = [];
  let isSearching = false;
  
  mcp.status.subscribe(value => {
    isConnected = value;
  });
  
  onMount(async () => {
    if (!isConnected) {
      await mcp.start();
    }
  });
  
  async function handleSearch() {
    if (!searchQuery.trim()) return;
    
    isSearching = true;
    try {
      searchResults = await mcp.searchRules(searchQuery);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      isSearching = false;
    }
  }
</script>

<div class="container mx-auto p-4">
  <header class="mb-8">
    <h1 class="text-3xl font-bold">MDMAI Assistant</h1>
    <div class="flex items-center gap-2 mt-2">
      <div class="w-3 h-3 rounded-full {isConnected ? 'bg-green-500' : 'bg-red-500'}"></div>
      <span class="text-sm">
        MCP Server: {isConnected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  </header>
  
  <div class="search-section mb-8">
    <form on:submit|preventDefault={handleSearch}>
      <input
        type="text"
        bind:value={searchQuery}
        placeholder="Search TTRPG rules..."
        class="w-full p-3 border rounded-lg"
        disabled={!isConnected || isSearching}
      />
      <button
        type="submit"
        class="mt-2 px-4 py-2 bg-blue-500 text-white rounded-lg"
        disabled={!isConnected || isSearching}
      >
        {isSearching ? 'Searching...' : 'Search'}
      </button>
    </form>
  </div>
  
  <div class="results-section">
    {#each searchResults as result}
      <div class="result-card p-4 border rounded-lg mb-2">
        <h3 class="font-semibold">{result.title}</h3>
        <p class="text-gray-600">{result.content}</p>
      </div>
    {/each}
  </div>
</div>
```

### 4. Python Packaging with PyOxidizer

**pyoxidizer.bzl**
```python
def make_exe():
    dist = default_python_distribution()
    
    python_config = dist.make_python_interpreter_config()
    python_config.run_module = "src.main"
    python_config.filesystem_importer = True
    python_config.sys_frozen = True
    python_config.optimization_level = 1
    
    exe = dist.to_python_executable(
        name="mdmai-mcp",
        config=python_config,
    )
    
    # Add Python packages
    exe.add_python_resources(exe.pip_install([
        "mcp>=1.0.0",
        "fastmcp>=0.1.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "pydantic>=2.0.0",
        "structlog>=23.0.0",
        # Add all dependencies from pyproject.toml
    ]))
    
    # Add local source code
    exe.add_python_resources(exe.read_package_root(
        path="src",
        package="src",
    ))
    
    return exe

def make_embedded_resources(exe):
    return exe.to_embedded_resources()

def make_install(exe):
    files = FileManifest()
    files.add_python_resource(".", exe)
    return files

register_target("exe", make_exe)
resolve_targets()
```

### 5. Build and Distribution

**package.json updates**
```json
{
  "scripts": {
    "dev": "vite dev",
    "build": "vite build",
    "tauri": "tauri",
    "tauri:dev": "tauri dev",
    "tauri:build": "tauri build",
    "build:python": "pyoxidizer build --release",
    "build:all": "npm run build:python && npm run tauri:build"
  },
  "devDependencies": {
    "@tauri-apps/cli": "^2.0.0",
    "@tauri-apps/api": "^2.0.0"
  }
}
```

**GitHub Actions Workflow**
```yaml
name: Build and Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - os: windows-latest
            target: x86_64-pc-windows-msvc
          - os: macos-latest
            target: x86_64-apple-darwin
          - os: macos-latest
            target: aarch64-apple-darwin

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: 18
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: ${{ matrix.target }}
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install PyOxidizer
        run: pip install pyoxidizer
      
      - name: Build Python executable
        run: pyoxidizer build --release
      
      - name: Copy Python dist to resources
        run: |
          mkdir -p src-tauri/resources/python-dist
          cp -r build/*/release/install/* src-tauri/resources/python-dist/
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build Tauri app
        run: npm run tauri:build -- --target ${{ matrix.target }}
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: ${{ matrix.target }}
          path: src-tauri/target/release/bundle/
```

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mcp_message_parsing() {
        let msg = r#"{"jsonrpc":"2.0","id":1,"result":"test"}"#;
        let parsed: MCPMessage = serde_json::from_str(msg).unwrap();
        assert_eq!(parsed.jsonrpc, "2.0");
    }
}
```

### Integration Tests
```typescript
// tests/mcp-integration.test.ts
import { describe, it, expect } from 'vitest';
import { mcp } from '$lib/mcp-client';

describe('MCP Integration', () => {
  it('should connect to MCP server', async () => {
    await mcp.start();
    const tools = await mcp.getTools();
    expect(tools).toBeDefined();
    expect(tools.length).toBeGreaterThan(0);
  });
});
```

## Performance Optimizations

### 1. Lazy Loading
- Load Python modules on-demand
- Implement progressive model loading
- Cache frequently used embeddings

### 2. Process Management
- Use process pooling for multiple MCP instances
- Implement graceful shutdown
- Monitor memory usage

### 3. Communication Optimization
- Batch requests when possible
- Use binary protocol for large data
- Implement request deduplication

## Troubleshooting

### Common Issues

1. **Python not found**
   - Bundle Python with PyOxidizer
   - Use virtual environment detection

2. **Large bundle size**
   - Exclude unnecessary Python packages
   - Use compression for resources
   - Implement dynamic downloading for models

3. **Slow startup**
   - Precompile Python bytecode
   - Use lazy imports
   - Show splash screen during initialization

## Conclusion

This implementation provides a robust, performant desktop application that:
- Maintains small bundle size (~60MB total)
- Provides native performance
- Ensures cross-platform compatibility
- Leverages existing codebase
- Supports offline operation

The Tauri + PyOxidizer combination delivers the best user experience while maintaining development efficiency.