# Tauri + PyOxidizer Integration Guide

This document explains how to integrate the PyOxidizer-packaged MCP server with the existing Tauri desktop application, replacing the development sidecar setup with production-ready standalone executables.

## Current vs New Architecture

### Current Development Setup
```
Tauri App → Python Sidecar → src/main.py (requires Python installation)
```

### New Production Setup  
```
Tauri App → PyOxidizer Executable (standalone, no Python required)
```

## Integration Steps

### 1. Build the PyOxidizer Executable

First, create the standalone executable:

```bash
# Install PyOxidizer
./scripts/install_pyoxidizer.sh

# Build for all platforms
python scripts/build_pyoxidizer.py --all

# Or build for specific platform
python scripts/build_pyoxidizer.py --platform linux
```

### 2. Update Tauri Configuration

#### A. Copy Executable to Tauri Directory

```bash
# Copy Linux executable
cp dist/pyoxidizer/mdmai-mcp-server-linux-x86_64/mdmai-mcp-server desktop/backend/

# Copy Windows executable  
cp dist/pyoxidizer/mdmai-mcp-server-windows-x86_64/mdmai-mcp-server.exe desktop/backend/

# Copy macOS executable
cp dist/pyoxidizer/mdmai-mcp-server-macos-x86_64/mdmai-mcp-server desktop/backend/
```

#### B. Update `tauri.conf.json`

Modify `desktop/frontend/src-tauri/tauri.conf.json`:

```json
{
  "tauri": {
    "bundle": {
      "externalBin": [
        "mdmai-mcp-server",
        "mdmai-mcp-server.exe"
      ]
    },
    "allowlist": {
      "shell": {
        "sidecar": true,
        "scope": [
          {
            "name": "mdmai-mcp-server",
            "cmd": "mdmai-mcp-server",
            "args": true
          }
        ]
      }
    }
  }
}
```

#### C. Update Rust Sidecar Code

Modify the Rust code to use the appropriate executable:

```rust
// In src-tauri/src/main.rs or relevant module

use tauri::api::shell::{Command, CommandEvent};

fn start_mcp_server() -> Result<tauri::api::shell::CommandChild, String> {
    // Determine executable name based on platform
    let exe_name = if cfg!(target_os = "windows") {
        "mdmai-mcp-server.exe"
    } else {
        "mdmai-mcp-server"
    };
    
    // Start the sidecar
    let (mut rx, child) = Command::new_sidecar(exe_name)
        .expect("failed to setup mdmai-mcp-server sidecar")
        .spawn()
        .expect("Failed to spawn sidecar");
    
    // Handle sidecar events
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    println!("MCP Server: {}", line);
                }
                CommandEvent::Stderr(line) => {
                    eprintln!("MCP Server Error: {}", line);
                }
                CommandEvent::Error(error) => {
                    eprintln!("MCP Server Command Error: {}", error);
                }
                CommandEvent::Terminated(payload) => {
                    println!("MCP Server terminated: {:?}", payload);
                    break;
                }
            }
        }
    });
    
    Ok(child)
}
```

### 3. Build Scripts Integration

#### A. Automated Build Pipeline

Create `scripts/build_desktop_with_pyoxidizer.py`:

```python
#!/usr/bin/env python3
"""
Integrated build script that:
1. Builds PyOxidizer executables
2. Copies them to Tauri backend directory
3. Builds Tauri desktop application
"""

import subprocess
import shutil
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    
    # Step 1: Build PyOxidizer executables
    print("Building PyOxidizer executables...")
    subprocess.run([
        "python", "scripts/build_pyoxidizer.py", "--all"
    ], cwd=project_root, check=True)
    
    # Step 2: Copy executables to desktop backend
    backend_dir = project_root / "desktop" / "backend"
    dist_dir = project_root / "dist" / "pyoxidizer"
    
    executables = [
        ("mdmai-mcp-server-linux-x86_64", "mdmai-mcp-server"),
        ("mdmai-mcp-server-windows-x86_64", "mdmai-mcp-server.exe"),
        ("mdmai-mcp-server-macos-x86_64", "mdmai-mcp-server"),
    ]
    
    for package_name, exe_name in executables:
        src = dist_dir / package_name / exe_name
        if src.exists():
            dst = backend_dir / exe_name
            print(f"Copying {src} -> {dst}")
            shutil.copy2(src, dst)
            # Make executable on Unix systems
            if not exe_name.endswith('.exe'):
                dst.chmod(0o755)
    
    # Step 3: Build Tauri application
    print("Building Tauri desktop application...")
    subprocess.run([
        "npm", "run", "tauri", "build"
    ], cwd=project_root / "desktop" / "frontend", check=True)
    
    print("✅ Desktop application built successfully!")

if __name__ == "__main__":
    main()
```

#### B. Update Existing Build Scripts

Modify `desktop/build_installer.py` to use PyOxidizer executables:

```python
# Add to the beginning of build_installer.py

def ensure_pyoxidizer_executables():
    """Ensure PyOxidizer executables are built and available."""
    
    backend_dir = Path(__file__).parent / "backend"
    
    # Check if executables exist
    executables = ["mdmai-mcp-server", "mdmai-mcp-server.exe"]
    missing = [exe for exe in executables if not (backend_dir / exe).exists()]
    
    if missing:
        print(f"Missing executables: {missing}")
        print("Building PyOxidizer executables...")
        
        # Build executables
        subprocess.run([
            "python", "../scripts/build_pyoxidizer.py", "--all"
        ], check=True)
        
        # Copy to backend directory
        # (copy logic here)
```

### 4. Development vs Production Modes

Create a configuration system to switch between development and production modes:

#### A. Environment Variable Control

```rust
// In Tauri Rust code
fn get_mcp_server_command() -> String {
    if std::env::var("MDMAI_DEV_MODE").is_ok() {
        // Development mode: use Python script
        if cfg!(target_os = "windows") {
            "python"
        } else {
            "python3"
        }
    } else {
        // Production mode: use PyOxidizer executable
        if cfg!(target_os = "windows") {
            "mdmai-mcp-server.exe"
        } else {
            "mdmai-mcp-server"
        }
    }
}

fn get_mcp_server_args() -> Vec<String> {
    if std::env::var("MDMAI_DEV_MODE").is_ok() {
        // Development mode: run Python script
        vec![
            "desktop/backend/mcp_stdio_wrapper.py".to_string()
        ]
    } else {
        // Production mode: no additional args needed
        vec![]
    }
}
```

#### B. Package.json Scripts

Update `desktop/frontend/package.json`:

```json
{
  "scripts": {
    "dev": "MDMAI_DEV_MODE=1 tauri dev",
    "build": "tauri build",
    "build-with-pyoxidizer": "python ../../scripts/build_desktop_with_pyoxidizer.py"
  }
}
```

### 5. Testing Integration

#### A. Test PyOxidizer Executable Independently

```bash
# Test the executable works with stdio
python scripts/test_pyoxidizer_stdio.py

# Test specific executable
python scripts/test_pyoxidizer_stdio.py --executable desktop/backend/mdmai-mcp-server
```

#### B. Test Tauri Integration

```bash
# Development mode (uses Python)
cd desktop/frontend
MDMAI_DEV_MODE=1 npm run tauri dev

# Production mode (uses PyOxidizer)
npm run tauri dev
```

#### C. End-to-End Testing

```python
# Add to desktop/test_mcp_stdio.py

def test_pyoxidizer_integration():
    """Test that PyOxidizer executable works with Tauri."""
    
    # Check if executable exists
    exe_path = Path("backend/mdmai-mcp-server")
    if platform.system() == "Windows":
        exe_path = Path("backend/mdmai-mcp-server.exe")
    
    assert exe_path.exists(), f"PyOxidizer executable not found: {exe_path}"
    
    # Test stdio communication
    process = subprocess.Popen(
        [str(exe_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send MCP initialize message
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}}
    }
    
    process.stdin.write(json.dumps(init_msg) + "\n")
    process.stdin.flush()
    
    # Wait for response
    response = process.stdout.readline()
    assert response, "No response from PyOxidizer executable"
    
    # Clean up
    process.terminate()
```

## Deployment Considerations

### 1. File Size

PyOxidizer executables are larger than Python scripts:

- **Development**: ~50KB (Python script)
- **Production**: ~800MB-1.2GB (PyOxidizer executable)

Consider:
- Distributing separate installers for different platforms
- Using compression in installers
- Delta updates for future versions

### 2. Startup Time

- **Cold start**: 5-10 seconds (first run)
- **Warm start**: 2-3 seconds (subsequent runs)

This is acceptable for desktop applications but slower than Python scripts.

### 3. Platform-Specific Builds

Each platform needs its own executable:

```
desktop/
├── backend/
│   ├── mdmai-mcp-server          # Linux
│   ├── mdmai-mcp-server.exe      # Windows  
│   └── mdmai-mcp-server-macos    # macOS (optional rename)
```

### 4. Code Signing

For production distribution:

**Windows:**
```bash
signtool sign /f certificate.pfx /p password desktop/backend/mdmai-mcp-server.exe
```

**macOS:**
```bash
codesign --force --verify --verbose --sign "Developer ID" desktop/backend/mdmai-mcp-server
```

## Migration Checklist

- [ ] Build PyOxidizer executables for all target platforms
- [ ] Copy executables to `desktop/backend/`
- [ ] Update `tauri.conf.json` configuration
- [ ] Modify Rust sidecar code to use new executables
- [ ] Update build scripts and package.json
- [ ] Test development and production modes
- [ ] Verify stdio communication works correctly
- [ ] Test end-to-end desktop application functionality
- [ ] Update documentation and README
- [ ] Consider code signing for distribution

## Troubleshooting

### Common Issues

1. **Executable not found**
   - Verify executable is in `desktop/backend/`
   - Check file permissions (755 on Unix)
   - Verify platform-specific naming

2. **Startup failures**
   - Check stderr output from sidecar
   - Verify all dependencies are included in PyOxidizer build
   - Test executable independently first

3. **Communication issues**
   - Verify MCP protocol compatibility
   - Check stdin/stdout buffering settings
   - Test with simple MCP messages

4. **Performance issues**
   - Monitor memory usage
   - Check for resource conflicts
   - Verify ChromaDB initialization

### Debugging Tips

```bash
# Test executable manually
./desktop/backend/mdmai-mcp-server

# Test with verbose logging
RUST_LOG=debug npm run tauri dev

# Check sidecar output
tail -f ~/.local/share/mdmai-desktop/logs/sidecar.log
```

## Benefits of PyOxidizer Integration

1. **No Python Dependency**: Users don't need Python installed
2. **Simplified Distribution**: Single executable per platform
3. **Better Security**: Embedded Python reduces attack surface
4. **Consistent Environment**: Same Python version across deployments
5. **Professional Packaging**: More suitable for production applications

## Conclusion

Integrating PyOxidizer with Tauri provides a professional, standalone desktop application that doesn't require Python installation on user systems. While the executables are larger and have slower startup times, the benefits of simplified distribution and consistent environments make this approach ideal for production deployments.