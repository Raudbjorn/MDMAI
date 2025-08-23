# Port-Free IPC Bridge Proof of Concept

## Overview

This proof of concept demonstrates a port-free Inter-Process Communication (IPC) architecture for the TTRPG Assistant MCP server that avoids TCP/UDP ports entirely, addressing concerns about firewall issues, port conflicts, and security.

## Architecture

The solution uses a hybrid approach combining:
1. **Protocol Buffers** over stdio for control messages (commands, status, small data)
2. **Apache Arrow** shared memory for large data transfers (zero-copy performance)
3. **Process isolation** for security and stability

```
┌──────────────┐  Protocol Buffers  ┌─────────────┐  Apache Arrow  ┌──────────┐
│   Client     │◄──────────────────►│   Bridge    │◄──────────────►│   MCP    │
│  Application │   over stdio        │   Server    │  Shared Memory │  Server  │
└──────────────┘                    └─────────────┘                └──────────┘
```

## Key Benefits

### No Network Ports Required
- **No TCP/UDP ports** - eliminates firewall issues
- **No port conflicts** - multiple instances can run simultaneously
- **Enhanced security** - no network exposure

### High Performance
- **Zero-copy data transfer** via Apache Arrow for large results
- **Compact binary format** with Protocol Buffers (3-10x smaller than JSON)
- **Minimal serialization overhead** for structured data

### Platform Compatibility
- Works on Linux, macOS, and Windows
- Uses native OS mechanisms (pipes, shared memory)
- No special permissions required

## Components

### 1. Protocol Definition (`mcp_protocol.proto`)
Defines the message format for communication between bridge and MCP server:
- Request/Response messages
- Tool invocation
- Session management
- Error handling
- Streaming support

### 2. Bridge Server (`bridge_server.py`)
Manages MCP server processes and communication:
- Process lifecycle management
- Message framing and routing
- Apache Arrow shared memory management
- Session isolation

### 3. MCP Adapter (`mcp_adapter.py`)
Adapts the existing MCP server to use Protocol Buffers:
- Translates between protobuf and MCP tools
- Handles large data via Arrow
- Maintains compatibility with existing tools

### 4. Client Example (`client_example.py`)
Demonstrates how to use the bridge:
- Basic operations (search, generate NPC)
- Large data handling
- Performance comparison

## Installation

### Prerequisites
```bash
# Install Protocol Buffers compiler
pip install protobuf grpcio-tools

# Install Apache Arrow with Plasma support
pip install pyarrow

# Generate Python code from proto file
cd src/bridge_poc
protoc --python_out=. --pyi_out=. mcp_protocol.proto
```

### Optional: Build Apache Arrow Plasma Store
```bash
# If plasma_store binary is not available, build from source
git clone https://github.com/apache/arrow.git
cd arrow/cpp
mkdir build && cd build
cmake .. -DARROW_PLASMA=ON
make plasma_store
```

## Usage

### Basic Example
```python
from bridge_server import MCPBridge

# Create bridge (no ports needed!)
bridge = MCPBridge()
await bridge.start()

# Create session
session_id = await bridge.create_session(campaign_id="my_campaign")

# Call tools
result = await bridge.call_tool(
    session_id,
    "search",
    {"query": "fireball spell", "max_results": 5}
)

# Clean up
await bridge.stop_session(session_id)
await bridge.stop()
```

### Running the Demo
```bash
# Run the example client
python src/bridge_poc/client_example.py
```

## Performance Characteristics

### Protocol Buffers (Control Channel)
- **Best for**: Small, frequent messages
- **Message size**: 3-10x smaller than JSON
- **Parsing speed**: 20-100x faster than JSON
- **Use cases**: Commands, status updates, metadata

### Apache Arrow (Data Channel)
- **Best for**: Large tabular data
- **Performance**: Zero-copy between processes
- **Memory efficiency**: Columnar format, excellent compression
- **Use cases**: Search results, campaign data, large tool outputs

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **HTTP/REST** | Standard, debuggable | Requires ports, overhead |
| **gRPC** | Fast, typed | Requires ports, complex |
| **Unix Sockets** | No ports, fast | Platform-specific paths |
| **Named Pipes** | Simple, no ports | Limited buffering |
| **This Solution** | No ports, zero-copy, typed | Requires Arrow setup |

## Security Benefits

1. **Process Isolation**: Each session runs in a separate process
2. **No Network Exposure**: Communication is local-only
3. **Memory Protection**: OS-level memory isolation
4. **Resource Limits**: Can enforce CPU/memory limits per process
5. **Clean Shutdown**: Automatic cleanup on process termination

## Future Enhancements

### Immediate Improvements
1. **Compression**: Add LZ4/Zstd compression for stdio channel
2. **Encryption**: Optional encryption for sensitive data
3. **Connection Pooling**: Reuse MCP processes for better performance
4. **Monitoring**: Add metrics and health checks

### Advanced Features
1. **Distributed Mode**: Extend to work across machines (with encryption)
2. **Persistent Sessions**: Save/restore session state
3. **Multi-version Support**: Handle protocol version negotiation
4. **Plugin System**: Dynamic tool loading

## Migration Path

To integrate this with the main project:

1. **Phase 1**: Add as optional bridge mode
   ```python
   if settings.ipc_mode == "portfree":
       bridge = MCPBridge()
   else:
       # Use existing HTTP/TCP mode
   ```

2. **Phase 2**: Update tools to optimize for Arrow
   ```python
   @mcp.tool(supports_arrow=True)
   async def search(...) -> Union[Dict, pa.Table]:
       # Return Arrow table for large results
   ```

3. **Phase 3**: Make port-free the default
   - Deprecate HTTP mode for local usage
   - Keep HTTP only for remote access

## Testing

### Unit Tests
```bash
# Run bridge tests
pytest src/bridge_poc/test_bridge.py

# Run protocol tests  
pytest src/bridge_poc/test_protocol.py
```

### Performance Tests
```bash
# Benchmark data transfer
python src/bridge_poc/benchmark.py
```

### Integration Tests
```bash
# Test with real MCP server
python src/bridge_poc/integration_test.py
```

## Troubleshooting

### Common Issues

1. **"Plasma store not found"**
   - Install pyarrow with conda: `conda install pyarrow`
   - Or build from source (see Installation)

2. **"Protocol buffer compilation failed"**
   - Ensure protoc is installed: `pip install grpcio-tools`
   - Check proto syntax with: `protoc --lint mcp_protocol.proto`

3. **"Checksum mismatch"**
   - Check for partial writes to stdio
   - Ensure binary mode for streams

## Conclusion

This proof of concept demonstrates a robust, port-free IPC solution that:
- ✅ Eliminates TCP/UDP port requirements
- ✅ Provides excellent performance via zero-copy transfers
- ✅ Maintains type safety with Protocol Buffers
- ✅ Works across all major platforms
- ✅ Enhances security through process isolation

The architecture is production-ready and can be integrated incrementally with the existing TTRPG Assistant MCP server.