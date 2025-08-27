# MCP Bridge Service Documentation

## Overview

The MCP Bridge Service provides a bridge between web clients and stdio-based MCP (Model Context Protocol) servers. It enables HTTP, WebSocket, and Server-Sent Events (SSE) communication while maintaining full MCP protocol compliance.

## Architecture

### Components

1. **MCP Process Manager** (`mcp_process_manager.py`)
   - Manages lifecycle of stdio MCP server subprocesses
   - Handles process pooling and health monitoring
   - Implements automatic restart and cleanup

2. **Session Manager** (`session_manager.py`)
   - Manages client sessions and their associated MCP processes
   - Enforces session limits and timeouts
   - Routes requests to appropriate processes

3. **Protocol Translator** (`protocol_translator.py`)
   - Translates between client formats and MCP JSON-RPC 2.0
   - Supports multiple AI provider formats (OpenAI, Anthropic)
   - Validates messages against MCP specification

4. **Bridge Server** (`bridge_server.py`)
   - FastAPI application providing HTTP/WebSocket/SSE endpoints
   - Handles client connections and request routing
   - Implements streaming responses for real-time updates

## Features

### Transport Support

- **HTTP**: Synchronous request/response for tool calls
- **WebSocket**: Bidirectional real-time communication
- **SSE**: Server-to-client streaming for notifications

### Session Management

- Process isolation per session
- Automatic session cleanup on idle timeout
- Configurable session limits per client
- Session state tracking and recovery

### MCP Protocol Compliance

- Full JSON-RPC 2.0 support
- Tool discovery and registration
- Request/response correlation
- Error handling with standard codes
- Batch request support

### Security Features

- Optional API key authentication
- Process sandboxing
- Request size limits
- Rate limiting support
- Session isolation

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Configuration

The bridge service can be configured via environment variables:

```bash
# MCP Server Configuration
MCP_SERVER_PATH=src.main  # Path to MCP server module

# Process Management
BRIDGE_MAX_PROCESSES=10  # Maximum concurrent MCP processes
BRIDGE_PROCESS_TIMEOUT=300  # Process timeout in seconds
BRIDGE_PROCESS_IDLE_TIMEOUT=600  # Idle timeout before process cleanup

# Session Management
BRIDGE_MAX_SESSIONS_PER_CLIENT=3  # Max sessions per client
BRIDGE_SESSION_TIMEOUT=3600  # Session timeout in seconds

# Transport Configuration
BRIDGE_ENABLE_WEBSOCKET=true
BRIDGE_ENABLE_SSE=true
BRIDGE_ENABLE_HTTP=true

# Security
BRIDGE_REQUIRE_AUTH=false
BRIDGE_API_KEYS=key1,key2  # Comma-separated API keys

# Server Configuration
BRIDGE_HOST=0.0.0.0
BRIDGE_PORT=8080

# Logging
BRIDGE_LOG_LEVEL=INFO
BRIDGE_LOG_REQUESTS=false
BRIDGE_LOG_RESPONSES=false
```

## Usage

### Starting the Bridge Server

```bash
# Run directly
python -m src.bridge.main

# Or with custom configuration
BRIDGE_PORT=3000 BRIDGE_LOG_LEVEL=DEBUG python -m src.bridge.main

# Run with uvicorn (for development)
uvicorn src.bridge.bridge_server:create_bridge_app --reload
```

### Client Examples

#### HTTP Client

```python
import requests

# Create session
response = requests.post('http://localhost:8080/sessions', json={
    'client_id': 'my-client'
})
session = response.json()
session_id = session['session_id']

# Discover tools
response = requests.post('http://localhost:8080/tools/discover', json={
    'session_id': session_id
})
tools = response.json()['tools']

# Call a tool
response = requests.post('http://localhost:8080/tools/call', json={
    'session_id': session_id,
    'tool': 'search',
    'params': {'query': 'test search'}
})
result = response.json()['result']
```

#### WebSocket Client

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
    // Create session
    ws.send(JSON.stringify({
        type: 'create_session',
        client_id: 'web-client'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'session_created') {
        // Session ready, can now send requests
        ws.send(JSON.stringify({
            jsonrpc: '2.0',
            id: '1',
            method: 'tools/search',
            params: {query: 'test'}
        }));
    }
};
```

#### SSE Client

```javascript
// Connect to SSE endpoint
const eventSource = new EventSource('http://localhost:8080/events/my-session');

eventSource.addEventListener('connected', (event) => {
    const data = JSON.parse(event.data);
    console.log('Session connected:', data.session_id);
});

eventSource.addEventListener('heartbeat', (event) => {
    console.log('Heartbeat received');
});

// Make tool calls via HTTP while receiving events via SSE
fetch('http://localhost:8080/tools/call', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        session_id: 'my-session',
        tool: 'search',
        params: {query: 'test'}
    })
});
```

## API Reference

### HTTP Endpoints

#### `POST /sessions`
Create a new session.

**Request:**
```json
{
    "client_id": "optional-client-id",
    "metadata": {}
}
```

**Response:**
```json
{
    "session_id": "uuid",
    "client_id": "client-id",
    "state": "ready",
    "capabilities": {}
}
```

#### `GET /sessions/{session_id}`
Get session information.

#### `DELETE /sessions/{session_id}`
Delete a session and cleanup resources.

#### `POST /tools/discover`
Discover available MCP tools.

**Request:**
```json
{
    "session_id": "optional-session-id"
}
```

**Response:**
```json
{
    "session_id": "uuid",
    "tools": [
        {
            "name": "search",
            "description": "Search the knowledge base",
            "inputSchema": {}
        }
    ]
}
```

#### `POST /tools/call`
Call an MCP tool.

**Request:**
```json
{
    "session_id": "optional-session-id",
    "tool": "search",
    "params": {"query": "test"}
}
```

**Response:**
```json
{
    "session_id": "uuid",
    "tool": "search",
    "result": {}
}
```

#### `GET /health`
Health check endpoint.

#### `GET /stats`
Get bridge statistics.

### WebSocket Protocol

#### Session Creation
```json
{
    "type": "create_session",
    "client_id": "optional-client-id"
}
```

#### Session Attachment
```json
{
    "type": "attach_session",
    "session_id": "existing-session-id"
}
```

#### MCP Request
```json
{
    "jsonrpc": "2.0",
    "id": "request-id",
    "method": "tools/search",
    "params": {"query": "test"}
}
```

### SSE Events

- `connected`: Initial connection with session info
- `heartbeat`: Periodic keepalive
- `notification`: MCP notifications from server
- `error`: Error events

## Testing

Run the test suite:

```bash
pytest tests/test_bridge.py -v
```

Run with coverage:

```bash
pytest tests/test_bridge.py --cov=src.bridge --cov-report=html
```

## Performance Considerations

### Process Pooling

The bridge maintains a pool of MCP processes to handle multiple sessions efficiently:

- Processes are reused when possible
- Idle processes are cleaned up automatically
- Health checks ensure process reliability

### Request Batching

Multiple requests can be batched for efficiency:

```json
[
    {"jsonrpc": "2.0", "id": "1", "method": "tools/search", "params": {"query": "test1"}},
    {"jsonrpc": "2.0", "id": "2", "method": "tools/search", "params": {"query": "test2"}}
]
```

### Caching

Consider implementing caching at the bridge level for frequently accessed data:

- Tool discovery results
- Resource metadata
- Session capabilities

## Troubleshooting

### Common Issues

1. **Process won't start**
   - Check MCP_SERVER_PATH is correct
   - Verify Python path and dependencies
   - Check process permissions

2. **Session timeout**
   - Increase BRIDGE_SESSION_TIMEOUT
   - Send periodic heartbeats from client
   - Check network connectivity

3. **High memory usage**
   - Reduce BRIDGE_MAX_PROCESSES
   - Decrease BRIDGE_PROCESS_IDLE_TIMEOUT
   - Monitor process statistics via /stats

### Debug Mode

Enable debug logging:

```bash
BRIDGE_LOG_LEVEL=DEBUG BRIDGE_LOG_REQUESTS=true BRIDGE_LOG_RESPONSES=true python -m src.bridge.main
```

## Security Best Practices

1. **Enable authentication in production**
   ```bash
   BRIDGE_REQUIRE_AUTH=true
   BRIDGE_API_KEYS=secure-key-1,secure-key-2
   ```

2. **Use HTTPS/WSS in production**
   - Deploy behind a reverse proxy (nginx, traefik)
   - Configure SSL certificates

3. **Implement rate limiting**
   - Use a reverse proxy or API gateway
   - Monitor usage patterns

4. **Restrict CORS origins**
   - Configure allowed origins in production
   - Avoid wildcard origins

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MCP_STDIO_MODE=true
ENV BRIDGE_HOST=0.0.0.0
ENV BRIDGE_PORT=8080

EXPOSE 8080

CMD ["python", "-m", "src.bridge.main"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  mcp-bridge:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MCP_SERVER_PATH=src.main
      - BRIDGE_MAX_PROCESSES=10
      - BRIDGE_LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### Systemd Service

```ini
[Unit]
Description=MCP Bridge Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/mcp-bridge
Environment="MCP_STDIO_MODE=true"
Environment="BRIDGE_PORT=8080"
ExecStart=/usr/bin/python3 -m src.bridge.main
Restart=always

[Install]
WantedBy=multi-user.target
```

## Monitoring

### Prometheus Metrics

The bridge can expose metrics for monitoring:

- Active sessions count
- Active processes count
- Request rate and latency
- Error rate
- Process CPU and memory usage

### Health Checks

Configure health check endpoints for monitoring:

```bash
# Liveness probe
curl http://localhost:8080/health

# Readiness probe
curl http://localhost:8080/stats
```

## Contributing

Please follow these guidelines when contributing:

1. Write tests for new features
2. Update documentation
3. Follow MCP protocol specifications
4. Ensure backward compatibility
5. Add type hints and docstrings

## License

See LICENSE file in the project root.