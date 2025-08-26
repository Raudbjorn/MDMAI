# MCP Bridge Service

A production-ready bridge service that enables web clients to communicate with stdio-based MCP (Model Context Protocol) servers through WebSocket, Server-Sent Events (SSE), and HTTP REST APIs.

## Features

### Core Capabilities
- **Multi-Transport Support**: WebSocket, SSE, and HTTP/REST endpoints
- **Process Management**: Automatic subprocess lifecycle management with health monitoring
- **Session Management**: Multi-session support with client isolation
- **Protocol Translation**: Seamless conversion between different protocol formats
- **Automatic Restart**: Failed processes are automatically restarted (configurable)
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Authentication**: Optional API key authentication
- **Metrics & Monitoring**: Real-time statistics and process monitoring

### Transport Options

#### WebSocket (`/ws`)
- Real-time bidirectional communication
- Session persistence
- Automatic reconnection support
- Batch request handling

#### Server-Sent Events (`/events/{session_id}`)
- Server-to-client streaming
- Automatic heartbeat
- Event-based notifications
- Lightweight alternative to WebSocket

#### HTTP REST API
- Stateless request/response
- Tool discovery and invocation
- Session management
- Statistics and health checks

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Web Client  │────▶│ Bridge Server│────▶│ MCP Process │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴───────┐
                    │              │
            ┌───────▼────┐  ┌──────▼──────┐
            │  Session   │  │  Protocol   │
            │  Manager   │  │ Translator  │
            └────────────┘  └─────────────┘
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Configuration

The bridge can be configured via environment variables or a `.env` file:

```bash
# Server Configuration
BRIDGE_HOST=0.0.0.0
BRIDGE_PORT=8080
BRIDGE_WORKERS=1

# MCP Server
MCP_SERVER_PATH=src.main
MCP_SERVER_ARGS=
MCP_SERVER_ENV=

# Process Management
BRIDGE_MAX_PROCESSES=10
BRIDGE_PROCESS_TIMEOUT=300
BRIDGE_PROCESS_IDLE_TIMEOUT=600
BRIDGE_HEALTH_CHECK_INTERVAL=30
BRIDGE_RESTART_ON_FAILURE=true
BRIDGE_MAX_RESTART_ATTEMPTS=3

# Session Management
BRIDGE_MAX_SESSIONS_PER_CLIENT=3
BRIDGE_SESSION_TIMEOUT=3600
BRIDGE_SESSION_CLEANUP_INTERVAL=60

# Transport
BRIDGE_ENABLE_WEBSOCKET=true
BRIDGE_ENABLE_SSE=true
BRIDGE_ENABLE_HTTP=true

# Security
BRIDGE_REQUIRE_AUTH=false
BRIDGE_API_KEYS=key1,key2,key3
BRIDGE_CORS_ORIGINS=*

# Rate Limiting
BRIDGE_ENABLE_RATE_LIMITING=true
BRIDGE_RATE_LIMIT_REQUESTS=100
BRIDGE_RATE_LIMIT_PERIOD=60

# Monitoring
BRIDGE_ENABLE_METRICS=true
BRIDGE_METRICS_PORT=9090

# Logging
BRIDGE_LOG_LEVEL=INFO
BRIDGE_LOG_FILE=/var/log/mcp-bridge.log
```

## Usage

### Starting the Bridge Server

```bash
# Start with default configuration
python -m src.bridge.main

# Start with custom config
BRIDGE_PORT=9000 python -m src.bridge.main

# Start with authentication enabled
BRIDGE_REQUIRE_AUTH=true BRIDGE_API_KEYS=secret-key python -m src.bridge.main
```

### Using the Test Client

Open the test client in your browser:

```bash
# Server should be running
open http://localhost:8080/static/test_client.html
```

### API Examples

#### Create Session
```bash
curl -X POST http://localhost:8080/sessions \
  -H "Content-Type: application/json" \
  -d '{"client_id": "my-client"}'
```

#### Discover Tools
```bash
curl -X POST http://localhost:8080/tools/discover \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID"}'
```

#### Call Tool
```bash
curl -X POST http://localhost:8080/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "SESSION_ID",
    "tool": "search",
    "params": {"query": "test"}
  }'
```

#### WebSocket Connection (JavaScript)
```javascript
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
    console.log('Received:', data);
    
    if (data.type === 'session_created') {
        // Send tool request
        ws.send(JSON.stringify({
            jsonrpc: '2.0',
            id: '1',
            method: 'tools/search',
            params: { query: 'test' }
        }));
    }
};
```

#### SSE Connection (JavaScript)
```javascript
const eventSource = new EventSource('http://localhost:8080/events/my-session');

eventSource.addEventListener('connected', (event) => {
    const data = JSON.parse(event.data);
    console.log('Connected:', data.session_id);
});

eventSource.addEventListener('heartbeat', (event) => {
    console.log('Heartbeat:', event.data);
});
```

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Bridge statistics |
| `/sessions` | POST | Create session |
| `/sessions/{id}` | GET | Get session info |
| `/sessions/{id}` | DELETE | Delete session |
| `/tools/discover` | POST | Discover available tools |
| `/tools/call` | POST | Call a tool |
| `/ws` | WebSocket | WebSocket endpoint |
| `/events/{id}` | GET | SSE endpoint |

### WebSocket Messages

#### Client to Server
```json
// Create session
{
    "type": "create_session",
    "client_id": "client-123",
    "metadata": {}
}

// Attach to existing session
{
    "type": "attach_session",
    "session_id": "session-123"
}

// MCP request
{
    "jsonrpc": "2.0",
    "id": "req-1",
    "method": "tools/search",
    "params": {"query": "test"}
}
```

#### Server to Client
```json
// Session created
{
    "type": "session_created",
    "session_id": "session-123",
    "capabilities": {}
}

// MCP response
{
    "type": "response",
    "data": {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": {}
    }
}

// Error
{
    "type": "error",
    "error": "Error message"
}
```

## Development

### Running Tests

```bash
# Run all bridge tests
pytest tests/test_bridge.py -v

# Run with coverage
pytest tests/test_bridge.py --cov=src.bridge --cov-report=html

# Run specific test
pytest tests/test_bridge.py::TestMCPProcessManagement -v
```

### Adding New Features

1. **New Transport**: Implement in `bridge_server.py`
2. **New Protocol Format**: Add to `protocol_translator.py`
3. **Process Features**: Modify `mcp_process_manager.py`
4. **Session Features**: Update `session_manager.py`

### Debugging

Enable debug logging:
```bash
BRIDGE_LOG_LEVEL=DEBUG python -m src.bridge.main
```

Monitor process health:
```bash
curl http://localhost:8080/stats | jq .process_stats
```

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "-m", "src.bridge.main"]
```

### Using systemd

```ini
[Unit]
Description=MCP Bridge Service
After=network.target

[Service]
Type=simple
User=mcp
WorkingDirectory=/opt/mcp-bridge
ExecStart=/usr/bin/python3 -m src.bridge.main
Restart=always
RestartSec=10
Environment="BRIDGE_PORT=8080"
Environment="BRIDGE_REQUIRE_AUTH=true"

[Install]
WantedBy=multi-user.target
```

### Nginx Proxy

```nginx
upstream mcp_bridge {
    server localhost:8080;
}

server {
    listen 443 ssl http2;
    server_name mcp-bridge.example.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    
    # WebSocket support
    location /ws {
        proxy_pass http://mcp_bridge;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # SSE support
    location /events/ {
        proxy_pass http://mcp_bridge;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection "";
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
    
    # REST API
    location / {
        proxy_pass http://mcp_bridge;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Prometheus Metrics (when enabled)

```yaml
scrape_configs:
  - job_name: 'mcp-bridge'
    static_configs:
      - targets: ['localhost:9090']
```

### Health Checks

```bash
# Simple health check
curl http://localhost:8080/health

# Detailed statistics
curl http://localhost:8080/stats
```

## Security Considerations

1. **Authentication**: Enable `BRIDGE_REQUIRE_AUTH` in production
2. **Rate Limiting**: Configure appropriate limits for your use case
3. **CORS**: Restrict `BRIDGE_CORS_ORIGINS` to specific domains
4. **Process Isolation**: Each session runs in a separate subprocess
5. **Input Validation**: All inputs are validated before processing
6. **Timeout Protection**: Configurable timeouts prevent resource exhaustion

## Troubleshooting

### Common Issues

#### Process won't start
- Check MCP_SERVER_PATH is correct
- Verify Python path and dependencies
- Check logs for detailed error messages

#### WebSocket connection fails
- Ensure BRIDGE_ENABLE_WEBSOCKET=true
- Check firewall/proxy settings
- Verify CORS configuration

#### Session timeout
- Increase BRIDGE_SESSION_TIMEOUT
- Check BRIDGE_PROCESS_IDLE_TIMEOUT
- Monitor with /stats endpoint

#### High memory usage
- Reduce BRIDGE_MAX_PROCESSES
- Lower BRIDGE_PROCESS_IDLE_TIMEOUT
- Enable process cleanup

## License

See LICENSE file in the repository root.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: [Full docs](https://docs.example.com)