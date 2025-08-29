# TTRPG MCP Server - Interactive Demos

This directory contains interactive demonstrations of the TTRPG MCP Server capabilities.

## Available Demos

### 1. Interactive Web Demo (`interactive-demo.html`)
A standalone HTML file that demonstrates core functionality without requiring server setup.

**Features:**
- 🎲 **Dice Roller** - Roll dice using standard RPG notation
- 📚 **Rule Search** - Search game rules and mechanics
- ⚔️ **Character Generator** - Generate random characters
- 📝 **Session Notes** - Add and categorize session notes
- ⚡ **Initiative Tracker** - Manage combat turn order
- 🗺️ **Campaign Info** - View campaign details, NPCs, and locations

**How to Use:**
1. Open `interactive-demo.html` in any modern web browser
2. No server or installation required - runs entirely in the browser
3. Simulates server responses for demonstration purposes

### 2. Live Server Demo (`live-demo.html`)
A full-featured demo that connects to a running MCP server.

**Prerequisites:**
- MCP server running locally (see deployment guide)
- Bridge server running on port 8000

**Features:**
- Real-time WebSocket connections
- Actual MCP tool execution
- Live collaborative features
- Server-Sent Events for updates

### 3. API Examples (`api-examples/`)

Collection of example code for integrating with the MCP server:

#### Python Client (`python-client.py`)
```python
# Example usage
from mcp_client import MCPClient

client = MCPClient("ws://localhost:8000/ws")
await client.connect()

# Roll dice
result = await client.call_tool("roll_dice", {"expression": "3d6+2"})

# Search rules
results = await client.call_tool("search_rules", {
    "query": "advantage",
    "limit": 5
})
```

#### JavaScript Client (`javascript-client.js`)
```javascript
// Example usage
const client = new MCPClient('ws://localhost:8000/ws');
await client.connect();

// Roll dice
const result = await client.callTool('roll_dice', {
    expression: '3d6+2'
});

// Search rules
const results = await client.callTool('search_rules', {
    query: 'advantage',
    limit: 5
});
```

#### cURL Examples (`curl-examples.sh`)
```bash
# Get session info
curl http://localhost:8000/api/sessions/current

# List available tools
curl http://localhost:8000/api/tools

# Execute a tool
curl -X POST http://localhost:8000/api/tools/execute \
  -H "Content-Type: application/json" \
  -d '{"tool": "roll_dice", "arguments": {"expression": "3d6+2"}}'
```

## Demo Scenarios

### Scenario 1: Running a Combat Encounter
1. Start a new session
2. Add combatants to initiative
3. Track HP and conditions
4. Roll dice for attacks and damage
5. Add combat notes

### Scenario 2: Character Creation Session
1. Generate multiple character options
2. Customize stats and equipment
3. Create backstories
4. Save to campaign

### Scenario 3: Rule Lookup During Play
1. Search for specific mechanics
2. Get quick rule clarifications
3. Find related rules
4. Reference page numbers

### Scenario 4: Campaign Management
1. Create a new campaign
2. Add NPCs and locations
3. Track plot points
4. Manage party inventory

## Testing the Demos

### Unit Testing
```bash
# Test demo functionality
pytest tests/demos/test_interactive_demo.py
```

### Performance Testing
```bash
# Load test the demo endpoints
python tests/demos/load_test_demo.py
```

## Customizing Demos

### Adding New Features
1. Edit the HTML/JavaScript in `interactive-demo.html`
2. Add new demo cards following the existing pattern
3. Implement mock responses for offline demo
4. Add real server calls for live demo

### Styling
- Uses inline CSS for portability
- Responsive design for mobile devices
- Gradient backgrounds and smooth animations
- Accessible color contrast ratios

## Deployment

### Hosting the Static Demo
```bash
# Using Python's built-in server
python -m http.server 8080

# Using Node.js
npx http-server -p 8080

# Using nginx
cp interactive-demo.html /var/www/html/
```

### Embedding in Documentation
```html
<iframe src="interactive-demo.html" width="100%" height="800px"></iframe>
```

### Docker Container
```dockerfile
FROM nginx:alpine
COPY interactive-demo.html /usr/share/nginx/html/index.html
EXPOSE 80
```

## Browser Compatibility

### Supported Browsers
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

### Required Features
- ES6 JavaScript support
- CSS Grid and Flexbox
- WebSocket API (for live demo)
- LocalStorage (for saving preferences)

## Troubleshooting

### Demo Not Loading
- Check browser console for errors
- Ensure JavaScript is enabled
- Try a different browser
- Clear browser cache

### Connection Issues (Live Demo)
- Verify MCP server is running
- Check bridge server is on port 8000
- Ensure no firewall blocking
- Check WebSocket support

### Performance Issues
- Reduce number of concurrent operations
- Clear result displays periodically
- Use Chrome DevTools Performance tab
- Check network latency

## Contributing

### Adding New Demos
1. Create a new HTML file in `docs/demos/`
2. Follow the existing structure and styling
3. Add documentation to this README
4. Submit a pull request

### Reporting Issues
- Use GitHub Issues
- Include browser version and OS
- Provide console error messages
- Include steps to reproduce

## License

These demos are part of the TTRPG MCP Server project and are released under the same license.