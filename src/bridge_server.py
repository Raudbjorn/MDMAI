#!/usr/bin/env python3
"""
Bridge Server - WebSocket to MCP Communication
Provides a WebSocket interface for web clients to communicate with the MCP server
"""

import asyncio
import json
import logging
import subprocess
import sys
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TTRPG MCP Bridge")

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MCPBridge:
    """Manages communication with MCP server process"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.reader_task: Optional[asyncio.Task] = None
        self.responses: Dict[str, asyncio.Future] = {}
        self.message_id = 0
        
    async def start(self):
        """Start the MCP server process"""
        logger.info("Starting MCP server process...")
        
        # Start MCP server as subprocess
        self.process = subprocess.Popen(
            [sys.executable, "src/mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False  # Use bytes for better control
        )
        
        # Start reader task
        self.reader_task = asyncio.create_task(self._read_output())
        
        # Send initialization
        await self.send_request({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0.0",
                "capabilities": {
                    "roots": {}
                },
                "clientInfo": {
                    "name": "TTRPG Bridge",
                    "version": "1.0.0"
                }
            },
            "id": self.get_next_id()
        })
        
        logger.info("MCP server started successfully")
        
    def get_next_id(self) -> str:
        """Get next message ID"""
        self.message_id += 1
        return str(self.message_id)
        
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to MCP server and wait for response"""
        if not self.process or self.process.poll() is not None:
            raise Exception("MCP server not running")
        
        # Add ID if not present
        if "id" not in request:
            request["id"] = self.get_next_id()
        
        request_id = request["id"]
        
        # Create future for response
        future = asyncio.Future()
        self.responses[request_id] = future
        
        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        self.process.stdin.flush()
        
        logger.info(f"Sent request: {request}")
        
        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            del self.responses[request_id]
            raise Exception("Request timeout")
            
    async def _read_output(self):
        """Read output from MCP server"""
        while self.process and self.process.poll() is None:
            try:
                # Read line from stdout
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self.process.stdout.readline
                )
                
                if not line:
                    break
                    
                # Try to parse as JSON
                try:
                    message = json.loads(line.decode())
                    logger.info(f"Received message: {message}")
                    
                    # Handle response
                    if "id" in message and message["id"] in self.responses:
                        self.responses[message["id"]].set_result(message)
                        del self.responses[message["id"]]
                        
                except json.JSONDecodeError:
                    # Not JSON, might be log output
                    logger.debug(f"Non-JSON output: {line.decode().strip()}")
                    
            except Exception as e:
                logger.error(f"Error reading output: {e}")
                break
                
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool"""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self.get_next_id()
        }
        
        response = await self.send_request(request)
        
        if "error" in response:
            raise Exception(f"Tool error: {response['error']}")
            
        return response.get("result", {}).get("content", [])
        
    async def list_tools(self) -> list:
        """List available MCP tools"""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self.get_next_id()
        }
        
        response = await self.send_request(request)
        return response.get("result", {}).get("tools", [])
        
    async def stop(self):
        """Stop the MCP server process"""
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.5)
            if self.process.poll() is None:
                self.process.kill()
            self.process = None
            
        if self.reader_task:
            self.reader_task.cancel()
            
# Global bridge instance
bridge = MCPBridge()

@app.on_event("startup")
async def startup_event():
    """Start MCP bridge on server startup"""
    await bridge.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop MCP bridge on server shutdown"""
    await bridge.stop()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for clients"""
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            logger.info(f"Received from client: {data}")
            
            # Handle different message types
            if data.get("type") == "list_tools":
                # List available tools
                tools = await bridge.list_tools()
                await websocket.send_json({
                    "type": "tools_list",
                    "tools": tools
                })
                
            elif data.get("type") == "call_tool":
                # Call MCP tool
                try:
                    result = await bridge.call_tool(
                        data["tool"],
                        data.get("arguments", {})
                    )
                    
                    # Send result back
                    await websocket.send_json({
                        "type": "tool_result",
                        "tool": data["tool"],
                        "result": result
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })
                    
            else:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Unknown message type: {data.get('type')}"
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/")
async def root():
    """Serve a simple test page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TTRPG MCP Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; margin: 5px; cursor: pointer; }
            input { padding: 8px; margin: 5px; width: 200px; }
            .result { background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 3px; }
            .error { background: #fee; color: #c00; }
            #log { background: #f9f9f9; padding: 10px; height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TTRPG MCP Demo</h1>
            
            <div class="section">
                <h2>Connection Status</h2>
                <div id="status">Disconnected</div>
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
            </div>
            
            <div class="section">
                <h2>Tools</h2>
                <button onclick="listTools()">List Available Tools</button>
                <div id="tools"></div>
            </div>
            
            <div class="section">
                <h2>Dice Roller</h2>
                <input type="text" id="diceExpression" placeholder="e.g., 3d6+2" value="3d6+2">
                <button onclick="rollDice()">Roll Dice</button>
                <div id="diceResult"></div>
            </div>
            
            <div class="section">
                <h2>Rule Search</h2>
                <input type="text" id="searchQuery" placeholder="Search rules..." value="advantage">
                <button onclick="searchRules()">Search</button>
                <div id="searchResult"></div>
            </div>
            
            <div class="section">
                <h2>Debug Log</h2>
                <div id="log"></div>
            </div>
        </div>
        
        <script>
            let ws = null;
            
            function log(message) {
                const logDiv = document.getElementById('log');
                const time = new Date().toLocaleTimeString();
                logDiv.innerHTML += `[${time}] ${message}<br>`;
                logDiv.scrollTop = logDiv.scrollHeight;
            }
            
            function connect() {
                if (ws) {
                    log('Already connected');
                    return;
                }
                
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = () => {
                    document.getElementById('status').textContent = 'Connected';
                    log('WebSocket connected');
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    log('Received: ' + JSON.stringify(data));
                    
                    if (data.type === 'tools_list') {
                        displayTools(data.tools);
                    } else if (data.type === 'tool_result') {
                        displayToolResult(data);
                    } else if (data.type === 'error') {
                        displayError(data.error);
                    }
                };
                
                ws.onerror = (error) => {
                    log('WebSocket error: ' + error);
                };
                
                ws.onclose = () => {
                    document.getElementById('status').textContent = 'Disconnected';
                    log('WebSocket disconnected');
                    ws = null;
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                }
            }
            
            function listTools() {
                if (!ws) {
                    alert('Please connect first');
                    return;
                }
                
                ws.send(JSON.stringify({ type: 'list_tools' }));
            }
            
            function rollDice() {
                if (!ws) {
                    alert('Please connect first');
                    return;
                }
                
                const expression = document.getElementById('diceExpression').value;
                
                ws.send(JSON.stringify({
                    type: 'call_tool',
                    tool: 'roll_dice',
                    arguments: { expression }
                }));
            }
            
            function searchRules() {
                if (!ws) {
                    alert('Please connect first');
                    return;
                }
                
                const query = document.getElementById('searchQuery').value;
                
                ws.send(JSON.stringify({
                    type: 'call_tool',
                    tool: 'search_rules',
                    arguments: { query, limit: 3 }
                }));
            }
            
            function displayTools(tools) {
                const toolsDiv = document.getElementById('tools');
                toolsDiv.innerHTML = '<h3>Available Tools:</h3>';
                tools.forEach(tool => {
                    toolsDiv.innerHTML += `
                        <div class="result">
                            <strong>${tool.name}</strong><br>
                            ${tool.description || 'No description'}
                        </div>
                    `;
                });
            }
            
            function displayToolResult(data) {
                if (data.tool === 'roll_dice') {
                    const result = data.result[0];
                    document.getElementById('diceResult').innerHTML = `
                        <div class="result">
                            <strong>Roll: ${result.expression}</strong><br>
                            Result: ${result.total}<br>
                            Breakdown: ${result.breakdown}<br>
                            Individual rolls: ${result.rolls.join(', ')}
                        </div>
                    `;
                } else if (data.tool === 'search_rules') {
                    const result = data.result[0];
                    const searchDiv = document.getElementById('searchResult');
                    searchDiv.innerHTML = '<h3>Search Results:</h3>';
                    
                    if (result.results && result.results.length > 0) {
                        result.results.forEach(rule => {
                            searchDiv.innerHTML += `
                                <div class="result">
                                    <strong>${rule.title}</strong><br>
                                    ${rule.content}<br>
                                    <small>${rule.source}, p.${rule.page}</small>
                                </div>
                            `;
                        });
                    } else {
                        searchDiv.innerHTML += '<div class="result">No results found</div>';
                    }
                }
            }
            
            function displayError(error) {
                log('Error: ' + error);
                alert('Error: ' + error);
            }
            
            // Auto-connect on load
            window.onload = () => {
                connect();
            };
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    print("Starting TTRPG MCP Bridge Server...")
    print("Access the demo at: http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)