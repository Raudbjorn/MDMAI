"""
WebSocket adapter for MCP server
Provides WebSocket interface to the existing MCP server via FastAPI
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import the existing MCP server
# This will be adjusted based on actual MCP implementation
# from src.main import mcp_server

logger = logging.getLogger(__name__)

class MCPWebSocketAdapter:
    """Adapter to expose MCP server via WebSocket"""
    
    def __init__(self, mcp_instance=None):
        self.mcp = mcp_instance
        self.app = FastAPI(title="MCP Desktop Server")
        self.active_connections: list[WebSocket] = []
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        """Configure CORS for local frontend access"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:*",
                "tauri://localhost",
                "https://tauri.localhost"
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup WebSocket and HTTP routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for connection verification"""
            return JSONResponse({
                "status": "healthy",
                "mcp_connected": self.mcp is not None
            })
        
        @self.app.websocket("/mcp")
        async def websocket_endpoint(websocket: WebSocket):
            """Main WebSocket endpoint for MCP communication"""
            await self.connect(websocket)
            try:
                while True:
                    # Receive JSON-RPC request
                    data = await websocket.receive_text()
                    
                    try:
                        request = json.loads(data)
                        response = await self.handle_mcp_request(request)
                        await websocket.send_text(json.dumps(response))
                    except json.JSONDecodeError as e:
                        error_response = {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32700,
                                "message": "Parse error",
                                "data": str(e)
                            },
                            "id": None
                        }
                        await websocket.send_text(json.dumps(error_response))
                    except Exception as e:
                        logger.error(f"Error handling request: {e}")
                        error_response = {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32603,
                                "message": "Internal error",
                                "data": str(e)
                            },
                            "id": request.get("id") if "request" in locals() else None
                        }
                        await websocket.send_text(json.dumps(error_response))
            
            except WebSocketDisconnect:
                self.disconnect(websocket)
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process JSON-RPC request and route to MCP server
        
        Args:
            request: JSON-RPC 2.0 request object
            
        Returns:
            JSON-RPC 2.0 response object
        """
        # Validate JSON-RPC structure
        if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32600,
                    "message": "Invalid Request",
                    "data": "Missing or invalid jsonrpc version"
                },
                "id": request.get("id")
            }
        
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        # Route to appropriate MCP handler
        try:
            # This would call the actual MCP server methods
            # result = await self.mcp.handle_request(method, params)
            
            # Placeholder response for testing
            result = {
                "status": "success",
                "method": method,
                "params_received": params
            }
            
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                },
                "id": request_id
            }
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


def start_websocket_server(host: str = "127.0.0.1", port: int = 8765):
    """Start the WebSocket server"""
    # Initialize MCP server
    # mcp = initialize_mcp_server()
    
    # Create adapter
    adapter = MCPWebSocketAdapter()  # Pass mcp when available
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    uvicorn.run(
        adapter.app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    # For testing the WebSocket server standalone
    start_websocket_server()