"""
MCP Server Adapter for Port-free IPC
Adapts the existing MCP server to work with Protocol Buffers and Arrow
"""

import asyncio
import json
import logging
import os
import struct
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import pyarrow as pa
import pyarrow.plasma as plasma
from google.protobuf import json_format, timestamp_pb2

try:
    from . import mcp_protocol_pb2 as pb
except ImportError:
    import mcp_protocol_pb2 as pb

from ..main import mcp, db, search_service  # Import existing MCP server

logger = logging.getLogger(__name__)


class MCPProtobufAdapter:
    """Adapts MCP server to use Protocol Buffers over stdio"""
    
    def __init__(self):
        self.session_id = os.environ.get("MCP_SESSION_ID", "unknown")
        self.plasma_socket = os.environ.get("MCP_PLASMA_SOCKET", "/tmp/plasma")
        self.plasma_client = None
        self.running = False
        
    async def start(self):
        """Start the adapter"""
        logger.info(f"Starting MCP Protobuf Adapter for session {self.session_id}")
        
        # Connect to Plasma store if available
        try:
            self.plasma_client = plasma.connect(self.plasma_socket)
            logger.info(f"Connected to Plasma store at {self.plasma_socket}")
        except Exception as e:
            logger.warning(f"Could not connect to Plasma store: {e}")
            logger.info("Will use JSON for all data transfer")
        
        self.running = True
        
        # Start message loop
        await self.message_loop()
    
    async def message_loop(self):
        """Main message processing loop"""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        
        # Connect stdin to async reader
        transport, _ = await asyncio.get_event_loop().connect_read_pipe(
            lambda: protocol, sys.stdin.buffer
        )
        
        try:
            while self.running:
                # Read framed message
                message_bytes = await self.read_framed_message(reader)
                if not message_bytes:
                    break  # EOF or error
                
                # Parse request
                request = pb.MCPRequest()
                request.ParseFromString(message_bytes)
                
                # Process request
                response = await self.process_request(request)
                
                # Send response
                await self.send_response(response)
                
        except Exception as e:
            logger.error(f"Error in message loop: {e}")
        finally:
            if transport:
                transport.close()
    
    async def read_framed_message(self, reader: asyncio.StreamReader) -> Optional[bytes]:
        """Read a framed protobuf message"""
        # Read magic bytes
        magic = await reader.read(4)
        if not magic or magic != b'MCP\x01':
            return None
        
        # Read length
        length_bytes = await reader.read(4)
        if len(length_bytes) != 4:
            return None
        length = struct.unpack('<I', length_bytes)[0]
        
        # Read message
        message = await reader.read(length)
        if len(message) != length:
            return None
        
        # Read and verify checksum
        checksum_bytes = await reader.read(4)
        if len(checksum_bytes) != 4:
            return None
        expected_checksum = struct.unpack('<I', checksum_bytes)[0]
        actual_checksum = sum(message) & 0xFFFFFFFF
        
        if expected_checksum != actual_checksum:
            logger.error("Checksum mismatch")
            return None
        
        return message
    
    async def send_response(self, response: pb.MCPResponse):
        """Send a protobuf response"""
        # Set common fields
        response.session_id = self.session_id
        response.timestamp.GetCurrentTime()
        
        # Serialize
        message_bytes = response.SerializeToString()
        
        # Frame the message
        framed = self.frame_message(message_bytes)
        
        # Write to stdout
        sys.stdout.buffer.write(framed)
        sys.stdout.buffer.flush()
    
    def frame_message(self, message: bytes) -> bytes:
        """Add framing to a message"""
        length = len(message)
        checksum = sum(message) & 0xFFFFFFFF
        
        return (
            b'MCP\x01' +
            struct.pack('<I', length) +
            message +
            struct.pack('<I', checksum)
        )
    
    async def process_request(self, request: pb.MCPRequest) -> pb.MCPResponse:
        """Process an incoming request"""
        response = pb.MCPResponse(request_id=request.request_id)
        
        try:
            if request.HasField("initialize"):
                # Handle initialization
                await self.handle_initialize(request.initialize, response)
                
            elif request.HasField("tool_call"):
                # Handle tool call
                await self.handle_tool_call(request.tool_call, response)
                
            elif request.HasField("list_tools"):
                # Handle list tools
                await self.handle_list_tools(request.list_tools, response)
                
            elif request.HasField("get_context"):
                # Handle get context
                await self.handle_get_context(request.get_context, response)
                
            elif request.HasField("shutdown"):
                # Handle shutdown
                await self.handle_shutdown(request.shutdown, response)
                
            else:
                # Unknown request type
                response.error.error_type = pb.Error.UNKNOWN
                response.error.message = "Unknown request type"
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            response.error.error_type = pb.Error.INTERNAL_ERROR
            response.error.message = str(e)
            
        return response
    
    async def handle_initialize(self, init: pb.Initialize, response: pb.MCPResponse):
        """Handle initialization request"""
        # Initialize database if needed
        if db is None:
            # Initialize database (from main.py logic)
            from ..core.database import ChromaDBManager
            global db
            db = ChromaDBManager()
            await db.initialize()
        
        response.acknowledgment.success = True
        response.acknowledgment.message = f"Session {self.session_id} initialized"
    
    async def handle_tool_call(self, tool_call: pb.ToolCall, response: pb.MCPResponse):
        """Handle tool invocation"""
        tool_name = tool_call.tool_name
        arguments = json_format.MessageToDict(tool_call.arguments)
        
        # Find the tool in MCP server
        tool_func = None
        for name, func in mcp._tools.items():
            if name == tool_name:
                tool_func = func
                break
        
        if not tool_func:
            response.error.error_type = pb.Error.TOOL_NOT_FOUND
            response.error.message = f"Tool '{tool_name}' not found"
            return
        
        try:
            # Execute the tool
            result = await tool_func(**arguments)
            
            # Determine if we should use Arrow for the result
            result_size = len(json.dumps(result))
            use_arrow = (
                tool_call.use_arrow_for_result and 
                self.plasma_client and 
                result_size > 10000  # Use Arrow for results > 10KB
            )
            
            if use_arrow:
                # Store in Arrow shared memory
                object_id = self.store_in_arrow(result)
                
                response.tool_result.success = True
                response.tool_result.arrow_reference.object_id = object_id
                response.tool_result.arrow_reference.size_bytes = result_size
                response.tool_result.arrow_reference.schema_type = "table"
                
            else:
                # Send via protobuf
                response.tool_result.success = True
                json_format.ParseDict(result, response.tool_result.json_result)
            
        except Exception as e:
            response.error.error_type = pb.Error.INTERNAL_ERROR
            response.error.message = f"Tool execution failed: {str(e)}"
    
    async def handle_list_tools(self, list_tools: pb.ListTools, response: pb.MCPResponse):
        """Handle list tools request"""
        for name, func in mcp._tools.items():
            tool_info = response.tool_list.tools.add()
            tool_info.name = name
            tool_info.description = func.__doc__ or ""
            tool_info.category = "general"  # Could be extracted from metadata
            
            # Extract parameter schema from function signature
            import inspect
            sig = inspect.signature(func)
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name not in ['self', 'cls']:
                    params[param_name] = {
                        "type": "string",  # Simplified - could use type hints
                        "required": param.default == inspect.Parameter.empty
                    }
            
            json_format.ParseDict({"properties": params}, tool_info.parameter_schema)
    
    async def handle_get_context(self, get_context: pb.GetContext, response: pb.MCPResponse):
        """Handle get context request"""
        # Get campaign data if available
        campaign_id = get_context.campaign_id
        
        context_data = {
            "campaign_id": campaign_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add campaign data if available
        if campaign_id and db:
            try:
                from ..campaign.campaign_manager import CampaignManager
                campaign_manager = CampaignManager(db)
                campaign = await campaign_manager.get_campaign(campaign_id)
                if campaign:
                    context_data["campaign"] = campaign
            except Exception as e:
                logger.warning(f"Could not load campaign: {e}")
        
        response.context_data.campaign_id = campaign_id
        json_format.ParseDict(context_data, response.context_data.current_state)
    
    async def handle_shutdown(self, shutdown: pb.Shutdown, response: pb.MCPResponse):
        """Handle shutdown request"""
        self.running = False
        
        # Clean up shared memory if requested
        if shutdown.cleanup_shared_memory and self.plasma_client:
            # Could iterate through known object IDs and delete them
            pass
        
        response.acknowledgment.success = True
        response.acknowledgment.message = "Shutting down"
    
    def store_in_arrow(self, data: Any) -> bytes:
        """Store data in Arrow shared memory"""
        # Convert to Arrow table
        if isinstance(data, dict):
            arrow_table = pa.Table.from_pydict(data)
        else:
            arrow_table = pa.Table.from_pydict({"result": [data]})
        
        # Generate object ID
        object_id = plasma.ObjectID(os.urandom(20))
        
        # Serialize and store
        buf = pa.serialize(arrow_table).to_buffer()
        self.plasma_client.put(buf, object_id)
        
        return object_id.binary()


async def main():
    """Main entry point for adapted MCP server"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if we're in bridge mode
    if os.environ.get("MCP_BRIDGE_MODE") != "true":
        # Run normal MCP server
        from ..main import main as original_main
        await original_main()
    else:
        # Run with protobuf adapter
        adapter = MCPProtobufAdapter()
        await adapter.start()


if __name__ == "__main__":
    asyncio.run(main())