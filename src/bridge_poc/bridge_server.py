"""
Port-free IPC Bridge Server
Uses Protocol Buffers over stdio for control and Apache Arrow for data
"""

import asyncio
import json
import logging
import os
import struct
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.plasma as plasma
from google.protobuf import json_format, timestamp_pb2
from google.protobuf.struct_pb2 import Struct

# Note: These imports assume protoc has been run on mcp_protocol.proto
# Run: protoc --python_out=. --pyi_out=. mcp_protocol.proto
try:
    from . import mcp_protocol_pb2 as pb
except ImportError:
    import mcp_protocol_pb2 as pb

logger = logging.getLogger(__name__)


@dataclass
class MCPSession:
    """Represents an MCP server session"""
    session_id: str
    process: subprocess.Popen
    plasma_client: Optional[plasma.PlasmaClient] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProtobufFramer:
    """Handles framing of protobuf messages over stdio"""
    
    MAGIC = b'MCP\x01'  # Magic bytes + version
    
    @staticmethod
    def frame_message(message: bytes) -> bytes:
        """Add framing to a protobuf message"""
        # Format: [MAGIC][LENGTH][MESSAGE][CHECKSUM]
        length = len(message)
        checksum = sum(message) & 0xFFFFFFFF  # Simple checksum
        
        frame = (
            ProtobufFramer.MAGIC +
            struct.pack('<I', length) +
            message +
            struct.pack('<I', checksum)
        )
        return frame
    
    @staticmethod
    async def read_message(stream: asyncio.StreamReader) -> Optional[bytes]:
        """Read a framed message from stream"""
        # Read magic bytes
        magic = await stream.read(4)
        if not magic or magic != ProtobufFramer.MAGIC:
            return None
        
        # Read length
        length_bytes = await stream.read(4)
        if len(length_bytes) != 4:
            return None
        length = struct.unpack('<I', length_bytes)[0]
        
        # Read message
        message = await stream.read(length)
        if len(message) != length:
            return None
        
        # Read and verify checksum
        checksum_bytes = await stream.read(4)
        if len(checksum_bytes) != 4:
            return None
        expected_checksum = struct.unpack('<I', checksum_bytes)[0]
        actual_checksum = sum(message) & 0xFFFFFFFF
        
        if expected_checksum != actual_checksum:
            logger.error("Checksum mismatch in protobuf message")
            return None
        
        return message


class ArrowDataManager:
    """Manages Apache Arrow shared memory operations"""
    
    def __init__(self, plasma_socket_path: str = "/tmp/plasma"):
        self.plasma_socket_path = plasma_socket_path
        self.plasma_client = None
        self.plasma_store_process = None
        
    async def start(self, memory_size: int = 2_000_000_000):  # 2GB default
        """Start Plasma store and connect client"""
        try:
            # Start Plasma store
            self.plasma_store_process = subprocess.Popen([
                "plasma_store",
                "-m", str(memory_size),
                "-s", self.plasma_socket_path
            ])
            
            # Wait for store to be ready
            await asyncio.sleep(0.5)
            
            # Connect client
            self.plasma_client = plasma.connect(self.plasma_socket_path)
            logger.info(f"Started Plasma store at {self.plasma_socket_path}")
            
        except Exception as e:
            logger.error(f"Failed to start Plasma store: {e}")
            raise
    
    async def stop(self):
        """Stop Plasma store and clean up"""
        if self.plasma_client:
            self.plasma_client.disconnect()
        
        if self.plasma_store_process:
            self.plasma_store_process.terminate()
            self.plasma_store_process.wait(timeout=5)
        
        # Clean up socket file
        socket_path = Path(self.plasma_socket_path)
        if socket_path.exists():
            socket_path.unlink()
    
    def store_data(self, data: Any, data_type: str = "table") -> bytes:
        """Store data in shared memory and return object ID"""
        # Convert data to Arrow format
        if data_type == "table":
            if isinstance(data, dict):
                arrow_data = pa.Table.from_pydict(data)
            elif isinstance(data, pa.Table):
                arrow_data = data
            else:
                # Convert to dict first
                arrow_data = pa.Table.from_pydict({"data": [data]})
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Generate object ID (20 bytes for Plasma)
        object_id = plasma.ObjectID(os.urandom(20))
        
        # Serialize and store
        buf = pa.serialize(arrow_data).to_buffer()
        self.plasma_client.put(buf, object_id)
        
        return object_id.binary()
    
    def retrieve_data(self, object_id_bytes: bytes) -> Any:
        """Retrieve data from shared memory"""
        object_id = plasma.ObjectID(object_id_bytes)
        
        # Get buffer (zero-copy)
        [buf] = self.plasma_client.get_buffers([object_id])
        
        # Deserialize
        return pa.deserialize(buf)
    
    def delete_data(self, object_id_bytes: bytes):
        """Delete data from shared memory"""
        object_id = plasma.ObjectID(object_id_bytes)
        self.plasma_client.delete([object_id])


class MCPBridge:
    """Main bridge server that manages MCP sessions without TCP/UDP ports"""
    
    def __init__(self, mcp_server_path: str = "src.main"):
        self.mcp_server_path = mcp_server_path
        self.sessions: Dict[str, MCPSession] = {}
        self.arrow_manager = ArrowDataManager()
        self.running = False
        
    async def start(self):
        """Start the bridge server"""
        logger.info("Starting MCP Bridge Server (port-free)")
        
        # Start Arrow shared memory
        await self.arrow_manager.start()
        
        self.running = True
        logger.info("MCP Bridge Server started")
    
    async def stop(self):
        """Stop the bridge server and clean up"""
        logger.info("Stopping MCP Bridge Server")
        self.running = False
        
        # Stop all sessions
        for session_id in list(self.sessions.keys()):
            await self.stop_session(session_id)
        
        # Stop Arrow manager
        await self.arrow_manager.stop()
        
        logger.info("MCP Bridge Server stopped")
    
    async def create_session(self, campaign_id: Optional[str] = None) -> str:
        """Create a new MCP server session"""
        session_id = str(uuid.uuid4())
        
        # Start MCP server process
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", self.mcp_server_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "MCP_SESSION_ID": session_id,
                "MCP_PLASMA_SOCKET": self.arrow_manager.plasma_socket_path,
                "MCP_BRIDGE_MODE": "true"
            }
        )
        
        # Create session
        session = MCPSession(
            session_id=session_id,
            process=process,
            plasma_client=self.arrow_manager.plasma_client
        )
        
        self.sessions[session_id] = session
        
        # Send initialization message
        init_request = pb.MCPRequest(
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=timestamp_pb2.Timestamp().GetCurrentTime()
        )
        init_request.initialize.campaign_id = campaign_id or ""
        init_request.initialize.plasma_socket_path = self.arrow_manager.plasma_socket_path
        
        await self._send_request(session, init_request)
        response = await self._receive_response(session)
        
        if response and response.HasField("acknowledgment"):
            logger.info(f"Created session {session_id}")
            return session_id
        else:
            await self.stop_session(session_id)
            raise RuntimeError("Failed to initialize MCP session")
    
    async def stop_session(self, session_id: str):
        """Stop an MCP server session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Send shutdown message
        shutdown_request = pb.MCPRequest(
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=timestamp_pb2.Timestamp().GetCurrentTime()
        )
        shutdown_request.shutdown.cleanup_shared_memory = True
        
        try:
            await self._send_request(session, shutdown_request)
            # Wait briefly for acknowledgment
            await asyncio.wait_for(
                self._receive_response(session),
                timeout=2.0
            )
        except Exception:
            pass  # Process might already be gone
        
        # Terminate process
        if session.process.returncode is None:
            session.process.terminate()
            try:
                await asyncio.wait_for(
                    session.process.wait(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                session.process.kill()
                await session.process.wait()
        
        # Clean up
        del self.sessions[session_id]
        logger.info(f"Stopped session {session_id}")
    
    async def call_tool(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        use_arrow_for_large_results: bool = True
    ) -> Dict[str, Any]:
        """Call an MCP tool and get the result"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.last_activity = datetime.now()
        
        # Create request
        request = pb.MCPRequest(
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=timestamp_pb2.Timestamp().GetCurrentTime()
        )
        
        # Set tool call details
        request.tool_call.tool_name = tool_name
        json_format.ParseDict(arguments, request.tool_call.arguments)
        request.tool_call.use_arrow_for_result = use_arrow_for_large_results
        
        # Send request and wait for response
        await self._send_request(session, request)
        response = await self._receive_response(session)
        
        if not response:
            raise RuntimeError("No response from MCP server")
        
        if response.HasField("error"):
            error = response.error
            raise RuntimeError(f"Tool error: {error.message}")
        
        if response.HasField("tool_result"):
            result = response.tool_result
            
            if result.HasField("json_result"):
                # Small result via protobuf
                return json_format.MessageToDict(result.json_result)
            
            elif result.HasField("arrow_reference"):
                # Large result via Arrow shared memory
                ref = result.arrow_reference
                data = self.arrow_manager.retrieve_data(ref.object_id)
                
                # Convert Arrow data back to Python
                if ref.schema_type == "table":
                    return data.to_pydict()
                else:
                    return data
            
            else:
                return {"success": result.success, "messages": list(result.messages)}
        
        raise RuntimeError("Unexpected response type")
    
    async def list_tools(self, session_id: str) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Create request
        request = pb.MCPRequest(
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=timestamp_pb2.Timestamp().GetCurrentTime()
        )
        request.list_tools.CopyFrom(pb.ListTools())
        
        # Send and receive
        await self._send_request(session, request)
        response = await self._receive_response(session)
        
        if response and response.HasField("tool_list"):
            tools = []
            for tool_info in response.tool_list.tools:
                tools.append({
                    "name": tool_info.name,
                    "description": tool_info.description,
                    "category": tool_info.category,
                    "parameters": json_format.MessageToDict(tool_info.parameter_schema),
                    "permissions": list(tool_info.required_permissions)
                })
            return tools
        
        return []
    
    async def _send_request(self, session: MCPSession, request: pb.MCPRequest):
        """Send a protobuf request to MCP server"""
        message_bytes = request.SerializeToString()
        framed_message = ProtobufFramer.frame_message(message_bytes)
        
        session.process.stdin.write(framed_message)
        await session.process.stdin.drain()
    
    async def _receive_response(self, session: MCPSession) -> Optional[pb.MCPResponse]:
        """Receive a protobuf response from MCP server"""
        try:
            # Create async reader from process stdout
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await asyncio.get_event_loop().connect_read_pipe(
                lambda: protocol, session.process.stdout
            )
            
            # Read framed message
            message_bytes = await ProtobufFramer.read_message(reader)
            if message_bytes:
                response = pb.MCPResponse()
                response.ParseFromString(message_bytes)
                return response
            
        except Exception as e:
            logger.error(f"Error receiving response: {e}")
        
        return None


# Example usage
async def main():
    """Example of using the port-free bridge"""
    logging.basicConfig(level=logging.INFO)
    
    bridge = MCPBridge()
    await bridge.start()
    
    try:
        # Create a session
        session_id = await bridge.create_session(campaign_id="test_campaign")
        print(f"Created session: {session_id}")
        
        # List available tools
        tools = await bridge.list_tools(session_id)
        print(f"Available tools: {[t['name'] for t in tools]}")
        
        # Call a tool (example)
        result = await bridge.call_tool(
            session_id,
            "search",
            {"query": "find fireball spell", "max_results": 5}
        )
        print(f"Search result: {result}")
        
        # Stop session
        await bridge.stop_session(session_id)
        
    finally:
        await bridge.stop()


if __name__ == "__main__":
    asyncio.run(main())