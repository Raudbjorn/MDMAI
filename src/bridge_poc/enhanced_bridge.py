"""
Enhanced Port-free IPC Bridge incorporating best practices from claude-ipc-mcp
Combines our Protocol Buffers/Arrow approach with their excellent patterns
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.plasma as plasma
from google.protobuf import json_format, timestamp_pb2

try:
    from . import mcp_protocol_pb2 as pb
except ImportError:
    import mcp_protocol_pb2 as pb

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for the enhanced bridge"""
    # IPC settings (no ports!)
    ipc_socket_path: str = "/tmp/mcp_bridge.sock"  # Unix domain socket
    plasma_socket_path: str = "/tmp/plasma"
    
    # Security settings
    require_auth: bool = True
    shared_secret: Optional[str] = None
    token_expiry_hours: int = 24
    
    # Rate limiting
    max_requests_per_minute: int = 100
    
    # Persistence
    db_path: str = "~/.ttrpg-mcp-data/bridge.db"
    
    # Session management
    max_sessions: int = 100
    session_ttl_hours: int = 24
    
    # Message broker settings
    message_retention_days: int = 7
    max_message_size_kb: int = 10240  # 10MB
    
    def __post_init__(self):
        """Initialize paths and create directories"""
        self.db_path = os.path.expanduser(self.db_path)
        os.makedirs(os.path.dirname(self.db_path), mode=0o700, exist_ok=True)
        
        # Get shared secret from environment if not provided
        if self.require_auth and not self.shared_secret:
            self.shared_secret = os.environ.get("MCP_SHARED_SECRET")
            if not self.shared_secret:
                logger.warning("No shared secret provided - running in open mode")
                self.require_auth = False


class RateLimiter:
    """Rate limiter for sessions"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
        
    async def check_rate_limit(self, session_id: str) -> bool:
        """Check if session is within rate limits"""
        now = time.time()
        
        if session_id not in self.requests:
            self.requests[session_id] = []
        
        # Clean old requests
        self.requests[session_id] = [
            ts for ts in self.requests[session_id]
            if now - ts < self.window_seconds
        ]
        
        # Check limit
        if len(self.requests[session_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[session_id].append(now)
        return True


class SessionManager:
    """Manages authentication and sessions"""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.tokens: Dict[str, str] = {}  # token -> session_id
        
    def create_session(self, instance_name: str) -> Tuple[str, str]:
        """Create a new authenticated session"""
        session_id = str(uuid.uuid4())
        token = secrets.token_urlsafe(32)
        
        # Hash token for storage
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        self.sessions[session_id] = {
            "instance_name": instance_name,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "token_hash": token_hash
        }
        
        self.tokens[token] = session_id
        
        return session_id, token
    
    def validate_token(self, token: str) -> Optional[str]:
        """Validate a session token"""
        if not self.config.require_auth:
            return "anonymous"
        
        session_id = self.tokens.get(token)
        if not session_id:
            return None
        
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Check expiry
        if datetime.now() - session["created_at"] > timedelta(hours=self.config.token_expiry_hours):
            del self.sessions[session_id]
            del self.tokens[token]
            return None
        
        # Update activity
        session["last_activity"] = datetime.now()
        return session_id


class MessageBroker:
    """Enhanced message broker with persistence and natural language support"""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.db_conn: Optional[sqlite3.Connection] = None
        self.instances: Dict[str, Dict[str, Any]] = {}  # instance_name -> info
        self.name_aliases: Dict[str, str] = {}  # alias -> canonical_name
        
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for message persistence"""
        self.db_conn = sqlite3.connect(
            self.config.db_path,
            check_same_thread=False
        )
        
        # Create tables
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                recipient TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                delivered BOOLEAN DEFAULT 0,
                delivered_at TIMESTAMP
            )
        """)
        
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS instances (
                name TEXT PRIMARY KEY,
                session_id TEXT,
                last_seen TIMESTAMP,
                metadata TEXT
            )
        """)
        
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS name_history (
                old_name TEXT,
                new_name TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_conn.commit()
        
        # Load instances
        self._load_instances()
    
    def _load_instances(self):
        """Load known instances from database"""
        cursor = self.db_conn.execute("SELECT name, metadata FROM instances")
        for row in cursor:
            self.instances[row[0]] = json.loads(row[1]) if row[1] else {}
    
    async def register_instance(self, name: str, session_id: str) -> bool:
        """Register an instance with a friendly name"""
        # Check if name is taken
        if name in self.instances and self.instances[name].get("session_id") != session_id:
            return False
        
        self.instances[name] = {
            "session_id": session_id,
            "registered_at": datetime.now().isoformat()
        }
        
        # Persist to database
        self.db_conn.execute(
            "INSERT OR REPLACE INTO instances (name, session_id, last_seen) VALUES (?, ?, ?)",
            (name, session_id, datetime.now())
        )
        self.db_conn.commit()
        
        logger.info(f"Registered instance '{name}' with session {session_id}")
        return True
    
    async def rename_instance(self, old_name: str, new_name: str, session_id: str) -> bool:
        """Rename an instance with forwarding"""
        if old_name not in self.instances:
            return False
        
        if self.instances[old_name].get("session_id") != session_id:
            return False  # Not authorized
        
        # Record name change
        self.db_conn.execute(
            "INSERT INTO name_history (old_name, new_name) VALUES (?, ?)",
            (old_name, new_name)
        )
        
        # Update instance
        self.instances[new_name] = self.instances.pop(old_name)
        self.name_aliases[old_name] = new_name
        
        # Update database
        self.db_conn.execute(
            "UPDATE instances SET name = ? WHERE name = ?",
            (new_name, old_name)
        )
        self.db_conn.execute(
            "UPDATE messages SET recipient = ? WHERE recipient = ? AND delivered = 0",
            (new_name, old_name)
        )
        self.db_conn.commit()
        
        logger.info(f"Renamed instance '{old_name}' to '{new_name}'")
        return True
    
    async def send_message(
        self,
        sender: str,
        recipient: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a message to another instance"""
        # Resolve aliases
        recipient = self.name_aliases.get(recipient, recipient)
        
        # Store in database
        cursor = self.db_conn.execute(
            """INSERT INTO messages (sender, recipient, content, metadata) 
               VALUES (?, ?, ?, ?)""",
            (sender, recipient, content, json.dumps(metadata) if metadata else None)
        )
        message_id = cursor.lastrowid
        self.db_conn.commit()
        
        logger.info(f"Message {message_id} from '{sender}' to '{recipient}' stored")
        return str(message_id)
    
    async def check_messages(self, recipient: str) -> List[Dict[str, Any]]:
        """Check for undelivered messages"""
        # Resolve aliases
        recipient = self.name_aliases.get(recipient, recipient)
        
        cursor = self.db_conn.execute(
            """SELECT id, sender, content, metadata, created_at 
               FROM messages 
               WHERE recipient = ? AND delivered = 0
               ORDER BY created_at ASC""",
            (recipient,)
        )
        
        messages = []
        for row in cursor:
            messages.append({
                "id": row[0],
                "sender": row[1],
                "content": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "timestamp": row[4]
            })
        
        # Mark as delivered
        if messages:
            message_ids = [msg["id"] for msg in messages]
            placeholders = ",".join("?" * len(message_ids))
            self.db_conn.execute(
                f"UPDATE messages SET delivered = 1, delivered_at = ? WHERE id IN ({placeholders})",
                [datetime.now()] + message_ids
            )
            self.db_conn.commit()
        
        return messages
    
    async def list_instances(self) -> List[Dict[str, Any]]:
        """List all known instances"""
        cursor = self.db_conn.execute(
            "SELECT name, last_seen FROM instances ORDER BY last_seen DESC"
        )
        
        instances = []
        for row in cursor:
            instances.append({
                "name": row[0],
                "last_seen": row[1],
                "online": (datetime.now() - datetime.fromisoformat(row[1])).seconds < 300
            })
        
        return instances


class NaturalLanguageProcessor:
    """Process natural language commands for messaging"""
    
    COMMAND_PATTERNS = {
        "register": [
            r"register (?:this instance |me )?as (\w+)",
            r"my name is (\w+)",
            r"call me (\w+)"
        ],
        "send": [
            r"send (?:a )?(?:message )?to (\w+):? (.+)",
            r"msg (\w+):? (.+)",
            r"tell (\w+):? (.+)"
        ],
        "check": [
            r"check (?:my )?messages?",
            r"any messages\??",
            r"what did (?:I |they )?(?:say|send)\??"
        ],
        "rename": [
            r"rename (?:me |this )?(?:from )?(\w+) to (\w+)",
            r"change (?:my )?name (?:from )?(\w+)? ?to (\w+)"
        ],
        "list": [
            r"(?:list|show) (?:all )?instances?",
            r"who(?:'s| is) online\??",
            r"whos here\??"
        ]
    }
    
    @staticmethod
    def parse_command(text: str) -> Optional[Dict[str, Any]]:
        """Parse natural language command"""
        import re
        
        text = text.lower().strip()
        
        for command_type, patterns in NaturalLanguageProcessor.COMMAND_PATTERNS.items():
            for pattern in patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    if command_type == "register":
                        return {"type": "register", "name": match.group(1)}
                    elif command_type == "send":
                        return {
                            "type": "send",
                            "recipient": match.group(1),
                            "content": match.group(2)
                        }
                    elif command_type == "check":
                        return {"type": "check"}
                    elif command_type == "rename":
                        groups = match.groups()
                        if len(groups) == 2:
                            return {
                                "type": "rename",
                                "old_name": groups[0] if groups[0] else None,
                                "new_name": groups[1]
                            }
                    elif command_type == "list":
                        return {"type": "list"}
        
        return None


class EnhancedMCPBridge:
    """Enhanced MCP Bridge with message broker and natural language support"""
    
    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self.sessions: Dict[str, MCPSession] = {}
        self.arrow_manager = ArrowDataManager(self.config.plasma_socket_path)
        self.session_manager = SessionManager(self.config)
        self.message_broker = MessageBroker(self.config)
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute)
        self.nlp = NaturalLanguageProcessor()
        
        # Unix domain socket for IPC (no TCP port!)
        self.ipc_socket_path = self.config.ipc_socket_path
        self.server_socket = None
        self.running = False
    
    async def start(self):
        """Start the enhanced bridge server"""
        logger.info("Starting Enhanced MCP Bridge (port-free with messaging)")
        
        # Start Arrow shared memory
        await self.arrow_manager.start()
        
        # Create Unix domain socket for IPC
        if os.path.exists(self.ipc_socket_path):
            os.unlink(self.ipc_socket_path)
        
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.ipc_socket_path)
        self.server_socket.listen(5)
        
        self.running = True
        
        # Start accepting connections
        asyncio.create_task(self._accept_connections())
        
        logger.info(f"Bridge server started on {self.ipc_socket_path}")
    
    async def process_natural_language(
        self,
        text: str,
        session_id: str,
        instance_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process natural language commands"""
        command = self.nlp.parse_command(text)
        
        if not command:
            return {
                "success": False,
                "error": "Could not understand command",
                "suggestion": "Try: 'Register as [name]', 'Send to [name]: [message]', 'Check messages'"
            }
        
        if command["type"] == "register":
            success = await self.message_broker.register_instance(
                command["name"], session_id
            )
            return {
                "success": success,
                "message": f"Registered as '{command['name']}'" if success else "Name already taken"
            }
        
        elif command["type"] == "send":
            if not instance_name:
                return {
                    "success": False,
                    "error": "You must register first (e.g., 'Register as claude')"
                }
            
            message_id = await self.message_broker.send_message(
                instance_name,
                command["recipient"],
                command["content"]
            )
            return {
                "success": True,
                "message": f"Message sent to '{command['recipient']}' (ID: {message_id})"
            }
        
        elif command["type"] == "check":
            if not instance_name:
                return {
                    "success": False,
                    "error": "You must register first"
                }
            
            messages = await self.message_broker.check_messages(instance_name)
            return {
                "success": True,
                "messages": messages,
                "count": len(messages)
            }
        
        elif command["type"] == "list":
            instances = await self.message_broker.list_instances()
            return {
                "success": True,
                "instances": instances
            }
        
        elif command["type"] == "rename":
            if not instance_name:
                return {
                    "success": False,
                    "error": "You must register first"
                }
            
            old_name = command["old_name"] or instance_name
            success = await self.message_broker.rename_instance(
                old_name, command["new_name"], session_id
            )
            return {
                "success": success,
                "message": f"Renamed from '{old_name}' to '{command['new_name']}'" if success else "Rename failed"
            }
        
        return {"success": False, "error": "Unknown command type"}


# Example usage combining best of both approaches
async def demo():
    """Demonstrate the enhanced bridge with messaging capabilities"""
    logging.basicConfig(level=logging.INFO)
    
    # Create bridge with configuration
    config = BridgeConfig(
        shared_secret="demo-secret-key",
        require_auth=True
    )
    
    bridge = EnhancedMCPBridge(config)
    await bridge.start()
    
    # Create two sessions (simulating two AI instances)
    session1, token1 = bridge.session_manager.create_session("claude")
    session2, token2 = bridge.session_manager.create_session("gemini")
    
    # Register instances
    await bridge.process_natural_language(
        "Register this instance as claude",
        session1,
        "claude"
    )
    
    await bridge.process_natural_language(
        "Register as gemini",
        session2,
        "gemini"
    )
    
    # Send a message
    result = await bridge.process_natural_language(
        "Send to gemini: Hey, can you help with this Python function?",
        session1,
        "claude"
    )
    print(f"Claude says: {result}")
    
    # Check messages
    result = await bridge.process_natural_language(
        "Check my messages",
        session2,
        "gemini"
    )
    print(f"Gemini received: {result}")
    
    # List instances
    result = await bridge.process_natural_language(
        "Who's online?",
        session1,
        "claude"
    )
    print(f"Online instances: {result}")


if __name__ == "__main__":
    asyncio.run(demo())