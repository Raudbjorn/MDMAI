"""MCP Process Manager for handling stdio subprocess lifecycle."""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
from structlog import get_logger

from .models import (
    BridgeConfig,
    MCPError,
    MCPNotification,
    MCPRequest,
    PendingRequest,
    ProcessStats,
)

logger = get_logger(__name__)


class MCPProcess:
    """Manages a single MCP server subprocess."""
    
    def __init__(
        self,
        session_id: str,
        config: BridgeConfig,
        env: Optional[Dict[str, str]] = None,
    ):
        self.session_id = session_id
        self.config = config
        self.env = env or {}
        self.process: Optional[asyncio.subprocess.Process] = None
        self.reader_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.pending_requests: Dict[str, PendingRequest] = {}
        self.capabilities: Dict[str, Any] = {}
        self.started_at: Optional[datetime] = None
        self.last_activity: Optional[datetime] = None
        self.num_requests: int = 0
        self.num_errors: int = 0
        self.restart_count: int = 0
        self.max_restarts: int = 3  # Maximum restart attempts
        self._lock = asyncio.Lock()
        self._running = False
        self._initialized = False
        self._auto_restart = True
    
    async def start(self) -> bool:
        """Start the MCP server subprocess."""
        async with self._lock:
            if self._running:
                logger.warning("Process already running", session_id=self.session_id)
                return True
            
            try:
                # Prepare environment
                process_env = os.environ.copy()
                process_env.update(self.config.mcp_server_env)
                process_env.update(self.env)
                
                # Set MCP mode to stdio
                process_env["MCP_STDIO_MODE"] = "true"
                
                # Start the subprocess
                cmd = [sys.executable, "-m", self.config.mcp_server_path]
                cmd.extend(self.config.mcp_server_args)
                
                logger.info(
                    "Starting MCP process",
                    session_id=self.session_id,
                    cmd=" ".join(cmd),
                )
                
                self.process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=process_env,
                    limit=1024 * 1024,  # 1MB buffer
                )
                
                self._running = True
                self.started_at = datetime.now()
                self.last_activity = datetime.now()
                
                # Start reader task
                self.reader_task = asyncio.create_task(self._read_output())
                
                # Start health check task
                self.health_check_task = asyncio.create_task(self._health_check_loop())
                
                # Initialize the MCP connection
                await self._initialize_mcp()
                
                logger.info(
                    "MCP process started",
                    session_id=self.session_id,
                    pid=self.process.pid,
                )
                
                return True
                
            except Exception as e:
                logger.error(
                    "Failed to start MCP process",
                    session_id=self.session_id,
                    error=str(e),
                )
                await self.stop()
                return False
    
    async def stop(self) -> None:
        """Stop the MCP server subprocess."""
        async with self._lock:
            self._running = False
            
            # Cancel tasks
            if self.reader_task:
                self.reader_task.cancel()
                try:
                    await self.reader_task
                except asyncio.CancelledError:
                    pass
            
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Terminate process
            if self.process:
                try:
                    # Send shutdown notification if initialized
                    if self._initialized:
                        await self._send_notification("shutdown", {})
                        await asyncio.sleep(0.5)  # Give it time to shutdown gracefully
                    
                    # Terminate the process
                    self.process.terminate()
                    
                    # Wait for termination with timeout
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Force kill if not terminated
                        self.process.kill()
                        await self.process.wait()
                    
                    logger.info(
                        "MCP process stopped",
                        session_id=self.session_id,
                        pid=self.process.pid,
                    )
                    
                except Exception as e:
                    logger.error(
                        "Error stopping MCP process",
                        session_id=self.session_id,
                        error=str(e),
                    )
                
                self.process = None
    
    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a request to the MCP server and await response."""
        if not self._running or not self.process:
            raise RuntimeError("MCP process not running")
        
        request = MCPRequest(method=method, params=params)
        request_id = str(request.id)
        
        # Create pending request
        future = asyncio.Future()
        pending = PendingRequest(
            request_id=request_id,
            method=method,
            params=params,
            timeout=timeout or self.config.request_timeout,
            callback=future,
        )
        
        self.pending_requests[request_id] = pending
        
        try:
            # Send request
            await self._write_message(request.dict())
            
            # Update activity
            self.last_activity = datetime.now()
            self.num_requests += 1
            
            # Wait for response with timeout
            response = await asyncio.wait_for(
                future,
                timeout=pending.timeout,
            )
            
            return response
            
        except asyncio.TimeoutError:
            self.num_errors += 1
            self.pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request {method} timed out after {pending.timeout}s")
        except Exception as e:
            self.num_errors += 1
            self.pending_requests.pop(request_id, None)
            raise
    
    async def send_notification(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a notification to the MCP server (no response expected)."""
        if not self._running or not self.process:
            raise RuntimeError("MCP process not running")
        
        notification = MCPNotification(method=method, params=params)
        
        await self._write_message(notification.dict())
        
        # Update activity
        self.last_activity = datetime.now()
    
    async def _initialize_mcp(self) -> None:
        """Initialize the MCP connection and discover capabilities."""
        try:
            # Send initialize request
            response = await self.send_request(
                "initialize",
                params={
                    "protocolVersion": "1.0",
                    "clientInfo": {
                        "name": "mcp-bridge",
                        "version": "0.1.0",
                    },
                },
                timeout=10.0,
            )
            
            # Store capabilities
            self.capabilities = response.get("capabilities", {})
            
            # Send initialized notification
            await self.send_notification("initialized", {})
            
            self._initialized = True
            
            logger.info(
                "MCP connection initialized",
                session_id=self.session_id,
                capabilities=self.capabilities,
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize MCP connection",
                session_id=self.session_id,
                error=str(e),
            )
            raise
    
    async def _write_message(self, message: Dict[str, Any]) -> None:
        """Write a message to the subprocess stdin."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Process stdin not available")
        
        try:
            # Serialize and write message
            data = json.dumps(message) + "\n"
            self.process.stdin.write(data.encode())
            await self.process.stdin.drain()
            
            if self.config.log_requests:
                logger.debug(
                    "Sent message to MCP",
                    session_id=self.session_id,
                    message=message,
                )
                
        except Exception as e:
            logger.error(
                "Failed to write message to MCP",
                session_id=self.session_id,
                error=str(e),
            )
            raise
    
    async def _read_output(self) -> None:
        """Read output from the subprocess stdout."""
        if not self.process or not self.process.stdout:
            return
        
        try:
            while self._running:
                # Read line from stdout
                line = await self.process.stdout.readline()
                
                if not line:
                    # Process has ended
                    logger.warning("MCP process ended", session_id=self.session_id)
                    break
                
                try:
                    # Parse JSON message
                    message = json.loads(line.decode().strip())
                    
                    if self.config.log_responses:
                        logger.debug(
                            "Received message from MCP",
                            session_id=self.session_id,
                            message=message,
                        )
                    
                    # Handle the message
                    await self._handle_message(message)
                    
                except json.JSONDecodeError as e:
                    # Log non-JSON output (might be debug logs)
                    logger.debug(
                        "Non-JSON output from MCP",
                        session_id=self.session_id,
                        output=line.decode().strip(),
                    )
                except Exception as e:
                    logger.error(
                        "Error handling MCP message",
                        session_id=self.session_id,
                        error=str(e),
                    )
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "Error reading MCP output",
                session_id=self.session_id,
                error=str(e),
            )
        finally:
            self._running = False
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle a message from the MCP server."""
        # Check if it's a response
        if "id" in message and message["id"] in self.pending_requests:
            request_id = str(message["id"])
            pending = self.pending_requests.pop(request_id, None)
            
            if pending and pending.callback:
                if "error" in message:
                    # Error response
                    error = MCPError(**message["error"])
                    pending.callback.set_exception(
                        RuntimeError(f"MCP error: {error.message}")
                    )
                else:
                    # Success response
                    pending.callback.set_result(message.get("result"))
        
        # Check if it's a notification
        elif "method" in message and "id" not in message:
            # Handle notification (could be forwarded to clients)
            await self._handle_notification(message)
    
    async def _handle_notification(self, notification: Dict[str, Any]) -> None:
        """Handle a notification from the MCP server."""
        method = notification.get("method")
        params = notification.get("params", {})
        
        logger.debug(
            "Received notification from MCP",
            session_id=self.session_id,
            method=method,
            params=params,
        )
        
        # TODO: Forward notifications to connected clients
    
    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a notification to the MCP server."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._write_message(notification)
    
    async def _health_check_loop(self) -> None:
        """Periodically check process health."""
        try:
            while self._running:
                await asyncio.sleep(self.config.process_health_check_interval)
                
                if self.process:
                    # Check if process is still alive
                    if self.process.returncode is not None:
                        logger.error(
                            "MCP process died unexpectedly",
                            session_id=self.session_id,
                            returncode=self.process.returncode,
                            restart_count=self.restart_count,
                        )
                        
                        # Attempt automatic restart if enabled and not exceeded max restarts
                        if self._auto_restart and self.restart_count < self.max_restarts:
                            logger.info(
                                "Attempting to restart MCP process",
                                session_id=self.session_id,
                                restart_attempt=self.restart_count + 1,
                            )
                            
                            self._running = False
                            self.restart_count += 1
                            
                            # Cleanup current process
                            await self._cleanup_process()
                            
                            # Wait a bit before restarting
                            await asyncio.sleep(2)
                            
                            # Attempt restart
                            if await self.start():
                                logger.info(
                                    "MCP process restarted successfully",
                                    session_id=self.session_id,
                                    restart_count=self.restart_count,
                                )
                                continue
                            else:
                                logger.error(
                                    "Failed to restart MCP process",
                                    session_id=self.session_id,
                                    restart_count=self.restart_count,
                                )
                                break
                        else:
                            self._running = False
                            break
                    
                    # Check for idle timeout
                    if self.last_activity:
                        idle_time = (datetime.now() - self.last_activity).total_seconds()
                        if idle_time > self.config.process_idle_timeout:
                            logger.info(
                                "MCP process idle timeout",
                                session_id=self.session_id,
                                idle_time=idle_time,
                            )
                            await self.stop()
                            break
                            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "Error in health check loop",
                session_id=self.session_id,
                error=str(e),
            )
    
    async def _cleanup_process(self) -> None:
        """Cleanup process resources without full stop."""
        if self.reader_task:
            self.reader_task.cancel()
            try:
                await self.reader_task
            except asyncio.CancelledError:
                pass
            self.reader_task = None
        
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.debug(
                    "Error during process cleanup",
                    session_id=self.session_id,
                    error=str(e),
                )
            self.process = None
    
    def get_stats(self) -> ProcessStats:
        """Get process statistics."""
        stats = ProcessStats(
            process_id=self.process.pid if self.process else -1,
            session_id=self.session_id,
            num_requests=self.num_requests,
            num_errors=self.num_errors,
            last_request=self.last_activity,
        )
        
        if self.started_at:
            stats.uptime_seconds = (datetime.now() - self.started_at).total_seconds()
        
        if self.process and self.process.pid:
            try:
                proc = psutil.Process(self.process.pid)
                stats.cpu_percent = proc.cpu_percent()
                stats.memory_mb = proc.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return stats


class MCPProcessManager:
    """Manages a pool of MCP server subprocesses."""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.processes: Dict[str, MCPProcess] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._started = False
    
    async def start(self) -> None:
        """Start the process manager."""
        if self._started:
            return
        
        self._started = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("MCP process manager started")
    
    async def stop(self) -> None:
        """Stop the process manager and all processes."""
        self._started = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Stop all processes
        async with self._lock:
            tasks = [process.stop() for process in self.processes.values()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self.processes.clear()
        
        logger.info("MCP process manager stopped")
    
    async def create_process(
        self,
        session_id: str,
        env: Optional[Dict[str, str]] = None,
    ) -> MCPProcess:
        """Create and start a new MCP process for a session."""
        async with self._lock:
            # Check if process already exists
            if session_id in self.processes:
                return self.processes[session_id]
            
            # Check max processes limit
            if len(self.processes) >= self.config.max_processes:
                # Try to clean up idle processes
                await self._cleanup_idle_processes()
                
                if len(self.processes) >= self.config.max_processes:
                    raise RuntimeError(f"Maximum number of processes ({self.config.max_processes}) reached")
            
            # Create new process
            process = MCPProcess(session_id, self.config, env)
            
            # Start the process
            if not await process.start():
                raise RuntimeError("Failed to start MCP process")
            
            # Store the process
            self.processes[session_id] = process
            
            logger.info(
                "Created MCP process",
                session_id=session_id,
                total_processes=len(self.processes),
            )
            
            return process
    
    async def get_process(self, session_id: str) -> Optional[MCPProcess]:
        """Get an existing MCP process for a session."""
        return self.processes.get(session_id)
    
    async def remove_process(self, session_id: str) -> None:
        """Remove and stop a MCP process."""
        async with self._lock:
            process = self.processes.pop(session_id, None)
            if process:
                await process.stop()
                
                logger.info(
                    "Removed MCP process",
                    session_id=session_id,
                    total_processes=len(self.processes),
                )
    
    async def _cleanup_loop(self) -> None:
        """Periodically cleanup idle and dead processes."""
        try:
            while self._started:
                await asyncio.sleep(self.config.session_cleanup_interval)
                await self._cleanup_idle_processes()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in cleanup loop", error=str(e))
    
    async def _cleanup_idle_processes(self) -> None:
        """Cleanup idle and dead processes."""
        async with self._lock:
            to_remove = []
            
            for session_id, process in self.processes.items():
                # Check if process is dead
                if not process._running:
                    to_remove.append(session_id)
                    continue
                
                # Check for idle timeout
                if process.last_activity:
                    idle_time = (datetime.now() - process.last_activity).total_seconds()
                    if idle_time > self.config.process_idle_timeout:
                        to_remove.append(session_id)
            
            # Remove idle/dead processes
            for session_id in to_remove:
                process = self.processes.pop(session_id, None)
                if process:
                    await process.stop()
                    
                    logger.info(
                        "Cleaned up idle/dead process",
                        session_id=session_id,
                        total_processes=len(self.processes),
                    )
    
    def get_stats(self) -> List[ProcessStats]:
        """Get statistics for all processes."""
        return [process.get_stats() for process in self.processes.values()]