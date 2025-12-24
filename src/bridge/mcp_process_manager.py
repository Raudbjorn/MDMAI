"""MCP Process Manager for handling stdio subprocess lifecycle with modern patterns."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

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

# Result pattern for error handling
T = TypeVar('T')
E = TypeVar('E')


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """Result type for error handling without exceptions."""
    value: Optional[T] = None
    error: Optional[E] = None
    
    @classmethod
    def ok(cls, value: T) -> 'Result[T, E]':
        """Create successful result."""
        return cls(value=value)
    
    @classmethod
    def err(cls, error: E) -> 'Result[T, E]':
        """Create error result."""
        return cls(error=error)
    
    @property
    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self.error is None and self.value is not None
    
    @property
    def is_err(self) -> bool:
        """Check if result contains error."""
        return self.error is not None
    
    def unwrap(self) -> T:
        """Get value or raise exception if error."""
        if self.is_err:
            raise ValueError(f"Unwrap called on error: {self.error}")
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Get value or default if error."""
        return self.value if self.is_ok else default


class ProcessState(StrEnum):
    """Process state enumeration."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RESTARTING = "restarting"


class RestartPolicy(StrEnum):
    """Restart policy for failed processes."""
    NEVER = "never"
    ON_FAILURE = "on_failure"
    ALWAYS = "always"


@dataclass
class ProcessMetrics:
    """Process performance and health metrics."""
    process_id: int
    session_id: str
    state: ProcessState
    uptime_seconds: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    num_requests: int = 0
    num_errors: int = 0
    restart_count: int = 0
    last_request: Optional[datetime] = None
    last_error: Optional[str] = None
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        total = self.num_requests + self.num_errors
        return (self.num_errors / total * 100) if total > 0 else 0.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if process is considered healthy."""
        return (
            self.state == ProcessState.RUNNING and
            self.error_rate < 10.0 and  # Less than 10% error rate
            self.memory_mb < 1024  # Less than 1GB memory usage
        )


@dataclass
class ProcessConfig:
    """Configuration for MCP process management."""
    max_restarts: int = 3
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE
    restart_delay_seconds: float = 2.0
    health_check_interval: float = 30.0
    idle_timeout_seconds: float = 300.0
    memory_limit_mb: int = 512
    cpu_limit_percent: float = 80.0


class MCPProcess:
    """Manages a single MCP server subprocess with enhanced error handling and monitoring."""
    
    def __init__(
        self,
        session_id: str,
        bridge_config: BridgeConfig,
        process_config: Optional[ProcessConfig] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self.session_id = session_id
        self.bridge_config = bridge_config
        self.process_config = process_config or ProcessConfig()
        self.env = env or {}
        
        # Process management
        self.process: Optional[asyncio.subprocess.Process] = None
        self.reader_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.state = ProcessState.STOPPED
        
        # Communication
        self.pending_requests: Dict[str, PendingRequest] = {}
        self.capabilities: Dict[str, Any] = {}
        
        # Metrics and timing
        self.started_at: Optional[datetime] = None
        self.last_activity: Optional[datetime] = None
        self.metrics = ProcessMetrics(
            process_id=-1,
            session_id=session_id,
            state=ProcessState.STOPPED
        )
        
        # Synchronization and flags
        self._lock = asyncio.Lock()
        self._initialized = False
        self._shutdown_requested = False
    
    async def start(self) -> Result[bool, str]:
        """Start the MCP server subprocess with comprehensive error handling."""
        async with self._lock:
            if self.state in [ProcessState.RUNNING, ProcessState.STARTING]:
                logger.warning("Process already starting or running", session_id=self.session_id)
                return Result.ok(True)
            
            self.state = ProcessState.STARTING
            
            try:
                # Prepare environment
                process_env = self._prepare_environment()
                
                # Build command
                cmd = self._build_command()
                
                logger.info(
                    "Starting MCP process",
                    session_id=self.session_id,
                    cmd=" ".join(cmd),
                    env_vars=len(process_env)
                )
                
                # Start the subprocess with resource limits
                self.process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=process_env,
                    limit=1024 * 1024,  # 1MB buffer
                )
                
                # Update state and timing
                self.state = ProcessState.RUNNING
                self.started_at = datetime.now()
                self.last_activity = datetime.now()
                self.metrics.process_id = self.process.pid
                self.metrics.state = ProcessState.RUNNING
                
                # Start background tasks
                await self._start_background_tasks()
                
                # Initialize the MCP connection
                init_result = await self._initialize_mcp()
                if init_result.is_err:
                    await self.stop()
                    return Result.err(f"MCP initialization failed: {init_result.error}")
                
                logger.info(
                    "MCP process started successfully",
                    session_id=self.session_id,
                    pid=self.process.pid,
                )
                
                return Result.ok(True)
                
            except Exception as e:
                error_msg = f"Failed to start MCP process: {str(e)}"
                logger.error(error_msg, session_id=self.session_id, exc_info=True)
                self.state = ProcessState.ERROR
                await self._cleanup()
                return Result.err(error_msg)
    
    def _prepare_environment(self) -> Dict[str, str]:
        """Prepare environment variables for the subprocess."""
        process_env = os.environ.copy()
        process_env.update(getattr(self.bridge_config, 'mcp_server_env', {}))
        process_env.update(self.env)
        process_env["MCP_STDIO_MODE"] = "true"
        return process_env
    
    def _build_command(self) -> List[str]:
        """Build the command to execute."""
        cmd = [sys.executable, "-m", getattr(self.bridge_config, 'mcp_server_path', 'mcp_server')]
        cmd.extend(getattr(self.bridge_config, 'mcp_server_args', []))
        return cmd
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        self.reader_task = asyncio.create_task(self._read_output())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self, force: bool = False) -> None:
        """Stop the MCP server subprocess gracefully or forcefully."""
        async with self._lock:
            if self.state in [ProcessState.STOPPED, ProcessState.STOPPING]:
                return
            
            self.state = ProcessState.STOPPING
            self._shutdown_requested = True
            
            logger.info("Stopping MCP process", session_id=self.session_id, force=force)
            
            # Cancel background tasks
            await self._cancel_tasks()
            
            # Terminate process
            if self.process:
                try:
                    if not force and self._initialized:
                        # Attempt graceful shutdown
                        await self._graceful_shutdown()
                    
                    # Force termination if needed
                    await self._force_termination(force)
                    
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
                        exc_info=True
                    )
                
                await self._cleanup()
    
    async def _cancel_tasks(self) -> None:
        """Cancel all background tasks."""
        for task in [self.reader_task, self.health_check_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _graceful_shutdown(self, timeout: float = 2.0) -> None:
        """Attempt graceful shutdown with timeout."""
        try:
            await self._send_notification("shutdown", {})
            await asyncio.sleep(0.5)  # Give process time to handle shutdown
        except Exception as e:
            logger.debug("Graceful shutdown notification failed", error=str(e))
    
    async def _force_termination(self, immediate: bool = False) -> None:
        """Force process termination."""
        if not self.process:
            return
        
        timeout = 0.1 if immediate else 5.0
        
        try:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Process didn't terminate gracefully, killing", session_id=self.session_id)
            self.process.kill()
            await self.process.wait()
    
    async def _cleanup(self) -> None:
        """Clean up process resources."""
        self.process = None
        self.state = ProcessState.STOPPED
        self._initialized = False
        self.pending_requests.clear()
        self.metrics.state = ProcessState.STOPPED
    
    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Result[Dict[str, Any], str]:
        """Send a request to the MCP server and await response using Result pattern."""
        if self.state != ProcessState.RUNNING or not self.process:
            return Result.err("MCP process not running")
        
        request = MCPRequest(method=method, params=params)
        request_id = str(request.id)
        
        # Create pending request
        future = asyncio.Future()
        request_timeout = timeout or getattr(self.bridge_config, 'request_timeout', 30.0)
        pending = PendingRequest(
            request_id=request_id,
            method=method,
            params=params,
            timeout=request_timeout,
            callback=future,
        )
        
        self.pending_requests[request_id] = pending
        
        try:
            # Send request
            write_result = await self._write_message(request.dict())
            if write_result.is_err:
                self.pending_requests.pop(request_id, None)
                return Result.err(f"Failed to send request: {write_result.error}")
            
            # Update metrics
            self.last_activity = datetime.now()
            self.metrics.num_requests += 1
            self.metrics.last_request = self.last_activity
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(future, timeout=request_timeout)
                return Result.ok(response)
            except asyncio.TimeoutError:
                error_msg = f"Request '{method}' timed out after {request_timeout}s"
                self._handle_request_error(request_id, error_msg)
                return Result.err(error_msg)
            
        except Exception as e:
            error_msg = f"Request '{method}' failed: {str(e)}"
            self._handle_request_error(request_id, error_msg)
            return Result.err(error_msg)
    
    def _handle_request_error(self, request_id: str, error_msg: str) -> None:
        """Handle request error and update metrics."""
        self.pending_requests.pop(request_id, None)
        self.metrics.num_errors += 1
        self.metrics.last_error = error_msg
        logger.warning(
            "Request failed",
            session_id=self.session_id,
            request_id=request_id,
            error=error_msg
        )
    
    async def send_notification(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Result[None, str]:
        """Send a notification to the MCP server (no response expected)."""
        if self.state != ProcessState.RUNNING or not self.process:
            return Result.err("MCP process not running")
        
        try:
            notification = MCPNotification(method=method, params=params)
            
            write_result = await self._write_message(notification.dict())
            if write_result.is_err:
                return Result.err(f"Failed to send notification: {write_result.error}")
            
            # Update activity
            self.last_activity = datetime.now()
            
            return Result.ok(None)
            
        except Exception as e:
            error_msg = f"Notification '{method}' failed: {str(e)}"
            logger.error(error_msg, session_id=self.session_id)
            return Result.err(error_msg)
    
    async def _initialize_mcp(self) -> Result[Dict[str, Any], str]:
        """Initialize the MCP connection and discover capabilities."""
        try:
            # Send initialize request
            init_params = {
                "protocolVersion": "1.0",
                "clientInfo": {
                    "name": "mcp-bridge",
                    "version": "0.1.0",
                },
            }
            
            response_result = await self.send_request(
                "initialize",
                params=init_params,
                timeout=10.0,
            )
            
            if response_result.is_err:
                return Result.err(f"Initialize request failed: {response_result.error}")
            
            response = response_result.unwrap()
            
            # Store capabilities
            self.capabilities = response.get("capabilities", {})
            
            # Send initialized notification
            notify_result = await self.send_notification("initialized", {})
            if notify_result.is_err:
                logger.warning("Initialized notification failed", error=notify_result.error)
            
            self._initialized = True
            
            logger.info(
                "MCP connection initialized successfully",
                session_id=self.session_id,
                capabilities=list(self.capabilities.keys()),
                protocol_version="1.0"
            )
            
            return Result.ok(self.capabilities)
            
        except Exception as e:
            error_msg = f"Failed to initialize MCP connection: {str(e)}"
            logger.error(error_msg, session_id=self.session_id, exc_info=True)
            return Result.err(error_msg)
    
    async def _write_message(self, message: Dict[str, Any]) -> Result[None, str]:
        """Write a message to the subprocess stdin with error handling."""
        if not self.process or not self.process.stdin:
            return Result.err("Process stdin not available")
        
        try:
            # Serialize and write message
            data = json.dumps(message, ensure_ascii=False) + "\n"
            encoded_data = data.encode('utf-8')
            
            self.process.stdin.write(encoded_data)
            await self.process.stdin.drain()
            
            # Log if configured (checking bridge_config for log_requests)
            if getattr(self.bridge_config, 'log_requests', False):
                logger.debug(
                    "Sent message to MCP",
                    session_id=self.session_id,
                    method=message.get('method', 'unknown'),
                    message_id=message.get('id'),
                )
            
            return Result.ok(None)
                
        except Exception as e:
            error_msg = f"Failed to write message to MCP: {str(e)}"
            logger.error(
                error_msg,
                session_id=self.session_id,
                message_type=message.get('method', 'unknown'),
                exc_info=True
            )
            return Result.err(error_msg)
    
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
        """Enhanced health check with process monitoring and automatic recovery."""
        try:
            while not self._shutdown_requested and self.state in [ProcessState.RUNNING, ProcessState.STARTING]:
                await asyncio.sleep(self.process_config.health_check_interval)
                
                if self.state == ProcessState.STOPPED:
                    break
                
                # Update metrics
                metrics = self.get_metrics()
                
                # Check if process is still alive
                if self.process and self.process.returncode is not None:
                    await self._handle_process_death(metrics)
                    break
                
                # Check resource limits
                await self._check_resource_limits(metrics)
                
                # Check for idle timeout
                await self._check_idle_timeout(metrics)
                
        except asyncio.CancelledError:
            logger.debug("Health check loop cancelled", session_id=self.session_id)
        except Exception as e:
            logger.error(
                "Error in health check loop",
                session_id=self.session_id,
                error=str(e),
                exc_info=True
            )
            self.state = ProcessState.ERROR
    
    async def _handle_process_death(self, metrics: ProcessMetrics) -> None:
        """Handle unexpected process termination."""
        logger.error(
            "MCP process died unexpectedly",
            session_id=self.session_id,
            returncode=self.process.returncode,
            restart_count=metrics.restart_count,
        )
        
        # Attempt automatic restart based on policy
        if self._should_restart(metrics):
            self.state = ProcessState.RESTARTING
            metrics.restart_count += 1
            
            logger.info(
                "Attempting to restart MCP process",
                session_id=self.session_id,
                restart_attempt=metrics.restart_count,
            )
            
            await self._cleanup()
            await asyncio.sleep(self.process_config.restart_delay_seconds)
            
            restart_result = await self.start()
            if restart_result.is_ok:
                logger.info(
                    "MCP process restarted successfully",
                    session_id=self.session_id,
                    restart_count=metrics.restart_count,
                )
            else:
                logger.error(
                    "Failed to restart MCP process",
                    session_id=self.session_id,
                    error=restart_result.error,
                )
                self.state = ProcessState.ERROR
        else:
            self.state = ProcessState.ERROR
    
    def _should_restart(self, metrics: ProcessMetrics) -> bool:
        """Determine if process should be restarted based on policy."""
        if self.process_config.restart_policy == RestartPolicy.NEVER:
            return False
        elif self.process_config.restart_policy == RestartPolicy.ALWAYS:
            return metrics.restart_count < self.process_config.max_restarts
        elif self.process_config.restart_policy == RestartPolicy.ON_FAILURE:
            # Only restart if process exited with non-zero code
            return (
                metrics.restart_count < self.process_config.max_restarts and
                self.process and
                self.process.returncode != 0
            )
        return False
    
    async def _check_resource_limits(self, metrics: ProcessMetrics) -> None:
        """Check if process is exceeding resource limits."""
        if metrics.memory_mb > self.process_config.memory_limit_mb:
            logger.warning(
                "Process exceeding memory limit",
                session_id=self.session_id,
                memory_mb=metrics.memory_mb,
                limit_mb=self.process_config.memory_limit_mb
            )
            # Consider restarting process or sending warning
        
        if metrics.cpu_percent > self.process_config.cpu_limit_percent:
            logger.warning(
                "Process exceeding CPU limit",
                session_id=self.session_id,
                cpu_percent=metrics.cpu_percent,
                limit_percent=self.process_config.cpu_limit_percent
            )
    
    async def _check_idle_timeout(self, metrics: ProcessMetrics) -> None:
        """Check if process has been idle too long."""
        if self.last_activity:
            idle_time = (datetime.now() - self.last_activity).total_seconds()
            if idle_time > self.process_config.idle_timeout_seconds:
                logger.info(
                    "MCP process idle timeout",
                    session_id=self.session_id,
                    idle_time=idle_time,
                )
                await self.stop()
    
    # This method is now replaced by _cleanup() for better organization
    async def _cleanup_process(self) -> None:
        """Cleanup process resources (backward compatibility)."""
        await self._cleanup()
    
    def get_metrics(self) -> ProcessMetrics:
        """Get comprehensive process metrics and health information."""
        # Update basic metrics
        if self.started_at:
            self.metrics.uptime_seconds = (datetime.now() - self.started_at).total_seconds()
        
        # Update system metrics if process is running
        if self.process and self.process.pid and self.state == ProcessState.RUNNING:
            try:
                proc = psutil.Process(self.process.pid)
                self.metrics.cpu_percent = proc.cpu_percent(interval=0.1)
                self.metrics.memory_mb = proc.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                logger.debug("Could not retrieve process metrics", session_id=self.session_id)
                pass
        
        return self.metrics
    
    # Backward compatibility
    def get_stats(self) -> ProcessStats:
        """Get process statistics (backward compatibility)."""
        metrics = self.get_metrics()
        return ProcessStats(
            process_id=metrics.process_id,
            session_id=metrics.session_id,
            num_requests=metrics.num_requests,
            num_errors=metrics.num_errors,
            last_request=metrics.last_request,
            uptime_seconds=metrics.uptime_seconds,
            cpu_percent=metrics.cpu_percent,
            memory_mb=metrics.memory_mb
        )


class MCPProcessManager:
    """Enhanced manager for a pool of MCP server subprocesses with health monitoring."""
    
    def __init__(
        self, 
        bridge_config: BridgeConfig,
        process_config: Optional[ProcessConfig] = None
    ):
        self.bridge_config = bridge_config
        self.process_config = process_config or ProcessConfig()
        self.processes: Dict[str, MCPProcess] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._started = False
        self._shutdown_requested = False
        
        # Manager metrics
        self.manager_started_at: Optional[datetime] = None
        self.total_processes_created: int = 0
        self.total_processes_failed: int = 0
    
    async def start(self) -> Result[None, str]:
        """Start the process manager with enhanced monitoring."""
        if self._started:
            return Result.ok(None)
        
        try:
            self._started = True
            self.manager_started_at = datetime.now()
            self._shutdown_requested = False
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            logger.info(
                "MCP process manager started",
                config={
                    "max_processes": getattr(self.bridge_config, 'max_processes', 10),
                    "cleanup_interval": getattr(self.bridge_config, 'session_cleanup_interval', 60),
                    "restart_policy": self.process_config.restart_policy
                }
            )
            
            return Result.ok(None)
            
        except Exception as e:
            error_msg = f"Failed to start process manager: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.err(error_msg)
    
    async def stop(self, force: bool = False) -> None:
        """Stop the process manager and all processes gracefully or forcefully."""
        self._started = False
        self._shutdown_requested = True
        
        logger.info("Stopping MCP process manager", process_count=len(self.processes), force=force)
        
        # Cancel background tasks
        for task in [self._cleanup_task, self._health_monitor_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop all processes
        async with self._lock:
            if self.processes:
                stop_tasks = [
                    process.stop(force=force) for process in self.processes.values()
                ]
                
                # Wait for all processes to stop
                stop_results = await asyncio.gather(*stop_tasks, return_exceptions=True)
                
                # Log any errors
                for i, result in enumerate(stop_results):
                    if isinstance(result, Exception):
                        logger.error(
                            "Error stopping process",
                            process_index=i,
                            error=str(result)
                        )
                
                self.processes.clear()
        
        uptime = (
            (datetime.now() - self.manager_started_at).total_seconds() 
            if self.manager_started_at else 0
        )
        
        logger.info(
            "MCP process manager stopped",
            uptime_seconds=uptime,
            total_created=self.total_processes_created,
            total_failed=self.total_processes_failed
        )
    
    async def create_process(
        self,
        session_id: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Result[MCPProcess, str]:
        """Create and start a new MCP process for a session."""
        async with self._lock:
            # Check if process already exists
            if session_id in self.processes:
                existing = self.processes[session_id]
                if existing.state == ProcessState.RUNNING:
                    return Result.ok(existing)
                else:
                    # Remove dead/failed process
                    await self._remove_process_internal(session_id)
            
            # Check max processes limit
            max_processes = getattr(self.bridge_config, 'max_processes', 10)
            if len(self.processes) >= max_processes:
                # Try to clean up idle processes
                await self._cleanup_idle_processes()
                
                if len(self.processes) >= max_processes:
                    return Result.err(f"Maximum number of processes ({max_processes}) reached")
            
            # Create new process
            try:
                process = MCPProcess(
                    session_id=session_id, 
                    bridge_config=self.bridge_config,
                    process_config=self.process_config,
                    env=env
                )
                
                # Start the process
                start_result = await process.start()
                if start_result.is_err:
                    self.total_processes_failed += 1
                    return Result.err(f"Failed to start MCP process: {start_result.error}")
                
                # Store the process
                self.processes[session_id] = process
                self.total_processes_created += 1
                
                logger.info(
                    "Created MCP process successfully",
                    session_id=session_id,
                    total_processes=len(self.processes),
                    total_created=self.total_processes_created
                )
                
                return Result.ok(process)
                
            except Exception as e:
                self.total_processes_failed += 1
                error_msg = f"Failed to create MCP process: {str(e)}"
                logger.error(error_msg, session_id=session_id, exc_info=True)
                return Result.err(error_msg)
    
    async def get_process(self, session_id: str) -> Optional[MCPProcess]:
        """Get an existing MCP process for a session."""
        return self.processes.get(session_id)
    
    async def remove_process(self, session_id: str) -> Result[None, str]:
        """Remove and stop a MCP process."""
        async with self._lock:
            return await self._remove_process_internal(session_id)
    
    async def _remove_process_internal(self, session_id: str) -> Result[None, str]:
        """Internal method to remove process without acquiring lock."""
        process = self.processes.pop(session_id, None)
        if not process:
            return Result.err(f"Process {session_id} not found")
        
        try:
            await process.stop()
            
            logger.info(
                "Removed MCP process",
                session_id=session_id,
                total_processes=len(self.processes),
            )
            
            return Result.ok(None)
            
        except Exception as e:
            error_msg = f"Failed to remove process {session_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.err(error_msg)
    
    async def _cleanup_loop(self) -> None:
        """Periodically cleanup idle and dead processes."""
        try:
            cleanup_interval = getattr(self.bridge_config, 'session_cleanup_interval', 60.0)
            while self._started and not self._shutdown_requested:
                await asyncio.sleep(cleanup_interval)
                await self._cleanup_idle_processes()
                
        except asyncio.CancelledError:
            logger.debug("Cleanup loop cancelled")
        except Exception as e:
            logger.error("Error in cleanup loop", error=str(e), exc_info=True)
    
    async def _health_monitor_loop(self) -> None:
        """Monitor overall health of all processes."""
        try:
            monitor_interval = 30.0  # 30 seconds
            while self._started and not self._shutdown_requested:
                await asyncio.sleep(monitor_interval)
                await self._monitor_process_health()
                
        except asyncio.CancelledError:
            logger.debug("Health monitor loop cancelled")
        except Exception as e:
            logger.error("Error in health monitor loop", error=str(e), exc_info=True)
    
    async def _monitor_process_health(self) -> None:
        """Check health of all managed processes."""
        if not self.processes:
            return
        
        unhealthy_processes = []
        total_memory = 0.0
        total_requests = 0
        total_errors = 0
        
        for session_id, process in self.processes.items():
            metrics = process.get_metrics()
            
            total_memory += metrics.memory_mb
            total_requests += metrics.num_requests
            total_errors += metrics.num_errors
            
            if not metrics.is_healthy:
                unhealthy_processes.append(session_id)
        
        if unhealthy_processes:
            logger.warning(
                "Unhealthy processes detected",
                unhealthy_sessions=unhealthy_processes,
                total_processes=len(self.processes)
            )
        
        logger.debug(
            "Process health summary",
            total_processes=len(self.processes),
            total_memory_mb=round(total_memory, 2),
            total_requests=total_requests,
            total_errors=total_errors,
            unhealthy_count=len(unhealthy_processes)
        )
    
    async def _cleanup_idle_processes(self) -> None:
        """Cleanup idle and dead processes with enhanced logic."""
        if not self.processes:
            return
        
        # Don't need lock here since _cleanup_loop already runs in background
        to_remove = []
        idle_timeout = getattr(self.bridge_config, 'process_idle_timeout', 300.0)
        
        for session_id, process in list(self.processes.items()):
            # Check if process is dead or in error state
            if process.state in [ProcessState.STOPPED, ProcessState.ERROR]:
                to_remove.append((session_id, "dead"))
                continue
            
            # Check for idle timeout
            if process.last_activity:
                idle_time = (datetime.now() - process.last_activity).total_seconds()
                if idle_time > idle_timeout:
                    to_remove.append((session_id, "idle"))
                    continue
        
        # Remove identified processes
        if to_remove:
            logger.info(
                "Cleaning up processes",
                count=len(to_remove),
                reasons={reason: sum(1 for _, r in to_remove if r == reason) for reason in ["dead", "idle"]}
            )
            
            async with self._lock:
                for session_id, reason in to_remove:
                    process = self.processes.pop(session_id, None)
                    if process:
                        try:
                            await process.stop()
                        except Exception as e:
                            logger.warning(
                                "Error stopping process during cleanup",
                                session_id=session_id,
                                reason=reason,
                                error=str(e)
                            )
                        
                        logger.debug(
                            "Cleaned up process",
                            session_id=session_id,
                            reason=reason,
                            remaining_processes=len(self.processes),
                        )
    
    def get_stats(self) -> List[ProcessStats]:
        """Get statistics for all processes (backward compatibility)."""
        return [process.get_stats() for process in self.processes.values()]
    
    def get_manager_metrics(self) -> Dict[str, Any]:
        """Get comprehensive manager metrics."""
        uptime = (
            (datetime.now() - self.manager_started_at).total_seconds()
            if self.manager_started_at else 0
        )
        
        # Aggregate process metrics
        total_memory = sum(p.get_metrics().memory_mb for p in self.processes.values())
        total_requests = sum(p.get_metrics().num_requests for p in self.processes.values())
        total_errors = sum(p.get_metrics().num_errors for p in self.processes.values())
        
        healthy_processes = sum(
            1 for p in self.processes.values() 
            if p.get_metrics().is_healthy
        )
        
        return {
            "manager": {
                "uptime_seconds": uptime,
                "total_processes_created": self.total_processes_created,
                "total_processes_failed": self.total_processes_failed,
                "current_processes": len(self.processes),
                "healthy_processes": healthy_processes,
                "max_processes": getattr(self.bridge_config, 'max_processes', 10),
            },
            "aggregates": {
                "total_memory_mb": round(total_memory, 2),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": (total_errors / (total_requests + total_errors) * 100) if (total_requests + total_errors) > 0 else 0.0,
            },
            "processes": {
                session_id: process.get_metrics() 
                for session_id, process in self.processes.items()
            }
        }