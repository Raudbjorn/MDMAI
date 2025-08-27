"""Process sandboxing for secure execution of user code."""

import asyncio
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

from config.logging_config import get_logger

logger = get_logger(__name__)


class SandboxPolicy(Enum):
    """Sandbox security policies."""

    STRICT = "strict"  # No network, limited filesystem
    MODERATE = "moderate"  # Limited network, restricted filesystem
    RELAXED = "relaxed"  # Full network, monitored filesystem
    CUSTOM = "custom"  # User-defined policy


class ResourceLimits(BaseModel):
    """Resource limits for sandboxed processes."""

    # CPU limits
    cpu_time_seconds: int = Field(default=30, description="Max CPU time in seconds")
    cpu_percent: float = Field(default=50.0, description="Max CPU usage percentage")
    
    # Memory limits
    memory_mb: int = Field(default=512, description="Max memory in MB")
    virtual_memory_mb: int = Field(default=1024, description="Max virtual memory in MB")
    
    # Process limits
    max_processes: int = Field(default=10, description="Max number of processes")
    max_threads: int = Field(default=50, description="Max number of threads")
    max_files: int = Field(default=100, description="Max number of open files")
    
    # I/O limits
    max_file_size_mb: int = Field(default=100, description="Max file size in MB")
    disk_quota_mb: int = Field(default=500, description="Max disk usage in MB")
    network_bandwidth_kbps: Optional[int] = Field(default=None, description="Max network bandwidth in kbps")
    
    # Time limits
    wall_time_seconds: int = Field(default=60, description="Max wall clock time in seconds")
    
    @validator("cpu_percent")
    def validate_cpu_percent(cls, v: float) -> float:
        """Validate CPU percentage."""
        if not 0 < v <= 100:
            raise ValueError("CPU percent must be between 0 and 100")
        return v


class FilesystemPolicy(BaseModel):
    """Filesystem access policy."""

    # Allowed paths
    allowed_read_paths: List[Path] = Field(default_factory=list)
    allowed_write_paths: List[Path] = Field(default_factory=list)
    
    # Blocked paths
    blocked_paths: List[Path] = Field(default_factory=lambda: [
        Path("/etc"),
        Path("/sys"),
        Path("/proc"),
        Path("/dev"),
        Path("/boot"),
        Path("/root"),
    ])
    
    # Allowed file extensions
    allowed_extensions: List[str] = Field(default_factory=lambda: [
        ".txt", ".json", ".yaml", ".yml", ".md", ".csv", ".log",
    ])
    
    # Temp directory
    use_temp_dir: bool = True
    cleanup_temp: bool = True


class NetworkPolicy(BaseModel):
    """Network access policy."""

    allow_network: bool = False
    allowed_protocols: List[str] = Field(default_factory=lambda: ["http", "https"])
    allowed_hosts: List[str] = Field(default_factory=list)
    blocked_hosts: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "0.0.0.0"])
    allowed_ports: List[int] = Field(default_factory=lambda: [80, 443])
    dns_servers: List[str] = Field(default_factory=lambda: ["8.8.8.8", "1.1.1.1"])


class SandboxConfig(BaseModel):
    """Complete sandbox configuration."""

    policy: SandboxPolicy = SandboxPolicy.STRICT
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    filesystem_policy: FilesystemPolicy = Field(default_factory=FilesystemPolicy)
    network_policy: NetworkPolicy = Field(default_factory=NetworkPolicy)
    
    # Security options
    use_chroot: bool = False  # Requires root
    use_namespaces: bool = True  # Linux namespaces
    use_seccomp: bool = True  # Syscall filtering
    use_apparmor: bool = False  # AppArmor profiles
    drop_privileges: bool = True
    
    # Execution environment
    environment_vars: Dict[str, str] = Field(default_factory=dict)
    working_dir: Optional[Path] = None
    user: Optional[str] = None
    group: Optional[str] = None
    
    # Monitoring
    enable_monitoring: bool = True
    log_syscalls: bool = False
    capture_output: bool = True
    
    @classmethod
    def from_policy(cls, policy: SandboxPolicy) -> "SandboxConfig":
        """Create config from predefined policy."""
        if policy == SandboxPolicy.STRICT:
            return cls(
                policy=policy,
                resource_limits=ResourceLimits(
                    cpu_time_seconds=10,
                    memory_mb=256,
                    max_processes=5,
                ),
                filesystem_policy=FilesystemPolicy(
                    allowed_read_paths=[],
                    allowed_write_paths=[],
                    use_temp_dir=True,
                ),
                network_policy=NetworkPolicy(
                    allow_network=False,
                ),
            )
        elif policy == SandboxPolicy.MODERATE:
            return cls(
                policy=policy,
                resource_limits=ResourceLimits(
                    cpu_time_seconds=30,
                    memory_mb=512,
                    max_processes=10,
                ),
                filesystem_policy=FilesystemPolicy(
                    use_temp_dir=True,
                ),
                network_policy=NetworkPolicy(
                    allow_network=True,
                    allowed_hosts=["api.example.com"],
                ),
            )
        elif policy == SandboxPolicy.RELAXED:
            return cls(
                policy=policy,
                resource_limits=ResourceLimits(
                    cpu_time_seconds=60,
                    memory_mb=1024,
                    max_processes=20,
                ),
                filesystem_policy=FilesystemPolicy(
                    use_temp_dir=False,
                ),
                network_policy=NetworkPolicy(
                    allow_network=True,
                ),
            )
        else:
            return cls(policy=policy)


class SandboxResult(BaseModel):
    """Result of sandboxed execution."""

    success: bool
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    
    # Resource usage
    cpu_time: float = 0.0
    memory_peak_mb: float = 0.0
    wall_time: float = 0.0
    
    # Security events
    violations: List[str] = Field(default_factory=list)
    blocked_syscalls: List[str] = Field(default_factory=list)
    
    # Files created/modified
    created_files: List[str] = Field(default_factory=list)
    modified_files: List[str] = Field(default_factory=list)
    
    # Network activity
    network_connections: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Errors
    error: Optional[str] = None
    timeout: bool = False


class ProcessSandbox:
    """Sandbox for secure process execution."""

    def __init__(self, config: Optional[SandboxConfig] = None):
        """Initialize process sandbox."""
        self.config = config or SandboxConfig.from_policy(SandboxPolicy.STRICT)
        self._temp_dir: Optional[Path] = None

    async def execute(
        self,
        command: List[str],
        stdin: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """Execute command in sandbox."""
        result = SandboxResult(success=False)
        
        try:
            # Set up sandbox environment
            with self._setup_sandbox() as sandbox_dir:
                # Build sandboxed command
                sandboxed_cmd = self._build_sandboxed_command(command, sandbox_dir)
                
                # Set resource limits
                preexec_fn = self._create_preexec_fn()
                
                # Execute with timeout
                timeout = timeout or self.config.resource_limits.wall_time_seconds
                
                process = await asyncio.create_subprocess_exec(
                    *sandboxed_cmd,
                    stdin=asyncio.subprocess.PIPE if stdin else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox_dir,
                    env=self._get_sandbox_env(),
                    preexec_fn=preexec_fn if sys.platform != "win32" else None,
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(stdin.encode() if stdin else None),
                        timeout=timeout,
                    )
                    
                    result.stdout = stdout.decode("utf-8", errors="replace")
                    result.stderr = stderr.decode("utf-8", errors="replace")
                    result.exit_code = process.returncode
                    result.success = process.returncode == 0
                    
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    result.timeout = True
                    result.error = f"Process exceeded timeout of {timeout} seconds"
                
                # Collect metrics
                result.wall_time = timeout if result.timeout else 0
                
                # Check for violations
                if sandbox_dir:
                    result.created_files = self._get_created_files(sandbox_dir)
                
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            result.error = str(e)
        
        return result

    async def execute_python(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """Execute Python code in sandbox."""
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = f.name
        
        try:
            # Execute Python script
            result = await self.execute(
                [sys.executable, "-u", script_path],
                timeout=timeout,
            )
        finally:
            # Clean up script file
            try:
                os.unlink(script_path)
            except Exception:
                pass
        
        return result

    def _setup_sandbox(self) -> Any:
        """Set up sandbox environment."""
        if self.config.filesystem_policy.use_temp_dir:
            return self._create_temp_sandbox()
        else:
            return contextmanager(lambda: (yield self.config.working_dir))()

    @contextmanager
    def _create_temp_sandbox(self) -> Any:
        """Create temporary sandbox directory."""
        temp_dir = tempfile.mkdtemp(prefix="sandbox_")
        self._temp_dir = Path(temp_dir)
        
        try:
            # Set permissions
            os.chmod(temp_dir, 0o700)
            
            # Copy allowed files
            for path in self.config.filesystem_policy.allowed_read_paths:
                if path.is_file():
                    dest = self._temp_dir / path.name
                    dest.write_bytes(path.read_bytes())
            
            yield self._temp_dir
            
        finally:
            # Cleanup
            if self.config.filesystem_policy.cleanup_temp:
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup sandbox: {e}")

    def _build_sandboxed_command(self, command: List[str], sandbox_dir: Path) -> List[str]:
        """Build sandboxed command with security wrappers."""
        sandboxed = []
        
        # Use firejail if available (Linux)
        if sys.platform == "linux" and shutil.which("firejail"):
            sandboxed.extend([
                "firejail",
                "--quiet",
                f"--private={sandbox_dir}",
                "--net=none" if not self.config.network_policy.allow_network else "",
                f"--rlimit-cpu={self.config.resource_limits.cpu_time_seconds}",
                f"--rlimit-as={self.config.resource_limits.memory_mb}m",
                f"--rlimit-nproc={self.config.resource_limits.max_processes}",
                f"--rlimit-nofile={self.config.resource_limits.max_files}",
                f"--timeout={self.config.resource_limits.wall_time_seconds}",
            ])
            
            # Add blocked paths
            for path in self.config.filesystem_policy.blocked_paths:
                sandboxed.append(f"--blacklist={path}")
        
        # Use Docker if available
        elif shutil.which("docker"):
            container_name = f"sandbox_{os.getpid()}"
            sandboxed.extend([
                "docker", "run",
                "--rm",
                f"--name={container_name}",
                f"--memory={self.config.resource_limits.memory_mb}m",
                f"--cpus={self.config.resource_limits.cpu_percent/100}",
                f"--pids-limit={self.config.resource_limits.max_processes}",
                "--network=none" if not self.config.network_policy.allow_network else "",
                f"--workdir=/sandbox",
                f"-v", f"{sandbox_dir}:/sandbox",
                "python:3.11-slim",
            ])
        
        sandboxed.extend(command)
        return sandboxed

    def _create_preexec_fn(self) -> Optional[Any]:
        """Create preexec function for resource limits."""
        if sys.platform == "win32":
            return None
        
        def set_limits() -> None:
            """Set resource limits for child process."""
            limits = self.config.resource_limits
            
            # CPU time limit
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (limits.cpu_time_seconds, limits.cpu_time_seconds),
            )
            
            # Memory limit
            memory_bytes = limits.memory_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_AS,
                (memory_bytes, memory_bytes),
            )
            
            # Process limit
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (limits.max_processes, limits.max_processes),
            )
            
            # File descriptor limit
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (limits.max_files, limits.max_files),
            )
            
            # File size limit
            file_size_bytes = limits.max_file_size_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (file_size_bytes, file_size_bytes),
            )
            
            # Drop privileges if configured
            if self.config.drop_privileges and os.getuid() == 0:
                import pwd
                import grp
                
                # Get nobody user
                nobody = pwd.getpwnam("nobody")
                os.setgroups([])
                os.setgid(nobody.pw_gid)
                os.setuid(nobody.pw_uid)
        
        return set_limits

    def _get_sandbox_env(self) -> Dict[str, str]:
        """Get sandboxed environment variables."""
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self._temp_dir or "/tmp"),
            "TMPDIR": str(self._temp_dir or "/tmp"),
            "USER": "sandbox",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        
        # Add custom environment variables
        env.update(self.config.environment_vars)
        
        # Remove sensitive variables
        sensitive_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "DATABASE_URL",
            "API_KEY",
            "SECRET_KEY",
        ]
        
        for var in sensitive_vars:
            env.pop(var, None)
        
        return env

    def _get_created_files(self, sandbox_dir: Path) -> List[str]:
        """Get list of files created in sandbox."""
        created = []
        
        try:
            for path in sandbox_dir.rglob("*"):
                if path.is_file():
                    created.append(str(path.relative_to(sandbox_dir)))
        except Exception as e:
            logger.warning(f"Failed to list created files: {e}")
        
        return created

    async def validate_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Validate code for security issues."""
        issues = []
        
        if language == "python":
            # Check for dangerous imports
            dangerous_imports = [
                "os", "sys", "subprocess", "socket", "requests",
                "__import__", "eval", "exec", "compile", "open",
            ]
            
            for imp in dangerous_imports:
                if imp in code:
                    issues.append(f"Potentially dangerous: {imp}")
            
            # Check for file operations
            file_ops = ["open(", "file(", "read(", "write("]
            for op in file_ops:
                if op in code:
                    issues.append(f"File operation detected: {op}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }


