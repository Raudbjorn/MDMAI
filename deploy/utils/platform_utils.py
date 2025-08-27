"""Cross-platform utility functions for deployment operations."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def get_platform() -> str:
    """
    Get the current platform identifier.
    
    Returns:
        str: Platform identifier ('windows', 'linux', 'darwin', or 'unknown')
    """
    system = platform.system().lower()
    if system == 'windows':
        return 'windows'
    elif system == 'linux':
        return 'linux'
    elif system == 'darwin':
        return 'darwin'
    else:
        return 'unknown'


def is_windows() -> bool:
    """Check if running on Windows."""
    return get_platform() == 'windows'


def is_linux() -> bool:
    """Check if running on Linux."""
    return get_platform() == 'linux'


def is_macos() -> bool:
    """Check if running on macOS."""
    return get_platform() == 'darwin'


def is_unix_like() -> bool:
    """Check if running on Unix-like system (Linux or macOS)."""
    return get_platform() in ('linux', 'darwin')


def set_file_permissions(path: Path, mode: int = 0o644) -> bool:
    """
    Set file permissions in a cross-platform way.
    
    Args:
        path: Path to file or directory
        mode: Permission mode (Unix-style octal, e.g., 0o644)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if is_unix_like():
            os.chmod(path, mode)
        # Windows doesn't support Unix-style permissions
        # but we can ensure the file exists and is accessible
        elif is_windows() and path.exists():
            # On Windows, just verify we can access the file
            path.touch(exist_ok=True)
        return True
    except Exception:
        return False


def set_owner(path: Path, user: Optional[str] = None, group: Optional[str] = None) -> bool:
    """
    Set file ownership in a cross-platform way.
    
    Args:
        path: Path to file or directory
        user: Username (Unix-like systems only)
        group: Group name (Unix-like systems only)
        
    Returns:
        bool: True if successful or not applicable, False on error
    """
    if is_windows():
        # Windows handles ownership differently, skip
        return True
    
    if not is_unix_like():
        return False
    
    try:
        import pwd
        import grp
        
        uid = -1
        gid = -1
        
        if user:
            try:
                uid = pwd.getpwnam(user).pw_uid
            except KeyError:
                # User doesn't exist
                return False
        
        if group:
            try:
                gid = grp.getgrnam(group).gr_gid
            except KeyError:
                # Group doesn't exist
                return False
        
        os.chown(path, uid, gid)
        return True
    except (ImportError, Exception):
        # pwd/grp modules not available or other error
        return False


def run_command(
    command: List[str],
    capture_output: bool = True,
    check: bool = True,
    shell: bool = False
) -> Tuple[bool, str, str]:
    """
    Run a command in a cross-platform way.
    
    Args:
        command: Command and arguments as list
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise exception on non-zero exit
        shell: Whether to run through shell
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        # Convert Path objects to strings
        command = [str(c) for c in command]
        
        if shell and is_windows():
            # On Windows, join command for shell execution
            result = subprocess.run(
                ' '.join(command),
                capture_output=capture_output,
                text=True,
                check=check,
                shell=True
            )
        else:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                check=check,
                shell=shell
            )
        
        stdout = result.stdout if capture_output else ""
        stderr = result.stderr if capture_output else ""
        return True, stdout, stderr
    except subprocess.CalledProcessError as e:
        stdout = e.stdout if hasattr(e, 'stdout') and e.stdout else ""
        stderr = e.stderr if hasattr(e, 'stderr') and e.stderr else ""
        return False, stdout, stderr
    except Exception as e:
        return False, "", str(e)


def manage_service(service_name: str, action: str) -> bool:
    """
    Manage system services in a cross-platform way.
    
    Args:
        service_name: Name of the service
        action: Action to perform ('start', 'stop', 'restart', 'status')
        
    Returns:
        bool: True if successful, False otherwise
    """
    platform_type = get_platform()
    
    if platform_type == 'linux':
        # Try systemctl first (systemd)
        success, _, _ = run_command(['systemctl', action, service_name], check=False)
        if success:
            return True
        
        # Fall back to service command (SysV init)
        success, _, _ = run_command(['service', service_name, action], check=False)
        return success
    
    elif platform_type == 'darwin':
        # macOS uses launchctl
        if action == 'start':
            launchctl_action = 'load'
        elif action == 'stop':
            launchctl_action = 'unload'
        else:
            # For restart, stop then start
            if action == 'restart':
                manage_service(service_name, 'stop')
                return manage_service(service_name, 'start')
            # Status is more complex on macOS
            elif action == 'status':
                success, stdout, _ = run_command(['launchctl', 'list'], check=False)
                return service_name in stdout
            return False
        
        # Assume service plist is in standard location
        plist_path = f"/Library/LaunchDaemons/{service_name}.plist"
        success, _, _ = run_command(['launchctl', launchctl_action, plist_path], check=False)
        return success
    
    elif platform_type == 'windows':
        # Windows uses sc or net commands
        windows_actions = {
            'start': 'start',
            'stop': 'stop',
            'restart': 'restart',
            'status': 'query'
        }
        
        if action not in windows_actions:
            return False
        
        # Try sc command first
        success, _, _ = run_command(
            ['sc', windows_actions[action], service_name],
            check=False
        )
        if success:
            return True
        
        # Fall back to net command for start/stop
        if action in ('start', 'stop'):
            success, _, _ = run_command(
                ['net', action, service_name],
                check=False
            )
            return success
        
        return False
    
    return False


def find_python() -> str:
    """
    Find the Python executable path in a cross-platform way.
    
    Returns:
        str: Path to Python executable
    """
    # First, try the current Python executable
    if sys.executable:
        return sys.executable
    
    # Try common Python commands
    for cmd in ['python3', 'python', 'py']:
        if shutil.which(cmd):
            return cmd
    
    # Default fallback
    return 'python'


def get_user_home() -> Path:
    """
    Get user home directory in a cross-platform way.
    
    Returns:
        Path: User home directory
    """
    return Path.home()


def get_config_dir(app_name: str = "mdmai") -> Path:
    """
    Get application configuration directory in a cross-platform way.
    
    Args:
        app_name: Application name for config directory
        
    Returns:
        Path: Configuration directory path
    """
    platform_type = get_platform()
    
    if platform_type == 'windows':
        # Windows: %APPDATA%/app_name
        app_data = os.environ.get('APPDATA')
        if app_data:
            return Path(app_data) / app_name
        return get_user_home() / 'AppData' / 'Roaming' / app_name
    
    elif platform_type == 'darwin':
        # macOS: ~/Library/Application Support/app_name
        return get_user_home() / 'Library' / 'Application Support' / app_name
    
    else:
        # Linux and others: ~/.config/app_name
        xdg_config = os.environ.get('XDG_CONFIG_HOME')
        if xdg_config:
            return Path(xdg_config) / app_name
        return get_user_home() / '.config' / app_name


def get_data_dir(app_name: str = "mdmai") -> Path:
    """
    Get application data directory in a cross-platform way.
    
    Args:
        app_name: Application name for data directory
        
    Returns:
        Path: Data directory path
    """
    platform_type = get_platform()
    
    if platform_type == 'windows':
        # Windows: %LOCALAPPDATA%/app_name
        local_app_data = os.environ.get('LOCALAPPDATA')
        if local_app_data:
            return Path(local_app_data) / app_name
        return get_user_home() / 'AppData' / 'Local' / app_name
    
    elif platform_type == 'darwin':
        # macOS: ~/Library/Application Support/app_name
        return get_user_home() / 'Library' / 'Application Support' / app_name
    
    else:
        # Linux and others: ~/.local/share/app_name
        xdg_data = os.environ.get('XDG_DATA_HOME')
        if xdg_data:
            return Path(xdg_data) / app_name
        return get_user_home() / '.local' / 'share' / app_name


def ensure_directory(path: Path, mode: int = 0o755) -> bool:
    """
    Ensure a directory exists with proper permissions.
    
    Args:
        path: Directory path
        mode: Permission mode (Unix-style octal)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return set_file_permissions(path, mode)
    except Exception:
        return False