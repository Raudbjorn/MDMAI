"""Security utilities for input validation and sanitization."""

import re
from pathlib import Path
from typing import Optional, Set

from config.logging_config import get_logger

logger = get_logger(__name__)


class SecurityError(Exception):
    """Base exception for security-related errors."""

    pass


class PathTraversalError(SecurityError):
    """Exception raised when path traversal is detected."""

    pass


class InputValidationError(SecurityError):
    """Exception raised when input validation fails."""

    pass


class PathValidator:
    """Validates file paths to prevent security vulnerabilities."""

    def __init__(self, allowed_dirs: Optional[Set[Path]] = None):
        """
        Initialize path validator.

        Args:
            allowed_dirs: Set of allowed directories. If None, uses current directory.
        """
        if allowed_dirs is None:
            # Default to data directory and temp directory
            self.allowed_dirs = {
                Path.cwd() / "data",
                Path("/tmp"),
            }
        else:
            self.allowed_dirs = allowed_dirs

        # Ensure all allowed dirs are absolute and resolved
        self.allowed_dirs = {dir_path.resolve() for dir_path in self.allowed_dirs}

    def validate_path(
        self,
        path: str,
        must_exist: bool = False,
        _visited: Optional[Set[Path]] = None,
        _depth: int = 0,
    ) -> Path:
        """
        Validate a file path for security issues.

        Args:
            path: Path to validate
            must_exist: Whether the path must exist
            _visited: Internal use - tracks visited symlinks to detect cycles
            _depth: Internal use - tracks recursion depth for symlink following

        Returns:
            Validated Path object

        Raises:
            PathTraversalError: If path traversal is detected
            FileNotFoundError: If must_exist=True and path doesn't exist
        """
        # Initialize visited set for symlink cycle detection
        if _visited is None:
            _visited = set()

        # Check recursion depth to prevent infinite loops
        if _depth > 10:
            raise PathTraversalError("Too many levels of symbolic links")

        # Convert to Path object
        try:
            path_obj = Path(path)
        except (ValueError, TypeError) as e:
            raise PathTraversalError(f"Invalid path format: {path}") from e

        # Resolve to absolute path (follows symlinks)
        try:
            resolved_path = path_obj.resolve()
        except (OSError, RuntimeError) as e:
            raise PathTraversalError(f"Cannot resolve path: {path}") from e

        # Check for null bytes
        if "\x00" in str(path):
            raise PathTraversalError("Path contains null bytes")

        # Check for suspicious patterns
        suspicious_patterns = [
            r"\.\./\.\./\.\.",  # Multiple parent directory traversals
            r"/\.\.",  # Hidden parent directory
            r"\.\.\/",  # Parent directory at start
        ]

        path_str = str(resolved_path)
        for pattern in suspicious_patterns:
            if re.search(pattern, path_str):
                logger.warning(f"Suspicious path pattern detected: {path}")
                raise PathTraversalError(f"Suspicious path pattern: {path}")

        # Check if path is within allowed directories
        is_allowed = False
        for allowed_dir in self.allowed_dirs:
            try:
                resolved_path.relative_to(allowed_dir)
                is_allowed = True
                break
            except ValueError:
                continue

        if not is_allowed:
            logger.warning(f"Path outside allowed directories: {resolved_path}")
            raise PathTraversalError(
                f"Path must be within allowed directories: {', '.join(str(d) for d in self.allowed_dirs)}"
            )

        # Check if path exists if required
        if must_exist and not resolved_path.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved_path}")

        # Additional security checks
        if resolved_path.is_symlink():
            # Check for symlink cycles
            if resolved_path in _visited:
                raise PathTraversalError("Symlink cycle detected during path validation")
            _visited.add(resolved_path)
            # Check if symlink target is also within allowed dirs
            target = resolved_path.readlink()
            if not target.is_absolute():
                target = resolved_path.parent / target
            self.validate_path(
                str(target),
                must_exist=False,
                _visited=_visited,
                _depth=_depth + 1,
            )

        return resolved_path

    def validate_filename(self, filename: str) -> str:
        """
        Validate a filename for security issues.

        Args:
            filename: Filename to validate

        Returns:
            Validated filename

        Raises:
            InputValidationError: If filename contains invalid characters
        """
        # Check for null bytes
        if "\x00" in filename:
            raise InputValidationError("Filename contains null bytes")

        # Check for path separators
        if "/" in filename or "\\" in filename:
            raise InputValidationError("Filename cannot contain path separators")

        # Check for special filenames
        invalid_names = {
            ".",
            "..",
            "~",
            "CON",
            "PRN",
            "AUX",
            "NUL",  # Windows reserved
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
        }

        base_name = filename.upper().split(".")[0]
        if base_name in invalid_names:
            raise InputValidationError(f"Invalid filename: {filename}")

        # Check for control characters
        if any(ord(char) < 32 for char in filename):
            raise InputValidationError("Filename contains control characters")

        # Limit filename length
        if len(filename) > 255:
            raise InputValidationError("Filename too long (max 255 characters)")

        return filename


class InputSanitizer:
    """Sanitizes user input to prevent injection attacks."""

    @staticmethod
    def sanitize_query(query: str, max_length: int = 1000) -> str:
        """
        Sanitize a search query.

        Args:
            query: Query string to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized query

        Raises:
            InputValidationError: If query is invalid
        """
        if not query or not isinstance(query, str):
            raise InputValidationError("Query must be a non-empty string")

        # Remove null bytes
        query = query.replace("\x00", "")

        # Limit length
        if len(query) > max_length:
            logger.warning(f"Query truncated from {len(query)} to {max_length} characters")
            query = query[:max_length]

        # Remove control characters except newlines and tabs
        sanitized = "".join(char for char in query if ord(char) >= 32 or char in "\n\t")

        # Strip excessive whitespace
        sanitized = " ".join(sanitized.split())

        if not sanitized:
            raise InputValidationError("Query contains only invalid characters")

        return sanitized

    @staticmethod
    def sanitize_metadata(metadata: dict, max_depth: int = 5) -> dict:
        """
        Sanitize metadata dictionary.

        Args:
            metadata: Metadata to sanitize
            max_depth: Maximum nesting depth

        Returns:
            Sanitized metadata

        Raises:
            InputValidationError: If metadata is invalid
        """

        def _sanitize_value(value, depth=0):
            if depth > max_depth:
                raise InputValidationError(f"Metadata nesting too deep (max {max_depth})")

            if value is None:
                return None
            elif isinstance(value, (bool, int, float)):
                return value
            elif isinstance(value, str):
                # Remove null bytes and control characters
                return "".join(char for char in value if ord(char) >= 32 or char in "\n\t")
            elif isinstance(value, dict):
                return {str(k)[:100]: _sanitize_value(v, depth + 1) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [_sanitize_value(item, depth + 1) for item in value[:1000]]
            else:
                # Convert unknown types to string
                return str(value)[:1000]

        if not isinstance(metadata, dict):
            raise InputValidationError("Metadata must be a dictionary")

        return _sanitize_value(metadata)

    @staticmethod
    def validate_system_name(name: str) -> str:
        """
        Validate a game system name.

        Args:
            name: System name to validate

        Returns:
            Validated system name

        Raises:
            InputValidationError: If name is invalid
        """
        if not name or not isinstance(name, str):
            raise InputValidationError("System name must be a non-empty string")

        # Allow alphanumeric, spaces, hyphens, and some special characters
        if not re.match(r"^[\w\s\-&:\'\.]+$", name):
            raise InputValidationError(
                "System name can only contain letters, numbers, spaces, and common punctuation"
            )

        # Limit length
        if len(name) > 100:
            raise InputValidationError("System name too long (max 100 characters)")

        return name.strip()

    @staticmethod
    def validate_pdf_metadata(metadata: dict) -> dict:
        """
        Validate PDF metadata for security issues.

        Args:
            metadata: PDF metadata to validate

        Returns:
            Validated metadata

        Raises:
            InputValidationError: If metadata contains security issues
        """
        # Fields that should be sanitized
        text_fields = ["title", "author", "subject", "creator", "producer", "keywords"]

        validated = {}
        for key, value in metadata.items():
            if key in text_fields and isinstance(value, str):
                # Sanitize text fields
                validated[key] = InputSanitizer.sanitize_query(value, max_length=500)
            elif key == "pages" and isinstance(value, int):
                # Validate page count
                if value < 1 or value > 10000:
                    raise InputValidationError(f"Invalid page count: {value}")
                validated[key] = value
            elif key in ["creation_date", "modification_date"]:
                # Keep dates as-is (should be datetime objects)
                validated[key] = value
            else:
                # Skip unknown fields for security
                logger.debug(f"Skipping unknown PDF metadata field: {key}")

        return validated


# Global instances for convenience
_default_path_validator = None
_input_sanitizer = InputSanitizer()


def get_path_validator(allowed_dirs: Optional[Set[Path]] = None) -> PathValidator:
    """
    Get a path validator instance.

    Args:
        allowed_dirs: Optional set of allowed directories

    Returns:
        PathValidator instance
    """
    global _default_path_validator

    if allowed_dirs is not None:
        return PathValidator(allowed_dirs)

    if _default_path_validator is None:
        _default_path_validator = PathValidator()

    return _default_path_validator


def validate_path(path: str, must_exist: bool = False) -> Path:
    """
    Validate a file path using the default validator.

    Args:
        path: Path to validate
        must_exist: Whether the path must exist

    Returns:
        Validated Path object

    Raises:
        PathTraversalError: If path traversal is detected
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    return get_path_validator().validate_path(path, must_exist)


def sanitize_query(query: str, max_length: int = 1000) -> str:
    """
    Sanitize a search query using the default sanitizer.

    Args:
        query: Query string to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized query

    Raises:
        InputValidationError: If query is invalid
    """
    return _input_sanitizer.sanitize_query(query, max_length)


def sanitize_metadata(metadata: dict) -> dict:
    """
    Sanitize metadata using the default sanitizer.

    Args:
        metadata: Metadata to sanitize

    Returns:
        Sanitized metadata

    Raises:
        InputValidationError: If metadata is invalid
    """
    return _input_sanitizer.sanitize_metadata(metadata)
