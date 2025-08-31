"""Advanced input validation and sanitization module."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from config.logging_config import get_logger

logger = get_logger(__name__)


# Validation error message constants
VALIDATION_ERRORS = {
    "PATH_NOT_IN_ALLOWED_DIRS": "Path not within allowed directories",
    "PATH_DOES_NOT_EXIST": "Path does not exist",
    "SQL_INJECTION": "Potential SQL injection detected",
    "XSS_ATTACK": "Potential XSS attack detected",
    "PATH_TRAVERSAL": "Path traversal attempt detected",
    "COMMAND_INJECTION": "Potential command injection detected",
    "INPUT_TOO_LONG": "Input exceeds maximum length",
    "INVALID_CHARACTERS": "Input contains invalid characters",
    "FORBIDDEN_PATTERN": "Input matches forbidden pattern",
    "TYPE_MISMATCH": "Expected type mismatch",
    "MISSING_FIELD": "Missing required field",
}


class ValidationResult:
    """Represents the result of a validation operation."""

    def __init__(
        self,
        is_valid: bool,
        value: Any = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        """Initialize validation result."""
        self.is_valid = is_valid
        self.value = value
        self.errors = errors or []
        self.warnings = warnings or []

    def __bool__(self) -> bool:
        """Return validity status when used as boolean."""
        return self.is_valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "value": self.value,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class SecurityValidationError(Exception):
    """Exception raised when security validation fails."""

    pass


class InjectionAttackError(SecurityValidationError):
    """Exception raised when injection attack detected."""

    pass


class PathTraversalError(SecurityValidationError):
    """Exception raised when path traversal detected."""

    pass


class DataTypeValidationError(SecurityValidationError):
    """Exception raised when data type validation fails."""

    pass


# Pydantic Models for Data Type Validation
class SearchParameters(BaseModel):
    """Validates search operation parameters."""

    query: str = Field(min_length=1, max_length=1000)
    rulebook: Optional[str] = Field(None, max_length=200)
    source_type: Optional[str] = Field(None, pattern="^(rulebook|flavor)$")
    content_type: Optional[str] = Field(
        None, pattern="^(rule|spell|monster|item|npc|location|lore)$"
    )
    max_results: int = Field(ge=1, le=100, default=5)
    use_hybrid: bool = Field(default=True)
    explain_results: bool = Field(default=False)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize query string."""
        # Remove null bytes
        v = v.replace("\x00", "")
        # Remove control characters except newlines and tabs
        v = "".join(char for char in v if ord(char) >= 32 or char in "\n\t")
        # Strip excessive whitespace
        v = " ".join(v.split())
        if not v:
            raise ValueError("Query contains only invalid characters")
        return v


class CampaignParameters(BaseModel):
    """Validates campaign operation parameters."""

    name: str = Field(min_length=1, max_length=200)
    system: str = Field(min_length=1, max_length=100)
    description: str = Field(max_length=2000, default="")
    setting: str = Field(max_length=200, default="")

    @field_validator("name", "system", "setting")
    @classmethod
    def validate_names(cls, v: str) -> str:
        """Validate campaign-related names."""
        if v:
            # Allow alphanumeric, spaces, hyphens, and some special characters
            if not re.match(r"^[\w\s\-&:'\.,]+$", v):
                raise ValueError(
                    "Field can only contain letters, numbers, spaces, and common punctuation"
                )
        return v.strip()


class FilePathParameters(BaseModel):
    """Validates file path parameters."""

    path: str = Field(min_length=1, max_length=4096)
    must_exist: bool = Field(default=False)
    allowed_extensions: Optional[List[str]] = Field(default=None)

    @model_validator(mode="after")
    def validate_path_security(self) -> "FilePathParameters":
        """Perform security validation on file path."""
        # Check for null bytes
        if "\x00" in self.path:
            raise ValueError("Path contains null bytes")

        # Check for suspicious patterns
        suspicious_patterns = [
            r"\.\./\.\./\.\.",  # Multiple parent directory traversals
            r"/\.\.",  # Hidden parent directory
            r"\.\.\\",  # Windows-style parent directory
            r"\.\.\/",  # Parent directory at start
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, self.path):
                raise ValueError(f"Suspicious path pattern detected: {pattern}")

        # Check extension if specified
        if self.allowed_extensions:
            path_obj = Path(self.path)
            if not any(
                path_obj.suffix.lower() == f".{ext.lower().lstrip('.')}"
                for ext in self.allowed_extensions
            ):
                raise ValueError(
                    f"File extension not allowed. Allowed: {', '.join(self.allowed_extensions)}"
                )

        return self


class InputValidator:
    """Advanced input validation with injection prevention."""

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|\#|\/\*|\*\/)",  # SQL comments
        r"(\bOR\b.*=.*)",  # OR 1=1 patterns
        r"(;.*\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b)",  # Chained SQL commands
        r"(xp_cmdshell|sp_executesql)",  # SQL Server stored procedures
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers
        r"<iframe[^>]*>",
        r"<embed[^>]*>",
        r"<object[^>]*>",
        r"eval\s*\(",
        r"document\.(write|cookie|location)",
        r"window\.(location|open)",
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",  # Shell metacharacters
        r"\$\([^)]+\)",  # Command substitution
        r"`[^`]+`",  # Backtick command execution
        r">\s*/dev/",  # Device file redirection
        r"\|\s*nc\b",  # Netcat piping
        r"(curl|wget|nc|netcat|bash|sh|cmd|powershell)\s+",
    ]

    # LDAP injection patterns
    LDAP_INJECTION_PATTERNS = [
        r"\*\|",
        r"\(\|\(",
        r"\)\|\)",
        r"[)(|&*]",  # LDAP metacharacters
    ]

    def __init__(self, strict_mode: bool = True):
        """
        Initialize input validator.

        Args:
            strict_mode: If True, raises exceptions on validation failure
        """
        self.strict_mode = strict_mode
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for performance."""
        self.sql_regex = re.compile(
            "|".join(self.SQL_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL
        )
        self.xss_regex = re.compile("|".join(self.XSS_PATTERNS), re.IGNORECASE | re.DOTALL)
        self.cmd_regex = re.compile(
            "|".join(self.COMMAND_INJECTION_PATTERNS), re.IGNORECASE
        )
        self.ldap_regex = re.compile("|".join(self.LDAP_INJECTION_PATTERNS))

    def validate_input(
        self,
        value: Any,
        input_type: str = "general",
        max_length: Optional[int] = None,
        allowed_chars: Optional[str] = None,
        custom_patterns: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Comprehensive input validation.

        Args:
            value: Input value to validate
            input_type: Type of input (general, query, path, metadata, etc.)
            max_length: Maximum allowed length
            allowed_chars: Regex pattern of allowed characters
            custom_patterns: Additional patterns to check

        Returns:
            ValidationResult with validation status and sanitized value
        """
        errors = []
        warnings = []

        # Type check
        if value is None:
            return ValidationResult(True, None)

        # Convert to string for validation
        str_value = str(value) if not isinstance(value, str) else value

        # Length check
        if max_length and len(str_value) > max_length:
            if self.strict_mode:
                errors.append(f"Input exceeds maximum length of {max_length}")
            else:
                str_value = str_value[:max_length]
                warnings.append(f"Input truncated to {max_length} characters")

        # Check for injection attacks based on input type
        if input_type in ["query", "general", "metadata"]:
            if self.detect_sql_injection(str_value):
                errors.append(VALIDATION_ERRORS["SQL_INJECTION"])

            if self.detect_xss(str_value):
                errors.append(VALIDATION_ERRORS["XSS_ATTACK"])

        if input_type in ["path", "filename"]:
            if self.detect_path_traversal(str_value):
                errors.append(VALIDATION_ERRORS["PATH_TRAVERSAL"])

            if self.detect_command_injection(str_value):
                errors.append(VALIDATION_ERRORS["COMMAND_INJECTION"])

        # Check allowed characters
        if allowed_chars and not re.match(allowed_chars, str_value):
            errors.append(f"Input contains invalid characters (allowed: {allowed_chars})")

        # Check custom patterns
        if custom_patterns:
            for pattern in custom_patterns:
                if re.search(pattern, str_value, re.IGNORECASE):
                    errors.append(f"Input matches forbidden pattern: {pattern}")

        # Sanitize the value
        sanitized = self.sanitize_string(str_value, input_type)

        if errors and self.strict_mode:
            raise SecurityValidationError(f"Validation failed: {'; '.join(errors)}")

        return ValidationResult(
            is_valid=len(errors) == 0, value=sanitized, errors=errors, warnings=warnings
        )

    def detect_sql_injection(self, value: str) -> bool:
        """
        Detect potential SQL injection attempts.

        Args:
            value: String to check

        Returns:
            True if SQL injection pattern detected
        """
        if self.sql_regex.search(value):
            logger.warning(f"SQL injection pattern detected: {value[:100]}")
            return True
        return False

    def detect_xss(self, value: str) -> bool:
        """
        Detect potential XSS attacks.

        Args:
            value: String to check

        Returns:
            True if XSS pattern detected
        """
        if self.xss_regex.search(value):
            logger.warning(f"XSS pattern detected: {value[:100]}")
            return True
        return False

    def detect_command_injection(self, value: str) -> bool:
        """
        Detect potential command injection.

        Args:
            value: String to check

        Returns:
            True if command injection pattern detected
        """
        if self.cmd_regex.search(value):
            logger.warning(f"Command injection pattern detected: {value[:100]}")
            return True
        return False

    def detect_path_traversal(self, value: str) -> bool:
        """
        Detect path traversal attempts.

        Args:
            value: String to check

        Returns:
            True if path traversal detected
        """
        patterns = [
            r"\.\./",  # Unix parent directory
            r"\.\.\\",  # Windows parent directory
            r"\.\.",  # Any parent directory reference
            r"%2e%2e",  # URL encoded parent directory
            r"\.%2e",  # Mixed encoding
            r"%00",  # Null byte
        ]

        for pattern in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Path traversal pattern detected: {value[:100]}")
                return True
        return False

    def detect_ldap_injection(self, value: str) -> bool:
        """
        Detect potential LDAP injection.

        Args:
            value: String to check

        Returns:
            True if LDAP injection pattern detected
        """
        if self.ldap_regex.search(value):
            logger.warning(f"LDAP injection pattern detected: {value[:100]}")
            return True
        return False

    def sanitize_string(self, value: str, context: str = "general") -> str:
        """
        Sanitize string based on context.

        Args:
            value: String to sanitize
            context: Context for sanitization (general, query, path, etc.)

        Returns:
            Sanitized string
        """
        # Remove null bytes
        value = value.replace("\x00", "")

        if context == "query":
            # Remove control characters except newlines and tabs
            value = "".join(char for char in value if ord(char) >= 32 or char in "\n\t")
            # Normalize whitespace
            value = " ".join(value.split())

        elif context == "path":
            # Remove any path traversal sequences
            value = re.sub(r"\.\.+[/\\]?", "", value)
            # Remove null bytes and control characters
            value = "".join(char for char in value if ord(char) >= 32)

        elif context == "filename":
            # Remove path separators
            value = value.replace("/", "").replace("\\", "")
            # Remove special characters that could be problematic
            value = re.sub(r'[<>:"|?*]', "", value)
            # Remove control characters
            value = "".join(char for char in value if ord(char) >= 32)

        elif context == "metadata":
            # HTML entity encode potentially dangerous characters
            replacements = {
                "<": "&lt;",
                ">": "&gt;",
                "&": "&amp;",
                '"': "&quot;",
                "'": "&#x27;",
                "/": "&#x2F;",
            }
            for char, replacement in replacements.items():
                value = value.replace(char, replacement)

        return value.strip()

    def validate_data_type(
        self, value: Any, expected_type: Type, model: Optional[Type[BaseModel]] = None
    ) -> ValidationResult:
        """
        Validate data type and structure.

        Args:
            value: Value to validate
            expected_type: Expected Python type
            model: Optional Pydantic model for validation

        Returns:
            ValidationResult with validation status
        """
        errors = []

        # Basic type check
        if not isinstance(value, expected_type):
            errors.append(
                f"Expected type {expected_type.__name__}, got {type(value).__name__}"
            )
            return ValidationResult(False, None, errors)

        # If Pydantic model provided, use it for validation
        if model:
            try:
                validated_value = model(**value) if isinstance(value, dict) else model(value)
                return ValidationResult(True, validated_value.model_dump())
            except ValidationError as e:
                errors.extend([str(err) for err in e.errors()])
                return ValidationResult(False, None, errors)

        return ValidationResult(True, value)

    def validate_json_structure(
        self, data: Dict[str, Any], schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate JSON structure against schema.

        Args:
            data: JSON data to validate
            schema: Expected schema structure

        Returns:
            ValidationResult with validation status
        """
        errors = []

        def check_structure(current_data: Any, current_schema: Any, path: str = "") -> None:
            if isinstance(current_schema, dict):
                if not isinstance(current_data, dict):
                    errors.append(f"{path}: Expected dict, got {type(current_data).__name__}")
                    return

                for key, value_schema in current_schema.items():
                    if key not in current_data:
                        if not key.endswith("?"):  # Optional field
                            errors.append(f"{path}.{key}: Missing required field")
                    else:
                        check_structure(
                            current_data[key], value_schema, f"{path}.{key}".lstrip(".")
                        )

            elif isinstance(current_schema, list):
                if not isinstance(current_data, list):
                    errors.append(f"{path}: Expected list, got {type(current_data).__name__}")
                elif current_schema:
                    # Check each item against the schema
                    for i, item in enumerate(current_data):
                        check_structure(item, current_schema[0], f"{path}[{i}]")

            elif isinstance(current_schema, type):
                if not isinstance(current_data, current_schema):
                    errors.append(
                        f"{path}: Expected {current_schema.__name__}, "
                        f"got {type(current_data).__name__}"
                    )

        check_structure(data, schema)

        return ValidationResult(is_valid=len(errors) == 0, value=data, errors=errors)

    def validate_file_path(
        self,
        path: str,
        allowed_dirs: Optional[List[Path]] = None,
        must_exist: bool = False,
        allowed_extensions: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Comprehensive file path validation.

        Args:
            path: Path to validate
            allowed_dirs: List of allowed parent directories
            must_exist: Whether file must exist
            allowed_extensions: List of allowed file extensions

        Returns:
            ValidationResult with validation status
        """
        try:
            # Use FilePathParameters for validation
            params = FilePathParameters(
                path=path,
                must_exist=must_exist,
                allowed_extensions=allowed_extensions,
            )

            path_obj = Path(params.path).resolve()

            # Check if path is within allowed directories
            if allowed_dirs:
                is_allowed = False
                for allowed_dir in allowed_dirs:
                    try:
                        path_obj.relative_to(allowed_dir.resolve())
                        is_allowed = True
                        break
                    except ValueError:
                        continue

                if not is_allowed:
                    return ValidationResult(
                        False, None, [VALIDATION_ERRORS["PATH_NOT_IN_ALLOWED_DIRS"]]
                    )

            # Check existence
            if must_exist and not path_obj.exists():
                return ValidationResult(False, None, [f"{VALIDATION_ERRORS['PATH_DOES_NOT_EXIST']}: {path_obj}"])

            return ValidationResult(True, str(path_obj))

        except ValidationError as e:
            errors = [str(err["msg"]) for err in e.errors()]
            return ValidationResult(False, None, errors)
        except Exception as e:
            return ValidationResult(False, None, [str(e)])

    def validate_parameter_range(
        self,
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        parameter_name: str = "parameter",
    ) -> ValidationResult:
        """
        Validate numeric parameter ranges.

        Args:
            value: Numeric value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            parameter_name: Name of parameter for error messages

        Returns:
            ValidationResult with validation status
        """
        errors = []

        if not isinstance(value, (int, float)):
            errors.append(f"{parameter_name} must be numeric")
            return ValidationResult(False, None, errors)

        if min_value is not None and value < min_value:
            errors.append(f"{parameter_name} must be >= {min_value}")

        if max_value is not None and value > max_value:
            errors.append(f"{parameter_name} must be <= {max_value}")

        return ValidationResult(is_valid=len(errors) == 0, value=value, errors=errors)

    def validate_enum(
        self, value: str, allowed_values: List[str], parameter_name: str = "parameter"
    ) -> ValidationResult:
        """
        Validate enum/choice parameters.

        Args:
            value: Value to validate
            allowed_values: List of allowed values
            parameter_name: Name of parameter for error messages

        Returns:
            ValidationResult with validation status
        """
        if value not in allowed_values:
            return ValidationResult(
                False,
                None,
                [f"{parameter_name} must be one of: {', '.join(allowed_values)}"],
            )
        return ValidationResult(True, value)


# Global validator instance
_validator = InputValidator()


def validate_and_sanitize(
    value: Any,
    input_type: str = "general",
    max_length: Optional[int] = None,
    strict: bool = True,
) -> ValidationResult:
    """
    Global validation function for convenience.

    Args:
        value: Value to validate
        input_type: Type of input
        max_length: Maximum length
        strict: Whether to use strict mode

    Returns:
        ValidationResult
    """
    _validator.strict_mode = strict
    return _validator.validate_input(value, input_type, max_length)