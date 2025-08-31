"""Comprehensive context validation and integrity checking system."""

import asyncio
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum, auto
from functools import cached_property, lru_cache
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, Union
from uuid import uuid4

import jsonschema
from typing_extensions import Self

from .models import (
    CollaborativeContext,
    Context,
    ContextState,
    ContextType,
    ConversationContext,
    SessionContext,
)

logger = logging.getLogger(__name__)


class ValidationSeverity(StrEnum):
    """Severity levels for validation issues."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass(frozen=True)
class ValidationIssue:
    """Immutable validation issue representation."""

    message: str
    severity: ValidationSeverity = ValidationSeverity.WARNING
    category: str = "general"
    field_path: Optional[str] = None
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    issue_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "issue_id": self.issue_id,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "field_path": self.field_path,
            "suggested_fix": self.suggested_fix,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationResult:
    """Result of context validation with builder pattern support."""

    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    corrected_data: Optional[Dict[str, Any]] = None
    validation_time: float = 0.0
    checks_performed: List[str] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> Self:
        """Add validation issue with fluent interface."""
        self.issues.append(issue)
        if issue.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}:
            self.is_valid = False
        return self

    @property
    def has_errors(self) -> bool:
        """Check if result has error-level issues."""
        return any(
            issue.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}
            for issue in self.issues
        )

    @cached_property
    def issues_by_severity(self) -> Dict[ValidationSeverity, List[ValidationIssue]]:
        """Group issues by severity level."""
        result = defaultdict(list)
        for issue in self.issues:
            result[issue.severity].append(issue)
        return dict(result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        severity_counts = Counter(issue.severity for issue in self.issues)
        return {
            "is_valid": self.is_valid,
            "validation_time": self.validation_time,
            "checks_performed": self.checks_performed,
            "issues": [issue.to_dict() for issue in self.issues],
            "issue_count_by_severity": dict(severity_counts),
        }


class ContextValidator:
    """Comprehensive context validation system with pluggable validators."""

    # Class-level constants
    DEFAULT_MAX_FIELD_LENGTH = 100_000
    DEFAULT_MAX_METADATA_SIZE = 10_000
    TIMESTAMP_FIELDS = frozenset(
        ["created_at", "last_modified", "last_accessed", "expires_at"]
    )
    CRITICAL_FIELDS = frozenset(["context_id", "context_type", "created_at"])
    VALID_MESSAGE_ROLES = frozenset(["user", "assistant", "system", "tool"])

    def __init__(
        self,
        *,
        enable_schema_validation: bool = True,
        enable_semantic_validation: bool = True,
        enable_integrity_checks: bool = True,
        enable_auto_correction: bool = True,
        max_field_length: int = DEFAULT_MAX_FIELD_LENGTH,
        max_metadata_size: int = DEFAULT_MAX_METADATA_SIZE,
    ):
        self.enable_schema_validation = enable_schema_validation
        self.enable_semantic_validation = enable_semantic_validation
        self.enable_integrity_checks = enable_integrity_checks
        self.enable_auto_correction = enable_auto_correction
        self.max_field_length = max_field_length
        self.max_metadata_size = max_metadata_size

        # Validation rules cache
        self._custom_rules: Dict[str, List[Callable]] = defaultdict(list)

        # Performance tracking using dataclass
        self._stats = ValidationStats()

        logger.info(
            "Context validator initialized",
            extra={
                "schema_validation": enable_schema_validation,
                "semantic_validation": enable_semantic_validation,
                "integrity_checks": enable_integrity_checks,
                "auto_correction": enable_auto_correction,
            },
        )

    @cached_property
    def _schemas(self) -> Dict[str, Dict[str, Any]]:
        """Lazily initialize and cache JSON schemas."""
        base_properties = {
            "context_id": {"type": "string", "format": "uuid"},
            "context_type": {
                "type": "string",
                "enum": [t.value for t in ContextType],
            },
            "title": {"type": "string", "maxLength": 500},
            "description": {"type": "string", "maxLength": 2000},
            "data": {"type": "object"},
            "metadata": {"type": "object"},
            "owner_id": {"type": ["string", "null"]},
            "collaborators": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 100,
            },
            "state": {"type": "string", "enum": [s.value for s in ContextState]},
            "created_at": {"type": "string", "format": "date-time"},
            "last_modified": {"type": "string", "format": "date-time"},
        }

        base_schema = {
            "type": "object",
            "properties": base_properties,
            "required": ["context_id", "context_type"],
            "additionalProperties": True,
        }

        return {
            "base": base_schema,
            "conversation": self._build_conversation_schema(base_schema),
            "session": self._build_session_schema(base_schema),
            "collaborative": self._build_collaborative_schema(base_schema),
        }

    @staticmethod
    def _build_conversation_schema(base: Dict[str, Any]) -> Dict[str, Any]:
        """Build conversation-specific schema."""
        return {
            **base,
            "properties": {
                **base["properties"],
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": ["user", "assistant", "system", "tool"],
                            },
                            "content": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"},
                            "tool_calls": {"type": "array"},
                            "tool_results": {"type": "array"},
                        },
                        "required": ["role", "content"],
                    },
                },
                "participants": {"type": "array", "items": {"type": "string"}},
                "current_turn": {"type": "integer", "minimum": 0},
                "max_turns": {"type": ["integer", "null"], "minimum": 1},
            },
        }

    @staticmethod
    def _build_session_schema(base: Dict[str, Any]) -> Dict[str, Any]:
        """Build session-specific schema."""
        return {
            **base,
            "properties": {
                **base["properties"],
                "session_id": {"type": "string"},
                "user_id": {"type": ["string", "null"]},
                "active_conversations": {"type": "array", "items": {"type": "string"}},
                "total_interactions": {"type": "integer", "minimum": 0},
            },
        }

    @staticmethod
    def _build_collaborative_schema(base: Dict[str, Any]) -> Dict[str, Any]:
        """Build collaborative-specific schema."""
        return {
            **base,
            "properties": {
                **base["properties"],
                "room_id": {"type": "string"},
                "active_participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 50,
                },
                "locked_by": {"type": ["string", "null"]},
                "lock_timeout_seconds": {"type": "integer", "minimum": 1},
            },
        }

    async def validate_context(
        self,
        context: Union[Context, Dict[str, Any]],
        context_type: Optional[Type[Context]] = None,
    ) -> ValidationResult:
        """Comprehensive context validation with performance tracking."""
        start_time = time.perf_counter()
        result = ValidationResult()

        try:
            context_data, context_type = self._prepare_context_data(context, context_type)

            # Run validation pipeline
            validators = self._get_validators()
            for validator_name, validator_func in validators:
                if validator_func:
                    await validator_func(context_data, context_type, result)
                    result.checks_performed.append(validator_name)

            # Apply auto-corrections if enabled
            if self.enable_auto_correction and not result.has_errors:
                result.corrected_data = await self._apply_auto_corrections(
                    context_data, result
                )

            # Run custom validation rules
            await self._run_custom_rules(context_data, context_type, result)

            # Update statistics
            result.validation_time = time.perf_counter() - start_time
            self._stats.record_validation(result.is_valid, result.validation_time)

            logger.debug(
                "Context validation completed",
                extra={
                    "is_valid": result.is_valid,
                    "issue_count": len(result.issues),
                    "execution_time": result.validation_time,
                },
            )

            return result

        except Exception as e:
            result.add_issue(
                ValidationIssue(
                    message=f"Validation process failed: {e!s}",
                    severity=ValidationSeverity.CRITICAL,
                    category="validation_error",
                )
            )
            logger.error(f"Context validation failed: {e}")
            return result

    def _prepare_context_data(
        self,
        context: Union[Context, Dict[str, Any]],
        context_type: Optional[Type[Context]] = None,
    ) -> tuple[Dict[str, Any], Type[Context]]:
        """Prepare context data for validation using pattern matching."""
        match context:
            case Context() as ctx:
                context_data = ctx.dict()
                context_type = type(ctx)
            case dict() as data:
                context_data = data
            case _:
                context_data = dict(context)

        if context_type is None:
            context_type = self._infer_context_type(context_data)

        return context_data, context_type

    @staticmethod
    def _infer_context_type(context_data: Dict[str, Any]) -> Type[Context]:
        """Infer context type from data using pattern matching."""
        match context_data.get("context_type", "base"):
            case "conversation":
                return ConversationContext
            case "session":
                return SessionContext
            case "collaborative":
                return CollaborativeContext
            case _:
                return Context

    def _get_validators(self) -> List[tuple[str, Optional[Callable]]]:
        """Get enabled validators."""
        return [
            (
                "schema_validation",
                self._validate_schema if self.enable_schema_validation else None,
            ),
            (
                "semantic_validation",
                self._validate_semantics if self.enable_semantic_validation else None,
            ),
            (
                "integrity_checks",
                self._validate_integrity if self.enable_integrity_checks else None,
            ),
        ]

    async def _validate_schema(
        self,
        context_data: Dict[str, Any],
        context_type: Type[Context],
        result: ValidationResult,
    ) -> None:
        """Validate context against JSON schema."""
        schema_map = {
            ConversationContext: "conversation",
            SessionContext: "session",
            CollaborativeContext: "collaborative",
        }
        schema_name = schema_map.get(context_type, "base")
        schema = self._schemas[schema_name]

        try:
            jsonschema.validate(context_data, schema)
        except jsonschema.ValidationError as e:
            result.add_issue(
                ValidationIssue(
                    message=f"Schema validation failed: {e.message}",
                    severity=ValidationSeverity.ERROR,
                    category="schema_validation",
                    field_path=".".join(str(p) for p in e.absolute_path)
                    if e.absolute_path
                    else None,
                    suggested_fix=self._suggest_schema_fix(e),
                )
            )
        except Exception as e:
            result.add_issue(
                ValidationIssue(
                    message=f"Schema validation error: {e!s}",
                    severity=ValidationSeverity.WARNING,
                    category="schema_validation",
                )
            )

    async def _validate_semantics(
        self,
        context_data: Dict[str, Any],
        context_type: Type[Context],
        result: ValidationResult,
    ) -> None:
        """Validate semantic consistency and business logic."""
        validators = [
            self._check_required_fields,
            self._check_field_lengths,
            self._check_timestamps,
            self._check_data_types,
        ]

        # Run general validators
        for validator in validators:
            await validator(context_data, result)

        # Run type-specific validators
        type_validators = {
            ConversationContext: self._validate_conversation_semantics,
            SessionContext: self._validate_session_semantics,
            CollaborativeContext: self._validate_collaborative_semantics,
        }

        if specific_validator := type_validators.get(context_type):
            await specific_validator(context_data, result)

    async def _validate_integrity(
        self,
        context_data: Dict[str, Any],
        context_type: Type[Context],
        result: ValidationResult,
    ) -> None:
        """Validate data integrity and referential consistency."""
        integrity_checks = [
            self._check_data_corruption,
            self._check_permissions_integrity,
            self._check_version_integrity,
            self._check_metadata_integrity,
        ]

        for check in integrity_checks:
            await check(context_data, result)

    async def _check_required_fields(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Check for required and critical fields."""
        # Check required fields exist
        for field in ["context_id", "context_type"]:
            if field not in context_data:
                result.add_issue(
                    ValidationIssue(
                        message=f"Required field '{field}' is missing",
                        severity=ValidationSeverity.ERROR,
                        category="data_consistency",
                        field_path=field,
                        suggested_fix=f"Add required field '{field}'",
                    )
                )

        # Check critical fields are not null
        for field in self.CRITICAL_FIELDS:
            if context_data.get(field) is None:
                result.add_issue(
                    ValidationIssue(
                        message=f"Critical field '{field}' has null value",
                        severity=ValidationSeverity.WARNING,
                        category="data_consistency",
                        field_path=field,
                    )
                )

    async def _check_field_lengths(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Check field length constraints."""
        length_limits = {"title": 500, "description": 2000, "context_id": 100}

        for field, max_length in length_limits.items():
            if value := context_data.get(field):
                if isinstance(value, str) and len(value) > max_length:
                    result.add_issue(
                        ValidationIssue(
                            message=f"Field '{field}' exceeds maximum length of {max_length}",
                            severity=ValidationSeverity.WARNING,
                            category="field_length",
                            field_path=field,
                            suggested_fix=f"Truncate '{field}' to {max_length} characters",
                        )
                    )

        # Check overall data size
        if data := context_data.get("data", {}):
            data_size = len(json.dumps(data))
            if data_size > self.max_field_length:
                result.add_issue(
                    ValidationIssue(
                        message=f"Context data size ({data_size}) exceeds limit ({self.max_field_length})",
                        severity=ValidationSeverity.WARNING,
                        category="field_length",
                        field_path="data",
                        suggested_fix="Consider compressing or splitting large data",
                    )
                )

    async def _check_timestamps(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Check timestamp consistency with improved parsing."""
        timestamps = {}
        now = datetime.now(timezone.utc)

        # Parse timestamps with error handling
        for field in self.TIMESTAMP_FIELDS:
            if value := context_data.get(field):
                timestamp = self._parse_timestamp(value, field, result)
                if timestamp:
                    timestamps[field] = timestamp

        # Check temporal relationships
        if timestamps:
            self._validate_temporal_order(timestamps, now, result)

    @staticmethod
    def _parse_timestamp(
        value: Any, field: str, result: ValidationResult
    ) -> Optional[datetime]:
        """Parse timestamp from various formats using modern Python 3.11+ patterns."""
        match value:
            case datetime() as dt:
                return dt
            case str() as s:
                try:
                    # Use modern fromisoformat with automatic timezone handling
                    return datetime.fromisoformat(s.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    result.add_issue(
                        ValidationIssue(
                            message=f"Invalid timestamp format in field '{field}'",
                            severity=ValidationSeverity.ERROR,
                            category="timestamp_validation",
                            field_path=field,
                            suggested_fix="Use ISO 8601 timestamp format",
                        )
                    )
            case _:
                return None

    @staticmethod
    def _validate_temporal_order(
        timestamps: Dict[str, datetime], now: datetime, result: ValidationResult
    ) -> None:
        """Validate temporal ordering of timestamps."""
        # Check for future timestamps (except expires_at)
        for field, ts in timestamps.items():
            if field != "expires_at" and ts > now:
                result.add_issue(
                    ValidationIssue(
                        message=f"Timestamp '{field}' is in the future",
                        severity=ValidationSeverity.WARNING,
                        category="timestamp_validation",
                        field_path=field,
                    )
                )

        # Check logical order
        if "created_at" in timestamps and "last_modified" in timestamps:
            if timestamps["created_at"] > timestamps["last_modified"]:
                result.add_issue(
                    ValidationIssue(
                        message="created_at is after last_modified",
                        severity=ValidationSeverity.ERROR,
                        category="timestamp_validation",
                        suggested_fix="Ensure created_at <= last_modified",
                    )
                )

    async def _check_data_types(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Check data type consistency."""
        type_checks = {
            "access_count": int,
            "current_version": int,
            "size_bytes": int,
            "total_interactions": int,
        }

        for field, expected_type in type_checks.items():
            if field in context_data:
                value = context_data[field]
                if not isinstance(value, expected_type):
                    result.add_issue(
                        ValidationIssue(
                            message=f"Field '{field}' has incorrect type. Expected {expected_type.__name__}",
                            severity=ValidationSeverity.WARNING,
                            category="data_consistency",
                            field_path=field,
                            suggested_fix=f"Convert '{field}' to {expected_type.__name__}",
                        )
                    )

    async def _validate_conversation_semantics(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate conversation-specific semantics."""
        messages = context_data.get("messages", [])

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                result.add_issue(
                    ValidationIssue(
                        message=f"Message at index {i} is not a dictionary",
                        severity=ValidationSeverity.ERROR,
                        category="conversation_semantics",
                        field_path=f"messages[{i}]",
                    )
                )
                continue

            # Validate message structure
            self._validate_message_structure(message, i, result)

        # Check turn count consistency
        current_turn = context_data.get("current_turn", 0)
        if current_turn != len(messages):
            result.add_issue(
                ValidationIssue(
                    message=f"current_turn ({current_turn}) doesn't match message count ({len(messages)})",
                    severity=ValidationSeverity.WARNING,
                    category="conversation_semantics",
                    suggested_fix="Sync current_turn with message count",
                )
            )

    def _validate_message_structure(
        self, message: Dict[str, Any], index: int, result: ValidationResult
    ) -> None:
        """Validate individual message structure."""
        # Check required fields
        for field in ["role", "content"]:
            if field not in message:
                result.add_issue(
                    ValidationIssue(
                        message=f"Message at index {index} missing '{field}' field",
                        severity=ValidationSeverity.ERROR,
                        category="conversation_semantics",
                        field_path=f"messages[{index}].{field}",
                    )
                )

        # Check role validity
        if role := message.get("role"):
            if role not in self.VALID_MESSAGE_ROLES:
                result.add_issue(
                    ValidationIssue(
                        message=f"Invalid role '{role}' in message {index}",
                        severity=ValidationSeverity.ERROR,
                        category="conversation_semantics",
                        field_path=f"messages[{index}].role",
                        suggested_fix=f"Use one of: {', '.join(self.VALID_MESSAGE_ROLES)}",
                    )
                )

    async def _validate_session_semantics(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate session-specific semantics."""
        if (interactions := context_data.get("total_interactions", 0)) < 0:
            result.add_issue(
                ValidationIssue(
                    message="total_interactions cannot be negative",
                    severity=ValidationSeverity.ERROR,
                    category="session_semantics",
                    field_path="total_interactions",
                    suggested_fix="Set total_interactions to 0 or positive value",
                )
            )

    async def _validate_collaborative_semantics(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate collaborative context semantics."""
        participants = context_data.get("active_participants", [])

        # Check participant limit
        if len(participants) > 50:
            result.add_issue(
                ValidationIssue(
                    message=f"Too many active participants ({len(participants)}), maximum is 50",
                    severity=ValidationSeverity.WARNING,
                    category="collaborative_semantics",
                    field_path="active_participants",
                    suggested_fix="Remove inactive participants",
                )
            )

        # Check lock consistency
        if locked_by := context_data.get("locked_by"):
            if locked_by not in participants:
                result.add_issue(
                    ValidationIssue(
                        message=f"Context locked by user '{locked_by}' who is not an active participant",
                        severity=ValidationSeverity.WARNING,
                        category="collaborative_semantics",
                        field_path="locked_by",
                        suggested_fix="Add lock holder to active participants or release lock",
                    )
                )

    async def _check_data_corruption(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Check for signs of data corruption."""
        # Check for control characters in string fields
        for field in ["title", "description"]:
            if value := context_data.get(field):
                if isinstance(value, str) and any(
                    ord(c) < 32 and c not in "\n\r\t" for c in value
                ):
                    result.add_issue(
                        ValidationIssue(
                            message=f"Field '{field}' contains control characters",
                            severity=ValidationSeverity.WARNING,
                            category="data_corruption",
                            field_path=field,
                            suggested_fix="Remove or escape control characters",
                        )
                    )

        # Check for circular references
        try:
            json.dumps(context_data)
        except ValueError as e:
            if "Circular reference" in str(e):
                result.add_issue(
                    ValidationIssue(
                        message="Circular reference detected in context data",
                        severity=ValidationSeverity.ERROR,
                        category="data_corruption",
                        suggested_fix="Remove circular references from data structure",
                    )
                )

    async def _check_permissions_integrity(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Check permissions consistency."""
        owner_id = context_data.get("owner_id")
        collaborators = context_data.get("collaborators", [])

        # Owner should not be in collaborators list
        if owner_id and owner_id in collaborators:
            result.add_issue(
                ValidationIssue(
                    message="Owner is listed as collaborator",
                    severity=ValidationSeverity.WARNING,
                    category="permissions_integrity",
                    suggested_fix="Remove owner from collaborators list",
                )
            )

    async def _check_version_integrity(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Check version consistency."""
        current_version = context_data.get("current_version", 1)
        version_history = context_data.get("version_history", [])

        if current_version < 1:
            result.add_issue(
                ValidationIssue(
                    message="current_version must be >= 1",
                    severity=ValidationSeverity.ERROR,
                    category="version_integrity",
                    field_path="current_version",
                    suggested_fix="Set current_version to 1 or higher",
                )
            )

        if len(version_history) > current_version:
            result.add_issue(
                ValidationIssue(
                    message="Version history count exceeds current version",
                    severity=ValidationSeverity.WARNING,
                    category="version_integrity",
                    suggested_fix="Sync version history with current version",
                )
            )

    async def _check_metadata_integrity(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Check metadata integrity."""
        if metadata := context_data.get("metadata", {}):
            metadata_size = len(json.dumps(metadata))
            if metadata_size > self.max_metadata_size:
                result.add_issue(
                    ValidationIssue(
                        message=f"Metadata size ({metadata_size}) exceeds limit ({self.max_metadata_size})",
                        severity=ValidationSeverity.WARNING,
                        category="metadata_integrity",
                        field_path="metadata",
                        suggested_fix="Reduce metadata size or move large data to main context",
                    )
                )

    async def _apply_auto_corrections(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> Dict[str, Any]:
        """Apply automatic corrections to fixable issues."""
        corrected_data = context_data.copy()
        corrections = 0

        # Auto-correct timestamps
        for field in self.TIMESTAMP_FIELDS:
            if value := corrected_data.get(field):
                if isinstance(value, str):
                    try:
                        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        corrected_data[field] = dt.isoformat()
                        corrections += 1
                    except (ValueError, AttributeError):
                        pass

        # Auto-correct negative counts
        for field in ["access_count", "current_version", "total_interactions"]:
            if field in corrected_data:
                value = corrected_data[field]
                if isinstance(value, (int, float)) and value < 0:
                    corrected_data[field] = max(0, 1 if field == "current_version" else 0)
                    corrections += 1

        if corrections > 0:
            self._stats.auto_corrections_applied += corrections
            result.add_issue(
                ValidationIssue(
                    message=f"Applied {corrections} automatic corrections",
                    severity=ValidationSeverity.INFO,
                    category="auto_correction",
                    metadata={"corrections_count": corrections},
                )
            )

        return corrected_data if corrections > 0 else context_data

    async def _run_custom_rules(
        self,
        context_data: Dict[str, Any],
        context_type: Type[Context],
        result: ValidationResult,
    ) -> None:
        """Run custom validation rules."""
        rules = self._custom_rules.get("all", [])
        rules.extend(self._custom_rules.get(context_type.__name__, []))

        for rule in rules:
            try:
                if asyncio.iscoroutinefunction(rule):
                    await rule(context_data, result)
                else:
                    rule(context_data, result)
            except Exception as e:
                result.add_issue(
                    ValidationIssue(
                        message=f"Custom validation rule failed: {e!s}",
                        severity=ValidationSeverity.ERROR,
                        category="custom_rule",
                    )
                )

    def add_custom_rule(
        self, rule_function: Callable, context_type: Optional[str] = None
    ) -> None:
        """Add a custom validation rule."""
        rule_category = context_type or "all"
        self._custom_rules[rule_category].append(rule_function)
        logger.info(f"Added custom validation rule for {rule_category}")

    @staticmethod
    @lru_cache(maxsize=128)
    def _suggest_schema_fix(validation_error: jsonschema.ValidationError) -> str:
        """Suggest a fix for schema validation error using pattern matching."""
        match validation_error.validator:
            case "required":
                return f"Add required property: {validation_error.message}"
            case "type":
                return f"Change type to {validation_error.validator_value}"
            case "enum":
                return f"Use one of: {', '.join(validation_error.validator_value)}"
            case "maxLength":
                return f"Reduce length to {validation_error.validator_value} characters"
            case _:
                return "Fix schema validation error"

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        return {
            "validation_stats": self._stats.to_dict(),
            "custom_rules_count": sum(len(rules) for rules in self._custom_rules.values()),
            "validation_features": {
                "schema_validation": self.enable_schema_validation,
                "semantic_validation": self.enable_semantic_validation,
                "integrity_checks": self.enable_integrity_checks,
                "auto_correction": self.enable_auto_correction,
            },
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self._custom_rules.clear()
        self._suggest_schema_fix.cache_clear()
        if hasattr(self, "_schemas"):
            del self._schemas
        logger.info("Context validator cleaned up")


@dataclass
class ValidationStats:
    """Performance tracking for validations."""

    validations_performed: int = 0
    validations_passed: int = 0
    validations_failed: int = 0
    auto_corrections_applied: int = 0
    total_validation_time: float = 0.0

    def record_validation(self, is_valid: bool, execution_time: float) -> None:
        """Record validation result."""
        self.validations_performed += 1
        if is_valid:
            self.validations_passed += 1
        else:
            self.validations_failed += 1
        self.total_validation_time += execution_time

    @property
    def avg_validation_time(self) -> float:
        """Calculate average validation time."""
        if self.validations_performed == 0:
            return 0.0
        return self.total_validation_time / self.validations_performed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validations_performed": self.validations_performed,
            "validations_passed": self.validations_passed,
            "validations_failed": self.validations_failed,
            "auto_corrections_applied": self.auto_corrections_applied,
            "avg_validation_time": self.avg_validation_time,
        }