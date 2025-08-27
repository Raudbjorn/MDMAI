"""Comprehensive context validation and integrity checking system."""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from uuid import uuid4

import jsonschema
from pydantic import ValidationError

from .models import (
    Context,
    ConversationContext,
    SessionContext,
    CollaborativeContext,
    ContextType,
    ContextState,
    ProviderContext,
)

logger = logging.getLogger(__name__)


class ValidationSeverity:
    """Severity levels for validation issues."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationIssue:
    """Represents a validation issue found in context data."""
    
    def __init__(
        self,
        issue_id: str,
        severity: str,
        category: str,
        message: str,
        field_path: Optional[str] = None,
        suggested_fix: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.issue_id = issue_id
        self.severity = severity
        self.category = category
        self.message = message
        self.field_path = field_path
        self.suggested_fix = suggested_fix
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
    
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


class ValidationResult:
    """Result of context validation."""
    
    def __init__(self):
        self.is_valid = True
        self.issues: List[ValidationIssue] = []
        self.corrected_data: Optional[Dict[str, Any]] = None
        self.validation_time = 0.0
        self.checks_performed = []
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
    
    def has_errors(self) -> bool:
        """Check if result has error-level issues."""
        return any(
            issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for issue in self.issues
        )
    
    def get_issues_by_severity(self, severity: str) -> List[ValidationIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "validation_time": self.validation_time,
            "checks_performed": self.checks_performed,
            "issues": [issue.to_dict() for issue in self.issues],
            "issue_count_by_severity": {
                ValidationSeverity.INFO: len(self.get_issues_by_severity(ValidationSeverity.INFO)),
                ValidationSeverity.WARNING: len(self.get_issues_by_severity(ValidationSeverity.WARNING)),
                ValidationSeverity.ERROR: len(self.get_issues_by_severity(ValidationSeverity.ERROR)),
                ValidationSeverity.CRITICAL: len(self.get_issues_by_severity(ValidationSeverity.CRITICAL)),
            },
        }


class ContextValidator:
    """Comprehensive context validation system with multiple validation strategies."""
    
    def __init__(
        self,
        enable_schema_validation: bool = True,
        enable_semantic_validation: bool = True,
        enable_integrity_checks: bool = True,
        enable_auto_correction: bool = True,
        max_field_length: int = 100000,
        max_metadata_size: int = 10000,
    ):
        self.enable_schema_validation = enable_schema_validation
        self.enable_semantic_validation = enable_semantic_validation
        self.enable_integrity_checks = enable_integrity_checks
        self.enable_auto_correction = enable_auto_correction
        self.max_field_length = max_field_length
        self.max_metadata_size = max_metadata_size
        
        # JSON schemas for validation
        self._schemas = self._initialize_schemas()
        
        # Performance tracking
        self._validation_stats = {
            "validations_performed": 0,
            "validations_passed": 0,
            "validations_failed": 0,
            "auto_corrections_applied": 0,
            "avg_validation_time": 0.0,
        }
        
        # Validation rules cache
        self._custom_rules: Dict[str, List[callable]] = {}
        
        logger.info(
            "Context validator initialized",
            schema_validation=enable_schema_validation,
            semantic_validation=enable_semantic_validation,
            integrity_checks=enable_integrity_checks,
            auto_correction=enable_auto_correction,
        )
    
    def _initialize_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Initialize JSON schemas for different context types."""
        base_context_schema = {
            "type": "object",
            "properties": {
                "context_id": {"type": "string", "format": "uuid"},
                "context_type": {"type": "string", "enum": [t.value for t in ContextType]},
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
            },
            "required": ["context_id", "context_type"],
            "additionalProperties": True,
        }
        
        conversation_schema = {
            **base_context_schema,
            "properties": {
                **base_context_schema["properties"],
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["user", "assistant", "system", "tool"]},
                            "content": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"},
                            "tool_calls": {"type": "array"},
                            "tool_results": {"type": "array"},
                        },
                        "required": ["role", "content"],
                    },
                },
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "current_turn": {"type": "integer", "minimum": 0},
                "max_turns": {"type": ["integer", "null"], "minimum": 1},
            },
        }
        
        session_schema = {
            **base_context_schema,
            "properties": {
                **base_context_schema["properties"],
                "session_id": {"type": "string"},
                "user_id": {"type": ["string", "null"]},
                "active_conversations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "total_interactions": {"type": "integer", "minimum": 0},
            },
        }
        
        collaborative_schema = {
            **base_context_schema,
            "properties": {
                **base_context_schema["properties"],
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
        
        return {
            "base": base_context_schema,
            "conversation": conversation_schema,
            "session": session_schema,
            "collaborative": collaborative_schema,
        }
    
    async def validate_context(
        self, context: Union[Context, Dict[str, Any]], context_type: Optional[Type[Context]] = None
    ) -> ValidationResult:
        """Comprehensive context validation."""
        start_time = time.time()
        result = ValidationResult()
        
        try:
            # Convert to dict if needed
            if isinstance(context, Context):
                context_data = context.dict()
                context_type = type(context)
            else:
                context_data = context
                
            # Determine context type for validation
            if context_type is None:
                context_type_str = context_data.get("context_type", "base")
                if context_type_str == "conversation":
                    context_type = ConversationContext
                elif context_type_str == "session":
                    context_type = SessionContext
                elif context_type_str == "collaborative":
                    context_type = CollaborativeContext
                else:
                    context_type = Context
            
            # Perform validation checks
            if self.enable_schema_validation:
                await self._validate_schema(context_data, context_type, result)
                result.checks_performed.append("schema_validation")
            
            if self.enable_semantic_validation:
                await self._validate_semantics(context_data, context_type, result)
                result.checks_performed.append("semantic_validation")
            
            if self.enable_integrity_checks:
                await self._validate_integrity(context_data, context_type, result)
                result.checks_performed.append("integrity_checks")
            
            # Apply auto-corrections if enabled
            if self.enable_auto_correction and not result.has_errors():
                corrected_data = await self._apply_auto_corrections(context_data, result)
                if corrected_data != context_data:
                    result.corrected_data = corrected_data
            
            # Run custom validation rules
            await self._run_custom_rules(context_data, context_type, result)
            
            # Update statistics
            execution_time = time.time() - start_time
            result.validation_time = execution_time
            self._validation_stats["validations_performed"] += 1
            
            if result.is_valid:
                self._validation_stats["validations_passed"] += 1
            else:
                self._validation_stats["validations_failed"] += 1
            
            self._update_avg_validation_time(execution_time)
            
            logger.debug(
                "Context validation completed",
                is_valid=result.is_valid,
                issue_count=len(result.issues),
                execution_time=execution_time,
            )
            
            return result
            
        except Exception as e:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.CRITICAL,
                category="validation_error",
                message=f"Validation process failed: {str(e)}",
            ))
            logger.error(f"Context validation failed: {e}")
            return result
    
    async def _validate_schema(
        self, context_data: Dict[str, Any], context_type: Type[Context], result: ValidationResult
    ) -> None:
        """Validate context against JSON schema."""
        try:
            # Determine schema to use
            schema_name = "base"
            if context_type == ConversationContext:
                schema_name = "conversation"
            elif context_type == SessionContext:
                schema_name = "session"
            elif context_type == CollaborativeContext:
                schema_name = "collaborative"
            
            schema = self._schemas.get(schema_name, self._schemas["base"])
            
            # Validate against schema
            jsonschema.validate(context_data, schema)
            
        except jsonschema.ValidationError as e:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.ERROR,
                category="schema_validation",
                message=f"Schema validation failed: {e.message}",
                field_path=".".join(str(p) for p in e.absolute_path) if e.absolute_path else None,
                suggested_fix=self._suggest_schema_fix(e),
            ))
        except Exception as e:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.WARNING,
                category="schema_validation",
                message=f"Schema validation error: {str(e)}",
            ))
    
    async def _validate_semantics(
        self, context_data: Dict[str, Any], context_type: Type[Context], result: ValidationResult
    ) -> None:
        """Validate semantic consistency and business logic."""
        # Check data consistency
        await self._check_data_consistency(context_data, result)
        
        # Check field lengths
        await self._check_field_lengths(context_data, result)
        
        # Check timestamps
        await self._check_timestamps(context_data, result)
        
        # Type-specific semantic checks
        if context_type == ConversationContext:
            await self._validate_conversation_semantics(context_data, result)
        elif context_type == SessionContext:
            await self._validate_session_semantics(context_data, result)
        elif context_type == CollaborativeContext:
            await self._validate_collaborative_semantics(context_data, result)
    
    async def _validate_integrity(
        self, context_data: Dict[str, Any], context_type: Type[Context], result: ValidationResult
    ) -> None:
        """Validate data integrity and referential consistency."""
        # Check for data corruption
        await self._check_data_corruption(context_data, result)
        
        # Check permissions consistency
        await self._check_permissions_integrity(context_data, result)
        
        # Check version consistency
        await self._check_version_integrity(context_data, result)
        
        # Check metadata integrity
        await self._check_metadata_integrity(context_data, result)
    
    async def _check_data_consistency(self, context_data: Dict[str, Any], result: ValidationResult) -> None:
        """Check for data consistency issues."""
        # Check required fields
        required_fields = ["context_id", "context_type"]
        for field in required_fields:
            if field not in context_data:
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.ERROR,
                    category="data_consistency",
                    message=f"Required field '{field}' is missing",
                    field_path=field,
                    suggested_fix=f"Add required field '{field}'",
                ))
        
        # Check for null values in critical fields
        critical_fields = ["context_id", "context_type", "created_at"]
        for field in critical_fields:
            if context_data.get(field) is None:
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.WARNING,
                    category="data_consistency",
                    message=f"Critical field '{field}' has null value",
                    field_path=field,
                ))
        
        # Check data types
        type_checks = {
            "access_count": int,
            "current_version": int,
            "size_bytes": int,
        }
        
        for field, expected_type in type_checks.items():
            if field in context_data and not isinstance(context_data[field], expected_type):
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.WARNING,
                    category="data_consistency",
                    message=f"Field '{field}' has incorrect type. Expected {expected_type.__name__}",
                    field_path=field,
                    suggested_fix=f"Convert '{field}' to {expected_type.__name__}",
                ))
    
    async def _check_field_lengths(self, context_data: Dict[str, Any], result: ValidationResult) -> None:
        """Check field length constraints."""
        length_limits = {
            "title": 500,
            "description": 2000,
            "context_id": 100,
        }
        
        for field, max_length in length_limits.items():
            if field in context_data and isinstance(context_data[field], str):
                if len(context_data[field]) > max_length:
                    result.add_issue(ValidationIssue(
                        issue_id=str(uuid4()),
                        severity=ValidationSeverity.WARNING,
                        category="field_length",
                        message=f"Field '{field}' exceeds maximum length of {max_length}",
                        field_path=field,
                        suggested_fix=f"Truncate '{field}' to {max_length} characters",
                    ))
        
        # Check overall data size
        data_str = json.dumps(context_data.get("data", {}))
        if len(data_str) > self.max_field_length:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.WARNING,
                category="field_length",
                message=f"Context data size ({len(data_str)}) exceeds limit ({self.max_field_length})",
                field_path="data",
                suggested_fix="Consider compressing or splitting large data",
            ))
    
    async def _check_timestamps(self, context_data: Dict[str, Any], result: ValidationResult) -> None:
        """Check timestamp consistency."""
        timestamp_fields = ["created_at", "last_modified", "last_accessed", "expires_at"]
        timestamps = {}
        
        # Parse timestamps
        for field in timestamp_fields:
            if field in context_data and context_data[field]:
                try:
                    if isinstance(context_data[field], str):
                        timestamps[field] = datetime.fromisoformat(context_data[field].replace("Z", "+00:00"))
                    elif isinstance(context_data[field], datetime):
                        timestamps[field] = context_data[field]
                except Exception:
                    result.add_issue(ValidationIssue(
                        issue_id=str(uuid4()),
                        severity=ValidationSeverity.ERROR,
                        category="timestamp_validation",
                        message=f"Invalid timestamp format in field '{field}'",
                        field_path=field,
                        suggested_fix="Use ISO 8601 timestamp format",
                    ))
        
        # Check timestamp relationships
        now = datetime.now(timezone.utc)
        
        # Future timestamps check
        for field, ts in timestamps.items():
            if field != "expires_at" and ts > now:
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.WARNING,
                    category="timestamp_validation",
                    message=f"Timestamp '{field}' is in the future",
                    field_path=field,
                ))
        
        # Logical order checks
        if "created_at" in timestamps and "last_modified" in timestamps:
            if timestamps["created_at"] > timestamps["last_modified"]:
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.ERROR,
                    category="timestamp_validation",
                    message="created_at is after last_modified",
                    suggested_fix="Ensure created_at <= last_modified",
                ))
    
    async def _validate_conversation_semantics(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate conversation-specific semantics."""
        messages = context_data.get("messages", [])
        
        # Check message consistency
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.ERROR,
                    category="conversation_semantics",
                    message=f"Message at index {i} is not a dictionary",
                    field_path=f"messages[{i}]",
                ))
                continue
            
            # Check required message fields
            if "role" not in message:
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.ERROR,
                    category="conversation_semantics",
                    message=f"Message at index {i} missing 'role' field",
                    field_path=f"messages[{i}].role",
                ))
            
            if "content" not in message:
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.ERROR,
                    category="conversation_semantics",
                    message=f"Message at index {i} missing 'content' field",
                    field_path=f"messages[{i}].content",
                ))
            
            # Check role validity
            valid_roles = ["user", "assistant", "system", "tool"]
            if message.get("role") not in valid_roles:
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.ERROR,
                    category="conversation_semantics",
                    message=f"Invalid role '{message.get('role')}' in message {i}",
                    field_path=f"messages[{i}].role",
                    suggested_fix=f"Use one of: {', '.join(valid_roles)}",
                ))
        
        # Check turn count consistency
        current_turn = context_data.get("current_turn", 0)
        if current_turn != len(messages):
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.WARNING,
                category="conversation_semantics",
                message=f"current_turn ({current_turn}) doesn't match message count ({len(messages)})",
                suggested_fix="Sync current_turn with message count",
            ))
    
    async def _validate_session_semantics(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate session-specific semantics."""
        # Check interaction count consistency
        total_interactions = context_data.get("total_interactions", 0)
        if total_interactions < 0:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.ERROR,
                category="session_semantics",
                message="total_interactions cannot be negative",
                field_path="total_interactions",
                suggested_fix="Set total_interactions to 0 or positive value",
            ))
    
    async def _validate_collaborative_semantics(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate collaborative context semantics."""
        active_participants = context_data.get("active_participants", [])
        
        # Check participant limit
        if len(active_participants) > 50:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.WARNING,
                category="collaborative_semantics",
                message=f"Too many active participants ({len(active_participants)}), maximum is 50",
                field_path="active_participants",
                suggested_fix="Remove inactive participants",
            ))
        
        # Check lock consistency
        locked_by = context_data.get("locked_by")
        if locked_by and locked_by not in active_participants:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.WARNING,
                category="collaborative_semantics",
                message=f"Context locked by user '{locked_by}' who is not an active participant",
                field_path="locked_by",
                suggested_fix="Add lock holder to active participants or release lock",
            ))
    
    async def _check_data_corruption(self, context_data: Dict[str, Any], result: ValidationResult) -> None:
        """Check for signs of data corruption."""
        # Check for malformed JSON in string fields
        string_fields = ["title", "description"]
        for field in string_fields:
            if field in context_data and isinstance(context_data[field], str):
                # Check for control characters
                if any(ord(c) < 32 and c not in ['\n', '\r', '\t'] for c in context_data[field]):
                    result.add_issue(ValidationIssue(
                        issue_id=str(uuid4()),
                        severity=ValidationSeverity.WARNING,
                        category="data_corruption",
                        message=f"Field '{field}' contains control characters",
                        field_path=field,
                        suggested_fix="Remove or escape control characters",
                    ))
        
        # Check for circular references in data
        try:
            json.dumps(context_data)
        except ValueError as e:
            if "Circular reference" in str(e):
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.ERROR,
                    category="data_corruption",
                    message="Circular reference detected in context data",
                    suggested_fix="Remove circular references from data structure",
                ))
    
    async def _check_permissions_integrity(self, context_data: Dict[str, Any], result: ValidationResult) -> None:
        """Check permissions consistency."""
        owner_id = context_data.get("owner_id")
        collaborators = context_data.get("collaborators", [])
        
        # Owner should not be in collaborators list
        if owner_id and owner_id in collaborators:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.WARNING,
                category="permissions_integrity",
                message="Owner is listed as collaborator",
                suggested_fix="Remove owner from collaborators list",
            ))
    
    async def _check_version_integrity(self, context_data: Dict[str, Any], result: ValidationResult) -> None:
        """Check version consistency."""
        current_version = context_data.get("current_version", 1)
        version_history = context_data.get("version_history", [])
        
        if current_version < 1:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.ERROR,
                category="version_integrity",
                message="current_version must be >= 1",
                field_path="current_version",
                suggested_fix="Set current_version to 1 or higher",
            ))
        
        if len(version_history) > current_version:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.WARNING,
                category="version_integrity",
                message="Version history count exceeds current version",
                suggested_fix="Sync version history with current version",
            ))
    
    async def _check_metadata_integrity(self, context_data: Dict[str, Any], result: ValidationResult) -> None:
        """Check metadata integrity."""
        metadata = context_data.get("metadata", {})
        
        # Check metadata size
        metadata_str = json.dumps(metadata)
        if len(metadata_str) > self.max_metadata_size:
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.WARNING,
                category="metadata_integrity",
                message=f"Metadata size ({len(metadata_str)}) exceeds limit ({self.max_metadata_size})",
                field_path="metadata",
                suggested_fix="Reduce metadata size or move large data to main context",
            ))
    
    async def _apply_auto_corrections(
        self, context_data: Dict[str, Any], result: ValidationResult
    ) -> Dict[str, Any]:
        """Apply automatic corrections to fixable issues."""
        corrected_data = context_data.copy()
        corrections_applied = 0
        
        # Auto-correct timestamp formats
        timestamp_fields = ["created_at", "last_modified", "last_accessed"]
        for field in timestamp_fields:
            if field in corrected_data and isinstance(corrected_data[field], str):
                try:
                    # Normalize timestamp format
                    dt = datetime.fromisoformat(corrected_data[field].replace("Z", "+00:00"))
                    corrected_data[field] = dt.isoformat()
                    corrections_applied += 1
                except Exception:
                    pass
        
        # Auto-correct negative counts
        count_fields = ["access_count", "current_version", "total_interactions"]
        for field in count_fields:
            if field in corrected_data and isinstance(corrected_data[field], (int, float)):
                if corrected_data[field] < 0:
                    corrected_data[field] = 0
                    corrections_applied += 1
        
        # Auto-correct version consistency
        if "current_version" in corrected_data and corrected_data["current_version"] < 1:
            corrected_data["current_version"] = 1
            corrections_applied += 1
        
        if corrections_applied > 0:
            self._validation_stats["auto_corrections_applied"] += corrections_applied
            result.add_issue(ValidationIssue(
                issue_id=str(uuid4()),
                severity=ValidationSeverity.INFO,
                category="auto_correction",
                message=f"Applied {corrections_applied} automatic corrections",
                metadata={"corrections_count": corrections_applied},
            ))
        
        return corrected_data
    
    async def _run_custom_rules(
        self, context_data: Dict[str, Any], context_type: Type[Context], result: ValidationResult
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
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.ERROR,
                    category="custom_rule",
                    message=f"Custom validation rule failed: {str(e)}",
                ))
    
    def add_custom_rule(
        self, rule_function: callable, context_type: Optional[str] = None
    ) -> None:
        """Add a custom validation rule."""
        rule_category = context_type or "all"
        if rule_category not in self._custom_rules:
            self._custom_rules[rule_category] = []
        self._custom_rules[rule_category].append(rule_function)
        
        logger.info(f"Added custom validation rule for {rule_category}")
    
    def _suggest_schema_fix(self, validation_error: jsonschema.ValidationError) -> str:
        """Suggest a fix for schema validation error."""
        if validation_error.validator == "required":
            return f"Add required property: {validation_error.message}"
        elif validation_error.validator == "type":
            return f"Change type to {validation_error.validator_value}"
        elif validation_error.validator == "enum":
            return f"Use one of: {', '.join(validation_error.validator_value)}"
        elif validation_error.validator == "maxLength":
            return f"Reduce length to {validation_error.validator_value} characters"
        else:
            return "Fix schema validation error"
    
    def _update_avg_validation_time(self, new_time: float) -> None:
        """Update average validation time."""
        count = self._validation_stats["validations_performed"]
        current_avg = self._validation_stats["avg_validation_time"]
        
        if count > 1:
            self._validation_stats["avg_validation_time"] = (
                (current_avg * (count - 1) + new_time) / count
            )
        else:
            self._validation_stats["avg_validation_time"] = new_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        return {
            "validation_stats": self._validation_stats,
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
        logger.info("Context validator cleaned up")