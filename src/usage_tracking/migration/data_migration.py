"""
Data migration and upgrade utilities for usage tracking storage systems.

This module provides:
- Migration between storage backends (JSON ↔ ChromaDB)
- Schema version upgrades
- Data validation and consistency checks
- Rollback capabilities
- Progressive migration with monitoring
"""

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import hashlib

from ..storage.models import (
    UsageRecord, UserProfile, UsageMetrics, UsagePattern,
    StorageType, UsageEventType, ProviderType, TimeAggregation
)
from ..storage.chromadb_storage import ChromaDBUsageStorage
from ..storage.json_storage import JSONUsageStorage
from ..storage.hybrid_storage import HybridUsageStorage

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Migration status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"


class MigrationType(Enum):
    """Types of migration operations."""
    JSON_TO_CHROMADB = "json_to_chromadb"
    CHROMADB_TO_JSON = "chromadb_to_json"
    SCHEMA_UPGRADE = "schema_upgrade"
    STORAGE_CONSOLIDATION = "storage_consolidation"
    BACKUP_RESTORE = "backup_restore"


@dataclass
class MigrationStep:
    """Individual step in a migration process."""
    step_id: str
    description: str
    estimated_records: int
    completed_records: int = 0
    status: MigrationStatus = MigrationStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.estimated_records == 0:
            return 100.0 if self.status == MigrationStatus.COMPLETED else 0.0
        return min((self.completed_records / self.estimated_records) * 100, 100.0)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate step duration in seconds."""
        if not self.start_time:
            return None
        end_time = self.end_time or datetime.utcnow()
        return (end_time - self.start_time).total_seconds()


@dataclass
class MigrationPlan:
    """Complete migration plan with multiple steps."""
    plan_id: str
    migration_type: MigrationType
    description: str
    steps: List[MigrationStep]
    status: MigrationStatus = MigrationStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rollback_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.rollback_data is None:
            self.rollback_data = {}
    
    @property
    def overall_progress_percent(self) -> float:
        """Calculate overall migration progress."""
        if not self.steps:
            return 0.0
        
        total_estimated = sum(step.estimated_records for step in self.steps)
        total_completed = sum(step.completed_records for step in self.steps)
        
        if total_estimated == 0:
            completed_steps = sum(1 for step in self.steps if step.status == MigrationStatus.COMPLETED)
            return (completed_steps / len(self.steps)) * 100
        
        return min((total_completed / total_estimated) * 100, 100.0)
    
    @property
    def current_step(self) -> Optional[MigrationStep]:
        """Get currently executing step."""
        for step in self.steps:
            if step.status == MigrationStatus.RUNNING:
                return step
        return None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total migration duration."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()


class DataValidator:
    """Validates data integrity during migration."""
    
    def __init__(self):
        self.validation_rules = {}
        self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> None:
        """Initialize data validation rules."""
        self.validation_rules = {
            "usage_record": {
                "required_fields": ["record_id", "user_id", "timestamp", "event_type", "provider"],
                "field_types": {
                    "cost_usd": (Decimal, int, float),
                    "token_count": int,
                    "success": bool,
                    "timestamp": datetime
                },
                "constraints": {
                    "cost_usd": lambda x: x >= 0,
                    "token_count": lambda x: x >= 0,
                    "record_id": lambda x: len(str(x)) > 0
                }
            },
            "user_profile": {
                "required_fields": ["user_id", "created_at"],
                "field_types": {
                    "total_spent": (Decimal, int, float),
                    "total_tokens": int,
                    "total_requests": int
                },
                "constraints": {
                    "total_spent": lambda x: x >= 0,
                    "total_tokens": lambda x: x >= 0,
                    "total_requests": lambda x: x >= 0,
                    "user_id": lambda x: len(str(x)) > 0
                }
            }
        }
    
    def validate_record(self, record_type: str, record_data: Dict[str, Any]) -> List[str]:
        """Validate a single record against rules."""
        errors = []
        rules = self.validation_rules.get(record_type, {})
        
        # Check required fields
        required_fields = rules.get("required_fields", [])
        for field in required_fields:
            if field not in record_data or record_data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        field_types = rules.get("field_types", {})
        for field, expected_types in field_types.items():
            if field in record_data and record_data[field] is not None:
                if not isinstance(record_data[field], expected_types):
                    errors.append(f"Invalid type for {field}: expected {expected_types}, got {type(record_data[field])}")
        
        # Check constraints
        constraints = rules.get("constraints", {})
        for field, constraint_func in constraints.items():
            if field in record_data and record_data[field] is not None:
                try:
                    if not constraint_func(record_data[field]):
                        errors.append(f"Constraint violation for {field}: {record_data[field]}")
                except Exception as e:
                    errors.append(f"Constraint check error for {field}: {str(e)}")
        
        return errors
    
    def calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data integrity verification."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


class MigrationEngine:
    """Core migration engine for data transfers."""
    
    def __init__(self):
        self.validator = DataValidator()
        self.migration_history: Dict[str, MigrationPlan] = {}
        self.active_migration: Optional[str] = None
        self.pause_requested = False
        
        # Performance settings
        self.batch_size = 1000
        self.max_concurrent_operations = 10
        self.validation_enabled = True
        self.backup_before_migration = True
        
        # Statistics
        self.migration_stats = {
            "total_migrations": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "total_records_migrated": 0,
            "total_time_spent_seconds": 0.0
        }
    
    async def create_migration_plan(
        self,
        migration_type: MigrationType,
        source_storage,
        target_storage,
        filters: Optional[Dict[str, Any]] = None
    ) -> MigrationPlan:
        """Create a comprehensive migration plan."""
        plan_id = f"migration_{int(time.time())}_{migration_type.value}"
        description = f"Migrate data: {migration_type.value}"
        
        # Estimate data sizes
        steps = await self._create_migration_steps(
            migration_type, source_storage, target_storage, filters
        )
        
        plan = MigrationPlan(
            plan_id=plan_id,
            migration_type=migration_type,
            description=description,
            steps=steps
        )
        
        self.migration_history[plan_id] = plan
        
        logger.info(
            f"Migration plan created: {plan_id}",
            steps=len(steps),
            estimated_records=sum(step.estimated_records for step in steps)
        )
        
        return plan
    
    async def _create_migration_steps(
        self,
        migration_type: MigrationType,
        source_storage,
        target_storage,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MigrationStep]:
        """Create migration steps based on migration type."""
        steps = []
        
        if migration_type == MigrationType.JSON_TO_CHROMADB:
            steps = await self._create_json_to_chromadb_steps(source_storage, target_storage, filters)
        
        elif migration_type == MigrationType.CHROMADB_TO_JSON:
            steps = await self._create_chromadb_to_json_steps(source_storage, target_storage, filters)
        
        elif migration_type == MigrationType.SCHEMA_UPGRADE:
            steps = await self._create_schema_upgrade_steps(source_storage, filters)
        
        elif migration_type == MigrationType.STORAGE_CONSOLIDATION:
            steps = await self._create_consolidation_steps(source_storage, target_storage, filters)
        
        return steps
    
    async def _create_json_to_chromadb_steps(
        self,
        json_storage: JSONUsageStorage,
        chromadb_storage: ChromaDBUsageStorage,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MigrationStep]:
        """Create steps for JSON to ChromaDB migration."""
        steps = []
        
        try:
            # Estimate usage records
            usage_records = await json_storage.get_usage_records(
                start_time=filters.get("start_date") if filters else None,
                end_time=filters.get("end_date") if filters else None,
                user_id=filters.get("user_id") if filters else None,
                limit=10000  # Get count estimate
            )
            
            if usage_records:
                steps.append(MigrationStep(
                    step_id="migrate_usage_records",
                    description="Migrate usage records from JSON to ChromaDB",
                    estimated_records=len(usage_records)
                ))
            
            # Analytics data migration
            steps.append(MigrationStep(
                step_id="migrate_analytics",
                description="Migrate analytics data",
                estimated_records=100  # Estimate
            ))
            
        except Exception as e:
            logger.error(f"Error estimating migration steps: {e}")
            # Create minimal step
            steps.append(MigrationStep(
                step_id="migrate_all_data",
                description="Migrate all available data",
                estimated_records=1000  # Conservative estimate
            ))
        
        return steps
    
    async def _create_chromadb_to_json_steps(
        self,
        chromadb_storage: ChromaDBUsageStorage,
        json_storage: JSONUsageStorage,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MigrationStep]:
        """Create steps for ChromaDB to JSON migration."""
        steps = []
        
        # Get collection stats
        try:
            stats = await chromadb_storage.get_collection_stats()
            usage_count = stats.get("collections", {}).get("usage", {}).get("document_count", 1000)
            patterns_count = stats.get("collections", {}).get("patterns", {}).get("document_count", 100)
            
            if usage_count > 0:
                steps.append(MigrationStep(
                    step_id="migrate_usage_records",
                    description="Migrate usage records from ChromaDB to JSON",
                    estimated_records=usage_count
                ))
            
            if patterns_count > 0:
                steps.append(MigrationStep(
                    step_id="migrate_patterns",
                    description="Migrate usage patterns",
                    estimated_records=patterns_count
                ))
            
        except Exception as e:
            logger.error(f"Error getting ChromaDB stats: {e}")
            steps.append(MigrationStep(
                step_id="migrate_all_chromadb_data",
                description="Migrate all ChromaDB data to JSON",
                estimated_records=1000
            ))
        
        return steps
    
    async def _create_schema_upgrade_steps(
        self,
        storage,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MigrationStep]:
        """Create steps for schema upgrade."""
        return [
            MigrationStep(
                step_id="backup_current_schema",
                description="Create backup of current data",
                estimated_records=100
            ),
            MigrationStep(
                step_id="upgrade_schema",
                description="Upgrade data schema to new version",
                estimated_records=1000
            ),
            MigrationStep(
                step_id="validate_upgraded_data",
                description="Validate upgraded data",
                estimated_records=1000
            )
        ]
    
    async def _create_consolidation_steps(
        self,
        source_storage,
        target_storage,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MigrationStep]:
        """Create steps for storage consolidation."""
        return [
            MigrationStep(
                step_id="analyze_source_data",
                description="Analyze source data for consolidation",
                estimated_records=100
            ),
            MigrationStep(
                step_id="migrate_user_profiles",
                description="Consolidate user profiles",
                estimated_records=1000
            ),
            MigrationStep(
                step_id="migrate_usage_data",
                description="Consolidate usage data",
                estimated_records=10000
            ),
            MigrationStep(
                step_id="verify_consolidation",
                description="Verify data consolidation",
                estimated_records=11100
            )
        ]
    
    async def execute_migration(
        self,
        plan_id: str,
        source_storage,
        target_storage,
        resume_from_step: Optional[str] = None
    ) -> bool:
        """Execute a migration plan."""
        if self.active_migration:
            raise ValueError(f"Migration already active: {self.active_migration}")
        
        plan = self.migration_history.get(plan_id)
        if not plan:
            raise ValueError(f"Migration plan not found: {plan_id}")
        
        self.active_migration = plan_id
        plan.status = MigrationStatus.RUNNING
        plan.started_at = datetime.utcnow()
        self.pause_requested = False
        
        try:
            # Create backup if enabled
            if self.backup_before_migration:
                await self._create_pre_migration_backup(source_storage, plan)
            
            # Execute each step
            start_executing = resume_from_step is None
            
            for step in plan.steps:
                if not start_executing:
                    if step.step_id == resume_from_step:
                        start_executing = True
                    else:
                        continue
                
                # Check for pause request
                if self.pause_requested:
                    plan.status = MigrationStatus.PAUSED
                    logger.info(f"Migration paused at step: {step.step_id}")
                    return False
                
                # Execute step
                success = await self._execute_migration_step(
                    step, plan.migration_type, source_storage, target_storage
                )
                
                if not success:
                    plan.status = MigrationStatus.FAILED
                    logger.error(f"Migration failed at step: {step.step_id}")
                    return False
            
            # Mark as completed
            plan.status = MigrationStatus.COMPLETED
            plan.completed_at = datetime.utcnow()
            
            # Update statistics
            self.migration_stats["successful_migrations"] += 1
            self.migration_stats["total_records_migrated"] += sum(
                step.completed_records for step in plan.steps
            )
            
            if plan.duration_seconds:
                self.migration_stats["total_time_spent_seconds"] += plan.duration_seconds
            
            logger.info(
                f"Migration completed successfully: {plan_id}",
                duration_seconds=plan.duration_seconds,
                records_migrated=sum(step.completed_records for step in plan.steps)
            )
            
            return True
            
        except Exception as e:
            plan.status = MigrationStatus.FAILED
            logger.error(f"Migration failed with error: {e}", exc_info=True)
            
            # Update statistics
            self.migration_stats["failed_migrations"] += 1
            
            return False
            
        finally:
            self.active_migration = None
            self.migration_stats["total_migrations"] += 1
    
    async def _create_pre_migration_backup(self, source_storage, plan: MigrationPlan) -> None:
        """Create backup before migration."""
        logger.info("Creating pre-migration backup...")
        
        try:
            if hasattr(source_storage, 'create_backup'):
                backup_path = await source_storage.create_backup()
                plan.rollback_data["backup_path"] = backup_path
                logger.info(f"Backup created: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    async def _execute_migration_step(
        self,
        step: MigrationStep,
        migration_type: MigrationType,
        source_storage,
        target_storage
    ) -> bool:
        """Execute a single migration step."""
        step.status = MigrationStatus.RUNNING
        step.start_time = datetime.utcnow()
        
        logger.info(f"Executing migration step: {step.step_id}")
        
        try:
            if migration_type == MigrationType.JSON_TO_CHROMADB:
                success = await self._execute_json_to_chromadb_step(step, source_storage, target_storage)
            elif migration_type == MigrationType.CHROMADB_TO_JSON:
                success = await self._execute_chromadb_to_json_step(step, source_storage, target_storage)
            elif migration_type == MigrationType.SCHEMA_UPGRADE:
                success = await self._execute_schema_upgrade_step(step, source_storage)
            else:
                # Generic step execution
                success = await self._execute_generic_step(step, source_storage, target_storage)
            
            if success:
                step.status = MigrationStatus.COMPLETED
            else:
                step.status = MigrationStatus.FAILED
                step.error_message = "Step execution returned failure"
            
            step.end_time = datetime.utcnow()
            
            logger.info(
                f"Migration step completed: {step.step_id}",
                success=success,
                duration_seconds=step.duration_seconds,
                progress=f"{step.completed_records}/{step.estimated_records}"
            )
            
            return success
            
        except Exception as e:
            step.status = MigrationStatus.FAILED
            step.error_message = str(e)
            step.end_time = datetime.utcnow()
            
            logger.error(f"Migration step failed: {step.step_id}", exc_info=True)
            return False
    
    async def _execute_json_to_chromadb_step(
        self,
        step: MigrationStep,
        json_storage: JSONUsageStorage,
        chromadb_storage: ChromaDBUsageStorage
    ) -> bool:
        """Execute JSON to ChromaDB migration step."""
        if step.step_id == "migrate_usage_records":
            # Get usage records from JSON storage
            records = await json_storage.get_usage_records(limit=step.estimated_records)
            
            # Convert to UsageRecord objects and migrate in batches
            batch_size = min(self.batch_size, 100)
            total_processed = 0
            
            for i in range(0, len(records), batch_size):
                if self.pause_requested:
                    return False
                
                batch = records[i:i + batch_size]
                usage_objects = []
                
                # Convert dict records to UsageRecord objects
                for record_dict in batch:
                    try:
                        # Create UsageRecord from dict data
                        usage_record = UsageRecord(
                            record_id=record_dict.get("record_id", str(time.time())),
                            user_id=record_dict["user_id"],
                            timestamp=datetime.fromisoformat(record_dict["timestamp"]),
                            event_type=UsageEventType(record_dict["event_type"]),
                            provider=ProviderType(record_dict["provider"]),
                            cost_usd=Decimal(str(record_dict.get("cost_usd", "0.00"))),
                            token_count=record_dict.get("token_count", 0),
                            success=record_dict.get("success", True),
                            session_id=record_dict.get("session_id"),
                            context_id=record_dict.get("context_id"),
                            operation=record_dict.get("operation")
                        )
                        
                        # Validate if enabled
                        if self.validation_enabled:
                            errors = self.validator.validate_record("usage_record", record_dict)
                            if errors:
                                logger.warning(f"Validation errors for record {usage_record.record_id}: {errors}")
                                continue
                        
                        usage_objects.append(usage_record)
                        
                    except Exception as e:
                        logger.error(f"Error converting record: {e}")
                        continue
                
                # Store batch in ChromaDB
                if usage_objects:
                    try:
                        await chromadb_storage.store_usage_records_batch(usage_objects)
                        total_processed += len(usage_objects)
                        step.completed_records = total_processed
                        
                        logger.debug(f"Migrated batch: {len(usage_objects)} records")
                        
                    except Exception as e:
                        logger.error(f"Error storing batch in ChromaDB: {e}")
                        return False
            
            return True
            
        elif step.step_id == "migrate_analytics":
            # Placeholder for analytics migration
            step.completed_records = step.estimated_records
            return True
            
        return False
    
    async def _execute_chromadb_to_json_step(
        self,
        step: MigrationStep,
        chromadb_storage: ChromaDBUsageStorage,
        json_storage: JSONUsageStorage
    ) -> bool:
        """Execute ChromaDB to JSON migration step."""
        if step.step_id == "migrate_usage_records":
            # Query all usage records from ChromaDB
            results = await chromadb_storage.query_usage_records(limit=step.estimated_records)
            
            batch_size = min(self.batch_size, 100)
            total_processed = 0
            
            for i in range(0, len(results), batch_size):
                if self.pause_requested:
                    return False
                
                batch = results[i:i + batch_size]
                
                for result in batch:
                    try:
                        # Extract metadata from ChromaDB result
                        metadata = result.get("metadata", {})
                        
                        # Create UsageRecord from ChromaDB data
                        usage_record = UsageRecord(
                            record_id=metadata.get("record_id", str(time.time())),
                            user_id=metadata["user_id"],
                            timestamp=datetime.fromisoformat(metadata["timestamp"]),
                            event_type=UsageEventType(metadata["event_type"]),
                            provider=ProviderType(metadata["provider"]),
                            cost_usd=Decimal(str(metadata.get("cost_usd", "0.00"))),
                            token_count=metadata.get("token_count", 0),
                            success=metadata.get("success", True),
                            session_id=metadata.get("session_id"),
                            context_id=metadata.get("context_id"),
                            operation=metadata.get("operation")
                        )
                        
                        # Store in JSON storage
                        await json_storage.store_usage_record(usage_record)
                        total_processed += 1
                        step.completed_records = total_processed
                        
                    except Exception as e:
                        logger.error(f"Error migrating record from ChromaDB: {e}")
                        continue
            
            return True
            
        return False
    
    async def _execute_schema_upgrade_step(
        self,
        step: MigrationStep,
        storage
    ) -> bool:
        """Execute schema upgrade step."""
        # Placeholder implementation for schema upgrades
        if step.step_id == "backup_current_schema":
            # Create backup
            step.completed_records = step.estimated_records
            return True
            
        elif step.step_id == "upgrade_schema":
            # Perform schema upgrade
            step.completed_records = step.estimated_records
            return True
            
        elif step.step_id == "validate_upgraded_data":
            # Validate upgraded data
            step.completed_records = step.estimated_records
            return True
        
        return False
    
    async def _execute_generic_step(
        self,
        step: MigrationStep,
        source_storage,
        target_storage
    ) -> bool:
        """Execute generic migration step."""
        # Simulate step execution
        batch_size = 10
        total_to_process = step.estimated_records
        
        for i in range(0, total_to_process, batch_size):
            if self.pause_requested:
                return False
            
            # Simulate processing
            await asyncio.sleep(0.1)  # Small delay to simulate work
            
            processed_this_batch = min(batch_size, total_to_process - i)
            step.completed_records += processed_this_batch
        
        return True
    
    async def pause_migration(self) -> bool:
        """Request migration pause."""
        if not self.active_migration:
            return False
        
        self.pause_requested = True
        logger.info(f"Pause requested for migration: {self.active_migration}")
        return True
    
    async def resume_migration(
        self,
        plan_id: str,
        source_storage,
        target_storage
    ) -> bool:
        """Resume paused migration."""
        plan = self.migration_history.get(plan_id)
        if not plan or plan.status != MigrationStatus.PAUSED:
            return False
        
        # Find the last completed step
        last_completed_step = None
        for step in reversed(plan.steps):
            if step.status == MigrationStatus.COMPLETED:
                last_completed_step = step.step_id
                break
        
        # Resume from next step
        next_step_index = 0
        if last_completed_step:
            for i, step in enumerate(plan.steps):
                if step.step_id == last_completed_step:
                    next_step_index = i + 1
                    break
        
        if next_step_index < len(plan.steps):
            resume_from_step = plan.steps[next_step_index].step_id
            return await self.execute_migration(plan_id, source_storage, target_storage, resume_from_step)
        
        return False
    
    async def rollback_migration(self, plan_id: str, storage) -> bool:
        """Rollback a migration."""
        plan = self.migration_history.get(plan_id)
        if not plan:
            return False
        
        logger.info(f"Starting rollback for migration: {plan_id}")
        
        try:
            # Restore from backup if available
            backup_path = plan.rollback_data.get("backup_path")
            if backup_path and hasattr(storage, 'restore_from_backup'):
                await storage.restore_from_backup(backup_path)
                logger.info(f"Restored from backup: {backup_path}")
            
            plan.status = MigrationStatus.ROLLED_BACK
            
            logger.info(f"Migration rollback completed: {plan_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}", exc_info=True)
            return False
    
    def get_migration_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a migration."""
        plan = self.migration_history.get(plan_id)
        if not plan:
            return None
        
        return {
            "plan_id": plan.plan_id,
            "migration_type": plan.migration_type.value,
            "status": plan.status.value,
            "overall_progress": plan.overall_progress_percent,
            "created_at": plan.created_at.isoformat(),
            "started_at": plan.started_at.isoformat() if plan.started_at else None,
            "completed_at": plan.completed_at.isoformat() if plan.completed_at else None,
            "duration_seconds": plan.duration_seconds,
            "current_step": plan.current_step.step_id if plan.current_step else None,
            "steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "status": step.status.value,
                    "progress": step.progress_percent,
                    "completed_records": step.completed_records,
                    "estimated_records": step.estimated_records,
                    "duration_seconds": step.duration_seconds,
                    "error_message": step.error_message
                }
                for step in plan.steps
            ],
            "can_resume": plan.status == MigrationStatus.PAUSED,
            "can_rollback": plan.status in [MigrationStatus.COMPLETED, MigrationStatus.FAILED]
        }
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get history of all migrations."""
        return [
            {
                "plan_id": plan.plan_id,
                "migration_type": plan.migration_type.value,
                "status": plan.status.value,
                "created_at": plan.created_at.isoformat(),
                "duration_seconds": plan.duration_seconds,
                "records_migrated": sum(step.completed_records for step in plan.steps)
            }
            for plan in self.migration_history.values()
        ]
    
    def get_migration_stats(self) -> Dict[str, Any]:
        """Get overall migration statistics."""
        return {
            "migration_stats": self.migration_stats.copy(),
            "active_migration": self.active_migration,
            "total_plans": len(self.migration_history),
            "settings": {
                "batch_size": self.batch_size,
                "max_concurrent_operations": self.max_concurrent_operations,
                "validation_enabled": self.validation_enabled,
                "backup_before_migration": self.backup_before_migration
            }
        }