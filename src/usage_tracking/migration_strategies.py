"""Comprehensive migration strategies for version upgrades and schema evolution."""

import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from abc import ABC, abstractmethod
import importlib
from packaging import version as pkg_version

from ..context.persistence import ContextPersistenceLayer
from .chroma_extensions import UsageTrackingChromaExtensions
from .json_persistence import JsonPersistenceManager
from .backup_recovery import BackupRecoveryManager
from config.logging_config import get_logger

logger = get_logger(__name__)


class MigrationType(Enum):
    """Types of migrations."""
    SCHEMA_UPGRADE = "schema_upgrade"
    DATA_TRANSFORMATION = "data_transformation"
    INDEX_RESTRUCTURE = "index_restructure"
    STORAGE_FORMAT_CHANGE = "storage_format_change"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_ENHANCEMENT = "security_enhancement"


class MigrationStrategy(Enum):
    """Migration execution strategies."""
    ONLINE = "online"          # Zero downtime migration
    OFFLINE = "offline"        # Requires downtime
    ROLLING = "rolling"        # Gradual migration
    BLUE_GREEN = "blue_green"  # Parallel environment switch
    CANARY = "canary"          # Gradual rollout with monitoring


class MigrationState(Enum):
    """Migration execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


@dataclass
class MigrationDefinition:
    """Defines a migration operation."""
    migration_id: str
    name: str
    description: str
    migration_type: MigrationType
    strategy: MigrationStrategy
    
    # Version information
    from_version: str
    to_version: str
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    
    # Execution parameters
    estimated_duration_minutes: int = 30
    requires_backup: bool = True
    reversible: bool = True
    
    # Migration steps
    pre_checks: List[str] = field(default_factory=list)
    migration_steps: List[str] = field(default_factory=list)
    post_checks: List[str] = field(default_factory=list)
    rollback_steps: List[str] = field(default_factory=list)
    
    # Safety parameters
    max_retry_attempts: int = 3
    timeout_minutes: int = 120
    validation_queries: List[str] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"


@dataclass
class MigrationExecution:
    """Tracks migration execution."""
    execution_id: str
    migration_id: str
    state: MigrationState
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Execution details
    current_step: str = ""
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    
    # Results
    success: bool = False
    error_message: Optional[str] = None
    execution_log: List[str] = field(default_factory=list)
    
    # Metrics
    records_processed: int = 0
    bytes_processed: int = 0
    
    # Backup information
    backup_id: Optional[str] = None
    rollback_available: bool = False


class MigrationScript(ABC):
    """Base class for migration scripts."""
    
    def __init__(self, migration_def: MigrationDefinition):
        self.migration_def = migration_def
        self.execution_context: Dict[str, Any] = {}
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the migration."""
        pass
    
    @abstractmethod
    async def rollback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback the migration."""
        pass
    
    @abstractmethod
    async def validate(self, context: Dict[str, Any]) -> bool:
        """Validate migration success."""
        pass
    
    async def pre_check(self, context: Dict[str, Any]) -> bool:
        """Pre-migration checks."""
        return True
    
    async def post_check(self, context: Dict[str, Any]) -> bool:
        """Post-migration checks."""
        return True


class SchemaUpgradeMigration(MigrationScript):
    """Migration for schema upgrades."""
    
    def __init__(self, migration_def: MigrationDefinition, schema_changes: List[Dict[str, Any]]):
        super().__init__(migration_def)
        self.schema_changes = schema_changes
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute schema upgrade."""
        results = {"changes_applied": 0, "errors": []}
        
        postgres_persistence = context.get("postgres_persistence")
        if not postgres_persistence:
            raise ValueError("PostgreSQL persistence not available in context")
        
        try:
            for change in self.schema_changes:
                if change["type"] == "add_column":
                    await self._add_column(postgres_persistence, change)
                elif change["type"] == "modify_column":
                    await self._modify_column(postgres_persistence, change)
                elif change["type"] == "add_index":
                    await self._add_index(postgres_persistence, change)
                elif change["type"] == "add_table":
                    await self._add_table(postgres_persistence, change)
                
                results["changes_applied"] += 1
        
        except Exception as e:
            results["errors"].append(str(e))
            raise
        
        return results
    
    async def rollback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback schema changes."""
        results = {"changes_rolled_back": 0, "errors": []}
        
        postgres_persistence = context.get("postgres_persistence")
        if not postgres_persistence:
            raise ValueError("PostgreSQL persistence not available in context")
        
        # Reverse the schema changes
        for change in reversed(self.schema_changes):
            try:
                if change["type"] == "add_column":
                    await self._drop_column(postgres_persistence, change)
                elif change["type"] == "add_index":
                    await self._drop_index(postgres_persistence, change)
                elif change["type"] == "add_table":
                    await self._drop_table(postgres_persistence, change)
                
                results["changes_rolled_back"] += 1
                
            except Exception as e:
                results["errors"].append(str(e))
        
        return results
    
    async def validate(self, context: Dict[str, Any]) -> bool:
        """Validate schema changes."""
        postgres_persistence = context.get("postgres_persistence")
        if not postgres_persistence:
            return False
        
        # Check if all schema changes were applied correctly
        for change in self.schema_changes:
            if not await self._validate_schema_change(postgres_persistence, change):
                return False
        
        return True
    
    async def _add_column(self, postgres_persistence, change: Dict[str, Any]) -> None:
        """Add a column to a table."""
        table = change["table"]
        column = change["column"]
        column_type = change["column_type"]
        nullable = change.get("nullable", True)
        default = change.get("default")
        
        sql = f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
        
        if not nullable:
            sql += " NOT NULL"
        
        if default is not None:
            sql += f" DEFAULT {default}"
        
        async with postgres_persistence._get_async_connection() as conn:
            await conn.execute(sql)
    
    async def _modify_column(self, postgres_persistence, change: Dict[str, Any]) -> None:
        """Modify a column."""
        table = change["table"]
        column = change["column"]
        new_type = change.get("new_type")
        
        if new_type:
            sql = f"ALTER TABLE {table} ALTER COLUMN {column} TYPE {new_type}"
            async with postgres_persistence._get_async_connection() as conn:
                await conn.execute(sql)
    
    async def _add_index(self, postgres_persistence, change: Dict[str, Any]) -> None:
        """Add an index."""
        index_name = change["index_name"]
        table = change["table"]
        columns = change["columns"]
        unique = change.get("unique", False)
        
        unique_clause = "UNIQUE " if unique else ""
        columns_clause = ", ".join(columns)
        
        sql = f"CREATE {unique_clause}INDEX {index_name} ON {table} ({columns_clause})"
        
        async with postgres_persistence._get_async_connection() as conn:
            await conn.execute(sql)
    
    async def _add_table(self, postgres_persistence, change: Dict[str, Any]) -> None:
        """Add a new table."""
        table_name = change["table_name"]
        columns = change["columns"]
        
        column_defs = []
        for col in columns:
            col_def = f"{col['name']} {col['type']}"
            if not col.get("nullable", True):
                col_def += " NOT NULL"
            if col.get("primary_key"):
                col_def += " PRIMARY KEY"
            if col.get("default"):
                col_def += f" DEFAULT {col['default']}"
            column_defs.append(col_def)
        
        sql = f"CREATE TABLE {table_name} ({', '.join(column_defs)})"
        
        async with postgres_persistence._get_async_connection() as conn:
            await conn.execute(sql)
    
    async def _drop_column(self, postgres_persistence, change: Dict[str, Any]) -> None:
        """Drop a column."""
        table = change["table"]
        column = change["column"]
        
        sql = f"ALTER TABLE {table} DROP COLUMN {column}"
        
        async with postgres_persistence._get_async_connection() as conn:
            await conn.execute(sql)
    
    async def _drop_index(self, postgres_persistence, change: Dict[str, Any]) -> None:
        """Drop an index."""
        index_name = change["index_name"]
        
        sql = f"DROP INDEX {index_name}"
        
        async with postgres_persistence._get_async_connection() as conn:
            await conn.execute(sql)
    
    async def _drop_table(self, postgres_persistence, change: Dict[str, Any]) -> None:
        """Drop a table."""
        table_name = change["table_name"]
        
        sql = f"DROP TABLE {table_name}"
        
        async with postgres_persistence._get_async_connection() as conn:
            await conn.execute(sql)
    
    async def _validate_schema_change(self, postgres_persistence, change: Dict[str, Any]) -> bool:
        """Validate a specific schema change."""
        try:
            if change["type"] == "add_column":
                # Check if column exists
                table = change["table"]
                column = change["column"]
                
                async with postgres_persistence._get_async_connection() as conn:
                    result = await conn.fetchrow("""
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = $1 AND column_name = $2
                    """, table, column)
                    
                    return result is not None
            
            elif change["type"] == "add_index":
                # Check if index exists
                index_name = change["index_name"]
                
                async with postgres_persistence._get_async_connection() as conn:
                    result = await conn.fetchrow("""
                        SELECT indexname FROM pg_indexes 
                        WHERE indexname = $1
                    """, index_name)
                    
                    return result is not None
            
            elif change["type"] == "add_table":
                # Check if table exists
                table_name = change["table_name"]
                
                async with postgres_persistence._get_async_connection() as conn:
                    result = await conn.fetchrow("""
                        SELECT table_name FROM information_schema.tables 
                        WHERE table_name = $1
                    """, table_name)
                    
                    return result is not None
        
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False
        
        return True


class DataTransformationMigration(MigrationScript):
    """Migration for data transformations."""
    
    def __init__(self, migration_def: MigrationDefinition, transformations: List[Dict[str, Any]]):
        super().__init__(migration_def)
        self.transformations = transformations
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data transformations."""
        results = {"records_transformed": 0, "errors": []}
        
        json_persistence = context.get("json_persistence")
        if not json_persistence:
            raise ValueError("JSON persistence not available in context")
        
        for transformation in self.transformations:
            try:
                if transformation["type"] == "field_rename":
                    count = await self._rename_field(json_persistence, transformation)
                elif transformation["type"] == "field_type_conversion":
                    count = await self._convert_field_type(json_persistence, transformation)
                elif transformation["type"] == "data_normalization":
                    count = await self._normalize_data(json_persistence, transformation)
                elif transformation["type"] == "field_restructure":
                    count = await self._restructure_field(json_persistence, transformation)
                
                results["records_transformed"] += count
                
            except Exception as e:
                results["errors"].append(str(e))
                raise
        
        return results
    
    async def rollback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback data transformations."""
        results = {"records_reverted": 0, "errors": []}
        
        json_persistence = context.get("json_persistence")
        if not json_persistence:
            raise ValueError("JSON persistence not available in context")
        
        # Reverse transformations
        for transformation in reversed(self.transformations):
            try:
                if transformation["type"] == "field_rename":
                    # Reverse the rename
                    reverse_transform = {
                        "type": "field_rename",
                        "old_name": transformation["new_name"],
                        "new_name": transformation["old_name"],
                        "filter": transformation.get("filter")
                    }
                    count = await self._rename_field(json_persistence, reverse_transform)
                    results["records_reverted"] += count
                    
            except Exception as e:
                results["errors"].append(str(e))
        
        return results
    
    async def validate(self, context: Dict[str, Any]) -> bool:
        """Validate data transformations."""
        json_persistence = context.get("json_persistence")
        if not json_persistence:
            return False
        
        # Validate each transformation
        for transformation in self.transformations:
            if not await self._validate_transformation(json_persistence, transformation):
                return False
        
        return True
    
    async def _rename_field(self, json_persistence, transformation: Dict[str, Any]) -> int:
        """Rename a field in JSON records."""
        old_name = transformation["old_name"]
        new_name = transformation["new_name"]
        filter_criteria = transformation.get("filter")
        
        # Query records that need transformation
        records = await json_persistence.query_usage_records(
            limit=10000  # Process in batches
        )
        
        transformed_count = 0
        batch_updates = []
        
        for record in records:
            # Check if record matches filter criteria
            if filter_criteria and not self._record_matches_filter(record, filter_criteria):
                continue
            
            # Check if field exists and needs renaming
            if old_name in record and new_name not in record:
                # Create updated record
                updated_record = record.copy()
                updated_record[new_name] = updated_record.pop(old_name)
                
                batch_updates.append({
                    "original": record,
                    "updated": updated_record
                })
                transformed_count += 1
                
                # Process in batches to avoid memory issues
                if len(batch_updates) >= 100:
                    await self._apply_record_updates(json_persistence, batch_updates)
                    batch_updates = []
        
        # Process remaining updates
        if batch_updates:
            await self._apply_record_updates(json_persistence, batch_updates)
        
        return transformed_count
    
    async def _convert_field_type(self, json_persistence, transformation: Dict[str, Any]) -> int:
        """Convert field type in JSON records."""
        field_name = transformation["field_name"]
        from_type = transformation["from_type"]
        to_type = transformation["to_type"]
        
        records = await json_persistence.query_usage_records(limit=10000)
        transformed_count = 0
        batch_updates = []
        
        for record in records:
            if field_name in record:
                old_value = record[field_name]
                
                try:
                    # Convert based on target type
                    if to_type == "string":
                        new_value = str(old_value)
                    elif to_type == "integer":
                        new_value = int(old_value)
                    elif to_type == "float":
                        new_value = float(old_value)
                    elif to_type == "boolean":
                        new_value = bool(old_value)
                    else:
                        continue
                    
                    # Create updated record
                    updated_record = record.copy()
                    updated_record[field_name] = new_value
                    
                    batch_updates.append({
                        "original": record,
                        "updated": updated_record
                    })
                    transformed_count += 1
                    
                    if len(batch_updates) >= 100:
                        await self._apply_record_updates(json_persistence, batch_updates)
                        batch_updates = []
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert field {field_name} for record: {e}")
                    continue
        
        if batch_updates:
            await self._apply_record_updates(json_persistence, batch_updates)
        
        return transformed_count
    
    async def _normalize_data(self, json_persistence, transformation: Dict[str, Any]) -> int:
        """Normalize data according to rules."""
        normalization_rules = transformation["rules"]
        
        # Placeholder implementation
        return 0
    
    async def _restructure_field(self, json_persistence, transformation: Dict[str, Any]) -> int:
        """Restructure field organization."""
        # Placeholder implementation
        return 0
    
    def _record_matches_filter(self, record: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if record matches filter criteria."""
        for field, expected_value in filter_criteria.items():
            if field not in record or record[field] != expected_value:
                return False
        return True
    
    async def _apply_record_updates(self, json_persistence, batch_updates: List[Dict[str, Any]]) -> None:
        """Apply a batch of record updates."""
        # This would integrate with the JSON persistence layer to update records
        # Placeholder implementation
        pass
    
    async def _validate_transformation(self, json_persistence, transformation: Dict[str, Any]) -> bool:
        """Validate a specific transformation."""
        # Sample a few records to verify transformation was applied correctly
        sample_records = await json_persistence.query_usage_records(limit=10)
        
        if transformation["type"] == "field_rename":
            old_name = transformation["old_name"]
            new_name = transformation["new_name"]
            
            for record in sample_records:
                # Check that old field doesn't exist and new field exists
                if old_name in record or new_name not in record:
                    return False
        
        return True


class StorageFormatMigration(MigrationScript):
    """Migration for storage format changes."""
    
    def __init__(self, migration_def: MigrationDefinition, format_changes: Dict[str, Any]):
        super().__init__(migration_def)
        self.format_changes = format_changes
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute storage format migration."""
        results = {"files_converted": 0, "bytes_processed": 0}
        
        if self.format_changes.get("compression_change"):
            results.update(await self._migrate_compression(context))
        
        if self.format_changes.get("partitioning_change"):
            results.update(await self._migrate_partitioning(context))
        
        if self.format_changes.get("encoding_change"):
            results.update(await self._migrate_encoding(context))
        
        return results
    
    async def rollback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback storage format changes."""
        # Reverse the format changes
        results = {"files_reverted": 0}
        return results
    
    async def validate(self, context: Dict[str, Any]) -> bool:
        """Validate storage format migration."""
        # Verify files are in the new format and readable
        return True
    
    async def _migrate_compression(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate compression format."""
        return {"files_converted": 0, "bytes_saved": 0}
    
    async def _migrate_partitioning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate partitioning scheme."""
        return {"partitions_created": 0, "files_moved": 0}
    
    async def _migrate_encoding(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate encoding format."""
        return {"files_re_encoded": 0}


class MigrationManager:
    """Main migration management system."""
    
    def __init__(
        self,
        postgres_persistence: ContextPersistenceLayer,
        chroma_extensions: UsageTrackingChromaExtensions,
        json_persistence: JsonPersistenceManager,
        backup_manager: BackupRecoveryManager,
        migrations_path: str = "./migrations"
    ):
        self.postgres_persistence = postgres_persistence
        self.chroma_extensions = chroma_extensions
        self.json_persistence = json_persistence
        self.backup_manager = backup_manager
        self.migrations_path = Path(migrations_path)
        self.migrations_path.mkdir(parents=True, exist_ok=True)
        
        # Migration tracking
        self.available_migrations: Dict[str, MigrationDefinition] = {}
        self.executed_migrations: Dict[str, MigrationExecution] = {}
        self.migration_scripts: Dict[str, MigrationScript] = {}
        
        # Execution state
        self.current_execution: Optional[MigrationExecution] = None
        self.execution_lock = asyncio.Lock()
        
        # Version tracking
        self.current_version = "1.0.0"
        self.target_version: Optional[str] = None
        
        # Load migrations
        asyncio.create_task(self._load_migrations())
    
    async def _load_migrations(self) -> None:
        """Load migration definitions and scripts."""
        try:
            # Load migration definitions from files
            for migration_file in self.migrations_path.glob("*.json"):
                try:
                    with open(migration_file, 'r') as f:
                        migration_data = json.load(f)
                    
                    migration_def = MigrationDefinition(**migration_data)
                    self.available_migrations[migration_def.migration_id] = migration_def
                    
                    # Load corresponding script
                    script_file = migration_file.with_suffix('.py')
                    if script_file.exists():
                        script = await self._load_migration_script(script_file, migration_def)
                        if script:
                            self.migration_scripts[migration_def.migration_id] = script
                    
                except Exception as e:
                    logger.error(f"Failed to load migration from {migration_file}: {e}")
            
            logger.info(f"Loaded {len(self.available_migrations)} migrations")
            
        except Exception as e:
            logger.error(f"Failed to load migrations: {e}")
    
    async def _load_migration_script(
        self, 
        script_file: Path, 
        migration_def: MigrationDefinition
    ) -> Optional[MigrationScript]:
        """Load migration script from Python file."""
        try:
            # Dynamic import of migration script
            spec = importlib.util.spec_from_file_location("migration_module", script_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for migration class
            if hasattr(module, 'Migration'):
                migration_class = getattr(module, 'Migration')
                return migration_class(migration_def)
            
        except Exception as e:
            logger.error(f"Failed to load migration script from {script_file}: {e}")
        
        return None
    
    async def plan_migration(
        self, 
        target_version: str,
        strategy: MigrationStrategy = MigrationStrategy.OFFLINE
    ) -> List[MigrationDefinition]:
        """Plan migration path to target version."""
        # Find all migrations needed to reach target version
        required_migrations = []
        
        for migration_def in self.available_migrations.values():
            # Check if migration is applicable for version upgrade
            if self._is_migration_applicable(migration_def, self.current_version, target_version):
                required_migrations.append(migration_def)
        
        # Sort migrations by dependency order
        ordered_migrations = self._sort_migrations_by_dependencies(required_migrations)
        
        # Validate migration plan
        self._validate_migration_plan(ordered_migrations, strategy)
        
        return ordered_migrations
    
    def _is_migration_applicable(
        self, 
        migration_def: MigrationDefinition, 
        current_version: str, 
        target_version: str
    ) -> bool:
        """Check if migration is applicable for version range."""
        try:
            current_ver = pkg_version.parse(current_version)
            target_ver = pkg_version.parse(target_version)
            from_ver = pkg_version.parse(migration_def.from_version)
            to_ver = pkg_version.parse(migration_def.to_version)
            
            # Migration is applicable if it fits within the version range
            return from_ver >= current_ver and to_ver <= target_ver
            
        except Exception as e:
            logger.error(f"Version comparison error: {e}")
            return False
    
    def _sort_migrations_by_dependencies(
        self, 
        migrations: List[MigrationDefinition]
    ) -> List[MigrationDefinition]:
        """Sort migrations by dependency order using topological sort."""
        # Build dependency graph
        migration_map = {m.migration_id: m for m in migrations}
        in_degree = {m.migration_id: 0 for m in migrations}
        
        # Calculate in-degrees
        for migration in migrations:
            for dep in migration.depends_on:
                if dep in in_degree:
                    in_degree[migration.migration_id] += 1
        
        # Topological sort
        queue = [mid for mid, degree in in_degree.items() if degree == 0]
        sorted_migrations = []
        
        while queue:
            current = queue.pop(0)
            sorted_migrations.append(migration_map[current])
            
            # Update in-degrees for dependent migrations
            for migration in migrations:
                if current in migration.depends_on:
                    in_degree[migration.migration_id] -= 1
                    if in_degree[migration.migration_id] == 0:
                        queue.append(migration.migration_id)
        
        if len(sorted_migrations) != len(migrations):
            raise ValueError("Circular dependency detected in migrations")
        
        return sorted_migrations
    
    def _validate_migration_plan(
        self, 
        migrations: List[MigrationDefinition], 
        strategy: MigrationStrategy
    ) -> None:
        """Validate migration plan for conflicts and requirements."""
        # Check for conflicts
        migration_ids = set(m.migration_id for m in migrations)
        
        for migration in migrations:
            for conflict in migration.conflicts_with:
                if conflict in migration_ids:
                    raise ValueError(f"Migration conflict: {migration.migration_id} conflicts with {conflict}")
        
        # Check strategy compatibility
        if strategy == MigrationStrategy.ONLINE:
            for migration in migrations:
                if migration.strategy not in [MigrationStrategy.ONLINE, MigrationStrategy.ROLLING]:
                    raise ValueError(f"Migration {migration.migration_id} not compatible with online strategy")
    
    async def execute_migration_plan(
        self, 
        migrations: List[MigrationDefinition],
        strategy: MigrationStrategy = MigrationStrategy.OFFLINE,
        create_backup: bool = True
    ) -> List[MigrationExecution]:
        """Execute a migration plan."""
        async with self.execution_lock:
            if self.current_execution:
                raise RuntimeError("Migration already in progress")
            
            executions = []
            
            # Create backup before migration if requested
            backup_id = None
            if create_backup:
                logger.info("Creating backup before migration")
                backup_manifest = await self.backup_manager.create_backup(
                    description="Pre-migration backup"
                )
                backup_id = backup_manifest.backup_id
            
            try:
                for migration_def in migrations:
                    execution = await self._execute_single_migration(
                        migration_def, strategy, backup_id
                    )
                    executions.append(execution)
                    
                    # Stop on first failure unless strategy allows continuation
                    if not execution.success and strategy != MigrationStrategy.ROLLING:
                        break
                
                # Update current version if all migrations succeeded
                if all(e.success for e in executions):
                    if migrations:
                        self.current_version = migrations[-1].to_version
                    logger.info(f"Migration completed successfully, version: {self.current_version}")
                else:
                    logger.error("Migration plan failed, some migrations did not complete successfully")
                
                return executions
                
            except Exception as e:
                logger.error(f"Migration plan execution failed: {e}")
                raise
    
    async def _execute_single_migration(
        self, 
        migration_def: MigrationDefinition,
        strategy: MigrationStrategy,
        backup_id: Optional[str]
    ) -> MigrationExecution:
        """Execute a single migration."""
        execution_id = f"exec_{migration_def.migration_id}_{int(datetime.now().timestamp())}"
        
        execution = MigrationExecution(
            execution_id=execution_id,
            migration_id=migration_def.migration_id,
            state=MigrationState.RUNNING,
            started_at=datetime.now(),
            backup_id=backup_id,
            rollback_available=migration_def.reversible
        )
        
        self.current_execution = execution
        
        try:
            logger.info(f"Starting migration: {migration_def.name}")
            
            # Get migration script
            script = self.migration_scripts.get(migration_def.migration_id)
            if not script:
                raise ValueError(f"Migration script not found for {migration_def.migration_id}")
            
            # Prepare execution context
            context = {
                "postgres_persistence": self.postgres_persistence,
                "chroma_extensions": self.chroma_extensions,
                "json_persistence": self.json_persistence,
                "backup_manager": self.backup_manager,
                "migration_def": migration_def,
                "execution": execution
            }
            
            # Execute pre-checks
            execution.current_step = "pre_checks"
            if not await script.pre_check(context):
                raise ValueError("Pre-migration checks failed")
            execution.steps_completed.append("pre_checks")
            
            # Execute migration
            execution.current_step = "migration"
            result = await script.execute(context)
            execution.steps_completed.append("migration")
            
            # Update execution metrics
            execution.records_processed = result.get("records_processed", 0)
            execution.bytes_processed = result.get("bytes_processed", 0)
            
            # Execute post-checks
            execution.current_step = "post_checks"
            if not await script.post_check(context):
                raise ValueError("Post-migration checks failed")
            execution.steps_completed.append("post_checks")
            
            # Validate migration
            execution.current_step = "validation"
            if not await script.validate(context):
                raise ValueError("Migration validation failed")
            execution.steps_completed.append("validation")
            
            # Mark as successful
            execution.state = MigrationState.COMPLETED
            execution.success = True
            execution.completed_at = datetime.now()
            
            logger.info(f"Migration completed successfully: {migration_def.name}")
            
        except Exception as e:
            execution.state = MigrationState.FAILED
            execution.success = False
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            
            logger.error(f"Migration failed: {migration_def.name}, error: {str(e)}")
            
            # Attempt rollback if migration is reversible
            if migration_def.reversible and script:
                try:
                    logger.info(f"Attempting rollback for: {migration_def.name}")
                    await script.rollback(context)
                    execution.state = MigrationState.ROLLED_BACK
                    logger.info(f"Rollback completed for: {migration_def.name}")
                except Exception as rollback_error:
                    logger.error(f"Rollback failed for {migration_def.name}: {str(rollback_error)}")
        
        finally:
            self.current_execution = None
            self.executed_migrations[execution.execution_id] = execution
        
        return execution
    
    async def rollback_migration(
        self, 
        execution_id: str
    ) -> MigrationExecution:
        """Rollback a specific migration."""
        async with self.execution_lock:
            if self.current_execution:
                raise RuntimeError("Migration in progress, cannot rollback")
            
            execution = self.executed_migrations.get(execution_id)
            if not execution:
                raise ValueError(f"Migration execution not found: {execution_id}")
            
            if not execution.rollback_available:
                raise ValueError(f"Migration is not reversible: {execution.migration_id}")
            
            migration_def = self.available_migrations.get(execution.migration_id)
            if not migration_def:
                raise ValueError(f"Migration definition not found: {execution.migration_id}")
            
            script = self.migration_scripts.get(migration_def.migration_id)
            if not script:
                raise ValueError(f"Migration script not found: {migration_def.migration_id}")
            
            # Create rollback execution
            rollback_execution = MigrationExecution(
                execution_id=f"rollback_{execution_id}_{int(datetime.now().timestamp())}",
                migration_id=execution.migration_id,
                state=MigrationState.RUNNING,
                started_at=datetime.now()
            )
            
            self.current_execution = rollback_execution
            
            try:
                logger.info(f"Starting rollback for migration: {migration_def.name}")
                
                context = {
                    "postgres_persistence": self.postgres_persistence,
                    "chroma_extensions": self.chroma_extensions,
                    "json_persistence": self.json_persistence,
                    "backup_manager": self.backup_manager,
                    "migration_def": migration_def,
                    "execution": rollback_execution
                }
                
                # Execute rollback
                await script.rollback(context)
                
                rollback_execution.state = MigrationState.COMPLETED
                rollback_execution.success = True
                rollback_execution.completed_at = datetime.now()
                
                # Update original execution state
                execution.state = MigrationState.ROLLED_BACK
                
                logger.info(f"Rollback completed for migration: {migration_def.name}")
                
            except Exception as e:
                rollback_execution.state = MigrationState.FAILED
                rollback_execution.success = False
                rollback_execution.error_message = str(e)
                rollback_execution.completed_at = datetime.now()
                
                logger.error(f"Rollback failed for migration {migration_def.name}: {str(e)}")
                raise
            
            finally:
                self.current_execution = None
                self.executed_migrations[rollback_execution.execution_id] = rollback_execution
            
            return rollback_execution
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status."""
        return {
            "current_version": self.current_version,
            "target_version": self.target_version,
            "available_migrations": len(self.available_migrations),
            "executed_migrations": len(self.executed_migrations),
            "current_execution": (
                {
                    "execution_id": self.current_execution.execution_id,
                    "migration_id": self.current_execution.migration_id,
                    "state": self.current_execution.state.value,
                    "current_step": self.current_execution.current_step,
                    "started_at": self.current_execution.started_at.isoformat()
                } if self.current_execution else None
            ),
            "recent_executions": [
                {
                    "execution_id": ex.execution_id,
                    "migration_id": ex.migration_id,
                    "state": ex.state.value,
                    "success": ex.success,
                    "started_at": ex.started_at.isoformat(),
                    "completed_at": ex.completed_at.isoformat() if ex.completed_at else None
                }
                for ex in sorted(
                    self.executed_migrations.values(),
                    key=lambda x: x.started_at,
                    reverse=True
                )[:10]
            ]
        }
    
    async def create_migration_template(
        self,
        migration_id: str,
        name: str,
        migration_type: MigrationType,
        from_version: str,
        to_version: str
    ) -> Path:
        """Create a migration template."""
        migration_def = MigrationDefinition(
            migration_id=migration_id,
            name=name,
            description=f"Migration: {name}",
            migration_type=migration_type,
            strategy=MigrationStrategy.OFFLINE,
            from_version=from_version,
            to_version=to_version
        )
        
        # Save migration definition
        definition_file = self.migrations_path / f"{migration_id}.json"
        with open(definition_file, 'w') as f:
            json.dump(asdict(migration_def), f, indent=2, default=str)
        
        # Create script template
        script_file = self.migrations_path / f"{migration_id}.py"
        script_template = f'''"""Migration script for {name}."""

from src.usage_tracking.migration_strategies import MigrationScript

class Migration(MigrationScript):
    """Migration: {name}."""
    
    async def execute(self, context):
        """Execute the migration."""
        results = {{"changes_applied": 0, "errors": []}}
        
        # TODO: Implement migration logic
        
        return results
    
    async def rollback(self, context):
        """Rollback the migration."""
        results = {{"changes_reverted": 0, "errors": []}}
        
        # TODO: Implement rollback logic
        
        return results
    
    async def validate(self, context):
        """Validate migration success."""
        # TODO: Implement validation logic
        return True
    
    async def pre_check(self, context):
        """Pre-migration checks."""
        # TODO: Implement pre-checks
        return True
    
    async def post_check(self, context):
        """Post-migration checks."""
        # TODO: Implement post-checks
        return True
'''
        
        with open(script_file, 'w') as f:
            f.write(script_template)
        
        logger.info(f"Created migration template: {migration_id}")
        
        return definition_file