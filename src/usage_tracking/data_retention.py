"""Comprehensive data retention policies and cleanup strategies for usage tracking."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path
import json

from ..context.persistence import ContextPersistenceLayer
from .chroma_extensions import UsageTrackingChromaExtensions
from .json_persistence import JsonPersistenceManager
from .state_synchronization import StateSynchronizationManager
from config.logging_config import get_logger

logger = get_logger(__name__)


class RetentionPolicy(Enum):
    """Data retention policy types."""
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based"
    COUNT_BASED = "count_based"
    CONDITIONAL = "conditional"
    COMPLIANCE = "compliance"


class DataCategory(Enum):
    """Categories of data for retention policies."""
    RAW_USAGE_DATA = "raw_usage_data"
    AGGREGATED_METRICS = "aggregated_metrics"
    ANALYTICS_INSIGHTS = "analytics_insights"
    USER_PROFILES = "user_profiles"
    AUDIT_LOGS = "audit_logs"
    BACKUP_DATA = "backup_data"
    TEMP_DATA = "temp_data"


class RetentionAction(Enum):
    """Actions to take when retention policy triggers."""
    DELETE = "delete"
    ARCHIVE = "archive"
    COMPRESS = "compress"
    AGGREGATE = "aggregate"
    EXPORT = "export"
    NOTIFY = "notify"


@dataclass
class RetentionRule:
    """Defines a data retention rule."""
    rule_id: str
    name: str
    description: str
    data_category: DataCategory
    policy_type: RetentionPolicy
    
    # Time-based parameters
    retention_days: Optional[int] = None
    
    # Size-based parameters
    max_size_gb: Optional[float] = None
    
    # Count-based parameters
    max_records: Optional[int] = None
    
    # Conditional parameters
    conditions: Optional[Dict[str, Any]] = None
    
    # Actions to take
    primary_action: RetentionAction = RetentionAction.DELETE
    fallback_actions: List[RetentionAction] = None
    
    # Scheduling
    check_interval_hours: int = 24
    enabled: bool = True
    
    # Compliance requirements
    compliance_requirements: List[str] = None
    
    # Backup before action
    backup_before_action: bool = False
    
    def __post_init__(self):
        """Initialize default values."""
        if self.fallback_actions is None:
            self.fallback_actions = []
        if self.compliance_requirements is None:
            self.compliance_requirements = []


@dataclass
class RetentionExecution:
    """Represents an execution of retention policies."""
    execution_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    rules_executed: List[str]
    results: Dict[str, Any]
    errors: List[str]
    total_records_processed: int
    total_size_freed_bytes: int
    success: bool


class DataRetentionManager:
    """Comprehensive data retention and cleanup manager."""
    
    def __init__(
        self,
        postgres_persistence: ContextPersistenceLayer,
        chroma_extensions: UsageTrackingChromaExtensions,
        json_persistence: JsonPersistenceManager,
        sync_manager: StateSynchronizationManager,
        retention_config_path: Optional[str] = None
    ):
        self.postgres_persistence = postgres_persistence
        self.chroma_extensions = chroma_extensions
        self.json_persistence = json_persistence
        self.sync_manager = sync_manager
        
        # Configuration
        self.config_path = retention_config_path or "./data/retention_config.json"
        self.archive_path = Path("./data/archives")
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # Retention rules
        self.retention_rules: Dict[str, RetentionRule] = {}
        
        # Execution tracking
        self.execution_history: List[RetentionExecution] = []
        self.current_execution: Optional[RetentionExecution] = None
        
        # Background tasks
        self._retention_tasks: List[asyncio.Task] = []
        self._running = False
        
        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_records_cleaned": 0,
            "total_size_freed_gb": 0.0,
            "avg_execution_time_minutes": 0.0,
            "last_execution": None
        }
        
        # Thread safety
        self.execution_lock = asyncio.Lock()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "retention_started": [],
            "retention_completed": [],
            "retention_failed": [],
            "rule_executed": [],
            "data_archived": [],
            "data_deleted": []
        }
        
        # Initialize default rules and load configuration
        self._initialize_default_rules()
        self._load_retention_configuration()
        
        logger.info("Data retention manager initialized")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default retention rules."""
        default_rules = [
            RetentionRule(
                rule_id="raw_usage_90d",
                name="Raw Usage Data - 90 Days",
                description="Delete raw usage data older than 90 days",
                data_category=DataCategory.RAW_USAGE_DATA,
                policy_type=RetentionPolicy.TIME_BASED,
                retention_days=90,
                primary_action=RetentionAction.ARCHIVE,
                fallback_actions=[RetentionAction.DELETE],
                backup_before_action=True
            ),
            
            RetentionRule(
                rule_id="aggregated_metrics_1y",
                name="Aggregated Metrics - 1 Year",
                description="Keep aggregated metrics for 1 year",
                data_category=DataCategory.AGGREGATED_METRICS,
                policy_type=RetentionPolicy.TIME_BASED,
                retention_days=365,
                primary_action=RetentionAction.COMPRESS
            ),
            
            RetentionRule(
                rule_id="analytics_insights_6m",
                name="Analytics Insights - 6 Months",
                description="Keep analytics insights for 6 months",
                data_category=DataCategory.ANALYTICS_INSIGHTS,
                policy_type=RetentionPolicy.TIME_BASED,
                retention_days=180,
                primary_action=RetentionAction.DELETE
            ),
            
            RetentionRule(
                rule_id="audit_logs_compliance",
                name="Audit Logs - Compliance",
                description="Keep audit logs for compliance requirements",
                data_category=DataCategory.AUDIT_LOGS,
                policy_type=RetentionPolicy.COMPLIANCE,
                retention_days=2555,  # 7 years for financial compliance
                primary_action=RetentionAction.ARCHIVE,
                compliance_requirements=["SOX", "GDPR", "PCI-DSS"]
            ),
            
            RetentionRule(
                rule_id="temp_data_1d",
                name="Temporary Data - 1 Day",
                description="Clean up temporary data daily",
                data_category=DataCategory.TEMP_DATA,
                policy_type=RetentionPolicy.TIME_BASED,
                retention_days=1,
                primary_action=RetentionAction.DELETE,
                check_interval_hours=6
            ),
            
            RetentionRule(
                rule_id="backup_data_30d",
                name="Backup Data - 30 Days",
                description="Keep backup data for 30 days",
                data_category=DataCategory.BACKUP_DATA,
                policy_type=RetentionPolicy.TIME_BASED,
                retention_days=30,
                primary_action=RetentionAction.DELETE
            ),
            
            RetentionRule(
                rule_id="large_files_size_limit",
                name="Large Files - Size Limit",
                description="Compress or archive files larger than 1GB",
                data_category=DataCategory.RAW_USAGE_DATA,
                policy_type=RetentionPolicy.SIZE_BASED,
                max_size_gb=1.0,
                primary_action=RetentionAction.COMPRESS,
                fallback_actions=[RetentionAction.ARCHIVE]
            ),
            
            RetentionRule(
                rule_id="high_volume_users",
                name="High Volume Users - Count Based",
                description="Limit records per user to prevent storage abuse",
                data_category=DataCategory.RAW_USAGE_DATA,
                policy_type=RetentionPolicy.COUNT_BASED,
                max_records=100000,
                primary_action=RetentionAction.AGGREGATE,
                conditions={"per_user_limit": True}
            )
        ]
        
        for rule in default_rules:
            self.retention_rules[rule.rule_id] = rule
    
    def _load_retention_configuration(self) -> None:
        """Load retention configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Load custom rules
                for rule_data in config_data.get("custom_rules", []):
                    rule = RetentionRule(
                        rule_id=rule_data["rule_id"],
                        name=rule_data["name"],
                        description=rule_data["description"],
                        data_category=DataCategory(rule_data["data_category"]),
                        policy_type=RetentionPolicy(rule_data["policy_type"]),
                        retention_days=rule_data.get("retention_days"),
                        max_size_gb=rule_data.get("max_size_gb"),
                        max_records=rule_data.get("max_records"),
                        conditions=rule_data.get("conditions"),
                        primary_action=RetentionAction(rule_data["primary_action"]),
                        fallback_actions=[RetentionAction(a) for a in rule_data.get("fallback_actions", [])],
                        check_interval_hours=rule_data.get("check_interval_hours", 24),
                        enabled=rule_data.get("enabled", True),
                        compliance_requirements=rule_data.get("compliance_requirements", []),
                        backup_before_action=rule_data.get("backup_before_action", False)
                    )
                    self.retention_rules[rule.rule_id] = rule
                
                # Override default rules if specified
                for rule_id, overrides in config_data.get("rule_overrides", {}).items():
                    if rule_id in self.retention_rules:
                        rule = self.retention_rules[rule_id]
                        for key, value in overrides.items():
                            if hasattr(rule, key):
                                setattr(rule, key, value)
                
                logger.info("Retention configuration loaded", 
                           custom_rules=len(config_data.get("custom_rules", [])),
                           overrides=len(config_data.get("rule_overrides", {})))
        
        except Exception as e:
            logger.warning("Failed to load retention configuration", error=str(e))
    
    async def start(self) -> None:
        """Start the data retention manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start background retention tasks for each rule
        for rule in self.retention_rules.values():
            if rule.enabled:
                task = asyncio.create_task(self._run_retention_schedule(rule))
                self._retention_tasks.append(task)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_retention_health())
        self._retention_tasks.append(monitor_task)
        
        logger.info("Data retention manager started", 
                   active_rules=len([r for r in self.retention_rules.values() if r.enabled]))
    
    async def stop(self) -> None:
        """Stop the data retention manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all background tasks
        for task in self._retention_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._retention_tasks:
            await asyncio.gather(*self._retention_tasks, return_exceptions=True)
        
        # Complete any ongoing execution
        if self.current_execution:
            await self._complete_execution(False, "Manager stopped")
        
        logger.info("Data retention manager stopped")
    
    async def execute_retention_policies(
        self, 
        rule_ids: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> RetentionExecution:
        """Execute retention policies manually or for specific rules."""
        async with self.execution_lock:
            if self.current_execution:
                raise RuntimeError("Retention execution already in progress")
            
            # Determine which rules to execute
            rules_to_execute = []
            if rule_ids:
                rules_to_execute = [
                    self.retention_rules[rule_id] for rule_id in rule_ids 
                    if rule_id in self.retention_rules and self.retention_rules[rule_id].enabled
                ]
            else:
                rules_to_execute = [rule for rule in self.retention_rules.values() if rule.enabled]
            
            if not rules_to_execute:
                raise ValueError("No enabled rules found to execute")
            
            # Create execution record
            execution = RetentionExecution(
                execution_id=f"retention_{int(datetime.now().timestamp())}",
                started_at=datetime.now(),
                completed_at=None,
                rules_executed=[rule.rule_id for rule in rules_to_execute],
                results={},
                errors=[],
                total_records_processed=0,
                total_size_freed_bytes=0,
                success=False
            )
            
            self.current_execution = execution
            
            try:
                await self._emit_event("retention_started", execution)
                
                logger.info("Starting retention execution", 
                           execution_id=execution.execution_id,
                           rules_count=len(rules_to_execute),
                           dry_run=dry_run)
                
                # Execute each rule
                for rule in rules_to_execute:
                    try:
                        rule_result = await self._execute_retention_rule(rule, dry_run)
                        execution.results[rule.rule_id] = rule_result
                        execution.total_records_processed += rule_result.get("records_processed", 0)
                        execution.total_size_freed_bytes += rule_result.get("size_freed_bytes", 0)
                        
                        await self._emit_event("rule_executed", {"rule": rule, "result": rule_result})
                        
                    except Exception as e:
                        error_msg = f"Failed to execute rule {rule.rule_id}: {str(e)}"
                        execution.errors.append(error_msg)
                        logger.error("Rule execution failed", rule_id=rule.rule_id, error=str(e))
                
                # Mark execution as successful if no critical errors
                execution.success = len(execution.errors) == 0
                
                await self._complete_execution(execution.success)
                
                # Update metrics
                self.metrics["total_executions"] += 1
                if execution.success:
                    self.metrics["successful_executions"] += 1
                else:
                    self.metrics["failed_executions"] += 1
                
                self.metrics["total_records_cleaned"] += execution.total_records_processed
                self.metrics["total_size_freed_gb"] += execution.total_size_freed_bytes / (1024**3)
                
                execution_time = (execution.completed_at - execution.started_at).total_seconds() / 60
                self.metrics["avg_execution_time_minutes"] = (
                    self.metrics["avg_execution_time_minutes"] * 0.9 + execution_time * 0.1
                )
                self.metrics["last_execution"] = execution.completed_at.isoformat()
                
                return execution
                
            except Exception as e:
                await self._complete_execution(False, str(e))
                raise
    
    async def _execute_retention_rule(self, rule: RetentionRule, dry_run: bool) -> Dict[str, Any]:
        """Execute a specific retention rule."""
        logger.info("Executing retention rule", 
                   rule_id=rule.rule_id, 
                   rule_name=rule.name,
                   dry_run=dry_run)
        
        result = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "started_at": datetime.now().isoformat(),
            "records_identified": 0,
            "records_processed": 0,
            "size_freed_bytes": 0,
            "actions_taken": [],
            "errors": []
        }
        
        try:
            # Identify data that matches retention criteria
            matching_data = await self._identify_matching_data(rule)
            result["records_identified"] = len(matching_data)
            
            if not matching_data:
                logger.debug("No data found matching retention rule", rule_id=rule.rule_id)
                return result
            
            # Execute primary action
            if not dry_run:
                action_result = await self._execute_retention_action(
                    rule.primary_action, matching_data, rule
                )
                result["records_processed"] += action_result["records_processed"]
                result["size_freed_bytes"] += action_result["size_freed_bytes"]
                result["actions_taken"].append({
                    "action": rule.primary_action.value,
                    "result": action_result
                })
                
                # Execute fallback actions if primary action failed partially
                if (action_result["records_processed"] < len(matching_data) and 
                    rule.fallback_actions):
                    
                    remaining_data = [
                        item for item in matching_data 
                        if item["id"] not in action_result.get("processed_ids", [])
                    ]
                    
                    for fallback_action in rule.fallback_actions:
                        if remaining_data:
                            fallback_result = await self._execute_retention_action(
                                fallback_action, remaining_data, rule
                            )
                            result["records_processed"] += fallback_result["records_processed"]
                            result["size_freed_bytes"] += fallback_result["size_freed_bytes"]
                            result["actions_taken"].append({
                                "action": fallback_action.value,
                                "result": fallback_result
                            })
                            
                            # Update remaining data
                            remaining_data = [
                                item for item in remaining_data
                                if item["id"] not in fallback_result.get("processed_ids", [])
                            ]
            else:
                # Dry run - just simulate
                result["actions_taken"].append({
                    "action": rule.primary_action.value,
                    "simulated": True,
                    "would_process": len(matching_data)
                })
            
            result["completed_at"] = datetime.now().isoformat()
            
            logger.info("Retention rule executed successfully", 
                       rule_id=rule.rule_id,
                       records_processed=result["records_processed"],
                       size_freed_mb=result["size_freed_bytes"] / (1024**2))
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to execute retention rule {rule.rule_id}: {str(e)}"
            result["errors"].append(error_msg)
            logger.error("Retention rule execution failed", rule_id=rule.rule_id, error=str(e))
            raise
    
    async def _identify_matching_data(self, rule: RetentionRule) -> List[Dict[str, Any]]:
        """Identify data that matches retention rule criteria."""
        matching_data = []
        
        try:
            if rule.data_category == DataCategory.RAW_USAGE_DATA:
                matching_data = await self._find_raw_usage_data(rule)
            
            elif rule.data_category == DataCategory.AGGREGATED_METRICS:
                matching_data = await self._find_aggregated_metrics(rule)
            
            elif rule.data_category == DataCategory.ANALYTICS_INSIGHTS:
                matching_data = await self._find_analytics_insights(rule)
            
            elif rule.data_category == DataCategory.AUDIT_LOGS:
                matching_data = await self._find_audit_logs(rule)
            
            elif rule.data_category == DataCategory.BACKUP_DATA:
                matching_data = await self._find_backup_data(rule)
            
            elif rule.data_category == DataCategory.TEMP_DATA:
                matching_data = await self._find_temp_data(rule)
            
            logger.debug("Data matching retention rule identified", 
                        rule_id=rule.rule_id,
                        matches=len(matching_data))
            
            return matching_data
            
        except Exception as e:
            logger.error("Failed to identify matching data", rule_id=rule.rule_id, error=str(e))
            return []
    
    async def _find_raw_usage_data(self, rule: RetentionRule) -> List[Dict[str, Any]]:
        """Find raw usage data matching retention criteria."""
        matching_data = []
        
        # Calculate cutoff date for time-based policies
        cutoff_date = None
        if rule.policy_type == RetentionPolicy.TIME_BASED and rule.retention_days:
            cutoff_date = datetime.now() - timedelta(days=rule.retention_days)
        
        # Query JSON files for old usage records
        try:
            if cutoff_date:
                json_records = await self.json_persistence.query_usage_records(
                    date_range=(datetime.min, cutoff_date),
                    limit=10000
                )
                
                for record in json_records:
                    matching_data.append({
                        "id": record.get("request_id"),
                        "type": "json_record",
                        "backend": "json_files",
                        "data": record,
                        "timestamp": record.get("timestamp"),
                        "size_bytes": len(json.dumps(record).encode())
                    })
        
        except Exception as e:
            logger.error("Failed to query JSON files for retention", error=str(e))
        
        # Query PostgreSQL for old usage contexts
        try:
            if cutoff_date:
                from ..context.models import ContextQuery, ContextType
                
                query = ContextQuery(
                    context_types=[ContextType.PROVIDER_SPECIFIC],
                    created_before=cutoff_date,
                    limit=10000
                )
                
                # Add filter for usage tracking contexts
                contexts = await self.postgres_persistence.query_contexts(query)
                usage_contexts = [
                    ctx for ctx in contexts 
                    if ctx.metadata.get("usage_tracking") == True
                ]
                
                for context in usage_contexts:
                    matching_data.append({
                        "id": context.context_id,
                        "type": "postgres_context",
                        "backend": "postgresql",
                        "data": context,
                        "timestamp": context.created_at.isoformat(),
                        "size_bytes": len(json.dumps(context.data).encode())
                    })
        
        except Exception as e:
            logger.error("Failed to query PostgreSQL for retention", error=str(e))
        
        # Apply additional filters
        if rule.policy_type == RetentionPolicy.SIZE_BASED and rule.max_size_gb:
            max_size_bytes = rule.max_size_gb * 1024**3
            matching_data = [
                item for item in matching_data 
                if item.get("size_bytes", 0) > max_size_bytes
            ]
        
        elif rule.policy_type == RetentionPolicy.COUNT_BASED and rule.max_records:
            # Sort by timestamp and take oldest records beyond limit
            matching_data.sort(key=lambda x: x.get("timestamp", ""))
            if len(matching_data) > rule.max_records:
                matching_data = matching_data[:-rule.max_records]
            else:
                matching_data = []
        
        return matching_data
    
    async def _find_aggregated_metrics(self, rule: RetentionRule) -> List[Dict[str, Any]]:
        """Find aggregated metrics matching retention criteria."""
        # This would query metrics storage for old aggregated data
        return []
    
    async def _find_analytics_insights(self, rule: RetentionRule) -> List[Dict[str, Any]]:
        """Find analytics insights matching retention criteria."""
        matching_data = []
        
        # Query ChromaDB for old analytics data
        try:
            cutoff_date = None
            if rule.policy_type == RetentionPolicy.TIME_BASED and rule.retention_days:
                cutoff_date = datetime.now() - timedelta(days=rule.retention_days)
            
            if cutoff_date:
                # Get analytics statistics to identify old data
                stats = await self.chroma_extensions.get_analytics_statistics()
                
                # This would involve querying each collection for old documents
                # Implementation would depend on ChromaDB's filtering capabilities
                
        except Exception as e:
            logger.error("Failed to query ChromaDB for retention", error=str(e))
        
        return matching_data
    
    async def _find_audit_logs(self, rule: RetentionRule) -> List[Dict[str, Any]]:
        """Find audit logs matching retention criteria."""
        # This would query audit log storage
        return []
    
    async def _find_backup_data(self, rule: RetentionRule) -> List[Dict[str, Any]]:
        """Find backup data matching retention criteria."""
        matching_data = []
        
        try:
            cutoff_date = None
            if rule.policy_type == RetentionPolicy.TIME_BASED and rule.retention_days:
                cutoff_date = datetime.now() - timedelta(days=rule.retention_days)
            
            # Find old backup files
            backup_dirs = [
                self.archive_path,
                Path("./data/backups"),
                Path("./data/usage_tracking").parent / "backup_*"
            ]
            
            for backup_dir in backup_dirs:
                if backup_dir.exists() and backup_dir.is_dir():
                    for backup_file in backup_dir.rglob("*"):
                        if backup_file.is_file():
                            file_stat = backup_file.stat()
                            file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
                            
                            if cutoff_date and file_mtime < cutoff_date:
                                matching_data.append({
                                    "id": str(backup_file),
                                    "type": "backup_file",
                                    "backend": "filesystem",
                                    "data": {"path": str(backup_file)},
                                    "timestamp": file_mtime.isoformat(),
                                    "size_bytes": file_stat.st_size
                                })
        
        except Exception as e:
            logger.error("Failed to find backup data for retention", error=str(e))
        
        return matching_data
    
    async def _find_temp_data(self, rule: RetentionRule) -> List[Dict[str, Any]]:
        """Find temporary data matching retention criteria."""
        matching_data = []
        
        try:
            cutoff_date = None
            if rule.policy_type == RetentionPolicy.TIME_BASED and rule.retention_days:
                cutoff_date = datetime.now() - timedelta(days=rule.retention_days)
            
            # Find temporary files
            temp_dirs = [
                Path("./data/temp"),
                Path("./data/cache"),
                Path("/tmp/mdmai_*")
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for temp_file in temp_dir.rglob("*"):
                        if temp_file.is_file():
                            file_stat = temp_file.stat()
                            file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
                            
                            if cutoff_date and file_mtime < cutoff_date:
                                matching_data.append({
                                    "id": str(temp_file),
                                    "type": "temp_file",
                                    "backend": "filesystem",
                                    "data": {"path": str(temp_file)},
                                    "timestamp": file_mtime.isoformat(),
                                    "size_bytes": file_stat.st_size
                                })
        
        except Exception as e:
            logger.error("Failed to find temp data for retention", error=str(e))
        
        return matching_data
    
    async def _execute_retention_action(
        self, 
        action: RetentionAction, 
        data_items: List[Dict[str, Any]], 
        rule: RetentionRule
    ) -> Dict[str, Any]:
        """Execute a retention action on data items."""
        result = {
            "action": action.value,
            "records_processed": 0,
            "records_failed": 0,
            "size_freed_bytes": 0,
            "processed_ids": [],
            "errors": []
        }
        
        try:
            if action == RetentionAction.DELETE:
                result = await self._delete_data(data_items, rule)
            
            elif action == RetentionAction.ARCHIVE:
                result = await self._archive_data(data_items, rule)
            
            elif action == RetentionAction.COMPRESS:
                result = await self._compress_data(data_items, rule)
            
            elif action == RetentionAction.AGGREGATE:
                result = await self._aggregate_data(data_items, rule)
            
            elif action == RetentionAction.EXPORT:
                result = await self._export_data(data_items, rule)
            
            await self._emit_event("data_" + action.value.lower() + "d", {
                "rule": rule,
                "result": result,
                "data_count": len(data_items)
            })
            
            return result
            
        except Exception as e:
            logger.error("Failed to execute retention action", 
                        action=action.value,
                        rule_id=rule.rule_id, 
                        error=str(e))
            result["errors"].append(str(e))
            return result
    
    async def _delete_data(self, data_items: List[Dict[str, Any]], rule: RetentionRule) -> Dict[str, Any]:
        """Delete data items."""
        result = {
            "action": "delete",
            "records_processed": 0,
            "records_failed": 0,
            "size_freed_bytes": 0,
            "processed_ids": [],
            "errors": []
        }
        
        for item in data_items:
            try:
                backend = item.get("backend")
                item_type = item.get("type")
                
                if backend == "json_files" and item_type == "json_record":
                    # For JSON files, we can't delete individual records easily
                    # This would require more sophisticated file management
                    pass
                
                elif backend == "postgresql" and item_type == "postgres_context":
                    # Delete context from PostgreSQL
                    context_id = item["id"]
                    await self.postgres_persistence.delete_context(
                        context_id, hard_delete=True
                    )
                
                elif backend == "chromadb":
                    # Delete from ChromaDB collections
                    # Implementation would depend on collection structure
                    pass
                
                elif backend == "filesystem":
                    # Delete file from filesystem
                    file_path = Path(item["data"]["path"])
                    if file_path.exists():
                        file_path.unlink()
                
                result["records_processed"] += 1
                result["size_freed_bytes"] += item.get("size_bytes", 0)
                result["processed_ids"].append(item["id"])
                
            except Exception as e:
                result["records_failed"] += 1
                result["errors"].append(f"Failed to delete {item['id']}: {str(e)}")
        
        return result
    
    async def _archive_data(self, data_items: List[Dict[str, Any]], rule: RetentionRule) -> Dict[str, Any]:
        """Archive data items."""
        result = {
            "action": "archive",
            "records_processed": 0,
            "records_failed": 0,
            "size_freed_bytes": 0,
            "processed_ids": [],
            "archive_path": None,
            "errors": []
        }
        
        try:
            # Create archive directory
            archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = self.archive_path / f"{rule.rule_id}_{archive_timestamp}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            result["archive_path"] = str(archive_dir)
            
            # Archive each item
            archived_items = []
            
            for item in data_items:
                try:
                    # Create archive entry
                    archive_entry = {
                        "original_id": item["id"],
                        "original_backend": item.get("backend"),
                        "original_type": item.get("type"),
                        "archived_at": datetime.now().isoformat(),
                        "data": item["data"],
                        "metadata": {
                            "size_bytes": item.get("size_bytes", 0),
                            "timestamp": item.get("timestamp"),
                            "rule_id": rule.rule_id
                        }
                    }
                    
                    archived_items.append(archive_entry)
                    result["processed_ids"].append(item["id"])
                    
                except Exception as e:
                    result["records_failed"] += 1
                    result["errors"].append(f"Failed to archive {item['id']}: {str(e)}")
            
            # Save archive file
            if archived_items:
                archive_file = archive_dir / "archived_data.json.gz"
                
                import gzip
                archive_data = json.dumps({
                    "rule_id": rule.rule_id,
                    "archived_at": datetime.now().isoformat(),
                    "total_items": len(archived_items),
                    "items": archived_items
                }, indent=2)
                
                with gzip.open(archive_file, 'wt') as f:
                    f.write(archive_data)
                
                result["records_processed"] = len(archived_items)
                
                # After successful archiving, delete original data
                if rule.primary_action == RetentionAction.ARCHIVE:
                    delete_result = await self._delete_data(data_items, rule)
                    result["size_freed_bytes"] = delete_result["size_freed_bytes"]
            
            return result
            
        except Exception as e:
            result["errors"].append(f"Archive operation failed: {str(e)}")
            return result
    
    async def _compress_data(self, data_items: List[Dict[str, Any]], rule: RetentionRule) -> Dict[str, Any]:
        """Compress data items."""
        # Implementation would compress files or data in place
        return {
            "action": "compress",
            "records_processed": 0,
            "records_failed": len(data_items),
            "size_freed_bytes": 0,
            "processed_ids": [],
            "errors": ["Compression not implemented"]
        }
    
    async def _aggregate_data(self, data_items: List[Dict[str, Any]], rule: RetentionRule) -> Dict[str, Any]:
        """Aggregate data items into summary records."""
        # Implementation would create aggregated summaries and delete originals
        return {
            "action": "aggregate",
            "records_processed": 0,
            "records_failed": len(data_items),
            "size_freed_bytes": 0,
            "processed_ids": [],
            "errors": ["Aggregation not implemented"]
        }
    
    async def _export_data(self, data_items: List[Dict[str, Any]], rule: RetentionRule) -> Dict[str, Any]:
        """Export data items to external storage."""
        # Implementation would export data and optionally delete originals
        return {
            "action": "export",
            "records_processed": 0,
            "records_failed": len(data_items),
            "size_freed_bytes": 0,
            "processed_ids": [],
            "errors": ["Export not implemented"]
        }
    
    async def _run_retention_schedule(self, rule: RetentionRule) -> None:
        """Run retention schedule for a specific rule."""
        while self._running:
            try:
                await asyncio.sleep(rule.check_interval_hours * 3600)
                
                if not self._running or not rule.enabled:
                    break
                
                # Execute the rule if not currently executing
                if not self.current_execution:
                    logger.debug("Executing scheduled retention rule", rule_id=rule.rule_id)
                    await self.execute_retention_policies([rule.rule_id])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in retention schedule", rule_id=rule.rule_id, error=str(e))
                await asyncio.sleep(3600)  # Wait an hour before retrying
    
    async def _monitor_retention_health(self) -> None:
        """Monitor retention system health."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                if not self._running:
                    break
                
                # Check for failed executions
                recent_failures = [
                    execution for execution in self.execution_history[-10:]
                    if not execution.success
                ]
                
                if len(recent_failures) > 5:
                    logger.warning("High retention failure rate detected", 
                                 recent_failures=len(recent_failures))
                
                # Check disk usage
                total_size = await self._calculate_total_data_size()
                if total_size > 10 * 1024**3:  # 10 GB threshold
                    logger.warning("High disk usage detected", 
                                 total_size_gb=total_size / 1024**3)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in retention health monitoring", error=str(e))
    
    async def _calculate_total_data_size(self) -> int:
        """Calculate total size of data managed by retention policies."""
        total_size = 0
        
        try:
            # JSON files
            json_stats = await self.json_persistence.get_storage_statistics()
            total_size += json_stats.get("total_size_bytes", 0)
            
            # Archive files
            if self.archive_path.exists():
                for file_path in self.archive_path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        
        except Exception as e:
            logger.error("Failed to calculate total data size", error=str(e))
        
        return total_size
    
    async def _complete_execution(self, success: bool, error_message: str = None) -> None:
        """Complete the current retention execution."""
        if self.current_execution:
            self.current_execution.completed_at = datetime.now()
            self.current_execution.success = success
            
            if error_message:
                self.current_execution.errors.append(error_message)
            
            # Add to history
            self.execution_history.append(self.current_execution)
            
            # Keep only recent history
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
            
            # Emit event
            if success:
                await self._emit_event("retention_completed", self.current_execution)
            else:
                await self._emit_event("retention_failed", self.current_execution)
            
            self.current_execution = None
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler for retention events."""
        self.event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit retention event to registered handlers."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    def add_retention_rule(self, rule: RetentionRule) -> None:
        """Add a new retention rule."""
        self.retention_rules[rule.rule_id] = rule
        logger.info("Retention rule added", rule_id=rule.rule_id, rule_name=rule.name)
    
    def remove_retention_rule(self, rule_id: str) -> bool:
        """Remove a retention rule."""
        if rule_id in self.retention_rules:
            del self.retention_rules[rule_id]
            logger.info("Retention rule removed", rule_id=rule_id)
            return True
        return False
    
    def update_retention_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update a retention rule."""
        if rule_id in self.retention_rules:
            rule = self.retention_rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            logger.info("Retention rule updated", rule_id=rule_id, updates=list(updates.keys()))
            return True
        return False
    
    async def get_retention_status(self) -> Dict[str, Any]:
        """Get comprehensive retention status."""
        return {
            "running": self._running,
            "active_rules": len([r for r in self.retention_rules.values() if r.enabled]),
            "total_rules": len(self.retention_rules),
            "current_execution": self.current_execution.execution_id if self.current_execution else None,
            "recent_executions": len(self.execution_history),
            "metrics": dict(self.metrics),
            "rules_status": {
                rule_id: {
                    "enabled": rule.enabled,
                    "last_check": None,  # Would track last execution time
                    "next_check": None   # Would calculate next execution time
                }
                for rule_id, rule in self.retention_rules.items()
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False