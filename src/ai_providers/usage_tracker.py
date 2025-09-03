"""
Usage tracking and cost management for MDMAI TTRPG Assistant.

This module provides usage tracking, cost calculation, and spending limit
enforcement for AI provider usage.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from .base_provider import ProviderType
from .pricing_config import get_pricing_manager

logger = logging.getLogger(__name__)


class SpendingLimitExceededException(Exception):
    """Raised when a user exceeds their spending limit."""
    
    def __init__(self, message: str, current_spend: float, limit: float):
        super().__init__(message)
        self.current_spend = current_spend
        self.limit = limit


@dataclass
class UsageRecord:
    """Record of AI provider usage."""
    user_id: str
    provider: ProviderType
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime
    session_id: str
    request_id: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['provider'] = self.provider.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UsageRecord':
        """Create from dictionary."""
        data = data.copy()
        data['provider'] = ProviderType(data['provider'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class SpendingLimit:
    """User spending limit configuration."""
    user_id: str
    daily_limit: float = 0.0
    weekly_limit: float = 0.0
    monthly_limit: float = 0.0
    enabled: bool = True
    alert_threshold: float = 0.8  # Alert at 80% of limit
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SpendingLimit':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class UsageTracker:
    """
    Tracks AI provider usage and manages spending limits.
    
    Features:
    - Local filesystem storage (JSON)
    - Optional ChromaDB integration
    - Cost calculation with provider-specific pricing
    - Spending limits (daily, weekly, monthly)
    - Usage analytics and reporting
    - Alert notifications
    """
    
    def __init__(self, storage_path: str = "./data/usage", use_chromadb: bool = False):
        """
        Initialize usage tracker.
        
        Args:
            storage_path: Path for local storage
            use_chromadb: Whether to use ChromaDB for persistence
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.use_chromadb = use_chromadb
        
        # Initialize storage
        if use_chromadb:
            self._init_chromadb()
        else:
            self._init_filesystem()
        
        # Use centralized pricing manager
        self.pricing_manager = get_pricing_manager()
        
        logger.info(f"UsageTracker initialized with storage at {self.storage_path}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB storage."""
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=str(self.storage_path / "chromadb"))
            self.usage_collection = self.client.get_or_create_collection(
                name="usage_records",
                metadata={"description": "LLM usage tracking"}
            )
            self.limits_collection = self.client.get_or_create_collection(
                name="user_limits",
                metadata={"description": "User spending limits"}
            )
            
            # Load existing data
            self.usage_records = self._load_records_chromadb()
            self.user_limits = self._load_limits_chromadb()
            
            logger.info("ChromaDB storage initialized")
        except ImportError:
            logger.warning("ChromaDB not available, falling back to filesystem storage")
            self.use_chromadb = False
            self._init_filesystem()
    
    def _init_filesystem(self):
        """Initialize filesystem storage."""
        self.records_file = self.storage_path / "usage_records.json"
        self.limits_file = self.storage_path / "user_limits.json"
        self.usage_records: List[UsageRecord] = self._load_records_json()
        self.user_limits: Dict[str, SpendingLimit] = self._load_limits_json()
        
        logger.info("Filesystem storage initialized")
    
    
    async def track_usage(
        self,
        user_id: str,
        provider: ProviderType,
        model: str,
        input_tokens: int,
        output_tokens: int,
        session_id: str,
        request_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> UsageRecord:
        """
        Track API usage and calculate costs.
        
        Args:
            user_id: User identifier
            provider: AI provider used
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            session_id: Session identifier
            request_id: Optional request identifier
            duration_ms: Request duration in milliseconds
            success: Whether the request was successful
            error_message: Error message if request failed
            
        Returns:
            UsageRecord: The created usage record
            
        Raises:
            SpendingLimitExceededException: If user exceeds spending limits
        """
        # Calculate cost
        cost = self._calculate_cost(provider, model, input_tokens, output_tokens)
        
        # Create usage record
        record = UsageRecord(
            user_id=user_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            request_id=request_id,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )
        
        # Add to records
        self.usage_records.append(record)
        
        # Check spending limits
        await self._check_spending_limits(user_id, cost)
        
        # Persist the records
        self._save_records()
        
        logger.debug(f"Tracked usage for {user_id}: {cost:.4f} USD")
        return record
    
    def set_spending_limit(
        self,
        user_id: str,
        daily_limit: float = 0.0,
        weekly_limit: float = 0.0,
        monthly_limit: float = 0.0,
        enabled: bool = True,
        alert_threshold: float = 0.8
    ) -> SpendingLimit:
        """
        Set spending limits for a user.
        
        Args:
            user_id: User identifier
            daily_limit: Daily spending limit in USD
            weekly_limit: Weekly spending limit in USD
            monthly_limit: Monthly spending limit in USD
            enabled: Whether limits are enabled
            alert_threshold: Threshold for alerts (0.0-1.0)
            
        Returns:
            SpendingLimit: The created/updated limit
        """
        limit = SpendingLimit(
            user_id=user_id,
            daily_limit=daily_limit,
            weekly_limit=weekly_limit,
            monthly_limit=monthly_limit,
            enabled=enabled,
            alert_threshold=alert_threshold
        )
        
        self.user_limits[user_id] = limit
        self._save_limits()
        
        logger.info(f"Set spending limits for {user_id}: daily=${daily_limit}, weekly=${weekly_limit}, monthly=${monthly_limit}")
        return limit
    
    def get_usage_summary(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: Optional[ProviderType] = None
    ) -> Dict:
        """
        Get usage summary for a user.
        
        Args:
            user_id: User identifier
            start_date: Start date for summary
            end_date: End date for summary
            provider: Optional provider filter
            
        Returns:
            Dict: Usage summary
        """
        # Filter records
        filtered_records = [
            r for r in self.usage_records 
            if r.user_id == user_id
        ]
        
        if start_date:
            filtered_records = [r for r in filtered_records if r.timestamp >= start_date]
        if end_date:
            filtered_records = [r for r in filtered_records if r.timestamp <= end_date]
        if provider:
            filtered_records = [r for r in filtered_records if r.provider == provider]
        
        if not filtered_records:
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'total_cost': 0.0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'by_provider': {},
                'by_model': {},
                'by_day': {}
            }
        
        # Calculate summary
        total_requests = len(filtered_records)
        successful_requests = sum(1 for r in filtered_records if r.success)
        total_cost = sum(r.cost for r in filtered_records)
        total_input_tokens = sum(r.input_tokens for r in filtered_records)
        total_output_tokens = sum(r.output_tokens for r in filtered_records)
        
        # Group by provider
        by_provider = {}
        for record in filtered_records:
            provider_name = record.provider.value
            if provider_name not in by_provider:
                by_provider[provider_name] = {
                    'requests': 0, 'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0
                }
            by_provider[provider_name]['requests'] += 1
            by_provider[provider_name]['cost'] += record.cost
            by_provider[provider_name]['input_tokens'] += record.input_tokens
            by_provider[provider_name]['output_tokens'] += record.output_tokens
        
        # Group by model
        by_model = {}
        for record in filtered_records:
            if record.model not in by_model:
                by_model[record.model] = {
                    'requests': 0, 'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0
                }
            by_model[record.model]['requests'] += 1
            by_model[record.model]['cost'] += record.cost
            by_model[record.model]['input_tokens'] += record.input_tokens
            by_model[record.model]['output_tokens'] += record.output_tokens
        
        # Group by day
        by_day = {}
        for record in filtered_records:
            date_key = record.timestamp.date().isoformat()
            if date_key not in by_day:
                by_day[date_key] = {
                    'requests': 0, 'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0
                }
            by_day[date_key]['requests'] += 1
            by_day[date_key]['cost'] += record.cost
            by_day[date_key]['input_tokens'] += record.input_tokens
            by_day[date_key]['output_tokens'] += record.output_tokens
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': (successful_requests / total_requests) * 100 if total_requests > 0 else 0,
            'total_cost': total_cost,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'average_cost_per_request': total_cost / total_requests if total_requests > 0 else 0,
            'by_provider': by_provider,
            'by_model': by_model,
            'by_day': by_day
        }
    
    def get_current_spending(self, user_id: str) -> Dict[str, float]:
        """
        Get current spending for different time periods.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict[str, float]: Spending amounts for different periods
        """
        now = datetime.utcnow()
        
        # Calculate date ranges
        today = now.date()
        week_start = today - timedelta(days=today.weekday())
        month_start = today.replace(day=1)
        
        # Filter records
        user_records = [r for r in self.usage_records if r.user_id == user_id]
        
        # Calculate spending
        daily_spend = sum(
            r.cost for r in user_records
            if r.timestamp.date() == today
        )
        
        weekly_spend = sum(
            r.cost for r in user_records
            if r.timestamp.date() >= week_start
        )
        
        monthly_spend = sum(
            r.cost for r in user_records
            if r.timestamp.date() >= month_start
        )
        
        return {
            'daily': daily_spend,
            'weekly': weekly_spend,
            'monthly': monthly_spend
        }
    
    def _calculate_cost(self, provider: ProviderType, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for API usage using centralized pricing."""
        return self.pricing_manager.calculate_cost(provider, model, input_tokens, output_tokens)
    
    async def _check_spending_limits(self, user_id: str, new_cost: float):
        """Check if user would exceed spending limits."""
        if user_id not in self.user_limits or not self.user_limits[user_id].enabled:
            return
        
        limits = self.user_limits[user_id]
        current_spending = self.get_current_spending(user_id)
        
        # Check daily limit
        if limits.daily_limit > 0:
            new_daily = current_spending['daily'] + new_cost
            if new_daily > limits.daily_limit:
                raise SpendingLimitExceededException(
                    f"Daily spending limit exceeded: ${new_daily:.2f} > ${limits.daily_limit:.2f}",
                    new_daily, limits.daily_limit
                )
            elif new_daily > limits.daily_limit * limits.alert_threshold:
                logger.warning(f"User {user_id} approaching daily limit: ${new_daily:.2f} / ${limits.daily_limit:.2f}")
        
        # Check weekly limit
        if limits.weekly_limit > 0:
            new_weekly = current_spending['weekly'] + new_cost
            if new_weekly > limits.weekly_limit:
                raise SpendingLimitExceededException(
                    f"Weekly spending limit exceeded: ${new_weekly:.2f} > ${limits.weekly_limit:.2f}",
                    new_weekly, limits.weekly_limit
                )
        
        # Check monthly limit
        if limits.monthly_limit > 0:
            new_monthly = current_spending['monthly'] + new_cost
            if new_monthly > limits.monthly_limit:
                raise SpendingLimitExceededException(
                    f"Monthly spending limit exceeded: ${new_monthly:.2f} > ${limits.monthly_limit:.2f}",
                    new_monthly, limits.monthly_limit
                )
    
    # Storage methods for JSON
    def _load_records_json(self) -> List[UsageRecord]:
        """Load usage records from JSON file."""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r') as f:
                    data = json.load(f)
                return [UsageRecord.from_dict(r) for r in data]
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error loading usage records: {e}")
                # Backup corrupted file
                backup_file = self.records_file.with_suffix('.json.backup')
                self.records_file.rename(backup_file)
        return []
    
    def _load_limits_json(self) -> Dict[str, SpendingLimit]:
        """Load spending limits from JSON file."""
        if self.limits_file.exists():
            try:
                with open(self.limits_file, 'r') as f:
                    data = json.load(f)
                return {k: SpendingLimit.from_dict(v) for k, v in data.items()}
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error loading spending limits: {e}")
                # Backup corrupted file
                backup_file = self.limits_file.with_suffix('.json.backup')
                self.limits_file.rename(backup_file)
        return {}
    
    def _save_records(self):
        """Save usage records to storage."""
        if self.use_chromadb:
            self._save_records_chromadb()
        else:
            self._save_records_json()
    
    def _save_limits(self):
        """Save spending limits to storage."""
        if self.use_chromadb:
            self._save_limits_chromadb()
        else:
            self._save_limits_json()
    
    def _save_records_json(self):
        """Save usage records to JSON file."""
        try:
            # Keep only recent records to prevent file from growing too large
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            recent_records = [r for r in self.usage_records if r.timestamp >= cutoff_date]
            
            data = [r.to_dict() for r in recent_records]
            temp_file = self.records_file.with_suffix('.tmp')
            
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.records_file)
            self.usage_records = recent_records  # Update in-memory list
            
        except Exception as e:
            logger.error(f"Error saving usage records: {e}")
    
    def _save_limits_json(self):
        """Save spending limits to JSON file."""
        try:
            data = {k: v.to_dict() for k, v in self.user_limits.items()}
            temp_file = self.limits_file.with_suffix('.tmp')
            
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.limits_file)
            
        except Exception as e:
            logger.error(f"Error saving spending limits: {e}")
    
    # ChromaDB methods (placeholder implementations)
    def _load_records_chromadb(self) -> List[UsageRecord]:
        """Load records from ChromaDB."""
        # Implementation would query ChromaDB collection
        return []
    
    def _load_limits_chromadb(self) -> Dict[str, SpendingLimit]:
        """Load limits from ChromaDB."""
        # Implementation would query ChromaDB collection
        return {}
    
    def _save_records_chromadb(self):
        """Save records to ChromaDB."""
        # Implementation would update ChromaDB collection
        pass
    
    def _save_limits_chromadb(self):
        """Save limits to ChromaDB."""
        # Implementation would update ChromaDB collection
        pass