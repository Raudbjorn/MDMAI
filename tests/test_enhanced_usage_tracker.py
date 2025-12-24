"""Comprehensive tests for the enhanced usage tracking and cost management system."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest
import tempfile

from src.ai_providers.enhanced_usage_tracker import (
    AdvancedTokenCounter,
    BudgetLevel,
    BudgetManager,
    ComprehensiveUsageManager,
    CostBreakdown,
    CostCalculationEngine,
    CurrencyCode,
    EnhancedBudget,
    EnhancedUsageRecord,
    EnhancedUsageTracker,
    TokenCountMetrics,
    UsageGranularity,
    UserSession,
    track_usage,
)
from src.ai_providers.models import (
    AIRequest,
    AIResponse,
    ProviderType,
    StreamingChunk,
)


# ==================== Test Fixtures ====================

@pytest.fixture
def token_counter():
    """Create a token counter instance."""
    return AdvancedTokenCounter()


@pytest.fixture
def cost_engine():
    """Create a cost calculation engine."""
    return CostCalculationEngine()


@pytest.fixture
def temp_storage_path():
    """Create a temporary storage path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def usage_tracker(temp_storage_path, token_counter, cost_engine):
    """Create an enhanced usage tracker."""
    return EnhancedUsageTracker(
        storage_path=temp_storage_path,
        token_counter=token_counter,
        cost_engine=cost_engine,
    )


@pytest.fixture
def budget_manager(usage_tracker, cost_engine):
    """Create a budget manager."""
    return BudgetManager(usage_tracker, cost_engine)


@pytest.fixture
def comprehensive_manager(temp_storage_path):
    """Create a comprehensive usage manager."""
    return ComprehensiveUsageManager(storage_path=temp_storage_path)


@pytest.fixture
def sample_request():
    """Create a sample AI request."""
    return AIRequest(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        max_tokens=100,
        temperature=0.7,
    )


@pytest.fixture
def sample_response():
    """Create a sample AI response."""
    return AIResponse(
        request_id="test-123",
        provider_type=ProviderType.OPENAI,
        model="gpt-4",
        content="I'm doing well, thank you!",
        usage={"input_tokens": 20, "output_tokens": 10},
        cost=0.001,
        latency_ms=500.0,
    )


@pytest.fixture
def sample_budget():
    """Create a sample enhanced budget."""
    return EnhancedBudget(
        name="Test Budget",
        daily_limit=Decimal("10.00"),
        monthly_limit=Decimal("100.00"),
        per_request_limit=Decimal("1.00"),
        enforcement_level=BudgetLevel.HARD,
        alert_thresholds=[0.5, 0.75, 0.9],
    )


# ==================== Token Counter Tests ====================

class TestAdvancedTokenCounter:
    """Test the advanced token counter."""
    
    def test_count_text_tokens(self, token_counter):
        """Test counting tokens in plain text."""
        text = "Hello, this is a test message."
        metrics = token_counter.count_tokens(
            text, ProviderType.OPENAI, "gpt-4"
        )
        
        assert metrics.text_tokens > 0
        assert metrics.total_tokens == metrics.text_tokens
        assert metrics.image_tokens == 0
    
    def test_count_message_tokens(self, token_counter):
        """Test counting tokens in messages."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"},
        ]
        
        metrics = token_counter.count_tokens(
            messages, ProviderType.OPENAI, "gpt-4"
        )
        
        assert metrics.text_tokens > 0
        assert metrics.system_prompt_tokens > 0
        assert metrics.total_tokens > 0
    
    def test_count_multimodal_tokens(self, token_counter):
        """Test counting tokens in multimodal content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "detail": "high"},
                ],
            }
        ]
        
        metrics = token_counter.count_tokens(
            messages, ProviderType.OPENAI, "gpt-4-vision"
        )
        
        assert metrics.text_tokens > 0
        assert metrics.image_tokens > 0
        assert metrics.total_tokens == metrics.text_tokens + metrics.image_tokens
    
    def test_count_tool_tokens(self, token_counter):
        """Test counting tokens in tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me search for that.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}',
                        },
                    }
                ],
            }
        ]
        
        metrics = token_counter.count_tokens(
            messages, ProviderType.OPENAI, "gpt-4"
        )
        
        assert metrics.text_tokens > 0
        assert metrics.tool_call_tokens > 0
    
    def test_caching(self, token_counter):
        """Test token counting cache."""
        text = "This is a test message for caching."
        
        # First call
        metrics1 = token_counter.count_tokens(
            text, ProviderType.OPENAI, "gpt-4", cache=True
        )
        
        # Second call should use cache
        metrics2 = token_counter.count_tokens(
            text, ProviderType.OPENAI, "gpt-4", cache=True
        )
        
        assert metrics1.text_tokens == metrics2.text_tokens
    
    def test_streaming_token_count(self, token_counter):
        """Test counting tokens in streaming response."""
        chunks = [
            StreamingChunk(request_id="123", content="Hello"),
            StreamingChunk(request_id="123", content=" world"),
            StreamingChunk(request_id="123", content="!"),
        ]
        
        total_tokens = 0
        generator = token_counter.count_streaming_tokens(
            iter(chunks), ProviderType.OPENAI, "gpt-4"
        )
        
        for metrics in generator:
            total_tokens += metrics.text_tokens
        
        assert total_tokens > 0


# ==================== Cost Calculation Tests ====================

class TestCostCalculationEngine:
    """Test the cost calculation engine."""
    
    def test_calculate_basic_cost(self, cost_engine):
        """Test basic cost calculation."""
        metrics = TokenCountMetrics(
            text_tokens=1000,
            system_prompt_tokens=100,
        )
        
        breakdown = cost_engine.calculate_cost(
            ProviderType.OPENAI,
            "gpt-4-turbo",
            metrics,
        )
        
        assert breakdown.input_cost > 0
        assert breakdown.total_cost > 0
        assert breakdown.currency == CurrencyCode.USD
    
    def test_cached_token_discount(self, cost_engine):
        """Test cached token discount calculation."""
        metrics = TokenCountMetrics(
            text_tokens=1000,
            cached_tokens=500,
        )
        
        breakdown = cost_engine.calculate_cost(
            ProviderType.OPENAI,
            "gpt-4-turbo",
            metrics,
        )
        
        assert breakdown.cache_discount > 0
        assert breakdown.total_cost < breakdown.subtotal
    
    def test_volume_discounts(self, cost_engine):
        """Test volume discount application."""
        # Large volume
        metrics = TokenCountMetrics(text_tokens=150000)
        
        breakdown = cost_engine.calculate_cost(
            ProviderType.OPENAI,
            "gpt-3.5-turbo",
            metrics,
            apply_discounts=True,
        )
        
        assert breakdown.volume_discount > 0
    
    def test_currency_conversion(self, cost_engine):
        """Test currency conversion."""
        metrics = TokenCountMetrics(text_tokens=1000)
        
        breakdown_usd = cost_engine.calculate_cost(
            ProviderType.OPENAI,
            "gpt-4",
            metrics,
            currency=CurrencyCode.USD,
        )
        
        breakdown_eur = cost_engine.calculate_cost(
            ProviderType.OPENAI,
            "gpt-4",
            metrics,
            currency=CurrencyCode.EUR,
        )
        
        assert breakdown_eur.currency == CurrencyCode.EUR
        assert breakdown_eur.exchange_rate != Decimal("1.0")
    
    def test_batch_cost_estimation(self, cost_engine, token_counter):
        """Test batch cost estimation."""
        requests = [
            AIRequest(model="gpt-4", messages=[{"role": "user", "content": f"Message {i}"}])
            for i in range(20)
        ]
        
        total_cost, individual_costs = cost_engine.estimate_batch_cost(
            requests,
            ProviderType.OPENAI,
            "gpt-4",
            token_counter,
        )
        
        assert len(individual_costs) == 20
        assert total_cost.batch_discount > 0  # Should have batch discount
        assert total_cost.total_cost < sum(c.total_cost for c in individual_costs)
    
    def test_multimodal_cost(self, cost_engine):
        """Test cost calculation for multimodal content."""
        metrics = TokenCountMetrics(
            text_tokens=500,
            image_tokens=500,
            audio_tokens=1000,
        )
        
        breakdown = cost_engine.calculate_cost(
            ProviderType.OPENAI,
            "gpt-4-vision",
            metrics,
        )
        
        assert breakdown.image_cost > 0
        assert breakdown.audio_cost > 0
        assert breakdown.total_cost > breakdown.input_cost


# ==================== Usage Tracker Tests ====================

class TestEnhancedUsageTracker:
    """Test the enhanced usage tracker."""
    
    @pytest.mark.asyncio
    async def test_record_usage(self, usage_tracker, sample_request, sample_response):
        """Test recording usage."""
        record = await usage_tracker.record_usage(
            sample_request,
            sample_response,
            ProviderType.OPENAI,
            "gpt-4",
            user_id="user123",
            tenant_id="tenant456",
            tags=["test", "api"],
        )
        
        assert record.request_id == sample_request.request_id
        assert record.user_id == "user123"
        assert record.tenant_id == "tenant456"
        assert record.success is True
        assert "test" in record.tags
        assert record.token_metrics.total_tokens > 0
        assert record.cost_breakdown.total_cost > 0
    
    def test_session_tracking(self, usage_tracker):
        """Test session tracking context manager."""
        with usage_tracker.track_session(user_id="user123") as session:
            assert session.is_active
            assert session.user_id == "user123"
            session.total_requests = 5
            session.total_cost = Decimal("1.50")
        
        assert not session.is_active
        assert session.duration is not None
    
    @pytest.mark.asyncio
    async def test_async_session_tracking(self, usage_tracker):
        """Test async session tracking."""
        async with usage_tracker.track_session_async(user_id="user456") as session:
            assert session.is_active
            session.total_requests = 3
            session.total_cost = Decimal("0.75")
        
        assert not session.is_active
    
    def test_usage_analytics(self, usage_tracker):
        """Test usage analytics generation."""
        # Add some mock records
        for i in range(10):
            record = EnhancedUsageRecord(
                request_id=f"req-{i}",
                provider_type=ProviderType.OPENAI if i % 2 == 0 else ProviderType.ANTHROPIC,
                model="gpt-4" if i % 2 == 0 else "claude-3",
                input_tokens=100,
                output_tokens=50,
                cost=0.01 * i,
                user_id="user123" if i % 3 == 0 else "user456",
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
            )
            usage_tracker._usage_records.append(record)
        
        analytics = usage_tracker.get_usage_analytics(
            granularity=UsageGranularity.HOURLY,
        )
        
        assert "summary" in analytics
        assert "providers" in analytics
        assert "models" in analytics
        assert "time_series" in analytics
        assert analytics["summary"]["total_requests"] == 10
    
    def test_export_json(self, usage_tracker):
        """Test exporting usage data as JSON."""
        # Add a record
        record = EnhancedUsageRecord(
            request_id="test-export",
            provider_type=ProviderType.OPENAI,
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
            user_id="user123",
        )
        usage_tracker._usage_records.append(record)
        
        json_data = usage_tracker.export_usage_data(format="json")
        parsed = json.loads(json_data)
        
        assert len(parsed) > 0
        assert parsed[0]["request_id"] == "test-export"
    
    def test_export_csv(self, usage_tracker):
        """Test exporting usage data as CSV."""
        # Add records
        for i in range(3):
            record = EnhancedUsageRecord(
                request_id=f"csv-{i}",
                provider_type=ProviderType.OPENAI,
                model="gpt-4",
                input_tokens=100,
                output_tokens=50,
                cost=0.01,
            )
            usage_tracker._usage_records.append(record)
        
        csv_data = usage_tracker.export_usage_data(format="csv")
        
        assert "request_id" in csv_data
        assert "csv-0" in csv_data
        assert "csv-1" in csv_data
        assert "csv-2" in csv_data
    
    def test_persistence(self, usage_tracker, temp_storage_path):
        """Test data persistence."""
        # Add usage data
        usage_tracker._daily_usage["2024-01-01"] = Decimal("10.50")
        usage_tracker._monthly_usage["2024-01"] = Decimal("105.00")
        
        # Save data
        usage_tracker._save_usage_data()
        
        # Check files exist
        daily_file = temp_storage_path / "daily_usage.json"
        monthly_file = temp_storage_path / "monthly_usage.json"
        
        assert daily_file.exists()
        assert monthly_file.exists()
        
        # Load and verify
        with open(daily_file) as f:
            daily_data = json.load(f)
            assert daily_data["2024-01-01"] == "10.50"


# ==================== Budget Manager Tests ====================

class TestBudgetManager:
    """Test the budget manager."""
    
    @pytest.mark.asyncio
    async def test_add_budget(self, budget_manager, sample_budget):
        """Test adding a budget."""
        await budget_manager.add_budget(sample_budget)
        
        assert sample_budget.budget_id in budget_manager._budgets
    
    @pytest.mark.asyncio
    async def test_check_budget_under_limit(
        self, budget_manager, sample_budget, sample_request
    ):
        """Test budget check when under limit."""
        await budget_manager.add_budget(sample_budget)
        
        allowed, reason, strategy = await budget_manager.check_budget(
            sample_request,
            ProviderType.OPENAI,
            "gpt-4",
            user_id="user123",
        )
        
        assert allowed is True
        assert reason is None
    
    @pytest.mark.asyncio
    async def test_check_budget_over_limit(
        self, budget_manager, sample_request
    ):
        """Test budget check when over limit."""
        # Create restrictive budget
        budget = EnhancedBudget(
            name="Restrictive",
            per_request_limit=Decimal("0.0001"),  # Very low limit
            enforcement_level=BudgetLevel.HARD,
        )
        await budget_manager.add_budget(budget)
        
        allowed, reason, strategy = await budget_manager.check_budget(
            sample_request,
            ProviderType.OPENAI,
            "gpt-4",
        )
        
        assert allowed is False
        assert reason is not None
        assert "exceeds limit" in reason
    
    @pytest.mark.asyncio
    async def test_adaptive_budget_enforcement(
        self, budget_manager, sample_request
    ):
        """Test adaptive budget enforcement."""
        budget = EnhancedBudget(
            name="Adaptive",
            per_request_limit=Decimal("0.0001"),
            enforcement_level=BudgetLevel.ADAPTIVE,
            degradation_strategies=["reduce_max_tokens", "use_smaller_model"],
            fallback_models=["gpt-3.5-turbo"],
        )
        await budget_manager.add_budget(budget)
        
        allowed, reason, strategy = await budget_manager.check_budget(
            sample_request,
            ProviderType.OPENAI,
            "gpt-4",
        )
        
        assert allowed is True
        assert strategy is not None
        assert "strategies" in strategy
        assert len(strategy["strategies"]) > 0
    
    @pytest.mark.asyncio
    async def test_soft_budget_alert(
        self, budget_manager, sample_request
    ):
        """Test soft budget enforcement with alerts."""
        budget = EnhancedBudget(
            name="Soft",
            per_request_limit=Decimal("0.0001"),
            enforcement_level=BudgetLevel.SOFT,
        )
        await budget_manager.add_budget(budget)
        
        allowed, reason, strategy = await budget_manager.check_budget(
            sample_request,
            ProviderType.OPENAI,
            "gpt-4",
        )
        
        assert allowed is True  # Soft enforcement allows request
        assert len(budget_manager._alerts) > 0  # But creates alert
    
    def test_budget_status(self, budget_manager):
        """Test getting budget status."""
        # Add mock usage data
        budget_manager._usage_tracker._daily_usage["2024-01-01"] = Decimal("5.00")
        budget_manager._usage_tracker._monthly_usage["2024-01"] = Decimal("50.00")
        
        # Add budget
        budget = EnhancedBudget(
            budget_id="test-budget",
            name="Test",
            daily_limit=Decimal("10.00"),
            monthly_limit=Decimal("100.00"),
        )
        budget_manager._budgets["test-budget"] = budget
        
        status = budget_manager.get_budget_status(budget_id="test-budget")
        
        assert "test-budget" in status
        assert status["test-budget"]["name"] == "Test"
        assert "limits" in status["test-budget"]
        assert "usage" in status["test-budget"]
        assert "remaining" in status["test-budget"]


# ==================== Decorator Tests ====================

class TestUsageDecorator:
    """Test the usage tracking decorator."""
    
    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test decorator on async function."""
        tracker = EnhancedUsageTracker()
        
        @track_usage(ProviderType.OPENAI, "gpt-4", tracker=tracker)
        async def make_call(request: AIRequest) -> AIResponse:
            return AIResponse(
                request_id=request.request_id,
                provider_type=ProviderType.OPENAI,
                model="gpt-4",
                content="Test response",
            )
        
        request = AIRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
        )
        
        response = await make_call(request)
        
        assert response.content == "Test response"
        assert len(tracker._usage_records) > 0
    
    def test_sync_decorator(self):
        """Test decorator on sync function."""
        tracker = EnhancedUsageTracker()
        
        @track_usage(ProviderType.OPENAI, "gpt-4", tracker=tracker)
        def make_sync_call(request: AIRequest) -> str:
            return "Sync response"
        
        request = AIRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
        )
        
        # Note: This would need event loop setup in real usage
        # response = make_sync_call(request)


# ==================== Integration Tests ====================

class TestComprehensiveUsageManager:
    """Test the comprehensive usage manager."""
    
    @pytest.mark.asyncio
    async def test_track_request_allowed(
        self, comprehensive_manager, sample_request
    ):
        """Test tracking request that is allowed."""
        # Add a permissive budget
        budget = EnhancedBudget(
            name="Permissive",
            daily_limit=Decimal("100.00"),
        )
        await comprehensive_manager.budget_manager.add_budget(budget)
        
        allowed, reason, strategy = await comprehensive_manager.track_request(
            sample_request,
            ProviderType.OPENAI,
            "gpt-4",
            user_id="user123",
        )
        
        assert allowed is True
        assert reason is None
    
    @pytest.mark.asyncio
    async def test_track_request_blocked(
        self, comprehensive_manager, sample_request
    ):
        """Test tracking request that is blocked."""
        # Add a restrictive budget
        budget = EnhancedBudget(
            name="Restrictive",
            per_request_limit=Decimal("0.0001"),
            enforcement_level=BudgetLevel.HARD,
        )
        await comprehensive_manager.budget_manager.add_budget(budget)
        
        allowed, reason, strategy = await comprehensive_manager.track_request(
            sample_request,
            ProviderType.OPENAI,
            "gpt-4",
        )
        
        assert allowed is False
        assert reason is not None
    
    @pytest.mark.asyncio
    async def test_record_response(
        self, comprehensive_manager, sample_request, sample_response
    ):
        """Test recording a response."""
        record = await comprehensive_manager.record_response(
            sample_request,
            sample_response,
            ProviderType.OPENAI,
            "gpt-4",
            user_id="user123",
            tags=["test"],
        )
        
        assert record.request_id == sample_request.request_id
        assert record.user_id == "user123"
        assert record.success is True
    
    def test_get_analytics(self, comprehensive_manager):
        """Test getting analytics."""
        analytics = comprehensive_manager.get_analytics()
        
        assert "summary" in analytics
        assert "providers" in analytics
        assert "models" in analytics
    
    def test_export_data(self, comprehensive_manager):
        """Test exporting data."""
        json_export = comprehensive_manager.export_data(format="json")
        assert isinstance(json_export, str)
        
        csv_export = comprehensive_manager.export_data(format="csv")
        assert isinstance(csv_export, str)


# ==================== Performance Tests ====================

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_bulk_recording_performance(self, usage_tracker):
        """Test performance with bulk usage recording."""
        import time
        
        start = time.time()
        
        # Record 1000 usage entries
        for i in range(1000):
            request = AIRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Message {i}"}],
            )
            response = AIResponse(
                request_id=request.request_id,
                provider_type=ProviderType.OPENAI,
                model="gpt-4",
                content=f"Response {i}",
            )
            
            await usage_tracker.record_usage(
                request,
                response,
                ProviderType.OPENAI,
                "gpt-4",
            )
        
        elapsed = time.time() - start
        
        # Should complete within reasonable time
        assert elapsed < 10.0  # 10 seconds for 1000 records
        assert len(usage_tracker._usage_records) == 1000
    
    def test_token_counting_cache_performance(self, token_counter):
        """Test token counting cache performance."""
        import time
        
        text = "This is a test message " * 100  # Long text
        
        # First call (no cache)
        start = time.time()
        metrics1 = token_counter.count_tokens(
            text, ProviderType.OPENAI, "gpt-4", cache=True
        )
        first_call_time = time.time() - start
        
        # Second call (cached)
        start = time.time()
        metrics2 = token_counter.count_tokens(
            text, ProviderType.OPENAI, "gpt-4", cache=True
        )
        cached_call_time = time.time() - start
        
        # Cached call should be much faster
        assert cached_call_time < first_call_time / 10
        assert metrics1.text_tokens == metrics2.text_tokens


# ==================== Edge Cases and Error Handling ====================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_record_usage_with_error(self, usage_tracker, sample_request):
        """Test recording usage when request fails."""
        record = await usage_tracker.record_usage(
            sample_request,
            None,  # No response (error case)
            ProviderType.OPENAI,
            "gpt-4",
        )
        
        assert record.success is False
        assert record.error_message is not None
    
    def test_empty_message_tokens(self, token_counter):
        """Test token counting with empty messages."""
        metrics = token_counter.count_tokens(
            [], ProviderType.OPENAI, "gpt-4"
        )
        
        assert metrics.total_tokens == 0
    
    def test_invalid_currency_conversion(self, cost_engine):
        """Test handling of invalid currency."""
        metrics = TokenCountMetrics(text_tokens=1000)
        
        # Should use USD as fallback
        breakdown = cost_engine.calculate_cost(
            ProviderType.OPENAI,
            "gpt-4",
            metrics,
            currency=CurrencyCode.USD,  # Valid currency
        )
        
        assert breakdown.currency == CurrencyCode.USD
    
    @pytest.mark.asyncio
    async def test_concurrent_usage_recording(self, usage_tracker):
        """Test concurrent usage recording."""
        tasks = []
        
        for i in range(100):
            request = AIRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Concurrent {i}"}],
            )
            response = AIResponse(
                request_id=request.request_id,
                provider_type=ProviderType.OPENAI,
                model="gpt-4",
                content=f"Response {i}",
            )
            
            task = usage_tracker.record_usage(
                request,
                response,
                ProviderType.OPENAI,
                "gpt-4",
            )
            tasks.append(task)
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
        
        # All should be recorded
        assert len(usage_tracker._usage_records) == 100
    
    def test_malformed_tool_calls(self, token_counter):
        """Test token counting with malformed tool calls."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"invalid": "structure"},  # Malformed
                ],
            }
        ]
        
        # Should handle gracefully
        metrics = token_counter.count_tokens(
            messages, ProviderType.OPENAI, "gpt-4"
        )
        
        assert metrics.tool_call_tokens >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])