"""
Tests for Result pattern implementation using returns library.

These tests demonstrate:
- Testing Success and Failure cases
- Working with AppError
- Testing decorated functions
- Async Result operations
- Result composition and chaining
"""

import asyncio
import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

from returns.result import Failure, Success

from src.campaign.campaign_manager_returns import Campaign, CampaignManager, Character
from src.core.result_pattern import (
    AppError,
    AsyncResult,
    ErrorKind,
    chain_results,
    collect_results,
    database_error,
    flat_map_async,
    map_error,
    not_found_error,
    result_to_response,
    unwrap_or_raise,
    validation_error,
    with_result,
)


class TestAppError:
    """Test AppError functionality."""

    def test_create_validation_error(self):
        """Test creating a validation error."""
        error = validation_error("Invalid input", field="email")
        
        assert error.kind == ErrorKind.VALIDATION
        assert error.message == "Invalid input"
        assert error.details["field"] == "email"
        assert not error.recoverable

    def test_create_not_found_error(self):
        """Test creating a not found error."""
        error = not_found_error("User", "123")
        
        assert error.kind == ErrorKind.NOT_FOUND
        assert "User not found: 123" in error.message
        assert error.details["resource"] == "User"
        assert error.details["id"] == "123"

    def test_error_to_dict(self):
        """Test converting error to dictionary."""
        error = database_error("Connection failed", operation="connect")
        error_dict = error.to_dict()
        
        assert error_dict["error"] == "database"
        assert error_dict["message"] == "Connection failed"
        assert error_dict["details"]["operation"] == "connect"

    def test_error_from_exception(self):
        """Test creating AppError from exception."""
        try:
            raise ValueError("Test exception")
        except Exception as e:
            error = AppError.from_exception(e, kind=ErrorKind.VALIDATION)
        
        assert error.kind == ErrorKind.VALIDATION
        assert error.message == "Test exception"
        assert error.details["exception_type"] == "ValueError"


class TestWithResultDecorator:
    """Test the with_result decorator."""

    def test_sync_function_success(self):
        """Test decorator with successful sync function."""
        
        @with_result(error_kind=ErrorKind.DATABASE)
        def fetch_data(id: str) -> Dict[str, Any]:
            return {"id": id, "name": "Test"}
        
        result = fetch_data("123")
        
        assert isinstance(result, Success)
        assert result.unwrap() == {"id": "123", "name": "Test"}

    def test_sync_function_failure(self):
        """Test decorator with failing sync function."""
        
        @with_result(error_kind=ErrorKind.DATABASE)
        def fetch_data(id: str) -> Dict[str, Any]:
            raise ValueError("Database connection failed")
        
        result = fetch_data("123")
        
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.kind == ErrorKind.DATABASE
        assert "Database connection failed" in error.message

    @pytest.mark.asyncio
    async def test_async_function_success(self):
        """Test decorator with successful async function."""
        
        @with_result(error_kind=ErrorKind.NETWORK)
        async def fetch_remote_data(url: str) -> Dict[str, Any]:
            return {"url": url, "status": "success"}
        
        result = await fetch_remote_data("http://example.com")
        
        assert isinstance(result, Success)
        assert result.unwrap()["status"] == "success"

    @pytest.mark.asyncio
    async def test_async_function_failure(self):
        """Test decorator with failing async function."""
        
        @with_result(error_kind=ErrorKind.NETWORK)
        async def fetch_remote_data(url: str) -> Dict[str, Any]:
            raise ConnectionError("Network timeout")
        
        result = await fetch_remote_data("http://example.com")
        
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.kind == ErrorKind.NETWORK
        assert "Network timeout" in error.message

    def test_decorator_with_custom_error_constructor(self):
        """Test decorator with custom error constructor."""
        
        def create_custom_error(msg: str) -> AppError:
            return validation_error(f"Custom: {msg}", field="test")
        
        @with_result(error_constructor=create_custom_error)
        def process_data(data: str) -> str:
            if not data:
                raise ValueError("Empty data")
            return data.upper()
        
        result = process_data("")
        
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.kind == ErrorKind.VALIDATION
        assert "Custom: Empty data" in error.message
        assert error.details["field"] == "test"

    def test_decorator_preserves_existing_result(self):
        """Test that decorator preserves functions that already return Result."""
        
        @with_result()
        def already_returns_result(value: int) -> Success[int]:
            if value < 0:
                return Failure(validation_error("Value must be positive"))
            return Success(value * 2)
        
        success_result = already_returns_result(5)
        assert isinstance(success_result, Success)
        assert success_result.unwrap() == 10
        
        failure_result = already_returns_result(-5)
        assert isinstance(failure_result, Failure)
        assert failure_result.failure().kind == ErrorKind.VALIDATION


class TestResultHelpers:
    """Test Result helper functions."""

    def test_collect_results_all_success(self):
        """Test collecting all successful results."""
        results = [
            Success(1),
            Success(2),
            Success(3),
        ]
        
        collected = collect_results(results)
        
        assert isinstance(collected, Success)
        assert collected.unwrap() == [1, 2, 3]

    def test_collect_results_with_failure(self):
        """Test collecting results with a failure."""
        error = validation_error("Invalid value")
        results = [
            Success(1),
            Failure(error),
            Success(3),
        ]
        
        collected = collect_results(results)
        
        assert isinstance(collected, Failure)
        assert collected.failure() == error

    @pytest.mark.asyncio
    async def test_collect_async_results(self):
        """Test collecting async results."""
        
        async def create_result(value: int) -> Success[int]:
            await asyncio.sleep(0.01)
            return Success(value)
        
        tasks = [
            asyncio.create_task(create_result(1)),
            asyncio.create_task(create_result(2)),
            asyncio.create_task(create_result(3)),
        ]
        
        from src.core.result_pattern import collect_async_results
        collected = await collect_async_results(tasks)
        
        assert isinstance(collected, Success)
        assert collected.unwrap() == [1, 2, 3]

    def test_map_error(self):
        """Test mapping error type."""
        original_error = validation_error("Test error")
        result = Failure(original_error)
        
        mapped = map_error(
            result,
            lambda e: database_error(f"Wrapped: {e.message}")
        )
        
        assert isinstance(mapped, Failure)
        new_error = mapped.failure()
        assert new_error.kind == ErrorKind.DATABASE
        assert "Wrapped: Test error" in new_error.message

    def test_map_error_preserves_success(self):
        """Test that map_error preserves Success."""
        result = Success(42)
        
        mapped = map_error(
            result,
            lambda e: database_error("Should not be called")
        )
        
        assert isinstance(mapped, Success)
        assert mapped.unwrap() == 42

    def test_unwrap_or_raise_success(self):
        """Test unwrap_or_raise with Success."""
        result = Success("test value")
        value = unwrap_or_raise(result)
        assert value == "test value"

    def test_unwrap_or_raise_failure(self):
        """Test unwrap_or_raise with Failure."""
        error = validation_error("Test error")
        result = Failure(error)
        
        with pytest.raises(RuntimeError) as exc_info:
            unwrap_or_raise(result)
        
        assert "[validation] Test error" in str(exc_info.value)

    def test_result_to_response_success(self):
        """Test converting Success to API response."""
        result = Success({"id": 123, "name": "Test"})
        response = result_to_response(result)
        
        assert response["success"] is True
        assert response["data"] == {"id": 123, "name": "Test"}
        assert "error" not in response

    def test_result_to_response_failure(self):
        """Test converting Failure to API response."""
        error = not_found_error("User", "123")
        result = Failure(error)
        response = result_to_response(result)
        
        assert response["success"] is False
        assert "data" not in response
        assert response["error"]["error"] == "not_found"
        assert "User not found" in response["error"]["message"]

    def test_chain_results(self):
        """Test chaining Result-returning operations."""
        
        def validate(x: int) -> Success[int]:
            if x < 0:
                return Failure(validation_error("Must be positive"))
            return Success(x)
        
        def double(x: int) -> Success[int]:
            return Success(x * 2)
        
        def add_ten(x: int) -> Success[int]:
            return Success(x + 10)
        
        process = chain_results(validate, double, add_ten)
        
        # Test successful chain
        result = process(5)
        assert isinstance(result, Success)
        assert result.unwrap() == 20  # (5 * 2) + 10
        
        # Test chain that fails at validation
        result = process(-5)
        assert isinstance(result, Failure)
        assert result.failure().kind == ErrorKind.VALIDATION


class TestCampaignManager:
    """Test CampaignManager with Result pattern."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db.add_document = AsyncMock()
        db.get_document = AsyncMock()
        db.query_by_metadata = AsyncMock()
        db.delete_document = AsyncMock()
        return db

    @pytest.fixture
    def manager(self, mock_db):
        """Create CampaignManager instance."""
        return CampaignManager(mock_db)

    @pytest.mark.asyncio
    async def test_create_campaign_success(self, manager, mock_db):
        """Test successful campaign creation."""
        result = await manager.create_campaign(
            name="Test Campaign",
            system="D&D 5e",
            description="A test campaign",
        )
        
        assert isinstance(result, Success)
        campaign = result.unwrap()
        assert campaign.name == "Test Campaign"
        assert campaign.system == "D&D 5e"
        assert mock_db.add_document.called

    @pytest.mark.asyncio
    async def test_create_campaign_validation_error(self, manager, mock_db):
        """Test campaign creation with validation error."""
        result = await manager.create_campaign(
            name="",  # Empty name should fail
            system="D&D 5e",
        )
        
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.kind == ErrorKind.VALIDATION
        assert "name cannot be empty" in error.message
        assert not mock_db.add_document.called

    @pytest.mark.asyncio
    async def test_create_campaign_database_error(self, manager, mock_db):
        """Test campaign creation with database error."""
        mock_db.add_document.side_effect = Exception("Database connection failed")
        
        result = await manager.create_campaign(
            name="Test Campaign",
            system="D&D 5e",
        )
        
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.kind == ErrorKind.DATABASE
        assert "Database connection failed" in error.message

    @pytest.mark.asyncio
    async def test_get_campaign_success(self, manager, mock_db):
        """Test successful campaign retrieval."""
        mock_db.get_document.return_value = [{
            "id": "123",
            "name": "Test Campaign",
            "system": "D&D 5e",
            "description": None,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }]
        
        result = await manager.get_campaign("123")
        
        assert isinstance(result, Success)
        data = result.unwrap()
        assert data["id"] == "123"
        assert data["name"] == "Test Campaign"

    @pytest.mark.asyncio
    async def test_get_campaign_not_found(self, manager, mock_db):
        """Test getting non-existent campaign."""
        mock_db.get_document.return_value = []
        
        result = await manager.get_campaign("nonexistent")
        
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.kind == ErrorKind.NOT_FOUND
        assert "Campaign not found" in error.message

    @pytest.mark.asyncio
    async def test_add_character_success(self, manager, mock_db):
        """Test successfully adding a character to campaign."""
        # Mock campaign exists
        mock_db.get_document.return_value = [{
            "id": "campaign-123",
            "name": "Test Campaign",
            "system": "D&D 5e",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }]
        
        result = await manager.add_character(
            campaign_id="campaign-123",
            character_data={"name": "Gandalf", "class": "Wizard"},
        )
        
        assert isinstance(result, Success)
        character = result.unwrap()
        assert character.name == "Gandalf"
        assert character.campaign_id == "campaign-123"

    @pytest.mark.asyncio
    async def test_add_character_campaign_not_found(self, manager, mock_db):
        """Test adding character to non-existent campaign."""
        mock_db.get_document.return_value = []
        
        result = await manager.add_character(
            campaign_id="nonexistent",
            character_data={"name": "Gandalf"},
        )
        
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.kind == ErrorKind.NOT_FOUND

    @pytest.mark.asyncio
    async def test_add_character_validation_error(self, manager, mock_db):
        """Test adding character with invalid data."""
        # Mock campaign exists
        mock_db.get_document.return_value = [{
            "id": "campaign-123",
            "name": "Test Campaign",
            "system": "D&D 5e",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }]
        
        result = await manager.add_character(
            campaign_id="campaign-123",
            character_data={},  # Missing required name
        )
        
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.kind == ErrorKind.VALIDATION
        assert "name is required" in error.message

    @pytest.mark.asyncio
    async def test_list_campaigns_success(self, manager, mock_db):
        """Test listing campaigns."""
        mock_db.query_by_metadata.return_value = [
            {"id": "1", "name": "Campaign 1"},
            {"id": "2", "name": "Campaign 2"},
        ]
        
        result = await manager.list_campaigns(limit=10)
        
        assert isinstance(result, Success)
        campaigns = result.unwrap()
        assert len(campaigns) == 2

    @pytest.mark.asyncio
    async def test_list_campaigns_invalid_limit(self, manager, mock_db):
        """Test listing campaigns with invalid limit."""
        result = await manager.list_campaigns(limit=150)  # Over max
        
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.kind == ErrorKind.VALIDATION
        assert "Limit must be between 1 and 100" in error.message


class TestAsyncResult:
    """Test AsyncResult helper class."""

    @pytest.mark.asyncio
    async def test_from_coroutine_success(self):
        """Test converting successful coroutine to Result."""
        
        async def async_operation() -> str:
            await asyncio.sleep(0.01)
            return "success"
        
        result = await AsyncResult.from_coroutine(
            async_operation(),
            error_kind=ErrorKind.NETWORK,
        )
        
        assert isinstance(result, Success)
        assert result.unwrap() == "success"

    @pytest.mark.asyncio
    async def test_from_coroutine_failure(self):
        """Test converting failing coroutine to Result."""
        
        async def async_operation() -> str:
            await asyncio.sleep(0.01)
            raise ConnectionError("Network error")
        
        result = await AsyncResult.from_coroutine(
            async_operation(),
            error_kind=ErrorKind.NETWORK,
        )
        
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.kind == ErrorKind.NETWORK
        assert "Network error" in error.message