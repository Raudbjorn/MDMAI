"""Comprehensive tests for enhanced security features."""

import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import jwt as pyjwt

from src.security.enhanced_security_manager import (
    EnhancedSecurityManager, EnhancedSecurityConfig
)
from src.security.auth_providers import OAuthProvider
from src.security.jwt_manager import JWTManager, JWTConfig, JWTAlgorithm
from src.security.session_store import SessionStore, SessionStoreConfig
from src.security.process_sandbox import ProcessSandbox, SandboxPolicy
from src.security.security_monitor import SecurityMonitor, ThreatLevel
from src.security.models import (
    AuthProvider, EnhancedUser, WebSession, TokenType,
    SecurityRole, SessionStatus, UserInfo
)
from src.security.security_audit import SecurityEventType


class TestJWTManager:
    """Test JWT token management."""

    @pytest.fixture
    def jwt_manager(self) -> JWTManager:
        """Create JWT manager for testing."""
        return JWTManager(JWTConfig(
            algorithm=JWTAlgorithm.HS256,
            secret_key="test_secret_key_for_testing_only",
            access_token_ttl=900,
            refresh_token_ttl=86400,
        ))

    def test_create_access_token(self, jwt_manager: JWTManager):
        """Test access token creation."""
        token = jwt_manager.create_access_token(
            subject="user123",
            claims={"role": "admin"},
        )
        
        assert token is not None
        assert isinstance(token, str)
        
        # Decode and verify
        claims = jwt_manager.verify_token(token, TokenType.ACCESS)
        assert claims.sub == "user123"
        assert claims.type == TokenType.ACCESS
        assert claims.custom_claims["role"] == "admin"

    def test_create_refresh_token(self, jwt_manager: JWTManager):
        """Test refresh token creation."""
        token = jwt_manager.create_refresh_token(
            subject="user123",
            claims={"role": "admin"},
        )
        
        assert token is not None
        assert isinstance(token, str)
        
        # Decode and verify
        claims = jwt_manager.verify_token(token, TokenType.REFRESH)
        assert claims.sub == "user123"
        assert claims.type == TokenType.REFRESH

    def test_create_token_pair(self, jwt_manager: JWTManager):
        """Test token pair creation."""
        pair = jwt_manager.create_token_pair(
            subject="user123",
            claims={"role": "admin"},
        )
        
        assert pair.access_token is not None
        assert pair.refresh_token is not None
        assert pair.token_type == "Bearer"
        assert pair.expires_in == 900

    def test_refresh_access_token(self, jwt_manager: JWTManager):
        """Test token refresh."""
        # Create initial token pair
        initial_pair = jwt_manager.create_token_pair("user123")
        
        # Refresh using refresh token
        new_pair = jwt_manager.refresh_access_token(initial_pair.refresh_token)
        
        assert new_pair is not None
        assert new_pair.access_token != initial_pair.access_token
        assert new_pair.refresh_token != initial_pair.refresh_token

    def test_token_expiration(self, jwt_manager: JWTManager):
        """Test token expiration validation."""
        # Create token with very short TTL
        token = jwt_manager.create_access_token(
            subject="user123",
            ttl_seconds=1,
        )
        
        # Token should be valid immediately
        claims = jwt_manager.verify_token(token)
        assert claims is not None
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Token should be expired
        with pytest.raises(pyjwt.ExpiredSignatureError):
            jwt_manager.verify_token(token)

    def test_token_revocation(self, jwt_manager: JWTManager):
        """Test token revocation."""
        token = jwt_manager.create_access_token("user123")
        
        # Token should be valid
        claims = jwt_manager.verify_token(token)
        assert claims is not None
        
        # Revoke token
        assert jwt_manager.revoke_token(token) is True
        
        # Token should be invalid after revocation
        with pytest.raises(pyjwt.InvalidTokenError):
            jwt_manager.verify_token(token)

    def test_invalid_token(self, jwt_manager: JWTManager):
        """Test invalid token handling."""
        # Invalid format
        with pytest.raises(pyjwt.InvalidTokenError):
            jwt_manager.verify_token("invalid_token")
        
        # Wrong secret
        wrong_token = pyjwt.encode(
            {"sub": "user123"},
            "wrong_secret",
            algorithm="HS256",
        )
        with pytest.raises(pyjwt.InvalidSignatureError):
            jwt_manager.verify_token(wrong_token)


class TestSessionStore:
    """Test session storage."""

    @pytest.fixture
    async def session_store(self) -> SessionStore:
        """Create session store for testing."""
        store = SessionStore(SessionStoreConfig(
            redis_url="redis://localhost:6379/1",  # Use test database
        ))
        # Use in-memory fallback for testing
        store._redis = None
        return store

    @pytest.mark.asyncio
    async def test_create_session(self, session_store: SessionStore):
        """Test session creation."""
        session = WebSession(
            user_id="user123",
            ip_address="127.0.0.1",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        
        success = await session_store.create_session(session)
        assert success is True
        
        # Retrieve session
        retrieved = await session_store.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.user_id == "user123"

    @pytest.mark.asyncio
    async def test_update_session(self, session_store: SessionStore):
        """Test session update."""
        session = WebSession(
            user_id="user123",
            ip_address="127.0.0.1",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        
        await session_store.create_session(session)
        
        # Update session
        session.metadata["test"] = "value"
        success = await session_store.update_session(session)
        assert success is True
        
        # Verify update
        retrieved = await session_store.get_session(session.session_id)
        assert retrieved.metadata["test"] == "value"

    @pytest.mark.asyncio
    async def test_delete_session(self, session_store: SessionStore):
        """Test session deletion."""
        session = WebSession(
            user_id="user123",
            ip_address="127.0.0.1",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        
        await session_store.create_session(session)
        
        # Delete session
        success = await session_store.delete_session(session.session_id)
        assert success is True
        
        # Session should not exist
        retrieved = await session_store.get_session(session.session_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_user_sessions(self, session_store: SessionStore):
        """Test getting all user sessions."""
        user_id = "user123"
        
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = WebSession(
                user_id=user_id,
                ip_address=f"127.0.0.{i}",
                expires_at=datetime.utcnow() + timedelta(hours=1),
            )
            await session_store.create_session(session)
            sessions.append(session)
        
        # Get user sessions
        user_sessions = await session_store.get_user_sessions(user_id)
        assert len(user_sessions) == 3
        
        # Delete all user sessions
        deleted = await session_store.delete_user_sessions(user_id)
        assert deleted == 3

    @pytest.mark.asyncio
    async def test_session_expiration(self, session_store: SessionStore):
        """Test session expiration handling."""
        # Create expired session
        session = WebSession(
            user_id="user123",
            ip_address="127.0.0.1",
            expires_at=datetime.utcnow() - timedelta(hours=1),  # Already expired
        )
        
        await session_store.create_session(session)
        
        # Session should be detected as expired
        assert session.is_expired() is True
        
        # Cleanup should remove expired sessions
        cleaned = await session_store.cleanup_expired()
        assert cleaned >= 1


class TestProcessSandbox:
    """Test process sandboxing."""

    @pytest.fixture
    def sandbox(self) -> ProcessSandbox:
        """Create sandbox for testing."""
        return ProcessSandbox()

    @pytest.mark.asyncio
    async def test_execute_command(self, sandbox: ProcessSandbox):
        """Test command execution in sandbox."""
        result = await sandbox.execute(
            ["echo", "Hello, World!"],
            timeout=5,
        )
        
        assert result.success is True
        assert "Hello, World!" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_python(self, sandbox: ProcessSandbox):
        """Test Python code execution in sandbox."""
        code = """
print("Test output")
result = 2 + 2
print(f"Result: {result}")
"""
        
        result = await sandbox.execute_python(code, timeout=5)
        
        assert result.success is True
        assert "Test output" in result.stdout
        assert "Result: 4" in result.stdout

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self, sandbox: ProcessSandbox):
        """Test timeout enforcement."""
        code = """
import time
time.sleep(10)  # Sleep longer than timeout
"""
        
        result = await sandbox.execute_python(code, timeout=2)
        
        assert result.timeout is True
        assert result.success is False

    @pytest.mark.asyncio
    async def test_code_validation(self, sandbox: ProcessSandbox):
        """Test code validation for security issues."""
        dangerous_code = """
import os
os.system("rm -rf /")
"""
        
        validation = await sandbox.validate_code(dangerous_code)
        
        assert validation["valid"] is False
        assert len(validation["issues"]) > 0
        assert any("os" in issue for issue in validation["issues"])


class TestSecurityMonitor:
    """Test security monitoring."""

    @pytest.fixture
    def monitor(self) -> SecurityMonitor:
        """Create security monitor for testing."""
        return SecurityMonitor()

    def test_record_event(self, monitor: SecurityMonitor):
        """Test event recording."""
        monitor.record_event(
            SecurityEventType.AUTH_LOGIN_FAILURE,
            ip_address="192.168.1.1",
            user_id="user123",
        )
        
        # Event should be in buffer
        assert len(monitor._event_buffer) == 1

    def test_block_ip(self, monitor: SecurityMonitor):
        """Test IP blocking."""
        ip = "192.168.1.1"
        
        monitor.block_ip(ip)
        assert monitor.is_ip_blocked(ip) is True
        
        monitor.unblock_ip(ip)
        assert monitor.is_ip_blocked(ip) is False

    def test_block_user(self, monitor: SecurityMonitor):
        """Test user blocking."""
        user_id = "user123"
        
        monitor.block_user(user_id)
        assert monitor.is_user_blocked(user_id) is True
        
        monitor.unblock_user(user_id)
        assert monitor.is_user_blocked(user_id) is False

    def test_threat_summary(self, monitor: SecurityMonitor):
        """Test threat summary generation."""
        # Record some events
        for _ in range(5):
            monitor.record_event(
                SecurityEventType.AUTH_LOGIN_FAILURE,
                ip_address="192.168.1.1",
            )
        
        summary = monitor.get_threat_summary()
        
        assert "overall_threat_level" in summary
        assert "active_alerts" in summary
        assert "blocked_ips" in summary


class TestEnhancedSecurityManager:
    """Test enhanced security manager."""

    @pytest.fixture
    async def security_manager(self) -> EnhancedSecurityManager:
        """Create enhanced security manager for testing."""
        config = EnhancedSecurityConfig(
            enable_authentication=True,
            enable_monitoring=True,
            enable_sandboxing=True,
        )
        manager = EnhancedSecurityManager(config)
        # Use in-memory session store for testing
        manager.session_store._redis = None
        return manager

    @pytest.mark.asyncio
    async def test_create_web_session(self, security_manager: EnhancedSecurityManager):
        """Test web session creation."""
        user = EnhancedUser(
            username="testuser",
            email="test@example.com",
            roles={SecurityRole.USER},
        )
        
        security_manager._users[user.user_id] = user
        
        session, tokens = await security_manager.create_web_session(
            user=user,
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        
        assert session is not None
        assert session.user_id == user.user_id
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None

    @pytest.mark.asyncio
    async def test_validate_jwt_token(self, security_manager: EnhancedSecurityManager):
        """Test JWT token validation."""
        user = EnhancedUser(
            username="testuser",
            email="test@example.com",
        )
        
        security_manager._users[user.user_id] = user
        
        # Create token
        token = security_manager.jwt_manager.create_access_token(user.user_id)
        
        # Validate token
        validated_user = await security_manager.validate_jwt_token(token)
        
        assert validated_user is not None
        assert validated_user.user_id == user.user_id

    def test_create_api_key(self, security_manager: EnhancedSecurityManager):
        """Test API key creation."""
        key, api_key_obj = security_manager.create_api_key(
            service_name="test_service",
            permissions={"read", "write"},
            expires_in_days=30,
        )
        
        assert key.startswith(security_manager.enhanced_config.api_key_prefix)
        assert api_key_obj.service_name == "test_service"
        assert "read" in api_key_obj.permissions
        assert "write" in api_key_obj.permissions

    def test_validate_api_key(self, security_manager: EnhancedSecurityManager):
        """Test API key validation."""
        key, _ = security_manager.create_api_key(
            service_name="test_service",
            permissions={"read"},
        )
        
        # Valid key with permission
        api_key = security_manager.validate_api_key(key, "read")
        assert api_key is not None
        
        # Valid key without required permission
        api_key = security_manager.validate_api_key(key, "write")
        assert api_key is None
        
        # Invalid key
        api_key = security_manager.validate_api_key("invalid_key")
        assert api_key is None

    @pytest.mark.asyncio
    async def test_execute_sandboxed(self, security_manager: EnhancedSecurityManager):
        """Test sandboxed execution."""
        result = await security_manager.execute_sandboxed(
            command=["echo", "test"],
            policy=SandboxPolicy.STRICT,
            timeout=5,
        )
        
        assert "stdout" in result
        assert "test" in result["stdout"]

    @pytest.mark.asyncio
    async def test_security_event_handling(self, security_manager: EnhancedSecurityManager):
        """Test security event handling."""
        await security_manager.handle_security_event(
            SecurityEventType.AUTH_LOGIN_FAILURE,
            ip_address="192.168.1.1",
            user_id="user123",
            details={"reason": "Invalid password"},
        )
        
        # Event should be recorded in monitor
        assert len(security_manager.security_monitor._event_buffer) > 0

    @pytest.mark.asyncio
    async def test_check_request_security(self, security_manager: EnhancedSecurityManager):
        """Test request security checking."""
        # Normal request should pass
        allowed, reason = await security_manager.check_request_security(
            ip_address="192.168.1.1",
            user_id="user123",
        )
        assert allowed is True
        assert reason is None
        
        # Block IP and test
        security_manager.security_monitor.block_ip("192.168.1.1")
        allowed, reason = await security_manager.check_request_security(
            ip_address="192.168.1.1",
        )
        assert allowed is False
        assert "blocked" in reason.lower()

    def test_security_dashboard(self, security_manager: EnhancedSecurityManager):
        """Test security dashboard data generation."""
        dashboard = security_manager.get_security_dashboard()
        
        assert "threat_summary" in dashboard
        assert "blocked_ips" in dashboard
        assert "blocked_users" in dashboard
        assert "recent_alerts" in dashboard
        assert "metrics" in dashboard


@pytest.mark.asyncio
async def test_oauth_flow():
    """Test OAuth authentication flow."""
    manager = EnhancedSecurityManager()
    
    # Mock OAuth provider
    with patch.object(manager, "_oauth_providers") as mock_providers:
        mock_provider = MagicMock()
        mock_provider.generate_state.return_value = MagicMock(
            state="test_state",
            is_expired=MagicMock(return_value=False),
        )
        mock_provider.get_authorization_url.return_value = "https://oauth.example.com/auth"
        mock_providers.get.return_value = mock_provider
        
        # Get login URL
        url = manager.get_oauth_login_url(OAuthProvider.GOOGLE)
        assert url is not None
        assert "oauth.example.com" in url


@pytest.mark.asyncio
async def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    manager = EnhancedSecurityManager()
    
    # Start monitoring
    await manager.security_monitor.start_monitoring()
    
    # Record multiple events
    for i in range(100):
        manager.security_monitor.record_event(
            SecurityEventType.AUTH_LOGIN_SUCCESS,
            ip_address=f"192.168.1.{i % 10}",
            user_id=f"user{i % 5}",
        )
    
    # Get metrics
    metrics = manager.security_monitor.get_metrics(hours=1)
    assert len(metrics) >= 0
    
    # Stop monitoring
    await manager.security_monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])