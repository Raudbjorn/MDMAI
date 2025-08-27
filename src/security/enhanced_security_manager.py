"""Enhanced security manager with OAuth2, JWT, and process isolation."""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from config.logging_config import get_logger
from config.settings import settings

from src.security.security_manager import SecurityManager, SecurityConfig
from src.security.auth_providers import (
    OAuthProvider, OAuthProviderFactory, OAuthState
)
from src.security.jwt_manager import JWTManager, JWTConfig
from src.security.session_store import SessionStore, SessionStoreConfig
from src.security.process_sandbox import ProcessSandbox, SandboxConfig, SandboxPolicy
from src.security.security_monitor import SecurityMonitor, ThreatLevel
from src.security.models import (
    AuthProvider, EnhancedUser, WebSession, ApiKey, TokenPair,
    SessionStatus, SecurityRole, TokenType
)
from src.security.security_audit import SecurityEventType, SecuritySeverity

logger = get_logger(__name__)


class EnhancedSecurityConfig(SecurityConfig):
    """Enhanced security configuration."""

    # OAuth2 configuration
    oauth_providers: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    oauth_redirect_base: str = "http://localhost:8000/auth/callback"
    
    # JWT configuration
    jwt_algorithm: str = "RS256"
    jwt_access_ttl: int = 900  # 15 minutes
    jwt_refresh_ttl: int = 2592000  # 30 days
    jwt_issuer: str = "ttrpg-assistant"
    
    # Session configuration
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_hours: int = 24
    max_sessions_per_user: int = 10
    
    # Sandboxing configuration
    sandbox_policy: SandboxPolicy = SandboxPolicy.STRICT
    enable_sandboxing: bool = True
    max_sandbox_cpu_seconds: int = 30
    max_sandbox_memory_mb: int = 512
    
    # Security monitoring
    enable_monitoring: bool = True
    threat_detection: bool = True
    auto_block_threats: bool = True
    
    # API key configuration
    api_key_prefix: str = "ttrpg_"
    api_key_length: int = 32
    
    # MFA configuration
    enable_mfa: bool = False
    mfa_issuer: str = "TTRPG Assistant"


class EnhancedSecurityManager(SecurityManager):
    """Enhanced security manager with web authentication features."""

    def __init__(self, config: Optional[EnhancedSecurityConfig] = None):
        """Initialize enhanced security manager."""
        # Initialize base security manager
        super().__init__(config or EnhancedSecurityConfig())
        self.enhanced_config: EnhancedSecurityConfig = self.config  # type: ignore
        
        # Initialize new components
        self.jwt_manager = JWTManager(
            JWTConfig(
                access_token_ttl=self.enhanced_config.jwt_access_ttl,
                refresh_token_ttl=self.enhanced_config.jwt_refresh_ttl,
                issuer=self.enhanced_config.jwt_issuer,
            )
        )
        
        self.session_store = SessionStore(
            SessionStoreConfig(
                redis_url=self.enhanced_config.redis_url,
                session_ttl_seconds=self.enhanced_config.session_ttl_hours * 3600,
                max_sessions_per_user=self.enhanced_config.max_sessions_per_user,
            )
        )
        
        self.security_monitor = SecurityMonitor()
        
        # OAuth providers
        self._oauth_providers: Dict[OAuthProvider, Any] = {}
        self._oauth_states: Dict[str, OAuthState] = {}
        
        # API keys
        self._api_keys: Dict[str, ApiKey] = {}
        
        # Enhanced users
        self._users: Dict[str, EnhancedUser] = {}
        
        # Initialize OAuth providers
        self._initialize_oauth_providers()
        
        logger.info(
            "Enhanced security manager initialized",
            jwt_enabled=True,
            oauth_providers=list(self._oauth_providers.keys()),
            monitoring_enabled=self.enhanced_config.enable_monitoring,
            sandboxing_enabled=self.enhanced_config.enable_sandboxing,
        )

    def _initialize_oauth_providers(self) -> None:
        """Initialize OAuth providers from configuration."""
        for provider_name, config in self.enhanced_config.oauth_providers.items():
            try:
                provider = OAuthProvider(provider_name.upper())
                redirect_uri = f"{self.enhanced_config.oauth_redirect_base}/{provider_name}"
                
                oauth_provider = OAuthProviderFactory.create(
                    provider=provider,
                    client_id=config["client_id"],
                    client_secret=config["client_secret"],
                    redirect_uri=redirect_uri,
                )
                
                self._oauth_providers[provider] = oauth_provider
                logger.info(f"Initialized OAuth provider: {provider_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize OAuth provider {provider_name}: {e}")

    async def start(self) -> None:
        """Start enhanced security services."""
        # Connect to Redis
        await self.session_store.connect()
        
        # Start security monitoring
        if self.enhanced_config.enable_monitoring:
            await self.security_monitor.start_monitoring()
        
        logger.info("Enhanced security services started")

    async def stop(self) -> None:
        """Stop enhanced security services."""
        # Stop monitoring
        await self.security_monitor.stop_monitoring()
        
        # Disconnect from Redis
        await self.session_store.disconnect()
        
        logger.info("Enhanced security services stopped")

    # OAuth2 Authentication Methods

    def get_oauth_login_url(
        self,
        provider: OAuthProvider,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Get OAuth login URL for provider."""
        oauth_provider = self._oauth_providers.get(provider)
        if not oauth_provider:
            return None
        
        # Generate state for CSRF protection
        state = oauth_provider.generate_state(metadata)
        self._oauth_states[state.state] = state
        
        # Get authorization URL
        return oauth_provider.get_authorization_url(state)

    async def handle_oauth_callback(
        self,
        provider: OAuthProvider,
        code: str,
        state: str,
    ) -> Optional[Tuple[EnhancedUser, TokenPair]]:
        """Handle OAuth callback and create user session."""
        # Validate state
        oauth_state = self._oauth_states.get(state)
        if not oauth_state or oauth_state.is_expired():
            logger.warning("Invalid or expired OAuth state")
            return None
        
        # Remove state (one-time use)
        del self._oauth_states[state]
        
        # Get OAuth provider
        oauth_provider = self._oauth_providers.get(provider)
        if not oauth_provider:
            return None
        
        try:
            # Exchange code for tokens
            token_response = await oauth_provider.exchange_code(code, oauth_state)
            
            # Get user info
            user_info = await oauth_provider.get_user_info(token_response.access_token)
            
            # Find or create user
            user = await self._find_or_create_oauth_user(user_info)
            
            # Create JWT tokens
            token_pair = self.jwt_manager.create_token_pair(
                subject=user.user_id,
                claims={
                    "username": user.username,
                    "email": user.email,
                    "roles": [r.value for r in user.roles],
                    "provider": user.auth_provider.value,
                },
            )
            
            # Update last login
            user.last_login_at = datetime.utcnow()
            
            # Log successful login
            self.audit_trail.log_event(
                SecurityEventType.AUTH_LOGIN_SUCCESS,
                SecuritySeverity.INFO,
                f"OAuth login via {provider.value}",
                user_id=user.user_id,
            )
            
            return user, token_pair
            
        except Exception as e:
            logger.error(f"OAuth callback failed: {e}")
            self.audit_trail.log_event(
                SecurityEventType.AUTH_LOGIN_FAILURE,
                SecuritySeverity.WARNING,
                f"OAuth login failed via {provider.value}: {str(e)}",
            )
            return None

    async def _find_or_create_oauth_user(self, user_info: Any) -> EnhancedUser:
        """Find existing user or create new one from OAuth info."""
        # Look for existing user by provider ID
        for user in self._users.values():
            if (
                user.auth_provider == user_info.provider
                and user.provider_user_id == user_info.provider_user_id
            ):
                return user
        
        # Look for existing user by email
        if user_info.email:
            for user in self._users.values():
                if user.email == user_info.email:
                    # Link OAuth provider to existing user
                    user.auth_provider = user_info.provider
                    user.provider_user_id = user_info.provider_user_id
                    return user
        
        # Create new user
        user = EnhancedUser(
            username=user_info.username or user_info.email or f"user_{user_info.provider_user_id}",
            email=user_info.email,
            email_verified=user_info.email_verified,
            auth_provider=user_info.provider,
            provider_user_id=user_info.provider_user_id,
            full_name=user_info.name,
            avatar_url=user_info.picture,
            locale=user_info.locale or "en-US",
            is_verified=user_info.email_verified,
            roles={SecurityRole.USER},
        )
        
        self._users[user.user_id] = user
        
        logger.info(f"Created new OAuth user: {user.username}")
        return user

    # JWT Token Methods

    async def create_web_session(
        self,
        user: EnhancedUser,
        ip_address: str,
        user_agent: Optional[str] = None,
        device_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[WebSession, TokenPair]:
        """Create web session with JWT tokens."""
        # Create JWT token pair
        token_pair = self.jwt_manager.create_token_pair(
            subject=user.user_id,
            claims={
                "username": user.username,
                "email": user.email,
                "roles": [r.value for r in user.roles],
            },
        )
        
        # Create session
        session = WebSession(
            user_id=user.user_id,
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + timedelta(hours=self.enhanced_config.session_ttl_hours),
            device_id=device_info.get("device_id") if device_info else None,
            device_name=device_info.get("device_name") if device_info else None,
            device_type=device_info.get("device_type") if device_info else None,
            browser=device_info.get("browser") if device_info else None,
            os=device_info.get("os") if device_info else None,
        )
        
        # Store session
        await self.session_store.create_session(session)
        
        # Log session creation
        self.audit_trail.log_event(
            SecurityEventType.SESSION_CREATED,
            SecuritySeverity.INFO,
            f"Web session created for user {user.username}",
            user_id=user.user_id,
            session_id=session.session_id,
            ip_address=ip_address,
        )
        
        return session, token_pair

    async def validate_jwt_token(
        self,
        token: str,
        expected_type: TokenType = TokenType.ACCESS,
    ) -> Optional[EnhancedUser]:
        """Validate JWT token and return user."""
        try:
            # Verify token
            claims = self.jwt_manager.verify_token(token, expected_type)
            
            # Get user
            user = self._users.get(claims.sub)
            if not user:
                logger.warning(f"User not found for token: {claims.sub}")
                return None
            
            # Check if user is blocked
            if self.security_monitor.is_user_blocked(user.user_id):
                logger.warning(f"Blocked user attempted access: {user.user_id}")
                return None
            
            # Check if account is locked
            if user.is_account_locked():
                logger.warning(f"Locked account attempted access: {user.user_id}")
                return None
            
            return user
            
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return None

    async def refresh_jwt_tokens(self, refresh_token: str) -> Optional[TokenPair]:
        """Refresh JWT tokens using refresh token."""
        try:
            return self.jwt_manager.refresh_access_token(refresh_token)
        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")
            return None

    # API Key Methods

    def create_api_key(
        self,
        service_name: str,
        user_id: Optional[str] = None,
        permissions: Optional[Set[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, ApiKey]:
        """Create new API key."""
        # Generate key
        raw_key = secrets.token_urlsafe(self.enhanced_config.api_key_length)
        full_key = f"{self.enhanced_config.api_key_prefix}{raw_key}"
        
        # Hash key for storage
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        # Create API key object
        api_key = ApiKey(
            key_hash=key_hash,
            key_prefix=full_key[:8],  # Store prefix for identification
            user_id=user_id,
            service_name=service_name,
            permissions=permissions or set(),
            expires_at=(
                datetime.utcnow() + timedelta(days=expires_in_days)
                if expires_in_days
                else None
            ),
        )
        
        # Store API key
        self._api_keys[api_key.key_id] = api_key
        
        logger.info(f"Created API key for service: {service_name}")
        
        return full_key, api_key

    def validate_api_key(
        self,
        key: str,
        required_permission: Optional[str] = None,
    ) -> Optional[ApiKey]:
        """Validate API key."""
        # Hash provided key
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Find matching API key
        for api_key in self._api_keys.values():
            if api_key.key_hash == key_hash:
                # Check if valid
                if not api_key.is_valid():
                    logger.warning(f"Invalid API key used: {api_key.key_id}")
                    return None
                
                # Check permission if required
                if required_permission and required_permission not in api_key.permissions:
                    logger.warning(f"API key lacks permission: {required_permission}")
                    return None
                
                # Update usage
                api_key.last_used_at = datetime.utcnow()
                api_key.usage_count += 1
                
                return api_key
        
        logger.warning("Unknown API key attempted")
        return None

    def revoke_api_key(self, key_id: str, reason: str = "") -> bool:
        """Revoke API key."""
        if key_id in self._api_keys:
            api_key = self._api_keys[key_id]
            api_key.is_revoked = True
            api_key.revoked_at = datetime.utcnow()
            api_key.revoked_reason = reason
            
            logger.info(f"Revoked API key: {key_id}")
            return True
        
        return False

    # Process Sandboxing Methods

    async def execute_sandboxed(
        self,
        command: List[str],
        user_id: Optional[str] = None,
        policy: Optional[SandboxPolicy] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute command in sandbox."""
        if not self.enhanced_config.enable_sandboxing:
            logger.warning("Sandboxing is disabled")
            return {"error": "Sandboxing is disabled"}
        
        # Use configured policy or default
        policy = policy or self.enhanced_config.sandbox_policy
        
        # Create sandbox configuration
        sandbox_config = SandboxConfig.from_policy(policy)
        
        # Override resource limits if needed
        if self.enhanced_config.max_sandbox_cpu_seconds:
            sandbox_config.resource_limits.cpu_time_seconds = self.enhanced_config.max_sandbox_cpu_seconds
        if self.enhanced_config.max_sandbox_memory_mb:
            sandbox_config.resource_limits.memory_mb = self.enhanced_config.max_sandbox_memory_mb
        
        # Create sandbox
        sandbox = ProcessSandbox(sandbox_config)
        
        # Log sandbox execution
        self.audit_trail.log_event(
            SecurityEventType.SANDBOX_EXECUTION,
            SecuritySeverity.INFO,
            f"Executing command in sandbox: {command[0]}",
            user_id=user_id,
            details={"policy": policy.value, "command": command},
        )
        
        # Execute
        result = await sandbox.execute(command, timeout=timeout)
        
        # Check for violations
        if result.violations:
            self.security_monitor.record_event(
                SecurityEventType.SECURITY_VIOLATION,
                user_id=user_id,
                details={"violations": result.violations},
            )
        
        return result.model_dump()

    async def execute_python_sandboxed(
        self,
        code: str,
        user_id: Optional[str] = None,
        policy: Optional[SandboxPolicy] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute Python code in sandbox."""
        if not self.enhanced_config.enable_sandboxing:
            logger.warning("Sandboxing is disabled")
            return {"error": "Sandboxing is disabled"}
        
        # Validate code first
        sandbox = ProcessSandbox()
        validation = await sandbox.validate_code(code, "python")
        
        if not validation["valid"]:
            logger.warning(f"Code validation failed: {validation['issues']}")
            return {"error": "Code validation failed", "issues": validation["issues"]}
        
        # Use configured policy or default
        policy = policy or self.enhanced_config.sandbox_policy
        
        # Create sandbox configuration
        sandbox_config = SandboxConfig.from_policy(policy)
        sandbox = ProcessSandbox(sandbox_config)
        
        # Execute
        result = await sandbox.execute_python(code, timeout=timeout)
        
        return result.model_dump()

    # Security Monitoring Methods

    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        return {
            "threat_summary": self.security_monitor.get_threat_summary(),
            "active_sessions": asyncio.create_task(self.session_store.get_session_count()),
            "active_users": len(asyncio.create_task(self.session_store.get_active_users())),
            "blocked_ips": list(self.security_monitor._blocked_ips),
            "blocked_users": list(self.security_monitor._blocked_users),
            "recent_alerts": [
                alert.model_dump()
                for alert in self.security_monitor.get_active_alerts()[:10]
            ],
            "metrics": [
                m.to_dict()
                for m in self.security_monitor.get_metrics(hours=24)
            ],
        }

    async def check_request_security(
        self,
        ip_address: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check if request should be allowed based on security rules."""
        # Check if IP is blocked
        if self.security_monitor.is_ip_blocked(ip_address):
            return False, "IP address is blocked"
        
        # Check if user is blocked
        if user_id and self.security_monitor.is_user_blocked(user_id):
            return False, "User account is blocked"
        
        # Check if session is suspicious
        if session_id and self.security_monitor.is_session_suspicious(session_id):
            return False, "Session flagged as suspicious"
        
        return True, None

    async def handle_security_event(
        self,
        event_type: SecurityEventType,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle security event and trigger monitoring."""
        # Record in audit trail
        self.audit_trail.log_event(
            event_type,
            SecuritySeverity.INFO,
            f"Security event: {event_type.value}",
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            details=details,
        )
        
        # Record in security monitor
        self.security_monitor.record_event(
            event_type,
            ip_address=ip_address,
            user_id=user_id,
            session_id=session_id,
            details=details,
        )

    async def perform_enhanced_maintenance(self) -> Dict[str, Any]:
        """Perform enhanced security maintenance tasks."""
        results = await super().perform_security_maintenance()
        
        # Clean up expired OAuth states
        expired_states = []
        for state_id, state in self._oauth_states.items():
            if state.is_expired(ttl_minutes=30):
                expired_states.append(state_id)
        
        for state_id in expired_states:
            del self._oauth_states[state_id]
        
        results["oauth_states_cleaned"] = len(expired_states)
        
        # Clean up expired sessions in store
        results["sessions_cleaned"] = await self.session_store.cleanup_expired()
        
        # Clean up JWT tokens
        refresh_cleaned, _ = self.jwt_manager.cleanup_expired()
        results["tokens_cleaned"] = refresh_cleaned
        
        return results