"""Security models and data structures."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl


class AuthProvider(Enum):
    """Authentication providers."""

    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    DISCORD = "discord"
    LDAP = "ldap"
    SAML = "saml"


class TokenType(Enum):
    """JWT token types."""

    ACCESS = "access"
    REFRESH = "refresh"
    ID = "id"
    API_KEY = "api_key"


class SessionStatus(Enum):
    """Session status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    LOCKED = "locked"


class SecurityRole(Enum):
    """Security roles."""

    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"
    SERVICE = "service"  # For service accounts


class ResourcePermission(BaseModel):
    """Resource-specific permission."""

    resource_type: str
    resource_id: str
    permissions: Set[str] = Field(default_factory=set)
    granted_by: Optional[str] = None
    granted_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if permission is expired."""
        return self.expires_at and datetime.utcnow() > self.expires_at


class EnhancedUser(BaseModel):
    """Enhanced user model with OAuth and session support."""

    user_id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    email: Optional[str] = None
    email_verified: bool = False
    phone: Optional[str] = None
    phone_verified: bool = False
    
    # Authentication
    auth_provider: AuthProvider = AuthProvider.LOCAL
    provider_user_id: Optional[str] = None
    password_hash: Optional[str] = None  # Only for local auth
    
    # Profile
    full_name: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[HttpUrl] = None
    locale: str = "en-US"
    timezone: str = "UTC"
    
    # Security
    roles: Set[SecurityRole] = Field(default_factory=lambda: {SecurityRole.USER})
    permissions: Set[str] = Field(default_factory=set)
    resource_permissions: List[ResourcePermission] = Field(default_factory=list)
    
    # Account status
    is_active: bool = True
    is_verified: bool = False
    is_locked: bool = False
    lockout_until: Optional[datetime] = None
    failed_login_attempts: int = 0
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Security settings
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    backup_codes: List[str] = Field(default_factory=list)
    trusted_devices: List[str] = Field(default_factory=list)
    
    def has_role(self, role: SecurityRole) -> bool:
        """Check if user has role."""
        return role in self.roles or SecurityRole.ADMIN in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has permission."""
        return (
            permission in self.permissions
            or SecurityRole.ADMIN in self.roles
        )

    def has_resource_permission(
        self,
        resource_type: str,
        resource_id: str,
        permission: str,
    ) -> bool:
        """Check if user has resource-specific permission."""
        if SecurityRole.ADMIN in self.roles:
            return True
        
        for rp in self.resource_permissions:
            if (
                rp.resource_type == resource_type
                and rp.resource_id == resource_id
                and permission in rp.permissions
                and not rp.is_expired()
            ):
                return True
        
        return False

    def is_account_locked(self) -> bool:
        """Check if account is locked."""
        if not self.is_locked:
            return False
        
        if self.lockout_until and datetime.utcnow() > self.lockout_until:
            # Auto-unlock if lockout period expired
            self.is_locked = False
            self.lockout_until = None
            self.failed_login_attempts = 0
            return False
        
        return True


class WebSession(BaseModel):
    """Enhanced web session with device tracking."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    
    # Tokens
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    csrf_token: str = Field(default_factory=lambda: str(uuid4()))
    
    # Session info
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    idle_timeout_minutes: int = 30
    absolute_timeout_hours: int = 24
    
    # Device/Client info
    ip_address: str
    user_agent: Optional[str] = None
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    device_type: Optional[str] = None  # mobile, tablet, desktop
    browser: Optional[str] = None
    os: Optional[str] = None
    
    # Security
    is_trusted: bool = False
    requires_mfa: bool = False
    mfa_verified: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        now = datetime.utcnow()
        
        # Check absolute timeout
        if now > self.expires_at:
            return True
        
        # Check idle timeout
        idle_limit = self.last_activity_at + timedelta(minutes=self.idle_timeout_minutes)
        if now > idle_limit:
            return True
        
        return False


class ApiKey(BaseModel):
    """API key for service authentication."""

    key_id: str = Field(default_factory=lambda: str(uuid4()))
    key_hash: str  # Hashed API key
    key_prefix: str  # First few chars for identification
    
    # Ownership
    user_id: Optional[str] = None
    service_name: str
    description: Optional[str] = None
    
    # Permissions
    roles: Set[SecurityRole] = Field(default_factory=lambda: {SecurityRole.SERVICE})
    permissions: Set[str] = Field(default_factory=set)
    allowed_ips: List[str] = Field(default_factory=list)
    allowed_origins: List[str] = Field(default_factory=list)
    
    # Rate limits
    rate_limit_per_minute: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None
    rate_limit_per_day: Optional[int] = None
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    usage_count: int = 0
    
    # Status
    is_active: bool = True
    is_revoked: bool = False
    revoked_at: Optional[datetime] = None
    revoked_reason: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        if not self.is_active or self.is_revoked:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True

    def check_ip(self, ip: str) -> bool:
        """Check if IP is allowed."""
        if not self.allowed_ips:
            return True
        return ip in self.allowed_ips

    def check_origin(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if not self.allowed_origins:
            return True
        return origin in self.allowed_origins


class OAuthTokenResponse(BaseModel):
    """OAuth token response."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: List[str] = Field(default_factory=list)
    id_token: Optional[str] = None


class UserInfo(BaseModel):
    """User info from OAuth provider."""

    provider: AuthProvider
    provider_user_id: str
    email: Optional[str] = None
    email_verified: bool = False
    name: Optional[str] = None
    username: Optional[str] = None
    picture: Optional[str] = None
    locale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JWTClaims(BaseModel):
    """JWT token claims."""

    sub: str  # Subject (user ID)
    type: TokenType
    iat: datetime  # Issued at
    exp: datetime  # Expires at
    nbf: datetime  # Not before
    iss: Optional[str] = None  # Issuer
    aud: Optional[List[str]] = None  # Audience
    jti: Optional[str] = None  # JWT ID
    custom_claims: Dict[str, Any] = Field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now(datetime.timezone.utc) > self.exp


class TokenPair(BaseModel):
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int


class SecurityMetrics(BaseModel):
    """Security metrics and statistics."""

    # Authentication metrics
    total_users: int = 0
    active_users: int = 0
    locked_users: int = 0
    verified_users: int = 0
    
    # Session metrics
    active_sessions: int = 0
    expired_sessions: int = 0
    average_session_duration: float = 0.0
    
    # API key metrics
    total_api_keys: int = 0
    active_api_keys: int = 0
    revoked_api_keys: int = 0
    
    # Security events
    failed_login_attempts: int = 0
    successful_logins: int = 0
    password_resets: int = 0
    mfa_challenges: int = 0
    
    # Rate limiting
    rate_limit_hits: int = 0
    blocked_requests: int = 0
    
    # Threats
    suspicious_activities: int = 0
    blocked_ips: List[str] = Field(default_factory=list)
    
    # Time range
    period_start: datetime
    period_end: datetime
    
