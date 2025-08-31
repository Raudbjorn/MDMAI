"""Web UI security integration with FastAPI."""

from typing import Any, Dict, Optional

from fastapi import (
    FastAPI, HTTPException, Depends, Security, status,
    Request, Cookie
)
from fastapi.security import (
    HTTPBearer, HTTPAuthorizationCredentials,
    OAuth2AuthorizationCodeBearer, APIKeyHeader
)
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from config.logging_config import get_logger
from src.security.enhanced_security_manager import (
    EnhancedSecurityManager
)
from src.security.auth_providers import OAuthProvider
from src.security.models import (
    EnhancedUser, SecurityRole
)
from src.security.security_audit import SecurityEventType

logger = get_logger(__name__)


# Security scheme definitions
bearer_scheme = HTTPBearer()
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
    tokenUrl="/auth/token",
    scopes={
        "openid": "OpenID Connect",
        "email": "Email access",
        "profile": "Profile information",
    },
)


class LoginRequest(BaseModel):
    """Login request model."""
    
    username: str
    password: str
    remember_me: bool = False
    device_info: Optional[Dict[str, Any]] = None


class RefreshRequest(BaseModel):
    """Token refresh request."""
    
    refresh_token: str


class SecurityMiddleware:
    """Security middleware for FastAPI."""

    def __init__(self, security_manager: EnhancedSecurityManager):
        """Initialize security middleware."""
        self.security_manager = security_manager

    async def __call__(self, request: Request, call_next: Any) -> Any:
        """Process request through security checks."""
        # Get client info
        ip_address = request.client.host if request.client else "unknown"
        
        # Check if IP is blocked
        allowed, reason = await self.security_manager.check_request_security(
            ip_address=ip_address,
        )
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": reason or "Access denied"},
            )
        
        # Process request
        response = await call_next(request)
        
        return response


class SecurityDependencies:
    """Security dependencies for FastAPI routes."""

    def __init__(self, security_manager: EnhancedSecurityManager):
        """Initialize security dependencies."""
        self.security_manager = security_manager

    async def get_current_user_bearer(
        self,
        credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
        request: Request = None,
    ) -> EnhancedUser:
        """Get current user from Bearer token."""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Validate JWT token
        user = await self.security_manager.validate_jwt_token(credentials.credentials)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Record activity
        ip_address = request.client.host if request and request.client else None
        await self.security_manager.handle_security_event(
            SecurityEventType.API_ACCESS,
            ip_address=ip_address,
            user_id=user.user_id,
        )
        
        return user

    async def get_current_user_api_key(
        self,
        api_key: Optional[str] = Security(api_key_scheme),
        request: Request = None,
    ) -> Optional[Dict[str, Any]]:
        """Validate API key."""
        if not api_key:
            return None
        
        # Validate API key
        api_key_obj = self.security_manager.validate_api_key(api_key)
        
        if not api_key_obj:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        
        # Check IP restrictions
        if request and request.client:
            ip_address = request.client.host
            if not api_key_obj.check_ip(ip_address):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key not authorized from this IP",
                )
        
        return {
            "service_name": api_key_obj.service_name,
            "permissions": list(api_key_obj.permissions),
            "key_id": api_key_obj.key_id,
        }

    async def get_current_user(
        self,
        bearer_user: Optional[EnhancedUser] = None,
        api_key_info: Optional[Dict[str, Any]] = None,
        session_token: Optional[str] = Cookie(None),
        request: Request = None,
    ) -> EnhancedUser:
        """Get current user from any authentication method."""
        # Try Bearer token first
        if bearer_user:
            return bearer_user
        
        # Try session cookie
        if session_token:
            user = await self.security_manager.validate_session(session_token)
            if user:
                return user
        
        # Try API key (service account)
        if api_key_info:
            # Create service user
            return EnhancedUser(
                user_id=api_key_info["key_id"],
                username=api_key_info["service_name"],
                roles={SecurityRole.SERVICE},
                permissions=set(api_key_info["permissions"]),
            )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    def require_permission(self, permission: str) -> Any:
        """Require specific permission."""
        async def permission_checker(
            user: EnhancedUser = Depends(self.get_current_user),
        ) -> EnhancedUser:
            if not user.has_permission(permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission}",
                )
            return user
        
        return permission_checker

    def require_role(self, role: SecurityRole) -> Any:
        """Require specific role."""
        async def role_checker(
            user: EnhancedUser = Depends(self.get_current_user),
        ) -> EnhancedUser:
            if not user.has_role(role):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role required: {role.value}",
                )
            return user
        
        return role_checker


def create_security_routes(
    app: FastAPI,
    security_manager: EnhancedSecurityManager,
) -> None:
    """Create security-related routes."""
    deps = SecurityDependencies(security_manager)
    
    @app.post("/auth/login")
    async def login(
        request: Request,
        login_data: LoginRequest,
    ) -> Dict[str, Any]:
        """Local authentication endpoint."""
        ip_address = request.client.host if request.client else "unknown"
        
        # Authenticate user
        session_token = security_manager.authenticate(
            username=login_data.username,
            password=login_data.password,
            ip_address=ip_address,
        )
        
        if not session_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        
        # Get user
        user = security_manager.validate_session(session_token)
        
        # Create web session with JWT
        session, tokens = await security_manager.create_web_session(
            user=user,
            ip_address=ip_address,
            user_agent=request.headers.get("User-Agent"),
            device_info=login_data.device_info,
        )
        
        return {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "token_type": tokens.token_type,
            "expires_in": tokens.expires_in,
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": [r.value for r in user.roles],
            },
        }
    
    @app.get("/auth/oauth/{provider}")
    async def oauth_login(provider: str) -> RedirectResponse:
        """Initiate OAuth login."""
        try:
            oauth_provider = OAuthProvider(provider.upper())
            login_url = security_manager.get_oauth_login_url(oauth_provider)
            
            if not login_url:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"OAuth provider not configured: {provider}",
                )
            
            return RedirectResponse(url=login_url)
            
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid OAuth provider: {provider}",
            )
    
    @app.get("/auth/callback/{provider}")
    async def oauth_callback(
        request: Request,
        provider: str,
        code: str,
        state: str,
    ) -> Dict[str, Any]:
        """OAuth callback endpoint."""
        try:
            oauth_provider = OAuthProvider(provider.upper())
            
            # Handle OAuth callback
            result = await security_manager.handle_oauth_callback(
                provider=oauth_provider,
                code=code,
                state=state,
            )
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="OAuth authentication failed",
                )
            
            user, tokens = result
            ip_address = request.client.host if request.client else "unknown"
            
            # Create web session
            session, _ = await security_manager.create_web_session(
                user=user,
                ip_address=ip_address,
                user_agent=request.headers.get("User-Agent"),
            )
            
            return {
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token,
                "token_type": tokens.token_type,
                "expires_in": tokens.expires_in,
                "user": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "roles": [r.value for r in user.roles],
                },
            }
            
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid OAuth provider: {provider}",
            )
    
    @app.post("/auth/refresh")
    async def refresh_token(refresh_data: RefreshRequest) -> Dict[str, Any]:
        """Refresh access token."""
        tokens = await security_manager.refresh_jwt_tokens(refresh_data.refresh_token)
        
        if not tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        
        return {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "token_type": tokens.token_type,
            "expires_in": tokens.expires_in,
        }
    
    @app.post("/auth/logout")
    async def logout(
        user: EnhancedUser = Depends(deps.get_current_user),
        session_token: Optional[str] = Cookie(None),
    ) -> Dict[str, str]:
        """Logout endpoint."""
        # Invalidate session
        if session_token:
            # Get session from store and delete
            sessions = await security_manager.session_store.get_user_sessions(user.user_id)
            for session in sessions:
                await security_manager.session_store.delete_session(session.session_id)
        
        return {"message": "Logged out successfully"}
    
    @app.get("/auth/me")
    async def get_current_user_info(
        user: EnhancedUser = Depends(deps.get_current_user),
    ) -> Dict[str, Any]:
        """Get current user information."""
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "roles": [r.value for r in user.roles],
            "permissions": list(user.permissions),
            "verified": user.is_verified,
            "mfa_enabled": user.mfa_enabled,
        }
    
    @app.get("/security/dashboard")
    async def security_dashboard(
        user: EnhancedUser = Depends(deps.require_role(SecurityRole.ADMIN)),
    ) -> Dict[str, Any]:
        """Get security dashboard data."""
        return security_manager.get_security_dashboard()
    
    @app.get("/security/alerts")
    async def get_security_alerts(
        user: EnhancedUser = Depends(deps.require_role(SecurityRole.ADMIN)),
    ) -> List[Dict[str, Any]]:
        """Get active security alerts."""
        alerts = security_manager.security_monitor.get_active_alerts()
        return [alert.model_dump() for alert in alerts]
    
    @app.post("/security/alerts/{alert_id}/resolve")
    async def resolve_alert(
        alert_id: str,
        user: EnhancedUser = Depends(deps.require_role(SecurityRole.ADMIN)),
    ) -> Dict[str, Any]:
        """Resolve security alert."""
        success = security_manager.security_monitor.resolve_alert(
            alert_id,
            resolved_by=user.user_id,
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found",
            )
        
        return {"message": "Alert resolved", "alert_id": alert_id}
    
    @app.post("/security/block/ip/{ip_address}")
    async def block_ip(
        ip_address: str,
        user: EnhancedUser = Depends(deps.require_role(SecurityRole.ADMIN)),
    ) -> Dict[str, str]:
        """Block IP address."""
        security_manager.security_monitor.block_ip(ip_address)
        return {"message": f"IP {ip_address} blocked"}
    
    @app.delete("/security/block/ip/{ip_address}")
    async def unblock_ip(
        ip_address: str,
        user: EnhancedUser = Depends(deps.require_role(SecurityRole.ADMIN)),
    ) -> Dict[str, str]:
        """Unblock IP address."""
        security_manager.security_monitor.unblock_ip(ip_address)
        return {"message": f"IP {ip_address} unblocked"}
    
    @app.post("/security/block/user/{user_id}")
    async def block_user(
        user_id: str,
        user: EnhancedUser = Depends(deps.require_role(SecurityRole.ADMIN)),
    ) -> Dict[str, str]:
        """Block user account."""
        security_manager.security_monitor.block_user(user_id)
        return {"message": f"User {user_id} blocked"}
    
    @app.delete("/security/block/user/{user_id}")
    async def unblock_user(
        user_id: str,
        user: EnhancedUser = Depends(deps.require_role(SecurityRole.ADMIN)),
    ) -> Dict[str, str]:
        """Unblock user account."""
        security_manager.security_monitor.unblock_user(user_id)
        return {"message": f"User {user_id} unblocked"}
    
    @app.post("/security/api-keys")
    async def create_api_key(
        service_name: str,
        permissions: List[str],
        expires_in_days: Optional[int] = None,
        user: EnhancedUser = Depends(deps.require_role(SecurityRole.ADMIN)),
    ) -> Dict[str, Any]:
        """Create new API key."""
        key, api_key_obj = security_manager.create_api_key(
            service_name=service_name,
            user_id=user.user_id,
            permissions=set(permissions),
            expires_in_days=expires_in_days,
        )
        
        return {
            "api_key": key,  # Only returned once
            "key_id": api_key_obj.key_id,
            "service_name": api_key_obj.service_name,
            "expires_at": api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else None,
        }
    
    @app.delete("/security/api-keys/{key_id}")
    async def revoke_api_key(
        key_id: str,
        reason: str = "",
        user: EnhancedUser = Depends(deps.require_role(SecurityRole.ADMIN)),
    ) -> Dict[str, str]:
        """Revoke API key."""
        success = security_manager.revoke_api_key(key_id, reason)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        
        return {"message": "API key revoked", "key_id": key_id}


from typing import List  # Add at top with other imports