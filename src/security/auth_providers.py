"""OAuth2 authentication providers for web authentication."""

import asyncio
import hashlib
import json
import secrets
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, parse_qs, urlparse

import aiohttp
from pydantic import BaseModel, Field, HttpUrl, validator

from config.logging_config import get_logger
from src.security.models import AuthProvider, OAuthTokenResponse, UserInfo

logger = get_logger(__name__)


class OAuthProvider(Enum):
    """Supported OAuth providers."""

    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    DISCORD = "discord"


class OAuthConfig(BaseModel):
    """OAuth provider configuration."""

    provider: OAuthProvider
    client_id: str
    client_secret: str
    redirect_uri: HttpUrl
    scopes: List[str] = Field(default_factory=list)
    authorization_endpoint: HttpUrl
    token_endpoint: HttpUrl
    userinfo_endpoint: HttpUrl
    revocation_endpoint: Optional[HttpUrl] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    @validator("scopes", pre=True)
    def validate_scopes(cls, v: List[str], values: Dict[str, Any]) -> List[str]:
        """Validate and set default scopes based on provider."""
        provider = values.get("provider")
        if not v and provider:
            # Set default scopes based on provider
            defaults = {
                OAuthProvider.GOOGLE: ["openid", "email", "profile"],
                OAuthProvider.GITHUB: ["user:email", "read:user"],
                OAuthProvider.MICROSOFT: ["openid", "email", "profile", "offline_access"],
                OAuthProvider.DISCORD: ["identify", "email"],
            }
            return defaults.get(provider, ["openid", "email", "profile"])
        return v


class OAuthState(BaseModel):
    """OAuth state for CSRF protection."""

    state: str
    provider: OAuthProvider
    created_at: datetime
    redirect_uri: str
    code_verifier: Optional[str] = None  # For PKCE
    nonce: Optional[str] = None  # For OpenID Connect
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_expired(self, ttl_minutes: int = 10) -> bool:
        """Check if state is expired."""
        return datetime.utcnow() > self.created_at + timedelta(minutes=ttl_minutes)


class AbstractOAuthProvider(ABC):
    """Abstract base class for OAuth providers."""

    def __init__(self, config: OAuthConfig):
        """Initialize OAuth provider."""
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AbstractOAuthProvider":
        """Enter async context."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        if self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if not self._session:
            self._session = aiohttp.ClientSession()
        return self._session

    def generate_state(self, metadata: Optional[Dict[str, Any]] = None) -> OAuthState:
        """Generate secure state for CSRF protection."""
        state = OAuthState(
            state=secrets.token_urlsafe(32),
            provider=self.config.provider,
            created_at=datetime.utcnow(),
            redirect_uri=str(self.config.redirect_uri),
            metadata=metadata or {},
        )
        
        # Add PKCE code verifier if supported
        if self.supports_pkce():
            state.code_verifier = secrets.token_urlsafe(64)
        
        # Add nonce for OpenID Connect
        if self.supports_openid():
            state.nonce = secrets.token_urlsafe(32)
        
        return state

    def get_authorization_url(self, state: OAuthState) -> str:
        """Get authorization URL for OAuth flow."""
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": str(self.config.redirect_uri),
            "response_type": "code",
            "state": state.state,
            "scope": " ".join(self.config.scopes),
        }
        
        # Add PKCE challenge if supported
        if state.code_verifier and self.supports_pkce():
            code_challenge = hashlib.sha256(state.code_verifier.encode()).digest()
            params["code_challenge"] = secrets.token_urlsafe(43)  # Base64 URL-safe
            params["code_challenge_method"] = "S256"
        
        # Add nonce for OpenID Connect
        if state.nonce and self.supports_openid():
            params["nonce"] = state.nonce
        
        # Add provider-specific parameters
        params.update(self.config.additional_params)
        
        return f"{self.config.authorization_endpoint}?{urlencode(params)}"

    async def exchange_code(
        self,
        code: str,
        state: OAuthState,
    ) -> OAuthTokenResponse:
        """Exchange authorization code for access token."""
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": str(self.config.redirect_uri),
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }
        
        # Add PKCE verifier if used
        if state.code_verifier and self.supports_pkce():
            data["code_verifier"] = state.code_verifier
        
        async with self.session.post(
            str(self.config.token_endpoint),
            data=data,
            headers={"Accept": "application/json"},
        ) as response:
            response.raise_for_status()
            token_data = await response.json()
            
            return OAuthTokenResponse(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                refresh_token=token_data.get("refresh_token"),
                scope=token_data.get("scope", "").split(),
                id_token=token_data.get("id_token"),
            )

    async def refresh_token(self, refresh_token: str) -> OAuthTokenResponse:
        """Refresh access token using refresh token."""
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }
        
        async with self.session.post(
            str(self.config.token_endpoint),
            data=data,
            headers={"Accept": "application/json"},
        ) as response:
            response.raise_for_status()
            token_data = await response.json()
            
            return OAuthTokenResponse(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                refresh_token=token_data.get("refresh_token", refresh_token),
                scope=token_data.get("scope", "").split(),
                id_token=token_data.get("id_token"),
            )

    async def revoke_token(self, token: str, token_type: str = "access_token") -> bool:
        """Revoke access or refresh token."""
        if not self.config.revocation_endpoint:
            logger.warning(f"Revocation not supported for {self.config.provider}")
            return False
        
        data = {
            "token": token,
            "token_type_hint": token_type,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }
        
        try:
            async with self.session.post(
                str(self.config.revocation_endpoint),
                data=data,
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False

    @abstractmethod
    async def get_user_info(self, access_token: str) -> UserInfo:
        """Get user information using access token."""
        pass

    @abstractmethod
    def supports_pkce(self) -> bool:
        """Check if provider supports PKCE."""
        pass

    @abstractmethod
    def supports_openid(self) -> bool:
        """Check if provider supports OpenID Connect."""
        pass


class GoogleOAuthProvider(AbstractOAuthProvider):
    """Google OAuth2 provider implementation."""

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """Initialize Google OAuth provider."""
        config = OAuthConfig(
            provider=OAuthProvider.GOOGLE,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=HttpUrl(redirect_uri),
            authorization_endpoint=HttpUrl("https://accounts.google.com/o/oauth2/v2/auth"),
            token_endpoint=HttpUrl("https://oauth2.googleapis.com/token"),
            userinfo_endpoint=HttpUrl("https://www.googleapis.com/oauth2/v2/userinfo"),
            revocation_endpoint=HttpUrl("https://oauth2.googleapis.com/revoke"),
            additional_params={"access_type": "offline", "prompt": "consent"},
        )
        super().__init__(config)

    async def get_user_info(self, access_token: str) -> UserInfo:
        """Get user info from Google."""
        async with self.session.get(
            str(self.config.userinfo_endpoint),
            headers={"Authorization": f"Bearer {access_token}"},
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            return UserInfo(
                provider=AuthProvider.GOOGLE,
                provider_user_id=data["id"],
                email=data.get("email"),
                email_verified=data.get("verified_email", False),
                name=data.get("name"),
                picture=data.get("picture"),
                locale=data.get("locale"),
                metadata=data,
            )

    def supports_pkce(self) -> bool:
        """Google supports PKCE."""
        return True

    def supports_openid(self) -> bool:
        """Google supports OpenID Connect."""
        return True


class GitHubOAuthProvider(AbstractOAuthProvider):
    """GitHub OAuth2 provider implementation."""

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """Initialize GitHub OAuth provider."""
        config = OAuthConfig(
            provider=OAuthProvider.GITHUB,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=HttpUrl(redirect_uri),
            authorization_endpoint=HttpUrl("https://github.com/login/oauth/authorize"),
            token_endpoint=HttpUrl("https://github.com/login/oauth/access_token"),
            userinfo_endpoint=HttpUrl("https://api.github.com/user"),
        )
        super().__init__(config)

    async def get_user_info(self, access_token: str) -> UserInfo:
        """Get user info from GitHub."""
        async with self.session.get(
            str(self.config.userinfo_endpoint),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github.v3+json",
            },
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Get primary email separately
            email = data.get("email")
            email_verified = False
            
            if not email:
                async with self.session.get(
                    "https://api.github.com/user/emails",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/vnd.github.v3+json",
                    },
                ) as email_response:
                    if email_response.status == 200:
                        emails = await email_response.json()
                        for e in emails:
                            if e.get("primary"):
                                email = e.get("email")
                                email_verified = e.get("verified", False)
                                break
            
            return UserInfo(
                provider=AuthProvider.GITHUB,
                provider_user_id=str(data["id"]),
                email=email,
                email_verified=email_verified,
                name=data.get("name") or data.get("login"),
                username=data.get("login"),
                picture=data.get("avatar_url"),
                metadata=data,
            )

    def supports_pkce(self) -> bool:
        """GitHub does not support PKCE."""
        return False

    def supports_openid(self) -> bool:
        """GitHub does not support OpenID Connect."""
        return False


class MicrosoftOAuthProvider(AbstractOAuthProvider):
    """Microsoft OAuth2 provider implementation."""

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, tenant: str = "common"):
        """Initialize Microsoft OAuth provider."""
        config = OAuthConfig(
            provider=OAuthProvider.MICROSOFT,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=HttpUrl(redirect_uri),
            authorization_endpoint=HttpUrl(f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"),
            token_endpoint=HttpUrl(f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"),
            userinfo_endpoint=HttpUrl("https://graph.microsoft.com/v1.0/me"),
            additional_params={"response_mode": "query"},
        )
        super().__init__(config)

    async def get_user_info(self, access_token: str) -> UserInfo:
        """Get user info from Microsoft Graph."""
        async with self.session.get(
            str(self.config.userinfo_endpoint),
            headers={"Authorization": f"Bearer {access_token}"},
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            return UserInfo(
                provider=AuthProvider.MICROSOFT,
                provider_user_id=data["id"],
                email=data.get("mail") or data.get("userPrincipalName"),
                email_verified=True,  # Microsoft accounts are pre-verified
                name=data.get("displayName"),
                username=data.get("userPrincipalName"),
                metadata=data,
            )

    def supports_pkce(self) -> bool:
        """Microsoft supports PKCE."""
        return True

    def supports_openid(self) -> bool:
        """Microsoft supports OpenID Connect."""
        return True


class OAuthProviderFactory:
    """Factory for creating OAuth provider instances."""

    _providers: Dict[OAuthProvider, type[AbstractOAuthProvider]] = {
        OAuthProvider.GOOGLE: GoogleOAuthProvider,
        OAuthProvider.GITHUB: GitHubOAuthProvider,
        OAuthProvider.MICROSOFT: MicrosoftOAuthProvider,
    }

    @classmethod
    def create(
        cls,
        provider: OAuthProvider,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        **kwargs: Any,
    ) -> AbstractOAuthProvider:
        """Create OAuth provider instance."""
        provider_class = cls._providers.get(provider)
        if not provider_class:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        return provider_class(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            **kwargs,
        )

    @classmethod
    def register_provider(
        cls,
        provider: OAuthProvider,
        provider_class: type[AbstractOAuthProvider],
    ) -> None:
        """Register custom OAuth provider."""
        cls._providers[provider] = provider_class