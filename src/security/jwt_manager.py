"""JWT token management for stateless authentication."""

import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import BaseModel, Field, validator

from config.logging_config import get_logger
from src.security.models import TokenType, JWTClaims, TokenPair

logger = get_logger(__name__)


class JWTAlgorithm(Enum):
    """JWT signing algorithms."""

    HS256 = "HS256"  # HMAC with SHA-256
    HS384 = "HS384"  # HMAC with SHA-384
    HS512 = "HS512"  # HMAC with SHA-512
    RS256 = "RS256"  # RSA with SHA-256
    RS384 = "RS384"  # RSA with SHA-384
    RS512 = "RS512"  # RSA with SHA-512
    ES256 = "ES256"  # ECDSA with SHA-256
    ES384 = "ES384"  # ECDSA with SHA-384
    ES512 = "ES512"  # ECDSA with SHA-512


class JWTConfig(BaseModel):
    """JWT configuration."""

    algorithm: JWTAlgorithm = JWTAlgorithm.RS256
    secret_key: Optional[str] = None  # For HMAC algorithms
    private_key: Optional[str] = None  # For RSA/ECDSA algorithms
    public_key: Optional[str] = None  # For RSA/ECDSA verification
    access_token_ttl: int = Field(default=900, description="Access token TTL in seconds (15 minutes)")
    refresh_token_ttl: int = Field(default=2592000, description="Refresh token TTL in seconds (30 days)")
    issuer: str = "ttrpg-assistant"
    audience: List[str] = Field(default_factory=lambda: ["ttrpg-web", "ttrpg-api"])
    verify_exp: bool = True
    verify_aud: bool = True
    verify_iss: bool = True
    leeway: int = Field(default=10, description="Leeway in seconds for time-based claims")

    @validator("secret_key")
    def validate_secret_key(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate secret key for HMAC algorithms."""
        algorithm = values.get("algorithm")
        if algorithm and algorithm.value.startswith("HS"):
            if not v:
                # Generate secure secret if not provided
                return secrets.token_urlsafe(64)
            elif len(v) < 32:
                raise ValueError("Secret key must be at least 32 characters for HMAC algorithms")
        return v

    @validator("private_key")
    def validate_private_key(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate private key for asymmetric algorithms."""
        algorithm = values.get("algorithm")
        if algorithm and algorithm.value.startswith(("RS", "ES")):
            if not v:
                # Generate RSA key pair if not provided
                key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend(),
                )
                return key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                ).decode()
        return v

    @validator("public_key")
    def validate_public_key(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Generate public key from private key if not provided."""
        algorithm = values.get("algorithm")
        private_key = values.get("private_key")
        
        if algorithm and algorithm.value.startswith(("RS", "ES")):
            if not v and private_key:
                # Extract public key from private key
                private_key_obj = serialization.load_pem_private_key(
                    private_key.encode(),
                    password=None,
                    backend=default_backend(),
                )
                public_key_obj = private_key_obj.public_key()
                return public_key_obj.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                ).decode()
        return v


class JWTManager:
    """Manages JWT token creation, validation, and refresh."""

    def __init__(self, config: Optional[JWTConfig] = None):
        """Initialize JWT manager."""
        self.config = config or JWTConfig()
        self._blacklist: set[str] = set()  # Token blacklist for revocation
        self._refresh_tokens: Dict[str, Dict[str, Any]] = {}  # Track refresh tokens

    def _get_signing_key(self) -> str:
        """Get signing key based on algorithm."""
        if self.config.algorithm.value.startswith("HS"):
            return self.config.secret_key or ""
        else:
            return self.config.private_key or ""

    def _get_verification_key(self) -> str:
        """Get verification key based on algorithm."""
        if self.config.algorithm.value.startswith("HS"):
            return self.config.secret_key or ""
        else:
            return self.config.public_key or ""

    def create_access_token(
        self,
        subject: str,
        claims: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Create access token."""
        now = datetime.now(timezone.utc)
        ttl = ttl_seconds or self.config.access_token_ttl
        
        payload = {
            "sub": subject,
            "type": TokenType.ACCESS.value,
            "iat": now,
            "exp": now + timedelta(seconds=ttl),
            "nbf": now,
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "jti": secrets.token_urlsafe(32),  # Unique token ID
        }
        
        # Add custom claims
        if claims:
            # Ensure we don't override standard claims
            for key, value in claims.items():
                if key not in payload:
                    payload[key] = value
        
        token = jwt.encode(
            payload,
            self._get_signing_key(),
            algorithm=self.config.algorithm.value,
        )
        
        logger.debug(f"Created access token for subject: {subject}")
        return token

    def create_refresh_token(
        self,
        subject: str,
        claims: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Create refresh token."""
        now = datetime.now(timezone.utc)
        ttl = ttl_seconds or self.config.refresh_token_ttl
        token_id = secrets.token_urlsafe(32)
        
        payload = {
            "sub": subject,
            "type": TokenType.REFRESH.value,
            "iat": now,
            "exp": now + timedelta(seconds=ttl),
            "nbf": now,
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "jti": token_id,
        }
        
        # Add custom claims
        if claims:
            for key, value in claims.items():
                if key not in payload:
                    payload[key] = value
        
        token = jwt.encode(
            payload,
            self._get_signing_key(),
            algorithm=self.config.algorithm.value,
        )
        
        # Track refresh token for rotation
        self._refresh_tokens[token_id] = {
            "subject": subject,
            "created_at": now,
            "expires_at": now + timedelta(seconds=ttl),
            "used": False,
        }
        
        logger.debug(f"Created refresh token for subject: {subject}")
        return token

    def create_token_pair(
        self,
        subject: str,
        claims: Optional[Dict[str, Any]] = None,
        access_ttl: Optional[int] = None,
        refresh_ttl: Optional[int] = None,
    ) -> TokenPair:
        """Create access and refresh token pair."""
        access_token = self.create_access_token(subject, claims, access_ttl)
        refresh_token = self.create_refresh_token(subject, claims, refresh_ttl)
        
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="Bearer",
            expires_in=access_ttl or self.config.access_token_ttl,
        )

    def verify_token(
        self,
        token: str,
        expected_type: Optional[TokenType] = None,
        verify_exp: Optional[bool] = None,
    ) -> JWTClaims:
        """Verify and decode JWT token."""
        try:
            # Check if token is blacklisted
            token_hash = self._hash_token(token)
            if token_hash in self._blacklist:
                raise jwt.InvalidTokenError("Token has been revoked")
            
            # Decode token
            payload = jwt.decode(
                token,
                self._get_verification_key(),
                algorithms=[self.config.algorithm.value],
                issuer=self.config.issuer if self.config.verify_iss else None,
                audience=self.config.audience if self.config.verify_aud else None,
                options={
                    "verify_exp": verify_exp if verify_exp is not None else self.config.verify_exp,
                    "verify_aud": self.config.verify_aud,
                    "verify_iss": self.config.verify_iss,
                },
                leeway=self.config.leeway,
            )
            
            # Verify token type
            token_type = TokenType(payload.get("type", TokenType.ACCESS.value))
            if expected_type and token_type != expected_type:
                raise jwt.InvalidTokenError(f"Invalid token type: expected {expected_type.value}, got {token_type.value}")
            
            # Check refresh token usage
            if token_type == TokenType.REFRESH:
                jti = payload.get("jti")
                if jti and jti in self._refresh_tokens:
                    if self._refresh_tokens[jti]["used"]:
                        raise jwt.InvalidTokenError("Refresh token has already been used")
            
            return JWTClaims(
                sub=payload["sub"],
                type=token_type,
                iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
                exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
                nbf=datetime.fromtimestamp(payload.get("nbf", payload["iat"]), tz=timezone.utc),
                iss=payload.get("iss"),
                aud=payload.get("aud"),
                jti=payload.get("jti"),
                custom_claims={k: v for k, v in payload.items() if k not in JWTClaims.__fields__},
            )
            
        except jwt.ExpiredSignatureError as e:
            logger.warning(f"Token expired: {e}")
            raise
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise jwt.InvalidTokenError(str(e))

    def refresh_access_token(
        self,
        refresh_token: str,
        claims: Optional[Dict[str, Any]] = None,
    ) -> TokenPair:
        """Use refresh token to get new access token."""
        # Verify refresh token
        refresh_claims = self.verify_token(refresh_token, TokenType.REFRESH)
        
        # Mark refresh token as used (for one-time use)
        if refresh_claims.jti and refresh_claims.jti in self._refresh_tokens:
            self._refresh_tokens[refresh_claims.jti]["used"] = True
        
        # Create new token pair
        new_claims = refresh_claims.custom_claims.copy()
        if claims:
            new_claims.update(claims)
        
        return self.create_token_pair(refresh_claims.sub, new_claims)

    def revoke_token(self, token: str) -> bool:
        """Revoke token by adding to blacklist."""
        try:
            # Verify token first
            claims = self.verify_token(token, verify_exp=False)
            
            # Add to blacklist
            token_hash = self._hash_token(token)
            self._blacklist.add(token_hash)
            
            # If it's a refresh token, mark as used
            if claims.type == TokenType.REFRESH and claims.jti:
                if claims.jti in self._refresh_tokens:
                    self._refresh_tokens[claims.jti]["used"] = True
            
            logger.info(f"Token revoked: {claims.jti}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False

    def _hash_token(self, token: str) -> str:
        """Hash token for blacklist storage."""
        import hashlib
        return hashlib.sha256(token.encode()).hexdigest()

    def cleanup_expired(self) -> Tuple[int, int]:
        """Clean up expired tokens from tracking."""
        now = datetime.now(timezone.utc)
        
        # Clean up expired refresh tokens
        expired_refresh = []
        for jti, info in self._refresh_tokens.items():
            if info["expires_at"] < now:
                expired_refresh.append(jti)
        
        for jti in expired_refresh:
            del self._refresh_tokens[jti]
        
        # Note: Blacklist cleanup would need TTL tracking
        # For production, use Redis with TTL instead
        
        return len(expired_refresh), 0

    def get_token_info(self, token: str) -> Dict[str, Any]:
        """Get token information without full verification."""
        try:
            # Decode without verification for inspection
            payload = jwt.decode(
                token,
                options={"verify_signature": False},
            )
            
            return {
                "subject": payload.get("sub"),
                "type": payload.get("type"),
                "issued_at": datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc),
                "expires_at": datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
                "issuer": payload.get("iss"),
                "audience": payload.get("aud"),
                "token_id": payload.get("jti"),
                "is_expired": datetime.now(timezone.utc) > datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
            }
        except Exception as e:
            logger.error(f"Failed to decode token info: {e}")
            return {}

    def validate_token_pair(self, access_token: str, refresh_token: str) -> bool:
        """Validate that access and refresh tokens belong together."""
        try:
            access_claims = self.verify_token(access_token, TokenType.ACCESS, verify_exp=False)
            refresh_claims = self.verify_token(refresh_token, TokenType.REFRESH, verify_exp=False)
            
            # Check subjects match
            return access_claims.sub == refresh_claims.sub
            
        except Exception:
            return False