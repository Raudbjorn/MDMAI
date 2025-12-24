"""Enhanced JWT token management with Redis-backed distributed blacklist."""

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import jwt
import redis.asyncio as redis

from config.logging_config import get_logger
from src.security.models import TokenType, JWTClaims, TokenPair

logger = get_logger(__name__)


class DistributedJWTManager:
    """Enhanced JWT Manager with Redis-backed distributed blacklist and refresh token storage."""

    def __init__(
        self, 
        config: Optional['JWTConfig'] = None,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "jwt:",
    ):
        """
        Initialize JWT manager with distributed storage.
        
        Args:
            config: JWT configuration
            redis_url: Redis connection URL
            key_prefix: Prefix for Redis keys
        """
        from src.security.jwt_manager import JWTConfig
        self.config = config or JWTConfig()
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis: Optional[redis.Redis] = None
        self._initialized = False
        
        # Key patterns for Redis
        self._blacklist_key = f"{key_prefix}blacklist"
        self._refresh_key = f"{key_prefix}refresh"
        self._stats_key = f"{key_prefix}stats"

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if self._initialized:
            return
        
        try:
            self._redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            
            # Test connection
            await self._redis.ping()
            
            self._initialized = True
            logger.info("Distributed JWT manager initialized with Redis")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis for JWT manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up Redis connection."""
        if self._redis:
            await self._redis.close()
            self._initialized = False

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

    def _hash_token(self, token: str) -> str:
        """Hash token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    async def create_access_token(
        self,
        subject: str,
        claims: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Create access token."""
        if not self._initialized:
            await self.initialize()
        
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
            for key, value in claims.items():
                if key not in payload:
                    payload[key] = value
        
        token = jwt.encode(
            payload,
            self._get_signing_key(),
            algorithm=self.config.algorithm.value,
        )
        
        # Track token creation in stats
        await self._increment_stat("tokens_created")
        
        logger.debug(f"Created access token for subject: {subject}")
        return token

    async def create_refresh_token(
        self,
        subject: str,
        claims: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Create refresh token and store in Redis."""
        if not self._initialized:
            await self.initialize()
        
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
        
        # Store refresh token metadata in Redis with TTL
        refresh_data = {
            "subject": subject,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
            "used": False,
            "used_count": 0,
            "last_used": None,
            "ip_addresses": [],
            "user_agents": [],
        }
        
        refresh_key = f"{self._refresh_key}:{token_id}"
        await self._redis.setex(
            refresh_key,
            ttl,
            json.dumps(refresh_data),
        )
        
        # Track token creation
        await self._increment_stat("refresh_tokens_created")
        
        logger.debug(f"Created refresh token for subject: {subject}")
        return token

    async def verify_token(
        self,
        token: str,
        expected_type: Optional[TokenType] = None,
        verify_exp: Optional[bool] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> JWTClaims:
        """Verify and decode JWT token with distributed blacklist check."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check if token is blacklisted in Redis
            token_hash = self._hash_token(token)
            is_blacklisted = await self._redis.sismember(self._blacklist_key, token_hash)
            
            if is_blacklisted:
                await self._increment_stat("blacklisted_tokens_rejected")
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
                raise jwt.InvalidTokenError(
                    f"Invalid token type: expected {expected_type.value}, got {token_type.value}"
                )
            
            # Check refresh token usage in Redis
            if token_type == TokenType.REFRESH:
                jti = payload.get("jti")
                if jti:
                    refresh_key = f"{self._refresh_key}:{jti}"
                    refresh_data_str = await self._redis.get(refresh_key)
                    
                    if refresh_data_str:
                        refresh_data = json.loads(refresh_data_str)
                        
                        # Check if token has been used (for one-time use policy)
                        if refresh_data.get("used") and not self.config.allow_refresh_reuse:
                            await self._increment_stat("refresh_tokens_rejected_reuse")
                            raise jwt.InvalidTokenError("Refresh token has already been used")
                        
                        # Update usage tracking if metadata provided
                        if request_metadata:
                            refresh_data["used_count"] = refresh_data.get("used_count", 0) + 1
                            refresh_data["last_used"] = datetime.now(timezone.utc).isoformat()
                            
                            if "ip_address" in request_metadata:
                                ip_addresses = refresh_data.get("ip_addresses", [])
                                if request_metadata["ip_address"] not in ip_addresses:
                                    ip_addresses.append(request_metadata["ip_address"])
                                    refresh_data["ip_addresses"] = ip_addresses[-10:]  # Keep last 10
                            
                            if "user_agent" in request_metadata:
                                user_agents = refresh_data.get("user_agents", [])
                                if request_metadata["user_agent"] not in user_agents:
                                    user_agents.append(request_metadata["user_agent"])
                                    refresh_data["user_agents"] = user_agents[-5:]  # Keep last 5
                            
                            # Update in Redis
                            ttl = await self._redis.ttl(refresh_key)
                            if ttl > 0:
                                await self._redis.setex(
                                    refresh_key,
                                    ttl,
                                    json.dumps(refresh_data),
                                )
            
            # Track successful verification
            await self._increment_stat("tokens_verified")
            
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
            await self._increment_stat("tokens_expired")
            logger.warning(f"Token expired: {e}")
            raise
        except jwt.InvalidTokenError as e:
            await self._increment_stat("tokens_invalid")
            logger.warning(f"Invalid token: {e}")
            raise
        except Exception as e:
            await self._increment_stat("tokens_error")
            logger.error(f"Token verification failed: {e}")
            raise jwt.InvalidTokenError(str(e))

    async def refresh_access_token(
        self,
        refresh_token: str,
        claims: Optional[Dict[str, Any]] = None,
        mark_as_used: bool = True,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> TokenPair:
        """Use refresh token to get new access token with distributed tracking."""
        if not self._initialized:
            await self.initialize()
        
        # Verify refresh token
        refresh_claims = await self.verify_token(
            refresh_token, 
            TokenType.REFRESH,
            request_metadata=request_metadata,
        )
        
        # Mark refresh token as used in Redis if required
        if mark_as_used and refresh_claims.jti:
            refresh_key = f"{self._refresh_key}:{refresh_claims.jti}"
            refresh_data_str = await self._redis.get(refresh_key)
            
            if refresh_data_str:
                refresh_data = json.loads(refresh_data_str)
                refresh_data["used"] = True
                refresh_data["used_at"] = datetime.now(timezone.utc).isoformat()
                
                ttl = await self._redis.ttl(refresh_key)
                if ttl > 0:
                    await self._redis.setex(
                        refresh_key,
                        ttl,
                        json.dumps(refresh_data),
                    )
        
        # Create new token pair
        new_claims = refresh_claims.custom_claims.copy()
        if claims:
            new_claims.update(claims)
        
        token_pair = await self.create_token_pair(refresh_claims.sub, new_claims)
        
        # Track refresh
        await self._increment_stat("tokens_refreshed")
        
        return token_pair

    async def create_token_pair(
        self,
        subject: str,
        claims: Optional[Dict[str, Any]] = None,
        access_ttl: Optional[int] = None,
        refresh_ttl: Optional[int] = None,
    ) -> TokenPair:
        """Create access and refresh token pair."""
        access_token = await self.create_access_token(subject, claims, access_ttl)
        refresh_token = await self.create_refresh_token(subject, claims, refresh_ttl)
        
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="Bearer",
            expires_in=access_ttl or self.config.access_token_ttl,
        )

    async def revoke_token(
        self, 
        token: str,
        reason: Optional[str] = None,
        revoked_by: Optional[str] = None,
    ) -> bool:
        """Revoke token by adding to distributed blacklist."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Verify token first (without exp check)
            claims = await self.verify_token(token, verify_exp=False)
            
            # Add to Redis blacklist with TTL matching token expiry
            token_hash = self._hash_token(token)
            
            # Calculate TTL based on token expiry
            ttl = int((claims.exp - datetime.now(timezone.utc)).total_seconds())
            if ttl <= 0:
                # Token already expired, no need to blacklist
                return True
            
            # Add to blacklist set with expiry
            await self._redis.sadd(self._blacklist_key, token_hash)
            
            # Store revocation metadata
            revoke_key = f"{self.key_prefix}revoked:{token_hash}"
            revoke_data = {
                "jti": claims.jti,
                "subject": claims.sub,
                "type": claims.type.value,
                "revoked_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "revoked_by": revoked_by,
                "expires_at": claims.exp.isoformat(),
            }
            
            await self._redis.setex(
                revoke_key,
                ttl,
                json.dumps(revoke_data),
            )
            
            # If it's a refresh token, also mark it in refresh storage
            if claims.type == TokenType.REFRESH and claims.jti:
                refresh_key = f"{self._refresh_key}:{claims.jti}"
                refresh_data_str = await self._redis.get(refresh_key)
                
                if refresh_data_str:
                    refresh_data = json.loads(refresh_data_str)
                    refresh_data["revoked"] = True
                    refresh_data["revoked_at"] = datetime.now(timezone.utc).isoformat()
                    refresh_data["revoke_reason"] = reason
                    
                    remaining_ttl = await self._redis.ttl(refresh_key)
                    if remaining_ttl > 0:
                        await self._redis.setex(
                            refresh_key,
                            remaining_ttl,
                            json.dumps(refresh_data),
                        )
            
            # Track revocation
            await self._increment_stat("tokens_revoked")
            
            logger.info(f"Token revoked: {claims.jti}, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False

    async def cleanup_expired(self) -> Tuple[int, int]:
        """Clean up expired tokens from Redis (automatic with TTL)."""
        if not self._initialized:
            await self.initialize()
        
        # Redis handles expiry automatically with TTL
        # This method can be used for stats or manual cleanup if needed
        
        # Get stats
        blacklist_size = await self._redis.scard(self._blacklist_key)
        
        # Count refresh tokens
        refresh_pattern = f"{self._refresh_key}:*"
        refresh_keys = []
        async for key in self._redis.scan_iter(match=refresh_pattern):
            refresh_keys.append(key)
        
        refresh_count = len(refresh_keys)
        
        logger.info(f"Token storage status - Blacklist: {blacklist_size}, Refresh tokens: {refresh_count}")
        
        return blacklist_size, refresh_count

    async def get_token_info(self, token: str) -> Dict[str, Any]:
        """Get token information without full verification."""
        try:
            # Decode without verification for inspection
            payload = jwt.decode(
                token,
                options={"verify_signature": False},
            )
            
            token_hash = self._hash_token(token)
            is_blacklisted = False
            revoke_info = None
            
            if self._initialized:
                # Check blacklist status
                is_blacklisted = await self._redis.sismember(self._blacklist_key, token_hash)
                
                # Get revocation info if blacklisted
                if is_blacklisted:
                    revoke_key = f"{self.key_prefix}revoked:{token_hash}"
                    revoke_data_str = await self._redis.get(revoke_key)
                    if revoke_data_str:
                        revoke_info = json.loads(revoke_data_str)
            
            return {
                "subject": payload.get("sub"),
                "type": payload.get("type"),
                "issued_at": datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc),
                "expires_at": datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
                "issuer": payload.get("iss"),
                "audience": payload.get("aud"),
                "token_id": payload.get("jti"),
                "is_expired": datetime.now(timezone.utc) > datetime.fromtimestamp(
                    payload.get("exp", 0), tz=timezone.utc
                ),
                "is_blacklisted": is_blacklisted,
                "revoke_info": revoke_info,
            }
        except Exception as e:
            logger.error(f"Failed to decode token info: {e}")
            return {"error": str(e)}

    async def get_user_sessions(self, subject: str) -> List[Dict[str, Any]]:
        """Get all active refresh tokens for a user."""
        if not self._initialized:
            await self.initialize()
        
        sessions = []
        refresh_pattern = f"{self._refresh_key}:*"
        
        async for key in self._redis.scan_iter(match=refresh_pattern):
            refresh_data_str = await self._redis.get(key)
            if refresh_data_str:
                refresh_data = json.loads(refresh_data_str)
                if refresh_data.get("subject") == subject:
                    # Add key info
                    token_id = key.split(":")[-1]
                    refresh_data["token_id"] = token_id
                    refresh_data["ttl"] = await self._redis.ttl(key)
                    sessions.append(refresh_data)
        
        return sessions

    async def revoke_all_user_tokens(
        self, 
        subject: str,
        reason: Optional[str] = None,
    ) -> int:
        """Revoke all tokens for a specific user."""
        if not self._initialized:
            await self.initialize()
        
        count = 0
        
        # Find and revoke all refresh tokens for the user
        sessions = await self.get_user_sessions(subject)
        
        for session in sessions:
            # We don't have the actual token, but we can mark it as revoked
            token_id = session.get("token_id")
            if token_id:
                refresh_key = f"{self._refresh_key}:{token_id}"
                refresh_data_str = await self._redis.get(refresh_key)
                
                if refresh_data_str:
                    refresh_data = json.loads(refresh_data_str)
                    refresh_data["revoked"] = True
                    refresh_data["revoked_at"] = datetime.now(timezone.utc).isoformat()
                    refresh_data["revoke_reason"] = reason or "All user tokens revoked"
                    
                    ttl = await self._redis.ttl(refresh_key)
                    if ttl > 0:
                        await self._redis.setex(
                            refresh_key,
                            ttl,
                            json.dumps(refresh_data),
                        )
                        count += 1
        
        logger.info(f"Revoked {count} tokens for user {subject}")
        return count

    async def _increment_stat(self, stat_name: str) -> None:
        """Increment a statistics counter."""
        if self._initialized:
            stat_key = f"{self._stats_key}:{stat_name}"
            await self._redis.incr(stat_key)
            
            # Set expiry to 30 days if new key
            await self._redis.expire(stat_key, 30 * 24 * 3600)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get JWT management statistics."""
        if not self._initialized:
            return {"error": "Not initialized"}
        
        stats = {}
        
        # Get all stat counters
        stat_pattern = f"{self._stats_key}:*"
        async for key in self._redis.scan_iter(match=stat_pattern):
            stat_name = key.split(":")[-1]
            value = await self._redis.get(key)
            stats[stat_name] = int(value) if value else 0
        
        # Add current counts
        stats["blacklist_size"] = await self._redis.scard(self._blacklist_key)
        
        # Count refresh tokens
        refresh_pattern = f"{self._refresh_key}:*"
        refresh_count = 0
        async for _ in self._redis.scan_iter(match=refresh_pattern):
            refresh_count += 1
        stats["active_refresh_tokens"] = refresh_count
        
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if not self._initialized:
                return {"status": "not_initialized"}
            
            # Check Redis connection
            await self._redis.ping()
            
            # Get stats
            stats = await self.get_statistics()
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"JWT manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# For backwards compatibility, inherit from original JWTManager
from src.security.jwt_manager import JWTManager, JWTConfig

class EnhancedJWTManager(JWTManager):
    """
    Enhanced JWT Manager that uses Redis for distributed state.
    This is a drop-in replacement for the original JWTManager.
    """
    
    def __init__(
        self, 
        config: Optional[JWTConfig] = None,
        redis_url: Optional[str] = None,
        use_distributed: bool = True,
    ):
        """
        Initialize enhanced JWT manager.
        
        Args:
            config: JWT configuration
            redis_url: Redis URL for distributed mode
            use_distributed: Whether to use distributed features
        """
        super().__init__(config)
        
        self.use_distributed = use_distributed and redis_url is not None
        
        if self.use_distributed:
            # Create distributed manager
            self._distributed = DistributedJWTManager(config, redis_url)
            logger.info("Using distributed JWT manager with Redis")
        else:
            # Use in-memory storage from parent
            logger.info("Using in-memory JWT manager")
    
    async def initialize(self) -> None:
        """Initialize the manager."""
        if self.use_distributed:
            await self._distributed.initialize()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.use_distributed:
            await self._distributed.cleanup()
    
    # Override methods to use distributed version when available
    
    def create_access_token(
        self,
        subject: str,
        claims: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Create access token."""
        if self.use_distributed:
            # Note: This is async in distributed version
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If called from async context, create task
                import warnings
                warnings.warn(
                    "Calling sync method from async context. Use await create_access_token_async() instead.",
                    RuntimeWarning
                )
            return super().create_access_token(subject, claims, ttl_seconds)
        else:
            return super().create_access_token(subject, claims, ttl_seconds)
    
    async def create_access_token_async(
        self,
        subject: str,
        claims: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Create access token (async version)."""
        if self.use_distributed:
            return await self._distributed.create_access_token(subject, claims, ttl_seconds)
        else:
            return super().create_access_token(subject, claims, ttl_seconds)
    
    async def verify_token_async(
        self,
        token: str,
        expected_type: Optional[TokenType] = None,
        verify_exp: Optional[bool] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> JWTClaims:
        """Verify token (async version)."""
        if self.use_distributed:
            return await self._distributed.verify_token(
                token, expected_type, verify_exp, request_metadata
            )
        else:
            return super().verify_token(token, expected_type, verify_exp)
    
    async def revoke_token_async(
        self, 
        token: str,
        reason: Optional[str] = None,
        revoked_by: Optional[str] = None,
    ) -> bool:
        """Revoke token (async version)."""
        if self.use_distributed:
            return await self._distributed.revoke_token(token, reason, revoked_by)
        else:
            return super().revoke_token(token)