"""Distributed JWT token management using ChromaDB for blacklist storage."""

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Set, Tuple

import chromadb
from chromadb.config import Settings
import jwt
from pydantic import BaseModel

from config.logging_config import get_logger
from src.security.jwt_manager import JWTManager, JWTConfig
from src.security.models import TokenType, JWTClaims, TokenPair

logger = get_logger(__name__)


class RevokedToken(BaseModel):
    """Revoked token record for storage."""
    
    token_hash: str
    token_id: Optional[str] = None
    subject: str
    token_type: TokenType
    revoked_at: datetime
    expires_at: datetime
    reason: Optional[str] = None
    revoked_by: Optional[str] = None


class DistributedJWTManager(JWTManager):
    """JWT Manager with distributed blacklist using ChromaDB.
    
    This implementation uses ChromaDB as a distributed store for the JWT blacklist,
    allowing multiple instances to share the same blacklist across the system.
    """
    
    def __init__(
        self,
        config: Optional[JWTConfig] = None,
        chroma_client: Optional[chromadb.Client] = None,
        collection_name: str = "jwt_blacklist",
        cleanup_interval: int = 3600,  # Cleanup expired tokens every hour
    ):
        """Initialize distributed JWT manager.
        
        Args:
            config: JWT configuration
            chroma_client: ChromaDB client (creates one if not provided)
            collection_name: Name of the ChromaDB collection for blacklist
            cleanup_interval: Seconds between cleanup runs
        """
        super().__init__(config)
        
        # Initialize ChromaDB client
        self.chroma_client = chroma_client or chromadb.Client(
            Settings(
                is_persistent=True,
                persist_directory="./chroma_db",
                anonymized_telemetry=False,
            )
        )
        
        # Create or get blacklist collection
        try:
            self.blacklist_collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "JWT token blacklist for distributed revocation"},
            )
        except ValueError:
            # Collection already exists
            self.blacklist_collection = self.chroma_client.get_collection(name=collection_name)
        
        # Create or get refresh token collection
        try:
            self.refresh_collection = self.chroma_client.create_collection(
                name=f"{collection_name}_refresh",
                metadata={"description": "Refresh token tracking for distributed systems"},
            )
        except ValueError:
            self.refresh_collection = self.chroma_client.get_collection(
                name=f"{collection_name}_refresh"
            )
        
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        
        # Local cache for performance
        self._blacklist_cache: Set[str] = set()
        self._cache_updated_at = time.time()
        self._cache_ttl = 60  # Refresh cache every 60 seconds
        
        logger.info(f"Initialized distributed JWT manager with ChromaDB collection: {collection_name}")
    
    def _hash_token(self, token: str) -> str:
        """Hash token for blacklist storage."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _refresh_cache(self) -> None:
        """Refresh local blacklist cache from ChromaDB."""
        now = time.time()
        if now - self._cache_updated_at < self._cache_ttl:
            return  # Cache still fresh
        
        try:
            # Get all blacklisted tokens from ChromaDB
            results = self.blacklist_collection.get()
            if results and results.get("ids"):
                self._blacklist_cache = set(results["ids"])
            else:
                self._blacklist_cache = set()
            
            self._cache_updated_at = now
            logger.debug(f"Refreshed blacklist cache with {len(self._blacklist_cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to refresh blacklist cache: {e}")
    
    def _is_token_blacklisted(self, token_hash: str) -> bool:
        """Check if token is blacklisted using cache and ChromaDB."""
        # Check local cache first
        self._refresh_cache()
        if token_hash in self._blacklist_cache:
            return True
        
        # Double-check with ChromaDB in case of cache miss
        try:
            results = self.blacklist_collection.get(ids=[token_hash])
            return bool(results and results.get("ids"))
        except Exception as e:
            logger.error(f"Failed to check blacklist: {e}")
            return False
    
    def verify_token(
        self,
        token: str,
        expected_type: Optional[TokenType] = None,
        verify_exp: Optional[bool] = None,
    ) -> JWTClaims:
        """Verify and decode JWT token with distributed blacklist check."""
        # Check distributed blacklist
        token_hash = self._hash_token(token)
        if self._is_token_blacklisted(token_hash):
            raise jwt.InvalidTokenError("Token has been revoked")
        
        # Perform standard verification
        claims = super().verify_token(token, expected_type, verify_exp)
        
        # Check refresh token usage in distributed store
        if claims.type == TokenType.REFRESH and claims.jti:
            try:
                results = self.refresh_collection.get(ids=[claims.jti])
                if results and results.get("metadatas"):
                    metadata = results["metadatas"][0]
                    if metadata.get("used") == "true":
                        raise jwt.InvalidTokenError("Refresh token has already been used")
            except jwt.InvalidTokenError:
                raise
            except Exception as e:
                logger.error(f"Failed to check refresh token usage: {e}")
        
        # Run cleanup periodically
        self._maybe_cleanup()
        
        return claims
    
    def revoke_token(self, token: str, reason: Optional[str] = None, revoked_by: Optional[str] = None) -> bool:
        """Revoke token by adding to distributed blacklist."""
        try:
            # Verify token first (without checking blacklist to avoid recursion)
            claims = super().verify_token(token, verify_exp=False)
            
            token_hash = self._hash_token(token)
            now = datetime.now(timezone.utc)
            
            # Create revoked token record
            revoked = RevokedToken(
                token_hash=token_hash,
                token_id=claims.jti,
                subject=claims.sub,
                token_type=claims.type,
                revoked_at=now,
                expires_at=claims.exp,
                reason=reason,
                revoked_by=revoked_by,
            )
            
            # Add to ChromaDB blacklist
            self.blacklist_collection.add(
                ids=[token_hash],
                metadatas=[{
                    "token_id": revoked.token_id or "",
                    "subject": revoked.subject,
                    "token_type": revoked.token_type.value,
                    "revoked_at": revoked.revoked_at.isoformat(),
                    "expires_at": revoked.expires_at.isoformat(),
                    "reason": revoked.reason or "",
                    "revoked_by": revoked_by or "",
                }],
                documents=[json.dumps({
                    "token_hash": token_hash,
                    "revoked_at": revoked.revoked_at.isoformat(),
                    "expires_at": revoked.expires_at.isoformat(),
                })],
            )
            
            # Update local cache immediately
            self._blacklist_cache.add(token_hash)
            
            # If it's a refresh token, mark as used in distributed store
            if claims.type == TokenType.REFRESH and claims.jti:
                self.refresh_collection.upsert(
                    ids=[claims.jti],
                    metadatas=[{
                        "subject": claims.sub,
                        "used": "true",
                        "used_at": now.isoformat(),
                    }],
                    documents=[json.dumps({
                        "jti": claims.jti,
                        "used": True,
                    })],
                )
            
            logger.info(f"Token revoked: {claims.jti} (reason: {reason})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    def create_refresh_token(
        self,
        subject: str,
        claims: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Create refresh token with distributed tracking."""
        # Create token using parent method
        token = super().create_refresh_token(subject, claims, ttl_seconds)
        
        # Decode to get JTI
        token_info = self.get_token_info(token)
        jti = token_info.get("token_id")
        
        if jti:
            # Track in distributed store
            now = datetime.now(timezone.utc)
            ttl = ttl_seconds or self.config.refresh_token_ttl
            
            self.refresh_collection.add(
                ids=[jti],
                metadatas=[{
                    "subject": subject,
                    "created_at": now.isoformat(),
                    "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
                    "used": "false",
                }],
                documents=[json.dumps({
                    "jti": jti,
                    "subject": subject,
                    "used": False,
                })],
            )
        
        return token
    
    def refresh_access_token(
        self,
        refresh_token: str,
        claims: Optional[Dict[str, Any]] = None,
    ) -> TokenPair:
        """Use refresh token to get new access token with distributed tracking."""
        # Verify refresh token
        refresh_claims = self.verify_token(refresh_token, TokenType.REFRESH)
        
        # Mark refresh token as used in distributed store
        if refresh_claims.jti:
            self.refresh_collection.upsert(
                ids=[refresh_claims.jti],
                metadatas=[{
                    "subject": refresh_claims.sub,
                    "used": "true",
                    "used_at": datetime.now(timezone.utc).isoformat(),
                }],
                documents=[json.dumps({
                    "jti": refresh_claims.jti,
                    "used": True,
                })],
            )
        
        # Create new token pair
        new_claims = refresh_claims.custom_claims.copy()
        if claims:
            new_claims.update(claims)
        
        return self.create_token_pair(refresh_claims.sub, new_claims)
    
    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        now = time.time()
        if now - self.last_cleanup >= self.cleanup_interval:
            self.cleanup_expired()
            self.last_cleanup = now
    
    def cleanup_expired(self) -> Tuple[int, int]:
        """Clean up expired tokens from distributed store."""
        now = datetime.now(timezone.utc)
        expired_blacklist = 0
        expired_refresh = 0
        
        try:
            # Clean expired from blacklist
            blacklist_results = self.blacklist_collection.get()
            if blacklist_results and blacklist_results.get("ids"):
                for i, metadata in enumerate(blacklist_results["metadatas"]):
                    expires_at_str = metadata.get("expires_at")
                    if expires_at_str:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if expires_at < now:
                            # Token has expired, safe to remove from blacklist
                            token_id = blacklist_results["ids"][i]
                            self.blacklist_collection.delete(ids=[token_id])
                            self._blacklist_cache.discard(token_id)
                            expired_blacklist += 1
            
            # Clean expired refresh tokens
            refresh_results = self.refresh_collection.get()
            if refresh_results and refresh_results.get("ids"):
                for i, metadata in enumerate(refresh_results["metadatas"]):
                    expires_at_str = metadata.get("expires_at")
                    if expires_at_str:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if expires_at < now:
                            # Refresh token has expired
                            jti = refresh_results["ids"][i]
                            self.refresh_collection.delete(ids=[jti])
                            expired_refresh += 1
            
            if expired_blacklist or expired_refresh:
                logger.info(
                    f"Cleaned up expired tokens - blacklist: {expired_blacklist}, "
                    f"refresh: {expired_refresh}"
                )
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired tokens: {e}")
        
        return expired_blacklist, expired_refresh
    
    def get_blacklist_stats(self) -> Dict[str, Any]:
        """Get statistics about the distributed blacklist."""
        try:
            blacklist_results = self.blacklist_collection.get()
            refresh_results = self.refresh_collection.get()
            
            blacklist_count = len(blacklist_results.get("ids", []))
            refresh_count = len(refresh_results.get("ids", []))
            
            # Count used refresh tokens
            used_refresh = 0
            if refresh_results.get("metadatas"):
                for metadata in refresh_results["metadatas"]:
                    if metadata.get("used") == "true":
                        used_refresh += 1
            
            return {
                "blacklisted_tokens": blacklist_count,
                "tracked_refresh_tokens": refresh_count,
                "used_refresh_tokens": used_refresh,
                "cache_size": len(self._blacklist_cache),
                "cache_age_seconds": time.time() - self._cache_updated_at,
                "last_cleanup": datetime.fromtimestamp(self.last_cleanup, tz=timezone.utc).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Failed to get blacklist stats: {e}")
            return {
                "error": str(e),
                "blacklisted_tokens": 0,
                "tracked_refresh_tokens": 0,
            }
    
    def bulk_revoke(
        self,
        subject: Optional[str] = None,
        token_type: Optional[TokenType] = None,
        reason: str = "Bulk revocation",
        revoked_by: Optional[str] = None,
    ) -> int:
        """Bulk revoke tokens by subject or type.
        
        Args:
            subject: Revoke all tokens for this subject
            token_type: Revoke all tokens of this type
            reason: Reason for revocation
            revoked_by: Who initiated the revocation
            
        Returns:
            Number of tokens revoked
        """
        count = 0
        
        try:
            # This would need to be implemented based on your token tracking strategy
            # For now, return 0 as we don't track all issued tokens
            logger.warning(
                f"Bulk revocation requested (subject: {subject}, type: {token_type}) "
                f"but full implementation requires token issuance tracking"
            )
            
        except Exception as e:
            logger.error(f"Failed to bulk revoke tokens: {e}")
        
        return count


# Convenience function to get a distributed JWT manager instance
def get_distributed_jwt_manager(
    config: Optional[JWTConfig] = None,
    chroma_client: Optional[chromadb.Client] = None,
) -> DistributedJWTManager:
    """Get a distributed JWT manager instance.
    
    Args:
        config: JWT configuration
        chroma_client: ChromaDB client (creates one if not provided)
        
    Returns:
        Configured DistributedJWTManager instance
    """
    return DistributedJWTManager(config=config, chroma_client=chroma_client)