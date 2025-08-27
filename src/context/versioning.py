"""Advanced context versioning and history tracking system."""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Context,
    ContextDiff,
    ContextVersion,
    CompressionType,
)
from .serialization import ContextCompressor, ContextSerializer

logger = logging.getLogger(__name__)


class VersioningStrategy:
    """Strategies for version management."""
    
    INCREMENTAL = "incremental"  # Store only differences
    FULL_SNAPSHOT = "full_snapshot"  # Store complete context
    HYBRID = "hybrid"  # Smart mix based on diff size
    COMPRESSED_DELTA = "compressed_delta"  # Compressed incremental


class ContextVersionManager:
    """Advanced context versioning with delta compression and intelligent storage."""
    
    def __init__(
        self,
        persistence_layer,
        serializer: Optional[ContextSerializer] = None,
        compressor: Optional[ContextCompressor] = None,
        versioning_strategy: str = VersioningStrategy.HYBRID,
        max_versions_per_branch: int = 100,
        auto_cleanup_days: int = 365,
        diff_compression_threshold: float = 0.3,  # Use delta if diff is < 30% of original
    ):
        self.persistence = persistence_layer
        self.serializer = serializer or ContextSerializer()
        self.compressor = compressor or ContextCompressor()
        self.versioning_strategy = versioning_strategy
        self.max_versions_per_branch = max_versions_per_branch
        self.auto_cleanup_days = auto_cleanup_days
        self.diff_compression_threshold = diff_compression_threshold
        
        # Performance tracking
        self._version_stats = {
            "versions_created": 0,
            "versions_retrieved": 0,
            "diffs_computed": 0,
            "space_saved_bytes": 0,
            "avg_version_creation_time": 0.0,
            "avg_version_retrieval_time": 0.0,
        }
        
        logger.info(
            "Context version manager initialized",
            strategy=versioning_strategy,
            max_versions=max_versions_per_branch,
            auto_cleanup_days=auto_cleanup_days,
        )
    
    async def create_version(
        self,
        context: Context,
        user_id: Optional[str] = None,
        message: str = "",
        tags: List[str] = None,
        branch: str = "main",
        parent_version: Optional[str] = None,
    ) -> ContextVersion:
        """Create a new version of a context with intelligent storage strategy."""
        start_time = time.time()
        
        try:
            # Get the previous version for diff calculation
            previous_version = None
            if context.current_version > 1:
                previous_version = await self.get_version(
                    context.context_id, context.current_version - 1, branch
                )
            
            # Calculate version metadata
            version = ContextVersion(
                context_id=context.context_id,
                version_number=context.current_version,
                created_by=user_id,
                parent_version=parent_version,
                branch=branch,
                tags=tags or [],
                metadata={"message": message, "auto_created": user_id is None},
            )
            
            # Determine storage strategy
            storage_strategy = await self._determine_storage_strategy(
                context, previous_version
            )
            
            # Store version based on strategy
            if storage_strategy == VersioningStrategy.FULL_SNAPSHOT:
                await self._store_full_version(context, version)
            elif storage_strategy == VersioningStrategy.INCREMENTAL:
                await self._store_incremental_version(context, version, previous_version)
            elif storage_strategy == VersioningStrategy.COMPRESSED_DELTA:
                await self._store_compressed_delta_version(context, version, previous_version)
            else:  # HYBRID
                await self._store_hybrid_version(context, version, previous_version)
            
            # Clean up old versions if needed
            await self._cleanup_old_versions(context.context_id, branch)
            
            # Update statistics
            execution_time = time.time() - start_time
            self._version_stats["versions_created"] += 1
            self._update_avg_stat("avg_version_creation_time", execution_time)
            
            logger.info(
                "Context version created",
                context_id=context.context_id,
                version=version.version_number,
                branch=branch,
                strategy=storage_strategy,
                execution_time=execution_time,
            )
            
            return version
            
        except Exception as e:
            logger.error(f"Failed to create version: {e}")
            raise
    
    async def get_version(
        self,
        context_id: str,
        version_number: int,
        branch: str = "main",
        user_id: Optional[str] = None,
    ) -> Optional[Context]:
        """Retrieve a specific version of a context."""
        start_time = time.time()
        
        try:
            async with self.persistence._get_async_connection() as conn:
                # Get version metadata
                version_row = await conn.fetchrow("""
                    SELECT cv.*, c.context_type 
                    FROM context_versions cv
                    JOIN contexts c ON c.context_id = cv.context_id
                    WHERE cv.context_id = $1 AND cv.version_number = $2 AND cv.branch = $3
                """, context_id, version_number, branch)
                
                if not version_row:
                    return None
                
                # Check permissions if user specified
                if user_id:
                    context_row = await conn.fetchrow("""
                        SELECT owner_id, collaborators FROM contexts 
                        WHERE context_id = $1
                    """, context_id)
                    
                    if (context_row["owner_id"] != user_id and 
                        user_id not in (context_row["collaborators"] or [])):
                        raise PermissionError(f"User {user_id} does not have access to context {context_id}")
                
                # Reconstruct context from version data
                context = await self._reconstruct_context_from_version(version_row)
                
                # Update statistics
                execution_time = time.time() - start_time
                self._version_stats["versions_retrieved"] += 1
                self._update_avg_stat("avg_version_retrieval_time", execution_time)
                
                logger.debug(
                    "Context version retrieved",
                    context_id=context_id,
                    version=version_number,
                    branch=branch,
                    execution_time=execution_time,
                )
                
                return context
                
        except Exception as e:
            logger.error(f"Failed to get version: {e}")
            raise
    
    async def get_version_history(
        self,
        context_id: str,
        branch: str = "main",
        limit: int = 50,
        offset: int = 0,
    ) -> List[ContextVersion]:
        """Get version history for a context."""
        try:
            async with self.persistence._get_async_connection() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM context_versions 
                    WHERE context_id = $1 AND branch = $2
                    ORDER BY version_number DESC
                    LIMIT $3 OFFSET $4
                """, context_id, branch, limit, offset)
                
                versions = []
                for row in rows:
                    version = ContextVersion(
                        version_id=str(row["version_id"]),
                        context_id=str(row["context_id"]),
                        version_number=row["version_number"],
                        created_at=row["created_at"],
                        created_by=row["created_by"],
                        parent_version=str(row["parent_version"]) if row["parent_version"] else None,
                        branch=row["branch"],
                        checksum=row["checksum"],
                        compressed=row["compressed"],
                        compression_type=CompressionType(row["compression_type"]),
                        size_bytes=row["size_bytes"],
                        metadata=row["metadata"] or {},
                        tags=row["tags"] or [],
                    )
                    versions.append(version)
                
                return versions
                
        except Exception as e:
            logger.error(f"Failed to get version history: {e}")
            raise
    
    async def compare_versions(
        self,
        context_id: str,
        version1: int,
        version2: int,
        branch: str = "main",
    ) -> ContextDiff:
        """Compare two versions and return differences."""
        start_time = time.time()
        
        try:
            # Get both versions
            context1 = await self.get_version(context_id, version1, branch)
            context2 = await self.get_version(context_id, version2, branch)
            
            if not context1 or not context2:
                raise ValueError(f"One or both versions not found: {version1}, {version2}")
            
            # Calculate diff
            diff = await self._calculate_diff(context1, context2, version1, version2)
            
            # Update statistics
            execution_time = time.time() - start_time
            self._version_stats["diffs_computed"] += 1
            
            logger.debug(
                "Versions compared",
                context_id=context_id,
                version1=version1,
                version2=version2,
                diff_size=diff.diff_size_bytes,
                execution_time=execution_time,
            )
            
            return diff
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            raise
    
    async def revert_to_version(
        self,
        context_id: str,
        target_version: int,
        user_id: Optional[str] = None,
        branch: str = "main",
        create_revert_version: bool = True,
    ) -> Context:
        """Revert context to a specific version."""
        try:
            # Get the target version
            target_context = await self.get_version(context_id, target_version, branch, user_id)
            if not target_context:
                raise ValueError(f"Version {target_version} not found")
            
            # Get current context for permission check
            current_context = await self.persistence.get_context(context_id, user_id)
            if not current_context:
                raise ValueError(f"Context {context_id} not found")
            
            # Create new version with reverted data if requested
            if create_revert_version:
                # Update the current context with target version data
                reverted_context = current_context.copy(deep=True)
                reverted_context.data = target_context.data.copy()
                reverted_context.metadata = target_context.metadata.copy()
                reverted_context.current_version += 1
                
                # Store as new version
                await self.create_version(
                    reverted_context,
                    user_id=user_id,
                    message=f"Reverted to version {target_version}",
                    tags=["revert"],
                    branch=branch,
                )
                
                # Update the main context
                await self.persistence.update_context(
                    context_id,
                    {
                        "data": reverted_context.data,
                        "metadata": reverted_context.metadata,
                    },
                    user_id=user_id,
                    create_version=False,  # We already created one
                )
                
                return reverted_context
            else:
                # Direct replacement
                await self.persistence.update_context(
                    context_id,
                    {
                        "data": target_context.data,
                        "metadata": target_context.metadata,
                    },
                    user_id=user_id,
                )
                
                return target_context
                
        except Exception as e:
            logger.error(f"Failed to revert to version: {e}")
            raise
    
    async def create_branch(
        self,
        context_id: str,
        new_branch: str,
        source_branch: str = "main",
        source_version: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Create a new branch from an existing version."""
        try:
            # Get source version
            if source_version is None:
                # Use latest version from source branch
                history = await self.get_version_history(context_id, source_branch, limit=1)
                if not history:
                    raise ValueError(f"No versions found in branch {source_branch}")
                source_version = history[0].version_number
            
            source_context = await self.get_version(context_id, source_version, source_branch, user_id)
            if not source_context:
                raise ValueError(f"Source version {source_version} not found in branch {source_branch}")
            
            # Create first version in new branch
            await self.create_version(
                source_context,
                user_id=user_id,
                message=f"Created branch from {source_branch}@{source_version}",
                tags=["branch"],
                branch=new_branch,
            )
            
            logger.info(
                "Branch created",
                context_id=context_id,
                new_branch=new_branch,
                source_branch=source_branch,
                source_version=source_version,
            )
            
            return new_branch
            
        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            raise
    
    async def merge_branches(
        self,
        context_id: str,
        source_branch: str,
        target_branch: str,
        user_id: Optional[str] = None,
        conflict_resolution: str = "manual",
    ) -> Context:
        """Merge one branch into another."""
        try:
            # Get latest versions from both branches
            source_history = await self.get_version_history(context_id, source_branch, limit=1)
            target_history = await self.get_version_history(context_id, target_branch, limit=1)
            
            if not source_history or not target_history:
                raise ValueError("One or both branches have no versions")
            
            source_context = await self.get_version(
                context_id, source_history[0].version_number, source_branch, user_id
            )
            target_context = await self.get_version(
                context_id, target_history[0].version_number, target_branch, user_id
            )
            
            # Calculate merge diff
            merge_diff = await self._calculate_diff(
                target_context, source_context,
                target_history[0].version_number,
                source_history[0].version_number,
            )
            
            # Perform merge based on conflict resolution strategy
            merged_context = await self._perform_merge(
                target_context, source_context, merge_diff, conflict_resolution
            )
            
            # Create merge version
            await self.create_version(
                merged_context,
                user_id=user_id,
                message=f"Merged {source_branch} into {target_branch}",
                tags=["merge"],
                branch=target_branch,
            )
            
            logger.info(
                "Branches merged",
                context_id=context_id,
                source_branch=source_branch,
                target_branch=target_branch,
                conflict_resolution=conflict_resolution,
            )
            
            return merged_context
            
        except Exception as e:
            logger.error(f"Failed to merge branches: {e}")
            raise
    
    async def _determine_storage_strategy(
        self, context: Context, previous_version: Optional[Context]
    ) -> str:
        """Determine the optimal storage strategy for a version."""
        if previous_version is None:
            return VersioningStrategy.FULL_SNAPSHOT
        
        if self.versioning_strategy == VersioningStrategy.FULL_SNAPSHOT:
            return VersioningStrategy.FULL_SNAPSHOT
        elif self.versioning_strategy == VersioningStrategy.INCREMENTAL:
            return VersioningStrategy.INCREMENTAL
        elif self.versioning_strategy == VersioningStrategy.COMPRESSED_DELTA:
            return VersioningStrategy.COMPRESSED_DELTA
        else:  # HYBRID
            # Calculate diff size to make decision
            diff = await self._calculate_diff(previous_version, context, 0, 0)
            diff_ratio = diff.diff_size_bytes / context.size_bytes if context.size_bytes > 0 else 1.0
            
            if diff_ratio < self.diff_compression_threshold:
                return VersioningStrategy.COMPRESSED_DELTA
            else:
                return VersioningStrategy.FULL_SNAPSHOT
    
    async def _store_full_version(self, context: Context, version: ContextVersion) -> None:
        """Store a complete version snapshot."""
        # Serialize and compress the full context
        serialized = self.serializer.serialize(context)
        compressed_data, compression_stats = self.compressor.compress(serialized)
        
        # Calculate checksum
        version.checksum = hashlib.sha256(compressed_data).hexdigest()
        version.compressed = compression_stats["compressed"]
        version.compression_type = CompressionType(compression_stats["compression_type"])
        version.size_bytes = compression_stats["original_size"]
        
        # Store in database
        async with self.persistence._get_async_connection() as conn:
            await conn.execute("""
                INSERT INTO context_versions (
                    version_id, context_id, version_number, created_by,
                    parent_version, branch, data_compressed, checksum,
                    compressed, compression_type, size_bytes, metadata, tags
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                version.version_id, version.context_id, version.version_number,
                version.created_by, version.parent_version, version.branch,
                compressed_data, version.checksum, version.compressed,
                version.compression_type.value, version.size_bytes,
                version.metadata, version.tags
            )
    
    async def _store_incremental_version(
        self, context: Context, version: ContextVersion, previous_version: Context
    ) -> None:
        """Store only the differences from the previous version."""
        # Calculate diff
        diff = await self._calculate_diff(previous_version, context, 0, 0)
        
        # Serialize and compress the diff
        diff_data = {
            "added": diff.added,
            "modified": diff.modified,
            "removed": diff.removed,
            "is_delta": True,
            "base_version": previous_version.current_version,
        }
        
        serialized = self.serializer.serialize(diff_data)
        compressed_data, compression_stats = self.compressor.compress(serialized)
        
        version.checksum = hashlib.sha256(compressed_data).hexdigest()
        version.compressed = compression_stats["compressed"]
        version.compression_type = CompressionType(compression_stats["compression_type"])
        version.size_bytes = compression_stats["original_size"]
        
        # Store delta version
        async with self.persistence._get_async_connection() as conn:
            await conn.execute("""
                INSERT INTO context_versions (
                    version_id, context_id, version_number, created_by,
                    parent_version, branch, data_compressed, checksum,
                    compressed, compression_type, size_bytes, metadata, tags
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                version.version_id, version.context_id, version.version_number,
                version.created_by, version.parent_version, version.branch,
                compressed_data, version.checksum, version.compressed,
                version.compression_type.value, version.size_bytes,
                version.metadata, version.tags
            )
        
        # Track space savings
        original_size = len(self.serializer.serialize(context))
        space_saved = original_size - version.size_bytes
        self._version_stats["space_saved_bytes"] += space_saved
    
    async def _store_compressed_delta_version(
        self, context: Context, version: ContextVersion, previous_version: Context
    ) -> None:
        """Store compressed delta version with maximum compression."""
        await self._store_incremental_version(context, version, previous_version)
    
    async def _store_hybrid_version(
        self, context: Context, version: ContextVersion, previous_version: Context
    ) -> None:
        """Store version using hybrid strategy."""
        # Decision was made in _determine_storage_strategy
        if hasattr(version, '_storage_strategy'):
            strategy = version._storage_strategy
        else:
            strategy = await self._determine_storage_strategy(context, previous_version)
        
        if strategy == VersioningStrategy.FULL_SNAPSHOT:
            await self._store_full_version(context, version)
        else:
            await self._store_compressed_delta_version(context, version, previous_version)
    
    async def _reconstruct_context_from_version(self, version_row) -> Context:
        """Reconstruct a context from version data."""
        # Decompress version data
        decompressed = self.compressor.decompress(
            version_row["data_compressed"],
            CompressionType(version_row["compression_type"])
        )
        
        # Deserialize
        version_data = self.serializer.deserialize(decompressed, dict)
        
        # Check if this is a delta version
        if version_data.get("is_delta", False):
            # Reconstruct from base version and apply delta
            base_version = version_data["base_version"]
            base_context = await self.get_version(
                str(version_row["context_id"]), base_version, version_row["branch"]
            )
            
            if not base_context:
                raise ValueError(f"Base version {base_version} not found for delta reconstruction")
            
            # Apply delta changes
            reconstructed_data = base_context.data.copy()
            
            # Apply modifications
            for key, value in version_data["modified"].items():
                self._apply_nested_change(reconstructed_data, key, value)
            
            # Apply additions
            for key, value in version_data["added"].items():
                self._apply_nested_change(reconstructed_data, key, value)
            
            # Remove deleted keys
            for key_path in version_data["removed"]:
                self._remove_nested_key(reconstructed_data, key_path)
            
            # Create context with reconstructed data
            base_context.data = reconstructed_data
            base_context.current_version = version_row["version_number"]
            
            return base_context
        else:
            # Full version - deserialize directly
            return self.serializer.deserialize(decompressed, Context)
    
    async def _calculate_diff(
        self, context1: Context, context2: Context, version1: int, version2: int
    ) -> ContextDiff:
        """Calculate differences between two contexts."""
        diff = ContextDiff(
            context_id=context1.context_id,
            from_version=version1,
            to_version=version2,
        )
        
        # Deep diff calculation
        diff.added, diff.modified, diff.removed = await self._deep_diff(
            context1.data, context2.data
        )
        
        # Calculate diff size
        diff_data = {
            "added": diff.added,
            "modified": diff.modified,
            "removed": diff.removed,
        }
        diff_serialized = self.serializer.serialize(diff_data)
        diff.diff_size_bytes = len(diff_serialized.encode('utf-8'))
        
        return diff
    
    async def _deep_diff(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> Tuple[Dict, Dict, List]:
        """Calculate deep differences between two dictionaries."""
        added = {}
        modified = {}
        removed = []
        
        # Find added and modified keys
        for key, value in obj2.items():
            if key not in obj1:
                added[key] = value
            elif obj1[key] != value:
                if isinstance(value, dict) and isinstance(obj1[key], dict):
                    # Recursive diff for nested objects
                    nested_added, nested_modified, nested_removed = await self._deep_diff(obj1[key], value)
                    if nested_added or nested_modified or nested_removed:
                        modified[key] = {
                            "added": nested_added,
                            "modified": nested_modified,
                            "removed": nested_removed,
                        }
                else:
                    modified[key] = value
        
        # Find removed keys
        for key in obj1:
            if key not in obj2:
                removed.append(key)
        
        return added, modified, removed
    
    def _apply_nested_change(self, data: Dict[str, Any], key_path: str, value: Any) -> None:
        """Apply a nested change to data dictionary."""
        keys = key_path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _remove_nested_key(self, data: Dict[str, Any], key_path: str) -> None:
        """Remove a nested key from data dictionary."""
        keys = key_path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                return  # Key doesn't exist
            current = current[key]
        
        if keys[-1] in current:
            del current[keys[-1]]
    
    async def _perform_merge(
        self,
        target_context: Context,
        source_context: Context,
        merge_diff: ContextDiff,
        conflict_resolution: str,
    ) -> Context:
        """Perform merge operation with conflict resolution."""
        merged_context = target_context.copy(deep=True)
        
        if conflict_resolution == "source_wins":
            # Source overwrites target
            merged_context.data = source_context.data.copy()
        elif conflict_resolution == "target_wins":
            # Keep target as is
            pass
        else:  # "manual" or other strategies
            # Intelligent merge - apply non-conflicting changes
            for key, value in merge_diff.added.items():
                if key not in merged_context.data:
                    self._apply_nested_change(merged_context.data, key, value)
            
            # For conflicts, keep target version (manual resolution needed)
        
        merged_context.current_version += 1
        return merged_context
    
    async def _cleanup_old_versions(self, context_id: str, branch: str) -> None:
        """Clean up old versions based on retention policy."""
        if self.max_versions_per_branch <= 0:
            return
        
        try:
            async with self.persistence._get_async_connection() as conn:
                # Get version count
                count_row = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM context_versions 
                    WHERE context_id = $1 AND branch = $2
                """, context_id, branch)
                
                if count_row["count"] > self.max_versions_per_branch:
                    # Delete oldest versions
                    versions_to_delete = count_row["count"] - self.max_versions_per_branch
                    
                    await conn.execute("""
                        DELETE FROM context_versions 
                        WHERE version_id IN (
                            SELECT version_id FROM context_versions 
                            WHERE context_id = $1 AND branch = $2
                            ORDER BY created_at ASC 
                            LIMIT $3
                        )
                    """, context_id, branch, versions_to_delete)
                    
                    logger.debug(
                        "Old versions cleaned up",
                        context_id=context_id,
                        branch=branch,
                        deleted_count=versions_to_delete,
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup old versions: {e}")
    
    def _update_avg_stat(self, stat_name: str, new_value: float) -> None:
        """Update running average statistic."""
        current_avg = self._version_stats[stat_name]
        count_key = stat_name.replace("avg_", "").replace("_time", "s_") + "created"
        if "retrieval" in stat_name:
            count_key = "versions_retrieved"
        else:
            count_key = "versions_created"
        
        count = self._version_stats[count_key]
        if count > 1:
            self._version_stats[stat_name] = (
                (current_avg * (count - 1) + new_value) / count
            )
        else:
            self._version_stats[stat_name] = new_value
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get versioning performance statistics."""
        return {
            "version_stats": self._version_stats,
            "strategy": self.versioning_strategy,
            "max_versions_per_branch": self.max_versions_per_branch,
            "auto_cleanup_days": self.auto_cleanup_days,
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self.serializer, 'cleanup'):
            self.serializer.cleanup()
        if hasattr(self.compressor, 'cleanup'):
            self.compressor.cleanup()
        
        logger.info("Context version manager cleaned up")