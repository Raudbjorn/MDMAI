"""Access control and permission management system."""

import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from config.logging_config import get_logger

logger = get_logger(__name__)


class Permission(Enum):
    """Available permissions in the system."""

    # Campaign permissions
    CAMPAIGN_CREATE = "campaign.create"
    CAMPAIGN_READ = "campaign.read"
    CAMPAIGN_UPDATE = "campaign.update"
    CAMPAIGN_DELETE = "campaign.delete"
    CAMPAIGN_ROLLBACK = "campaign.rollback"

    # Source management permissions
    SOURCE_ADD = "source.add"
    SOURCE_READ = "source.read"
    SOURCE_DELETE = "source.delete"
    SOURCE_UPDATE = "source.update"

    # Search permissions
    SEARCH_BASIC = "search.basic"
    SEARCH_ADVANCED = "search.advanced"
    SEARCH_ANALYTICS = "search.analytics"

    # Character generation permissions
    CHARACTER_CREATE = "character.create"
    CHARACTER_READ = "character.read"
    CHARACTER_UPDATE = "character.update"
    CHARACTER_DELETE = "character.delete"

    # Session management permissions
    SESSION_CREATE = "session.create"
    SESSION_READ = "session.read"
    SESSION_UPDATE = "session.update"
    SESSION_DELETE = "session.delete"

    # Personality management permissions
    PERSONALITY_CREATE = "personality.create"
    PERSONALITY_READ = "personality.read"
    PERSONALITY_UPDATE = "personality.update"
    PERSONALITY_DELETE = "personality.delete"
    PERSONALITY_APPLY = "personality.apply"

    # System permissions
    SYSTEM_ADMIN = "system.admin"
    SYSTEM_CONFIG = "system.config"
    SYSTEM_MONITOR = "system.monitor"

    # Cache management permissions
    CACHE_READ = "cache.read"
    CACHE_CLEAR = "cache.clear"
    CACHE_CONFIG = "cache.config"


class ResourceType(Enum):
    """Types of resources that can be controlled."""

    CAMPAIGN = "campaign"
    SOURCE = "source"
    CHARACTER = "character"
    NPC = "npc"
    SESSION = "session"
    PERSONALITY = "personality"
    SEARCH = "search"
    CACHE = "cache"
    SYSTEM = "system"


class AccessLevel(Enum):
    """Access levels for resources."""

    NONE = 0
    READ = 1
    WRITE = 2
    DELETE = 3
    ADMIN = 4


class User(BaseModel):
    """User model for access control."""

    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    permissions: Set[Permission] = Field(default_factory=set)
    campaign_access: Dict[str, AccessLevel] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        use_enum_values = False


class Role(BaseModel):
    """Role model for grouping permissions."""

    role_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    permissions: Set[Permission] = Field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        use_enum_values = False


class Session(BaseModel):
    """User session for authentication."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    token: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


class AccessControlManager:
    """Manages access control and permissions."""

    # Default roles
    SYSTEM_ROLES = {
        "admin": {
            "description": "System administrator with full access",
            "permissions": [p for p in Permission],  # All permissions
        },
        "game_master": {
            "description": "Game master with campaign management access",
            "permissions": [
                Permission.CAMPAIGN_CREATE,
                Permission.CAMPAIGN_READ,
                Permission.CAMPAIGN_UPDATE,
                Permission.CAMPAIGN_DELETE,
                Permission.CAMPAIGN_ROLLBACK,
                Permission.SOURCE_ADD,
                Permission.SOURCE_READ,
                Permission.CHARACTER_CREATE,
                Permission.CHARACTER_READ,
                Permission.CHARACTER_UPDATE,
                Permission.CHARACTER_DELETE,
                Permission.SESSION_CREATE,
                Permission.SESSION_READ,
                Permission.SESSION_UPDATE,
                Permission.SESSION_DELETE,
                Permission.PERSONALITY_CREATE,
                Permission.PERSONALITY_READ,
                Permission.PERSONALITY_APPLY,
                Permission.SEARCH_BASIC,
                Permission.SEARCH_ADVANCED,
            ],
        },
        "player": {
            "description": "Player with limited access",
            "permissions": [
                Permission.CAMPAIGN_READ,
                Permission.CHARACTER_READ,
                Permission.CHARACTER_UPDATE,
                Permission.SESSION_READ,
                Permission.SEARCH_BASIC,
                Permission.PERSONALITY_READ,
            ],
        },
        "viewer": {
            "description": "Read-only access",
            "permissions": [
                Permission.CAMPAIGN_READ,
                Permission.SOURCE_READ,
                Permission.CHARACTER_READ,
                Permission.SESSION_READ,
                Permission.SEARCH_BASIC,
            ],
        },
    }

    def __init__(self, enable_auth: bool = False, session_timeout_minutes: int = 60):
        """
        Initialize access control manager.

        Args:
            enable_auth: Whether to enable authentication
            session_timeout_minutes: Session timeout in minutes
        """
        self.enable_auth = enable_auth
        self.session_timeout = timedelta(minutes=session_timeout_minutes)

        # In-memory storage (replace with database in production)
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.sessions: Dict[str, Session] = {}

        # Campaign ownership
        self.campaign_owners: Dict[str, str] = {}  # campaign_id -> user_id
        self.campaign_members: Dict[str, Set[str]] = {}  # campaign_id -> set of user_ids

        # Initialize system roles
        self._initialize_system_roles()

        # Create default admin user if auth is disabled (for testing)
        if not enable_auth:
            self._create_default_admin()

        logger.info(
            "Access control manager initialized",
            auth_enabled=enable_auth,
            session_timeout=session_timeout_minutes,
        )

    def _initialize_system_roles(self) -> None:
        """Initialize system-defined roles."""
        for role_name, role_data in self.SYSTEM_ROLES.items():
            role = Role(
                role_id=f"system_{role_name}",
                name=role_name,
                description=role_data["description"],
                permissions=set(role_data["permissions"]),
                is_system_role=True,
            )
            self.roles[role.role_id] = role
            logger.debug(f"Initialized system role: {role_name}")

    def _create_default_admin(self) -> None:
        """Create default admin user for testing."""
        admin_user = User(
            user_id="default_admin",
            username="admin",
            email="admin@local",
            roles=["system_admin"],
            permissions=set(Permission),  # All permissions
        )
        self.users[admin_user.user_id] = admin_user
        logger.info("Created default admin user (auth disabled)")

    def authenticate_user(
        self, username: str, password: str, ip_address: Optional[str] = None
    ) -> Optional[Session]:
        """
        Authenticate a user and create a session.

        Args:
            username: Username
            password: Password (would be hashed in production)
            ip_address: Client IP address

        Returns:
            Session object if authentication successful
        """
        if not self.enable_auth:
            # Return default session when auth is disabled
            user = self.users.get("default_admin")
            if user:
                session = Session(
                    user_id=user.user_id,
                    expires_at=datetime.utcnow() + self.session_timeout,
                    ip_address=ip_address,
                )
                self.sessions[session.session_id] = session
                return session
            return None

        # Find user by username (in production, check hashed password)
        user = None
        for u in self.users.values():
            if u.username == username and u.is_active:
                user = u
                break

        if not user:
            logger.warning(f"Authentication failed for username: {username}")
            return None

        # Create session
        session = Session(
            user_id=user.user_id,
            expires_at=datetime.utcnow() + self.session_timeout,
            ip_address=ip_address,
        )
        self.sessions[session.session_id] = session

        # Update user last active
        user.last_active = datetime.utcnow()

        logger.info(f"User authenticated: {username}")
        return session

    def validate_session(self, session_token: str) -> Optional[User]:
        """
        Validate a session token and return the user.

        Args:
            session_token: Session token to validate

        Returns:
            User object if session is valid
        """
        if not self.enable_auth:
            # Return default admin when auth is disabled
            return self.users.get("default_admin")

        # Find session by token
        session = None
        for s in self.sessions.values():
            if s.token == session_token and s.is_active:
                session = s
                break

        if not session:
            logger.debug("Invalid session token")
            return None

        # Check expiration
        if datetime.utcnow() > session.expires_at:
            session.is_active = False
            logger.debug(f"Session expired for user: {session.user_id}")
            return None

        # Get user
        user = self.users.get(session.user_id)
        if user and user.is_active:
            user.last_active = datetime.utcnow()
            return user

        return None

    def check_permission(
        self,
        user: User,
        permission: Permission,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
    ) -> bool:
        """
        Check if user has a specific permission.

        Args:
            user: User to check
            permission: Permission to check
            resource_type: Type of resource (for resource-specific checks)
            resource_id: Resource ID (for resource-specific checks)

        Returns:
            True if user has permission
        """
        if not self.enable_auth:
            return True  # All permissions granted when auth is disabled

        # Check if user has system admin permission
        if Permission.SYSTEM_ADMIN in user.permissions:
            return True

        # Check direct user permissions
        if permission in user.permissions:
            logger.debug(f"User {user.username} has direct permission: {permission.value}")
            return True

        # Check role-based permissions
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if role and permission in role.permissions:
                logger.debug(
                    f"User {user.username} has permission {permission.value} via role {role.name}"
                )
                return True

        # Check resource-specific permissions
        if resource_type == ResourceType.CAMPAIGN and resource_id:
            return self._check_campaign_permission(user, permission, resource_id)

        logger.debug(f"User {user.username} denied permission: {permission.value}")
        return False

    def _check_campaign_permission(
        self, user: User, permission: Permission, campaign_id: str
    ) -> bool:
        """
        Check campaign-specific permissions.

        Args:
            user: User to check
            permission: Permission to check
            campaign_id: Campaign ID

        Returns:
            True if user has permission for this campaign
        """
        # Check if user is campaign owner
        if self.campaign_owners.get(campaign_id) == user.user_id:
            return True

        # Check if user is campaign member
        if campaign_id in self.campaign_members:
            if user.user_id in self.campaign_members[campaign_id]:
                # Members have read access
                if permission in [
                    Permission.CAMPAIGN_READ,
                    Permission.CHARACTER_READ,
                    Permission.SESSION_READ,
                ]:
                    return True

        # Check user's specific campaign access level
        access_level = user.campaign_access.get(campaign_id)
        if access_level:
            required_level = self._permission_to_access_level(permission)
            if access_level.value >= required_level.value:
                return True

        return False

    def _permission_to_access_level(self, permission: Permission) -> AccessLevel:
        """
        Convert permission to required access level.

        Args:
            permission: Permission to convert

        Returns:
            Required access level
        """
        if "read" in permission.value.lower():
            return AccessLevel.READ
        elif "update" in permission.value.lower():
            return AccessLevel.WRITE
        elif "delete" in permission.value.lower():
            return AccessLevel.DELETE
        elif "create" in permission.value.lower():
            return AccessLevel.WRITE
        elif "admin" in permission.value.lower():
            return AccessLevel.ADMIN
        else:
            return AccessLevel.READ

    def grant_campaign_access(
        self, user_id: str, campaign_id: str, access_level: AccessLevel
    ) -> bool:
        """
        Grant user access to a campaign.

        Args:
            user_id: User ID
            campaign_id: Campaign ID
            access_level: Access level to grant

        Returns:
            True if access granted
        """
        user = self.users.get(user_id)
        if not user:
            logger.warning(f"User not found: {user_id}")
            return False

        user.campaign_access[campaign_id] = access_level

        # Add to campaign members
        if campaign_id not in self.campaign_members:
            self.campaign_members[campaign_id] = set()
        self.campaign_members[campaign_id].add(user_id)

        logger.info(
            f"Granted {access_level.name} access to campaign {campaign_id} for user {user_id}"
        )
        return True

    def revoke_campaign_access(self, user_id: str, campaign_id: str) -> bool:
        """
        Revoke user access to a campaign.

        Args:
            user_id: User ID
            campaign_id: Campaign ID

        Returns:
            True if access revoked
        """
        user = self.users.get(user_id)
        if not user:
            return False

        # Remove from campaign access
        if campaign_id in user.campaign_access:
            del user.campaign_access[campaign_id]

        # Remove from campaign members
        if campaign_id in self.campaign_members:
            self.campaign_members[campaign_id].discard(user_id)

        logger.info(f"Revoked access to campaign {campaign_id} for user {user_id}")
        return True

    def set_campaign_owner(self, campaign_id: str, user_id: str) -> None:
        """
        Set the owner of a campaign.

        Args:
            campaign_id: Campaign ID
            user_id: User ID of the owner
        """
        self.campaign_owners[campaign_id] = user_id
        self.grant_campaign_access(user_id, campaign_id, AccessLevel.ADMIN)
        logger.info(f"Set user {user_id} as owner of campaign {campaign_id}")

    def create_user(
        self,
        username: str,
        email: Optional[str] = None,
        roles: Optional[List[str]] = None,
        permissions: Optional[Set[Permission]] = None,
    ) -> User:
        """
        Create a new user.

        Args:
            username: Username
            email: Email address
            roles: List of role IDs
            permissions: Direct permissions

        Returns:
            Created user
        """
        user = User(
            username=username,
            email=email,
            roles=roles or [],
            permissions=permissions or set(),
        )
        self.users[user.user_id] = user

        logger.info(f"Created user: {username} (ID: {user.user_id})")
        return user

    def create_role(
        self, name: str, description: str = "", permissions: Optional[Set[Permission]] = None
    ) -> Role:
        """
        Create a custom role.

        Args:
            name: Role name
            description: Role description
            permissions: Role permissions

        Returns:
            Created role
        """
        role = Role(
            name=name,
            description=description,
            permissions=permissions or set(),
            is_system_role=False,
        )
        self.roles[role.role_id] = role

        logger.info(f"Created role: {name} (ID: {role.role_id})")
        return role

    def assign_role(self, user_id: str, role_id: str) -> bool:
        """
        Assign a role to a user.

        Args:
            user_id: User ID
            role_id: Role ID

        Returns:
            True if role assigned
        """
        user = self.users.get(user_id)
        role = self.roles.get(role_id)

        if not user or not role:
            return False

        if role_id not in user.roles:
            user.roles.append(role_id)
            logger.info(f"Assigned role {role.name} to user {user.username}")

        return True

    def remove_role(self, user_id: str, role_id: str) -> bool:
        """
        Remove a role from a user.

        Args:
            user_id: User ID
            role_id: Role ID

        Returns:
            True if role removed
        """
        user = self.users.get(user_id)
        if not user or role_id not in user.roles:
            return False

        user.roles.remove(role_id)
        logger.info(f"Removed role {role_id} from user {user.username}")
        return True

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        Get all permissions for a user (direct + role-based).

        Args:
            user_id: User ID

        Returns:
            Set of all user permissions
        """
        user = self.users.get(user_id)
        if not user:
            return set()

        # Collect all permissions
        all_permissions = user.permissions.copy()

        # Add role permissions
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if role:
                all_permissions.update(role.permissions)

        return all_permissions

    def list_user_campaigns(self, user_id: str) -> List[tuple[str, AccessLevel]]:
        """
        List all campaigns a user has access to.

        Args:
            user_id: User ID

        Returns:
            List of (campaign_id, access_level) tuples
        """
        user = self.users.get(user_id)
        if not user:
            return []

        campaigns = []

        # Add owned campaigns
        for campaign_id, owner_id in self.campaign_owners.items():
            if owner_id == user_id:
                campaigns.append((campaign_id, AccessLevel.ADMIN))

        # Add campaigns with specific access
        for campaign_id, access_level in user.campaign_access.items():
            if campaign_id not in [c[0] for c in campaigns]:
                campaigns.append((campaign_id, access_level))

        return campaigns

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        now = datetime.utcnow()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if now > session.expires_at or not session.is_active:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)