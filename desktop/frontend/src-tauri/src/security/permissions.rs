//! Granular Permission Management
//! 
//! This module provides fine-grained permission control system:
//! - Role-based access control (RBAC)
//! - Resource-based permissions
//! - Dynamic permission evaluation
//! - Permission inheritance and delegation
//! - Audit trail for permission changes

use super::*;
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// Permission manager
pub struct PermissionManager {
    config: SecurityConfig,
    roles: Arc<RwLock<HashMap<String, Role>>>,
    permissions: Arc<RwLock<HashMap<String, Permission>>>,
    user_assignments: Arc<RwLock<HashMap<String, UserPermissions>>>,
    resource_policies: Arc<RwLock<HashMap<String, ResourcePolicy>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: String,
    pub name: String,
    pub description: String,
    pub permissions: HashSet<String>,
    pub parent_roles: HashSet<String>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub id: String,
    pub name: String,
    pub description: String,
    pub resource_type: String,
    pub actions: HashSet<String>,
    pub constraints: HashMap<String, serde_json::Value>,
    pub created_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPermissions {
    pub user_id: String,
    pub roles: HashSet<String>,
    pub direct_permissions: HashSet<String>,
    pub denied_permissions: HashSet<String>,
    pub effective_permissions: HashSet<String>,
    pub last_calculated: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePolicy {
    pub resource_id: String,
    pub resource_type: String,
    pub required_permissions: HashSet<String>,
    pub access_conditions: HashMap<String, serde_json::Value>,
    pub inherit_parent_permissions: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionRequest {
    pub user_id: String,
    pub resource_id: String,
    pub action: String,
    pub context: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionResult {
    pub granted: bool,
    pub reason: String,
    pub matched_permissions: Vec<String>,
    pub applied_policies: Vec<String>,
}

impl PermissionManager {
    pub fn new(config: &SecurityConfig) -> SecurityResult<Self> {
        Ok(Self {
            config: config.clone(),
            roles: Arc::new(RwLock::new(HashMap::new())),
            permissions: Arc::new(RwLock::new(HashMap::new())),
            user_assignments: Arc::new(RwLock::new(HashMap::new())),
            resource_policies: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn initialize(&self) -> SecurityResult<()> {
        // Initialize default roles and permissions
        self.create_default_roles().await?;
        self.create_default_permissions().await?;

        log::info!("Permission manager initialized");
        Ok(())
    }

    /// Create a new role
    pub async fn create_role(&self, role: Role) -> SecurityResult<()> {
        // Validate role hierarchy to prevent cycles
        self.validate_role_hierarchy(&role).await?;

        self.roles.write().await.insert(role.id.clone(), role);
        
        // Recalculate effective permissions for affected users
        self.recalculate_affected_permissions(&role.id).await?;

        Ok(())
    }

    /// Create a new permission
    pub async fn create_permission(&self, permission: Permission) -> SecurityResult<()> {
        self.permissions.write().await.insert(permission.id.clone(), permission);
        Ok(())
    }

    /// Assign role to user
    pub async fn assign_role(&self, user_id: &str, role_id: &str) -> SecurityResult<()> {
        // Verify role exists
        let roles = self.roles.read().await;
        if !roles.contains_key(role_id) {
            return Err(SecurityError::AuthorizationDenied {
                operation: format!("Role not found: {}", role_id),
            });
        }
        drop(roles);

        let mut assignments = self.user_assignments.write().await;
        let user_perms = assignments.entry(user_id.to_string()).or_insert_with(|| {
            UserPermissions {
                user_id: user_id.to_string(),
                roles: HashSet::new(),
                direct_permissions: HashSet::new(),
                denied_permissions: HashSet::new(),
                effective_permissions: HashSet::new(),
                last_calculated: SystemTime::now(),
            }
        });

        user_perms.roles.insert(role_id.to_string());
        drop(assignments);

        // Recalculate effective permissions
        self.calculate_effective_permissions(user_id).await?;

        Ok(())
    }

    /// Grant direct permission to user
    pub async fn grant_permission(&self, user_id: &str, permission_id: &str) -> SecurityResult<()> {
        // Verify permission exists
        let permissions = self.permissions.read().await;
        if !permissions.contains_key(permission_id) {
            return Err(SecurityError::AuthorizationDenied {
                operation: format!("Permission not found: {}", permission_id),
            });
        }
        drop(permissions);

        let mut assignments = self.user_assignments.write().await;
        let user_perms = assignments.entry(user_id.to_string()).or_insert_with(|| {
            UserPermissions {
                user_id: user_id.to_string(),
                roles: HashSet::new(),
                direct_permissions: HashSet::new(),
                denied_permissions: HashSet::new(),
                effective_permissions: HashSet::new(),
                last_calculated: SystemTime::now(),
            }
        });

        user_perms.direct_permissions.insert(permission_id.to_string());
        drop(assignments);

        // Recalculate effective permissions
        self.calculate_effective_permissions(user_id).await?;

        Ok(())
    }

    /// Check if user has permission
    pub async fn check_permission(&self, request: &PermissionRequest) -> SecurityResult<PermissionResult> {
        // Get user's effective permissions
        let assignments = self.user_assignments.read().await;
        let user_perms = assignments.get(&request.user_id).ok_or_else(|| {
            SecurityError::AuthorizationDenied {
                operation: format!("User not found: {}", request.user_id),
            }
        })?;

        // Check if permission is explicitly denied
        if user_perms.denied_permissions.iter().any(|p| self.permission_matches(p, &request.resource_id, &request.action)) {
            return Ok(PermissionResult {
                granted: false,
                reason: "Permission explicitly denied".to_string(),
                matched_permissions: Vec::new(),
                applied_policies: Vec::new(),
            });
        }

        // Check resource policies
        let resource_policies = self.resource_policies.read().await;
        let mut applied_policies = Vec::new();
        let mut required_perms = HashSet::new();

        if let Some(policy) = resource_policies.get(&request.resource_id) {
            required_perms.extend(policy.required_permissions.clone());
            applied_policies.push(policy.resource_id.clone());

            // Check access conditions
            if !self.evaluate_access_conditions(&policy.access_conditions, &request.context) {
                return Ok(PermissionResult {
                    granted: false,
                    reason: "Access conditions not met".to_string(),
                    matched_permissions: Vec::new(),
                    applied_policies,
                });
            }
        }

        // Find matching permissions
        let permissions = self.permissions.read().await;
        let mut matched_permissions = Vec::new();

        for perm_id in &user_perms.effective_permissions {
            if let Some(permission) = permissions.get(perm_id) {
                if self.permission_applies(permission, &request.resource_id, &request.action) {
                    matched_permissions.push(perm_id.clone());
                }
            }
        }

        // Check if any required permissions are satisfied
        let granted = if required_perms.is_empty() {
            !matched_permissions.is_empty()
        } else {
            required_perms.iter().any(|req_perm| {
                user_perms.effective_permissions.contains(req_perm)
            })
        };

        Ok(PermissionResult {
            granted,
            reason: if granted { "Access granted".to_string() } else { "Insufficient permissions".to_string() },
            matched_permissions,
            applied_policies,
        })
    }

    /// Calculate effective permissions for a user
    async fn calculate_effective_permissions(&self, user_id: &str) -> SecurityResult<()> {
        let mut effective_permissions = HashSet::new();

        let mut assignments = self.user_assignments.write().await;
        if let Some(user_perms) = assignments.get_mut(user_id) {
            // Add direct permissions
            effective_permissions.extend(user_perms.direct_permissions.clone());

            // Add permissions from roles (including inherited)
            let roles = self.roles.read().await;
            for role_id in &user_perms.roles {
                if let Some(role) = roles.get(role_id) {
                    effective_permissions.extend(role.permissions.clone());
                    
                    // Add permissions from parent roles
                    self.add_inherited_permissions(&roles, role, &mut effective_permissions);
                }
            }

            // Remove denied permissions
            effective_permissions.retain(|p| !user_perms.denied_permissions.contains(p));

            user_perms.effective_permissions = effective_permissions;
            user_perms.last_calculated = SystemTime::now();
        }

        Ok(())
    }

    /// Add permissions from parent roles recursively
    fn add_inherited_permissions(
        &self,
        roles: &HashMap<String, Role>,
        role: &Role,
        permissions: &mut HashSet<String>,
    ) {
        for parent_role_id in &role.parent_roles {
            if let Some(parent_role) = roles.get(parent_role_id) {
                permissions.extend(parent_role.permissions.clone());
                // Recursively add from parent's parents
                self.add_inherited_permissions(roles, parent_role, permissions);
            }
        }
    }

    /// Check if a permission matches a resource and action
    fn permission_matches(&self, permission_id: &str, resource_id: &str, action: &str) -> bool {
        // Simple string matching for now
        // In a real implementation, this would be more sophisticated
        permission_id.contains(resource_id) || permission_id.contains(action)
    }

    /// Check if a permission applies to a resource and action
    fn permission_applies(&self, permission: &Permission, resource_id: &str, action: &str) -> bool {
        // Check resource type match
        let resource_matches = permission.resource_type == "*" || resource_id.starts_with(&permission.resource_type);
        
        // Check action match
        let action_matches = permission.actions.contains("*") || permission.actions.contains(action);

        resource_matches && action_matches
    }

    /// Evaluate access conditions
    fn evaluate_access_conditions(
        &self,
        conditions: &HashMap<String, serde_json::Value>,
        context: &HashMap<String, serde_json::Value>,
    ) -> bool {
        for (condition_key, condition_value) in conditions {
            if let Some(context_value) = context.get(condition_key) {
                if context_value != condition_value {
                    return false;
                }
            } else {
                return false; // Required condition not provided in context
            }
        }
        true
    }

    /// Validate role hierarchy to prevent cycles
    async fn validate_role_hierarchy(&self, role: &Role) -> SecurityResult<()> {
        let roles = self.roles.read().await;
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        self.check_role_cycle(&roles, &role.id, &role.parent_roles, &mut visited, &mut stack)?;

        Ok(())
    }

    /// Recursively check for cycles in role hierarchy
    fn check_role_cycle(
        &self,
        roles: &HashMap<String, Role>,
        role_id: &str,
        parent_roles: &HashSet<String>,
        visited: &mut HashSet<String>,
        stack: &mut Vec<String>,
    ) -> SecurityResult<()> {
        if stack.contains(&role_id.to_string()) {
            return Err(SecurityError::PolicyViolation {
                policy: format!("Role hierarchy cycle detected: {:?}", stack),
            });
        }

        if visited.contains(role_id) {
            return Ok(());
        }

        visited.insert(role_id.to_string());
        stack.push(role_id.to_string());

        for parent_id in parent_roles {
            if let Some(parent_role) = roles.get(parent_id) {
                self.check_role_cycle(roles, parent_id, &parent_role.parent_roles, visited, stack)?;
            }
        }

        stack.pop();
        Ok(())
    }

    /// Recalculate permissions for users affected by role changes
    async fn recalculate_affected_permissions(&self, role_id: &str) -> SecurityResult<()> {
        let assignments = self.user_assignments.read().await;
        let affected_users: Vec<String> = assignments
            .iter()
            .filter(|(_, user_perms)| user_perms.roles.contains(role_id))
            .map(|(user_id, _)| user_id.clone())
            .collect();
        drop(assignments);

        for user_id in affected_users {
            self.calculate_effective_permissions(&user_id).await?;
        }

        Ok(())
    }

    /// Create default roles and permissions
    async fn create_default_roles(&self) -> SecurityResult<()> {
        let default_roles = vec![
            Role {
                id: "admin".to_string(),
                name: "Administrator".to_string(),
                description: "Full system access".to_string(),
                permissions: ["*".to_string()].iter().cloned().collect(),
                parent_roles: HashSet::new(),
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
            },
            Role {
                id: "user".to_string(),
                name: "Standard User".to_string(),
                description: "Basic user access".to_string(),
                permissions: [
                    "campaign.read".to_string(),
                    "campaign.create".to_string(),
                    "campaign.update".to_string(),
                    "character.read".to_string(),
                    "character.create".to_string(),
                    "character.update".to_string(),
                ].iter().cloned().collect(),
                parent_roles: HashSet::new(),
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
            },
            Role {
                id: "readonly".to_string(),
                name: "Read Only".to_string(),
                description: "Read-only access".to_string(),
                permissions: [
                    "campaign.read".to_string(),
                    "character.read".to_string(),
                ].iter().cloned().collect(),
                parent_roles: HashSet::new(),
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
            },
        ];

        let mut roles = self.roles.write().await;
        for role in default_roles {
            roles.insert(role.id.clone(), role);
        }

        Ok(())
    }

    /// Create default permissions
    async fn create_default_permissions(&self) -> SecurityResult<()> {
        let default_permissions = vec![
            Permission {
                id: "campaign.read".to_string(),
                name: "Read Campaigns".to_string(),
                description: "View campaign data".to_string(),
                resource_type: "campaign".to_string(),
                actions: ["read".to_string()].iter().cloned().collect(),
                constraints: HashMap::new(),
                created_at: SystemTime::now(),
            },
            Permission {
                id: "campaign.create".to_string(),
                name: "Create Campaigns".to_string(),
                description: "Create new campaigns".to_string(),
                resource_type: "campaign".to_string(),
                actions: ["create".to_string()].iter().cloned().collect(),
                constraints: HashMap::new(),
                created_at: SystemTime::now(),
            },
            Permission {
                id: "campaign.update".to_string(),
                name: "Update Campaigns".to_string(),
                description: "Modify existing campaigns".to_string(),
                resource_type: "campaign".to_string(),
                actions: ["update".to_string()].iter().cloned().collect(),
                constraints: HashMap::new(),
                created_at: SystemTime::now(),
            },
            Permission {
                id: "campaign.delete".to_string(),
                name: "Delete Campaigns".to_string(),
                description: "Delete campaigns".to_string(),
                resource_type: "campaign".to_string(),
                actions: ["delete".to_string()].iter().cloned().collect(),
                constraints: HashMap::new(),
                created_at: SystemTime::now(),
            },
        ];

        let mut permissions = self.permissions.write().await;
        for permission in default_permissions {
            permissions.insert(permission.id.clone(), permission);
        }

        Ok(())
    }

    /// Get user's effective permissions
    pub async fn get_user_permissions(&self, user_id: &str) -> SecurityResult<UserPermissions> {
        let assignments = self.user_assignments.read().await;
        assignments.get(user_id).cloned().ok_or_else(|| {
            SecurityError::AuthorizationDenied {
                operation: format!("User not found: {}", user_id),
            }
        })
    }

    /// List all roles
    pub async fn list_roles(&self) -> Vec<Role> {
        self.roles.read().await.values().cloned().collect()
    }

    /// List all permissions
    pub async fn list_permissions(&self) -> Vec<Permission> {
        self.permissions.read().await.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_permission_system() {
        let config = SecurityConfig::default();
        let perm_manager = PermissionManager::new(&config).unwrap();
        perm_manager.initialize().await.unwrap();

        // Assign user role
        perm_manager.assign_role("test_user", "user").await.unwrap();

        // Check permission
        let request = PermissionRequest {
            user_id: "test_user".to_string(),
            resource_id: "campaign_123".to_string(),
            action: "read".to_string(),
            context: HashMap::new(),
        };

        let result = perm_manager.check_permission(&request).await.unwrap();
        assert!(result.granted);
    }

    #[tokio::test]
    async fn test_role_hierarchy_cycle_detection() {
        let config = SecurityConfig::default();
        let perm_manager = PermissionManager::new(&config).unwrap();
        perm_manager.initialize().await.unwrap();

        // Create roles with cycle
        let role_a = Role {
            id: "role_a".to_string(),
            name: "Role A".to_string(),
            description: "Test role A".to_string(),
            permissions: HashSet::new(),
            parent_roles: ["role_b".to_string()].iter().cloned().collect(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };

        let role_b = Role {
            id: "role_b".to_string(),
            name: "Role B".to_string(),
            description: "Test role B".to_string(),
            permissions: HashSet::new(),
            parent_roles: ["role_a".to_string()].iter().cloned().collect(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };

        // This should detect the cycle
        perm_manager.create_role(role_b).await.unwrap();
        assert!(perm_manager.create_role(role_a).await.is_err());
    }
}