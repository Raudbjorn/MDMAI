/*!
 * Startup Optimization Module
 * 
 * Provides intelligent startup optimization including parallel initialization,
 * dependency management, and startup caching.
 */

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, HashSet, VecDeque};
use tokio::sync::{RwLock, Semaphore};
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use log::{info, debug, warn, error};
use super::{PerformanceConfig, StartupConfig};

/// Startup task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupTask {
    pub name: String,
    pub priority: u8, // 0 = highest, 255 = lowest
    pub dependencies: Vec<String>,
    pub estimated_duration_ms: u64,
    pub critical: bool, // If true, startup fails if this task fails
    pub cache_key: Option<String>,
}

/// Startup task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_name: String,
    pub success: bool,
    pub duration: Duration,
    pub error: Option<String>,
    pub cached: bool,
}

/// Dependency graph node
#[derive(Debug, Clone)]
struct DependencyNode {
    task: StartupTask,
    dependencies: HashSet<String>,
    dependents: HashSet<String>,
    completed: bool,
}

/// Startup optimizer
pub struct StartupOptimizer {
    config: Arc<RwLock<PerformanceConfig>>,
    task_registry: Arc<RwLock<HashMap<String, StartupTask>>>,
    cache: Arc<RwLock<HashMap<String, (Instant, serde_json::Value)>>>,
    execution_stats: Arc<RwLock<HashMap<String, Vec<TaskResult>>>>,
}

impl StartupOptimizer {
    pub fn new(config: Arc<RwLock<PerformanceConfig>>) -> Self {
        Self {
            config,
            task_registry: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            execution_stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a startup task
    pub async fn register_task(&self, task: StartupTask) {
        debug!("Registering startup task: {}", task.name);
        self.task_registry.write().await.insert(task.name.clone(), task);
    }

    /// Register multiple startup tasks
    pub async fn register_tasks(&self, tasks: Vec<StartupTask>) {
        for task in tasks {
            self.register_task(task).await;
        }
    }

    /// Execute startup sequence with optimization
    pub async fn execute_startup(&self) -> Result<Vec<TaskResult>, String> {
        let start_time = Instant::now();
        info!("Starting optimized startup sequence");

        // Get configuration
        let config = self.config.read().await.startup.clone();
        
        // Build dependency graph
        let execution_order = self.build_execution_order().await?;
        
        // Clean expired cache entries
        self.clean_cache(&config).await;
        
        // Execute tasks with concurrency control
        let results = self.execute_tasks_parallel(execution_order, &config).await?;
        
        let total_duration = start_time.elapsed();
        info!("Startup sequence completed in {:?}", total_duration);
        
        // Update execution statistics
        self.update_execution_stats(&results).await;
        
        Ok(results)
    }

    /// Build optimal execution order considering dependencies and priorities
    async fn build_execution_order(&self) -> Result<Vec<Vec<String>>, String> {
        let registry = self.task_registry.read().await;
        
        if registry.is_empty() {
            return Ok(vec![]);
        }

        // Build dependency graph
        let mut graph: HashMap<String, DependencyNode> = HashMap::new();
        
        for (name, task) in registry.iter() {
            graph.insert(name.clone(), DependencyNode {
                task: task.clone(),
                dependencies: task.dependencies.iter().cloned().collect(),
                dependents: HashSet::new(),
                completed: false,
            });
        }
        
        // Build reverse dependencies
        for (name, node) in graph.iter() {
            for dep in &node.dependencies {
                if let Some(dep_node) = graph.get_mut(dep) {
                    dep_node.dependents.insert(name.clone());
                }
            }
        }
        
        // Validate dependencies exist
        for (name, node) in &graph {
            for dep in &node.dependencies {
                if !graph.contains_key(dep) {
                    return Err(format!("Task '{}' depends on non-existent task '{}'", name, dep));
                }
            }
        }
        
        // Detect circular dependencies
        self.detect_circular_dependencies(&graph)?;
        
        // Build execution levels using topological sort
        let mut levels = Vec::new();
        let mut remaining: HashSet<String> = graph.keys().cloned().collect();
        
        while !remaining.is_empty() {
            // Find tasks with no remaining dependencies
            let ready: Vec<String> = remaining.iter()
                .filter(|name| {
                    graph[*name].dependencies.iter()
                        .all(|dep| !remaining.contains(dep))
                })
                .cloned()
                .collect();
            
            if ready.is_empty() {
                return Err("Circular dependency detected in remaining tasks".to_string());
            }
            
            // Sort by priority (lower number = higher priority)
            let mut ready_sorted = ready;
            ready_sorted.sort_by_key(|name| graph[name].task.priority);
            
            levels.push(ready_sorted.clone());
            
            // Remove completed tasks
            for task_name in ready_sorted {
                remaining.remove(&task_name);
            }
        }
        
        debug!("Built execution order with {} levels", levels.len());
        for (i, level) in levels.iter().enumerate() {
            debug!("Level {}: {:?}", i, level);
        }
        
        Ok(levels)
    }

    /// Detect circular dependencies using DFS
    fn detect_circular_dependencies(&self, graph: &HashMap<String, DependencyNode>) -> Result<(), String> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        
        for name in graph.keys() {
            if !visited.contains(name) {
                if self.has_cycle_util(name, graph, &mut visited, &mut rec_stack) {
                    return Err(format!("Circular dependency detected involving task '{}'", name));
                }
            }
        }
        
        Ok(())
    }
    
    fn has_cycle_util(
        &self,
        name: &str,
        graph: &HashMap<String, DependencyNode>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>
    ) -> bool {
        visited.insert(name.to_string());
        rec_stack.insert(name.to_string());
        
        if let Some(node) = graph.get(name) {
            for dep in &node.dependencies {
                if !visited.contains(dep) {
                    if self.has_cycle_util(dep, graph, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(dep) {
                    return true;
                }
            }
        }
        
        rec_stack.remove(name);
        false
    }

    /// Execute tasks in parallel with concurrency control
    async fn execute_tasks_parallel(
        &self,
        execution_levels: Vec<Vec<String>>,
        config: &StartupConfig,
    ) -> Result<Vec<TaskResult>, String> {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_tasks));
        let mut all_results = Vec::new();
        
        for (level_index, level) in execution_levels.iter().enumerate() {
            debug!("Executing level {} with {} tasks", level_index, level.len());
            
            let level_start = Instant::now();
            let mut level_futures = Vec::new();
            
            for task_name in level {
                let task = {
                    let registry = self.task_registry.read().await;
                    registry.get(task_name).cloned()
                        .ok_or_else(|| format!("Task '{}' not found in registry", task_name))?
                };
                
                let semaphore_permit = semaphore.clone();
                let cache = self.cache.clone();
                let task_name = task_name.clone();
                
                let future = async move {
                    let _permit = semaphore_permit.acquire().await.unwrap();
                    Self::execute_single_task(task, cache, config).await
                };
                
                level_futures.push(future);
            }
            
            // Execute all tasks in this level concurrently
            let level_results = try_join_all(level_futures).await
                .map_err(|e| format!("Failed to execute level {}: {}", level_index, e))?;
            
            // Check for critical task failures
            for result in &level_results {
                if !result.success {
                    let registry = self.task_registry.read().await;
                    if let Some(task) = registry.get(&result.task_name) {
                        if task.critical {
                            return Err(format!(
                                "Critical task '{}' failed: {}", 
                                result.task_name,
                                result.error.as_deref().unwrap_or("Unknown error")
                            ));
                        }
                    }
                }
            }
            
            all_results.extend(level_results);
            
            let level_duration = level_start.elapsed();
            debug!("Level {} completed in {:?}", level_index, level_duration);
        }
        
        Ok(all_results)
    }

    /// Execute a single startup task
    async fn execute_single_task(
        task: StartupTask,
        cache: Arc<RwLock<HashMap<String, (Instant, serde_json::Value)>>>,
        config: &StartupConfig,
    ) -> Result<TaskResult, String> {
        let start_time = Instant::now();
        debug!("Executing startup task: {}", task.name);
        
        // Check cache first
        if config.enable_cache {
            if let Some(cache_key) = &task.cache_key {
                let cache_guard = cache.read().await;
                if let Some((cache_time, _cached_result)) = cache_guard.get(cache_key) {
                    let age = start_time.duration_since(*cache_time);
                    if age.as_secs() < config.cache_duration {
                        debug!("Using cached result for task: {}", task.name);
                        return Ok(TaskResult {
                            task_name: task.name,
                            success: true,
                            duration: Duration::from_millis(1), // Minimal cache lookup time
                            error: None,
                            cached: true,
                        });
                    }
                }
            }
        }
        
        // Execute the task with timeout
        let task_future = Self::simulate_task_execution(&task);
        let timeout_duration = Duration::from_secs(config.timeout_seconds);
        
        let result = match tokio::time::timeout(timeout_duration, task_future).await {
            Ok(Ok(result)) => {
                // Cache successful results
                if config.enable_cache && result.is_some() {
                    if let Some(cache_key) = &task.cache_key {
                        let mut cache_guard = cache.write().await;
                        cache_guard.insert(cache_key.clone(), (start_time, result.unwrap_or(serde_json::Value::Null)));
                    }
                }
                
                TaskResult {
                    task_name: task.name,
                    success: true,
                    duration: start_time.elapsed(),
                    error: None,
                    cached: false,
                }
            }
            Ok(Err(error)) => {
                TaskResult {
                    task_name: task.name,
                    success: false,
                    duration: start_time.elapsed(),
                    error: Some(error),
                    cached: false,
                }
            }
            Err(_) => {
                TaskResult {
                    task_name: task.name,
                    success: false,
                    duration: timeout_duration,
                    error: Some("Task timeout".to_string()),
                    cached: false,
                }
            }
        };
        
        if result.success {
            debug!("Task '{}' completed successfully in {:?}", task.name, result.duration);
        } else {
            warn!("Task '{}' failed in {:?}: {}", 
                  task.name, result.duration, 
                  result.error.as_deref().unwrap_or("Unknown error"));
        }
        
        Ok(result)
    }

    /// Simulate task execution (replace with actual task execution logic)
    async fn simulate_task_execution(task: &StartupTask) -> Result<Option<serde_json::Value>, String> {
        // Simulate work based on estimated duration
        let duration = Duration::from_millis(task.estimated_duration_ms);
        tokio::time::sleep(duration).await;
        
        // Simulate occasional failures for non-critical tasks
        if !task.critical && task.name.contains("optional") && rand::random::<f32>() < 0.1 {
            return Err("Simulated optional task failure".to_string());
        }
        
        Ok(Some(serde_json::json!({
            "task": task.name,
            "completed_at": chrono::Utc::now(),
            "priority": task.priority
        })))
    }

    /// Clean expired cache entries
    async fn clean_cache(&self, config: &StartupConfig) {
        if !config.enable_cache {
            return;
        }
        
        let now = Instant::now();
        let max_age = Duration::from_secs(config.cache_duration);
        
        let mut cache = self.cache.write().await;
        let initial_size = cache.len();
        
        cache.retain(|_, (timestamp, _)| {
            now.duration_since(*timestamp) < max_age
        });
        
        let removed = initial_size - cache.len();
        if removed > 0 {
            debug!("Cleaned {} expired cache entries", removed);
        }
    }

    /// Update execution statistics
    async fn update_execution_stats(&self, results: &[TaskResult]) {
        let mut stats = self.execution_stats.write().await;
        
        for result in results {
            stats.entry(result.task_name.clone())
                .or_insert_with(Vec::new)
                .push(result.clone());
            
            // Keep only recent results (last 10)
            let task_stats = stats.get_mut(&result.task_name).unwrap();
            if task_stats.len() > 10 {
                task_stats.remove(0);
            }
        }
    }

    /// Get execution statistics for a task
    pub async fn get_task_stats(&self, task_name: &str) -> Option<Vec<TaskResult>> {
        self.execution_stats.read().await.get(task_name).cloned()
    }

    /// Get average execution time for a task
    pub async fn get_average_execution_time(&self, task_name: &str) -> Option<Duration> {
        let stats = self.execution_stats.read().await;
        let task_results = stats.get(task_name)?;
        
        if task_results.is_empty() {
            return None;
        }
        
        let total_duration: Duration = task_results.iter()
            .filter(|r| r.success && !r.cached)
            .map(|r| r.duration)
            .sum();
        
        let successful_count = task_results.iter()
            .filter(|r| r.success && !r.cached)
            .count();
        
        if successful_count == 0 {
            None
        } else {
            Some(total_duration / successful_count as u32)
        }
    }

    /// Clear cache manually
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        let size = cache.len();
        cache.clear();
        info!("Cleared {} cache entries", size);
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        let total_entries = cache.len();
        
        // Count expired entries
        let now = Instant::now();
        let config = self.config.read().await.startup.clone();
        let max_age = Duration::from_secs(config.cache_duration);
        
        let expired_entries = cache.iter()
            .filter(|(_, (timestamp, _))| now.duration_since(*timestamp) >= max_age)
            .count();
        
        (total_entries, expired_entries)
    }

    /// Handle configuration updates
    pub async fn on_config_updated(&self) {
        debug!("Startup optimizer configuration updated");
        
        // Clean cache with new settings
        let config = self.config.read().await.startup.clone();
        self.clean_cache(&config).await;
    }

    /// Create default startup tasks for the TTRPG Assistant
    pub async fn register_default_tasks(&self) {
        let default_tasks = vec![
            StartupTask {
                name: "core_initialization".to_string(),
                priority: 0,
                dependencies: vec![],
                estimated_duration_ms: 100,
                critical: true,
                cache_key: Some("core_init".to_string()),
            },
            StartupTask {
                name: "logging_setup".to_string(),
                priority: 1,
                dependencies: vec![],
                estimated_duration_ms: 50,
                critical: true,
                cache_key: None,
            },
            StartupTask {
                name: "configuration_load".to_string(),
                priority: 2,
                dependencies: vec!["core_initialization".to_string()],
                estimated_duration_ms: 200,
                critical: true,
                cache_key: Some("config_load".to_string()),
            },
            StartupTask {
                name: "native_features_init".to_string(),
                priority: 10,
                dependencies: vec!["core_initialization".to_string()],
                estimated_duration_ms: 300,
                critical: false,
                cache_key: Some("native_features".to_string()),
            },
            StartupTask {
                name: "mcp_bridge_init".to_string(),
                priority: 20,
                dependencies: vec!["configuration_load".to_string()],
                estimated_duration_ms: 500,
                critical: true,
                cache_key: None,
            },
            StartupTask {
                name: "process_manager_init".to_string(),
                priority: 15,
                dependencies: vec!["configuration_load".to_string()],
                estimated_duration_ms: 100,
                critical: true,
                cache_key: None,
            },
            StartupTask {
                name: "data_manager_lazy_load".to_string(),
                priority: 50,
                dependencies: vec!["configuration_load".to_string()],
                estimated_duration_ms: 800,
                critical: false,
                cache_key: Some("data_manager".to_string()),
            },
            StartupTask {
                name: "security_manager_lazy_load".to_string(),
                priority: 60,
                dependencies: vec!["configuration_load".to_string()],
                estimated_duration_ms: 400,
                critical: false,
                cache_key: Some("security_manager".to_string()),
            },
            StartupTask {
                name: "ui_framework_init".to_string(),
                priority: 30,
                dependencies: vec!["native_features_init".to_string()],
                estimated_duration_ms: 600,
                critical: true,
                cache_key: Some("ui_framework".to_string()),
            },
            StartupTask {
                name: "optional_plugins_load".to_string(),
                priority: 100,
                dependencies: vec!["ui_framework_init".to_string()],
                estimated_duration_ms: 1000,
                critical: false,
                cache_key: Some("optional_plugins".to_string()),
            },
        ];
        
        self.register_tasks(default_tasks).await;
        info!("Registered {} default startup tasks", 10);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_dependency_resolution() {
        let config = Arc::new(RwLock::new(PerformanceConfig::default()));
        let optimizer = StartupOptimizer::new(config);
        
        let tasks = vec![
            StartupTask {
                name: "task_a".to_string(),
                priority: 1,
                dependencies: vec!["task_b".to_string()],
                estimated_duration_ms: 100,
                critical: true,
                cache_key: None,
            },
            StartupTask {
                name: "task_b".to_string(),
                priority: 2,
                dependencies: vec![],
                estimated_duration_ms: 50,
                critical: true,
                cache_key: None,
            },
        ];
        
        optimizer.register_tasks(tasks).await;
        let order = optimizer.build_execution_order().await.unwrap();
        
        assert_eq!(order.len(), 2);
        assert_eq!(order[0], vec!["task_b"]);
        assert_eq!(order[1], vec!["task_a"]);
    }
    
    #[tokio::test]
    async fn test_circular_dependency_detection() {
        let config = Arc::new(RwLock::new(PerformanceConfig::default()));
        let optimizer = StartupOptimizer::new(config);
        
        let tasks = vec![
            StartupTask {
                name: "task_a".to_string(),
                priority: 1,
                dependencies: vec!["task_b".to_string()],
                estimated_duration_ms: 100,
                critical: true,
                cache_key: None,
            },
            StartupTask {
                name: "task_b".to_string(),
                priority: 2,
                dependencies: vec!["task_a".to_string()],
                estimated_duration_ms: 50,
                critical: true,
                cache_key: None,
            },
        ];
        
        optimizer.register_tasks(tasks).await;
        let result = optimizer.build_execution_order().await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Circular dependency"));
    }
}