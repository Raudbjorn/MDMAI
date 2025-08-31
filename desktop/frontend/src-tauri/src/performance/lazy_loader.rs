/*!
 * Lazy Loading Module
 * 
 * Provides intelligent lazy loading for application components, reducing startup time
 * and memory usage through on-demand initialization.
 */

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::{RwLock, Mutex, oneshot};
use serde::{Deserialize, Serialize};
use log::{info, debug, warn, error};
use super::{PerformanceConfig, LazyLoadingConfig};

/// Component loading state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadingState {
    Unloaded,
    Loading,
    Loaded,
    Failed(String),
}

/// Lazy component definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LazyComponent {
    pub name: String,
    pub priority: u8,
    pub dependencies: Vec<String>,
    pub estimated_load_time_ms: u64,
    pub memory_usage_estimate_mb: u32,
    pub preload_condition: PreloadCondition,
    pub mandatory: bool,
}

/// Conditions for preloading components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreloadCondition {
    Never,
    OnStartup,
    OnFirstAccess,
    OnDelay(u64), // milliseconds after startup
    OnUserAction(String),
    OnMemoryAvailable(u32), // MB of available memory
}

/// Component loading result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadingResult {
    pub component_name: String,
    pub success: bool,
    pub load_time: Duration,
    pub memory_used_mb: u32,
    pub error: Option<String>,
    pub cached: bool,
}

/// Component loading context
pub struct LoadingContext {
    pub component: LazyComponent,
    pub start_time: Instant,
    pub triggered_by: String,
}

/// Lazy component wrapper with initialization logic
pub trait ComponentFactory: Send + Sync {
    type Component: Send + Sync;
    type Error: std::fmt::Display + Send + Sync;

    fn create(&self) -> impl std::future::Future<Output = Result<Self::Component, Self::Error>> + Send;
    fn cleanup(&self, component: Self::Component) -> impl std::future::Future<Output = ()> + Send;
}

/// Generic lazy component holder
pub struct LazyComponentHolder<T> {
    factory: Arc<dyn ComponentFactory<Component = T, Error = String> + Send + Sync>,
    instance: Arc<RwLock<Option<Arc<T>>>>,
    state: Arc<RwLock<LoadingState>>,
    loading_mutex: Arc<Mutex<()>>,
}

impl<T: Send + Sync + 'static> LazyComponentHolder<T> {
    pub fn new(factory: Arc<dyn ComponentFactory<Component = T, Error = String> + Send + Sync>) -> Self {
        Self {
            factory,
            instance: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(LoadingState::Unloaded)),
            loading_mutex: Arc::new(Mutex::new(())),
        }
    }

    /// Get the component instance, loading it if necessary
    pub async fn get(&self) -> Result<Arc<T>, String> {
        // Quick check if already loaded
        {
            let state = self.state.read().await;
            if *state == LoadingState::Loaded {
                let instance = self.instance.read().await;
                if let Some(ref component) = *instance {
                    return Ok(Arc::clone(component));
                }
            }
        }

        // Ensure only one loading attempt at a time
        let _guard = self.loading_mutex.lock().await;

        // Double-check after acquiring the lock
        {
            let state = self.state.read().await;
            if *state == LoadingState::Loaded {
                let instance = self.instance.read().await;
                if let Some(ref component) = *instance {
                    return Ok(Arc::clone(component));
                }
            }
            
            if *state == LoadingState::Loading {
                return Err("Component is already being loaded".to_string());
            }
            
            if let LoadingState::Failed(ref error) = *state {
                return Err(format!("Component failed to load previously: {}", error));
            }
        }

        // Set loading state
        *self.state.write().await = LoadingState::Loading;

        // Load the component
        match self.factory.create().await {
            Ok(component) => {
                let arc_component = Arc::new(component);
                *self.instance.write().await = Some(Arc::clone(&arc_component));
                *self.state.write().await = LoadingState::Loaded;
                Ok(arc_component)
            }
            Err(e) => {
                *self.state.write().await = LoadingState::Failed(e.to_string());
                Err(e.to_string())
            }
        }
    }

    /// Check if component is loaded
    pub async fn is_loaded(&self) -> bool {
        *self.state.read().await == LoadingState::Loaded
    }

    /// Get current loading state
    pub async fn get_state(&self) -> LoadingState {
        self.state.read().await.clone()
    }

    /// Unload the component
    pub async fn unload(&self) {
        let _guard = self.loading_mutex.lock().await;

        if let Some(component) = self.instance.write().await.take() {
            // Try to unwrap Arc, but only if we're the sole owner
            match Arc::try_unwrap(component) {
                Ok(component) => self.factory.cleanup(component).await,
                Err(arc_component) => {
                    // There are other references, so we can't cleanup immediately
                    // Put it back and log a warning
                    *self.instance.write().await = Some(arc_component);
                    log::warn!("Cannot unload component: other references exist");
                    return;
                }
            }
        }

        *self.state.write().await = LoadingState::Unloaded;
    }
}

/// Main lazy loader managing all lazy components
pub struct LazyLoader {
    config: Arc<RwLock<PerformanceConfig>>,
    components: Arc<RwLock<HashMap<String, LazyComponent>>>,
    holders: Arc<RwLock<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>>,
    loading_queue: Arc<Mutex<Vec<String>>>,
    stats: Arc<RwLock<LazyLoaderStats>>,
    preload_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LazyLoaderStats {
    pub components_registered: usize,
    pub components_loaded: usize,
    pub components_failed: usize,
    pub total_load_time_ms: u64,
    pub average_load_time_ms: f64,
    pub memory_saved_mb: u32,
    pub preloads_triggered: u64,
    pub on_demand_loads: u64,
}

impl LazyLoader {
    pub fn new(config: Arc<RwLock<PerformanceConfig>>) -> Self {
        Self {
            config,
            components: Arc::new(RwLock::new(HashMap::new())),
            holders: Arc::new(RwLock::new(HashMap::new())),
            loading_queue: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(RwLock::new(LazyLoaderStats {
                components_registered: 0,
                components_loaded: 0,
                components_failed: 0,
                total_load_time_ms: 0,
                average_load_time_ms: 0.0,
                memory_saved_mb: 0,
                preloads_triggered: 0,
                on_demand_loads: 0,
            })),
            preload_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize the lazy loader
    pub async fn initialize(&self) -> Result<(), String> {
        info!("Initializing Lazy Loader");

        // Register default components
        self.register_default_components().await;

        // Start preloading background task
        self.start_preload_task().await;

        info!("Lazy Loader initialized with {} components", 
              self.components.read().await.len());

        Ok(())
    }

    /// Register a lazy component
    pub async fn register_component(&self, component: LazyComponent) {
        debug!("Registering lazy component: {}", component.name);
        
        let mut components = self.components.write().await;
        components.insert(component.name.clone(), component);

        let mut stats = self.stats.write().await;
        stats.components_registered = components.len();
        stats.memory_saved_mb += components.values()
            .filter(|c| c.preload_condition != PreloadCondition::OnStartup)
            .map(|c| c.memory_usage_estimate_mb)
            .sum::<u32>();
    }

    /// Register multiple components
    pub async fn register_components(&self, components: Vec<LazyComponent>) {
        for component in components {
            self.register_component(component).await;
        }
    }

    /// Register a component with its factory
    pub async fn register_component_with_factory<T, F>(
        &self, 
        component: LazyComponent, 
        factory: F
    ) 
    where
        T: Send + Sync + 'static,
        F: ComponentFactory<Component = T, Error = String> + Send + Sync + 'static,
    {
        let holder = LazyComponentHolder::new(Arc::new(factory));
        
        {
            let mut holders = self.holders.write().await;
            holders.insert(component.name.clone(), Box::new(holder));
        }
        
        self.register_component(component).await;
    }

    /// Load a component on demand
    pub async fn load_component(&self, name: &str, triggered_by: &str) -> Result<LoadingResult, String> {
        let start_time = Instant::now();
        debug!("Loading component '{}' triggered by '{}'", name, triggered_by);

        let component = {
            let components = self.components.read().await;
            components.get(name).cloned()
                .ok_or_else(|| format!("Component '{}' not registered", name))?
        };

        // Check dependencies
        for dep in &component.dependencies {
            if !self.is_component_loaded(dep).await {
                debug!("Loading dependency '{}' for '{}'", dep, name);
                self.load_component(dep, &format!("dependency_of_{}", name)).await?;
            }
        }

        // Simulate component loading (in reality, this would call the actual factory)
        let load_result = self.simulate_component_load(&component).await;

        let load_time = start_time.elapsed();
        let result = LoadingResult {
            component_name: name.to_string(),
            success: load_result.is_ok(),
            load_time,
            memory_used_mb: if load_result.is_ok() { component.memory_usage_estimate_mb } else { 0 },
            error: load_result.err(),
            cached: false,
        };

        // Update statistics
        self.update_loading_stats(&result, triggered_by).await;

        if result.success {
            debug!("Component '{}' loaded successfully in {:?}", name, load_time);
            Ok(result)
        } else {
            error!("Failed to load component '{}': {}", name, 
                   result.error.as_deref().unwrap_or("Unknown error"));
            Err(result.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }

    /// Check if a component is loaded
    pub async fn is_component_loaded(&self, name: &str) -> bool {
        // In a real implementation, you'd check the component holder
        // For this example, we'll track loaded components in stats
        let components = self.components.read().await;
        components.contains_key(name) // Simplified check
    }

    /// Get component loading state
    pub async fn get_component_state(&self, name: &str) -> Option<LoadingState> {
        let holders = self.holders.read().await;
        if holders.contains_key(name) {
            Some(LoadingState::Unloaded) // Simplified - would check actual state
        } else {
            None
        }
    }

    /// Preload components based on conditions
    pub async fn preload_components(&self, trigger: &str) -> Vec<LoadingResult> {
        let mut results = Vec::new();
        let components = self.components.read().await.clone();

        for (name, component) in components {
            let should_preload = match &component.preload_condition {
                PreloadCondition::OnStartup if trigger == "startup" => true,
                PreloadCondition::OnUserAction(action) if trigger == action => true,
                PreloadCondition::OnDelay(delay_ms) => {
                    // Check if enough time has passed since startup
                    // This is simplified - you'd track startup time
                    trigger == "delay_check"
                }
                PreloadCondition::OnMemoryAvailable(required_mb) => {
                    // Check if enough memory is available
                    self.check_memory_available(*required_mb).await
                }
                _ => false,
            };

            if should_preload && !self.is_component_loaded(&name).await {
                debug!("Preloading component '{}' due to trigger '{}'", name, trigger);
                match self.load_component(&name, &format!("preload_{}", trigger)).await {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        results.push(LoadingResult {
                            component_name: name,
                            success: false,
                            load_time: Duration::ZERO,
                            memory_used_mb: 0,
                            error: Some(e),
                            cached: false,
                        });
                    }
                }
            }
        }

        if !results.is_empty() {
            let mut stats = self.stats.write().await;
            stats.preloads_triggered += results.len() as u64;
            info!("Preloaded {} components due to trigger '{}'", results.len(), trigger);
        }

        results
    }

    /// Start background preloading task
    async fn start_preload_task(&self) {
        let config = self.config.read().await;
        if !config.lazy_loading.background_preload {
            return;
        }

        let loader = self.clone();
        let preload_order = config.lazy_loading.preload_order.clone();
        let preload_threshold = config.lazy_loading.preload_threshold_ms;

        let preload_task = tokio::spawn(async move {
            // Wait for initial startup to complete
            tokio::time::sleep(Duration::from_millis(preload_threshold)).await;

            // Preload components in order
            for component_name in preload_order {
                if !loader.is_component_loaded(&component_name).await {
                    debug!("Background preloading component: {}", component_name);
                    if let Err(e) = loader.load_component(&component_name, "background_preload").await {
                        warn!("Failed to preload component '{}': {}", component_name, e);
                    }
                    
                    // Small delay between preloads to avoid overwhelming the system
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }

            info!("Background preloading completed");
        });

        *self.preload_handle.write().await = Some(preload_task);
    }

    /// Simulate component loading (replace with actual loading logic)
    async fn simulate_component_load(&self, component: &LazyComponent) -> Result<(), String> {
        // Simulate loading time
        let load_time = Duration::from_millis(component.estimated_load_time_ms);
        tokio::time::sleep(load_time).await;

        // Simulate occasional failures for non-mandatory components
        if !component.mandatory && component.name.contains("optional") && rand::random::<f32>() < 0.1 {
            return Err("Simulated loading failure".to_string());
        }

        Ok(())
    }

    /// Check if sufficient memory is available
    async fn check_memory_available(&self, required_mb: u32) -> bool {
        // Use sysinfo to check available memory
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        
        let available_mb = system.available_memory() / 1024 / 1024;
        available_mb >= required_mb as u64
    }

    /// Update loading statistics
    async fn update_loading_stats(&self, result: &LoadingResult, triggered_by: &str) {
        let mut stats = self.stats.write().await;

        if result.success {
            stats.components_loaded += 1;
        } else {
            stats.components_failed += 1;
        }

        stats.total_load_time_ms += result.load_time.as_millis() as u64;

        let total_loads = stats.components_loaded + stats.components_failed;
        if total_loads > 0 {
            stats.average_load_time_ms = stats.total_load_time_ms as f64 / total_loads as f64;
        }

        if triggered_by.contains("demand") || triggered_by.contains("user") {
            stats.on_demand_loads += 1;
        }
    }

    /// Register default application components
    async fn register_default_components(&self) {
        let default_components = vec![
            LazyComponent {
                name: "data_manager".to_string(),
                priority: 10,
                dependencies: vec!["core".to_string()],
                estimated_load_time_ms: 800,
                memory_usage_estimate_mb: 25,
                preload_condition: PreloadCondition::OnDelay(5000),
                mandatory: false,
            },
            LazyComponent {
                name: "security_manager".to_string(),
                priority: 15,
                dependencies: vec!["core".to_string()],
                estimated_load_time_ms: 400,
                memory_usage_estimate_mb: 15,
                preload_condition: PreloadCondition::OnFirstAccess,
                mandatory: false,
            },
            LazyComponent {
                name: "advanced_features".to_string(),
                priority: 50,
                dependencies: vec!["data_manager".to_string()],
                estimated_load_time_ms: 1200,
                memory_usage_estimate_mb: 35,
                preload_condition: PreloadCondition::OnUserAction("advanced_mode".to_string()),
                mandatory: false,
            },
            LazyComponent {
                name: "backup_system".to_string(),
                priority: 30,
                dependencies: vec!["data_manager".to_string()],
                estimated_load_time_ms: 600,
                memory_usage_estimate_mb: 20,
                preload_condition: PreloadCondition::OnMemoryAvailable(100),
                mandatory: false,
            },
            LazyComponent {
                name: "plugin_system".to_string(),
                priority: 60,
                dependencies: vec!["core".to_string()],
                estimated_load_time_ms: 1000,
                memory_usage_estimate_mb: 30,
                preload_condition: PreloadCondition::Never,
                mandatory: false,
            },
        ];

        self.register_components(default_components).await;
        info!("Registered {} default lazy components", 5);
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> LazyLoaderStats {
        self.stats.read().await.clone()
    }

    /// Get detailed component information
    pub async fn get_component_info(&self) -> HashMap<String, ComponentInfo> {
        let components = self.components.read().await;
        let mut info = HashMap::new();

        for (name, component) in components.iter() {
            let state = self.get_component_state(name).await.unwrap_or(LoadingState::Unloaded);
            
            info.insert(name.clone(), ComponentInfo {
                component: component.clone(),
                state,
                loaded_at: None, // Would track loading timestamp
                load_time: None, // Would track actual load time
            });
        }

        info
    }

    /// Handle configuration updates
    pub async fn on_config_updated(&self) {
        debug!("Lazy loader configuration updated");
        
        // Restart preload task if configuration changed
        let config = self.config.read().await;
        if config.lazy_loading.background_preload {
            // Stop existing preload task
            if let Some(handle) = self.preload_handle.write().await.take() {
                handle.abort();
            }
            
            // Start new preload task
            self.start_preload_task().await;
        }
    }

    /// Shutdown lazy loader
    pub async fn shutdown(&self) -> Result<(), String> {
        info!("Shutting down Lazy Loader");

        // Stop preload task
        if let Some(handle) = self.preload_handle.write().await.take() {
            handle.abort();
        }

        // Unload all components
        let components: Vec<String> = self.components.read().await.keys().cloned().collect();
        for name in components {
            // In a real implementation, you'd call unload on each component holder
            debug!("Unloading component: {}", name);
        }

        info!("Lazy Loader shutdown complete");
        Ok(())
    }
}

impl Clone for LazyLoader {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            components: self.components.clone(),
            holders: self.holders.clone(),
            loading_queue: self.loading_queue.clone(),
            stats: self.stats.clone(),
            preload_handle: self.preload_handle.clone(),
        }
    }
}

/// Component information for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentInfo {
    pub component: LazyComponent,
    pub state: LoadingState,
    pub loaded_at: Option<chrono::DateTime<chrono::Utc>>,
    pub load_time: Option<Duration>,
}

/// Example component factory implementations
pub struct DataManagerFactory;

impl ComponentFactory for DataManagerFactory {
    type Component = String; // Simplified - would be actual data manager
    type Error = String;

    async fn create(&self) -> Result<Self::Component, Self::Error> {
        tokio::time::sleep(Duration::from_millis(800)).await;
        Ok("DataManager initialized".to_string())
    }

    async fn cleanup(&self, _component: Self::Component) {
        // Cleanup logic here
    }
}

pub struct SecurityManagerFactory;

impl ComponentFactory for SecurityManagerFactory {
    type Component = String; // Simplified - would be actual security manager
    type Error = String;

    async fn create(&self) -> Result<Self::Component, Self::Error> {
        tokio::time::sleep(Duration::from_millis(400)).await;
        Ok("SecurityManager initialized".to_string())
    }

    async fn cleanup(&self, _component: Self::Component) {
        // Cleanup logic here
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lazy_component_registration() {
        let config = Arc::new(RwLock::new(super::super::PerformanceConfig::default()));
        let loader = LazyLoader::new(config);
        
        let component = LazyComponent {
            name: "test_component".to_string(),
            priority: 10,
            dependencies: vec![],
            estimated_load_time_ms: 100,
            memory_usage_estimate_mb: 10,
            preload_condition: PreloadCondition::OnFirstAccess,
            mandatory: false,
        };
        
        loader.register_component(component).await;
        
        let stats = loader.get_stats().await;
        assert_eq!(stats.components_registered, 1);
    }

    #[tokio::test]
    async fn test_component_loading() {
        let config = Arc::new(RwLock::new(super::super::PerformanceConfig::default()));
        let loader = LazyLoader::new(config);
        
        let component = LazyComponent {
            name: "test_component".to_string(),
            priority: 10,
            dependencies: vec![],
            estimated_load_time_ms: 50,
            memory_usage_estimate_mb: 10,
            preload_condition: PreloadCondition::OnFirstAccess,
            mandatory: true,
        };
        
        loader.register_component(component).await;
        
        let result = loader.load_component("test_component", "test").await;
        assert!(result.is_ok());
        
        let loading_result = result.unwrap();
        assert!(loading_result.success);
        assert_eq!(loading_result.component_name, "test_component");
    }

    #[tokio::test]
    async fn test_dependency_loading() {
        let config = Arc::new(RwLock::new(super::super::PerformanceConfig::default()));
        let loader = LazyLoader::new(config);
        
        let components = vec![
            LazyComponent {
                name: "base_component".to_string(),
                priority: 5,
                dependencies: vec![],
                estimated_load_time_ms: 50,
                memory_usage_estimate_mb: 5,
                preload_condition: PreloadCondition::OnFirstAccess,
                mandatory: true,
            },
            LazyComponent {
                name: "dependent_component".to_string(),
                priority: 10,
                dependencies: vec!["base_component".to_string()],
                estimated_load_time_ms: 100,
                memory_usage_estimate_mb: 10,
                preload_condition: PreloadCondition::OnFirstAccess,
                mandatory: true,
            },
        ];
        
        loader.register_components(components).await;
        
        // Loading dependent should also load base
        let result = loader.load_component("dependent_component", "test").await;
        assert!(result.is_ok());
        
        let stats = loader.get_stats().await;
        assert_eq!(stats.components_loaded, 2); // Both base and dependent should be loaded
    }
}