#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

// Module declarations
mod ipc;
mod mcp_bridge_v2;
mod process_manager;

use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::Manager;
use log::{info, error};
use process_manager::{ProcessManagerState, ProcessConfig};

// Use the enhanced MCP bridge
use mcp_bridge_v2::MCPBridge;

// Application state for thread-safe operation
#[derive(Clone)]
pub struct AppState {
    mcp_bridge: Arc<Mutex<Option<MCPBridge>>>,
    process_manager: Arc<process_manager::ProcessManager>,
}

impl AppState {
    pub fn new() -> Self {
        // Configure process manager with desktop-optimized settings
        let process_config = ProcessConfig {
            max_restart_attempts: 5,
            restart_delay_ms: 2000,
            health_check_interval_ms: 15000,
            health_check_timeout_ms: 5000,
            max_health_check_failures: 3,
            resource_monitor_interval_ms: 10000,
            cpu_alert_threshold: 85.0,
            memory_alert_threshold: 1000.0, // 1GB
            auto_restart_on_crash: true,
            graceful_shutdown_timeout_ms: 10000,
        };
        
        let process_manager = Arc::new(
            process_manager::ProcessManager::with_config(process_config)
        );
        
        AppState {
            mcp_bridge: Arc::new(Mutex::new(None)),
            process_manager,
        }
    }
    
    /// Get or create MCP bridge
    pub async fn get_or_create_bridge(&self) -> Arc<MCPBridge> {
        let mut bridge_guard = self.mcp_bridge.lock().await;
        
        if bridge_guard.is_none() {
            info!("Creating new MCP bridge instance");
            *bridge_guard = Some(MCPBridge::new(self.process_manager.clone()));
        }
        
        // Clone the Arc to return, safe because MCPBridge is thread-safe
        Arc::new(
            bridge_guard
                .as_ref()
                .expect("Bridge should exist after creation")
                .clone()
        )
    }
    
    /// Safely shutdown all services
    pub async fn shutdown(&self) -> Result<(), String> {
        info!("Shutting down application services");
        
        // Stop MCP bridge if running
        let bridge_guard = self.mcp_bridge.lock().await;
        if let Some(bridge) = bridge_guard.as_ref() {
            bridge.stop().await?;
        }
        drop(bridge_guard);
        
        info!("Application shutdown complete");
        Ok(())
    }
}

// Enhanced Tauri command handlers with proper error handling and thread safety

#[tauri::command]
async fn start_mcp_backend(
    app_state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    info!("Starting MCP backend");
    
    let bridge = app_state.get_or_create_bridge().await;
    
    // Check if already healthy
    if bridge.is_healthy().await {
        info!("MCP backend already running and healthy");
        return Ok(());
    }
    
    // Start the bridge
    bridge.start(&app_handle).await?;
    
    info!("MCP backend started successfully");
    Ok(())
}

#[tauri::command]
async fn stop_mcp_backend(
    app_state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    info!("Stopping MCP backend");
    
    let bridge_guard = app_state.mcp_bridge.lock().await;
    if let Some(bridge) = bridge_guard.as_ref() {
        bridge.stop().await?;
        info!("MCP backend stopped successfully");
    } else {
        info!("MCP backend was not running");
    }
    
    Ok(())
}

#[tauri::command]
async fn restart_mcp_backend(
    app_state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    info!("Restarting MCP backend");
    
    let bridge = app_state.get_or_create_bridge().await;
    bridge.restart().await?;
    
    info!("MCP backend restarted successfully");
    Ok(())
}

#[tauri::command]
async fn mcp_call(
    app_state: tauri::State<'_, AppState>,
    method: String,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let bridge_guard = app_state.mcp_bridge.lock().await;
    
    if let Some(bridge) = bridge_guard.as_ref() {
        bridge.call(&method, params).await
    } else {
        Err("MCP backend not initialized".to_string())
    }
}

#[tauri::command]
async fn mcp_call_streaming(
    app_state: tauri::State<'_, AppState>,
    method: String,
    params: serde_json::Value,
    timeout_ms: Option<u64>,
) -> Result<serde_json::Value, String> {
    let bridge_guard = app_state.mcp_bridge.lock().await;
    
    if let Some(bridge) = bridge_guard.as_ref() {
        let timeout = timeout_ms.map(std::time::Duration::from_millis);
        bridge.call_streaming(&method, params, timeout).await
    } else {
        Err("MCP backend not initialized".to_string())
    }
}

#[tauri::command]
async fn check_mcp_health(
    app_state: tauri::State<'_, AppState>,
) -> Result<bool, String> {
    let bridge_guard = app_state.mcp_bridge.lock().await;
    
    if let Some(bridge) = bridge_guard.as_ref() {
        Ok(bridge.is_healthy().await)
    } else {
        Ok(false)
    }
}

#[tauri::command]
async fn get_mcp_metrics(
    app_state: tauri::State<'_, AppState>,
) -> Result<ipc::PerformanceMetrics, String> {
    let bridge_guard = app_state.mcp_bridge.lock().await;
    
    if let Some(bridge) = bridge_guard.as_ref() {
        Ok(bridge.get_metrics().await)
    } else {
        Err("MCP backend not initialized".to_string())
    }
}

#[tauri::command]
async fn reset_mcp_metrics(
    app_state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let bridge_guard = app_state.mcp_bridge.lock().await;
    
    if let Some(bridge) = bridge_guard.as_ref() {
        bridge.reset_metrics().await;
        Ok(())
    } else {
        Err("MCP backend not initialized".to_string())
    }
}

#[tauri::command]
async fn cancel_mcp_request(
    app_state: tauri::State<'_, AppState>,
    request_id: u64,
) -> Result<bool, String> {
    let bridge_guard = app_state.mcp_bridge.lock().await;
    
    if let Some(bridge) = bridge_guard.as_ref() {
        Ok(bridge.cancel_request(request_id).await)
    } else {
        Err("MCP backend not initialized".to_string())
    }
}

#[tauri::command]
async fn cancel_all_mcp_requests(
    app_state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let bridge_guard = app_state.mcp_bridge.lock().await;
    
    if let Some(bridge) = bridge_guard.as_ref() {
        bridge.cancel_all_requests().await;
        Ok(())
    } else {
        Err("MCP backend not initialized".to_string())
    }
}

#[tauri::command]
async fn update_ipc_config(
    app_state: tauri::State<'_, AppState>,
    max_concurrent_requests: Option<usize>,
    max_queue_size: Option<usize>,
    default_timeout_ms: Option<u64>,
) -> Result<(), String> {
    let bridge_guard = app_state.mcp_bridge.lock().await;
    
    if let Some(bridge) = bridge_guard.as_ref() {
        let mut new_config = ipc::QueueConfig::default();
        
        if let Some(max_concurrent) = max_concurrent_requests {
            new_config.max_concurrent_requests = max_concurrent;
        }
        if let Some(max_queue) = max_queue_size {
            new_config.max_queue_size = max_queue;
        }
        if let Some(timeout) = default_timeout_ms {
            new_config.default_timeout_ms = timeout;
        }
        
        bridge.update_ipc_config(new_config).await;
        info!("IPC configuration updated");
        Ok(())
    } else {
        Err("MCP backend not initialized".to_string())
    }
}

// Process management commands
#[tauri::command]
async fn get_process_stats(
    app_state: tauri::State<'_, AppState>,
) -> Result<process_manager::ProcessStats, String> {
    Ok(app_state.process_manager.get_stats().await)
}

#[tauri::command]
async fn get_process_events(
    app_state: tauri::State<'_, AppState>,
    limit: Option<usize>,
) -> Result<Vec<process_manager::ProcessEvent>, String> {
    Ok(app_state.process_manager.get_recent_events(limit.unwrap_or(100)).await)
}

#[tauri::command]
async fn update_process_config(
    app_state: tauri::State<'_, AppState>,
    config: process_manager::ProcessConfig,
) -> Result<(), String> {
    app_state.process_manager.update_config(config).await;
    Ok(())
}

#[tauri::command]
async fn reset_process_restart_count(
    app_state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    app_state.process_manager.reset_restart_count().await;
    Ok(())
}

#[tauri::command]
async fn clear_process_events(
    app_state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    app_state.process_manager.clear_events().await;
    Ok(())
}

// Health check endpoint for monitoring
#[tauri::command]
async fn get_application_health(
    app_state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let bridge_guard = app_state.mcp_bridge.lock().await;
    let mcp_healthy = if let Some(bridge) = bridge_guard.as_ref() {
        bridge.is_healthy().await
    } else {
        false
    };
    drop(bridge_guard);
    
    let process_stats = app_state.process_manager.get_stats().await;
    let metrics = if let Some(bridge) = app_state.mcp_bridge.lock().await.as_ref() {
        Some(bridge.get_metrics().await)
    } else {
        None
    };
    
    Ok(serde_json::json!({
        "mcp_healthy": mcp_healthy,
        "process_state": process_stats.state,
        "process_health": process_stats.health,
        "active_requests": metrics.as_ref().map(|m| m.active_requests).unwrap_or(0),
        "queued_requests": metrics.as_ref().map(|m| m.queued_requests).unwrap_or(0),
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }))
}

fn main() {
    // Initialize logging with appropriate level
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    
    info!("Starting MDMAI Desktop Application");
    
    // Create application state
    let app_state = AppState::new();
    
    tauri::Builder::default()
        // Add required plugins
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        
        // Manage application state
        .manage(app_state.clone())
        
        // Register command handlers
        .invoke_handler(tauri::generate_handler![
            // MCP Bridge commands
            start_mcp_backend,
            stop_mcp_backend,
            restart_mcp_backend,
            mcp_call,
            mcp_call_streaming,
            check_mcp_health,
            get_mcp_metrics,
            reset_mcp_metrics,
            cancel_mcp_request,
            cancel_all_mcp_requests,
            update_ipc_config,
            
            // Process management commands
            get_process_stats,
            get_process_events,
            update_process_config,
            reset_process_restart_count,
            clear_process_events,
            
            // Health monitoring
            get_application_health,
        ])
        
        .setup(|app| {
            info!("Application setup complete");
            Ok(())
        })
        
        // Handle window events for cleanup
        .on_window_event(|window, event| {
            use tauri::WindowEvent;
            match event {
                WindowEvent::CloseRequested { .. } => {
                    let app_state = window.state::<AppState>();
                    
                    // Perform graceful shutdown
                    tauri::async_runtime::block_on(async {
                        if let Err(e) = app_state.shutdown().await {
                            error!("Error during shutdown: {}", e);
                        }
                    });
                }
                WindowEvent::Destroyed => {
                    info!("Window destroyed, application exiting");
                }
                _ => {}
            }
        })
        
        // Handle app events
        .on_menu_event(|window, event| {
            match event.menu_item_id() {
                "quit" => {
                    let app_state = window.state::<AppState>();
                    
                    tauri::async_runtime::spawn(async move {
                        if let Err(e) = app_state.shutdown().await {
                            error!("Error during menu quit: {}", e);
                        }
                        std::process::exit(0);
                    });
                }
                _ => {}
            }
        })
        
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}

// Ensure proper cleanup on panic
fn setup_panic_handler() {
    std::panic::set_hook(Box::new(|panic_info| {
        error!("Application panic: {:?}", panic_info);
        
        // Attempt graceful shutdown
        // Note: This is a last resort and may not work reliably
        std::process::exit(1);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_app_state_creation() {
        let state = AppState::new();
        
        // Should be able to get process stats
        let stats = state.process_manager.get_stats().await;
        assert_eq!(stats.state, process_manager::ProcessState::Stopped);
        
        // Should be able to create bridge
        let bridge = state.get_or_create_bridge().await;
        assert!(!bridge.is_healthy().await); // Should not be healthy initially
    }
    
    #[tokio::test]
    async fn test_graceful_shutdown() {
        let state = AppState::new();
        
        // Should be able to shutdown even without starting anything
        let result = state.shutdown().await;
        assert!(result.is_ok());
    }
}