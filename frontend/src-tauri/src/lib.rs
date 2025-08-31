//! MDMAI Desktop Application Backend
//! 
//! This library provides the Rust backend for the MDMAI desktop application built with Tauri.
//! It includes comprehensive data management capabilities with a focus on performance,
//! memory efficiency, and data safety.
//! 
//! ## Features
//! 
//! - **Thread-safe encryption**: AES-256-GCM with Argon2 key derivation
//! - **High-performance caching**: O(1) operations with intelligent eviction
//! - **Streaming file processing**: Memory-efficient handling of large files
//! - **Data integrity verification**: Blake3 and SHA-256 checksums with streaming calculation
//! - **Type-safe database operations**: Proper SQL type handling to prevent data corruption
//! - **Native system integration**: Platform-specific features without memory leaks
//! 
//! ## Architecture
//! 
//! The backend is organized into several key modules:
//! - `data_manager`: Core data management functionality
//! - `native_features`: Platform-specific system integration
//! - Command handlers for Tauri frontend communication

#![warn(missing_docs)]
#![deny(unsafe_code)]

use tauri::{
    api::shell,
    generate_context, generate_handler,
    Builder, Context, Manager, State, Window,
};

// Internal modules
pub mod data_manager;
pub mod data_manager_commands;
pub mod native_features;

#[cfg(test)]
mod tests;

// Re-exports for convenience
pub use data_manager::{
    DataManagerConfig, DataManagerStats, validate_config, init_logging
};
pub use data_manager_commands::DataManagerState;

/// Initialize and configure the Tauri application
pub fn create_app() -> tauri::App {
    // Initialize logging
    if let Err(e) = init_logging() {
        eprintln!("Failed to initialize logging: {}", e);
    }

    // Create default configuration
    let config = DataManagerConfig::default();
    
    // Validate configuration
    if let Err(e) = validate_config(&config) {
        panic!("Invalid configuration: {}", e);
    }

    // Get the tauri config for initializing components that need it
    let tauri_config = generate_context!().config();
    
    // Initialize data manager state
    let data_manager_state = match DataManagerState::new(config, &tauri_config) {
        Ok(state) => state,
        Err(e) => {
            panic!("Failed to initialize data manager: {}", e);
        }
    };

    // Create native features manager
    let native_features_manager = native_features::NativeFeaturesManager::new();

    Builder::default()
        .manage(data_manager_state)
        .manage(native_features_manager)
        .invoke_handler(generate_handler![
            // Data manager commands
            data_manager_commands::initialize_data_manager,
            data_manager_commands::is_data_manager_initialized,
            data_manager_commands::store_encrypted_data,
            data_manager_commands::retrieve_encrypted_data,
            data_manager_commands::delete_encrypted_data,
            data_manager_commands::get_cache_stats,
            data_manager_commands::clear_cache,
            data_manager_commands::create_backup,
            data_manager_commands::restore_backup,
            data_manager_commands::list_backups,
            data_manager_commands::verify_data_integrity,
            // Salt management commands
            data_manager_commands::check_salt_exists,
            data_manager_commands::regenerate_salt,
            data_manager_commands::get_salt_storage_status,
            // Native features commands  
            get_system_status,
            get_file_info,
            send_notification,
            get_system_info,
        ])
        .setup(|app| {
            log::info!("MDMAI Desktop application initialized");
            
            // Perform any additional setup here
            let app_handle = app.handle();
            
            // Initialize native features if needed
            tauri::async_runtime::spawn(async move {
                // Any async initialization can go here
                log::debug!("Async initialization completed");
            });
            
            Ok(())
        })
        .build(generate_context!())
        .expect("Failed to create Tauri application")
}

/// Get system status
#[tauri::command]
async fn get_system_status(
    native_manager: State<'_, native_features::NativeFeaturesManager>,
) -> Result<native_features::SystemStatus, String> {
    native_manager.get_system_status().await
        .map_err(|e| format!("Failed to get system status: {}", e))
}

/// Get file system information
#[tauri::command]
async fn get_file_info(
    file_path: String,
    native_manager: State<'_, native_features::NativeFeaturesManager>,
) -> Result<native_features::FileSystemInfo, String> {
    native_manager.get_file_info(file_path).await
        .map_err(|e| format!("Failed to get file info: {}", e))
}

/// Send system notification
#[tauri::command]
async fn send_notification(
    config: native_features::NotificationConfig,
    native_manager: State<'_, native_features::NativeFeaturesManager>,
) -> Result<(), String> {
    native_manager.send_notification(config).await
        .map_err(|e| format!("Failed to send notification: {}", e))
}

/// Get comprehensive system information
#[tauri::command]
async fn get_system_info(
    native_manager: State<'_, native_features::NativeFeaturesManager>,
) -> Result<std::collections::HashMap<String, serde_json::Value>, String> {
    native_manager.get_system_info().await
        .map_err(|e| format!("Failed to get system info: {}", e))
}

/// Run the Tauri application
pub fn run_app() {
    let app = create_app();
    
    app.run(|_app_handle, event| match event {
        tauri::RunEvent::Updater(updater_event) => {
            match updater_event {
                tauri::UpdaterEvent::UpdateAvailable { body, date, version } => {
                    log::info!("Update available: version {}, date: {:?}", version, date);
                    if let Some(body) = body {
                        log::debug!("Update notes: {}", body);
                    }
                }
                tauri::UpdaterEvent::Pending => {
                    log::info!("Update is pending...");
                }
                tauri::UpdaterEvent::DownloadProgress { chunk_length, content_length } => {
                    let progress = if let Some(total) = content_length {
                        (chunk_length as f64 / total as f64) * 100.0
                    } else {
                        0.0
                    };
                    log::debug!("Download progress: {:.1}%", progress);
                }
                tauri::UpdaterEvent::Downloaded => {
                    log::info!("Update downloaded successfully");
                }
                tauri::UpdaterEvent::Updated => {
                    log::info!("Application updated successfully");
                }
                tauri::UpdaterEvent::AlreadyUpToDate => {
                    log::debug!("Application is already up to date");
                }
                tauri::UpdaterEvent::Error(error) => {
                    log::error!("Update error: {}", error);
                }
            }
        }
        tauri::RunEvent::ExitRequested { api, .. } => {
            log::info!("Application exit requested");
            api.prevent_exit();
            
            // Perform cleanup here
            tauri::async_runtime::block_on(async {
                // Any async cleanup can go here
                log::info!("Cleanup completed");
            });
            
            // Allow the app to exit after cleanup
            api.prevent_exit();
        }
        tauri::RunEvent::WindowEvent { label, event, .. } => {
            match event {
                tauri::WindowEvent::CloseRequested { api, .. } => {
                    log::debug!("Window {} close requested", label);
                    // Could implement custom close handling here
                }
                tauri::WindowEvent::Focused(focused) => {
                    log::debug!("Window {} focused: {}", label, focused);
                }
                tauri::WindowEvent::Resized(size) => {
                    log::debug!("Window {} resized to: {}x{}", label, size.width, size.height);
                }
                _ => {}
            }
        }
        _ => {}
    });
}