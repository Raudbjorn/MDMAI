#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod mcp_bridge;
mod process_manager;
mod native_features;
mod error_handling;
mod data_manager;
mod data_manager_commands;

use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::{Manager, Emitter};
use process_manager::ProcessManagerState;
use native_features::NativeFeaturesState;
use data_manager_commands::DataManagerStateWrapper;

fn main() {
    // Initialize logging
    env_logger::init();
    
    tauri::Builder::default()
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(Arc::new(Mutex::new(None::<mcp_bridge::MCPBridge>)))
        .manage(ProcessManagerState::new())
        .manage(NativeFeaturesState::new())
        .manage(DataManagerStateWrapper::new())
        .invoke_handler(tauri::generate_handler![
            mcp_bridge::start_mcp_backend,
            mcp_bridge::stop_mcp_backend,
            mcp_bridge::restart_mcp_backend,
            mcp_bridge::mcp_call,
            mcp_bridge::check_mcp_health,
            process_manager::get_process_stats,
            process_manager::get_process_events,
            process_manager::update_process_config,
            process_manager::reset_process_restart_count,
            process_manager::clear_process_events,
            native_features::show_native_file_dialog,
            native_features::show_save_dialog,
            native_features::send_native_notification,
            native_features::handle_drag_drop_event,
            native_features::update_tray_status,
            // Data Manager Commands
            data_manager_commands::initialize_data_manager,
            data_manager_commands::initialize_data_manager_with_password,
            data_manager_commands::get_data_manager_config,
            data_manager_commands::update_data_manager_config,
            data_manager_commands::create_campaign,
            data_manager_commands::get_campaign,
            data_manager_commands::list_campaigns,
            data_manager_commands::update_campaign,
            data_manager_commands::delete_campaign,
            data_manager_commands::create_character,
            data_manager_commands::create_backup,
            data_manager_commands::list_backups,
            data_manager_commands::restore_backup,
            data_manager_commands::delete_backup,
            data_manager_commands::check_data_integrity,
            data_manager_commands::repair_data_integrity,
            data_manager_commands::store_file,
            data_manager_commands::retrieve_file,
            data_manager_commands::export_file,
            data_manager_commands::delete_stored_file,
            data_manager_commands::get_storage_stats,
            data_manager_commands::find_duplicate_files,
            data_manager_commands::optimize_storage,
            data_manager_commands::get_cache_stats,
            data_manager_commands::clear_cache,
            data_manager_commands::cleanup_expired_cache,
            data_manager_commands::get_migration_status,
            data_manager_commands::run_pending_migrations,
            data_manager_commands::get_database_stats,
            data_manager_commands::perform_database_maintenance,
            data_manager_commands::shutdown_data_manager,
        ])
        .setup(|app| {
            // Initialize native features
            let native_features = app.state::<NativeFeaturesState>();
            let native_features_clone = native_features.inner().clone();
            let app_handle = app.handle().clone();
            
            tauri::async_runtime::block_on(async {
                if let Err(e) = native_features_clone.initialize(&app_handle).await {
                    log::error!("Failed to initialize native features: {}", e);
                }
            });
            
            Ok(())
        })
        .on_window_event(|window, event| {
            use tauri::WindowEvent;
            match event {
                WindowEvent::CloseRequested { .. } => {
                    // Clean shutdown on window close
                    let state: tauri::State<Arc<Mutex<Option<mcp_bridge::MCPBridge>>>> = window.state();
                    let state_clone = state.inner().clone();
                    
                    // Stop MCP backend gracefully
                    tauri::async_runtime::block_on(async {
                        if let Some(bridge) = state_clone.lock().await.as_ref() {
                            let _ = bridge.stop().await;
                        }
                    });
                },
                WindowEvent::DragDrop(drag_event) => {
                    use tauri::DragDropEvent;
                    match drag_event {
                        DragDropEvent::Drop { paths, position } => {
                            let files: Vec<String> = paths.iter()
                                .map(|p| p.to_string_lossy().to_string())
                                .collect();
                            
                            let drop_event = native_features::DragDropEvent {
                                event_type: "drop".to_string(),
                                files,
                                position: Some((position.x, position.y)),
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs(),
                            };
                            
                            // Emit drag drop event to frontend
                            if let Err(e) = window.emit("drag-drop", &drop_event) {
                                log::error!("Failed to emit drag-drop event: {}", e);
                            }
                        },
                        DragDropEvent::Enter { paths, position } => {
                            let files: Vec<String> = paths.iter()
                                .map(|p| p.to_string_lossy().to_string())
                                .collect();
                            
                            let drop_event = native_features::DragDropEvent {
                                event_type: "enter".to_string(),
                                files,
                                position: Some((position.x, position.y)),
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs(),
                            };
                            
                            if let Err(e) = window.emit("drag-enter", &drop_event) {
                                log::error!("Failed to emit drag-enter event: {}", e);
                            }
                        },
                        DragDropEvent::Over { position } => {
                            let drop_event = native_features::DragDropEvent {
                                event_type: "over".to_string(),
                                files: vec![],
                                position: Some((position.x, position.y)),
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs(),
                            };
                            
                            if let Err(e) = window.emit("drag-over", &drop_event) {
                                log::error!("Failed to emit drag-over event: {}", e);
                            }
                        },
                        DragDropEvent::Leave => {
                            let drop_event = native_features::DragDropEvent {
                                event_type: "leave".to_string(),
                                files: vec![],
                                position: None,
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs(),
                            };
                            
                            if let Err(e) = window.emit("drag-leave", &drop_event) {
                                log::error!("Failed to emit drag-leave event: {}", e);
                            }
                        },
                        _ => {}
                    }
                },
                _ => {}
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}