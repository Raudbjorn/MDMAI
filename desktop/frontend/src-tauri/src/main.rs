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

use std::{sync::Arc, time::{SystemTime, UNIX_EPOCH}};
use tokio::sync::Mutex;
use tauri::{Manager, Emitter, DragDropEvent};
use process_manager::ProcessManagerState;
use native_features::{NativeFeaturesState, DragDropEvent as CustomDragDropEvent};
use data_manager_commands::DataManagerStateWrapper;

/// Convert paths to string vector
fn paths_to_strings(paths: &[std::path::PathBuf]) -> Vec<String> {
    paths.iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect()
}

/// Helper function to create drag drop events and emit them to the window
fn create_and_emit_drag_event(
    window: &tauri::Window,
    event_type: &str,
    files: Vec<String>,
    position: Option<(f64, f64)>,
    event_name: &str,
) {
    let drop_event = CustomDragDropEvent {
        event_type: event_type.to_string(),
        files,
        position,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };
    
    if let Err(e) = window.emit(event_name, &drop_event) {
        log::error!("Failed to emit {}: {}", event_name, e);
    }
}

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
            let app_handle = app.handle().clone();
            
            tauri::async_runtime::block_on(async {
                if let Err(e) = native_features.inner().initialize(&app_handle).await {
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
                    match drag_event {
                        DragDropEvent::Drop { paths, position } => {
                            create_and_emit_drag_event(window, "drop", paths_to_strings(paths), Some((position.x, position.y)), "drag-drop");
                        },
                        DragDropEvent::Enter { paths, position } => {
                            create_and_emit_drag_event(window, "enter", paths_to_strings(paths), Some((position.x, position.y)), "drag-enter");
                        },
                        DragDropEvent::Over { position } => {
                            create_and_emit_drag_event(window, "over", vec![], Some((position.x, position.y)), "drag-over");
                        },
                        DragDropEvent::Leave => {
                            create_and_emit_drag_event(window, "leave", vec![], None, "drag-leave");
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