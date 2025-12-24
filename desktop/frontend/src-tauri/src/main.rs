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
mod security;
mod security_commands;
mod secure_mcp_bridge;
mod performance;
mod performance_commands;

use std::{sync::Arc, time::{SystemTime, UNIX_EPOCH, Instant}};
use tokio::sync::Mutex;
use tauri::{Manager, Emitter, DragDropEvent};
use process_manager::ProcessManagerState;
use native_features::{NativeFeaturesState, DragDropEvent as CustomDragDropEvent};
use data_manager_commands::DataManagerStateWrapper;
use security::{SecurityManager, SecurityConfig};
use security_commands::SecurityManagerState;
use performance_commands::PerformanceManagerState;

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
        .manage(SecurityManagerState::new())
        .manage(PerformanceManagerState::new())
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
            // Security Commands
            security_commands::initialize_security_manager,
            security_commands::create_security_session,
            security_commands::validate_session_permission,
            security_commands::validate_input,
            security_commands::sanitize_string,
            security_commands::validate_file_path,
            security_commands::store_credential,
            security_commands::retrieve_credential,
            security_commands::delete_credential,
            security_commands::check_permission,
            security_commands::get_security_stats,
            security_commands::get_security_alerts,
            security_commands::generate_secure_random,
            security_commands::hash_data,
            security_commands::create_sandboxed_process,
            security_commands::terminate_sandboxed_process,
            security_commands::get_process_status,
            security_commands::log_security_event,
            security_commands::cleanup_expired_items,
            // Secure MCP Bridge Commands
            secure_mcp_bridge::secure_mcp_call,
            secure_mcp_bridge::validate_mcp_method,
            secure_mcp_bridge::get_secure_mcp_stats,
            // Performance Commands
            performance_commands::initialize_performance_manager,
            performance_commands::begin_performance_startup,
            performance_commands::get_performance_metrics,
            performance_commands::update_performance_config,
            performance_commands::run_performance_benchmarks,
            performance_commands::get_resource_stats,
            performance_commands::optimize_memory,
            performance_commands::get_cache_stats,
            performance_commands::clear_memory_caches,
            performance_commands::get_memory_pool_stats,
            performance_commands::get_ipc_optimizer_stats,
            performance_commands::reset_ipc_optimizer,
            performance_commands::get_lazy_loader_stats,
            performance_commands::get_component_info,
            performance_commands::load_component,
            performance_commands::preload_components,
            performance_commands::get_benchmark_history,
            performance_commands::get_benchmark_baselines,
            performance_commands::clear_benchmark_baselines,
            performance_commands::run_startup_benchmark,
            performance_commands::get_metrics_history,
            performance_commands::get_metrics_range,
            performance_commands::get_active_alerts,
            performance_commands::get_performance_trends,
            performance_commands::get_optimization_recommendations,
            performance_commands::get_historical_resource_data,
            performance_commands::get_system_info_summary,
            performance_commands::create_optimization_report,
            performance_commands::export_performance_data,
            performance_commands::shutdown_performance_manager,
            performance_commands::get_performance_status,
            performance_commands::force_optimization,
        ])
        .setup(|app| {
            let startup_time = Instant::now();
            log::info!("Starting TTRPG Assistant with performance optimization");

            // Initialize performance manager
            let performance_manager = app.state::<PerformanceManagerState>();
            
            // Initialize native features
            let native_features = app.state::<NativeFeaturesState>();
            let app_handle = app.handle().clone();
            
            tauri::async_runtime::block_on(async {
                // Begin performance tracking
                let startup_context = performance_manager.0.begin_startup().await;
                
                // Initialize performance manager
                let perf_task = startup_context.begin_task("performance_manager_init").await;
                if let Err(e) = performance_manager.0.initialize().await {
                    log::error!("Failed to initialize performance manager: {}", e);
                    perf_task.complete_with_error(&e).await;
                } else {
                    perf_task.complete().await;
                }
                
                // Initialize native features with performance tracking
                let native_task = startup_context.begin_task("native_features_init").await;
                if let Err(e) = native_features.inner().initialize(&app_handle).await {
                    log::error!("Failed to initialize native features: {}", e);
                    native_task.complete_with_error(&format!("{}", e)).await;
                } else {
                    native_task.complete().await;
                }
                
                // Complete startup tracking
                performance_manager.0.complete_startup(startup_context).await;
                
                let total_startup_time = startup_time.elapsed();
                log::info!("TTRPG Assistant startup completed in {:?}", total_startup_time);
            });
            
            Ok(())
        })
        .on_window_event(|window, event| {
            use tauri::WindowEvent;
            match event {
                WindowEvent::CloseRequested { .. } => {
                    log::info!("Application shutdown requested");
                    
                    // Clean shutdown on window close
                    let mcp_state: tauri::State<Arc<Mutex<Option<mcp_bridge::MCPBridge>>>> = window.state();
                    let mcp_state_clone = mcp_state.inner().clone();
                    
                    let performance_state: tauri::State<PerformanceManagerState> = window.state();
                    let performance_manager = performance_state.0.clone();
                    
                    // Stop services gracefully
                    tauri::async_runtime::block_on(async {
                        // Stop performance manager first
                        if let Err(e) = performance_manager.shutdown().await {
                            log::error!("Error shutting down performance manager: {}", e);
                        }
                        
                        // Stop MCP backend
                        if let Some(bridge) = mcp_state_clone.lock().await.as_ref() {
                            let _ = bridge.stop().await;
                        }
                        
                        log::info!("Application shutdown completed");
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