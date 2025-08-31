#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod mcp_bridge;

use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::Manager;

fn main() {
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
        .invoke_handler(tauri::generate_handler![
            mcp_bridge::start_mcp_backend,
            mcp_bridge::stop_mcp_backend,
            mcp_bridge::mcp_call,
            mcp_bridge::check_mcp_health,
        ])
        .setup(|_app| {
            // Tray icon is now configured in tauri.conf.json
            // No need to set it up here in Tauri 2.x
            
            Ok(())
        })
        .on_window_event(|window, event| {
            // Clean shutdown on window close
            use tauri::WindowEvent;
            if let WindowEvent::CloseRequested { .. } = event {
                let state: tauri::State<Arc<Mutex<Option<mcp_bridge::MCPBridge>>>> = window.state();
                let state_clone = state.inner().clone();
                
                // Stop MCP backend gracefully
                tauri::async_runtime::block_on(async {
                    if let Some(bridge) = state_clone.lock().await.as_ref() {
                        let _ = bridge.stop().await;
                    }
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}