#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod mcp_bridge;

use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::{Manager, State, WindowEvent};

fn main() {
    tauri::Builder::default()
        .manage(Arc::new(Mutex::new(None::<mcp_bridge::MCPBridge>)))
        .invoke_handler(tauri::generate_handler![
            mcp_bridge::start_mcp_backend,
            mcp_bridge::stop_mcp_backend,
            mcp_bridge::mcp_call,
            mcp_bridge::check_mcp_health,
        ])
        .setup(|app| {
            // Create system tray
            #[cfg(desktop)]
            {
                use tauri::SystemTray;
                let tray = SystemTray::new();
                app.handle().tray_handle_by_id(&tray.id());
            }
            
            Ok(())
        })
        .on_window_event(|event| {
            // Clean shutdown on window close
            if let WindowEvent::CloseRequested { .. } = event.event() {
                let window = event.window();
                let state = window.state::<Arc<Mutex<Option<mcp_bridge::MCPBridge>>>>();
                
                // Stop MCP backend gracefully
                tauri::async_runtime::block_on(async {
                    if let Some(bridge) = state.lock().await.as_ref() {
                        let _ = bridge.stop().await;
                    }
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}