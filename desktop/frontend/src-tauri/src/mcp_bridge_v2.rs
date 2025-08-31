use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, mpsc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri_plugin_shell::{ShellExt, process::{CommandEvent, CommandChild}};
use log::{info, error, debug, warn};
use crate::process_manager::{ProcessManager, ProcessState};
use crate::ipc::{IpcManager, JsonRpcMessage, JsonRpcNotification, QueueConfig, parse_jsonrpc_message};

/// Enhanced MCP Bridge with IPC Manager integration
pub struct MCPBridge {
    // IPC management
    ipc_manager: Arc<IpcManager>,
    
    // Process management
    child_process: Arc<Mutex<Option<CommandChild>>>,
    process_manager: Arc<ProcessManager>,
    
    // Communication channels
    stdout_handler: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    stdin_writer: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    
    // State
    is_running: Arc<RwLock<bool>>,
    app_handle: Arc<Mutex<Option<tauri::AppHandle>>>,
    
    // Configuration
    reconnect_attempts: Arc<RwLock<u32>>,
    max_reconnect_attempts: u32,
}

impl MCPBridge {
    /// Create a new MCP bridge with enhanced IPC capabilities
    pub fn new(process_manager: Arc<ProcessManager>) -> Self {
        let ipc_config = QueueConfig {
            max_concurrent_requests: 20,
            max_queue_size: 200,
            default_timeout_ms: 30000,
            max_retries: 3,
            retry_delay_ms: 1000,
            enable_priority_queue: true,
        };
        
        MCPBridge {
            ipc_manager: Arc::new(IpcManager::with_config(ipc_config)),
            child_process: Arc::new(Mutex::new(None)),
            process_manager,
            stdout_handler: Arc::new(Mutex::new(None)),
            stdin_writer: Arc::new(Mutex::new(None)),
            is_running: Arc::new(RwLock::new(false)),
            app_handle: Arc::new(Mutex::new(None)),
            reconnect_attempts: Arc::new(RwLock::new(0)),
            max_reconnect_attempts: 3,
        }
    }
    
    /// Start the MCP server process
    pub async fn start(&self, app_handle: &tauri::AppHandle) -> Result<(), String> {
        // Check if already running
        if *self.is_running.read().await {
            debug!("MCP server already running");
            return Ok(());
        }
        
        info!("Starting MCP server process");
        
        // Store app handle
        *self.app_handle.lock().await = Some(app_handle.clone());
        self.process_manager.set_app_handle(app_handle.clone()).await;
        
        // Start the process
        self.start_process_internal(app_handle).await
    }
    
    /// Internal process start logic
    async fn start_process_internal(&self, app_handle: &tauri::AppHandle) -> Result<(), String> {
        // Create communication channels
        let (stdin_tx, mut stdin_rx) = mpsc::channel::<String>(100);
        let (notification_tx, mut notification_rx) = mpsc::channel::<JsonRpcNotification>(100);
        
        // Set channels in IPC manager
        self.ipc_manager.set_stdin_channel(stdin_tx.clone()).await;
        self.ipc_manager.set_notification_channel(notification_tx).await;
        
        // Start Python MCP server as sidecar
        let (mut rx, child) = app_handle.shell()
            .sidecar("mcp-server")
            .map_err(|e| format!("Failed to create sidecar command: {}", e))?
            .env("MCP_STDIO_MODE", "true")
            .env("PYTHONUNBUFFERED", "1")
            .spawn()
            .map_err(|e| format!("Failed to start MCP server: {}", e))?;
        
        let child_pid = child.pid();
        info!("MCP server started with PID: {}", child_pid);
        
        // Store child process
        *self.child_process.lock().await = Some(child);
        *self.is_running.write().await = true;
        
        // Notify process manager
        self.process_manager.on_process_started(child_pid).await;
        
        // Handle stdin writing
        let child_process_clone = self.child_process.clone();
        let stdin_handle = tokio::spawn(async move {
            debug!("Stdin writer task started");
            while let Some(request) = stdin_rx.recv().await {
                let mut child_guard = child_process_clone.lock().await;
                if let Some(child) = child_guard.as_mut() {
                    if let Err(e) = child.write(request.as_bytes()) {
                        error!("Failed to write to stdin: {}", e);
                        break;
                    }
                    debug!("Sent to stdin: {}", request.trim());
                } else {
                    error!("Child process not available");
                    break;
                }
            }
            debug!("Stdin writer task ended");
        });
        *self.stdin_writer.lock().await = Some(stdin_handle);
        
        // Handle stdout reading and response processing
        let ipc_manager = self.ipc_manager.clone();
        let is_running = self.is_running.clone();
        let process_manager = self.process_manager.clone();
        let app_handle_clone = app_handle.clone();
        
        let stdout_handle = tokio::spawn(async move {
            debug!("Stdout handler task started");
            let mut buffer = String::new();
            
            while let Some(event) = rx.recv().await {
                match event {
                    CommandEvent::Stdout(bytes) => {
                        let output = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&output);
                        
                        // Process complete JSON messages
                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].trim().to_string();
                            buffer = buffer[newline_pos + 1..].to_string();
                            
                            if line.is_empty() {
                                continue;
                            }
                            
                            debug!("Received from stdout: {}", line);
                            
                            // Parse and handle JSON-RPC message
                            match parse_jsonrpc_message(&line) {
                                Ok(JsonRpcMessage::Response(response)) => {
                                    ipc_manager.handle_response(response).await;
                                }
                                Ok(JsonRpcMessage::Notification(notification)) => {
                                    ipc_manager.handle_notification(notification.clone()).await;
                                    
                                    // Emit as Tauri event
                                    let _ = app_handle_clone.emit_all(
                                        &format!("mcp-notification-{}", notification.method),
                                        &notification.params
                                    );
                                }
                                Ok(JsonRpcMessage::Request(request)) => {
                                    // Server is requesting something from us
                                    warn!("Received request from server: {}", request.method);
                                    // Handle server-initiated requests if needed
                                }
                                Err(e) => {
                                    warn!("Failed to parse JSON-RPC message: {} - Line: {}", e, line);
                                }
                            }
                        }
                    }
                    CommandEvent::Stderr(bytes) => {
                        let error = String::from_utf8_lossy(&bytes);
                        warn!("MCP stderr: {}", error);
                        
                        // Emit stderr as event for debugging
                        let _ = app_handle_clone.emit_all("mcp-stderr", error.to_string());
                    }
                    CommandEvent::Error(e) => {
                        error!("MCP process error: {}", e);
                        let _ = app_handle_clone.emit_all("mcp-error", e.to_string());
                    }
                    CommandEvent::Terminated(payload) => {
                        error!("MCP process terminated with code: {:?}", payload.code);
                        *is_running.write().await = false;
                        
                        // Cleanup IPC manager
                        ipc_manager.disconnect().await;
                        
                        // Notify process manager
                        process_manager.on_process_stopped(payload.code).await;
                        
                        // Emit termination event
                        let _ = app_handle_clone.emit_all("mcp-terminated", payload);
                        
                        break;
                    }
                    _ => {}
                }
            }
            debug!("Stdout handler task ended");
        });
        *self.stdout_handler.lock().await = Some(stdout_handle);
        
        // Handle notifications in separate task
        let app_handle_clone2 = app_handle.clone();
        tokio::spawn(async move {
            while let Some(notification) = notification_rx.recv().await {
                // Emit notification as Tauri event
                let _ = app_handle_clone2.emit_all(
                    "mcp-notification",
                    &notification
                );
            }
        });
        
        // Initialize connection
        self.initialize().await?;
        
        // Reset reconnect counter on successful start
        *self.reconnect_attempts.write().await = 0;
        
        Ok(())
    }
    
    /// Initialize MCP connection
    async fn initialize(&self) -> Result<(), String> {
        debug!("Initializing MCP connection");
        
        // Wait a moment for process to start
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        // Call server_info to verify connection
        let result = self.call("server_info", Value::Null).await?;
        
        info!("MCP server initialized: {:?}", result);
        Ok(())
    }
    
    /// Stop the MCP server process
    pub async fn stop(&self) -> Result<(), String> {
        if !*self.is_running.read().await {
            debug!("MCP server not running");
            return Ok(());
        }
        
        info!("Stopping MCP server");
        
        // Try graceful shutdown
        let _ = self.call("shutdown", Value::Null).await;
        
        // Give process time to shutdown
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        // Stop handler tasks
        if let Some(handle) = self.stdout_handler.lock().await.take() {
            handle.abort();
        }
        if let Some(handle) = self.stdin_writer.lock().await.take() {
            handle.abort();
        }
        
        // Force kill if still running
        let mut child_guard = self.child_process.lock().await;
        if let Some(child) = child_guard.take() {
            if let Err(e) = child.kill() {
                error!("Failed to kill MCP server: {}", e);
            } else {
                info!("MCP server terminated");
            }
        }
        
        // Cleanup state
        *self.is_running.write().await = false;
        self.ipc_manager.disconnect().await;
        
        // Notify process manager
        self.process_manager.on_process_stopped(Some(0)).await;
        
        Ok(())
    }
    
    /// Restart the MCP server
    pub async fn restart(&self) -> Result<(), String> {
        info!("Restarting MCP server");
        
        self.stop().await?;
        tokio::time::sleep(Duration::from_millis(1000)).await;
        
        if let Some(app_handle) = self.app_handle.lock().await.as_ref() {
            self.start(app_handle).await
        } else {
            Err("App handle not available".to_string())
        }
    }
    
    /// Call an MCP method
    pub async fn call(&self, method: &str, params: Value) -> Result<Value, String> {
        // Check if running
        if !*self.is_running.read().await {
            // Try to auto-reconnect
            if *self.reconnect_attempts.read().await < self.max_reconnect_attempts {
                *self.reconnect_attempts.write().await += 1;
                
                if let Some(app_handle) = self.app_handle.lock().await.as_ref() {
                    self.start(app_handle).await?;
                } else {
                    return Err("MCP server not running and cannot reconnect".to_string());
                }
            } else {
                return Err("MCP server not running".to_string());
            }
        }
        
        // Send request through IPC manager
        match self.ipc_manager.send_request(
            method.to_string(),
            params,
            None,
            None,
            false,
        ).await {
            Ok(result) => Ok(result),
            Err(e) => Err(format!("{}: {}", e.code, e.message)),
        }
    }
    
    /// Call an MCP method with streaming support
    pub async fn call_streaming(
        &self,
        method: &str,
        params: Value,
        timeout: Option<Duration>,
    ) -> Result<Value, String> {
        // Check if running
        if !*self.is_running.read().await {
            return Err("MCP server not running".to_string());
        }
        
        // Send request with streaming enabled
        match self.ipc_manager.send_request(
            method.to_string(),
            params,
            timeout,
            None,
            true,
        ).await {
            Ok(result) => Ok(result),
            Err(e) => Err(format!("{}: {}", e.code, e.message)),
        }
    }
    
    /// Check if the MCP server is healthy
    pub async fn is_healthy(&self) -> bool {
        if !*self.is_running.read().await {
            return false;
        }
        
        // Perform health check
        match self.call("server_info", Value::Null).await {
            Ok(_) => {
                self.process_manager.on_health_check_result(true, None).await;
                true
            }
            Err(e) => {
                self.process_manager.on_health_check_result(false, Some(e)).await;
                false
            }
        }
    }
    
    /// Get IPC performance metrics
    pub async fn get_metrics(&self) -> crate::ipc::PerformanceMetrics {
        self.ipc_manager.get_metrics().await
    }
    
    /// Reset IPC metrics
    pub async fn reset_metrics(&self) {
        self.ipc_manager.reset_metrics().await;
    }
    
    /// Update IPC configuration
    pub async fn update_ipc_config(&self, config: QueueConfig) {
        self.ipc_manager.update_config(config).await;
    }
    
    /// Cancel a specific request
    pub async fn cancel_request(&self, request_id: u64) -> bool {
        self.ipc_manager.cancel_request(request_id).await
    }
    
    /// Cancel all pending requests
    pub async fn cancel_all_requests(&self) {
        self.ipc_manager.cancel_all_requests().await;
    }
}

// Tauri command handlers
#[tauri::command]
pub async fn start_mcp_backend(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
    process_state: tauri::State<'_, crate::process_manager::ProcessManagerState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let mut bridge_opt = state.lock().await;
    
    if bridge_opt.is_none() {
        info!("Creating new MCP bridge");
        let bridge = MCPBridge::new(process_state.0.clone());
        bridge.start(&app_handle).await?;
        *bridge_opt = Some(bridge);
    } else if let Some(bridge) = bridge_opt.as_ref() {
        if !bridge.is_healthy().await {
            info!("MCP bridge unhealthy, restarting");
            bridge.restart().await?;
        }
    }
    
    Ok(())
}

#[tauri::command]
pub async fn stop_mcp_backend(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
) -> Result<(), String> {
    if let Some(bridge) = state.lock().await.as_ref() {
        bridge.stop().await?;
    }
    Ok(())
}

#[tauri::command]
pub async fn restart_mcp_backend(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
) -> Result<(), String> {
    if let Some(bridge) = state.lock().await.as_ref() {
        bridge.restart().await?;
    } else {
        Err("MCP backend not initialized".to_string())
    }
}

#[tauri::command]
pub async fn mcp_call(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
    method: String,
    params: Value,
) -> Result<Value, String> {
    let bridge_opt = state.lock().await;
    
    if let Some(bridge) = bridge_opt.as_ref() {
        bridge.call(&method, params).await
    } else {
        Err("MCP backend not started".to_string())
    }
}

#[tauri::command]
pub async fn mcp_call_streaming(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
    method: String,
    params: Value,
    timeout_ms: Option<u64>,
) -> Result<Value, String> {
    let bridge_opt = state.lock().await;
    
    if let Some(bridge) = bridge_opt.as_ref() {
        let timeout = timeout_ms.map(Duration::from_millis);
        bridge.call_streaming(&method, params, timeout).await
    } else {
        Err("MCP backend not started".to_string())
    }
}

#[tauri::command]
pub async fn check_mcp_health(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
) -> Result<bool, String> {
    let bridge_opt = state.lock().await;
    
    if let Some(bridge) = bridge_opt.as_ref() {
        Ok(bridge.is_healthy().await)
    } else {
        Ok(false)
    }
}

#[tauri::command]
pub async fn get_mcp_metrics(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
) -> Result<crate::ipc::PerformanceMetrics, String> {
    let bridge_opt = state.lock().await;
    
    if let Some(bridge) = bridge_opt.as_ref() {
        Ok(bridge.get_metrics().await)
    } else {
        Err("MCP backend not started".to_string())
    }
}

#[tauri::command]
pub async fn reset_mcp_metrics(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
) -> Result<(), String> {
    let bridge_opt = state.lock().await;
    
    if let Some(bridge) = bridge_opt.as_ref() {
        bridge.reset_metrics().await;
        Ok(())
    } else {
        Err("MCP backend not started".to_string())
    }
}

#[tauri::command]
pub async fn cancel_mcp_request(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
    request_id: u64,
) -> Result<bool, String> {
    let bridge_opt = state.lock().await;
    
    if let Some(bridge) = bridge_opt.as_ref() {
        Ok(bridge.cancel_request(request_id).await)
    } else {
        Err("MCP backend not started".to_string())
    }
}

#[tauri::command]
pub async fn cancel_all_mcp_requests(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
) -> Result<(), String> {
    let bridge_opt = state.lock().await;
    
    if let Some(bridge) = bridge_opt.as_ref() {
        bridge.cancel_all_requests().await;
        Ok(())
    } else {
        Err("MCP backend not started".to_string())
    }
}