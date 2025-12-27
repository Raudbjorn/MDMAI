use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri_plugin_shell::{ShellExt, process::{CommandEvent, CommandChild}};
use log::{info, error, debug, warn};
use crate::process_manager::ProcessManager;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

pub struct MCPBridge {
    stdin_tx: Arc<Mutex<Option<tokio::sync::mpsc::Sender<String>>>>,
    request_id: Arc<RwLock<u64>>,
    pending: Arc<RwLock<HashMap<u64, tokio::sync::oneshot::Sender<Result<Value, String>>>>>,
    is_running: Arc<RwLock<bool>>,
    child_process: Arc<Mutex<Option<CommandChild>>>,
    process_manager: Arc<ProcessManager>,
    app_handle: Arc<Mutex<Option<tauri::AppHandle>>>,
    restart_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl MCPBridge {
    pub fn new(process_manager: Arc<ProcessManager>) -> Self {
        MCPBridge {
            stdin_tx: Arc::new(Mutex::new(None)),
            request_id: Arc::new(RwLock::new(0)),
            pending: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
            child_process: Arc::new(Mutex::new(None)),
            process_manager,
            app_handle: Arc::new(Mutex::new(None)),
            restart_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self, app_handle: &tauri::AppHandle) -> Result<(), String> {
        // Check if already running
        if *self.is_running.read().await {
            info!("MCP server already running, skipping start");
            return Ok(());
        }

        info!("Starting MCP server sidecar process");
        eprintln!("[MCP_BRIDGE] Starting MCP server sidecar process");
        
        // Store app handle for event emission
        *self.app_handle.lock().await = Some(app_handle.clone());
        self.process_manager.set_app_handle(app_handle.clone()).await;

        // Start the process
        self.start_process_internal(app_handle.clone()).await
    }
    
    async fn start_process_internal(&self, app_handle: tauri::AppHandle) -> Result<(), String> {
        // Create channel for stdin communication
        let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(100);
        *self.stdin_tx.lock().await = Some(stdin_tx);

        // Start Python MCP server using Tauri sidecar
        eprintln!("[MCP_BRIDGE] Creating sidecar command...");
        let sidecar_cmd = app_handle.shell()
            .sidecar("mcp-server")
            .map_err(|e| {
                eprintln!("[MCP_BRIDGE] Failed to create sidecar command: {}", e);
                format!("Failed to create mcp-server sidecar command: {}", e)
            })?
            .env("MCP_STDIO_MODE", "true");

        eprintln!("[MCP_BRIDGE] Spawning sidecar process...");
        let (mut rx, child) = sidecar_cmd
            .spawn()
            .map_err(|e| {
                eprintln!("[MCP_BRIDGE] Failed to spawn sidecar: {}", e);
                format!("Failed to start MCP server: {}", e)
            })?;

        eprintln!("[MCP_BRIDGE] Sidecar spawned with PID: {}", child.pid());

        let child_pid = child.pid();
        
        // Store the child process handle
        *self.child_process.lock().await = Some(child);
        *self.is_running.write().await = true;
        
        // Notify process manager
        self.process_manager.on_process_started(child_pid).await;

        // Handle stdin writing in separate task
        let child_process_clone = self.child_process.clone();
        tokio::spawn(async move {
            while let Some(request) = stdin_rx.recv().await {
                let mut child_guard = child_process_clone.lock().await;
                if let Some(child) = child_guard.as_mut() {
                    if let Err(e) = child.write(request.as_bytes()) {
                        error!("Failed to write to stdin: {}", e);
                        break;
                    }
                } else {
                    error!("Child process not available for writing");
                    break;
                }
            }
            debug!("Stdin writer task ended for process {}", child_pid);
        });

        // Start reading responses in background
        let pending = self.pending.clone();
        let is_running = self.is_running.clone();
        let child_process_cleanup = self.child_process.clone();
        let process_manager = self.process_manager.clone();
        
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    CommandEvent::Stdout(line) => {
                        let line_str = String::from_utf8_lossy(&line);
                        debug!("MCP stdout: {}", line_str);
                        
                        // Parse JSON-RPC response
                        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line_str) {
                            if let Some(sender) = pending.write().await.remove(&response.id) {
                                if let Some(result) = response.result {
                                    let _ = sender.send(Ok(result));
                                } else if let Some(error) = response.error {
                                    let _ = sender.send(Err(error.message));
                                }
                            }
                        }
                    }
                    CommandEvent::Stderr(line) => {
                        let line_str = String::from_utf8_lossy(&line);
                        debug!("MCP stderr: {}", line_str);
                    }
                    CommandEvent::Error(e) => {
                        error!("MCP process error: {}", e);
                    }
                    CommandEvent::Terminated(payload) => {
                        error!("MCP process terminated with code: {:?}, signal: {:?}", payload.code, payload.signal);
                        *is_running.write().await = false;
                        
                        // Notify process manager
                        process_manager.on_process_stopped(payload.code).await;
                        
                        // Clean up the child process handle
                        *child_process_cleanup.lock().await = None;
                        
                        // Check if we should restart
                        if process_manager.should_restart().await {
                            // We'll handle restart separately
                            info!("MCP server should be restarted - marking for restart");
                            // The restart will be handled by the process manager or external logic
                        }
                        
                        break;
                    }
                    _ => {}
                }
            }
        });

        // Wait for MCP server to initialize (it takes ~8 seconds to load models)
        // This delay ensures the server is ready to receive JSON-RPC requests
        info!("Waiting for MCP server to initialize...");
        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

        // Initialize connection with retries
        let mut init_attempts = 0;
        let max_init_attempts = 3;
        loop {
            init_attempts += 1;
            match self.initialize().await {
                Ok(()) => {
                    info!("MCP connection initialized successfully");
                    break;
                }
                Err(e) if init_attempts < max_init_attempts => {
                    warn!("Initialize attempt {} failed: {}, retrying...", init_attempts, e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
                Err(e) => {
                    error!("Failed to initialize MCP connection after {} attempts: {}", init_attempts, e);
                    return Err(e);
                }
            }
        }

        // Reset restart count on successful start
        self.process_manager.reset_restart_count().await;

        Ok(())
    }
    

    pub async fn stop(&self) -> Result<(), String> {
        if *self.is_running.read().await {
            info!("Stopping MCP server process");
            
            // Cancel any pending restart
            if let Some(handle) = self.restart_handle.lock().await.take() {
                handle.abort();
            }
            
            // Try to send shutdown command gracefully
            let shutdown_result = self.call("shutdown", Value::Null).await;
            
            // Give process time to shutdown gracefully
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            // Force kill the process if it's still running
            let mut child_guard = self.child_process.lock().await;
            if let Some(child) = child_guard.take() {
                // Check if graceful shutdown failed
                if shutdown_result.is_err() {
                    warn!("Graceful shutdown failed, forcefully terminating process");
                }
                
                // Kill the process
                if let Err(e) = child.kill() {
                    error!("Failed to kill MCP server process: {}", e);
                    // Process might have already exited, which is fine
                } else {
                    info!("MCP server process terminated successfully");
                }
            }
            
            // Clear state
            *self.is_running.write().await = false;
            *self.stdin_tx.lock().await = None;
            
            // Notify process manager
            self.process_manager.on_process_stopped(Some(0)).await;
        }

        self.pending.write().await.clear();

        Ok(())
    }
    
    pub async fn restart(&self) -> Result<(), String> {
        info!("Restarting MCP server");
        
        // Stop the current process
        self.stop().await?;
        
        // Wait a moment
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
        
        // Start again
        if let Some(app_handle) = self.app_handle.lock().await.as_ref() {
            self.start(app_handle).await
        } else {
            Err("App handle not available for restart".to_string())
        }
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value, String> {
        // Check if process is running
        if !*self.is_running.read().await {
            return Err("MCP server not running".to_string());
        }

        // Get next request ID
        let request_id = {
            let mut id = self.request_id.write().await;
            *id += 1;
            *id
        };

        // Create JSON-RPC request
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: request_id,
            method: method.to_string(),
            params,
        };

        // Create response channel
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.write().await.insert(request_id, tx);

        // Send request to Python process via stdin channel
        let request_str = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;
        
        // Send via channel
        if let Some(stdin_tx) = self.stdin_tx.lock().await.as_ref() {
            stdin_tx.send(format!("{}\n", request_str)).await
                .map_err(|e| {
                    format!("Failed to send to stdin channel: {}", e)
                })?;
        } else {
            self.pending.write().await.remove(&request_id);
            return Err("Stdin channel not available".to_string());
        }

        // Wait for response with timeout - use shorter timeout for tests
        let timeout_duration = if cfg!(test) { 
            tokio::time::Duration::from_millis(100) 
        } else { 
            tokio::time::Duration::from_secs(30) 
        };
        match tokio::time::timeout(timeout_duration, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("Response channel closed".to_string()),
            Err(_) => {
                self.pending.write().await.remove(&request_id);
                Err("Request timeout".to_string())
            }
        }
    }

    /// Send a JSON-RPC notification (no ID, no response expected)
    async fn notify(&self, method: &str) -> Result<(), String> {
        if !*self.is_running.read().await {
            return Err("MCP server not running".to_string());
        }

        // Create JSON-RPC notification (no ID field)
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method
        });

        let notification_str = serde_json::to_string(&notification)
            .map_err(|e| format!("Failed to serialize notification: {}", e))?;

        if let Some(stdin_tx) = self.stdin_tx.lock().await.as_ref() {
            stdin_tx.send(format!("{}\n", notification_str)).await
                .map_err(|e| format!("Failed to send notification: {}", e))?;
        } else {
            return Err("Stdin channel not available".to_string());
        }

        Ok(())
    }

    async fn initialize(&self) -> Result<(), String> {
        debug!("Initializing MCP connection");

        // Call MCP initialize with required protocol parameters
        let init_params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "TTRPG Assistant Desktop",
                "version": "1.0.0"
            }
        });

        let result = self.call("initialize", init_params).await?;
        info!("MCP server initialized: {:?}", result);

        // Send notifications/initialized notification (required by MCP protocol)
        // This must be sent before any other requests
        self.notify("notifications/initialized").await?;
        info!("MCP initialized notification sent");

        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        if !*self.is_running.read().await {
            return false;
        }

        // Perform health check using tools/list which is always available
        match self.call("tools/list", Value::Null).await {
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
}

// Tauri commands
#[tauri::command]
pub async fn start_mcp_backend(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
    process_state: tauri::State<'_, crate::process_manager::ProcessManagerState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    eprintln!("[MCP_BRIDGE] start_mcp_backend command invoked");
    let mut bridge_opt = state.lock().await;

    if bridge_opt.is_none() {
        info!("Creating new MCP bridge");
        eprintln!("[MCP_BRIDGE] Creating new MCP bridge");
        let bridge = MCPBridge::new(process_state.0.clone());
        bridge.start(&app_handle).await?;
        *bridge_opt = Some(bridge);
    } else if let Some(bridge) = bridge_opt.as_ref() {
        // Check if process is actually running
        if !bridge.is_healthy().await {
            info!("MCP bridge exists but process is not healthy, restarting");
            bridge.restart().await?;
        } else {
            debug!("MCP bridge already running and healthy");
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
        bridge.restart().await
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