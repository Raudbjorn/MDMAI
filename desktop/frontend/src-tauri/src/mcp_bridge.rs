use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri_plugin_shell::{ShellExt, process::{CommandEvent, CommandChild}};
use log::{info, error, debug, warn};

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
}

impl MCPBridge {
    pub fn new() -> Self {
        MCPBridge {
            stdin_tx: Arc::new(Mutex::new(None)),
            request_id: Arc::new(RwLock::new(0)),
            pending: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
            child_process: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self, app_handle: &tauri::AppHandle) -> Result<(), String> {
        // Check if already running
        if *self.is_running.read().await {
            return Ok(());
        }

        info!("Starting MCP server sidecar process");

        // Create channel for stdin communication
        let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(100);
        *self.stdin_tx.lock().await = Some(stdin_tx);

        // Start Python MCP server using Tauri sidecar
        let (mut rx, child) = app_handle.shell()
            .sidecar("mcp-server")
            .map_err(|e| format!("Failed to create mcp-server sidecar command: {}", e))?
            .env("MCP_STDIO_MODE", "true")
            .spawn()
            .map_err(|e| format!("Failed to start MCP server: {}", e))?;

        let child_id = child.pid();
        
        // Store the child process handle
        *self.child_process.lock().await = Some(child);
        *self.is_running.write().await = true;

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
            debug!("Stdin writer task ended for process {}", child_id);
        });

        // Start reading responses in background
        let pending = self.pending.clone();
        let is_running = self.is_running.clone();
        let child_process_cleanup = self.child_process.clone();
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
                        warn!("MCP stderr: {}", line_str);
                    }
                    CommandEvent::Error(e) => {
                        error!("MCP process error: {}", e);
                    }
                    CommandEvent::Terminated(payload) => {
                        error!("MCP process terminated with code: {:?}", payload.code);
                        *is_running.write().await = false;
                        // Clean up the child process handle
                        *child_process_cleanup.lock().await = None;
                        break;
                    }
                    _ => {}
                }
            }
        });

        // Initialize connection
        self.initialize().await?;

        Ok(())
    }

    pub async fn stop(&self) -> Result<(), String> {
        if *self.is_running.read().await {
            info!("Stopping MCP server process");
            
            // Try to send shutdown command gracefully
            let shutdown_result = self.call("shutdown", Value::Null).await;
            
            // Give process time to shutdown gracefully
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            // Force kill the process if it's still running
            let mut child_guard = self.child_process.lock().await;
            if let Some(mut child) = child_guard.take() {
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
        }

        self.pending.write().await.clear();

        Ok(())
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

        // Wait for response with timeout
        match tokio::time::timeout(tokio::time::Duration::from_secs(30), rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("Response channel closed".to_string()),
            Err(_) => {
                self.pending.write().await.remove(&request_id);
                Err("Request timeout".to_string())
            }
        }
    }

    async fn initialize(&self) -> Result<(), String> {
        debug!("Initializing MCP connection");
        
        // Call server_info to verify connection
        let result = self.call("server_info", Value::Null).await?;
        
        info!("MCP server initialized successfully: {:?}", result);
        Ok(())
    }
    
    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }
}

// Tauri commands
#[tauri::command]
pub async fn start_mcp_backend(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let mut bridge_opt = state.lock().await;
    
    if bridge_opt.is_none() {
        info!("Creating new MCP bridge");
        let bridge = MCPBridge::new();
        bridge.start(&app_handle).await?;
        *bridge_opt = Some(bridge);
    } else {
        debug!("MCP bridge already exists");
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