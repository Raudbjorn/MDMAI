use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::api::process::{Command, CommandEvent, CommandChild};
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
    child: Arc<Mutex<Option<CommandChild>>>,
    request_id: Arc<RwLock<u64>>,
    pending: Arc<RwLock<HashMap<u64, tokio::sync::oneshot::Sender<Result<Value, String>>>>>,
}

impl MCPBridge {
    pub fn new() -> Self {
        MCPBridge {
            child: Arc::new(Mutex::new(None)),
            request_id: Arc::new(RwLock::new(0)),
            pending: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn start(&self) -> Result<(), String> {
        // Check if already running
        if self.child.lock().await.is_some() {
            return Ok(());
        }

        info!("Starting MCP server sidecar process");

        // Start Python MCP server using Tauri sidecar
        let (mut rx, child) = Command::new_sidecar("mcp-server")
            .map_err(|e| format!("Failed to create mcp-server sidecar command: {}", e))?
            .env("MCP_STDIO_MODE", "true")
            .spawn()
            .map_err(|e| format!("Failed to start MCP server: {}", e))?;

        // Store process handle
        *self.child.lock().await = Some(child);

        // Start reading responses in background
        let pending = self.pending.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    CommandEvent::Stdout(line) => {
                        debug!("MCP stdout: {}", line);
                        // Parse JSON-RPC response
                        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line) {
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
                        warn!("MCP stderr: {}", line);
                    }
                    CommandEvent::Error(e) => {
                        error!("MCP process error: {}", e);
                    }
                    CommandEvent::Terminated(payload) => {
                        error!("MCP process terminated with code: {:?}", payload.code);
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
        if let Some(child) = self.child.lock().await.take() {
            info!("Stopping MCP server process");
            
            // Send shutdown command if possible
            let _ = self.call("shutdown", Value::Null).await;
            
            // Give process time to shutdown gracefully
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            // Force kill if still running
            let _ = child.kill();
        }

        self.pending.write().await.clear();

        Ok(())
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value, String> {
        // Check if process is running
        let child_guard = self.child.lock().await;
        if let Some(child) = child_guard.as_ref() {
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

            // Send request to Python process via stdin
            let request_str = serde_json::to_string(&request)
                .map_err(|e| format!("Failed to serialize request: {}", e))?;
            
            child.write(format!("{}\n", request_str).as_bytes())
                .map_err(|e| {
                    self.pending.write().await.remove(&request_id);
                    format!("Failed to write to stdin: {}", e)
                })?;

            // Wait for response with timeout
            match tokio::time::timeout(tokio::time::Duration::from_secs(30), rx).await {
                Ok(Ok(result)) => result,
                Ok(Err(_)) => Err("Response channel closed".to_string()),
                Err(_) => {
                    self.pending.write().await.remove(&request_id);
                    Err("Request timeout".to_string())
                }
            }
        } else {
            Err("MCP server not running".to_string())
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
        self.child.lock().await.is_some()
    }
}

// Tauri commands
#[tauri::command]
pub async fn start_mcp_backend(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
) -> Result<(), String> {
    let mut bridge_opt = state.lock().await;
    
    if bridge_opt.is_none() {
        info!("Creating new MCP bridge");
        let bridge = MCPBridge::new();
        bridge.start().await?;
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