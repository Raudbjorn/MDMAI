use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    request_id: Arc<RwLock<u64>>,
    pending: Arc<RwLock<HashMap<u64, tokio::sync::oneshot::Sender<Result<Value, String>>>>>,
    python_path: String,
    mcp_script_path: String,
}

impl MCPBridge {
    pub fn new(python_path: String, mcp_script_path: String) -> Self {
        MCPBridge {
            child: Arc::new(Mutex::new(None)),
            stdin: Arc::new(Mutex::new(None)),
            request_id: Arc::new(RwLock::new(0)),
            pending: Arc::new(RwLock::new(HashMap::new())),
            python_path,
            mcp_script_path,
        }
    }

    pub async fn start(&self) -> Result<(), String> {
        // Check if already running
        if self.child.lock().await.is_some() {
            return Ok(());
        }

        // Start Python MCP server process
        let mut child = Command::new(&self.python_path)
            .arg(&self.mcp_script_path)
            .env("MCP_STDIO_MODE", "true")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start MCP server: {}", e))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Failed to get stdin".to_string())?;
        
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to get stdout".to_string())?;

        // Store process handles
        *self.stdin.lock().await = Some(stdin);
        *self.child.lock().await = Some(child);

        // Start reading responses in background
        let pending = self.pending.clone();
        tokio::spawn(async move {
            Self::read_responses(stdout, pending).await;
        });

        // Initialize connection
        self.initialize().await?;

        Ok(())
    }

    pub async fn stop(&self) -> Result<(), String> {
        if let Some(mut child) = self.child.lock().await.take() {
            // Send shutdown command if possible
            let _ = self.call("shutdown", Value::Null).await;
            
            // Give process time to shutdown gracefully
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            // Force kill if still running
            let _ = child.kill().await;
        }

        *self.stdin.lock().await = None;
        self.pending.write().await.clear();

        Ok(())
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value, String> {
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

        // Send request to Python process
        if let Some(stdin) = &mut *self.stdin.lock().await {
            let request_str = serde_json::to_string(&request)
                .map_err(|e| format!("Failed to serialize request: {}", e))?;
            
            stdin
                .write_all(format!("{}\n", request_str).as_bytes())
                .await
                .map_err(|e| format!("Failed to write to stdin: {}", e))?;
            
            stdin
                .flush()
                .await
                .map_err(|e| format!("Failed to flush stdin: {}", e))?;
        } else {
            self.pending.write().await.remove(&request_id);
            return Err("MCP server not running".to_string());
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

    async fn read_responses(
        stdout: ChildStdout,
        pending: Arc<RwLock<HashMap<u64, tokio::sync::oneshot::Sender<Result<Value, String>>>>>,
    ) {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => {
                    // EOF - process ended
                    println!("MCP server process ended");
                    break;
                }
                Ok(_) => {
                    // Try to parse JSON-RPC response
                    if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line) {
                        if let Some(tx) = pending.write().await.remove(&response.id) {
                            let result = if let Some(error) = response.error {
                                Err(format!("{} ({})", error.message, error.code))
                            } else if let Some(result) = response.result {
                                Ok(result)
                            } else {
                                Ok(Value::Null)
                            };
                            let _ = tx.send(result);
                        }
                    } else if !line.trim().is_empty() {
                        // Log non-JSON output for debugging
                        println!("MCP stdout: {}", line.trim());
                    }
                }
                Err(e) => {
                    println!("Error reading from MCP server: {}", e);
                    break;
                }
            }
        }

        // Clear all pending requests on disconnect
        let mut pending = pending.write().await;
        for (_, tx) in pending.drain() {
            let _ = tx.send(Err("MCP server disconnected".to_string()));
        }
    }

    async fn initialize(&self) -> Result<(), String> {
        let result = self.call("initialize", serde_json::json!({
            "protocolVersion": "0.1.0",
            "clientInfo": {
                "name": "TTRPG Desktop",
                "version": "1.0.0"
            }
        })).await?;

        if result.get("protocolVersion").is_none() {
            return Err("Invalid initialization response".to_string());
        }

        Ok(())
    }

    pub async fn health_check(&self) -> Result<bool, String> {
        match self.call("server_info", Value::Null).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

// Tauri commands
#[tauri::command]
pub async fn start_mcp_backend(
    state: tauri::State<'_, Arc<Mutex<Option<MCPBridge>>>>,
) -> Result<(), String> {
    let mut bridge_opt = state.lock().await;
    
    if bridge_opt.is_none() {
        // Get paths from environment or config
        let python_path = std::env::var("PYTHON_PATH")
            .unwrap_or_else(|_| "python".to_string());
        let mcp_script_path = std::env::var("MCP_SCRIPT_PATH")
            .unwrap_or_else(|_| "../../backend/main.py".to_string());
        
        let bridge = MCPBridge::new(python_path, mcp_script_path);
        bridge.start().await?;
        *bridge_opt = Some(bridge);
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
        bridge.health_check().await
    } else {
        Ok(false)
    }
}