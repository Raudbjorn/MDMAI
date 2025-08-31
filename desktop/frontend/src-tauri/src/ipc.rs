use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock, oneshot, mpsc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use log::{info, error, debug, warn, trace};

// JSON-RPC 2.0 protocol structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<RequestId>,
    pub method: String,
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    Number(u64),
    String(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<RequestId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

// JSON-RPC error codes
pub mod error_codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
    pub const SERVER_ERROR: i32 = -32000;
    pub const TIMEOUT_ERROR: i32 = -32001;
    pub const CANCELLED_ERROR: i32 = -32002;
}

// Notification (no id field)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    pub params: Value,
}

// Stream chunk for large responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: RequestId,
    pub sequence: u32,
    pub total_chunks: Option<u32>,
    pub data: Vec<u8>,
    pub is_final: bool,
}

// Request metadata for tracking
#[derive(Debug, Clone)]
pub struct RequestMetadata {
    pub id: RequestId,
    pub method: String,
    pub timestamp: Instant,
    pub timeout: Duration,
    pub retries: u32,
    pub priority: u8,
}

// Response handling
pub type ResponseSender = oneshot::Sender<Result<Value, JsonRpcError>>;

// Stream handling
pub type StreamSender = mpsc::Sender<StreamChunk>;
pub type StreamReceiver = mpsc::Receiver<StreamChunk>;

// Pending request tracker
pub struct PendingRequest {
    pub metadata: RequestMetadata,
    pub sender: ResponseSender,
    pub stream_sender: Option<StreamSender>,
}

// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_requests: u64,
    pub successful_responses: u64,
    pub failed_responses: u64,
    pub timeouts: u64,
    pub average_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub requests_per_second: f64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub active_requests: u32,
    pub queued_requests: u32,
    pub last_update: u64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        PerformanceMetrics {
            total_requests: 0,
            successful_responses: 0,
            failed_responses: 0,
            timeouts: 0,
            average_latency_ms: 0.0,
            min_latency_ms: f64::MAX,
            max_latency_ms: 0.0,
            requests_per_second: 0.0,
            bytes_sent: 0,
            bytes_received: 0,
            active_requests: 0,
            queued_requests: 0,
            last_update: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
}

// Queue configuration
#[derive(Debug, Clone)]
pub struct QueueConfig {
    pub max_concurrent_requests: usize,
    pub max_queue_size: usize,
    pub default_timeout_ms: u64,
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub enable_priority_queue: bool,
}

impl Default for QueueConfig {
    fn default() -> Self {
        QueueConfig {
            max_concurrent_requests: 10,
            max_queue_size: 100,
            default_timeout_ms: 30000,
            max_retries: 3,
            retry_delay_ms: 1000,
            enable_priority_queue: true,
        }
    }
}

// IPC Manager for handling all communication
pub struct IpcManager {
    // Request tracking
    request_counter: Arc<RwLock<u64>>,
    pending_requests: Arc<RwLock<HashMap<u64, PendingRequest>>>,
    request_queue: Arc<Mutex<VecDeque<(JsonRpcRequest, ResponseSender, Option<StreamSender>)>>>,
    
    // Stream handling
    active_streams: Arc<RwLock<HashMap<u64, StreamReceiver>>>,
    stream_buffers: Arc<RwLock<HashMap<u64, Vec<StreamChunk>>>>,
    
    // Communication channels
    stdin_tx: Arc<Mutex<Option<mpsc::Sender<String>>>>,
    notification_tx: Arc<Mutex<Option<mpsc::Sender<JsonRpcNotification>>>>,
    
    // Performance tracking
    metrics: Arc<RwLock<PerformanceMetrics>>,
    latency_samples: Arc<Mutex<Vec<f64>>>,
    
    // Configuration
    config: Arc<RwLock<QueueConfig>>,
    
    // State
    is_connected: Arc<RwLock<bool>>,
}

impl IpcManager {
    pub fn new() -> Self {
        IpcManager {
            request_counter: Arc::new(RwLock::new(0)),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            active_streams: Arc::new(RwLock::new(HashMap::new())),
            stream_buffers: Arc::new(RwLock::new(HashMap::new())),
            stdin_tx: Arc::new(Mutex::new(None)),
            notification_tx: Arc::new(Mutex::new(None)),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            latency_samples: Arc::new(Mutex::new(Vec::with_capacity(1000))),
            config: Arc::new(RwLock::new(QueueConfig::default())),
            is_connected: Arc::new(RwLock::new(false)),
        }
    }
    
    pub fn with_config(config: QueueConfig) -> Self {
        let mut manager = Self::new();
        *manager.config.blocking_write() = config;
        manager
    }
    
    // Set the stdin channel for sending requests
    pub async fn set_stdin_channel(&self, tx: mpsc::Sender<String>) {
        *self.stdin_tx.lock().await = Some(tx);
        *self.is_connected.write().await = true;
    }
    
    // Set the notification channel for receiving notifications
    pub async fn set_notification_channel(&self, tx: mpsc::Sender<JsonRpcNotification>) {
        *self.notification_tx.lock().await = Some(tx);
    }
    
    // Generate next request ID
    async fn next_request_id(&self) -> RequestId {
        let mut counter = self.request_counter.write().await;
        *counter += 1;
        RequestId::Number(*counter)
    }
    
    // Send a request with optional streaming support
    pub async fn send_request(
        &self,
        method: String,
        params: Value,
        timeout: Option<Duration>,
        priority: Option<u8>,
        enable_streaming: bool,
    ) -> Result<Value, JsonRpcError> {
        // Check connection
        if !*self.is_connected.read().await {
            return Err(JsonRpcError {
                code: error_codes::INTERNAL_ERROR,
                message: "Not connected to server".to_string(),
                data: None,
            });
        }
        
        // Generate request ID
        let id = self.next_request_id().await;
        let id_num = match &id {
            RequestId::Number(n) => *n,
            RequestId::String(s) => s.parse().unwrap_or(0),
        };
        
        // Create request
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(id.clone()),
            method: method.clone(),
            params,
        };
        
        // Create response channel
        let (tx, rx) = oneshot::channel();
        
        // Create stream channel if needed
        let stream_channel = if enable_streaming {
            let (stream_tx, stream_rx) = mpsc::channel(100);
            self.active_streams.write().await.insert(id_num, stream_rx);
            Some(stream_tx)
        } else {
            None
        };
        
        // Check queue limits
        let config = self.config.read().await;
        let queue_size = self.request_queue.lock().await.len();
        if queue_size >= config.max_queue_size {
            return Err(JsonRpcError {
                code: error_codes::SERVER_ERROR,
                message: "Request queue is full".to_string(),
                data: Some(serde_json::json!({ "queue_size": queue_size })),
            });
        }
        
        // Create metadata
        let metadata = RequestMetadata {
            id: id.clone(),
            method: method.clone(),
            timestamp: Instant::now(),
            timeout: timeout.unwrap_or(Duration::from_millis(config.default_timeout_ms)),
            retries: 0,
            priority: priority.unwrap_or(5),
        };
        
        // Add to pending requests
        self.pending_requests.write().await.insert(
            id_num,
            PendingRequest {
                metadata,
                sender: tx,
                stream_sender: stream_channel.clone(),
            },
        );
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_requests += 1;
            metrics.active_requests += 1;
            metrics.queued_requests = queue_size as u32 + 1;
        }
        
        // Queue or send request
        if self.should_queue_request().await {
            self.request_queue.lock().await.push_back((request, tx, stream_channel));
            debug!("Request queued: {} (queue size: {})", method, queue_size + 1);
        } else {
            self.send_request_internal(request).await?;
        }
        
        // Wait for response with timeout
        let timeout_duration = timeout.unwrap_or(Duration::from_millis(config.default_timeout_ms));
        match tokio::time::timeout(timeout_duration, rx).await {
            Ok(Ok(result)) => {
                self.update_success_metrics(id_num).await;
                result
            }
            Ok(Err(_)) => {
                self.update_failure_metrics(id_num, true).await;
                Err(JsonRpcError {
                    code: error_codes::CANCELLED_ERROR,
                    message: "Request cancelled".to_string(),
                    data: None,
                })
            }
            Err(_) => {
                self.update_failure_metrics(id_num, true).await;
                self.pending_requests.write().await.remove(&id_num);
                Err(JsonRpcError {
                    code: error_codes::TIMEOUT_ERROR,
                    message: format!("Request timeout after {:?}", timeout_duration),
                    data: None,
                })
            }
        }
    }
    
    // Internal request sending
    async fn send_request_internal(&self, request: JsonRpcRequest) -> Result<(), JsonRpcError> {
        let request_str = serde_json::to_string(&request)
            .map_err(|e| JsonRpcError {
                code: error_codes::PARSE_ERROR,
                message: format!("Failed to serialize request: {}", e),
                data: None,
            })?;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.bytes_sent += request_str.len() as u64;
        }
        
        // Send via stdin channel
        if let Some(tx) = self.stdin_tx.lock().await.as_ref() {
            tx.send(format!("{}\n", request_str)).await
                .map_err(|e| JsonRpcError {
                    code: error_codes::INTERNAL_ERROR,
                    message: format!("Failed to send request: {}", e),
                    data: None,
                })?;
            
            trace!("Sent request: {}", request_str);
            Ok(())
        } else {
            Err(JsonRpcError {
                code: error_codes::INTERNAL_ERROR,
                message: "Stdin channel not available".to_string(),
                data: None,
            })
        }
    }
    
    // Check if request should be queued
    async fn should_queue_request(&self) -> bool {
        let config = self.config.read().await;
        let active = self.pending_requests.read().await.len();
        active >= config.max_concurrent_requests
    }
    
    // Process queued requests
    pub async fn process_queue(&self) {
        while !self.should_queue_request().await {
            let next = self.request_queue.lock().await.pop_front();
            if let Some((request, _, _)) = next {
                if let Err(e) = self.send_request_internal(request).await {
                    error!("Failed to send queued request: {:?}", e);
                }
            } else {
                break;
            }
        }
    }
    
    // Handle incoming response
    pub async fn handle_response(&self, response: JsonRpcResponse) {
        let id_num = match &response.id {
            Some(RequestId::Number(n)) => *n,
            Some(RequestId::String(s)) => s.parse().unwrap_or(0),
            None => {
                warn!("Received response without ID");
                return;
            }
        };
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.bytes_received += serde_json::to_string(&response).unwrap_or_default().len() as u64;
        }
        
        // Find pending request
        if let Some(pending) = self.pending_requests.write().await.remove(&id_num) {
            // Calculate latency
            let latency = pending.metadata.timestamp.elapsed();
            self.record_latency(latency.as_millis() as f64).await;
            
            // Send response
            if let Some(result) = response.result {
                let _ = pending.sender.send(Ok(result));
            } else if let Some(error) = response.error {
                let _ = pending.sender.send(Err(error));
            }
            
            debug!("Response handled for request {} in {:?}", id_num, latency);
        } else {
            warn!("Received response for unknown request: {}", id_num);
        }
        
        // Process any queued requests
        self.process_queue().await;
    }
    
    // Handle stream chunk
    pub async fn handle_stream_chunk(&self, chunk: StreamChunk) {
        let id_num = match &chunk.id {
            RequestId::Number(n) => *n,
            RequestId::String(s) => s.parse().unwrap_or(0),
        };
        
        // Store chunk in buffer
        let mut buffers = self.stream_buffers.write().await;
        let buffer = buffers.entry(id_num).or_insert_with(Vec::new);
        buffer.push(chunk.clone());
        
        // If final chunk, assemble and deliver
        if chunk.is_final {
            if let Some(chunks) = buffers.remove(&id_num) {
                // Send to stream receiver if exists
                if let Some(mut receiver) = self.active_streams.write().await.remove(&id_num) {
                    for chunk in chunks {
                        // Note: In real implementation, send chunks through the receiver
                        debug!("Stream chunk {} of request {}", chunk.sequence, id_num);
                    }
                }
            }
        }
    }
    
    // Handle notification
    pub async fn handle_notification(&self, notification: JsonRpcNotification) {
        if let Some(tx) = self.notification_tx.lock().await.as_ref() {
            if let Err(e) = tx.send(notification.clone()).await {
                error!("Failed to forward notification: {}", e);
            }
        }
        
        // Emit as Tauri event
        debug!("Received notification: {}", notification.method);
    }
    
    // Record latency sample
    async fn record_latency(&self, latency_ms: f64) {
        let mut samples = self.latency_samples.lock().await;
        samples.push(latency_ms);
        
        // Keep only last 1000 samples
        if samples.len() > 1000 {
            samples.remove(0);
        }
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.average_latency_ms = samples.iter().sum::<f64>() / samples.len() as f64;
        metrics.min_latency_ms = samples.iter().fold(f64::MAX, |a, &b| a.min(b));
        metrics.max_latency_ms = samples.iter().fold(0.0, |a, &b| a.max(b));
    }
    
    // Update success metrics
    async fn update_success_metrics(&self, request_id: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.successful_responses += 1;
        metrics.active_requests = metrics.active_requests.saturating_sub(1);
        metrics.last_update = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
    
    // Update failure metrics
    async fn update_failure_metrics(&self, request_id: u64, is_timeout: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.failed_responses += 1;
        if is_timeout {
            metrics.timeouts += 1;
        }
        metrics.active_requests = metrics.active_requests.saturating_sub(1);
        metrics.last_update = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
    
    // Get current metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        let metrics = self.metrics.read().await.clone();
        metrics
    }
    
    // Reset metrics
    pub async fn reset_metrics(&self) {
        *self.metrics.write().await = PerformanceMetrics::default();
        self.latency_samples.lock().await.clear();
    }
    
    // Update configuration
    pub async fn update_config(&self, config: QueueConfig) {
        *self.config.write().await = config;
    }
    
    // Cancel a pending request
    pub async fn cancel_request(&self, request_id: u64) -> bool {
        if let Some(pending) = self.pending_requests.write().await.remove(&request_id) {
            let _ = pending.sender.send(Err(JsonRpcError {
                code: error_codes::CANCELLED_ERROR,
                message: "Request cancelled by user".to_string(),
                data: None,
            }));
            true
        } else {
            false
        }
    }
    
    // Cancel all pending requests
    pub async fn cancel_all_requests(&self) {
        let mut pending = self.pending_requests.write().await;
        for (_, request) in pending.drain() {
            let _ = request.sender.send(Err(JsonRpcError {
                code: error_codes::CANCELLED_ERROR,
                message: "All requests cancelled".to_string(),
                data: None,
            }));
        }
        
        self.request_queue.lock().await.clear();
    }
    
    // Disconnect and cleanup
    pub async fn disconnect(&self) {
        *self.is_connected.write().await = false;
        self.cancel_all_requests().await;
        *self.stdin_tx.lock().await = None;
        *self.notification_tx.lock().await = None;
        self.active_streams.write().await.clear();
        self.stream_buffers.write().await.clear();
    }
}

// Helper function to parse JSON-RPC message
pub fn parse_jsonrpc_message(input: &str) -> Result<JsonRpcMessage, serde_json::Error> {
    // Try to parse as response first
    if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(input) {
        return Ok(JsonRpcMessage::Response(response));
    }
    
    // Try to parse as notification
    if let Ok(notification) = serde_json::from_str::<JsonRpcNotification>(input) {
        return Ok(JsonRpcMessage::Notification(notification));
    }
    
    // Try to parse as request
    if let Ok(request) = serde_json::from_str::<JsonRpcRequest>(input) {
        return Ok(JsonRpcMessage::Request(request));
    }
    
    Err(serde::de::Error::custom("Unknown JSON-RPC message type"))
}

// Enum for different message types
#[derive(Debug)]
pub enum JsonRpcMessage {
    Request(JsonRpcRequest),
    Response(JsonRpcResponse),
    Notification(JsonRpcNotification),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_request_id_generation() {
        let manager = IpcManager::new();
        let id1 = manager.next_request_id().await;
        let id2 = manager.next_request_id().await;
        
        match (id1, id2) {
            (RequestId::Number(n1), RequestId::Number(n2)) => {
                assert_eq!(n2, n1 + 1);
            }
            _ => panic!("Expected numeric IDs"),
        }
    }
    
    #[tokio::test]
    async fn test_queue_management() {
        let config = QueueConfig {
            max_concurrent_requests: 2,
            max_queue_size: 5,
            ..Default::default()
        };
        let manager = IpcManager::with_config(config);
        
        // Should not queue when under limit
        assert!(!manager.should_queue_request().await);
        
        // Add pending requests
        for i in 0..2 {
            manager.pending_requests.write().await.insert(
                i,
                PendingRequest {
                    metadata: RequestMetadata {
                        id: RequestId::Number(i as u64),
                        method: "test".to_string(),
                        timestamp: Instant::now(),
                        timeout: Duration::from_secs(30),
                        retries: 0,
                        priority: 5,
                    },
                    sender: oneshot::channel().0,
                    stream_sender: None,
                },
            );
        }
        
        // Should queue when at limit
        assert!(manager.should_queue_request().await);
    }
    
    #[tokio::test]
    async fn test_metrics_tracking() {
        let manager = IpcManager::new();
        
        // Record some latencies
        manager.record_latency(10.0).await;
        manager.record_latency(20.0).await;
        manager.record_latency(15.0).await;
        
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.average_latency_ms, 15.0);
        assert_eq!(metrics.min_latency_ms, 10.0);
        assert_eq!(metrics.max_latency_ms, 20.0);
    }
}