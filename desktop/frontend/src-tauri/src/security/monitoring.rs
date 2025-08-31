//! Security Monitoring and Alerting
//! 
//! This module provides real-time security monitoring:
//! - Threat detection and analysis
//! - Behavioral anomaly detection
//! - Resource usage monitoring
//! - Real-time alerting
//! - Security dashboard metrics

use super::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

/// Security monitor for real-time threat detection
pub struct SecurityMonitor {
    config: SecurityConfig,
    threat_patterns: Arc<RwLock<Vec<ThreatPattern>>>,
    security_metrics: Arc<RwLock<SecurityMetrics>>,
    alert_history: Arc<Mutex<VecDeque<SecurityAlert>>>,
    resource_monitors: Arc<RwLock<HashMap<String, ResourceMonitor>>>,
    behavior_baselines: Arc<RwLock<HashMap<String, BehaviorBaseline>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatPattern {
    pub id: String,
    pub name: String,
    pub description: String,
    pub severity: ThreatSeverity,
    pub pattern_type: ThreatType,
    pub conditions: Vec<ThreatCondition>,
    pub threshold: f64,
    pub time_window: Duration,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatSeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatType {
    BruteForce,
    CommandInjection,
    PathTraversal,
    ResourceExhaustion,
    AnomalousAccess,
    PrivilegeEscalation,
    DataExfiltration,
    MaliciousPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatCondition {
    pub field: String,
    pub operator: ConditionOperator,
    pub value: serde_json::Value,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    GreaterThan,
    LessThan,
    Regex,
    RateLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAlert {
    pub id: Uuid,
    pub timestamp: SystemTime,
    pub severity: ThreatSeverity,
    pub threat_type: ThreatType,
    pub title: String,
    pub description: String,
    pub source: String,
    pub affected_resources: Vec<String>,
    pub detection_confidence: f64,
    pub mitigation_suggestions: Vec<String>,
    pub raw_event: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub events_processed: u64,
    pub threats_detected: u64,
    pub alerts_generated: u64,
    pub false_positives: u64,
    pub last_update: SystemTime,
    pub threat_distribution: HashMap<String, u32>,
    pub severity_distribution: HashMap<String, u32>,
    pub recent_activity: Vec<ActivitySample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivitySample {
    pub timestamp: SystemTime,
    pub event_count: u32,
    pub threat_count: u32,
    pub avg_risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitor {
    pub resource_id: String,
    pub resource_type: String,
    pub current_usage: ResourceUsage,
    pub usage_history: VecDeque<ResourceUsage>,
    pub thresholds: ResourceThresholds,
    pub last_alert: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub timestamp: SystemTime,
    pub cpu_percent: f32,
    pub memory_bytes: u64,
    pub disk_io_bytes: u64,
    pub network_io_bytes: u64,
    pub file_handles: u32,
    pub process_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    pub cpu_warning: f32,
    pub cpu_critical: f32,
    pub memory_warning: u64,
    pub memory_critical: u64,
    pub disk_io_warning: u64,
    pub network_io_warning: u64,
    pub file_handles_warning: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorBaseline {
    pub user_id: String,
    pub typical_access_patterns: HashMap<String, f64>,
    pub typical_resources: HashSet<String>,
    pub typical_timeframes: Vec<(u8, u8)>, // (hour_start, hour_end)
    pub baseline_established: SystemTime,
    pub last_updated: SystemTime,
    pub confidence_score: f64,
}

impl SecurityMonitor {
    pub async fn new(config: &SecurityConfig) -> SecurityResult<Self> {
        Ok(Self {
            config: config.clone(),
            threat_patterns: Arc::new(RwLock::new(Vec::new())),
            security_metrics: Arc::new(RwLock::new(SecurityMetrics::default())),
            alert_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            resource_monitors: Arc::new(RwLock::new(HashMap::new())),
            behavior_baselines: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn initialize(&self) -> SecurityResult<()> {
        // Initialize threat patterns
        self.create_default_threat_patterns().await?;

        // Start monitoring tasks
        self.start_monitoring_tasks().await;

        log::info!("Security monitor initialized");
        Ok(())
    }

    /// Process a security event and check for threats
    pub async fn process_security_event(&self, event: &SecurityEvent) -> SecurityResult<()> {
        // Update metrics
        self.update_metrics().await;

        // Check threat patterns
        let alerts = self.analyze_event_for_threats(event).await?;

        // Generate alerts if threats detected
        for alert in alerts {
            self.generate_alert(alert).await?;
        }

        // Update behavior baselines
        if let Some(session_id) = event.session_id {
            self.update_behavior_baseline(&session_id.to_string(), event).await?;
        }

        Ok(())
    }

    /// Analyze an event against threat patterns
    async fn analyze_event_for_threats(&self, event: &SecurityEvent) -> SecurityResult<Vec<SecurityAlert>> {
        let mut alerts = Vec::new();
        let patterns = self.threat_patterns.read().await;

        for pattern in patterns.iter().filter(|p| p.enabled) {
            let risk_score = self.calculate_risk_score(event, pattern).await?;

            if risk_score >= pattern.threshold {
                let alert = SecurityAlert {
                    id: Uuid::new_v4(),
                    timestamp: SystemTime::now(),
                    severity: pattern.severity.clone(),
                    threat_type: pattern.pattern_type.clone(),
                    title: format!("Threat Detected: {}", pattern.name),
                    description: format!("{} (Risk Score: {:.2})", pattern.description, risk_score),
                    source: event.source_component.clone(),
                    affected_resources: vec![event.source_component.clone()],
                    detection_confidence: risk_score,
                    mitigation_suggestions: self.get_mitigation_suggestions(&pattern.pattern_type),
                    raw_event: serde_json::to_value(event).unwrap_or_default(),
                };

                alerts.push(alert);
            }
        }

        Ok(alerts)
    }

    /// Calculate risk score for an event against a pattern
    async fn calculate_risk_score(&self, event: &SecurityEvent, pattern: &ThreatPattern) -> SecurityResult<f64> {
        let mut total_score = 0.0;
        let mut max_possible_score = 0.0;

        for condition in &pattern.conditions {
            max_possible_score += condition.weight;

            if self.evaluate_condition(condition, event).await? {
                total_score += condition.weight;
            }
        }

        // Normalize score to 0-1 range
        Ok(if max_possible_score > 0.0 {
            total_score / max_possible_score
        } else {
            0.0
        })
    }

    /// Evaluate a single threat condition
    async fn evaluate_condition(&self, condition: &ThreatCondition, event: &SecurityEvent) -> SecurityResult<bool> {
        let event_value = self.extract_field_value(&condition.field, event)?;

        match condition.operator {
            ConditionOperator::Equals => {
                Ok(event_value == condition.value)
            }
            ConditionOperator::NotEquals => {
                Ok(event_value != condition.value)
            }
            ConditionOperator::Contains => {
                if let (Some(event_str), Some(condition_str)) = (event_value.as_str(), condition.value.as_str()) {
                    Ok(event_str.contains(condition_str))
                } else {
                    Ok(false)
                }
            }
            ConditionOperator::NotContains => {
                if let (Some(event_str), Some(condition_str)) = (event_value.as_str(), condition.value.as_str()) {
                    Ok(!event_str.contains(condition_str))
                } else {
                    Ok(true)
                }
            }
            ConditionOperator::GreaterThan => {
                if let (Some(event_num), Some(condition_num)) = (event_value.as_f64(), condition.value.as_f64()) {
                    Ok(event_num > condition_num)
                } else {
                    Ok(false)
                }
            }
            ConditionOperator::LessThan => {
                if let (Some(event_num), Some(condition_num)) = (event_value.as_f64(), condition.value.as_f64()) {
                    Ok(event_num < condition_num)
                } else {
                    Ok(false)
                }
            }
            ConditionOperator::Regex => {
                if let (Some(event_str), Some(pattern_str)) = (event_value.as_str(), condition.value.as_str()) {
                    match regex::Regex::new(pattern_str) {
                        Ok(regex) => Ok(regex.is_match(event_str)),
                        Err(_) => Ok(false),
                    }
                } else {
                    Ok(false)
                }
            }
            ConditionOperator::RateLimit => {
                // Rate limiting would require maintaining state over time
                // This is a simplified implementation
                Ok(false)
            }
        }
    }

    /// Extract field value from event
    fn extract_field_value(&self, field: &str, event: &SecurityEvent) -> SecurityResult<serde_json::Value> {
        match field {
            "event_type" => Ok(serde_json::to_value(&event.event_type).unwrap_or_default()),
            "severity" => Ok(serde_json::to_value(&event.severity).unwrap_or_default()),
            "message" => Ok(serde_json::Value::String(event.message.clone())),
            "source" => Ok(serde_json::Value::String(event.source_component.clone())),
            _ => {
                // Try to extract from details
                if let Some(value) = event.details.get(field) {
                    Ok(value.clone())
                } else {
                    Ok(serde_json::Value::Null)
                }
            }
        }
    }

    /// Generate and process an alert
    async fn generate_alert(&self, alert: SecurityAlert) -> SecurityResult<()> {
        // Add to alert history
        let mut history = self.alert_history.lock().await;
        history.push_back(alert.clone());

        // Keep only recent alerts
        if history.len() > 1000 {
            history.pop_front();
        }
        drop(history);

        // Log the alert
        log::warn!(
            "[SECURITY ALERT] {} - {} - {} (Confidence: {:.2})",
            severity_to_string(&alert.severity),
            threat_type_to_string(&alert.threat_type),
            alert.title,
            alert.detection_confidence
        );

        // Send real-time notifications if enabled
        if self.config.security_monitoring_enabled {
            self.send_alert_notification(&alert).await?;
        }

        Ok(())
    }

    /// Send alert notification
    async fn send_alert_notification(&self, alert: &SecurityAlert) -> SecurityResult<()> {
        // In a real implementation, this would send notifications via:
        // - Email
        // - Slack/Discord webhooks
        // - Desktop notifications
        // - System logging
        
        log::info!("Alert notification sent: {}", alert.title);
        Ok(())
    }

    /// Update security metrics
    async fn update_metrics(&self) -> SecurityResult<()> {
        let mut metrics = self.security_metrics.write().await;
        metrics.events_processed += 1;
        metrics.last_update = SystemTime::now();

        // Add activity sample
        let now = SystemTime::now();
        if let Some(last_sample) = metrics.recent_activity.last() {
            if now.duration_since(last_sample.timestamp).unwrap_or(Duration::ZERO) > Duration::from_secs(60) {
                // Add new sample every minute
                let sample = ActivitySample {
                    timestamp: now,
                    event_count: 1,
                    threat_count: 0,
                    avg_risk_score: 0.0,
                };
                metrics.recent_activity.push(sample);

                // Keep only recent samples (last 24 hours)
                metrics.recent_activity.retain(|sample| {
                    now.duration_since(sample.timestamp).unwrap_or(Duration::MAX) < Duration::from_secs(24 * 3600)
                });
            }
        } else {
            // First sample
            let sample = ActivitySample {
                timestamp: now,
                event_count: 1,
                threat_count: 0,
                avg_risk_score: 0.0,
            };
            metrics.recent_activity.push(sample);
        }

        Ok(())
    }

    /// Update behavior baseline for a user
    async fn update_behavior_baseline(&self, user_id: &str, event: &SecurityEvent) -> SecurityResult<()> {
        let mut baselines = self.behavior_baselines.write().await;
        
        let baseline = baselines.entry(user_id.to_string()).or_insert_with(|| {
            BehaviorBaseline {
                user_id: user_id.to_string(),
                typical_access_patterns: HashMap::new(),
                typical_resources: HashSet::new(),
                typical_timeframes: Vec::new(),
                baseline_established: SystemTime::now(),
                last_updated: SystemTime::now(),
                confidence_score: 0.0,
            }
        });

        // Update access patterns
        let pattern_key = format!("{}:{}", event.event_type_to_string(), event.source_component);
        *baseline.typical_access_patterns.entry(pattern_key).or_insert(0.0) += 1.0;

        // Update typical resources
        baseline.typical_resources.insert(event.source_component.clone());

        // Update last activity time
        baseline.last_updated = SystemTime::now();

        // Increase confidence score
        baseline.confidence_score = (baseline.confidence_score + 0.01).min(1.0);

        Ok(())
    }

    /// Create default threat patterns
    async fn create_default_threat_patterns(&self) -> SecurityResult<()> {
        let mut patterns = self.threat_patterns.write().await;

        // Brute force detection
        patterns.push(ThreatPattern {
            id: "brute_force_auth".to_string(),
            name: "Brute Force Authentication".to_string(),
            description: "Multiple failed authentication attempts".to_string(),
            severity: ThreatSeverity::High,
            pattern_type: ThreatType::BruteForce,
            conditions: vec![
                ThreatCondition {
                    field: "event_type".to_string(),
                    operator: ConditionOperator::Equals,
                    value: serde_json::json!("Authentication"),
                    weight: 0.5,
                },
                ThreatCondition {
                    field: "message".to_string(),
                    operator: ConditionOperator::Contains,
                    value: serde_json::json!("failed"),
                    weight: 0.5,
                },
            ],
            threshold: 0.8,
            time_window: Duration::from_secs(300),
            enabled: true,
        });

        // Command injection detection
        patterns.push(ThreatPattern {
            id: "command_injection".to_string(),
            name: "Command Injection Attempt".to_string(),
            description: "Suspicious command patterns detected".to_string(),
            severity: ThreatSeverity::Critical,
            pattern_type: ThreatType::CommandInjection,
            conditions: vec![
                ThreatCondition {
                    field: "message".to_string(),
                    operator: ConditionOperator::Regex,
                    value: serde_json::json!(r"(\$\(|\`|\\|&|;|\|\|)"),
                    weight: 1.0,
                },
            ],
            threshold: 0.7,
            time_window: Duration::from_secs(60),
            enabled: true,
        });

        // Resource exhaustion detection
        patterns.push(ThreatPattern {
            id: "resource_exhaustion".to_string(),
            name: "Resource Exhaustion".to_string(),
            description: "Abnormal resource usage detected".to_string(),
            severity: ThreatSeverity::Medium,
            pattern_type: ThreatType::ResourceExhaustion,
            conditions: vec![
                ThreatCondition {
                    field: "event_type".to_string(),
                    operator: ConditionOperator::Equals,
                    value: serde_json::json!("ResourceUsage"),
                    weight: 1.0,
                },
            ],
            threshold: 0.6,
            time_window: Duration::from_secs(120),
            enabled: true,
        });

        Ok(())
    }

    /// Start background monitoring tasks
    async fn start_monitoring_tasks(&self) {
        // Resource monitoring task
        let resource_monitors = self.resource_monitors.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Monitor system resources
                if let Err(e) = Self::monitor_system_resources(&resource_monitors, &config).await {
                    log::error!("System resource monitoring failed: {}", e);
                }
            }
        });

        // Alert cleanup task
        let alert_history = self.alert_history.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600)); // 1 hour
            
            loop {
                interval.tick().await;
                
                // Clean up old alerts
                let mut history = alert_history.lock().await;
                let cutoff = SystemTime::now() - Duration::from_secs(24 * 3600 * 7); // 7 days
                
                history.retain(|alert| alert.timestamp > cutoff);
            }
        });
    }

    /// Monitor system resources
    async fn monitor_system_resources(
        _monitors: &Arc<RwLock<HashMap<String, ResourceMonitor>>>,
        _config: &SecurityConfig,
    ) -> SecurityResult<()> {
        // Implementation would monitor actual system resources
        // This is a placeholder
        Ok(())
    }

    /// Get recent alerts
    pub async fn get_recent_alerts(&self, limit: usize) -> Vec<SecurityAlert> {
        let history = self.alert_history.lock().await;
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get security metrics
    pub async fn get_security_metrics(&self) -> SecurityMetrics {
        self.security_metrics.read().await.clone()
    }

    /// Get mitigation suggestions for a threat type
    fn get_mitigation_suggestions(&self, threat_type: &ThreatType) -> Vec<String> {
        match threat_type {
            ThreatType::BruteForce => vec![
                "Enable account lockout after failed attempts".to_string(),
                "Implement CAPTCHA verification".to_string(),
                "Use multi-factor authentication".to_string(),
            ],
            ThreatType::CommandInjection => vec![
                "Validate and sanitize all user inputs".to_string(),
                "Use parameterized queries".to_string(),
                "Implement strict input filtering".to_string(),
            ],
            ThreatType::ResourceExhaustion => vec![
                "Implement rate limiting".to_string(),
                "Monitor resource usage closely".to_string(),
                "Set resource usage thresholds".to_string(),
            ],
            _ => vec!["Review security logs and investigate further".to_string()],
        }
    }
}

impl Default for SecurityMetrics {
    fn default() -> Self {
        Self {
            events_processed: 0,
            threats_detected: 0,
            alerts_generated: 0,
            false_positives: 0,
            last_update: SystemTime::now(),
            threat_distribution: HashMap::new(),
            severity_distribution: HashMap::new(),
            recent_activity: Vec::new(),
        }
    }
}

// Helper functions for serialization
trait EventTypeToString {
    fn event_type_to_string(&self) -> String;
}

impl EventTypeToString for SecurityEvent {
    fn event_type_to_string(&self) -> String {
        format!("{:?}", self.event_type)
    }
}

fn threat_type_to_string(threat_type: &ThreatType) -> String {
    match threat_type {
        ThreatType::BruteForce => "BruteForce".to_string(),
        ThreatType::CommandInjection => "CommandInjection".to_string(),
        ThreatType::PathTraversal => "PathTraversal".to_string(),
        ThreatType::ResourceExhaustion => "ResourceExhaustion".to_string(),
        ThreatType::AnomalousAccess => "AnomalousAccess".to_string(),
        ThreatType::PrivilegeEscalation => "PrivilegeEscalation".to_string(),
        ThreatType::DataExfiltration => "DataExfiltration".to_string(),
        ThreatType::MaliciousPayload => "MaliciousPayload".to_string(),
    }
}

fn severity_to_string(severity: &ThreatSeverity) -> String {
    match severity {
        ThreatSeverity::Info => "Info".to_string(),
        ThreatSeverity::Low => "Low".to_string(),
        ThreatSeverity::Medium => "Medium".to_string(),
        ThreatSeverity::High => "High".to_string(),
        ThreatSeverity::Critical => "Critical".to_string(),
    }
}