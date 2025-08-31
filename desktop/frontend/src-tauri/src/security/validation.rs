//! Input Validation and Sanitization
//! 
//! This module provides comprehensive input validation and sanitization:
//! - Command injection prevention
//! - Path traversal protection
//! - Data type validation
//! - Format validation (email, URLs, etc.)
//! - Sanitization of user inputs
//! - Schema-based validation

use super::*;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use url::Url;

/// Input validator with comprehensive validation rules
pub struct InputValidator {
    config: SecurityConfig,
    validation_rules: Arc<RwLock<HashMap<String, ValidationRule>>>,
    compiled_patterns: Arc<RwLock<HashMap<String, Regex>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub name: String,
    pub rule_type: ValidationType,
    pub required: bool,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub allowed_values: Option<Vec<String>>,
    pub custom_validator: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    String,
    Integer,
    Float,
    Boolean,
    Email,
    Url,
    Path,
    Command,
    Json,
    Uuid,
    Base64,
    Hex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRequest {
    pub field_name: String,
    pub value: serde_json::Value,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub sanitized_value: Option<serde_json::Value>,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub field: String,
    pub error_type: String,
    pub message: String,
    pub severity: ValidationSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl InputValidator {
    pub fn new(config: &SecurityConfig) -> SecurityResult<Self> {
        Ok(Self {
            config: config.clone(),
            validation_rules: Arc::new(RwLock::new(HashMap::new())),
            compiled_patterns: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn initialize(&self) -> SecurityResult<()> {
        // Initialize default validation rules
        self.create_default_validation_rules().await?;

        log::info!("Input validator initialized");
        Ok(())
    }

    /// Validate a single input value
    pub async fn validate_input(&self, request: &ValidationRequest) -> SecurityResult<ValidationResult> {
        let rules = self.validation_rules.read().await;
        
        if let Some(rule) = rules.get(&request.field_name) {
            self.validate_against_rule(&request.value, rule, &request.field_name).await
        } else {
            // No specific rule, apply basic validation
            self.validate_basic(&request.value, &request.field_name).await
        }
    }

    /// Validate multiple inputs
    pub async fn validate_inputs(&self, requests: &[ValidationRequest]) -> SecurityResult<HashMap<String, ValidationResult>> {
        let mut results = HashMap::new();
        
        for request in requests {
            let result = self.validate_input(request).await?;
            results.insert(request.field_name.clone(), result);
        }

        Ok(results)
    }

    /// Sanitize a string input to prevent injection attacks
    pub async fn sanitize_string(&self, input: &str) -> String {
        let mut sanitized = input.to_string();

        // Remove null bytes
        sanitized = sanitized.replace('\0', "");

        // Escape dangerous characters for command injection
        sanitized = sanitized.replace("$(", "\\$(")
                           .replace("`", "\\`")
                           .replace("|", "\\|")
                           .replace("&", "\\&")
                           .replace(";", "\\;");

        // Remove or escape HTML/XML tags
        sanitized = self.escape_html(&sanitized);

        // Normalize Unicode
        sanitized = self.normalize_unicode(&sanitized);

        sanitized
    }

    /// Validate and sanitize a file path
    pub async fn validate_path(&self, path_str: &str) -> SecurityResult<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for path traversal attempts
        if path_str.contains("..") {
            errors.push(ValidationError {
                field: "path".to_string(),
                error_type: "path_traversal".to_string(),
                message: "Path traversal attempt detected".to_string(),
                severity: ValidationSeverity::High,
            });
        }

        // Check for absolute paths when not allowed
        let path = Path::new(path_str);
        if path.is_absolute() {
            warnings.push("Absolute path detected".to_string());
        }

        // Check for dangerous path components
        let dangerous_components = ["/dev", "/proc", "/sys", "\\System32", "\\Windows"];
        for component in &dangerous_components {
            if path_str.contains(component) {
                errors.push(ValidationError {
                    field: "path".to_string(),
                    error_type: "dangerous_path".to_string(),
                    message: format!("Access to dangerous path component: {}", component),
                    severity: ValidationSeverity::Critical,
                });
            }
        }

        // Normalize path
        let normalized = self.normalize_path(path)?;
        
        Ok(ValidationResult {
            valid: errors.is_empty(),
            sanitized_value: Some(serde_json::json!(normalized.to_string_lossy().to_string())),
            errors,
            warnings,
        })
    }

    /// Validate a command and its arguments
    pub async fn validate_command(&self, command: &str, args: &[String]) -> SecurityResult<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check command against allowlist
        if !self.is_command_allowed(command).await {
            errors.push(ValidationError {
                field: "command".to_string(),
                error_type: "forbidden_command".to_string(),
                message: format!("Command not allowed: {}", command),
                severity: ValidationSeverity::Critical,
            });
        }

        // Validate arguments
        for (i, arg) in args.iter().enumerate() {
            // Check for injection attempts
            if self.contains_injection_patterns(arg) {
                errors.push(ValidationError {
                    field: format!("arg_{}", i),
                    error_type: "injection_attempt".to_string(),
                    message: format!("Potential injection in argument: {}", arg),
                    severity: ValidationSeverity::High,
                });
            }

            // Check argument length
            if arg.len() > 1000 {
                warnings.push(format!("Very long argument at position {}", i));
            }
        }

        // Check total argument count
        if args.len() > crate::security::constants::MAX_COMMAND_ARGS {
            errors.push(ValidationError {
                field: "args".to_string(),
                error_type: "too_many_args".to_string(),
                message: format!("Too many arguments: {} (max: {})", args.len(), crate::security::constants::MAX_COMMAND_ARGS),
                severity: ValidationSeverity::Medium,
            });
        }

        Ok(ValidationResult {
            valid: errors.is_empty(),
            sanitized_value: Some(serde_json::json!({
                "command": command,
                "args": args
            })),
            errors,
            warnings,
        })
    }

    /// Validate JSON data structure
    pub async fn validate_json(&self, json_str: &str, schema_name: Option<&str>) -> SecurityResult<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Parse JSON
        let parsed_json = match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(json) => json,
            Err(e) => {
                errors.push(ValidationError {
                    field: "json".to_string(),
                    error_type: "invalid_json".to_string(),
                    message: format!("Invalid JSON: {}", e),
                    severity: ValidationSeverity::High,
                });
                return Ok(ValidationResult {
                    valid: false,
                    sanitized_value: None,
                    errors,
                    warnings,
                });
            }
        };

        // Check JSON size
        if json_str.len() > crate::security::constants::MAX_INPUT_LENGTH {
            errors.push(ValidationError {
                field: "json".to_string(),
                error_type: "size_limit".to_string(),
                message: "JSON exceeds maximum size limit".to_string(),
                severity: ValidationSeverity::Medium,
            });
        }

        // Validate against schema if provided
        if let Some(schema) = schema_name {
            self.validate_json_schema(&parsed_json, schema, &mut errors).await?;
        }

        // Check for potentially dangerous content
        self.scan_json_for_threats(&parsed_json, &mut warnings);

        Ok(ValidationResult {
            valid: errors.is_empty(),
            sanitized_value: Some(parsed_json),
            errors,
            warnings,
        })
    }

    /// Validate email format
    pub async fn validate_email(&self, email: &str) -> ValidationResult {
        let mut errors = Vec::new();
        let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();

        if !email_regex.is_match(email) {
            errors.push(ValidationError {
                field: "email".to_string(),
                error_type: "invalid_format".to_string(),
                message: "Invalid email format".to_string(),
                severity: ValidationSeverity::Medium,
            });
        }

        // Check for length limits
        if email.len() > 254 {
            errors.push(ValidationError {
                field: "email".to_string(),
                error_type: "too_long".to_string(),
                message: "Email address too long".to_string(),
                severity: ValidationSeverity::Medium,
            });
        }

        ValidationResult {
            valid: errors.is_empty(),
            sanitized_value: Some(serde_json::json!(email.to_lowercase().trim())),
            errors,
            warnings: Vec::new(),
        }
    }

    /// Validate URL format and safety
    pub async fn validate_url(&self, url_str: &str) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        match Url::parse(url_str) {
            Ok(url) => {
                // Check allowed schemes
                let allowed_schemes = ["http", "https", "ftp", "ftps"];
                if !allowed_schemes.contains(&url.scheme()) {
                    errors.push(ValidationError {
                        field: "url".to_string(),
                        error_type: "forbidden_scheme".to_string(),
                        message: format!("URL scheme not allowed: {}", url.scheme()),
                        severity: ValidationSeverity::High,
                    });
                }

                // Check for localhost/private IP access
                if let Some(host) = url.host_str() {
                    if self.is_private_host(host) {
                        warnings.push("URL points to private/localhost address".to_string());
                    }
                }

                // Check for suspicious URLs
                if self.is_suspicious_url(&url) {
                    warnings.push("URL appears suspicious".to_string());
                }
            }
            Err(e) => {
                errors.push(ValidationError {
                    field: "url".to_string(),
                    error_type: "invalid_url".to_string(),
                    message: format!("Invalid URL: {}", e),
                    severity: ValidationSeverity::Medium,
                });
            }
        }

        ValidationResult {
            valid: errors.is_empty(),
            sanitized_value: Some(serde_json::json!(url_str)),
            errors,
            warnings,
        }
    }

    // Private helper methods

    async fn validate_against_rule(&self, value: &serde_json::Value, rule: &ValidationRule, field_name: &str) -> SecurityResult<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check if required
        if rule.required && (value.is_null() || (value.is_string() && value.as_str().unwrap().is_empty())) {
            errors.push(ValidationError {
                field: field_name.to_string(),
                error_type: "required".to_string(),
                message: format!("Field {} is required", field_name),
                severity: ValidationSeverity::High,
            });
        }

        // Type validation
        match rule.rule_type {
            ValidationType::String => {
                if let Some(s) = value.as_str() {
                    if let Some(min_len) = rule.min_length {
                        if s.len() < min_len {
                            errors.push(ValidationError {
                                field: field_name.to_string(),
                                error_type: "too_short".to_string(),
                                message: format!("String too short: {} < {}", s.len(), min_len),
                                severity: ValidationSeverity::Medium,
                            });
                        }
                    }

                    if let Some(max_len) = rule.max_length {
                        if s.len() > max_len {
                            errors.push(ValidationError {
                                field: field_name.to_string(),
                                error_type: "too_long".to_string(),
                                message: format!("String too long: {} > {}", s.len(), max_len),
                                severity: ValidationSeverity::Medium,
                            });
                        }
                    }
                } else if !value.is_null() {
                    errors.push(ValidationError {
                        field: field_name.to_string(),
                        error_type: "wrong_type".to_string(),
                        message: "Expected string".to_string(),
                        severity: ValidationSeverity::Medium,
                    });
                }
            }
            ValidationType::Email => {
                if let Some(email) = value.as_str() {
                    let email_result = self.validate_email(email).await;
                    errors.extend(email_result.errors);
                    warnings.extend(email_result.warnings);
                }
            }
            ValidationType::Url => {
                if let Some(url) = value.as_str() {
                    let url_result = self.validate_url(url).await;
                    errors.extend(url_result.errors);
                    warnings.extend(url_result.warnings);
                }
            }
            // Add other type validations...
            _ => {}
        }

        // Pattern validation
        if let Some(pattern) = &rule.pattern {
            if let Some(s) = value.as_str() {
                let compiled_patterns = self.compiled_patterns.read().await;
                if let Some(regex) = compiled_patterns.get(pattern) {
                    if !regex.is_match(s) {
                        errors.push(ValidationError {
                            field: field_name.to_string(),
                            error_type: "pattern_mismatch".to_string(),
                            message: "Value doesn't match required pattern".to_string(),
                            severity: ValidationSeverity::Medium,
                        });
                    }
                }
            }
        }

        // Allowed values validation
        if let Some(allowed) = &rule.allowed_values {
            if let Some(s) = value.as_str() {
                if !allowed.contains(&s.to_string()) {
                    errors.push(ValidationError {
                        field: field_name.to_string(),
                        error_type: "invalid_value".to_string(),
                        message: "Value not in allowed list".to_string(),
                        severity: ValidationSeverity::Medium,
                    });
                }
            }
        }

        Ok(ValidationResult {
            valid: errors.is_empty(),
            sanitized_value: Some(value.clone()),
            errors,
            warnings,
        })
    }

    async fn validate_basic(&self, value: &serde_json::Value, field_name: &str) -> SecurityResult<ValidationResult> {
        let mut warnings = Vec::new();

        // Basic sanitization for strings
        let sanitized_value = if let Some(s) = value.as_str() {
            let sanitized = self.sanitize_string(s).await;
            if sanitized != s {
                warnings.push("Input was sanitized".to_string());
            }
            Some(serde_json::json!(sanitized))
        } else {
            Some(value.clone())
        };

        Ok(ValidationResult {
            valid: true,
            sanitized_value,
            errors: Vec::new(),
            warnings,
        })
    }

    async fn create_default_validation_rules(&self) -> SecurityResult<()> {
        let mut rules = self.validation_rules.write().await;
        let mut patterns = self.compiled_patterns.write().await;

        // Campaign name validation
        rules.insert("campaign_name".to_string(), ValidationRule {
            name: "Campaign Name".to_string(),
            rule_type: ValidationType::String,
            required: true,
            min_length: Some(1),
            max_length: Some(100),
            pattern: Some("^[a-zA-Z0-9\\s\\-_]+$".to_string()),
            allowed_values: None,
            custom_validator: None,
        });

        patterns.insert("^[a-zA-Z0-9\\s\\-_]+$".to_string(), 
            Regex::new(r"^[a-zA-Z0-9\s\-_]+$").unwrap());

        // Email validation
        rules.insert("email".to_string(), ValidationRule {
            name: "Email Address".to_string(),
            rule_type: ValidationType::Email,
            required: false,
            min_length: None,
            max_length: Some(254),
            pattern: None,
            allowed_values: None,
            custom_validator: None,
        });

        // File path validation
        rules.insert("file_path".to_string(), ValidationRule {
            name: "File Path".to_string(),
            rule_type: ValidationType::Path,
            required: false,
            min_length: None,
            max_length: Some(4096),
            pattern: None,
            allowed_values: None,
            custom_validator: None,
        });

        Ok(())
    }

    fn escape_html(&self, input: &str) -> String {
        input.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace("\"", "&quot;")
             .replace("'", "&#x27;")
    }

    fn normalize_unicode(&self, input: &str) -> String {
        // Basic Unicode normalization (NFC)
        // In a real implementation, use proper Unicode normalization
        input.to_string()
    }

    fn normalize_path(&self, path: &Path) -> SecurityResult<PathBuf> {
        let mut normalized = PathBuf::new();
        
        for component in path.components() {
            match component {
                std::path::Component::Normal(name) => normalized.push(name),
                std::path::Component::RootDir => normalized.push("/"),
                _ => continue, // Skip .. and . components
            }
        }

        Ok(normalized)
    }

    async fn is_command_allowed(&self, command: &str) -> bool {
        // Check against allowlist of safe commands from config
        let command_name = Path::new(command)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(command);

        self.config.allowed_commands.contains(&command_name.to_string())
    }

    fn contains_injection_patterns(&self, input: &str) -> bool {
        let dangerous_patterns = [
            "$(", "`", "|", "&", ";", "||", "&&",
            "../", "..\\", "/dev/", "\\System32\\",
        ];

        dangerous_patterns.iter().any(|pattern| input.contains(pattern))
    }

    async fn validate_json_schema(&self, _json: &serde_json::Value, _schema: &str, _errors: &mut Vec<ValidationError>) -> SecurityResult<()> {
        // JSON Schema validation would be implemented here
        // For now, this is a placeholder
        Ok(())
    }

    fn scan_json_for_threats(&self, json: &serde_json::Value, warnings: &mut Vec<String>) {
        match json {
            serde_json::Value::String(s) => {
                if self.contains_injection_patterns(s) {
                    warnings.push("Potential injection pattern in JSON string".to_string());
                }
            }
            serde_json::Value::Object(map) => {
                for (key, value) in map {
                    if key.contains("password") || key.contains("secret") {
                        warnings.push("Sensitive field detected in JSON".to_string());
                    }
                    self.scan_json_for_threats(value, warnings);
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.scan_json_for_threats(item, warnings);
                }
            }
            _ => {}
        }
    }

    fn is_private_host(&self, host: &str) -> bool {
        host == "localhost" || 
        host.starts_with("127.") ||
        host.starts_with("192.168.") ||
        host.starts_with("10.") ||
        host.starts_with("172.")
    }

    fn is_suspicious_url(&self, url: &Url) -> bool {
        let host = url.host_str().unwrap_or("");
        
        // Check for suspicious patterns
        host.contains("bit.ly") || // URL shorteners
        host.contains("tinyurl") ||
        url.as_str().contains("..") || // Path traversal
        url.as_str().len() > 2000 // Very long URLs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_string_sanitization() {
        let config = SecurityConfig::default();
        let validator = InputValidator::new(&config).unwrap();
        validator.initialize().await.unwrap();

        let malicious_input = "$(cat /etc/passwd) && rm -rf /";
        let sanitized = validator.sanitize_string(malicious_input).await;
        
        assert!(!sanitized.contains("$("));
        assert!(!sanitized.contains(" && "));
    }

    #[tokio::test]
    async fn test_path_validation() {
        let config = SecurityConfig::default();
        let validator = InputValidator::new(&config).unwrap();
        validator.initialize().await.unwrap();

        // Test path traversal
        let result = validator.validate_path("../../../etc/passwd").await.unwrap();
        assert!(!result.valid);
        assert!(!result.errors.is_empty());

        // Test safe path
        let result = validator.validate_path("documents/campaign.json").await.unwrap();
        assert!(result.valid);
    }

    #[tokio::test]
    async fn test_command_validation() {
        let config = SecurityConfig::default();
        let validator = InputValidator::new(&config).unwrap();
        validator.initialize().await.unwrap();

        // Test dangerous command
        let result = validator.validate_command("rm", &["-rf".to_string(), "/".to_string()]).await.unwrap();
        assert!(!result.valid);

        // Test safe command
        let result = validator.validate_command("python", &["script.py".to_string()]).await.unwrap();
        assert!(result.valid);
    }

    #[tokio::test]
    async fn test_email_validation() {
        let config = SecurityConfig::default();
        let validator = InputValidator::new(&config).unwrap();
        validator.initialize().await.unwrap();

        let result = validator.validate_email("test@example.com").await;
        assert!(result.valid);

        let result = validator.validate_email("invalid-email").await;
        assert!(!result.valid);
    }
}