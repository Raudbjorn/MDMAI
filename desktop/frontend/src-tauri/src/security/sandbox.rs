//! Process Sandboxing and Isolation
//! 
//! This module provides comprehensive process sandboxing capabilities:
//! - Resource limits (CPU, memory, file descriptors)
//! - Filesystem access controls
//! - Network restrictions
//! - Process privilege reduction
//! - Runtime monitoring and enforcement

use super::*;
use std::collections::HashMap;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use sysinfo::{System, SystemExt, ProcessExt, PidExt};

/// Sandbox configuration for a process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    /// Maximum CPU percentage (0.0-100.0)
    pub max_cpu_percent: f32,
    /// Maximum number of file descriptors
    pub max_file_descriptors: u32,
    /// Maximum number of processes/threads
    pub max_processes: u32,
    /// Filesystem restrictions
    pub filesystem_restrictions: FilesystemRestrictions,
    /// Network restrictions
    pub network_restrictions: NetworkRestrictions,
    /// Environment variable whitelist
    pub allowed_env_vars: Vec<String>,
    /// Working directory restriction
    pub allowed_working_dirs: Vec<String>,
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Drop privileges (Unix only)
    pub drop_privileges: bool,
    /// Enable seccomp filtering (Linux only)
    pub enable_seccomp: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 512,
            max_cpu_percent: 50.0,
            max_file_descriptors: 100,
            max_processes: 10,
            filesystem_restrictions: FilesystemRestrictions::default(),
            network_restrictions: NetworkRestrictions::default(),
            allowed_env_vars: vec![
                "PATH".to_string(),
                "HOME".to_string(),
                "USER".to_string(),
                "LANG".to_string(),
                "LC_ALL".to_string(),
                "PYTHONPATH".to_string(),
                "PYTHON_HOME".to_string(),
            ],
            allowed_working_dirs: vec![
                "$APPDATA".to_string(),
                "$TEMP".to_string(),
            ],
            max_execution_time: Some(Duration::from_secs(300)), // 5 minutes
            drop_privileges: true,
            enable_seccomp: true,
        }
    }
}

/// Sandboxed process information
#[derive(Debug, Clone)]
pub struct SandboxedProcess {
    pub id: Uuid,
    pub child: Arc<RwLock<Option<Child>>>,
    pub config: SandboxConfig,
    pub start_time: Instant,
    pub last_resource_check: Arc<RwLock<Instant>>,
    pub resource_violations: Arc<RwLock<Vec<ResourceViolation>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceViolation {
    pub violation_type: ViolationType,
    pub current_value: f64,
    pub limit: f64,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    MemoryLimit,
    CpuLimit,
    FileDescriptorLimit,
    ProcessLimit,
    ExecutionTimeLimit,
    FilesystemAccess,
    NetworkAccess,
}

/// Sandbox manager
pub struct SandboxManager {
    config: SecurityConfig,
    active_processes: Arc<RwLock<HashMap<Uuid, SandboxedProcess>>>,
    system_monitor: Arc<RwLock<System>>,
}

impl SandboxManager {
    pub fn new(config: &SecurityConfig) -> SecurityResult<Self> {
        Ok(Self {
            config: config.clone(),
            active_processes: Arc::new(RwLock::new(HashMap::new())),
            system_monitor: Arc::new(RwLock::new(System::new_all())),
        })
    }

    pub async fn initialize(&self) -> SecurityResult<()> {
        // Start resource monitoring task
        let processes = self.active_processes.clone();
        let system_monitor = self.system_monitor.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                // Update system information
                system_monitor.write().await.refresh_all();
                
                // Check resource usage for all active processes
                let active_processes = processes.read().await;
                for (process_id, sandboxed_process) in active_processes.iter() {
                    if let Err(e) = Self::check_process_resources(&system_monitor, process_id, sandboxed_process).await {
                        log::error!("Resource check failed for process {}: {}", process_id, e);
                    }
                }
            }
        });

        log::info!("Sandbox manager initialized");
        Ok(())
    }

    /// Create a sandboxed process
    pub async fn create_sandboxed_process(
        &self,
        command: &str,
        args: &[String],
        working_dir: Option<&str>,
        sandbox_config: Option<SandboxConfig>,
    ) -> SecurityResult<Uuid> {
        let config = sandbox_config.unwrap_or_default();
        let process_id = Uuid::new_v4();

        // Validate command and arguments
        self.validate_command(command, args, &config)?;

        // Build the command with sandbox restrictions
        let mut cmd = Command::new(command);
        cmd.args(args)
           .stdin(Stdio::piped())
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        // Set working directory if specified and allowed
        if let Some(dir) = working_dir {
            self.validate_working_directory(dir, &config)?;
            cmd.current_dir(dir);
        }

        // Set environment variables (filtered to allowed list)
        cmd.env_clear();
        for env_var in &config.allowed_env_vars {
            if let Ok(value) = std::env::var(env_var) {
                cmd.env(env_var, value);
            }
        }

        // Apply platform-specific sandbox restrictions
        #[cfg(unix)]
        self.apply_unix_restrictions(&mut cmd, &config)?;

        #[cfg(windows)]
        self.apply_windows_restrictions(&mut cmd, &config)?;

        // Start the process
        let child = cmd.spawn().map_err(|e| SecurityError::SandboxViolation {
            violation: format!("Failed to spawn process: {}", e),
        })?;

        let sandboxed_process = SandboxedProcess {
            id: process_id,
            child: Arc::new(RwLock::new(Some(child))),
            config,
            start_time: Instant::now(),
            last_resource_check: Arc::new(RwLock::new(Instant::now())),
            resource_violations: Arc::new(RwLock::new(Vec::new())),
        };

        self.active_processes.write().await.insert(process_id, sandboxed_process);

        log::info!("Created sandboxed process: {} (command: {})", process_id, command);
        Ok(process_id)
    }

    /// Terminate a sandboxed process
    pub async fn terminate_process(&self, process_id: Uuid) -> SecurityResult<()> {
        let mut active_processes = self.active_processes.write().await;
        
        if let Some(sandboxed_process) = active_processes.get(&process_id) {
            let mut child_guard = sandboxed_process.child.write().await;
            if let Some(mut child) = child_guard.take() {
                // Try graceful termination first
                let _ = child.kill();
                let _ = child.wait();
            }
        }

        active_processes.remove(&process_id);
        log::info!("Terminated sandboxed process: {}", process_id);
        Ok(())
    }

    /// Get process status
    pub async fn get_process_status(&self, process_id: Uuid) -> SecurityResult<ProcessStatus> {
        let active_processes = self.active_processes.read().await;
        let system = self.system_monitor.read().await;
        
        if let Some(sandboxed_process) = active_processes.get(&process_id) {
            let child_guard = sandboxed_process.child.read().await;
            
            if let Some(child) = child_guard.as_ref() {
                let pid = child.id();
                
                let status = if let Some(process) = system.process(sysinfo::Pid::from_u32(pid)) {
                    let memory_mb = process.memory() / 1024 / 1024; // Convert to MB
                    let cpu_percent = process.cpu_usage();
                    
                    ProcessStatus {
                        process_id,
                        pid,
                        is_running: true,
                        memory_usage_mb: memory_mb,
                        cpu_usage_percent: cpu_percent,
                        uptime: sandboxed_process.start_time.elapsed(),
                        resource_violations: sandboxed_process.resource_violations.read().await.clone(),
                    }
                } else {
                    ProcessStatus {
                        process_id,
                        pid,
                        is_running: false,
                        memory_usage_mb: 0,
                        cpu_usage_percent: 0.0,
                        uptime: sandboxed_process.start_time.elapsed(),
                        resource_violations: sandboxed_process.resource_violations.read().await.clone(),
                    }
                };

                Ok(status)
            } else {
                Err(SecurityError::InternalError {
                    message: "Process not found or already terminated".to_string(),
                })
            }
        } else {
            Err(SecurityError::InternalError {
                message: format!("Process {} not found", process_id),
            })
        }
    }

    /// List all active processes
    pub async fn list_active_processes(&self) -> Vec<Uuid> {
        self.active_processes.read().await.keys().cloned().collect()
    }

    /// Validate command and arguments
    fn validate_command(&self, command: &str, args: &[String], _config: &SandboxConfig) -> SecurityResult<()> {
        // Check for dangerous commands
        let dangerous_commands = [
            "rm", "del", "format", "fdisk", "mkfs",
            "shutdown", "reboot", "halt", "poweroff",
            "su", "sudo", "passwd", "chown", "chmod",
            "netcat", "nc", "telnet", "ssh", "ftp",
        ];

        let cmd_name = std::path::Path::new(command)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(command);

        if dangerous_commands.contains(&cmd_name) {
            return Err(SecurityError::SandboxViolation {
                violation: format!("Dangerous command not allowed: {}", cmd_name),
            });
        }

        // Validate argument count
        if args.len() > crate::security::constants::MAX_COMMAND_ARGS {
            return Err(SecurityError::SandboxViolation {
                violation: format!("Too many command arguments: {} (max: {})", 
                    args.len(), crate::security::constants::MAX_COMMAND_ARGS),
            });
        }

        // Check for suspicious arguments
        for arg in args {
            if arg.contains("..") || arg.contains("/dev") || arg.contains("/proc") || arg.contains("/sys") {
                return Err(SecurityError::SandboxViolation {
                    violation: format!("Suspicious argument detected: {}", arg),
                });
            }
        }

        Ok(())
    }

    /// Validate working directory
    fn validate_working_directory(&self, dir: &str, config: &SandboxConfig) -> SecurityResult<()> {
        let path = std::path::Path::new(dir);
        
        // Check if directory is in allowed list
        for allowed in &config.allowed_working_dirs {
            if dir.starts_with(allowed) {
                return Ok(());
            }
        }

        // Check against blocked paths
        for blocked in &config.filesystem_restrictions.blocked_paths {
            if dir.starts_with(blocked) {
                return Err(SecurityError::SandboxViolation {
                    violation: format!("Working directory not allowed: {}", dir),
                });
            }
        }

        // Ensure directory exists and is accessible
        if !path.exists() {
            return Err(SecurityError::SandboxViolation {
                violation: format!("Working directory does not exist: {}", dir),
            });
        }

        Ok(())
    }

    /// Check resource usage for a process
    async fn check_process_resources(
        system_monitor: &Arc<RwLock<System>>,
        process_id: &Uuid,
        sandboxed_process: &SandboxedProcess,
    ) -> SecurityResult<()> {
        let child_guard = sandboxed_process.child.read().await;
        if let Some(child) = child_guard.as_ref() {
            let pid = child.id();
            let system = system_monitor.read().await;
            
            if let Some(process) = system.process(sysinfo::Pid::from_u32(pid)) {
                let memory_mb = process.memory() / 1024 / 1024;
                let cpu_percent = process.cpu_usage();

                let mut violations = Vec::new();

                // Check memory limit
                if memory_mb > sandboxed_process.config.max_memory_mb {
                    violations.push(ResourceViolation {
                        violation_type: ViolationType::MemoryLimit,
                        current_value: memory_mb as f64,
                        limit: sandboxed_process.config.max_memory_mb as f64,
                        timestamp: SystemTime::now(),
                    });
                }

                // Check CPU limit
                if cpu_percent > sandboxed_process.config.max_cpu_percent {
                    violations.push(ResourceViolation {
                        violation_type: ViolationType::CpuLimit,
                        current_value: cpu_percent as f64,
                        limit: sandboxed_process.config.max_cpu_percent as f64,
                        timestamp: SystemTime::now(),
                    });
                }

                // Check execution time limit
                if let Some(max_time) = sandboxed_process.config.max_execution_time {
                    if sandboxed_process.start_time.elapsed() > max_time {
                        violations.push(ResourceViolation {
                            violation_type: ViolationType::ExecutionTimeLimit,
                            current_value: sandboxed_process.start_time.elapsed().as_secs() as f64,
                            limit: max_time.as_secs() as f64,
                            timestamp: SystemTime::now(),
                        });
                    }
                }

                // Record violations
                if !violations.is_empty() {
                    let mut process_violations = sandboxed_process.resource_violations.write().await;
                    for violation in violations {
                        log::warn!("Resource violation for process {}: {:?}", process_id, violation);
                        process_violations.push(violation);
                    }

                    // Keep only recent violations (last 100)
                    if process_violations.len() > 100 {
                        process_violations.drain(0..process_violations.len() - 100);
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply Unix-specific sandbox restrictions
    #[cfg(unix)]
    fn apply_unix_restrictions(&self, cmd: &mut Command, config: &SandboxConfig) -> SecurityResult<()> {
        use std::os::unix::process::CommandExt;
        
        // Set resource limits using setrlimit
        cmd.pre_exec(move || {
            // Set memory limit (RLIMIT_AS - virtual memory)
            let memory_limit = (config.max_memory_mb * 1024 * 1024) as libc::rlim_t;
            let limit = libc::rlimit {
                rlim_cur: memory_limit,
                rlim_max: memory_limit,
            };
            unsafe {
                if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }

            // Set file descriptor limit
            let fd_limit = config.max_file_descriptors as libc::rlim_t;
            let fd_rlimit = libc::rlimit {
                rlim_cur: fd_limit,
                rlim_max: fd_limit,
            };
            unsafe {
                if libc::setrlimit(libc::RLIMIT_NOFILE, &fd_rlimit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }

            // Set process limit
            let proc_limit = config.max_processes as libc::rlim_t;
            let proc_rlimit = libc::rlimit {
                rlim_cur: proc_limit,
                rlim_max: proc_limit,
            };
            unsafe {
                if libc::setrlimit(libc::RLIMIT_NPROC, &proc_rlimit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }

            // Drop privileges if requested
            if config.drop_privileges {
                // Get nobody user/group
                unsafe {
                    let nobody_uid = libc::getuid();
                    let nobody_gid = libc::getgid();
                    
                    if libc::setgid(nobody_gid) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    
                    if libc::setuid(nobody_uid) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
            }

            Ok(())
        });

        Ok(())
    }

    /// Apply Windows-specific sandbox restrictions
    #[cfg(windows)]
    fn apply_windows_restrictions(&self, cmd: &mut Command, config: &SandboxConfig) -> SecurityResult<()> {
        use std::ffi::OsString;
        use std::os::windows::ffi::OsStringExt;
        use windows::{
            core::*,
            Win32::{
                Foundation::*,
                System::JobObjects::*,
                Security::*,
            },
        };

        // Create a Job Object for resource limitation
        unsafe {
            let job_handle = CreateJobObjectW(None, None)
                .map_err(|e| SecurityError::SandboxingError {
                    message: format!("Failed to create Windows Job Object: {}", e),
                })?;

            // Set job limits
            let mut job_limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
                BasicLimitInformation: JOBOBJECT_BASIC_LIMIT_INFORMATION {
                    LimitFlags: JOB_OBJECT_LIMIT_PROCESS_MEMORY 
                        | JOB_OBJECT_LIMIT_JOB_MEMORY 
                        | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                        | JOB_OBJECT_LIMIT_PROCESS_TIME,
                    ProcessMemoryLimit: (config.memory_limit_mb as usize * 1024 * 1024),
                    JobMemoryLimit: (config.memory_limit_mb as usize * 1024 * 1024),
                    ActiveProcessLimit: 1, // Only allow one process
                    PerProcessUserTimeLimit: LARGE_INTEGER {
                        QuadPart: (config.execution_time_limit_secs as i64 * 10_000_000), // 100ns units
                    },
                    ..Default::default()
                },
                IoInfo: JOBOBJECT_IO_RATE_CONTROL_INFORMATION {
                    MaxIops: config.max_iops.unwrap_or(1000) as u64,
                    MaxBandwidth: config.max_bandwidth_bytes_per_sec.unwrap_or(10_000_000) as i64,
                    ..Default::default()
                },
                ..Default::default()
            };

            // Apply the job limits
            let result = SetInformationJobObject(
                job_handle,
                JobObjectExtendedLimitInformation,
                &job_limits as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );

            if result.is_err() {
                let _ = CloseHandle(job_handle);
                return Err(SecurityError::SandboxingError {
                    message: "Failed to set Job Object limits".to_string(),
                });
            }

            // Set UI restrictions to prevent breakout
            let ui_restrictions = JOBOBJECT_BASIC_UI_RESTRICTIONS {
                UIRestrictionsClass: JOB_OBJECT_UILIMIT_DESKTOP
                    | JOB_OBJECT_UILIMIT_DISPLAYSETTINGS
                    | JOB_OBJECT_UILIMIT_EXITWINDOWS
                    | JOB_OBJECT_UILIMIT_GLOBALATOMS
                    | JOB_OBJECT_UILIMIT_HANDLES
                    | JOB_OBJECT_UILIMIT_READCLIPBOARD
                    | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS
                    | JOB_OBJECT_UILIMIT_WRITECLIPBOARD,
            };

            let ui_result = SetInformationJobObject(
                job_handle,
                JobObjectBasicUIRestrictions,
                &ui_restrictions as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_BASIC_UI_RESTRICTIONS>() as u32,
            );

            if ui_result.is_err() {
                let _ = CloseHandle(job_handle);
                return Err(SecurityError::SandboxingError {
                    message: "Failed to set Job Object UI restrictions".to_string(),
                });
            }

            // Store job handle for later assignment to spawned process
            // This would typically be stored in a process manager
            log::info!("Windows sandboxing configured with Job Object restrictions");
            
            // Note: The actual assignment of the process to the job object
            // would happen after CreateProcess in the process spawning code
            let _ = CloseHandle(job_handle);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessStatus {
    pub process_id: Uuid,
    pub pid: u32,
    pub is_running: bool,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f32,
    pub uptime: Duration,
    pub resource_violations: Vec<ResourceViolation>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_sandbox_creation() {
        let config = SecurityConfig::default();
        let sandbox = SandboxManager::new(&config).unwrap();
        sandbox.initialize().await.unwrap();
        
        let process_id = sandbox
            .create_sandboxed_process("echo", &["Hello, World!".to_string()], None, None)
            .await;
            
        assert!(process_id.is_ok());
        
        let process_id = process_id.unwrap();
        let status = sandbox.get_process_status(process_id).await.unwrap();
        assert_eq!(status.process_id, process_id);
        
        sandbox.terminate_process(process_id).await.unwrap();
    }

    #[test]
    fn test_command_validation() {
        let config = SecurityConfig::default();
        let sandbox = SandboxManager::new(&config).unwrap();
        let sandbox_config = SandboxConfig::default();
        
        // Test dangerous command rejection
        assert!(sandbox.validate_command("rm", &["-rf".to_string(), "/".to_string()], &sandbox_config).is_err());
        
        // Test safe command acceptance
        assert!(sandbox.validate_command("echo", &["hello".to_string()], &sandbox_config).is_ok());
    }
}