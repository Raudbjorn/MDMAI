/*!
 * Benchmarking Module
 * 
 * Provides comprehensive performance benchmarking including startup benchmarks,
 * IPC latency tests, memory usage profiling, and regression detection.
 */

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use log::{info, debug, warn, error};
use sysinfo::{System, SystemExt, ProcessExt};

/// Benchmark test definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTest {
    pub name: String,
    pub description: String,
    pub category: TestCategory,
    pub warmup_iterations: u32,
    pub test_iterations: u32,
    pub timeout_seconds: u64,
    pub expected_range: Option<PerformanceRange>,
    pub enabled: bool,
}

/// Test categories for organization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestCategory {
    Startup,
    IpcLatency,
    MemoryUsage,
    FileSystem,
    Database,
    Network,
    Security,
    UserInterface,
    Background,
}

/// Expected performance range for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRange {
    pub min_ms: f64,
    pub max_ms: f64,
    pub target_ms: f64,
    pub memory_limit_mb: Option<u32>,
}

/// Benchmark result for a single test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub category: TestCategory,
    pub iterations: u32,
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub median_duration: Duration,
    pub p95_duration: Duration,
    pub p99_duration: Duration,
    pub standard_deviation: f64,
    pub memory_usage: MemoryMetrics,
    pub cpu_usage: CpuMetrics,
    pub success_rate: f64,
    pub errors: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub baseline_comparison: Option<BaselineComparison>,
}

/// Memory usage metrics during benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub initial_mb: u32,
    pub peak_mb: u32,
    pub final_mb: u32,
    pub allocated_mb: u32,
    pub average_mb: u32,
}

/// CPU usage metrics during benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub initial_percent: f32,
    pub peak_percent: f32,
    pub average_percent: f32,
    pub user_time_ms: u64,
    pub system_time_ms: u64,
}

/// Comparison with baseline performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub baseline_duration: Duration,
    pub current_duration: Duration,
    pub performance_change_percent: f64,
    pub regression_detected: bool,
    pub improvement_detected: bool,
}

/// Complete benchmark suite result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteResult {
    pub suite_name: String,
    pub results: Vec<BenchmarkResult>,
    pub total_duration: Duration,
    pub passed_tests: u32,
    pub failed_tests: u32,
    pub success_rate: f64,
    pub performance_summary: PerformanceSummary,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Overall performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub startup_time_ms: f64,
    pub average_ipc_latency_ms: f64,
    pub memory_efficiency_score: f64,
    pub overall_score: f64,
    pub regressions_detected: u32,
    pub improvements_detected: u32,
}

/// Benchmark execution context
pub struct BenchmarkContext {
    pub test: PerformanceTest,
    pub start_time: Instant,
    pub iterations_completed: u32,
    pub memory_samples: Vec<u32>,
    pub cpu_samples: Vec<f32>,
    pub durations: Vec<Duration>,
    pub errors: Vec<String>,
}

impl BenchmarkContext {
    pub fn new(test: PerformanceTest) -> Self {
        Self {
            test,
            start_time: Instant::now(),
            iterations_completed: 0,
            memory_samples: Vec::new(),
            cpu_samples: Vec::new(),
            durations: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Record a test iteration result
    pub fn record_iteration(&mut self, duration: Duration, memory_mb: u32, cpu_percent: f32) {
        self.iterations_completed += 1;
        self.durations.push(duration);
        self.memory_samples.push(memory_mb);
        self.cpu_samples.push(cpu_percent);
    }

    /// Record an error
    pub fn record_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Calculate final benchmark result
    pub fn finalize(self, baseline: Option<&BenchmarkResult>) -> BenchmarkResult {
        let mut durations = self.durations;
        durations.sort();

        let total_duration: Duration = durations.iter().sum();
        let average_duration = if durations.is_empty() { 
            Duration::ZERO 
        } else { 
            total_duration / durations.len() as u32 
        };

        let min_duration = durations.first().copied().unwrap_or(Duration::ZERO);
        let max_duration = durations.last().copied().unwrap_or(Duration::ZERO);
        
        let median_duration = if durations.is_empty() {
            Duration::ZERO
        } else {
            durations[durations.len() / 2]
        };

        let p95_index = ((durations.len() as f64) * 0.95) as usize;
        let p95_duration = durations.get(p95_index).copied().unwrap_or(Duration::ZERO);

        let p99_index = ((durations.len() as f64) * 0.99) as usize;
        let p99_duration = durations.get(p99_index).copied().unwrap_or(Duration::ZERO);

        // Calculate standard deviation
        let avg_ms = average_duration.as_millis() as f64;
        let variance: f64 = durations.iter()
            .map(|d| {
                let diff = d.as_millis() as f64 - avg_ms;
                diff * diff
            })
            .sum::<f64>() / durations.len() as f64;
        let standard_deviation = variance.sqrt();

        // Calculate memory metrics
        let memory_usage = MemoryMetrics {
            initial_mb: self.memory_samples.first().copied().unwrap_or(0),
            peak_mb: self.memory_samples.iter().max().copied().unwrap_or(0),
            final_mb: self.memory_samples.last().copied().unwrap_or(0),
            allocated_mb: self.memory_samples.iter().max().copied().unwrap_or(0) - 
                         self.memory_samples.first().copied().unwrap_or(0),
            average_mb: if self.memory_samples.is_empty() { 0 } else {
                (self.memory_samples.iter().sum::<u32>() / self.memory_samples.len() as u32)
            },
        };

        // Calculate CPU metrics
        let cpu_usage = CpuMetrics {
            initial_percent: self.cpu_samples.first().copied().unwrap_or(0.0),
            peak_percent: self.cpu_samples.iter().fold(0.0f32, |a, &b| a.max(b)),
            average_percent: if self.cpu_samples.is_empty() { 0.0 } else {
                self.cpu_samples.iter().sum::<f32>() / self.cpu_samples.len() as f32
            },
            user_time_ms: 0, // Would need actual process time tracking
            system_time_ms: 0, // Would need actual process time tracking
        };

        // Calculate success rate
        let total_attempts = self.test.test_iterations;
        let successful_attempts = self.iterations_completed;
        let success_rate = if total_attempts > 0 {
            successful_attempts as f64 / total_attempts as f64
        } else {
            0.0
        };

        // Compare with baseline if provided
        let baseline_comparison = baseline.map(|baseline_result| {
            let current_avg = average_duration.as_millis() as f64;
            let baseline_avg = baseline_result.average_duration.as_millis() as f64;
            let change_percent = ((current_avg - baseline_avg) / baseline_avg) * 100.0;
            
            BaselineComparison {
                baseline_duration: baseline_result.average_duration,
                current_duration: average_duration,
                performance_change_percent: change_percent,
                regression_detected: change_percent > 10.0, // 10% slower is a regression
                improvement_detected: change_percent < -5.0, // 5% faster is an improvement
            }
        });

        BenchmarkResult {
            test_name: self.test.name,
            category: self.test.category,
            iterations: self.iterations_completed,
            total_duration,
            average_duration,
            min_duration,
            max_duration,
            median_duration,
            p95_duration,
            p99_duration,
            standard_deviation,
            memory_usage,
            cpu_usage,
            success_rate,
            errors: self.errors,
            timestamp: chrono::Utc::now(),
            baseline_comparison,
        }
    }
}

/// Main benchmark suite manager
pub struct BenchmarkSuite {
    tests: Arc<RwLock<HashMap<String, PerformanceTest>>>,
    baselines: Arc<RwLock<HashMap<String, BenchmarkResult>>>,
    history: Arc<RwLock<VecDeque<BenchmarkSuiteResult>>>,
    max_history_size: usize,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            tests: Arc::new(RwLock::new(HashMap::new())),
            baselines: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(VecDeque::new())),
            max_history_size: 50, // Keep last 50 benchmark runs
        }
    }

    /// Register a performance test
    pub async fn register_test(&self, test: PerformanceTest) {
        debug!("Registering benchmark test: {}", test.name);
        self.tests.write().await.insert(test.name.clone(), test);
    }

    /// Register multiple tests
    pub async fn register_tests(&self, tests: Vec<PerformanceTest>) {
        for test in tests {
            self.register_test(test).await;
        }
    }

    /// Run a single benchmark test
    pub async fn run_test(&self, test_name: &str) -> Result<BenchmarkResult, String> {
        let test = {
            let tests = self.tests.read().await;
            tests.get(test_name).cloned()
                .ok_or_else(|| format!("Test '{}' not found", test_name))?
        };

        if !test.enabled {
            return Err(format!("Test '{}' is disabled", test_name));
        }

        info!("Running benchmark test: {}", test_name);
        let mut context = BenchmarkContext::new(test.clone());

        // Warmup iterations
        for _ in 0..test.warmup_iterations {
            if let Err(e) = self.execute_test_iteration(&test, false).await {
                warn!("Warmup iteration failed: {}", e);
            }
        }

        // Actual test iterations
        for i in 0..test.test_iterations {
            match self.execute_test_iteration(&test, true).await {
                Ok((duration, memory_mb, cpu_percent)) => {
                    context.record_iteration(duration, memory_mb, cpu_percent);
                }
                Err(e) => {
                    context.record_error(format!("Iteration {}: {}", i, e));
                }
            }

            // Check timeout
            if context.start_time.elapsed().as_secs() > test.timeout_seconds {
                context.record_error("Test timeout reached".to_string());
                break;
            }
        }

        // Get baseline for comparison
        let baseline = self.baselines.read().await.get(test_name).cloned();
        let result = context.finalize(baseline.as_ref());

        // Update baseline if this is a significant improvement
        if let Some(baseline_cmp) = &result.baseline_comparison {
            if baseline_cmp.improvement_detected && result.success_rate > 0.9 {
                debug!("Updating baseline for test '{}' due to improvement", test_name);
                self.baselines.write().await.insert(test_name.to_string(), result.clone());
            }
        } else {
            // Set initial baseline
            if result.success_rate > 0.9 {
                debug!("Setting initial baseline for test '{}'", test_name);
                self.baselines.write().await.insert(test_name.to_string(), result.clone());
            }
        }

        info!("Benchmark test '{}' completed: {:.2}ms avg", 
              test_name, result.average_duration.as_millis());

        Ok(result)
    }

    /// Execute a single test iteration
    async fn execute_test_iteration(&self, test: &PerformanceTest, measure: bool) -> Result<(Duration, u32, f32), String> {
        let start_time = Instant::now();
        let initial_memory = self.get_current_memory_usage();
        let initial_cpu = self.get_current_cpu_usage();

        // Execute the actual test based on category
        let result = match test.category {
            TestCategory::Startup => self.benchmark_startup().await,
            TestCategory::IpcLatency => self.benchmark_ipc_latency().await,
            TestCategory::MemoryUsage => self.benchmark_memory_usage().await,
            TestCategory::FileSystem => self.benchmark_file_system().await,
            TestCategory::Database => self.benchmark_database().await,
            TestCategory::Network => self.benchmark_network().await,
            TestCategory::Security => self.benchmark_security().await,
            TestCategory::UserInterface => self.benchmark_ui().await,
            TestCategory::Background => self.benchmark_background_tasks().await,
        };

        let duration = start_time.elapsed();
        let final_memory = self.get_current_memory_usage();
        let final_cpu = self.get_current_cpu_usage();

        result?;

        if measure {
            Ok((duration, final_memory, final_cpu))
        } else {
            Ok((Duration::ZERO, 0, 0.0))
        }
    }

    /// Get current memory usage in MB
    fn get_current_memory_usage(&self) -> u32 {
        let mut system = System::new();
        system.refresh_processes();
        
        if let Ok(current_pid) = sysinfo::get_current_pid() {
            if let Some(process) = system.process(current_pid) {
                return (process.memory() / 1024 / 1024) as u32;
            }
        }
        
        0
    }

    /// Get current CPU usage percentage
    fn get_current_cpu_usage(&self) -> f32 {
        let mut system = System::new();
        system.refresh_processes();
        
        if let Ok(current_pid) = sysinfo::get_current_pid() {
            if let Some(process) = system.process(current_pid) {
                return process.cpu_usage();
            }
        }
        
        0.0
    }

    /// Benchmark implementations for different categories
    async fn benchmark_startup(&self) -> Result<(), String> {
        // Simulate startup operations
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }

    async fn benchmark_ipc_latency(&self) -> Result<(), String> {
        // Simulate IPC call
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(())
    }

    async fn benchmark_memory_usage(&self) -> Result<(), String> {
        // Simulate memory intensive operations
        let _data: Vec<u8> = vec![0; 1024 * 1024]; // Allocate 1MB
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    async fn benchmark_file_system(&self) -> Result<(), String> {
        // Simulate file operations
        tokio::time::sleep(Duration::from_millis(20)).await;
        Ok(())
    }

    async fn benchmark_database(&self) -> Result<(), String> {
        // Simulate database operations
        tokio::time::sleep(Duration::from_millis(15)).await;
        Ok(())
    }

    async fn benchmark_network(&self) -> Result<(), String> {
        // Simulate network operations
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(())
    }

    async fn benchmark_security(&self) -> Result<(), String> {
        // Simulate security operations (hashing, encryption)
        tokio::time::sleep(Duration::from_millis(25)).await;
        Ok(())
    }

    async fn benchmark_ui(&self) -> Result<(), String> {
        // Simulate UI operations
        tokio::time::sleep(Duration::from_millis(8)).await;
        Ok(())
    }

    async fn benchmark_background_tasks(&self) -> Result<(), String> {
        // Simulate background processing
        tokio::time::sleep(Duration::from_millis(30)).await;
        Ok(())
    }

    /// Run the complete benchmark suite
    pub async fn run_full_suite(&self) -> Result<Vec<BenchmarkResult>, String> {
        info!("Running full benchmark suite");
        let start_time = Instant::now();

        let test_names: Vec<String> = {
            let tests = self.tests.read().await;
            tests.keys().cloned().collect()
        };

        let mut results = Vec::new();
        let mut passed = 0;
        let mut failed = 0;

        for test_name in test_names {
            match self.run_test(&test_name).await {
                Ok(result) => {
                    if result.success_rate > 0.5 {
                        passed += 1;
                    } else {
                        failed += 1;
                    }
                    results.push(result);
                }
                Err(e) => {
                    error!("Failed to run test '{}': {}", test_name, e);
                    failed += 1;
                }
            }
        }

        let total_duration = start_time.elapsed();
        let success_rate = if passed + failed > 0 {
            passed as f64 / (passed + failed) as f64
        } else {
            0.0
        };

        // Calculate performance summary
        let performance_summary = self.calculate_performance_summary(&results).await;

        let suite_result = BenchmarkSuiteResult {
            suite_name: "Full Performance Suite".to_string(),
            results: results.clone(),
            total_duration,
            passed_tests: passed,
            failed_tests: failed,
            success_rate,
            performance_summary,
            timestamp: chrono::Utc::now(),
        };

        // Add to history
        {
            let mut history = self.history.write().await;
            history.push_back(suite_result);
            
            // Maintain history size limit
            while history.len() > self.max_history_size {
                history.pop_front();
            }
        }

        info!("Benchmark suite completed: {}/{} tests passed in {:?}", 
              passed, passed + failed, total_duration);

        Ok(results)
    }

    /// Run startup-specific benchmarks
    pub async fn run_startup_benchmark(&self) -> Result<BenchmarkResult, String> {
        self.run_test("startup_performance").await
    }

    /// Calculate overall performance summary
    async fn calculate_performance_summary(&self, results: &[BenchmarkResult]) -> PerformanceSummary {
        let startup_time_ms = results.iter()
            .find(|r| r.category == TestCategory::Startup)
            .map(|r| r.average_duration.as_millis() as f64)
            .unwrap_or(0.0);

        let average_ipc_latency_ms = results.iter()
            .filter(|r| r.category == TestCategory::IpcLatency)
            .map(|r| r.average_duration.as_millis() as f64)
            .sum::<f64>() / results.iter()
            .filter(|r| r.category == TestCategory::IpcLatency)
            .count().max(1) as f64;

        // Calculate memory efficiency (inverse of memory usage)
        let memory_efficiency_score = results.iter()
            .map(|r| {
                let max_memory = r.memory_usage.peak_mb as f64;
                if max_memory > 0.0 { 100.0 / max_memory } else { 100.0 }
            })
            .sum::<f64>() / results.len().max(1) as f64;

        let regressions_detected = results.iter()
            .filter(|r| r.baseline_comparison.as_ref()
                .map_or(false, |bc| bc.regression_detected))
            .count() as u32;

        let improvements_detected = results.iter()
            .filter(|r| r.baseline_comparison.as_ref()
                .map_or(false, |bc| bc.improvement_detected))
            .count() as u32;

        // Calculate overall score (simplified scoring algorithm)
        let overall_score = {
            let startup_score = (2000.0 - startup_time_ms).max(0.0) / 2000.0 * 100.0;
            let latency_score = (100.0 - average_ipc_latency_ms).max(0.0) / 100.0 * 100.0;
            let memory_score = memory_efficiency_score.min(100.0);
            let regression_penalty = regressions_detected as f64 * 10.0;
            
            ((startup_score + latency_score + memory_score) / 3.0 - regression_penalty).max(0.0)
        };

        PerformanceSummary {
            startup_time_ms,
            average_ipc_latency_ms,
            memory_efficiency_score,
            overall_score,
            regressions_detected,
            improvements_detected,
        }
    }

    /// Get benchmark history
    pub async fn get_history(&self) -> Vec<BenchmarkSuiteResult> {
        self.history.read().await.iter().cloned().collect()
    }

    /// Clear all baselines
    pub async fn clear_baselines(&self) {
        self.baselines.write().await.clear();
        info!("Cleared all benchmark baselines");
    }

    /// Get current baselines
    pub async fn get_baselines(&self) -> HashMap<String, BenchmarkResult> {
        self.baselines.read().await.clone()
    }

    /// Register default benchmark tests
    pub async fn register_default_tests(&self) {
        let default_tests = vec![
            PerformanceTest {
                name: "startup_performance".to_string(),
                description: "Measures application startup time".to_string(),
                category: TestCategory::Startup,
                warmup_iterations: 2,
                test_iterations: 5,
                timeout_seconds: 30,
                expected_range: Some(PerformanceRange {
                    min_ms: 500.0,
                    max_ms: 2000.0,
                    target_ms: 1000.0,
                    memory_limit_mb: Some(150),
                }),
                enabled: true,
            },
            PerformanceTest {
                name: "ipc_latency".to_string(),
                description: "Measures IPC communication latency".to_string(),
                category: TestCategory::IpcLatency,
                warmup_iterations: 5,
                test_iterations: 100,
                timeout_seconds: 60,
                expected_range: Some(PerformanceRange {
                    min_ms: 1.0,
                    max_ms: 5.0,
                    target_ms: 2.0,
                    memory_limit_mb: None,
                }),
                enabled: true,
            },
            PerformanceTest {
                name: "memory_usage".to_string(),
                description: "Measures memory allocation and cleanup".to_string(),
                category: TestCategory::MemoryUsage,
                warmup_iterations: 3,
                test_iterations: 10,
                timeout_seconds: 30,
                expected_range: Some(PerformanceRange {
                    min_ms: 10.0,
                    max_ms: 100.0,
                    target_ms: 50.0,
                    memory_limit_mb: Some(200),
                }),
                enabled: true,
            },
            PerformanceTest {
                name: "database_operations".to_string(),
                description: "Measures database query performance".to_string(),
                category: TestCategory::Database,
                warmup_iterations: 5,
                test_iterations: 50,
                timeout_seconds: 60,
                expected_range: Some(PerformanceRange {
                    min_ms: 5.0,
                    max_ms: 50.0,
                    target_ms: 15.0,
                    memory_limit_mb: None,
                }),
                enabled: true,
            },
            PerformanceTest {
                name: "file_system_operations".to_string(),
                description: "Measures file I/O performance".to_string(),
                category: TestCategory::FileSystem,
                warmup_iterations: 3,
                test_iterations: 20,
                timeout_seconds: 45,
                expected_range: Some(PerformanceRange {
                    min_ms: 10.0,
                    max_ms: 100.0,
                    target_ms: 30.0,
                    memory_limit_mb: None,
                }),
                enabled: true,
            },
        ];

        self.register_tests(default_tests).await;
        info!("Registered {} default benchmark tests", 5);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_benchmark_registration() {
        let suite = BenchmarkSuite::new();
        
        let test = PerformanceTest {
            name: "test_benchmark".to_string(),
            description: "Test benchmark".to_string(),
            category: TestCategory::Startup,
            warmup_iterations: 1,
            test_iterations: 3,
            timeout_seconds: 10,
            expected_range: None,
            enabled: true,
        };
        
        suite.register_test(test).await;
        
        let tests = suite.tests.read().await;
        assert!(tests.contains_key("test_benchmark"));
    }

    #[tokio::test]
    async fn test_benchmark_execution() {
        let suite = BenchmarkSuite::new();
        
        let test = PerformanceTest {
            name: "startup_test".to_string(),
            description: "Quick startup test".to_string(),
            category: TestCategory::Startup,
            warmup_iterations: 1,
            test_iterations: 2,
            timeout_seconds: 5,
            expected_range: None,
            enabled: true,
        };
        
        suite.register_test(test).await;
        
        let result = suite.run_test("startup_test").await;
        assert!(result.is_ok());
        
        let benchmark_result = result.unwrap();
        assert_eq!(benchmark_result.test_name, "startup_test");
        assert_eq!(benchmark_result.iterations, 2);
        assert!(benchmark_result.success_rate > 0.0);
    }

    #[tokio::test]
    async fn test_benchmark_context() {
        let test = PerformanceTest {
            name: "context_test".to_string(),
            description: "Test context".to_string(),
            category: TestCategory::IpcLatency,
            warmup_iterations: 0,
            test_iterations: 3,
            timeout_seconds: 5,
            expected_range: None,
            enabled: true,
        };
        
        let mut context = BenchmarkContext::new(test);
        
        context.record_iteration(Duration::from_millis(100), 50, 10.0);
        context.record_iteration(Duration::from_millis(150), 55, 15.0);
        context.record_iteration(Duration::from_millis(120), 52, 12.0);
        
        let result = context.finalize(None);
        
        assert_eq!(result.iterations, 3);
        assert_eq!(result.min_duration, Duration::from_millis(100));
        assert_eq!(result.max_duration, Duration::from_millis(150));
        assert_eq!(result.median_duration, Duration::from_millis(120));
        assert!(result.success_rate > 0.0);
    }
}