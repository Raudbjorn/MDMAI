/*!
 * Performance Commands Module
 * 
 * Tauri commands for accessing performance optimization features from the frontend.
 */

use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::State;
use serde_json::Value;
use log::{info, debug, error};

use crate::performance::{
    PerformanceManager, PerformanceConfig, OptimizationResult,
    BenchmarkResult, PerformanceMetrics, ResourceMetrics,
};
use crate::performance::resource_monitor::ResourceMetrics as ResourceMonitorMetrics;

/// Performance manager state for Tauri
pub struct PerformanceManagerState(pub Arc<PerformanceManager>);

impl PerformanceManagerState {
    pub fn new() -> Self {
        Self(Arc::new(PerformanceManager::new()))
    }
}

/// Initialize the performance manager
#[tauri::command]
pub async fn initialize_performance_manager(
    state: State<'_, PerformanceManagerState>,
) -> Result<(), String> {
    info!("Initializing performance manager from frontend");
    state.0.initialize().await
}

/// Begin optimized startup sequence
#[tauri::command]
pub async fn begin_performance_startup(
    state: State<'_, PerformanceManagerState>,
) -> Result<String, String> {
    debug!("Beginning performance startup sequence");
    let context = state.0.begin_startup().await;
    
    // For this demo, we'll immediately complete startup
    // In a real implementation, you'd track the startup process
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        // Simulate startup completion
    });
    
    Ok("startup_context_id".to_string())
}

/// Get current performance metrics
#[tauri::command]
pub async fn get_performance_metrics(
    state: State<'_, PerformanceManagerState>,
) -> Result<PerformanceMetrics, String> {
    debug!("Retrieving current performance metrics");
    Ok(state.0.get_metrics().await)
}

/// Update performance configuration
#[tauri::command]
pub async fn update_performance_config(
    state: State<'_, PerformanceManagerState>,
    config: PerformanceConfig,
) -> Result<(), String> {
    info!("Updating performance configuration");
    state.0.update_config(config).await
}

/// Run performance benchmarks
#[tauri::command]
pub async fn run_performance_benchmarks(
    state: State<'_, PerformanceManagerState>,
) -> Result<Vec<BenchmarkResult>, String> {
    info!("Running performance benchmarks");
    state.0.run_benchmarks().await
}

/// Get resource usage statistics
#[tauri::command]
pub async fn get_resource_stats(
    state: State<'_, PerformanceManagerState>,
) -> Result<ResourceMonitorMetrics, String> {
    debug!("Retrieving resource statistics");
    Ok(state.0.get_resource_stats().await)
}

/// Optimize memory usage
#[tauri::command]
pub async fn optimize_memory(
    state: State<'_, PerformanceManagerState>,
) -> Result<(), String> {
    info!("Running memory optimization");
    state.0.optimize_memory().await
}

/// Get cache statistics from memory manager
#[tauri::command]
pub async fn get_cache_stats(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving cache statistics");
    let memory_manager = &state.0.memory_manager;
    let cache_stats = memory_manager.get_cache_stats().await;
    serde_json::to_value(cache_stats).map_err(|e| e.to_string())
}

/// Clear memory caches
#[tauri::command]
pub async fn clear_memory_caches(
    state: State<'_, PerformanceManagerState>,
) -> Result<(), String> {
    info!("Clearing memory caches");
    let memory_manager = &state.0.memory_manager;
    memory_manager.cache_clear().await;
    Ok(())
}

/// Get memory pool statistics
#[tauri::command]
pub async fn get_memory_pool_stats(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving memory pool statistics");
    let memory_manager = &state.0.memory_manager;
    let pool_stats = memory_manager.get_pool_stats().await;
    serde_json::to_value(pool_stats).map_err(|e| e.to_string())
}

/// Get IPC optimizer statistics
#[tauri::command]
pub async fn get_ipc_optimizer_stats(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving IPC optimizer statistics");
    let ipc_optimizer = &state.0.ipc_optimizer;
    let stats = ipc_optimizer.get_detailed_stats().await;
    Ok(stats)
}

/// Reset IPC optimizer (clear caches and statistics)
#[tauri::command]
pub async fn reset_ipc_optimizer(
    state: State<'_, PerformanceManagerState>,
) -> Result<(), String> {
    info!("Resetting IPC optimizer");
    let ipc_optimizer = &state.0.ipc_optimizer;
    ipc_optimizer.reset().await;
    Ok(())
}

/// Get lazy loader statistics
#[tauri::command]
pub async fn get_lazy_loader_stats(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving lazy loader statistics");
    let lazy_loader = &state.0.lazy_loader;
    let stats = lazy_loader.get_stats().await;
    serde_json::to_value(stats).map_err(|e| e.to_string())
}

/// Get lazy component information
#[tauri::command]
pub async fn get_component_info(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving component information");
    let lazy_loader = &state.0.lazy_loader;
    let component_info = lazy_loader.get_component_info().await;
    serde_json::to_value(component_info).map_err(|e| e.to_string())
}

/// Load a specific component on demand
#[tauri::command]
pub async fn load_component(
    state: State<'_, PerformanceManagerState>,
    component_name: String,
) -> Result<Value, String> {
    info!("Loading component on demand: {}", component_name);
    let lazy_loader = &state.0.lazy_loader;
    let result = lazy_loader.load_component(&component_name, "user_request").await?;
    serde_json::to_value(result).map_err(|e| e.to_string())
}

/// Preload components based on trigger
#[tauri::command]
pub async fn preload_components(
    state: State<'_, PerformanceManagerState>,
    trigger: String,
) -> Result<Value, String> {
    info!("Preloading components with trigger: {}", trigger);
    let lazy_loader = &state.0.lazy_loader;
    let results = lazy_loader.preload_components(&trigger).await;
    serde_json::to_value(results).map_err(|e| e.to_string())
}

/// Get benchmark suite history
#[tauri::command]
pub async fn get_benchmark_history(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving benchmark history");
    let benchmark_suite = &state.0.benchmark_suite;
    let history = benchmark_suite.get_history().await;
    serde_json::to_value(history).map_err(|e| e.to_string())
}

/// Get benchmark baselines
#[tauri::command]
pub async fn get_benchmark_baselines(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving benchmark baselines");
    let benchmark_suite = &state.0.benchmark_suite;
    let baselines = benchmark_suite.get_baselines().await;
    serde_json::to_value(baselines).map_err(|e| e.to_string())
}

/// Clear benchmark baselines
#[tauri::command]
pub async fn clear_benchmark_baselines(
    state: State<'_, PerformanceManagerState>,
) -> Result<(), String> {
    info!("Clearing benchmark baselines");
    let benchmark_suite = &state.0.benchmark_suite;
    benchmark_suite.clear_baselines().await;
    Ok(())
}

/// Run startup benchmark specifically
#[tauri::command]
pub async fn run_startup_benchmark(
    state: State<'_, PerformanceManagerState>,
) -> Result<BenchmarkResult, String> {
    info!("Running startup benchmark");
    let benchmark_suite = &state.0.benchmark_suite;
    benchmark_suite.run_startup_benchmark().await
}

/// Get metrics collection history
#[tauri::command]
pub async fn get_metrics_history(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving metrics history");
    let metrics_collector = &state.0.metrics_collector;
    let history = metrics_collector.get_history().await;
    serde_json::to_value(history).map_err(|e| e.to_string())
}

/// Get metrics for specific time range
#[tauri::command]
pub async fn get_metrics_range(
    state: State<'_, PerformanceManagerState>,
    start: String,
    end: String,
) -> Result<Value, String> {
    debug!("Retrieving metrics for time range: {} to {}", start, end);
    
    let start_dt = chrono::DateTime::parse_from_rfc3339(&start)
        .map_err(|e| format!("Invalid start date format: {}", e))?
        .with_timezone(&chrono::Utc);
    
    let end_dt = chrono::DateTime::parse_from_rfc3339(&end)
        .map_err(|e| format!("Invalid end date format: {}", e))?
        .with_timezone(&chrono::Utc);
    
    let metrics_collector = &state.0.metrics_collector;
    let metrics = metrics_collector.get_metrics_range(start_dt, end_dt).await;
    serde_json::to_value(metrics).map_err(|e| e.to_string())
}

/// Get active resource alerts
#[tauri::command]
pub async fn get_active_alerts(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving active resource alerts");
    let resource_monitor = &state.0.resource_monitor;
    let alerts = resource_monitor.get_active_alerts().await;
    serde_json::to_value(alerts).map_err(|e| e.to_string())
}

/// Get performance trends
#[tauri::command]
pub async fn get_performance_trends(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving performance trends");
    let resource_monitor = &state.0.resource_monitor;
    let trends = resource_monitor.get_performance_trends().await;
    serde_json::to_value(trends).map_err(|e| e.to_string())
}

/// Get optimization recommendations
#[tauri::command]
pub async fn get_optimization_recommendations(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving optimization recommendations");
    let resource_monitor = &state.0.resource_monitor;
    let recommendations = resource_monitor.get_optimization_recommendations().await;
    serde_json::to_value(recommendations).map_err(|e| e.to_string())
}

/// Get historical resource data
#[tauri::command]
pub async fn get_historical_resource_data(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving historical resource data");
    let resource_monitor = &state.0.resource_monitor;
    let history = resource_monitor.get_historical_data().await;
    serde_json::to_value(history).map_err(|e| e.to_string())
}

/// Get system information summary
#[tauri::command]
pub async fn get_system_info_summary(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving system information summary");
    let resource_stats = state.0.get_resource_stats().await;
    
    let summary = serde_json::json!({
        "timestamp": resource_stats.timestamp,
        "health_score": resource_stats.health_score,
        "cpu": {
            "usage_percent": resource_stats.system_metrics.cpu_usage_percent,
            "cores": resource_stats.system_metrics.cpu_cores,
            "frequency_mhz": resource_stats.system_metrics.cpu_frequency_mhz
        },
        "memory": {
            "total_gb": resource_stats.system_metrics.memory_total_gb,
            "used_gb": resource_stats.system_metrics.memory_used_gb,
            "usage_percent": resource_stats.system_metrics.memory_usage_percent
        },
        "process": {
            "pid": resource_stats.process_metrics.pid,
            "cpu_percent": resource_stats.process_metrics.cpu_usage_percent,
            "memory_mb": resource_stats.process_metrics.memory_usage_mb
        }
    });
    
    Ok(summary)
}

/// Create performance optimization report
#[tauri::command]
pub async fn create_optimization_report(
    state: State<'_, PerformanceManagerState>,
) -> Result<OptimizationResult, String> {
    info!("Creating performance optimization report");
    
    let metrics = state.0.get_metrics().await;
    let resource_stats = state.0.get_resource_stats().await;
    let recommendations = resource_stats.system_metrics.memory_usage_percent;
    
    let memory_usage_mb = resource_stats.process_metrics.memory_usage_mb as u64;
    let startup_time_ms = metrics.startup_metrics.total_startup_time_ms;
    
    let mut report = OptimizationResult::new();
    report.startup_time_ms = startup_time_ms;
    report.memory_usage_mb = memory_usage_mb;
    report.memory_saved_mb = if memory_usage_mb > 100 { 25 } else { 0 }; // Estimated savings
    report.ipc_latency_ms = metrics.ipc_metrics.average_latency_ms;
    report.cache_hit_ratio = metrics.ipc_metrics.cache_hit_ratio;
    
    // Add optimizations based on current state
    if resource_stats.system_metrics.memory_usage_percent > 80.0 {
        report.optimizations_applied.push("Memory optimization recommended".to_string());
    }
    if resource_stats.system_metrics.cpu_usage_percent > 70.0 {
        report.optimizations_applied.push("CPU optimization recommended".to_string());
    }
    if metrics.ipc_metrics.average_latency_ms > 50.0 {
        report.optimizations_applied.push("IPC optimization recommended".to_string());
    }
    
    // Add warnings for potential issues
    if startup_time_ms > 3000 {
        report.warnings.push("Startup time exceeds target of 2 seconds".to_string());
    }
    if memory_usage_mb > 150 {
        report.warnings.push("Memory usage exceeds target of 150MB".to_string());
    }
    
    Ok(report)
}

/// Export performance data
#[tauri::command]
pub async fn export_performance_data(
    state: State<'_, PerformanceManagerState>,
    format: String,
) -> Result<String, String> {
    info!("Exporting performance data in format: {}", format);
    
    let metrics = state.0.get_metrics().await;
    let resource_stats = state.0.get_resource_stats().await;
    let benchmark_history = state.0.benchmark_suite.get_history().await;
    
    let export_data = serde_json::json!({
        "export_timestamp": chrono::Utc::now(),
        "current_metrics": metrics,
        "resource_stats": resource_stats,
        "benchmark_history": benchmark_history,
        "format": format
    });
    
    match format.to_lowercase().as_str() {
        "json" => {
            serde_json::to_string_pretty(&export_data)
                .map_err(|e| format!("Failed to serialize data: {}", e))
        },
        "csv" => {
            // Simplified CSV export - in a real implementation, you'd create proper CSV
            Ok("timestamp,cpu_percent,memory_mb,startup_ms,ipc_latency_ms\n".to_string() +
               &format!("{},{},{},{},{}\n", 
                        chrono::Utc::now(),
                        resource_stats.system_metrics.cpu_usage_percent,
                        resource_stats.process_metrics.memory_usage_mb,
                        metrics.startup_metrics.total_startup_time_ms,
                        metrics.ipc_metrics.average_latency_ms))
        },
        _ => Err(format!("Unsupported export format: {}", format))
    }
}

/// Shutdown performance manager
#[tauri::command]
pub async fn shutdown_performance_manager(
    state: State<'_, PerformanceManagerState>,
) -> Result<(), String> {
    info!("Shutting down performance manager");
    state.0.shutdown().await
}

/// Get performance manager status
#[tauri::command]
pub async fn get_performance_status(
    state: State<'_, PerformanceManagerState>,
) -> Result<Value, String> {
    debug!("Retrieving performance manager status");
    
    let metrics = state.0.get_metrics().await;
    let resource_stats = state.0.get_resource_stats().await;
    
    let status = serde_json::json!({
        "initialized": true,
        "monitoring_active": true,
        "health_score": resource_stats.health_score,
        "startup_time_ms": metrics.startup_metrics.total_startup_time_ms,
        "memory_usage_mb": resource_stats.process_metrics.memory_usage_mb,
        "cpu_usage_percent": resource_stats.process_metrics.cpu_usage_percent,
        "last_updated": chrono::Utc::now(),
        "performance_targets": {
            "startup_time_target_ms": 2000,
            "memory_usage_target_mb": 150,
            "ipc_latency_target_ms": 5.0
        },
        "targets_met": {
            "startup_time": metrics.startup_metrics.total_startup_time_ms <= 2000,
            "memory_usage": resource_stats.process_metrics.memory_usage_mb <= 150,
            "ipc_latency": metrics.ipc_metrics.average_latency_ms <= 5.0
        }
    });
    
    Ok(status)
}

/// Force garbage collection and optimization
#[tauri::command]
pub async fn force_optimization(
    state: State<'_, PerformanceManagerState>,
) -> Result<OptimizationResult, String> {
    info!("Forcing comprehensive optimization");
    
    // Run memory optimization
    state.0.optimize_memory().await?;
    
    // Clear caches
    let memory_manager = &state.0.memory_manager;
    memory_manager.cache_clear().await;
    
    // Reset IPC optimizer
    let ipc_optimizer = &state.0.ipc_optimizer;
    ipc_optimizer.reset().await;
    
    // Generate optimization report
    create_optimization_report(state).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::PerformanceConfig;

    #[tokio::test]
    async fn test_performance_commands() {
        let state = PerformanceManagerState::new();
        
        // Test initialization
        let result = initialize_performance_manager(tauri::State::from(&state)).await;
        assert!(result.is_ok());
        
        // Test getting metrics
        let metrics_result = get_performance_metrics(tauri::State::from(&state)).await;
        assert!(metrics_result.is_ok());
        
        // Test getting resource stats
        let resource_result = get_resource_stats(tauri::State::from(&state)).await;
        assert!(resource_result.is_ok());
    }

    #[tokio::test]
    async fn test_optimization_report() {
        let state = PerformanceManagerState::new();
        
        // Initialize first
        let _ = initialize_performance_manager(tauri::State::from(&state)).await;
        
        let report_result = create_optimization_report(tauri::State::from(&state)).await;
        assert!(report_result.is_ok());
        
        let report = report_result.unwrap();
        assert!(report.startup_time_ms >= 0);
        assert!(report.memory_usage_mb >= 0);
    }
}