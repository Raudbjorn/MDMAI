<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { MetricsCollector } from '$lib/performance/metrics-collector';
  import { CacheManager } from '$lib/cache/cache-manager';
  import type { PerformanceMetric, WebVitalsMetrics, PerformanceAlert } from '$lib/performance/metrics-collector';
  import WebVitalsCard from './WebVitalsCard.svelte';
  import MetricsChart from './MetricsChart.svelte';
  import CacheStats from './CacheStats.svelte';
  import AlertsList from './AlertsList.svelte';

  let metricsCollector = $state(new MetricsCollector());
  let cacheManager = $state(new CacheManager());
  
  let webVitals = $state<WebVitalsMetrics>({
    FCP: null,
    LCP: null,
    FID: null,
    CLS: null,
    TTFB: null,
    INP: null
  });
  
  let metrics = $state<PerformanceMetric[]>([]);
  let alerts = $state<PerformanceAlert[]>([]);
  let cacheStats = $state<any>(null);
  let selectedMetric = $state('response_time');
  let timeWindow = $state(300000); // 5 minutes
  let autoRefresh = $state(true);
  
  let unsubscribeMetrics: (() => void) | null = null;
  let unsubscribeAlerts: (() => void) | null = null;
  let refreshInterval: number | null = null;

  onMount(() => {
    // Set up thresholds
    metricsCollector.setThreshold('response_time', 1000, 3000);
    metricsCollector.setThreshold('memory_usage', 70, 90);
    metricsCollector.setThreshold('frame_rate', 30, 15);
    metricsCollector.setThreshold('long_task', 50, 100);
    
    // Subscribe to metrics
    unsubscribeMetrics = metricsCollector.subscribe((updatedMetrics) => {
      metrics = updatedMetrics;
    });
    
    // Subscribe to alerts
    unsubscribeAlerts = metricsCollector.subscribeToAlerts((alert) => {
      alerts = [alert, ...alerts].slice(0, 50); // Keep last 50 alerts
    });
    
    // Start auto-refresh
    startAutoRefresh();
    
    // Initial load
    refreshData();
  });

  onDestroy(() => {
    unsubscribeMetrics?.();
    unsubscribeAlerts?.();
    stopAutoRefresh();
    metricsCollector.destroy();
  });

  function startAutoRefresh() {
    if (!autoRefresh) return;
    
    }, 10000); // Refresh every 10 seconds
  }

  function stopAutoRefresh() {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      refreshInterval = null;
    }
  }

  function toggleAutoRefresh() {
    autoRefresh = !autoRefresh;
    if (autoRefresh) {
      startAutoRefresh();
    } else {
      stopAutoRefresh();
    }
  }

  async function refreshData() {
    webVitals = metricsCollector.getWebVitals();
    cacheStats = await cacheManager.getStats();
  }

  function clearCache() {
    cacheManager.clear();
    refreshData();
  }

  function clearAlerts() {
    alerts = [];
  }

  function exportMetrics() {
    const report = metricsCollector.generateReport();
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `performance-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const metricOptions = [
    { value: 'response_time', label: 'Response Time' },
    { value: 'memory_used', label: 'Memory Usage' },
    { value: 'frame_rate', label: 'Frame Rate' },
    { value: 'cache_hit_rate', label: 'Cache Hit Rate' },
    { value: 'long_task', label: 'Long Tasks' }
  ];

  const timeWindowOptions = [
    { value: 60000, label: '1 minute' },
    { value: 300000, label: '5 minutes' },
    { value: 900000, label: '15 minutes' },
    { value: 3600000, label: '1 hour' }
  ];
</script>

<div class="performance-dashboard">
  <div class="dashboard-header">
    <h2 class="text-2xl font-bold">Performance Dashboard</h2>
    
    <div class="controls">
      <select bind:value={selectedMetric} class="select">
        {#each metricOptions as option}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
      
      <select bind:value={timeWindow} class="select">
        {#each timeWindowOptions as option}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
      
      <button 
        onclick={toggleAutoRefresh}
        class="btn btn-secondary"
        class:active={autoRefresh}
      >
        {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
      </button>
      
      <button onclick={exportMetrics} class="btn btn-primary">
        Export Report
      </button>
    </div>
  </div>

  <div class="metrics-grid">
    <!-- Web Vitals -->
    <div class="metric-section">
      <h3 class="section-title">Core Web Vitals</h3>
      <WebVitalsCard {webVitals} />
    </div>

    <!-- Cache Statistics -->
    <div class="metric-section">
      <h3 class="section-title">Cache Performance</h3>
      <CacheStats stats={cacheStats} onClear={clearCache} />
    </div>

    <!-- Metrics Chart -->
    <div class="metric-section chart-section">
      <h3 class="section-title">Metrics Timeline</h3>
      <MetricsChart 
        {metrics} 
        metric={selectedMetric} 
        window={timeWindow} 
      />
    </div>

    <!-- Alerts -->
    <div class="metric-section alerts-section">
      <h3 class="section-title">
        Performance Alerts 
        {#if alerts.length > 0}
          <span class="alert-count">({alerts.length})</span>
        {/if}
      </h3>
      <AlertsList {alerts} onClear={clearAlerts} />
    </div>
  </div>

  <!-- Summary Statistics -->
  <div class="summary-stats">
    <div class="stat-card">
      <div class="stat-label">Avg Response Time</div>
      <div class="stat-value">
        {metricsCollector.getAggregatedMetrics('response_time', 'avg', timeWindow)?.toFixed(2) ?? 'N/A'} ms
      </div>
    </div>
    
    <div class="stat-card">
      <div class="stat-label">P95 Response Time</div>
      <div class="stat-value">
        {metricsCollector.getAggregatedMetrics('response_time', 'p95', timeWindow)?.toFixed(2) ?? 'N/A'} ms
      </div>
    </div>
    
    <div class="stat-card">
      <div class="stat-label">Cache Hit Rate</div>
      <div class="stat-value">
        {cacheStats?.memory?.hitRate ? (cacheStats.memory.hitRate * 100).toFixed(1) : '0'}%
      </div>
    </div>
    
    <div class="stat-card">
      <div class="stat-label">Memory Usage</div>
      <div class="stat-value">
        {cacheStats?.memory?.sizeInfo ? 
          `${(cacheStats.memory.sizeInfo.used / 1024 / 1024).toFixed(2)} MB` : 
          'N/A'}
      </div>
    </div>
  </div>
</div>

<style>
  .performance-dashboard {
    padding: 1.5rem;
    background: var(--background, #fff);
    border-radius: 0.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    flex-wrap: wrap;
    gap: 1rem;
  }

  .controls {
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
  }

  .select {
    padding: 0.5rem;
    border: 1px solid var(--border-color, #ddd);
    border-radius: 0.25rem;
    background: var(--input-bg, #fff);
    min-width: 150px;
  }

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 0.25rem;
    border: none;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.2s;
  }

  .btn-primary {
    background: var(--primary, #3b82f6);
    color: white;
  }

  .btn-primary:hover {
    background: var(--primary-hover, #2563eb);
  }

  .btn-secondary {
    background: var(--secondary, #64748b);
    color: white;
  }

  .btn-secondary.active {
    background: var(--success, #10b981);
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
  }

  .metric-section {
    background: var(--card-bg, #f9fafb);
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid var(--border-color, #e5e7eb);
  }

  .chart-section {
    grid-column: span 2;
  }

  .alerts-section {
    grid-column: span 2;
  }

  .section-title {
    font-size: 1.125rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .alert-count {
    font-size: 0.875rem;
    padding: 0.125rem 0.5rem;
    background: var(--warning, #f59e0b);
    color: white;
    border-radius: 9999px;
  }

  .summary-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }

  .stat-card {
    background: var(--card-bg, #f9fafb);
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid var(--border-color, #e5e7eb);
  }

  .stat-label {
    font-size: 0.875rem;
    color: var(--text-secondary, #6b7280);
    margin-bottom: 0.5rem;
  }

  .stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary, #111827);
  }

  @media (max-width: 768px) {
    .metrics-grid {
      grid-template-columns: 1fr;
    }

    .chart-section,
    .alerts-section {
      grid-column: span 1;
    }

    .controls {
      width: 100%;
      justify-content: flex-start;
    }
  }
</style>