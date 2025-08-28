<script lang="ts">
  import type { PerformanceAlert } from '$lib/performance/metrics-collector';

  interface Props {
    alerts: PerformanceAlert[];
    onClear: () => void;
  }

  let { alerts, onClear }: Props = $props();

  function formatTime(timestamp: number): string {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  }

  function getAlertIcon(type: 'warning' | 'critical'): string {
    return type === 'critical' ? '🚨' : '⚠️';
  }

  let sortedAlerts = $derived(
    [...alerts].sort((a, b) => b.timestamp - a.timestamp)
  );

  let criticalCount = $derived(
    alerts.filter(a => a.type === 'critical').length
  );

  let warningCount = $derived(
    alerts.filter(a => a.type === 'warning').length
  );
</script>

<div class="alerts-list">
  {#if alerts.length > 0}
    <div class="alerts-header">
      <div class="alert-counts">
        {#if criticalCount > 0}
          <span class="count critical">
            {criticalCount} Critical
          </span>
        {/if}
        {#if warningCount > 0}
          <span class="count warning">
            {warningCount} Warning{warningCount !== 1 ? 's' : ''}
          </span>
        {/if}
      </div>
      
      <button onclick={onClear} class="btn-clear">
        Clear All
      </button>
    </div>

    <div class="alerts-container">
      {#each sortedAlerts as alert (alert.id)}
        <div class="alert alert-{alert.type}">
          <div class="alert-icon">
            {getAlertIcon(alert.type)}
          </div>
          
          <div class="alert-content">
            <div class="alert-message">
              {alert.message}
            </div>
            
            <div class="alert-meta">
              <span class="alert-metric">{alert.metric}</span>
              <span class="alert-value">
                Value: {alert.value.toFixed(2)} 
                (Threshold: {alert.threshold})
              </span>
              <span class="alert-time">{formatTime(alert.timestamp)}</span>
            </div>
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <div class="no-alerts">
      <p>No performance alerts</p>
      <p class="subtitle">System is running smoothly</p>
    </div>
  {/if}
</div>

<style>
  .alerts-list {
    width: 100%;
  }

  .alerts-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color, #e5e7eb);
  }

  .alert-counts {
    display: flex;
    gap: 0.75rem;
  }

  .count {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 600;
  }

  .count.critical {
    background: var(--error-light, #fee2e2);
    color: var(--error, #ef4444);
  }

  .count.warning {
    background: var(--warning-light, #fef3c7);
    color: var(--warning-dark, #d97706);
  }

  .btn-clear {
    padding: 0.375rem 0.75rem;
    background: transparent;
    border: 1px solid var(--border-color, #e5e7eb);
    border-radius: 0.375rem;
    font-size: 0.875rem;
    color: var(--text-secondary, #6b7280);
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-clear:hover {
    background: var(--hover-bg, #f9fafb);
    color: var(--text-primary, #111827);
  }

  .alerts-container {
    max-height: 400px;
    overflow-y: auto;
    space-y: 0.5rem;
  }

  .alert {
    display: flex;
    gap: 0.75rem;
    padding: 0.75rem;
    background: var(--card-bg, #ffffff);
    border: 1px solid var(--border-color, #e5e7eb);
    border-radius: 0.375rem;
    margin-bottom: 0.5rem;
    transition: all 0.2s;
  }

  .alert:hover {
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  }

  .alert-critical {
    border-left: 4px solid var(--error, #ef4444);
    background: var(--error-light, #fef2f2);
  }

  .alert-warning {
    border-left: 4px solid var(--warning, #f59e0b);
    background: var(--warning-light, #fffbeb);
  }

  .alert-icon {
    font-size: 1.25rem;
    line-height: 1;
  }

  .alert-content {
    flex: 1;
  }

  .alert-message {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-primary, #111827);
    margin-bottom: 0.375rem;
  }

  .alert-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.75rem;
    color: var(--text-secondary, #6b7280);
    flex-wrap: wrap;
  }

  .alert-metric {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.025em;
  }

  .alert-value {
    font-family: monospace;
  }

  .alert-time {
    margin-left: auto;
  }

  .no-alerts {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-secondary, #6b7280);
  }

  .no-alerts p {
    margin: 0;
  }

  .no-alerts .subtitle {
    margin-top: 0.5rem;
    font-size: 0.875rem;
    color: var(--text-tertiary, #9ca3af);
  }

  /* Custom scrollbar for alerts container */
  .alerts-container::-webkit-scrollbar {
    width: 6px;
  }

  .alerts-container::-webkit-scrollbar-track {
    background: var(--scrollbar-track, #f1f1f1);
    border-radius: 3px;
  }

  .alerts-container::-webkit-scrollbar-thumb {
    background: var(--scrollbar-thumb, #888);
    border-radius: 3px;
  }

  .alerts-container::-webkit-scrollbar-thumb:hover {
    background: var(--scrollbar-thumb-hover, #555);
  }
</style>