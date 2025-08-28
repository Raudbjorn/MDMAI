<script lang="ts">
  import type { WebVitalsMetrics } from '$lib/performance/metrics-collector';

  interface Props {
    webVitals: WebVitalsMetrics;
  }

  let { webVitals }: Props = $props();

  function getVitalStatus(metric: string, value: number | null): 'good' | 'needs-improvement' | 'poor' | 'unknown' {
    if (value === null) return 'unknown';

    const thresholds: Record<string, { good: number; poor: number }> = {
      FCP: { good: 1800, poor: 3000 },
      LCP: { good: 2500, poor: 4000 },
      FID: { good: 100, poor: 300 },
      CLS: { good: 0.1, poor: 0.25 },
      TTFB: { good: 800, poor: 1800 },
      INP: { good: 200, poor: 500 }
    };

    const threshold = thresholds[metric];
    if (!threshold) return 'unknown';

    if (value <= threshold.good) return 'good';
    if (value >= threshold.poor) return 'poor';
    return 'needs-improvement';
  }

  function formatValue(metric: string, value: number | null): string {
    if (value === null) return 'N/A';

    if (metric === 'CLS') {
      return value.toFixed(3);
    }

    return `${Math.round(value)} ms`;
  }

  const vitalDescriptions = {
    FCP: 'First Contentful Paint - Time to first content render',
    LCP: 'Largest Contentful Paint - Time to largest element render',
    FID: 'First Input Delay - Time from first interaction to response',
    CLS: 'Cumulative Layout Shift - Visual stability score',
    TTFB: 'Time to First Byte - Server response time',
    INP: 'Interaction to Next Paint - Overall interaction responsiveness'
  };
</script>

<div class="web-vitals-card">
  <div class="vitals-grid">
    {#each Object.entries(webVitals) as [key, value]}
      {@const status = getVitalStatus(key, value)}
      <div class="vital-item" data-status={status}>
        <div class="vital-header">
          <span class="vital-name">{key}</span>
          <span class="vital-status status-{status}">
            {status === 'good' ? '✓' : status === 'poor' ? '✗' : status === 'needs-improvement' ? '!' : '?'}
          </span>
        </div>
        <div class="vital-value">
          {formatValue(key, value)}
        </div>
        <div class="vital-description">
          {vitalDescriptions[key as keyof typeof vitalDescriptions]}
        </div>
      </div>
    {/each}
  </div>
</div>

<style>
  .web-vitals-card {
    width: 100%;
  }

  .vitals-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
  }

  .vital-item {
    padding: 1rem;
    background: var(--card-bg, #ffffff);
    border: 1px solid var(--border-color, #e5e7eb);
    border-radius: 0.5rem;
    transition: all 0.2s;
  }

  .vital-item:hover {
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .vital-item[data-status="good"] {
    border-left: 4px solid var(--success, #10b981);
  }

  .vital-item[data-status="needs-improvement"] {
    border-left: 4px solid var(--warning, #f59e0b);
  }

  .vital-item[data-status="poor"] {
    border-left: 4px solid var(--error, #ef4444);
  }

  .vital-item[data-status="unknown"] {
    border-left: 4px solid var(--gray, #9ca3af);
  }

  .vital-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .vital-name {
    font-weight: 600;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.025em;
  }

  .vital-status {
    width: 1.5rem;
    height: 1.5rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.75rem;
  }

  .status-good {
    background: var(--success, #10b981);
    color: white;
  }

  .status-needs-improvement {
    background: var(--warning, #f59e0b);
    color: white;
  }

  .status-poor {
    background: var(--error, #ef4444);
    color: white;
  }

  .status-unknown {
    background: var(--gray, #9ca3af);
    color: white;
  }

  .vital-value {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: var(--text-primary, #111827);
  }

  .vital-description {
    font-size: 0.75rem;
    color: var(--text-secondary, #6b7280);
    line-height: 1.4;
  }
</style>