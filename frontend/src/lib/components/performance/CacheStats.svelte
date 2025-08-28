<script lang="ts">
  interface Props {
    stats: any;
    onClear: () => void;
  }

  let { stats, onClear }: Props = $props();

  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  }

  function formatNumber(num: number): string {
    return new Intl.NumberFormat().format(num);
  }

  function formatPercentage(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  let memoryStats = $derived(stats?.memory || null);
  let persistentStats = $derived(stats?.persistent || null);
  let deduplicationActive = $derived(stats?.deduplicationActive || 0);
  let warmingQueueSize = $derived(stats?.warmingQueueSize || 0);
</script>

<div class="cache-stats">
  {#if stats}
    <div class="stats-grid">
      <!-- Memory Cache Stats -->
      <div class="stat-group">
        <h4 class="stat-group-title">Memory Cache</h4>
        
        <div class="stat-row">
          <span class="stat-label">Hit Rate:</span>
          <span class="stat-value" class:good={memoryStats?.hitRate > 0.7}>
            {formatPercentage(memoryStats?.hitRate || 0)}
          </span>
        </div>

        <div class="stat-row">
          <span class="stat-label">Items:</span>
          <span class="stat-value">{formatNumber(memoryStats?.itemCount || 0)}</span>
        </div>

        <div class="stat-row">
          <span class="stat-label">Size:</span>
          <span class="stat-value">
            {formatBytes(memoryStats?.sizeInfo?.used || 0)} / 
            {formatBytes(memoryStats?.sizeInfo?.max || 0)}
          </span>
        </div>

        <div class="stat-row">
          <span class="stat-label">Usage:</span>
          <div class="progress-bar">
            <div 
              class="progress-fill"
              style="width: {memoryStats?.sizeInfo?.percentage || 0}%"
              class:warning={memoryStats?.sizeInfo?.percentage > 70}
              class:critical={memoryStats?.sizeInfo?.percentage > 90}
            ></div>
          </div>
        </div>

        <div class="stat-row">
          <span class="stat-label">Hits:</span>
          <span class="stat-value">{formatNumber(memoryStats?.hits || 0)}</span>
        </div>

        <div class="stat-row">
          <span class="stat-label">Misses:</span>
          <span class="stat-value">{formatNumber(memoryStats?.misses || 0)}</span>
        </div>

        <div class="stat-row">
          <span class="stat-label">Evictions:</span>
          <span class="stat-value">{formatNumber(memoryStats?.evictions || 0)}</span>
        </div>
      </div>

      <!-- Persistent Cache Stats -->
      <div class="stat-group">
        <h4 class="stat-group-title">Persistent Cache (IndexedDB)</h4>
        
        <div class="stat-row">
          <span class="stat-label">Items:</span>
          <span class="stat-value">{formatNumber(persistentStats?.totalItems || 0)}</span>
        </div>

        <div class="stat-row">
          <span class="stat-label">Size:</span>
          <span class="stat-value">{formatBytes(persistentStats?.totalSize || 0)}</span>
        </div>

        {#if persistentStats?.oldestEntry}
          <div class="stat-row">
            <span class="stat-label">Oldest:</span>
            <span class="stat-value">
              {new Date(persistentStats.oldestEntry).toLocaleTimeString()}
            </span>
          </div>
        {/if}

        {#if persistentStats?.newestEntry}
          <div class="stat-row">
            <span class="stat-label">Newest:</span>
            <span class="stat-value">
              {new Date(persistentStats.newestEntry).toLocaleTimeString()}
            </span>
          </div>
        {/if}
      </div>

      <!-- Active Operations -->
      <div class="stat-group">
        <h4 class="stat-group-title">Active Operations</h4>
        
        <div class="stat-row">
          <span class="stat-label">Deduplication:</span>
          <span class="stat-value" class:active={deduplicationActive > 0}>
            {formatNumber(deduplicationActive)} requests
          </span>
        </div>

        <div class="stat-row">
          <span class="stat-label">Warming Queue:</span>
          <span class="stat-value" class:active={warmingQueueSize > 0}>
            {formatNumber(warmingQueueSize)} items
          </span>
        </div>
      </div>
    </div>

    <div class="cache-actions">
      <button onclick={onClear} class="btn btn-danger">
        Clear All Caches
      </button>
    </div>
  {:else}
    <div class="no-stats">
      <p>No cache statistics available</p>
    </div>
  {/if}
</div>

<style>
  .cache-stats {
    width: 100%;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-bottom: 1rem;
  }

  .stat-group {
    background: var(--card-bg, #ffffff);
    padding: 1rem;
    border-radius: 0.375rem;
    border: 1px solid var(--border-color, #e5e7eb);
  }

  .stat-group-title {
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: var(--text-primary, #111827);
    text-transform: uppercase;
    letter-spacing: 0.025em;
  }

  .stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.375rem 0;
    border-bottom: 1px solid var(--border-light, #f3f4f6);
  }

  .stat-row:last-child {
    border-bottom: none;
  }

  .stat-label {
    font-size: 0.875rem;
    color: var(--text-secondary, #6b7280);
  }

  .stat-value {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary, #111827);
  }

  .stat-value.good {
    color: var(--success, #10b981);
  }

  .stat-value.active {
    color: var(--warning, #f59e0b);
  }

  .progress-bar {
    width: 120px;
    height: 8px;
    background: var(--progress-bg, #e5e7eb);
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--primary, #3b82f6);
    transition: width 0.3s ease;
  }

  .progress-fill.warning {
    background: var(--warning, #f59e0b);
  }

  .progress-fill.critical {
    background: var(--error, #ef4444);
  }

  .cache-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 1rem;
  }

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    border: none;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.2s;
    font-size: 0.875rem;
  }

  .btn-danger {
    background: var(--error, #ef4444);
    color: white;
  }

  .btn-danger:hover {
    background: var(--error-hover, #dc2626);
  }

  .no-stats {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary, #6b7280);
  }
</style>