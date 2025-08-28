<script lang="ts">
  import PerformanceDashboard from '$lib/components/performance/PerformanceDashboard.svelte';
  import { useCacheStats, usePerformanceMetrics } from '$lib/hooks/use-cache';
  import { api } from '$lib/api/optimized-client';
  
  const { stats: cacheStats } = useCacheStats();
  const { webVitals, metrics } = usePerformanceMetrics();
  
  // Example of prefetching data
  $effect(() => {
    // Prefetch common resources
    api.prefetch([
      '/api/campaigns',
      '/api/characters',
      '/api/rules/systems'
    ], {
      ttl: 3600000, // 1 hour
      persistent: true
    });
  });
</script>

<div class="performance-page">
  <div class="page-header">
    <h1>Performance Monitoring</h1>
    <p class="subtitle">
      Real-time performance metrics and cache statistics
    </p>
  </div>

  <PerformanceDashboard />

  <div class="additional-info">
    <div class="info-card">
      <h3>Cache Optimization</h3>
      <p>
        The application uses a multi-layer caching strategy with in-memory LRU cache
        and persistent IndexedDB storage for offline support.
      </p>
      <ul>
        <li>Automatic cache warming for frequently used resources</li>
        <li>Request deduplication to prevent redundant API calls</li>
        <li>Predictive prefetching based on user patterns</li>
        <li>TTL-based cache invalidation with smart expiration</li>
      </ul>
    </div>

    <div class="info-card">
      <h3>Performance Features</h3>
      <ul>
        <li>Request batching for bulk operations</li>
        <li>Debounced search and autocomplete</li>
        <li>WebSocket connection pooling</li>
        <li>Service Worker for offline functionality</li>
        <li>Lazy loading and code splitting</li>
        <li>Web Vitals monitoring (FCP, LCP, CLS, etc.)</li>
      </ul>
    </div>

    <div class="info-card">
      <h3>Quick Actions</h3>
      <div class="action-buttons">
        <button 
          onclick={() => api.invalidateCache()}
          class="btn btn-secondary"
        >
          Clear All Caches
        </button>
        
        <button 
          onclick={() => api.invalidateCache(/\/api\/search/)}
          class="btn btn-secondary"
        >
          Clear Search Cache
        </button>
        
        <button 
          onclick={() => window.location.reload()}
          class="btn btn-secondary"
        >
          Force Reload
        </button>
        
        <button 
          onclick={() => {
            if ('serviceWorker' in navigator) {
              navigator.serviceWorker.getRegistration().then(reg => {
                reg?.update();
              });
            }
          }}
          class="btn btn-secondary"
        >
          Update Service Worker
        </button>
      </div>
    </div>
  </div>
</div>

<style>
  .performance-page {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
  }

  .page-header {
    margin-bottom: 2rem;
  }

  .page-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: var(--text-primary, #111827);
  }

  .subtitle {
    font-size: 1.125rem;
    color: var(--text-secondary, #6b7280);
  }

  .additional-info {
    margin-top: 3rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 2rem;
  }

  .info-card {
    background: var(--card-bg, #ffffff);
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid var(--border-color, #e5e7eb);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .info-card h3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-primary, #111827);
  }

  .info-card p {
    margin-bottom: 1rem;
    line-height: 1.6;
    color: var(--text-secondary, #6b7280);
  }

  .info-card ul {
    list-style: none;
    padding: 0;
  }

  .info-card li {
    padding: 0.5rem 0;
    padding-left: 1.5rem;
    position: relative;
    color: var(--text-secondary, #6b7280);
  }

  .info-card li::before {
    content: '✓';
    position: absolute;
    left: 0;
    color: var(--success, #10b981);
    font-weight: bold;
  }

  .action-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
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

  .btn-secondary {
    background: var(--secondary, #64748b);
    color: white;
  }

  .btn-secondary:hover {
    background: var(--secondary-hover, #475569);
  }

  @media (max-width: 768px) {
    .performance-page {
      padding: 1rem;
    }

    .additional-info {
      grid-template-columns: 1fr;
    }
  }
</style>