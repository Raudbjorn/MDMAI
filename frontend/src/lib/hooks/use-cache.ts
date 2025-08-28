/**
 * Svelte hooks for cache integration
 */

import { CacheManager } from '$lib/cache/cache-manager';
import { MetricsCollector } from '$lib/performance/metrics-collector';
import type { CacheConfig, CacheKey } from '$lib/cache/types';

const cacheManager = new CacheManager();
const metricsCollector = new MetricsCollector();

/**
 * Hook for cached data fetching
 */
export function useCachedFetch<T>(
  key: string | CacheKey,
  fetcher: () => Promise<T>,
  config?: Partial<CacheConfig>
) {
  let data = $state<T | null>(null);
  let loading = $state(true);
  let error = $state<Error | null>(null);

  async function load() {
    loading = true;
    error = null;

    try {
      const start = performance.now();
      
      // Try cache first
      const cached = await cacheManager.get<T>(key);
      if (cached !== null) {
        data = cached;
        loading = false;
        
        metricsCollector.recordMetric({
          name: 'cache_hit',
          value: performance.now() - start,
          timestamp: Date.now(),
          unit: 'ms',
          tags: { key: typeof key === 'string' ? key : key.namespace }
        });
        
        return;
      }

      // Cache miss - fetch data
      metricsCollector.recordMetric({
        name: 'cache_miss',
        value: 1,
        timestamp: Date.now(),
        tags: { key: typeof key === 'string' ? key : key.namespace }
      });

      const result = await cacheManager.withCache(key, fetcher, config);
      data = result;
      
      metricsCollector.recordMetric({
        name: 'response_time',
        value: performance.now() - start,
        timestamp: Date.now(),
        unit: 'ms',
        tags: { key: typeof key === 'string' ? key : key.namespace }
      });
    } catch (err) {
      error = err instanceof Error ? err : new Error('Unknown error');
      
      metricsCollector.recordMetric({
        name: 'request_error',
        value: 1,
        timestamp: Date.now(),
        tags: { 
          key: typeof key === 'string' ? key : key.namespace,
          error: error.message
        }
      });
    } finally {
      loading = false;
    }
  }

  async function invalidate() {
    await cacheManager.delete(key);
    await load();
  }

  async function update(newData: T) {
    data = newData;
    await cacheManager.set(key, newData, config);
  }

  // Load on mount
  $effect(() => {
    load();
  });

  return {
    get data() { return data; },
    get loading() { return loading; },
    get error() { return error; },
    reload: load,
    invalidate,
    update
  };
}

/**
 * Hook for batch cached fetching
 */
export function useBatchCache<T>(
  keys: Array<string | CacheKey>,
  fetcher: (keys: Array<string | CacheKey>) => Promise<Map<string, T>>,
  config?: Partial<CacheConfig>
) {
  let data = $state<Map<string, T | null>>(new Map());
  let loading = $state(true);
  let error = $state<Error | null>(null);

  async function load() {
    loading = true;
    error = null;

    try {
      // Check cache for all keys
      const cached = await cacheManager.batchGet<T>(keys);
      
      // Separate hits and misses
      const hits: Map<string, T> = new Map();
      const misses: Array<string | CacheKey> = [];
      
      for (const [key, value] of cached.entries()) {
        if (value !== null) {
          hits.set(key, value);
        } else {
          const originalKey = keys.find(k => 
            typeof k === 'string' ? k === key : cacheManager.generateKey(k) === key
          );
          if (originalKey) misses.push(originalKey);
        }
      }

      // Update data with cache hits
      data = new Map(hits);

      // Fetch missing data if any
      if (misses.length > 0) {
        const fetchedData = await fetcher(misses);
        
        // Cache and update fetched data
        const updates: Array<{ key: string | CacheKey; value: T; config?: Partial<CacheConfig> }> = [];
        
        for (const [key, value] of fetchedData.entries()) {
          data.set(key, value);
          const originalKey = misses.find(k => 
            typeof k === 'string' ? k === key : cacheManager.generateKey(k) === key
          );
          if (originalKey) {
            updates.push({ key: originalKey, value, config });
          }
        }
        
        await cacheManager.batchSet(updates);
      }
    } catch (err) {
      error = err instanceof Error ? err : new Error('Unknown error');
    } finally {
      loading = false;
    }
  }

  async function invalidateAll() {
    await Promise.all(keys.map(key => cacheManager.delete(key)));
    await load();
  }

  // Load on mount
  $effect(() => {
    load();
  });

  return {
    get data() { return data; },
    get loading() { return loading; },
    get error() { return error; },
    reload: load,
    invalidateAll
  };
}

/**
 * Hook for prefetching data
 */
export function usePrefetch() {
  async function prefetch<T>(
    key: string | CacheKey,
    fetcher: () => Promise<T>,
    config?: Partial<CacheConfig>
  ) {
    // Check if already cached
    const cached = await cacheManager.get(key);
    if (cached !== null) return;

    // Prefetch in background
    cacheManager.withCache(key, fetcher, config).catch(error => {
      console.error('Prefetch error:', error);
    });
  }

  async function prefetchBatch<T>(
    items: Array<{
      key: string | CacheKey;
      fetcher: () => Promise<T>;
      config?: Partial<CacheConfig>;
    }>
  ) {
    const promises = items.map(item => 
      prefetch(item.key, item.fetcher, item.config)
    );
    
    await Promise.all(promises);
  }

  return {
    prefetch,
    prefetchBatch
  };
}

/**
 * Hook for cache statistics
 */
export function useCacheStats() {
  let stats = $state<any>(null);
  let loading = $state(false);

  async function refresh() {
    loading = true;
    try {
      stats = await cacheManager.getStats();
    } finally {
      loading = false;
    }
  }

  // Auto-refresh stats periodically
  $effect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  });

  return {
    get stats() { return stats; },
    get loading() { return loading; },
    refresh
  };
}

/**
 * Hook for performance metrics
 */
export function usePerformanceMetrics() {
  let metrics = $state<any[]>([]);
  let webVitals = $state(metricsCollector.getWebVitals());
  
  $effect(() => {
    const unsubscribe = metricsCollector.subscribe((updatedMetrics) => {
      metrics = updatedMetrics;
      webVitals = metricsCollector.getWebVitals();
    });
    
    return unsubscribe;
  });

  function recordTiming(name: string, duration: number, tags?: Record<string, string>) {
    metricsCollector.recordTiming(name, duration, tags);
  }

  async function measureAsync<T>(
    name: string,
    fn: () => Promise<T>,
    tags?: Record<string, string>
  ): Promise<T> {
    return metricsCollector.measureAsync(name, fn, tags);
  }

  function measureSync<T>(
    name: string,
    fn: () => T,
    tags?: Record<string, string>
  ): T {
    return metricsCollector.measureSync(name, fn, tags);
  }

  return {
    get metrics() { return metrics; },
    get webVitals() { return webVitals; },
    recordTiming,
    measureAsync,
    measureSync
  };
}