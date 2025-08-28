/**
 * Unified cache manager that coordinates memory and persistent caching
 */

import { MemoryCache } from './memory-cache';
import { IndexedDBCache } from './indexed-db';
import type { 
  CacheConfig, 
  CacheKey, 
  CacheWarmingStrategy,
  CacheInvalidationRule,
  CacheObserver,
  CacheEvent 
} from './types';

export class CacheManager {
  private memoryCache: MemoryCache;
  private persistentCache: IndexedDBCache;
  private config: Map<string, CacheConfig> = new Map();
  private warmingQueue: Set<string> = new Set();
  private invalidationRules: Map<string, CacheInvalidationRule> = new Map();
  private requestDeduplication: Map<string, Promise<any>> = new Map();

  constructor() {
    this.memoryCache = new MemoryCache();
    this.persistentCache = new IndexedDBCache();
    this.init();
  }

  private async init(): Promise<void> {
    await this.persistentCache.init();
    
    // Start periodic cleanup
    setInterval(() => {
      this.cleanupExpired();
    }, 60000); // Every minute

    // Load warming strategies
    this.startWarmingScheduler();
  }

  /**
   * Generate cache key from components
   */
  generateKey(key: CacheKey): string {
    const parts = [key.namespace, key.resource];
    
    if (key.params) {
      const sortedParams = Object.keys(key.params)
        .sort()
        .map(k => `${k}=${JSON.stringify(key.params![k])}`)
        .join('&');
      parts.push(sortedParams);
    }
    
    if (key.version) {
      parts.push(key.version);
    }
    
    return parts.join(':');
  }

  /**
   * Get value from cache (memory first, then persistent)
   */
  async get<T>(key: string | CacheKey): Promise<T | null> {
    const cacheKey = typeof key === 'string' ? key : this.generateKey(key);
    
    // Check memory cache first
    const memoryResult = this.memoryCache.get<T>(cacheKey);
    if (memoryResult !== null) {
      return memoryResult;
    }
    
    // Check persistent cache
    const persistentResult = await this.persistentCache.get<T>(cacheKey);
    if (persistentResult) {
      // Promote to memory cache
      this.memoryCache.set(
        cacheKey, 
        persistentResult.value, 
        persistentResult.expiresAt - Date.now()
      );
      return persistentResult.value;
    }
    
    return null;
  }

  /**
   * Set value in cache with configuration
   */
  async set<T>(
    key: string | CacheKey, 
    value: T, 
    config?: Partial<CacheConfig>
  ): Promise<void> {
    const cacheKey = typeof key === 'string' ? key : this.generateKey(key);
    const finalConfig = this.mergeConfig(cacheKey, config);
    
    // Set in memory cache
    this.memoryCache.set(cacheKey, value, finalConfig.ttl);
    
    // Set in persistent cache if configured
    if (finalConfig.persistent) {
      const size = this.estimateSize(value);
      await this.persistentCache.set({
        key: cacheKey,
        value,
        timestamp: Date.now(),
        expiresAt: Date.now() + finalConfig.ttl,
        size,
        hits: 0,
        lastAccessed: Date.now(),
        metadata: typeof key === 'object' ? { namespace: key.namespace } : {}
      });
    }
  }

  /**
   * Delete from both caches
   */
  async delete(key: string | CacheKey): Promise<boolean> {
    const cacheKey = typeof key === 'string' ? key : this.generateKey(key);
    
    const memoryDeleted = this.memoryCache.delete(cacheKey);
    await this.persistentCache.delete(cacheKey);
    
    return memoryDeleted;
  }

  /**
   * Invalidate cache entries by pattern
   */
  async invalidate(pattern: string | RegExp | CacheInvalidationRule): Promise<number> {
    let invalidated = 0;
    
    if (typeof pattern === 'string' || pattern instanceof RegExp) {
      // Simple pattern invalidation
      invalidated = this.memoryCache.invalidateByPattern(pattern);
      
      // Also invalidate in persistent cache
      const keys = await this.getKeysByPattern(pattern);
      for (const key of keys) {
        await this.persistentCache.delete(key);
      }
    } else {
      // Rule-based invalidation
      const rule = pattern as CacheInvalidationRule;
      
      if (rule.pattern) {
        invalidated = await this.invalidate(rule.pattern);
      }
      
      if (rule.dependencies) {
        for (const dep of rule.dependencies) {
          invalidated += await this.invalidate(dep);
        }
      }
      
      if (rule.tags) {
        for (const tag of rule.tags) {
          invalidated += await this.invalidateByTag(tag);
        }
      }
    }
    
    return invalidated;
  }

  /**
   * Cache wrapper with deduplication
   */
  async withCache<T>(
    key: string | CacheKey,
    fetcher: () => Promise<T>,
    config?: Partial<CacheConfig>
  ): Promise<T> {
    const cacheKey = typeof key === 'string' ? key : this.generateKey(key);
    
    // Check cache first
    const cached = await this.get<T>(cacheKey);
    if (cached !== null) {
      return cached;
    }
    
    // Check if request is already in flight (deduplication)
    if (this.requestDeduplication.has(cacheKey)) {
      return this.requestDeduplication.get(cacheKey)!;
    }
    
    // Make the request
    const promise = fetcher()
      .then(async (result) => {
        await this.set(cacheKey, result, config);
        this.requestDeduplication.delete(cacheKey);
        return result;
      })
      .catch(error => {
        this.requestDeduplication.delete(cacheKey);
        throw error;
      });
    
    this.requestDeduplication.set(cacheKey, promise);
    return promise;
  }

  /**
   * Batch get multiple keys
   */
  async batchGet<T>(keys: Array<string | CacheKey>): Promise<Map<string, T | null>> {
    const results = new Map<string, T | null>();
    
    await Promise.all(
      keys.map(async (key) => {
        const cacheKey = typeof key === 'string' ? key : this.generateKey(key);
        const value = await this.get<T>(key);
        results.set(cacheKey, value);
      })
    );
    
    return results;
  }

  /**
   * Batch set multiple entries
   */
  async batchSet<T>(
    entries: Array<{ key: string | CacheKey; value: T; config?: Partial<CacheConfig> }>
  ): Promise<void> {
    await Promise.all(
      entries.map(entry => this.set(entry.key, entry.value, entry.config))
    );
  }

  /**
   * Warm cache with predefined strategies
   */
  async warmCache(strategy: CacheWarmingStrategy): Promise<void> {
    const warmingTasks = strategy.resources.map(async (resource) => {
      if (this.warmingQueue.has(resource)) return;
      
      this.warmingQueue.add(resource);
      
      try {
        // Implementation depends on resource type
        // This is a placeholder for actual warming logic
        console.log(`Warming cache for resource: ${resource}`);
      } catch (error) {
        console.error(`Failed to warm cache for ${resource}:`, error);
      } finally {
        this.warmingQueue.delete(resource);
      }
    });
    
    if (strategy.schedule === 'immediate') {
      await Promise.all(warmingTasks);
    } else if (strategy.schedule === 'idle') {
      if ('requestIdleCallback' in window) {
        requestIdleCallback(() => {
          Promise.all(warmingTasks);
        });
      } else {
        setTimeout(() => Promise.all(warmingTasks), 0);
      }
    } else {
      // Background
      Promise.all(warmingTasks);
    }
  }

  /**
   * Preload resources based on user patterns
   */
  async predictivePreload(currentResource: string): Promise<void> {
    // This would analyze user patterns and preload likely next resources
    // Implementation would depend on analytics and user behavior tracking
    const predictions = this.getPredictedResources(currentResource);
    
    for (const resource of predictions) {
      // Preload in background
      this.warmCache({
        resources: [resource],
        priority: 'low',
        schedule: 'background'
      });
    }
  }

  /**
   * Subscribe to cache events
   */
  subscribe(observer: CacheObserver): () => void {
    return this.memoryCache.subscribe(observer);
  }

  /**
   * Get cache statistics
   */
  async getStats() {
    const memoryStats = this.memoryCache.getStats();
    const persistentStats = await this.persistentCache.getStats();
    const sizeInfo = this.memoryCache.getSizeInfo();
    
    return {
      memory: {
        ...memoryStats,
        sizeInfo
      },
      persistent: persistentStats,
      deduplicationActive: this.requestDeduplication.size,
      warmingQueueSize: this.warmingQueue.size
    };
  }

  /**
   * Clear all caches
   */
  async clear(): Promise<void> {
    this.memoryCache.clear();
    await this.persistentCache.clear();
    this.requestDeduplication.clear();
  }

  private mergeConfig(key: string, config?: Partial<CacheConfig>): CacheConfig {
    const defaultConfig: CacheConfig = {
      ttl: 3600000, // 1 hour
      strategy: 'LRU',
      persistent: false
    };
    
    const storedConfig = this.config.get(key) || {};
    
    return {
      ...defaultConfig,
      ...storedConfig,
      ...config
    };
  }

  private async getKeysByPattern(pattern: string | RegExp): Promise<string[]> {
    // This would query IndexedDB for matching keys
    // For now, returning memory cache keys
    return this.memoryCache.keys().filter(key => {
      const regex = typeof pattern === 'string' 
        ? new RegExp(pattern.replace(/\*/g, '.*'))
        : pattern;
      return regex.test(key);
    });
  }

  private async invalidateByTag(tag: string): Promise<number> {
    // This would invalidate all entries with a specific tag
    // Tags would be stored in metadata
    return 0; // Placeholder
  }

  private getPredictedResources(currentResource: string): string[] {
    // This would use ML or heuristics to predict next resources
    // For now, returning empty array
    return [];
  }

  private async cleanupExpired(): Promise<void> {
    this.memoryCache.cleanupExpired();
    await this.persistentCache.deleteExpired();
  }

  private startWarmingScheduler(): void {
    // This would periodically warm high-priority caches
    setInterval(() => {
      // Placeholder for warming logic
    }, 300000); // Every 5 minutes
  }

  private estimateSize(value: unknown): number {
    if (typeof value === 'string') {
      return value.length * 2;
    } else if (value instanceof ArrayBuffer) {
      return value.byteLength;
    } else {
      try {
        return JSON.stringify(value).length * 2;
      } catch {
        return 1024;
      }
    }
  }
}