/**
 * Cache system types and interfaces
 */

export interface CacheConfig {
  ttl: number; // Time to live in milliseconds
  maxSize?: number; // Maximum cache size in bytes
  strategy?: 'LRU' | 'LFU' | 'FIFO';
  persistent?: boolean; // Use IndexedDB
}

export interface CacheEntry<T = unknown> {
  key: string;
  value: T;
  timestamp: number;
  expiresAt: number;
  size: number;
  hits: number;
  lastAccessed: number;
  metadata?: Record<string, any>;
}

export interface CacheStats {
  hits: number;
  misses: number;
  evictions: number;
  size: number;
  itemCount: number;
  hitRate: number;
}

export interface CacheKey {
  namespace: string;
  resource: string;
  params?: Record<string, any>;
  version?: string;
}

export type CacheInvalidationRule = {
  pattern?: string | RegExp;
  maxAge?: number;
  dependencies?: string[];
  tags?: string[];
};

export interface CacheWarmingStrategy {
  resources: string[];
  priority: 'high' | 'medium' | 'low';
  schedule?: 'immediate' | 'idle' | 'background';
  dependencies?: string[];
}

export interface PerformanceMetrics {
  cacheHitRate: number;
  averageResponseTime: number;
  p95ResponseTime: number;
  p99ResponseTime: number;
  requestsPerSecond: number;
  errorRate: number;
  cacheSize: number;
  memoryUsage: number;
}

export type CacheEventType = 
  | 'hit'
  | 'miss'
  | 'set'
  | 'delete'
  | 'evict'
  | 'expire'
  | 'clear';

export interface CacheEvent {
  type: CacheEventType;
  key: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface CacheObserver {
  onCacheEvent(event: CacheEvent): void;
}