/**
 * In-memory cache with LRU eviction
 */

import type { CacheEntry, CacheStats, CacheEvent, CacheObserver } from './types';

export class MemoryCache {
  private cache = new Map<string, CacheEntry>();
  private accessOrder: string[] = [];
  private maxSize: number;
  private currentSize = 0;
  private stats: CacheStats = {
    hits: 0,
    misses: 0,
    evictions: 0,
    size: 0,
    itemCount: 0,
    hitRate: 0
  };
  private observers: Set<CacheObserver> = new Set();

  constructor(maxSize: number = 50 * 1024 * 1024) { // 50MB default
    this.maxSize = maxSize;
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    
    if (!entry) {
      this.stats.misses++;
      this.updateHitRate();
      this.notifyObservers({ type: 'miss', key, timestamp: Date.now() });
      return null;
    }

    // Check expiration
    if (entry.expiresAt <= Date.now()) {
      this.delete(key);
      this.stats.misses++;
      this.updateHitRate();
      this.notifyObservers({ type: 'expire', key, timestamp: Date.now() });
      return null;
    }

    // Update access order (LRU)
    this.updateAccessOrder(key);
    
    // Update entry stats
    entry.lastAccessed = Date.now();
    entry.hits++;
    
    this.stats.hits++;
    this.updateHitRate();
    this.notifyObservers({ type: 'hit', key, timestamp: Date.now() });
    
    return entry.value as T;
  }

  set<T>(key: string, value: T, ttl: number = 3600000): void { // 1 hour default TTL
    const size = this.estimateSize(value);
    
    // Check if we need to evict entries
    while (this.currentSize + size > this.maxSize && this.accessOrder.length > 0) {
      this.evictLRU();
    }

    const entry: CacheEntry<T> = {
      key,
      value,
      timestamp: Date.now(),
      expiresAt: Date.now() + ttl,
      size,
      hits: 0,
      lastAccessed: Date.now()
    };

    // Remove old entry if it exists
    if (this.cache.has(key)) {
      const oldEntry = this.cache.get(key)!;
      this.currentSize -= oldEntry.size;
    }

    this.cache.set(key, entry);
    this.currentSize += size;
    this.updateAccessOrder(key);
    this.updateStats();
    
    this.notifyObservers({ type: 'set', key, timestamp: Date.now() });
  }

  delete(key: string): boolean {
    const entry = this.cache.get(key);
    
    if (!entry) return false;
    
    this.cache.delete(key);
    this.currentSize -= entry.size;
    this.accessOrder = this.accessOrder.filter(k => k !== key);
    this.updateStats();
    
    this.notifyObservers({ type: 'delete', key, timestamp: Date.now() });
    
    return true;
  }

  clear(): void {
    const keys = Array.from(this.cache.keys());
    this.cache.clear();
    this.accessOrder = [];
    this.currentSize = 0;
    this.updateStats();
    
    keys.forEach(key => {
      this.notifyObservers({ type: 'clear', key, timestamp: Date.now() });
    });
  }

  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;
    
    // Check expiration
    if (entry.expiresAt <= Date.now()) {
      this.delete(key);
      return false;
    }
    
    return true;
  }

  getStats(): CacheStats {
    return { ...this.stats };
  }

  subscribe(observer: CacheObserver): () => void {
    this.observers.add(observer);
    return () => this.observers.delete(observer);
  }

  // Get entries by pattern
  getByPattern(pattern: string | RegExp): Map<string, CacheEntry> {
    const regex = typeof pattern === 'string' 
      ? new RegExp(pattern.replace(/\*/g, '.*'))
      : pattern;
    
    const results = new Map<string, CacheEntry>();
    
    for (const [key, entry] of this.cache.entries()) {
      if (regex.test(key) && entry.expiresAt > Date.now()) {
        results.set(key, entry);
      }
    }
    
    return results;
  }

  // Invalidate by pattern
  invalidateByPattern(pattern: string | RegExp): number {
    const entries = this.getByPattern(pattern);
    let count = 0;
    
    for (const key of entries.keys()) {
      if (this.delete(key)) count++;
    }
    
    return count;
  }

  // Get all keys
  keys(): string[] {
    return Array.from(this.cache.keys()).filter(key => {
      const entry = this.cache.get(key);
      return entry && entry.expiresAt > Date.now();
    });
  }

  // Get cache size info
  getSizeInfo(): { used: number; max: number; percentage: number } {
    return {
      used: this.currentSize,
      max: this.maxSize,
      percentage: (this.currentSize / this.maxSize) * 100
    };
  }

  private evictLRU(): void {
    if (this.accessOrder.length === 0) return;
    
    const keyToEvict = this.accessOrder[0];
    const entry = this.cache.get(keyToEvict);
    
    if (entry) {
      this.cache.delete(keyToEvict);
      this.currentSize -= entry.size;
      this.accessOrder.shift();
      this.stats.evictions++;
      this.updateStats();
      
      this.notifyObservers({ 
        type: 'evict', 
        key: keyToEvict, 
        timestamp: Date.now(),
        metadata: { reason: 'LRU' }
      });
    }
  }

  private updateAccessOrder(key: string): void {
    // Remove key from current position
    this.accessOrder = this.accessOrder.filter(k => k !== key);
    // Add to end (most recently used)
    this.accessOrder.push(key);
  }

  private estimateSize(value: unknown): number {
    // Simple size estimation
    if (typeof value === 'string') {
      return value.length * 2; // 2 bytes per character
    } else if (value instanceof ArrayBuffer) {
      return value.byteLength;
    } else if (value instanceof Blob) {
      return value.size;
    } else {
      // For objects, use JSON stringify as estimation
      try {
        return JSON.stringify(value).length * 2;
      } catch {
        return 1024; // Default 1KB for non-serializable objects
      }
    }
  }

  private updateStats(): void {
    this.stats.size = this.currentSize;
    this.stats.itemCount = this.cache.size;
  }

  private updateHitRate(): void {
    const total = this.stats.hits + this.stats.misses;
    this.stats.hitRate = total > 0 ? this.stats.hits / total : 0;
  }

  private notifyObservers(event: CacheEvent): void {
    this.observers.forEach(observer => {
      try {
        observer.onCacheEvent(event);
      } catch (error) {
        console.error('Cache observer error:', error);
      }
    });
  }

  // Clean up expired entries
  cleanupExpired(): number {
    const now = Date.now();
    let count = 0;
    
    for (const [key, entry] of this.cache.entries()) {
      if (entry.expiresAt <= now) {
        this.delete(key);
        count++;
      }
    }
    
    return count;
  }

  // Get most frequently used entries
  getMostFrequent(limit: number = 10): Array<[string, CacheEntry]> {
    return Array.from(this.cache.entries())
      .filter(([_, entry]) => entry.expiresAt > Date.now())
      .sort((a, b) => b[1].hits - a[1].hits)
      .slice(0, limit);
  }

  // Get least recently used entries
  getLeastRecent(limit: number = 10): Array<[string, CacheEntry]> {
    return this.accessOrder
      .slice(0, limit)
      .map(key => [key, this.cache.get(key)])
      .filter((entry): entry is [string, CacheEntry] => entry[1] !== undefined);
  }
}