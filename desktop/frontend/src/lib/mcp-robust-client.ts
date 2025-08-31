/**
 * Robust MCP Client with retry patterns, caching, and graceful degradation
 * Based on best practices from SvelteKit desktop guide
 */

import { invoke } from '@tauri-apps/api/core';
import { writable, type Writable, get } from 'svelte/store';
import { browser } from '$app/environment';

// Result type for error handling (error-as-values pattern)
export type Result<T> = { ok: true; data: T } | { ok: false; error: string };

// Connection status
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error' | 'degraded';

// Cache entry type
interface CacheEntry<T> {
    data: T;
    timestamp: number;
    hits: number;
}

// Retry configuration
interface RetryConfig {
    maxAttempts: number;
    backoffMs: number;
    maxBackoffMs: number;
    jitter: boolean;
}

export class RobustMCPClient {
    private initialized = false;
    private cache = new Map<string, CacheEntry<any>>();
    private cacheTTL = 5 * 60 * 1000; // 5 minutes default
    private maxCacheSize = 100;
    
    // Retry configuration
    private retryConfig: RetryConfig = {
        maxAttempts: 3,
        backoffMs: 1000,
        maxBackoffMs: 10000,
        jitter: true
    };
    
    // Performance metrics
    private metrics = {
        totalCalls: 0,
        successfulCalls: 0,
        failedCalls: 0,
        cacheHits: 0,
        averageLatency: 0
    };
    
    // Stores
    public status: Writable<ConnectionStatus> = writable('disconnected');
    public lastError: Writable<string | null> = writable(null);
    public isLoading: Writable<boolean> = writable(false);
    
    constructor() {
        // Auto-connect if in Tauri environment
        if (browser && window.__TAURI__) {
            this.connect();
        }
        
        // Periodic cache cleanup
        if (browser) {
            setInterval(() => this.cleanupCache(), 60000); // Every minute
        }
    }
    
    /**
     * Connect to the MCP server
     */
    async connect(): Promise<Result<void>> {
        try {
            this.status.set('connecting');
            
            // Start the MCP backend
            await invoke('start_mcp_backend');
            
            // Verify connection
            const result = await this.callWithRetry('server_info', {});
            
            if (result.ok) {
                this.initialized = true;
                this.status.set('connected');
                return { ok: true, data: undefined };
            }
            
            this.status.set('error');
            return result;
            
        } catch (e) {
            const error = this.formatError(e);
            this.status.set('error');
            this.lastError.set(error);
            return { ok: false, error };
        }
    }
    
    /**
     * Disconnect from the MCP server
     */
    async disconnect(): Promise<void> {
        try {
            await invoke('stop_mcp_backend');
            this.initialized = false;
            this.status.set('disconnected');
            this.cache.clear();
        } catch (e) {
            console.error('Error stopping MCP backend:', e);
        }
    }
    
    /**
     * Call an MCP method with retry logic and caching
     */
    async callWithRetry<T>(
        method: string,
        params: any = {},
        options: { 
            skipCache?: boolean;
            cacheTTL?: number;
            fallback?: T;
        } = {}
    ): Promise<Result<T>> {
        const startTime = performance.now();
        this.metrics.totalCalls++;
        
        // Check cache first (unless skipped)
        if (!options.skipCache) {
            const cached = this.getFromCache<T>(method, params);
            if (cached) {
                this.metrics.cacheHits++;
                return { ok: true, data: cached };
            }
        }
        
        // Attempt call with retries
        const result = await this.callWithExponentialBackoff<T>(
            method,
            params,
            1,
            options.fallback
        );
        
        // Update metrics
        const latency = performance.now() - startTime;
        this.updateLatencyMetric(latency);
        
        if (result.ok) {
            this.metrics.successfulCalls++;
            // Cache successful results
            this.saveToCache(method, params, result.data, options.cacheTTL);
        } else {
            this.metrics.failedCalls++;
        }
        
        return result;
    }
    
    /**
     * Internal call with exponential backoff
     */
    private async callWithExponentialBackoff<T>(
        method: string,
        params: any,
        attempt: number,
        fallback?: T
    ): Promise<Result<T>> {
        try {
            this.isLoading.set(true);
            
            // Make the actual call
            const result = await invoke<T>('mcp_call', { method, params });
            
            this.isLoading.set(false);
            return { ok: true, data: result };
            
        } catch (error) {
            this.isLoading.set(false);
            const errorStr = this.formatError(error);
            
            console.warn(`MCP call failed (attempt ${attempt}/${this.retryConfig.maxAttempts}):`, {
                method,
                params: this.sanitizeForLogging(params),
                error: errorStr
            });
            
            // Check if we should retry
            if (attempt < this.retryConfig.maxAttempts && this.isRetriableError(errorStr)) {
                const delay = this.calculateBackoff(attempt);
                await this.delay(delay);
                
                return this.callWithExponentialBackoff(method, params, attempt + 1, fallback);
            }
            
            // Try fallback if provided
            if (fallback !== undefined) {
                console.info('Using fallback value for failed MCP call:', method);
                this.status.set('degraded');
                return { ok: true, data: fallback };
            }
            
            // Check for cached stale data as last resort
            const staleCache = this.getFromCache<T>(method, params, true);
            if (staleCache) {
                console.info('Using stale cache for failed MCP call:', method);
                this.status.set('degraded');
                return { ok: true, data: staleCache };
            }
            
            this.lastError.set(errorStr);
            return { ok: false, error: errorStr };
        }
    }
    
    /**
     * Calculate exponential backoff with jitter
     */
    private calculateBackoff(attempt: number): number {
        let delay = Math.min(
            this.retryConfig.backoffMs * Math.pow(2, attempt - 1),
            this.retryConfig.maxBackoffMs
        );
        
        if (this.retryConfig.jitter) {
            // Add random jitter (±25%)
            const jitter = delay * 0.25 * (Math.random() * 2 - 1);
            delay += jitter;
        }
        
        return Math.max(delay, 0);
    }
    
    /**
     * Check if error is retriable
     */
    private isRetriableError(error: string): boolean {
        const retriablePatterns = [
            'not running',
            'disconnected',
            'timeout',
            'ECONNREFUSED',
            'EPIPE',
            'ENOTFOUND'
        ];
        
        return retriablePatterns.some(pattern => 
            error.toLowerCase().includes(pattern.toLowerCase())
        );
    }
    
    /**
     * Get from cache
     */
    private getFromCache<T>(
        method: string,
        params: any,
        includeStale: boolean = false
    ): T | null {
        const cacheKey = this.getCacheKey(method, params);
        const entry = this.cache.get(cacheKey);
        
        if (!entry) return null;
        
        const age = Date.now() - entry.timestamp;
        
        if (age <= this.cacheTTL || includeStale) {
            entry.hits++;
            return entry.data;
        }
        
        return null;
    }
    
    /**
     * Save to cache
     */
    private saveToCache<T>(
        method: string,
        params: any,
        data: T,
        customTTL?: number
    ): void {
        // Enforce cache size limit
        if (this.cache.size >= this.maxCacheSize) {
            this.evictLeastUsed();
        }
        
        const cacheKey = this.getCacheKey(method, params);
        this.cache.set(cacheKey, {
            data,
            timestamp: Date.now(),
            hits: 0
        });
        
        // Set custom TTL if provided
        if (customTTL) {
            setTimeout(() => {
                this.cache.delete(cacheKey);
            }, customTTL);
        }
    }
    
    /**
     * Evict least recently used cache entry
     */
    private evictLeastUsed(): void {
        let leastUsed: [string, CacheEntry<any>] | null = null;
        let minScore = Infinity;
        
        for (const entry of this.cache.entries()) {
            const score = entry[1].hits + (Date.now() - entry[1].timestamp) / 1000;
            if (score < minScore) {
                minScore = score;
                leastUsed = entry;
            }
        }
        
        if (leastUsed) {
            this.cache.delete(leastUsed[0]);
        }
    }
    
    /**
     * Clean up expired cache entries
     */
    private cleanupCache(): void {
        const now = Date.now();
        const expired: string[] = [];
        
        for (const [key, entry] of this.cache.entries()) {
            if (now - entry.timestamp > this.cacheTTL) {
                expired.push(key);
            }
        }
        
        expired.forEach(key => this.cache.delete(key));
    }
    
    /**
     * Generate cache key
     */
    private getCacheKey(method: string, params: any): string {
        return `${method}:${JSON.stringify(params)}`;
    }
    
    /**
     * Sanitize parameters for logging
     */
    private sanitizeForLogging(params: any): any {
        const sensitive = ['password', 'token', 'secret', 'key', 'auth'];
        const sanitized = { ...params };
        
        for (const key of Object.keys(sanitized)) {
            if (sensitive.some(s => key.toLowerCase().includes(s))) {
                sanitized[key] = '[REDACTED]';
            }
        }
        
        return sanitized;
    }
    
    /**
     * Format error for consistency
     */
    private formatError(error: any): string {
        if (typeof error === 'string') return error;
        if (error?.message) return error.message;
        if (error?.toString) return error.toString();
        return 'Unknown error occurred';
    }
    
    /**
     * Delay helper
     */
    private delay(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * Update latency metric
     */
    private updateLatencyMetric(latency: number): void {
        const weight = 0.1; // Exponential moving average weight
        this.metrics.averageLatency = 
            this.metrics.averageLatency * (1 - weight) + latency * weight;
    }
    
    /**
     * Get performance metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            cacheHitRate: this.metrics.cacheHits / Math.max(1, this.metrics.totalCalls),
            successRate: this.metrics.successfulCalls / Math.max(1, this.metrics.totalCalls),
            cacheSize: this.cache.size
        };
    }
    
    /**
     * Clear all caches
     */
    clearCache(): void {
        this.cache.clear();
    }
    
    /**
     * Reset metrics
     */
    resetMetrics(): void {
        this.metrics = {
            totalCalls: 0,
            successfulCalls: 0,
            failedCalls: 0,
            cacheHits: 0,
            averageLatency: 0
        };
    }
    
    /**
     * Get health status
     */
    async getHealth(): Promise<{ status: ConnectionStatus; metrics: any; errors: string[] }> {
        const errors: string[] = [];
        const currentError = get(this.lastError);
        
        if (currentError) {
            errors.push(currentError);
        }
        
        return {
            status: get(this.status),
            metrics: this.getMetrics(),
            errors
        };
    }
    
    // Convenience methods with proper typing and fallbacks
    
    async search(query: string, options: any = {}): Promise<Result<any>> {
        return this.callWithRetry('search', { query, ...options }, {
            fallback: { results: [], query, error: 'Search temporarily unavailable' }
        });
    }
    
    async rollDice(notation: string): Promise<Result<any>> {
        // Don't cache dice rolls
        return this.callWithRetry('roll_dice', { notation }, {
            skipCache: true
        });
    }
    
    async listSources(): Promise<Result<any>> {
        return this.callWithRetry('list_sources', {}, {
            cacheTTL: 10 * 60 * 1000, // Cache for 10 minutes
            fallback: { sources: [] }
        });
    }
}

// Create singleton instance with lazy initialization
let mcpClientInstance: RobustMCPClient | null = null;

export function getMCPClient(): RobustMCPClient {
    if (!mcpClientInstance && browser) {
        mcpClientInstance = new RobustMCPClient();
    }
    return mcpClientInstance!;
}

// Export convenience stores
export const mcpStatus = writable<ConnectionStatus>('disconnected');
export const mcpError = writable<string | null>(null);
export const mcpLoading = writable<boolean>(false);