/**
 * Robust MCP Client with retry patterns, caching, and graceful degradation
 * Based on best practices from SvelteKit desktop guide
 */

import { invoke } from '@tauri-apps/api/core';
import { writable, type Writable, get } from 'svelte/store';
import { browser } from '$app/environment';

// Enhanced Result type for error handling (error-as-values pattern)
export interface Result<T, E = string> {
  readonly ok: boolean;
  readonly data?: T;
  readonly error?: E;
  readonly metadata?: Record<string, unknown>;
}

export type Success<T> = { ok: true; data: T; metadata?: Record<string, unknown> };
export type Failure<E = string> = { ok: false; error: E; metadata?: Record<string, unknown> };

// Helper functions for Result type
export const createSuccess = <T>(data: T, metadata?: Record<string, unknown>): Success<T> => ({ 
  ok: true, 
  data, 
  ...(metadata && { metadata }) 
});

export const createFailure = <E = string>(error: E, metadata?: Record<string, unknown>): Failure<E> => ({ 
  ok: false, 
  error, 
  ...(metadata && { metadata }) 
});

// Result utility functions
export const isSuccess = <T, E>(result: Result<T, E>): result is Success<T> => result.ok;
export const isFailure = <T, E>(result: Result<T, E>): result is Failure<E> => !result.ok;

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
        if (browser && (window as any).__TAURI__) {
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
            return result as Result<void, string>;
            
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
        params: Record<string, unknown> = {},
        options: { 
            skipCache?: boolean;
            cacheTTL?: number;
            fallback?: T;
            timeout?: number;
            retryAttempts?: number;
        } = {}
    ): Promise<Result<T>> {
        const startTime = performance.now();
        this.metrics.totalCalls++;
        
        // Check cache first (unless skipped)
        if (!options.skipCache) {
            const cached = this.getFromCache<T>(method, params);
            if (cached) {
                this.metrics.cacheHits++;
                return createSuccess(cached, { cached: true, method });
            }
        }
        
        // Attempt call with retries
        const result = await this.callWithExponentialBackoff<T>(
            method,
            params,
            1,
            options.fallback,
            options.retryAttempts ?? this.retryConfig.maxAttempts,
            options.timeout
        );
        
        // Update metrics
        const latency = performance.now() - startTime;
        this.updateLatencyMetric(latency);
        
        if (result.ok && result.data !== undefined) {
            this.metrics.successfulCalls++;
            // Cache successful results
            this.saveToCache(method, params, result.data, options.cacheTTL);
        } else {
            this.metrics.failedCalls++;
        }
        
        return result;
    }
    
    /**
     * Internal call with exponential backoff and comprehensive error handling
     */
    private async callWithExponentialBackoff<T>(
        method: string,
        params: Record<string, unknown>,
        attempt: number,
        fallback?: T,
        maxAttempts?: number,
        timeout?: number
    ): Promise<Result<T>> {
        const actualMaxAttempts = maxAttempts ?? this.retryConfig.maxAttempts;
        
        try {
            this.isLoading.set(true);
            
            // Create timeout promise if specified
            let callPromise: Promise<T> = invoke<T>('mcp_call', { method, params });
            
            if (timeout) {
                const timeoutPromise = new Promise<never>((_, reject) =>
                    setTimeout(() => reject(new Error(`Call timeout after ${timeout}ms`)), timeout)
                );
                callPromise = Promise.race([callPromise, timeoutPromise]);
            }
            
            // Make the actual call
            const result = await callPromise;
            
            this.isLoading.set(false);
            return createSuccess(result, {
                method,
                attempt,
                timestamp: Date.now()
            });
            
        } catch (error) {
            this.isLoading.set(false);
            const errorInfo = this.categorizeError(error);
            
            console.warn(`MCP call failed (attempt ${attempt}/${actualMaxAttempts}):`, {
                method,
                params: this.sanitizeForLogging(params),
                error: errorInfo.message,
                category: errorInfo.category,
                retriable: errorInfo.retriable
            });
            
            // Check if we should retry
            if (attempt < actualMaxAttempts && errorInfo.retriable) {
                const delay = this.calculateBackoff(attempt);
                await this.delay(delay);
                
                return this.callWithExponentialBackoff(
                    method, 
                    params, 
                    attempt + 1, 
                    fallback, 
                    maxAttempts, 
                    timeout
                );
            }
            
            // Try fallback if provided
            if (fallback !== undefined) {
                console.info('Using fallback value for failed MCP call:', method);
                this.status.set('degraded');
                return createSuccess(fallback, {
                    method,
                    fallback: true,
                    originalError: errorInfo.message
                });
            }
            
            // Check for cached stale data as last resort
            const staleCache = this.getFromCache<T>(method, params, true);
            if (staleCache) {
                console.info('Using stale cache for failed MCP call:', method);
                this.status.set('degraded');
                return createSuccess(staleCache, {
                    method,
                    staleCache: true,
                    originalError: errorInfo.message
                });
            }
            
            this.lastError.set(errorInfo.message);
            return createFailure(errorInfo.message, {
                method,
                attempt,
                maxAttempts: actualMaxAttempts,
                category: errorInfo.category,
                retriable: errorInfo.retriable,
                timestamp: Date.now()
            });
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
     * Categorize errors for better handling
     */
    private categorizeError(error: unknown): {
        message: string;
        category: 'network' | 'timeout' | 'process' | 'validation' | 'unknown';
        retriable: boolean;
    } {
        const message = this.formatError(error);
        const lowerMessage = message.toLowerCase();
        
        // Network errors (retriable)
        if (lowerMessage.includes('econnrefused') || 
            lowerMessage.includes('enotfound') ||
            lowerMessage.includes('network') ||
            lowerMessage.includes('dns')) {
            return { message, category: 'network', retriable: true };
        }
        
        // Timeout errors (retriable)
        if (lowerMessage.includes('timeout') ||
            lowerMessage.includes('timed out')) {
            return { message, category: 'timeout', retriable: true };
        }
        
        // Process errors (retriable)
        if (lowerMessage.includes('not running') ||
            lowerMessage.includes('disconnected') ||
            lowerMessage.includes('epipe') ||
            lowerMessage.includes('process died')) {
            return { message, category: 'process', retriable: true };
        }
        
        // Validation errors (not retriable)
        if (lowerMessage.includes('invalid') ||
            lowerMessage.includes('validation') ||
            lowerMessage.includes('malformed') ||
            lowerMessage.includes('bad request')) {
            return { message, category: 'validation', retriable: false };
        }
        
        // Default to unknown (not retriable to be safe)
        return { message, category: 'unknown', retriable: false };
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
            // Update timestamp for LRU tracking
            entry.timestamp = Date.now();
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
        let oldestKey: string | null = null;
        let oldestTimestamp = Infinity;

        // Find the entry with the oldest timestamp (Least Recently Used)
        for (const [key, entry] of this.cache.entries()) {
            if (entry.timestamp < oldestTimestamp) {
                oldestTimestamp = entry.timestamp;
                oldestKey = key;
            }
        }

        if (oldestKey) {
            this.cache.delete(oldestKey);
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
     * Enhanced error formatting with context preservation
     */
    private formatError(error: unknown): string {
        if (typeof error === 'string') {
            return error;
        }
        
        if (error instanceof Error) {
            // Include stack trace in development
            if (import.meta.env.DEV && error.stack) {
                return `${error.message}\nStack: ${error.stack}`;
            }
            return error.message;
        }
        
        if (error && typeof error === 'object') {
            // Handle Tauri error format
            if ('message' in error && typeof error.message === 'string') {
                return error.message;
            }
            
            // Try toString
            if ('toString' in error && typeof error.toString === 'function') {
                return error.toString();
            }
            
            // Fallback to JSON stringification
            try {
                return JSON.stringify(error);
            } catch {
                return '[Object object - could not serialize]';
            }
        }
        
        return `Unknown error occurred (type: ${typeof error})`;
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
    
    // Enhanced convenience methods with proper typing and fallbacks
    
    async search(
        query: string, 
        options: Record<string, unknown> = {}
    ): Promise<Result<{
        results: Array<{
            title: string;
            content: string;
            source: string;
            relevance: number;
            page?: number;
        }>;
        query: string;
        total: number;
        metadata: Record<string, unknown>;
    }>> {
        return this.callWithRetry('search', { query, ...options }, {
            fallback: {
                results: [],
                query,
                total: 0,
                metadata: { error: 'Search temporarily unavailable' }
            },
            timeout: 15000 // 15 second timeout for searches
        });
    }
    
    async rollDice(notation: string): Promise<Result<{
        notation: string;
        result: number;
        breakdown: Array<{ die: string; roll: number }>;
        timestamp: string;
    }>> {
        // Don't cache dice rolls - they should always be fresh
        return this.callWithRetry('roll_dice', { notation }, {
            skipCache: true,
            retryAttempts: 2, // Fewer retries for dice rolls
            timeout: 5000 // Quick timeout
        });
    }
    
    async listSources(): Promise<Result<Array<{
        id: string;
        name: string;
        system: string;
        pageCount: number;
        status: 'processing' | 'ready' | 'error';
        addedAt: string;
    }>>> {
        return this.callWithRetry('list_sources', {}, {
            cacheTTL: 10 * 60 * 1000, // Cache for 10 minutes
            fallback: [],
            timeout: 10000
        });
    }
    
    /**
     * Add source document (PDF rulebook)
     */
    async addSource(
        pdfPath: string,
        rulebookName: string,
        system: string,
        metadata?: Record<string, unknown>
    ): Promise<Result<{ sourceId: string; pagesProcessed: number }>> {
        return this.callWithRetry('add_source', {
            pdf_path: pdfPath,
            rulebook_name: rulebookName,
            system,
            ...(metadata && { metadata })
        }, {
            timeout: 30000, // PDF processing can take time
            retryAttempts: 1 // Don't retry file operations
        });
    }
    
    /**
     * Batch multiple calls with error isolation
     */
    async batchCall<T extends Record<string, { method: string; params: Record<string, unknown> }>>(
        calls: T,
        options: {
            continueOnError?: boolean;
            timeout?: number;
        } = {}
    ): Promise<{ [K in keyof T]: Result<unknown> }> {
        const results = {} as { [K in keyof T]: Result<unknown> };
        const { continueOnError = true, timeout } = options;
        
        for (const [key, call] of Object.entries(calls) as Array<[keyof T, typeof calls[keyof T]]>) {
            try {
                results[key] = await this.callWithRetry(call.method, call.params, timeout ? { timeout } : {});
                
                // If we get an error and shouldn't continue, break early
                if (!continueOnError && !results[key]!.ok) {
                    break;
                }
            } catch (error) {
                results[key] = createFailure(this.formatError(error), {
                    batchCall: true,
                    key: key as string
                });
                
                if (!continueOnError) {
                    break;
                }
            }
        }
        
        return results;
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
export const mcpStatus = browser ? getMCPClient().status : writable<ConnectionStatus>('disconnected');
export const mcpError = browser ? getMCPClient().lastError : writable<string | null>(null);
export const mcpLoading = browser ? getMCPClient().isLoading : writable<boolean>(false);