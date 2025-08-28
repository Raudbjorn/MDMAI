/**
 * Optimized API client with caching, batching, and performance monitoring
 */

import { CacheManager } from '$lib/cache/cache-manager';
import { MetricsCollector } from '$lib/performance/metrics-collector';
import { RequestOptimizer } from '$lib/performance/request-optimizer';
import type { CacheConfig } from '$lib/cache/types';

export interface ApiRequestConfig extends RequestInit {
  cache?: Partial<CacheConfig>;
  batch?: boolean;
  debounce?: number;
  retry?: {
    attempts: number;
    delay: number;
    backoff?: number;
  };
}

export class OptimizedApiClient {
  private baseUrl: string;
  private cacheManager: CacheManager;
  private metricsCollector: MetricsCollector;
  private requestOptimizer: RequestOptimizer;
  private defaultHeaders: Record<string, string>;
  private webSocketPool: any = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    this.cacheManager = new CacheManager();
    this.metricsCollector = new MetricsCollector();
    this.requestOptimizer = new RequestOptimizer();
    this.defaultHeaders = {
      'Content-Type': 'application/json'
    };

    this.setupBatching();
    this.setupWebSocketPool();
  }

  private setupBatching() {
    // Configure batching for search endpoint
    this.requestOptimizer.configureBatching('/api/search', {
      maxBatchSize: 10,
      maxWaitTime: 100,
      batchProcessor: async (requests) => {
        const response = await fetch(`${this.baseUrl}/api/search/batch`, {
          method: 'POST',
          headers: this.defaultHeaders,
          body: JSON.stringify({ queries: requests })
        });
        
        if (!response.ok) {
          throw new Error(`Batch request failed: ${response.statusText}`);
        }
        
        return response.json();
      }
    });

    // Configure batching for rules lookup
    this.requestOptimizer.configureBatching('/api/rules', {
      maxBatchSize: 20,
      maxWaitTime: 50,
      batchProcessor: async (requests) => {
        const response = await fetch(`${this.baseUrl}/api/rules/batch`, {
          method: 'POST',
          headers: this.defaultHeaders,
          body: JSON.stringify({ ids: requests })
        });
        
        if (!response.ok) {
          throw new Error(`Batch request failed: ${response.statusText}`);
        }
        
        return response.json();
      }
    });
  }

  private setupWebSocketPool() {
    // Create WebSocket pool for real-time features (singleton)
    if (!this.webSocketPool) {
      this.webSocketPool = this.requestOptimizer.createConnectionPool({
        maxConnections: 3,
        url: `${this.baseUrl.replace('http', 'ws')}/ws`,
        reconnectDelay: 1000
      });
    }
  }

  /**
   * Make an optimized API request
   */
  async request<T>(
    endpoint: string,
    config: ApiRequestConfig = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const cacheKey = this.generateCacheKey(url, config);
    
    // Check if request should be cached
    if (config.method === 'GET' || !config.method) {
      return this.cachedRequest<T>(url, cacheKey, config);
    }
    
    // Non-cacheable request
    return this.directRequest<T>(url, config);
  }

  private async cachedRequest<T>(
    url: string,
    cacheKey: string,
    config: ApiRequestConfig
  ): Promise<T> {
    return this.cacheManager.withCache(
      cacheKey,
      () => this.directRequest<T>(url, config),
      config.cache
    );
  }

  private async directRequest<T>(
    url: string,
    config: ApiRequestConfig
  ): Promise<T> {
    const start = performance.now();
    
    // Apply debouncing if configured
    if (config.debounce) {
      const debouncedFetch = this.requestOptimizer.debounce(
        () => this.fetchWithRetry(url, config),
        { delay: config.debounce }
      );
      
      return debouncedFetch() as Promise<T>;
    }
    
    // Apply deduplication
    return this.requestOptimizer.deduplicate(
      url,
      async () => {
        try {
          const result = await this.fetchWithRetry(url, config);
          
          // Record metrics
          this.metricsCollector.recordTiming(
            'api_request',
            performance.now() - start,
            { endpoint: new URL(url).pathname }
          );
          
          return result as T;
        } catch (error) {
          // Record error metrics
          this.metricsCollector.recordMetric({
            name: 'api_error',
            value: 1,
            timestamp: Date.now(),
            tags: {
              endpoint: new URL(url).pathname,
              error: error instanceof Error ? error.message : 'Unknown error'
            }
          });
          
          throw error;
        }
      }
    );
  }

  private async fetchWithRetry(
    url: string,
    config: ApiRequestConfig
  ): Promise<any> {
    const retry = config.retry || { attempts: 3, delay: 1000, backoff: 2 };
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt < retry.attempts; attempt++) {
      try {
        const response = await fetch(url, {
          ...config,
          headers: {
            ...this.defaultHeaders,
            ...config.headers
          }
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response.json();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error');
        
        if (attempt < retry.attempts - 1) {
          const delay = retry.delay * Math.pow(retry.backoff || 1, attempt);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw lastError;
  }

  /**
   * Batch multiple requests
   */
  async batch<T>(
    endpoint: string,
    requests: any[]
  ): Promise<T[]> {
    const results: T[] = [];
    
    for (const request of requests) {
      const result = await this.requestOptimizer.addToBatch<T>(endpoint, request);
      results.push(result);
    }
    
    return results;
  }

  /**
   * Stream data using Server-Sent Events
   */
  streamSSE(
    endpoint: string,
    onMessage: (data: any) => void,
    onError?: (error: Error) => void
  ): () => void {
    const url = `${this.baseUrl}${endpoint}`;
    const eventSource = new EventSource(url);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse SSE message:', error);
      }
    };
    
    eventSource.onerror = (event) => {
      const error = new Error('SSE connection error');
      onError?.(error);
      
      this.metricsCollector.recordMetric({
        name: 'sse_error',
        value: 1,
        timestamp: Date.now(),
        tags: { endpoint }
      });
    };
    
    // Return cleanup function
    return () => {
      eventSource.close();
    };
  }

  /**
   * WebSocket connection with pooling
   */
  async websocket(
    handler: (ws: WebSocket) => Promise<void>
  ): Promise<void> {
    // Use existing pool instead of creating a new one each time
    if (!this.webSocketPool) {
      this.setupWebSocketPool();
    }
    
    return this.webSocketPool.execute(handler);
  }

  /**
   * Prefetch and warm cache
   */
  async prefetch(endpoints: string[], config?: Partial<CacheConfig>): Promise<void> {
    const promises = endpoints.map(endpoint => {
      const url = `${this.baseUrl}${endpoint}`;
      const cacheKey = this.generateCacheKey(url, {});
      
      return this.cacheManager.withCache(
        cacheKey,
        () => this.directRequest(url, {}),
        config
      );
    });
    
    await Promise.all(promises);
  }

  /**
   * Invalidate cache
   */
  async invalidateCache(pattern?: string | RegExp): Promise<number> {
    if (pattern) {
      return this.cacheManager.invalidate(pattern);
    }
    
    await this.cacheManager.clear();
    return 0;
  }

  /**
   * Get performance metrics
   */
  getMetrics() {
    return {
      performance: this.metricsCollector.generateReport(),
      cache: this.cacheManager.getStats()
    };
  }

  private generateCacheKey(url: string, config: ApiRequestConfig): string {
    const parts = [url];
    
    if (config.method && config.method !== 'GET') {
      parts.push(config.method);
    }
    
    if (config.body) {
      parts.push(JSON.stringify(config.body));
    }
    
    return parts.join(':');
  }
}

// Singleton instance
export const apiClient = new OptimizedApiClient(
  import.meta.env.VITE_API_URL || 'http://localhost:8000'
);

// Convenience methods
export const api = {
  get: <T>(endpoint: string, config?: ApiRequestConfig) => 
    apiClient.request<T>(endpoint, { ...config, method: 'GET' }),
  
  post: <T>(endpoint: string, body?: any, config?: ApiRequestConfig) => 
    apiClient.request<T>(endpoint, { ...config, method: 'POST', body: JSON.stringify(body) }),
  
  put: <T>(endpoint: string, body?: any, config?: ApiRequestConfig) => 
    apiClient.request<T>(endpoint, { ...config, method: 'PUT', body: JSON.stringify(body) }),
  
  delete: <T>(endpoint: string, config?: ApiRequestConfig) => 
    apiClient.request<T>(endpoint, { ...config, method: 'DELETE' }),
  
  batch: <T>(endpoint: string, requests: any[]) => 
    apiClient.batch<T>(endpoint, requests),
  
  stream: apiClient.streamSSE.bind(apiClient),
  websocket: apiClient.websocket.bind(apiClient),
  prefetch: apiClient.prefetch.bind(apiClient),
  invalidateCache: apiClient.invalidateCache.bind(apiClient),
  metrics: apiClient.getMetrics.bind(apiClient)
};