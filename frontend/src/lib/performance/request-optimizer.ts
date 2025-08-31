/**
 * Request optimization utilities for batching, debouncing, and deduplication
 */

export interface BatchConfig {
  maxBatchSize: number;
  maxWaitTime: number;
  batchProcessor: (items: any[]) => Promise<any[]>;
}

export interface DebounceConfig {
  delay: number;
  maxWait?: number;
  leading?: boolean;
  trailing?: boolean;
}

export class RequestOptimizer {
  private batchQueues = new Map<string, any[]>();
  private batchTimers = new Map<string, number>();
  private batchConfigs = new Map<string, BatchConfig>();
  private debouncedFunctions = new Map<string, any>();
  private pendingRequests = new Map<string, Promise<any>>();

  /**
   * Configure batching for a specific endpoint
   */
  configureBatching(endpoint: string, config: BatchConfig): void {
    this.batchConfigs.set(endpoint, config);
    if (!this.batchQueues.has(endpoint)) {
      this.batchQueues.set(endpoint, []);
    }
  }

  /**
   * Add request to batch queue
   */
  async addToBatch<T>(
    endpoint: string,
    request: any
  ): Promise<T> {
    const config = this.batchConfigs.get(endpoint);
    if (!config) {
      throw new Error(`No batch config for endpoint: ${endpoint}`);
    }

    return new Promise((resolve, reject) => {
      const queue = this.batchQueues.get(endpoint) || [];
      queue.push({ request, resolve, reject });
      this.batchQueues.set(endpoint, queue);

      // Process if batch is full
      if (queue.length >= config.maxBatchSize) {
        this.processBatch(endpoint);
      } else {
        // Schedule batch processing
        this.scheduleBatch(endpoint, config.maxWaitTime);
      }
    });
  }

  /**
   * Create debounced version of a function
   */
  debounce<T extends (...args: any[]) => any>(
    fn: T,
    config: DebounceConfig
  ): (...args: Parameters<T>) => Promise<ReturnType<T>> {
    let timeoutId: number | null = null;
    let maxTimeoutId: number | null = null;
    let lastCallTime = 0;
    let lastArgs: Parameters<T> | null = null;
    let result: ReturnType<T> | null = null;
    let pendingPromise: Promise<ReturnType<T>> | null = null;
    let resolvers: Array<(value: ReturnType<T>) => void> = [];

    const invokeFunc = (args: Parameters<T>) => {
      const value = fn(...args);
      result = value;
      resolvers.forEach(resolve => resolve(value));
      resolvers = [];
      pendingPromise = null;
      return value;
    };

    const debounced = (...args: Parameters<T>): Promise<ReturnType<T>> => {
      lastArgs = args;
      lastCallTime = Date.now();

      if (pendingPromise) {
        return pendingPromise;
      }

      pendingPromise = new Promise<ReturnType<T>>((resolve) => {
        resolvers.push(resolve);

        const shouldCallNow = config.leading && !timeoutId;

        if (timeoutId) {
          clearTimeout(timeoutId);
        }

        if (shouldCallNow) {
          resolve(invokeFunc(args));
        } else {
          timeoutId = setTimeout(() => {
            if (config.trailing !== false && lastArgs) {
              invokeFunc(lastArgs);
            }
            timeoutId = null;
            maxTimeoutId = null;
          }, config.delay);

          // Set max wait timer if configured
          if (config.maxWait && !maxTimeoutId) {
            maxTimeoutId = setTimeout(() => {
              if (lastArgs) {
                invokeFunc(lastArgs);
              }
              if (timeoutId) {
                clearTimeout(timeoutId);
                timeoutId = null;
              }
              maxTimeoutId = null;
            }, config.maxWait);
          }
        }
      });

      return pendingPromise;
    };

    debounced.cancel = () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      if (maxTimeoutId) {
        clearTimeout(maxTimeoutId);
        maxTimeoutId = null;
      }
      lastArgs = null;
      resolvers = [];
      pendingPromise = null;
    };

    debounced.flush = () => {
      if (timeoutId && lastArgs) {
        invokeFunc(lastArgs);
        if (timeoutId) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }
        if (maxTimeoutId) {
          clearTimeout(maxTimeoutId);
          maxTimeoutId = null;
        }
      }
      return result;
    };

    return debounced;
  }

  /**
   * Throttle function calls
   */
  throttle<T extends (...args: any[]) => any>(
    fn: T,
    delay: number
  ): (...args: Parameters<T>) => ReturnType<T> | undefined {
    let lastCall = 0;
    let lastResult: ReturnType<T> | undefined;

    return (...args: Parameters<T>): ReturnType<T> | undefined => {
      const now = Date.now();
      
      if (now - lastCall >= delay) {
        lastCall = now;
        lastResult = fn(...args);
        return lastResult;
      }
      
      return lastResult;
    };
  }

  /**
   * Deduplicate concurrent requests
   */
  async deduplicate<T>(
    key: string,
    fetcher: () => Promise<T>
  ): Promise<T> {
    // Check if request is already in flight
    if (this.pendingRequests.has(key)) {
      return this.pendingRequests.get(key)!;
    }

    // Create new request
    const promise = fetcher()
      .then(result => {
        this.pendingRequests.delete(key);
        return result;
      })
      .catch(error => {
        this.pendingRequests.delete(key);
        throw error;
      });

    this.pendingRequests.set(key, promise);
    return promise;
  }

  /**
   * Create request pipeline with multiple optimizations
   */
  createPipeline<T>(options: {
    endpoint: string;
    debounce?: DebounceConfig;
    batch?: BatchConfig;
    deduplicate?: boolean;
  }): (request: any) => Promise<T> {
    let pipeline: any = async (request: any) => request;

    if (options.batch) {
      this.configureBatching(options.endpoint, options.batch);
      const batchFn = pipeline;
      pipeline = (request: any) => this.addToBatch(options.endpoint, request);
    }

    if (options.debounce) {
      pipeline = this.debounce(pipeline, options.debounce);
    }

    if (options.deduplicate) {
      const deduplicatedPipeline = pipeline;
      pipeline = (request: any) => {
        const key = `${options.endpoint}:${JSON.stringify(request)}`;
        return this.deduplicate(key, () => deduplicatedPipeline(request));
      };
    }

    return pipeline;
  }

  /**
   * Connection pooling for WebSocket
   */
  createConnectionPool(options: {
    maxConnections: number;
    url: string;
    reconnectDelay?: number;
  }): WebSocketPool {
    return new WebSocketPool(options);
  }

  private scheduleBatch(endpoint: string, maxWaitTime: number): void {
    // Clear existing timer
    const existingTimer = this.batchTimers.get(endpoint);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    // Schedule new batch processing
    const timer = setTimeout(() => {
      this.processBatch(endpoint);
    }, maxWaitTime) as any;

    this.batchTimers.set(endpoint, timer);
  }

  private async processBatch(endpoint: string): Promise<void> {
    const queue = this.batchQueues.get(endpoint);
    const config = this.batchConfigs.get(endpoint);
    
    if (!queue || queue.length === 0 || !config) {
      return;
    }

    // Clear timer
    const timer = this.batchTimers.get(endpoint);
    if (timer) {
      clearTimeout(timer);
      this.batchTimers.delete(endpoint);
    }

    // Get items to process
    const items = queue.splice(0, config.maxBatchSize);
    this.batchQueues.set(endpoint, queue);

    try {
      // Process batch
      const requests = items.map(item => item.request);
      const results = await config.batchProcessor(requests);

      // Resolve individual promises
      items.forEach((item, index) => {
        item.resolve(results[index]);
      });
    } catch (error) {
      // Reject all promises in batch
      items.forEach(item => {
        item.reject(error);
      });
    }
  }
}

/**
 * WebSocket connection pool
 */
export class WebSocketPool {
  private connections: WebSocket[] = [];
  private availableConnections: WebSocket[] = [];
  private pendingRequests: Array<(ws: WebSocket) => void> = [];
  private options: {
    maxConnections: number;
    url: string;
    reconnectDelay: number;
  };

  constructor(options: {
    maxConnections: number;
    url: string;
    reconnectDelay?: number;
  }) {
    this.options = {
      ...options,
      reconnectDelay: options.reconnectDelay || 1000
    };
    this.initialize();
  }

  private initialize(): void {
    for (let i = 0; i < this.options.maxConnections; i++) {
      this.createConnection();
    }
  }

  private createConnection(): void {
    const ws = new WebSocket(this.options.url);

    ws.addEventListener('open', () => {
      this.availableConnections.push(ws);
      this.processPendingRequests();
    });

    ws.addEventListener('close', () => {
      this.removeConnection(ws);
      // Reconnect after delay
      setTimeout(() => this.createConnection(), this.options.reconnectDelay);
    });

    ws.addEventListener('error', (error) => {
      console.error('WebSocket error:', error);
      this.removeConnection(ws);
    });

    this.connections.push(ws);
  }

  private removeConnection(ws: WebSocket): void {
    this.connections = this.connections.filter(conn => conn !== ws);
    this.availableConnections = this.availableConnections.filter(conn => conn !== ws);
  }

  private processPendingRequests(): void {
    while (this.pendingRequests.length > 0 && this.availableConnections.length > 0) {
      const resolver = this.pendingRequests.shift()!;
      const ws = this.availableConnections.shift()!;
      resolver(ws);
    }
  }

  async getConnection(): Promise<WebSocket> {
    if (this.availableConnections.length > 0) {
      return this.availableConnections.shift()!;
    }

    return new Promise((resolve) => {
      this.pendingRequests.push(resolve);
    });
  }

  releaseConnection(ws: WebSocket): void {
    if (ws.readyState === WebSocket.OPEN) {
      this.availableConnections.push(ws);
      this.processPendingRequests();
    }
  }

  async execute<T>(
    fn: (ws: WebSocket) => Promise<T>
  ): Promise<T> {
    const ws = await this.getConnection();
    try {
      return await fn(ws);
    } finally {
      this.releaseConnection(ws);
    }
  }

  close(): void {
    this.connections.forEach(ws => ws.close());
    this.connections = [];
    this.availableConnections = [];
    this.pendingRequests = [];
  }

  getStats() {
    return {
      total: this.connections.length,
      available: this.availableConnections.length,
      pending: this.pendingRequests.length
    };
  }
}