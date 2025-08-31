/**
 * Performance metrics collection and monitoring
 */

export interface PerformanceMetric {
  name: string;
  value: number;
  timestamp: number;
  tags?: Record<string, string>;
  unit?: string;
}

export interface WebVitalsMetrics {
  FCP: number | null;  // First Contentful Paint
  LCP: number | null;  // Largest Contentful Paint
  FID: number | null;  // First Input Delay
  CLS: number | null;  // Cumulative Layout Shift
  TTFB: number | null; // Time to First Byte
  INP: number | null;  // Interaction to Next Paint
}

export interface ResourceTiming {
  name: string;
  duration: number;
  transferSize: number;
  encodedBodySize: number;
  decodedBodySize: number;
  startTime: number;
  responseEnd: number;
}

export interface PerformanceAlert {
  id: string;
  type: 'warning' | 'critical';
  metric: string;
  threshold: number;
  value: number;
  message: string;
  timestamp: number;
}

export class MetricsCollector {
  private metrics: PerformanceMetric[] = [];
  private webVitals: WebVitalsMetrics = {
    FCP: null,
    LCP: null,
    FID: null,
    CLS: null,
    TTFB: null,
    INP: null
  };
  private observers: Set<(metrics: PerformanceMetric[]) => void> = new Set();
  private alertObservers: Set<(alert: PerformanceAlert) => void> = new Set();
  private thresholds = new Map<string, { warning: number; critical: number }>();
  private collectionInterval: number | null = null;

  constructor() {
    this.initializeWebVitals();
    this.setupPerformanceObservers();
    this.startCollection();
  }

  private initializeWebVitals(): void {
    if (typeof window === 'undefined') return;

    // Observe paint timing
    this.observePaintTiming();
    
    // Observe layout shift
    this.observeLayoutShift();
    
    // Observe first input delay
    this.observeFirstInput();
    
    // Observe interaction to next paint
    this.observeINP();
  }

  private observePaintTiming(): void {
    if (!('PerformanceObserver' in window)) return;

    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.name === 'first-contentful-paint') {
            this.webVitals.FCP = entry.startTime;
            this.recordMetric({
              name: 'web_vitals_fcp',
              value: entry.startTime,
              timestamp: Date.now(),
              unit: 'ms'
            });
          }
        }
      });
      
      observer.observe({ type: 'paint', buffered: true });
    } catch (error) {
      console.error('Failed to observe paint timing:', error);
    }

    // Observe LCP
    try {
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1];
        if (lastEntry) {
          this.webVitals.LCP = lastEntry.startTime;
          this.recordMetric({
            name: 'web_vitals_lcp',
            value: lastEntry.startTime,
            timestamp: Date.now(),
            unit: 'ms'
          });
        }
      });
      
      lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });
    } catch (error) {
      console.error('Failed to observe LCP:', error);
    }
  }

  private observeLayoutShift(): void {
    if (!('PerformanceObserver' in window)) return;

    try {
      let cls = 0;
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (!(entry as any).hadRecentInput) {
            cls += (entry as any).value;
          }
        }
        this.webVitals.CLS = cls;
        this.recordMetric({
          name: 'web_vitals_cls',
          value: cls,
          timestamp: Date.now(),
          unit: 'score'
        });
      });
      
      observer.observe({ type: 'layout-shift', buffered: true });
    } catch (error) {
      console.error('Failed to observe layout shift:', error);
    }
  }

  private observeFirstInput(): void {
    if (!('PerformanceObserver' in window)) return;

    try {
      const observer = new PerformanceObserver((list) => {
        const firstInput = list.getEntries()[0] as PerformanceEventTiming;
        if (firstInput && firstInput.processingStart) {
          const fid = firstInput.processingStart - firstInput.startTime;
          this.webVitals.FID = fid;
          this.recordMetric({
            name: 'web_vitals_fid',
            value: fid,
            timestamp: Date.now(),
            unit: 'ms'
          });
        }
      });
      
      observer.observe({ type: 'first-input', buffered: true });
    } catch (error) {
      console.error('Failed to observe first input:', error);
    }
  }

  private observeINP(): void {
    if (!('PerformanceObserver' in window)) return;

    try {
      let maxDuration = 0;
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.duration > maxDuration) {
            maxDuration = entry.duration;
            this.webVitals.INP = maxDuration;
            this.recordMetric({
              name: 'web_vitals_inp',
              value: maxDuration,
              timestamp: Date.now(),
              unit: 'ms'
            });
          }
        }
      });
      
      observer.observe({ type: 'event', buffered: true });
    } catch (error) {
      console.error('Failed to observe INP:', error);
    }
  }

  private setupPerformanceObservers(): void {
    if (!('PerformanceObserver' in window)) return;

    // Observe navigation timing
    this.observeNavigationTiming();
    
    // Observe resource timing
    this.observeResourceTiming();
    
    // Observe long tasks
    this.observeLongTasks();
  }

  private observeNavigationTiming(): void {
    if (!('performance' in window)) return;

    const navigation = performance.getEntriesByType('navigation')[0] as any;
    if (navigation) {
      this.webVitals.TTFB = navigation.responseStart - navigation.fetchStart;
      this.recordMetric({
        name: 'web_vitals_ttfb',
        value: this.webVitals.TTFB,
        timestamp: Date.now(),
        unit: 'ms'
      });
    }
  }

  private observeResourceTiming(): void {
    if (!('PerformanceObserver' in window)) return;

    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          const resourceEntry = entry as PerformanceResourceTiming;
          
          this.recordMetric({
            name: 'resource_duration',
            value: resourceEntry.duration,
            timestamp: Date.now(),
            tags: {
              resource: resourceEntry.name,
              type: resourceEntry.initiatorType
            },
            unit: 'ms'
          });

          // Check for slow resources
          if (resourceEntry.duration > 1000) {
            this.checkThreshold('resource_duration', resourceEntry.duration, {
              resource: resourceEntry.name
            });
          }
        }
      });
      
      observer.observe({ type: 'resource', buffered: false });
    } catch (error) {
      console.error('Failed to observe resource timing:', error);
    }
  }

  private observeLongTasks(): void {
    if (!('PerformanceObserver' in window)) return;

    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          this.recordMetric({
            name: 'long_task',
            value: entry.duration,
            timestamp: Date.now(),
            unit: 'ms'
          });

          // Alert on very long tasks
          if (entry.duration > 100) {
            this.checkThreshold('long_task', entry.duration);
          }
        }
      });
      
      observer.observe({ type: 'longtask', buffered: false });
    } catch (error) {
      console.error('Failed to observe long tasks:', error);
    }
  }

  private startCollection(): void {
    // Collect memory usage periodically
    this.collectionInterval = window.setInterval(() => {
      this.collectMemoryMetrics();
      this.collectCustomMetrics();
      this.pruneOldMetrics();
    }, 10000); // Every 10 seconds
  }

  private collectMemoryMetrics(): void {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      
      this.recordMetric({
        name: 'memory_used',
        value: memory.usedJSHeapSize,
        timestamp: Date.now(),
        unit: 'bytes'
      });

      this.recordMetric({
        name: 'memory_limit',
        value: memory.jsHeapSizeLimit,
        timestamp: Date.now(),
        unit: 'bytes'
      });

      const usage = (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100;
      if (usage > 80) {
        this.checkThreshold('memory_usage', usage);
      }
    }
  }

  private collectCustomMetrics(): void {
    // Collect frame rate
    this.measureFrameRate();
    
    // Collect connection info
    this.collectConnectionMetrics();
  }

  private measureFrameRate(): void {
    let lastTime = performance.now();
    let frames = 0;
    let fps = 0;

    const measureFrame = () => {
      frames++;
      const currentTime = performance.now();
      
      if (currentTime >= lastTime + 1000) {
        fps = Math.round((frames * 1000) / (currentTime - lastTime));
        this.recordMetric({
          name: 'frame_rate',
          value: fps,
          timestamp: Date.now(),
          unit: 'fps'
        });
        
        frames = 0;
        lastTime = currentTime;
        
        if (fps < 30) {
          this.checkThreshold('frame_rate', fps);
        }
      }
      
      requestAnimationFrame(measureFrame);
    };

    requestAnimationFrame(measureFrame);
  }

  private collectConnectionMetrics(): void {
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      
      this.recordMetric({
        name: 'connection_rtt',
        value: connection.rtt || 0,
        timestamp: Date.now(),
        unit: 'ms'
      });

      this.recordMetric({
        name: 'connection_downlink',
        value: connection.downlink || 0,
        timestamp: Date.now(),
        unit: 'mbps'
      });
    }
  }

  recordMetric(metric: PerformanceMetric): void {
    this.metrics.push(metric);
    this.notifyObservers();
  }

  recordTiming(name: string, duration: number, tags?: Record<string, string>): void {
    this.recordMetric({
      name,
      value: duration,
      timestamp: Date.now(),
      tags,
      unit: 'ms'
    });
  }

  measureAsync<T>(
    name: string,
    fn: () => Promise<T>,
    tags?: Record<string, string>
  ): Promise<T> {
    const start = performance.now();
    
    return fn()
      .then(result => {
        const duration = performance.now() - start;
        this.recordTiming(name, duration, tags);
        return result;
      })
      .catch(error => {
        const duration = performance.now() - start;
        this.recordTiming(name, duration, { ...tags, error: 'true' });
        throw error;
      });
  }

  measureSync<T>(
    name: string,
    fn: () => T,
    tags?: Record<string, string>
  ): T {
    const start = performance.now();
    
    try {
      const result = fn();
      const duration = performance.now() - start;
      this.recordTiming(name, duration, tags);
      return result;
    } catch (error) {
      const duration = performance.now() - start;
      this.recordTiming(name, duration, { ...tags, error: 'true' });
      throw error;
    }
  }

  setThreshold(
    metric: string,
    warning: number,
    critical: number
  ): void {
    this.thresholds.set(metric, { warning, critical });
  }

  private checkThreshold(
    metric: string,
    value: number,
    metadata?: Record<string, any>
  ): void {
    const threshold = this.thresholds.get(metric);
    if (!threshold) return;

    let alert: PerformanceAlert | null = null;

    if (value >= threshold.critical) {
      alert = {
        id: `${metric}-${Date.now()}`,
        type: 'critical',
        metric,
        threshold: threshold.critical,
        value,
        message: `${metric} exceeded critical threshold: ${value} > ${threshold.critical}`,
        timestamp: Date.now()
      };
    } else if (value >= threshold.warning) {
      alert = {
        id: `${metric}-${Date.now()}`,
        type: 'warning',
        metric,
        threshold: threshold.warning,
        value,
        message: `${metric} exceeded warning threshold: ${value} > ${threshold.warning}`,
        timestamp: Date.now()
      };
    }

    if (alert) {
      this.notifyAlertObservers(alert);
    }
  }

  getWebVitals(): WebVitalsMetrics {
    return { ...this.webVitals };
  }

  getMetrics(since?: number): PerformanceMetric[] {
    if (since) {
      return this.metrics.filter(m => m.timestamp >= since);
    }
    return [...this.metrics];
  }

  getAggregatedMetrics(
    name: string,
    aggregation: 'avg' | 'min' | 'max' | 'sum' | 'p50' | 'p95' | 'p99',
    window?: number
  ): number | null {
    const cutoff = window ? Date.now() - window : 0;
    const relevantMetrics = this.metrics
      .filter(m => m.name === name && m.timestamp >= cutoff)
      .map(m => m.value)
      .sort((a, b) => a - b);

    if (relevantMetrics.length === 0) return null;

    switch (aggregation) {
      case 'avg':
        return relevantMetrics.reduce((a, b) => a + b, 0) / relevantMetrics.length;
      case 'min':
        return Math.min(...relevantMetrics);
      case 'max':
        return Math.max(...relevantMetrics);
      case 'sum':
        return relevantMetrics.reduce((a, b) => a + b, 0);
      case 'p50':
        return this.percentile(relevantMetrics, 0.5);
      case 'p95':
        return this.percentile(relevantMetrics, 0.95);
      case 'p99':
        return this.percentile(relevantMetrics, 0.99);
      default:
        return null;
    }
  }

  private percentile(values: number[], p: number): number {
    const index = Math.ceil(values.length * p) - 1;
    return values[Math.max(0, index)];
  }

  subscribe(observer: (metrics: PerformanceMetric[]) => void): () => void {
    this.observers.add(observer);
    return () => this.observers.delete(observer);
  }

  subscribeToAlerts(observer: (alert: PerformanceAlert) => void): () => void {
    this.alertObservers.add(observer);
    return () => this.alertObservers.delete(observer);
  }

  private notifyObservers(): void {
    this.observers.forEach(observer => {
      try {
        observer(this.getMetrics());
      } catch (error) {
        console.error('Metrics observer error:', error);
      }
    });
  }

  private notifyAlertObservers(alert: PerformanceAlert): void {
    this.alertObservers.forEach(observer => {
      try {
        observer(alert);
      } catch (error) {
        console.error('Alert observer error:', error);
      }
    });
  }

  private pruneOldMetrics(): void {
    // Keep only last hour of metrics
    const cutoff = Date.now() - 3600000;
    this.metrics = this.metrics.filter(m => m.timestamp >= cutoff);
  }

  generateReport(): {
    webVitals: WebVitalsMetrics;
    summary: Record<string, any>;
    alerts: PerformanceAlert[];
  } {
    const window = 300000; // Last 5 minutes

    return {
      webVitals: this.getWebVitals(),
      summary: {
        avgResponseTime: this.getAggregatedMetrics('response_time', 'avg', window),
        p95ResponseTime: this.getAggregatedMetrics('response_time', 'p95', window),
        errorRate: this.calculateErrorRate(window),
        cacheHitRate: this.calculateCacheHitRate(window),
        avgMemoryUsage: this.getAggregatedMetrics('memory_used', 'avg', window),
        avgFrameRate: this.getAggregatedMetrics('frame_rate', 'avg', window)
      },
      alerts: [] // Would be populated with recent alerts
    };
  }

  private calculateErrorRate(window: number): number {
    const cutoff = Date.now() - window;
    const requests = this.metrics.filter(
      m => m.name === 'request' && m.timestamp >= cutoff
    );
    const errors = requests.filter(m => m.tags?.error === 'true');
    
    return requests.length > 0 ? (errors.length / requests.length) * 100 : 0;
  }

  private calculateCacheHitRate(window: number): number {
    const cutoff = Date.now() - window;
    const cacheMetrics = this.metrics.filter(
      m => (m.name === 'cache_hit' || m.name === 'cache_miss') && m.timestamp >= cutoff
    );
    const hits = cacheMetrics.filter(m => m.name === 'cache_hit');
    
    return cacheMetrics.length > 0 ? (hits.length / cacheMetrics.length) * 100 : 0;
  }

  destroy(): void {
    if (this.collectionInterval) {
      clearInterval(this.collectionInterval);
    }
    this.observers.clear();
    this.alertObservers.clear();
    this.metrics = [];
  }
}