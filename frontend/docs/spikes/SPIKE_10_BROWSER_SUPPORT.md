# Spike 10: Browser Support & Compatibility

## Overview
This spike defines browser support requirements, polyfills, and progressive enhancement strategies for the TTRPG MCP Server frontend.

## 1. Browser Support Matrix

### 1.1 Supported Browsers
```yaml
# Minimum supported versions
browsers:
  desktop:
    chrome: 90+      # ~95% feature support
    firefox: 88+     # ~93% feature support  
    safari: 14.1+    # ~90% feature support
    edge: 90+        # Chrome-based
    
  mobile:
    ios_safari: 14.5+
    chrome_android: 90+
    firefox_android: 88+
    samsung_internet: 14+
    
  optional:
    opera: 76+
    brave: latest
    vivaldi: latest
```

### 1.2 Feature Detection
```typescript
// feature-detection.ts
export class FeatureDetector {
  private features = new Map<string, boolean>();
  
  detectFeatures(): FeatureSupport {
    return {
      // Core Web APIs
      webSockets: 'WebSocket' in window,
      serviceWorker: 'serviceWorker' in navigator,
      indexedDB: 'indexedDB' in window,
      webCrypto: 'crypto' in window && 'subtle' in window.crypto,
      
      // Modern JavaScript
      es2020: this.checkES2020Support(),
      asyncAwait: this.checkAsyncSupport(),
      modules: 'noModule' in HTMLScriptElement.prototype,
      
      // CSS Features
      grid: CSS.supports('display', 'grid'),
      customProperties: CSS.supports('--test', '0'),
      containerQueries: CSS.supports('container-type', 'inline-size'),
      
      // Media
      webP: this.checkWebPSupport(),
      webM: this.checkWebMSupport(),
      avif: this.checkAVIFSupport(),
      
      // Performance
      webWorkers: 'Worker' in window,
      sharedArrayBuffer: 'SharedArrayBuffer' in window,
      webAssembly: 'WebAssembly' in window,
      
      // Permissions
      notifications: 'Notification' in window,
      clipboard: navigator.clipboard !== undefined,
      geolocation: 'geolocation' in navigator,
      
      // Storage
      localStorage: this.checkLocalStorage(),
      sessionStorage: this.checkSessionStorage(),
      cookies: navigator.cookieEnabled
    };
  }
  
  private checkES2020Support(): boolean {
    try {
      // Check for nullish coalescing and optional chaining
      eval('const x = null ?? 5; const y = {}?.prop');
      return true;
    } catch {
      return false;
    }
  }
  
  private checkAsyncSupport(): boolean {
    try {
      eval('(async () => {})');
      return true;
    } catch {
      return false;
    }
  }
  
  private async checkWebPSupport(): Promise<boolean> {
    const webP = new Image();
    webP.src = 'data:image/webp;base64,UklGRjoAAABXRUJQVlA4IC4AAACyAgCdASoCAAIALmk0mk0iIiIiIgBoSygABc6WWgAA/veff/0PP8bA//LwYAAA';
    
    return new Promise((resolve) => {
      webP.onload = webP.onerror = () => {
        resolve(webP.height === 2);
      };
    });
  }
}
```

## 2. Polyfills & Fallbacks

### 2.1 Core Polyfills
```typescript
// polyfills.ts
export async function loadPolyfills() {
  const promises = [];
  
  // Promise.allSettled polyfill
  if (!Promise.allSettled) {
    promises.push(import('core-js/features/promise/all-settled'));
  }
  
  // Array.prototype.at polyfill
  if (!Array.prototype.at) {
    promises.push(import('core-js/features/array/at'));
  }
  
  // Object.hasOwn polyfill
  if (!Object.hasOwn) {
    promises.push(import('core-js/features/object/has-own'));
  }
  
  // structuredClone polyfill
  if (!window.structuredClone) {
    promises.push(import('core-js/features/structured-clone'));
  }
  
  // Intl.RelativeTimeFormat polyfill
  if (!Intl.RelativeTimeFormat) {
    promises.push(import('@formatjs/intl-relativetimeformat/polyfill'));
  }
  
  // ResizeObserver polyfill
  if (!window.ResizeObserver) {
    promises.push(import('resize-observer-polyfill').then(module => {
      window.ResizeObserver = module.default;
    }));
  }
  
  // IntersectionObserver polyfill
  if (!window.IntersectionObserver) {
    promises.push(import('intersection-observer'));
  }
  
  await Promise.all(promises);
}
```

### 2.2 WebSocket Fallback
```typescript
// websocket-fallback.ts
export class WebSocketWithFallback {
  private ws: WebSocket | null = null;
  private polling: PollingTransport | null = null;
  private usePolling = false;
  
  constructor(private url: string) {
    this.detectWebSocketSupport();
  }
  
  private detectWebSocketSupport() {
    if (!('WebSocket' in window)) {
      this.usePolling = true;
      return;
    }
    
    // Test WebSocket connectivity
    try {
      const testWs = new WebSocket(this.url.replace('http', 'ws'));
      testWs.onopen = () => {
        testWs.close();
        this.usePolling = false;
      };
      testWs.onerror = () => {
        this.usePolling = true;
      };
    } catch {
      this.usePolling = true;
    }
  }
  
  connect() {
    if (this.usePolling) {
      this.polling = new PollingTransport(this.url);
      return this.polling.connect();
    } else {
      this.ws = new WebSocket(this.url.replace('http', 'ws'));
      return new Promise((resolve, reject) => {
        this.ws!.onopen = resolve;
        this.ws!.onerror = reject;
      });
    }
  }
  
  send(data: any) {
    if (this.usePolling) {
      this.polling?.send(data);
    } else {
      this.ws?.send(JSON.stringify(data));
    }
  }
}

// Long polling fallback
class PollingTransport {
  private pollInterval = 1000;
  private polling = false;
  
  constructor(private url: string) {}
  
  async connect() {
    this.polling = true;
    this.poll();
  }
  
  private async poll() {
    while (this.polling) {
      try {
        const response = await fetch(`${this.url}/poll`, {
          method: 'GET',
          credentials: 'include'
        });
        
        if (response.ok) {
          const messages = await response.json();
          messages.forEach(msg => this.handleMessage(msg));
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
      
      await new Promise(resolve => setTimeout(resolve, this.pollInterval));
    }
  }
  
  async send(data: any) {
    await fetch(`${this.url}/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
      credentials: 'include'
    });
  }
  
  private handleMessage(message: any) {
    // Dispatch message to handlers
    window.dispatchEvent(new CustomEvent('polling-message', { detail: message }));
  }
}
```

## 3. Progressive Enhancement

### 3.1 Core Functionality First
```svelte
<!-- DiceRoller.svelte with progressive enhancement -->
<script lang="ts">
  import { browser } from '$app/environment';
  
  let expression = $state('');
  let result = $state<number | null>(null);
  let canUse3D = $state(false);
  let canUseWebGL = $state(false);
  
  $effect(() => {
    if (browser) {
      // Check for advanced features
      canUseWebGL = !!document.createElement('canvas').getContext('webgl');
      canUse3D = canUseWebGL && CSS.supports('transform-style', 'preserve-3d');
    }
  });
  
  function rollDice() {
    // Basic functionality - always works
    const basicResult = calculateDiceRoll(expression);
    result = basicResult;
    
    // Enhanced functionality if available
    if (canUse3D) {
      show3DDiceAnimation(expression, basicResult);
    } else if (canUseWebGL) {
      show2DDiceAnimation(expression, basicResult);
    }
    
    // Share if available
    if (navigator.share) {
      offerToShare(expression, basicResult);
    }
  }
  
  function calculateDiceRoll(expr: string): number {
    // Pure JavaScript dice rolling - works everywhere
    const match = expr.match(/(\d+)d(\d+)([+-]\d+)?/);
    if (!match) return 0;
    
    const [, count, sides, modifier] = match;
    let total = 0;
    
    for (let i = 0; i < parseInt(count); i++) {
      total += Math.floor(Math.random() * parseInt(sides)) + 1;
    }
    
    if (modifier) {
      total += parseInt(modifier);
    }
    
    return total;
  }
</script>

<!-- HTML form fallback for no-JS -->
<form method="POST" action="/api/dice/roll">
  <label for="expression">Dice Expression:</label>
  <input 
    id="expression"
    name="expression"
    type="text" 
    bind:value={expression}
    pattern="\\d+d\\d+([+-]\\d+)?"
    required
  />
  
  <button type="submit" onclick={browser ? rollDice : undefined}>
    Roll Dice
  </button>
  
  {#if result !== null}
    <output>{result}</output>
  {/if}
</form>

<!-- Progressive enhancement notice -->
{#if browser && !canUseWebGL}
  <p class="enhancement-notice">
    Enable hardware acceleration for 3D dice animations
  </p>
{/if}
```

### 3.2 CSS Progressive Enhancement
```css
/* base.css - Works everywhere */
.dice-roller {
  padding: 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.dice-result {
  font-size: 2rem;
  font-weight: bold;
  text-align: center;
}

/* Modern browsers only */
@supports (display: grid) {
  .dice-roller {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 1rem;
  }
}

/* Container queries for responsive components */
@supports (container-type: inline-size) {
  .dice-roller {
    container-type: inline-size;
  }
  
  @container (min-width: 400px) {
    .dice-result {
      font-size: 3rem;
    }
  }
}

/* Dark mode with fallback */
.dice-roller {
  background: white;
  color: black;
}

@media (prefers-color-scheme: dark) {
  .dice-roller {
    background: #1a1a1a;
    color: white;
  }
}

/* Advanced animations for capable browsers */
@supports (animation-timeline: scroll()) {
  .dice-result {
    animation: reveal linear both;
    animation-timeline: scroll();
    animation-range: entry 25% cover 50%;
  }
}

/* WebP background with fallback */
.hero {
  background-image: url('/images/hero.jpg');
}

.webp .hero {
  background-image: url('/images/hero.webp');
}

.avif .hero {
  background-image: url('/images/hero.avif');
}
```

## 4. Mobile Optimization

### 4.1 Touch Interactions
```typescript
// touch-handler.ts
export class TouchHandler {
  private touchStartX = 0;
  private touchStartY = 0;
  private touchEndX = 0;
  private touchEndY = 0;
  
  setupTouchHandlers(element: HTMLElement) {
    // Prevent 300ms delay on touch devices
    element.style.touchAction = 'manipulation';
    
    element.addEventListener('touchstart', this.handleTouchStart.bind(this), 
      { passive: true }
    );
    
    element.addEventListener('touchmove', this.handleTouchMove.bind(this),
      { passive: false }
    );
    
    element.addEventListener('touchend', this.handleTouchEnd.bind(this));
    
    // Prevent zoom on double tap
    let lastTap = 0;
    element.addEventListener('touchend', (e) => {
      const currentTime = new Date().getTime();
      const tapLength = currentTime - lastTap;
      if (tapLength < 500 && tapLength > 0) {
        e.preventDefault();
      }
      lastTap = currentTime;
    });
  }
  
  private handleTouchStart(e: TouchEvent) {
    this.touchStartX = e.touches[0].clientX;
    this.touchStartY = e.touches[0].clientY;
  }
  
  private handleTouchMove(e: TouchEvent) {
    // Prevent scrolling while dragging
    if (this.isDragging) {
      e.preventDefault();
    }
  }
  
  private handleTouchEnd(e: TouchEvent) {
    this.touchEndX = e.changedTouches[0].clientX;
    this.touchEndY = e.changedTouches[0].clientY;
    
    this.detectGesture();
  }
  
  private detectGesture() {
    const deltaX = this.touchEndX - this.touchStartX;
    const deltaY = this.touchEndY - this.touchStartY;
    
    // Swipe detection
    if (Math.abs(deltaX) > 50) {
      if (deltaX > 0) {
        this.onSwipeRight?.();
      } else {
        this.onSwipeLeft?.();
      }
    }
    
    if (Math.abs(deltaY) > 50) {
      if (deltaY > 0) {
        this.onSwipeDown?.();
      } else {
        this.onSwipeUp?.();
      }
    }
  }
}
```

### 4.2 Responsive Images
```svelte
<!-- ResponsiveImage.svelte -->
<script lang="ts">
  interface Props {
    src: string;
    alt: string;
    sizes?: string;
    loading?: 'lazy' | 'eager';
  }
  
  let { src, alt, sizes = '100vw', loading = 'lazy' }: Props = $props();
  
  // Generate srcset for different resolutions
  function generateSrcSet(baseSrc: string): string {
    const widths = [320, 640, 960, 1280, 1920];
    return widths
      .map(w => `${baseSrc}?w=${w} ${w}w`)
      .join(', ');
  }
  
  // Generate picture sources for different formats
  function getImageSources(baseSrc: string) {
    const base = baseSrc.replace(/\.[^.]+$/, '');
    return [
      { type: 'image/avif', srcset: `${base}.avif` },
      { type: 'image/webp', srcset: `${base}.webp` },
      { type: 'image/jpeg', srcset: baseSrc }
    ];
  }
</script>

<picture>
  {#each getImageSources(src) as source}
    <source 
      type={source.type}
      srcset={generateSrcSet(source.srcset)}
      {sizes}
    />
  {/each}
  <img
    {src}
    {alt}
    {loading}
    {sizes}
    decoding="async"
    onload="this.classList.add('loaded')"
  />
</picture>

<style>
  img {
    opacity: 0;
    transition: opacity 0.3s;
  }
  
  img.loaded {
    opacity: 1;
  }
</style>
```

## 5. Performance Optimizations

### 5.1 Bundle Optimization
```javascript
// vite.config.js
import { defineConfig } from 'vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    sveltekit(),
    visualizer({
      filename: './stats.html',
      gzipSize: true,
      brotliSize: true
    })
  ],
  
  build: {
    target: ['es2020', 'chrome90', 'firefox88', 'safari14'],
    
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunks
          'vendor-svelte': ['svelte', '@sveltejs/kit'],
          'vendor-utils': ['lodash-es', 'date-fns'],
          'vendor-ui': ['@floating-ui/dom'],
          
          // Feature chunks
          'feature-dice': [
            './src/lib/components/dice',
            './src/lib/utils/dice-parser'
          ],
          'feature-map': [
            './src/lib/components/map',
            './src/lib/utils/canvas'
          ]
        }
      }
    },
    
    // Enable compression
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  },
  
  // Optimize dependencies
  optimizeDeps: {
    include: ['svelte', 'esm-env'],
    exclude: ['@sveltejs/kit']
  }
});
```

### 5.2 Lazy Loading
```typescript
// lazy-loader.ts
export class LazyLoader {
  private observer: IntersectionObserver;
  private loadedModules = new Set<string>();
  
  constructor() {
    this.observer = new IntersectionObserver(
      this.handleIntersection.bind(this),
      {
        rootMargin: '50px',
        threshold: 0.01
      }
    );
  }
  
  observeElement(element: HTMLElement, moduleId: string) {
    element.dataset.moduleId = moduleId;
    this.observer.observe(element);
  }
  
  private async handleIntersection(entries: IntersectionObserverEntry[]) {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        const element = entry.target as HTMLElement;
        const moduleId = element.dataset.moduleId;
        
        if (moduleId && !this.loadedModules.has(moduleId)) {
          await this.loadModule(moduleId, element);
          this.loadedModules.add(moduleId);
          this.observer.unobserve(element);
        }
      }
    }
  }
  
  private async loadModule(moduleId: string, element: HTMLElement) {
    switch (moduleId) {
      case 'dice-3d':
        const { Dice3D } = await import('./modules/dice-3d');
        new Dice3D(element);
        break;
        
      case 'map-canvas':
        const { MapCanvas } = await import('./modules/map-canvas');
        new MapCanvas(element);
        break;
        
      case 'character-sheet':
        const { CharacterSheet } = await import('./modules/character-sheet');
        new CharacterSheet(element);
        break;
    }
  }
}
```

## 6. Accessibility Support

### 6.1 Screen Reader Support
```svelte
<!-- AccessibleDiceRoller.svelte -->
<script lang="ts">
  let expression = $state('');
  let result = $state<number | null>(null);
  let isRolling = $state(false);
  
  async function rollDice() {
    isRolling = true;
    
    // Announce to screen readers
    announceToScreenReader('Rolling dice...');
    
    // Simulate roll
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    result = Math.floor(Math.random() * 20) + 1;
    isRolling = false;
    
    // Announce result
    announceToScreenReader(`You rolled ${result}`);
  }
  
  function announceToScreenReader(message: string) {
    const announcement = document.createElement('div');
    announcement.setAttribute('role', 'status');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    setTimeout(() => announcement.remove(), 1000);
  }
</script>

<div role="region" aria-label="Dice Roller">
  <form onsubmit={rollDice}>
    <label for="dice-expression">
      Enter dice expression (e.g., 1d20+5):
    </label>
    
    <input
      id="dice-expression"
      type="text"
      bind:value={expression}
      aria-describedby="expression-help"
      aria-invalid={!isValidExpression(expression)}
      required
    />
    
    <span id="expression-help" class="sr-only">
      Format: number of dice, letter d, number of sides, optional modifier
    </span>
    
    <button 
      type="submit"
      disabled={isRolling}
      aria-busy={isRolling}
    >
      {isRolling ? 'Rolling...' : 'Roll Dice'}
    </button>
    
    {#if result !== null}
      <output 
        aria-label="Dice result"
        aria-live="polite"
      >
        Result: {result}
      </output>
    {/if}
  </form>
</div>

<style>
  /* Utility class for screen readers */
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
  }
  
  /* Focus styles for keyboard navigation */
  button:focus,
  input:focus {
    outline: 2px solid #4A90E2;
    outline-offset: 2px;
  }
  
  /* High contrast mode support */
  @media (prefers-contrast: high) {
    button {
      border: 2px solid;
    }
    
    input {
      border: 2px solid;
    }
  }
  
  /* Reduced motion support */
  @media (prefers-reduced-motion: reduce) {
    * {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }
</style>
```

## 7. Testing Browser Compatibility

### 7.1 Cross-Browser Testing Setup
```javascript
// playwright.config.js
export default {
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] }
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] }
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] }
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] }
    }
  ],
  
  use: {
    // Test against production build
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    
    // Capture screenshots on failure
    screenshot: 'only-on-failure',
    
    // Record video on failure
    video: 'retain-on-failure',
    
    // Test in different viewports
    viewport: { width: 1280, height: 720 }
  }
};
```

### 7.2 Browser-Specific Tests
```typescript
// browser-compat.test.ts
import { test, expect } from '@playwright/test';

test.describe('Browser Compatibility', () => {
  test('WebSocket fallback in older browsers', async ({ page, browserName }) => {
    if (browserName === 'webkit') {
      // Simulate WebSocket not available
      await page.addInitScript(() => {
        delete (window as any).WebSocket;
      });
    }
    
    await page.goto('/');
    
    // Should fall back to polling
    await expect(page.locator('.connection-status')).toContainText('Connected (Polling)');
  });
  
  test('Progressive enhancement for dice roller', async ({ page }) => {
    // Disable JavaScript
    await page.setJavaScriptEnabled(false);
    
    await page.goto('/dice');
    
    // Form should still work
    await page.fill('input[name="expression"]', '1d20');
    await page.click('button[type="submit"]');
    
    // Server-side rendering should show result
    await expect(page.locator('output')).toBeVisible();
  });
  
  test('Touch interactions on mobile', async ({ page, isMobile }) => {
    if (!isMobile) {
      test.skip();
    }
    
    await page.goto('/map');
    
    // Test pinch to zoom
    await page.locator('.map-canvas').tap();
    await page.locator('.map-canvas').pinch({
      scale: 2,
      position: { x: 100, y: 100 }
    });
    
    // Verify zoom applied
    const transform = await page.locator('.map-canvas').evaluate(
      el => window.getComputedStyle(el).transform
    );
    expect(transform).toContain('scale');
  });
});
```

## Implementation Timeline

### Week 1: Core Support
- [ ] Browser detection
- [ ] Polyfill loading
- [ ] Feature detection

### Week 2: Progressive Enhancement  
- [ ] Fallback implementations
- [ ] Graceful degradation
- [ ] No-JS support

### Week 3: Mobile Optimization
- [ ] Touch handlers
- [ ] Responsive images
- [ ] Performance optimization

### Week 4: Testing & Validation
- [ ] Cross-browser testing
- [ ] Accessibility testing
- [ ] Performance testing

## Conclusion

This browser support strategy ensures the TTRPG MCP Server works across a wide range of browsers while taking advantage of modern features when available through progressive enhancement.