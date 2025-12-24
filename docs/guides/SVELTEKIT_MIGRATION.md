# SvelteKit Migration & MCP Integration Architecture

## Executive Summary

This document outlines the architectural changes and integration patterns required for migrating the TTRPG Assistant MCP Server frontend from React to Svelte/SvelteKit while maintaining full MCP protocol compliance. The migration focuses on leveraging SvelteKit's SSR capabilities, simplified state management, and built-in API routes for optimal MCP bridge communication.

## Key Technology Stack Changes

### Frontend Migration
- **FROM**: React 18 + TypeScript + Zustand + Vite
- **TO**: SvelteKit 2.x + TypeScript + Built-in stores + Vite

### Backend (Unchanged)
- **MCP Server**: FastMCP (Python) with stdio communication
- **Bridge Service**: FastAPI with WebSocket/SSE support
- **Database**: ChromaDB for vector storage

### Removed Components
- **Mobile App (Phase 21)**: Focusing on responsive web-only approach
- **React Native**: Replaced with SvelteKit's responsive design

## Architecture Overview

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   SvelteKit App     │────▶│  Bridge Service  │────▶│   MCP Server    │
│   (SSR + CSR)       │ WS  │    (FastAPI)     │stdio │   (FastMCP)     │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
         │                           │                         │
         ├──────────────┐           │                         │
         │              │           │                         │
    ┌────▼────┐   ┌────▼────┐  ┌───▼────┐            ┌──────▼──────┐
    │  +page  │   │  +api   │  │Session │            │  ChromaDB   │
    │ routes  │   │ routes  │  │Manager │            │   Vector    │
    └─────────┘   └─────────┘  └────────┘            │   Storage   │
```

## 1. SvelteKit SSR Integration with MCP

### 1.1 Server-Side MCP Communication Architecture

SvelteKit's server-side rendering provides unique advantages for MCP integration:

```typescript
// src/lib/server/mcp-client.ts
import type { Result } from '$lib/types/result';
import { WebSocket } from 'ws';
import { error, ok } from '$lib/utils/result';

export class MCPServerClient {
    private ws: WebSocket | null = null;
    private sessionId: string | null = null;
    private messageQueue: Map<string, (result: Result<any>) => void> = new Map();
    
    constructor(
        private bridgeUrl: string = import.meta.env.VITE_BRIDGE_URL || 'ws://localhost:8080/ws'
    ) {}
    
    async connect(): Promise<Result<string>> {
        try {
            this.ws = new WebSocket(this.bridgeUrl);
            
            return new Promise((resolve) => {
                this.ws!.on('open', () => {
                    this.initializeSession().then(resolve);
                });
                
                this.ws!.on('message', (data) => {
                    this.handleMessage(data.toString());
                });
                
                this.ws!.on('error', (err) => {
                    resolve(error(`WebSocket error: ${err.message}`));
                });
            });
        } catch (err) {
            return error(`Failed to connect: ${err}`);
        }
    }
    
    private async initializeSession(): Promise<Result<string>> {
        const response = await this.sendRequest({
            type: 'create_session',
            client_id: 'sveltekit-server',
            metadata: {
                platform: 'sveltekit',
                ssr: true
            }
        });
        
        if (response.success && response.data?.session_id) {
            this.sessionId = response.data.session_id;
            return ok(this.sessionId);
        }
        
        return error('Failed to create session');
    }
    
    async callTool<T>(method: string, params: any): Promise<Result<T>> {
        if (!this.sessionId) {
            return error('No active session');
        }
        
        const id = crypto.randomUUID();
        
        return new Promise((resolve) => {
            this.messageQueue.set(id, resolve);
            
            this.ws?.send(JSON.stringify({
                jsonrpc: '2.0',
                id,
                method: `tools/${method}`,
                params
            }));
            
            // Timeout after 30 seconds
            setTimeout(() => {
                if (this.messageQueue.has(id)) {
                    this.messageQueue.delete(id);
                    resolve(error('Request timeout'));
                }
            }, 30000);
        });
    }
    
    private handleMessage(data: string) {
        try {
            const message = JSON.parse(data);
            
            if (message.id && this.messageQueue.has(message.id)) {
                const callback = this.messageQueue.get(message.id)!;
                this.messageQueue.delete(message.id);
                
                if (message.error) {
                    callback(error(message.error.message));
                } else {
                    callback(ok(message.result));
                }
            }
        } catch (err) {
            console.error('Failed to parse message:', err);
        }
    }
}

// Singleton instance for SSR
let mcpClient: MCPServerClient | null = null;

export function getMCPClient(): MCPServerClient {
    if (!mcpClient) {
        mcpClient = new MCPServerClient();
    }
    return mcpClient;
}
```

### 1.2 Load Functions with MCP Integration

```typescript
// src/routes/campaigns/[id]/+page.server.ts
import type { PageServerLoad } from './$types';
import { getMCPClient } from '$lib/server/mcp-client';
import { error } from '@sveltejs/kit';

export const load: PageServerLoad = async ({ params, setHeaders }) => {
    const client = getMCPClient();
    
    // Ensure connection
    const connectionResult = await client.connect();
    if (!connectionResult.success) {
        throw error(503, 'MCP service unavailable');
    }
    
    // Fetch campaign data via MCP
    const campaignResult = await client.callTool('get_campaign_data', {
        campaign_id: params.id,
        include_related: true
    });
    
    if (!campaignResult.success) {
        throw error(404, 'Campaign not found');
    }
    
    // Set cache headers for SSR optimization
    setHeaders({
        'cache-control': 'private, max-age=60'
    });
    
    return {
        campaign: campaignResult.data,
        streamedData: {
            // Stream additional data for progressive enhancement
            npcs: client.callTool('get_campaign_npcs', { campaign_id: params.id }),
            sessions: client.callTool('get_campaign_sessions', { campaign_id: params.id })
        }
    };
};
```

## 2. API Routes for MCP Tool Exposure

### 2.1 RESTful API Endpoints

```typescript
// src/routes/api/mcp/tools/+server.ts
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getMCPClient } from '$lib/server/mcp-client';
import { z } from 'zod';

const ToolCallSchema = z.object({
    tool: z.string(),
    params: z.record(z.any())
});

export const POST: RequestHandler = async ({ request }) => {
    try {
        const body = await request.json();
        const validation = ToolCallSchema.safeParse(body);
        
        if (!validation.success) {
            return json(
                { error: 'Invalid request', details: validation.error.flatten() },
                { status: 400 }
            );
        }
        
        const client = getMCPClient();
        const result = await client.callTool(validation.data.tool, validation.data.params);
        
        if (!result.success) {
            return json(
                { error: result.error },
                { status: 500 }
            );
        }
        
        return json({
            success: true,
            data: result.data
        });
    } catch (err) {
        return json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
};
```

### 2.2 Form Actions for Progressive Enhancement

```typescript
// src/routes/campaigns/create/+page.server.ts
import type { Actions } from './$types';
import { fail, redirect } from '@sveltejs/kit';
import { getMCPClient } from '$lib/server/mcp-client';

export const actions = {
    default: async ({ request }) => {
        const formData = await request.formData();
        const name = formData.get('name');
        const system = formData.get('system');
        const description = formData.get('description');
        
        if (!name || !system) {
            return fail(400, {
                error: 'Name and system are required',
                values: { name, system, description }
            });
        }
        
        const client = getMCPClient();
        const result = await client.callTool('create_campaign', {
            name: name.toString(),
            system: system.toString(),
            description: description?.toString()
        });
        
        if (!result.success) {
            return fail(500, {
                error: result.error,
                values: { name, system, description }
            });
        }
        
        throw redirect(303, `/campaigns/${result.data.id}`);
    }
} satisfies Actions;
```

## 3. WebSocket/SSE Real-time Updates

### 3.1 WebSocket Store for Client-Side Updates

```typescript
// src/lib/stores/websocket.ts
import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

interface WebSocketState {
    connected: boolean;
    sessionId: string | null;
    error: string | null;
}

function createWebSocketStore() {
    const { subscribe, set, update } = writable<WebSocketState>({
        connected: false,
        sessionId: null,
        error: null
    });
    
    let ws: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;
    
    function connect() {
        if (!browser) return;
        
        ws = new WebSocket(import.meta.env.VITE_WS_URL || 'ws://localhost:8080/ws');
        
        ws.onopen = () => {
            update(s => ({ ...s, connected: true, error: null }));
            
            // Create client session
            ws!.send(JSON.stringify({
                type: 'create_session',
                client_id: 'sveltekit-client',
                metadata: { browser: true }
            }));
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'session_created') {
                update(s => ({ ...s, sessionId: data.session_id }));
            }
            
            // Dispatch custom events for tool responses
            if (data.type === 'response') {
                window.dispatchEvent(new CustomEvent('mcp-response', {
                    detail: data.data
                }));
            }
        };
        
        ws.onerror = (error) => {
            update(s => ({ ...s, error: 'WebSocket error' }));
        };
        
        ws.onclose = () => {
            update(s => ({ ...s, connected: false }));
            
            // Auto-reconnect after 5 seconds
            clearTimeout(reconnectTimeout);
            reconnectTimeout = setTimeout(connect, 5000);
        };
    }
    
    function send(data: any) {
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(data));
        }
    }
    
    function disconnect() {
        clearTimeout(reconnectTimeout);
        ws?.close();
        ws = null;
    }
    
    return {
        subscribe,
        connect,
        send,
        disconnect
    };
}

export const websocket = createWebSocketStore();
```

### 3.2 Server-Sent Events for Unidirectional Updates

```typescript
// src/routes/api/events/[session]/+server.ts
import type { RequestHandler } from './$types';
import { getMCPClient } from '$lib/server/mcp-client';

export const GET: RequestHandler = async ({ params, setHeaders }) => {
    setHeaders({
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    });
    
    const stream = new ReadableStream({
        async start(controller) {
            const client = getMCPClient();
            
            // Send initial connection event
            controller.enqueue(`event: connected\ndata: ${JSON.stringify({
                session_id: params.session,
                timestamp: Date.now()
            })}\n\n`);
            
            // Set up heartbeat
            const heartbeat = setInterval(() => {
                controller.enqueue(`event: heartbeat\ndata: ${Date.now()}\n\n`);
            }, 30000);
            
            // Note: MCPServerClient needs EventEmitter functionality for server-pushed events
            // Alternative implementation using polling until EventEmitter is added:
            let lastUpdateTime = Date.now();
            const pollInterval = setInterval(async () => {
                try {
                    const updates = await client.callTool('get_session_updates', {
                        session_id: params.session,
                        since: lastUpdateTime
                    });
                    if (updates.events?.length > 0) {
                        updates.events.forEach(event => {
                            controller.enqueue(`event: ${event.type}\ndata: ${JSON.stringify(event.data)}\n\n`);
                        });
                        lastUpdateTime = Date.now();
                    }
                } catch (error) {
                    console.error('Error polling for updates:', error);
                }
            }, 1000); // Poll every second
            
            // Cleanup on close
            return () => {
                clearInterval(heartbeat);
                clearInterval(pollInterval);
            };
        }
    });
    
    return new Response(stream);
};
```

## 4. Error Handling with Result Pattern

### 4.1 Result Type Definition

```typescript
// src/lib/types/result.ts
export type Result<T, E = string> = 
    | { success: true; data: T }
    | { success: false; error: E };

export type AsyncResult<T, E = string> = Promise<Result<T, E>>;

// Helper functions
export const ok = <T>(data: T): Result<T> => ({ success: true, data });
export const error = <E = string>(error: E): Result<never, E> => ({ success: false, error });

// Utility functions for Result manipulation
export function map<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => U
): Result<U, E> {
    return result.success ? ok(fn(result.data)) : result;
}

export function flatMap<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => Result<U, E>
): Result<U, E> {
    return result.success ? fn(result.data) : result;
}

export function withDefault<T, E>(
    result: Result<T, E>,
    defaultValue: T
): T {
    return result.success ? result.data : defaultValue;
}
```

### 4.2 Error Boundary Component

```svelte
<!-- src/lib/components/ErrorBoundary.svelte -->
<script lang="ts">
    import { onMount } from 'svelte';
    import type { Result } from '$lib/types/result';
    
    export let result: Result<any> | undefined = undefined;
    export let loading = false;
    export let retry: (() => void) | undefined = undefined;
    
    let error: string | null = null;
    
    $: if (result && !result.success) {
        error = result.error;
    }
    
    onMount(() => {
        // Global error handler for unhandled MCP errors
        window.addEventListener('mcp-error', (event: CustomEvent) => {
            error = event.detail.message;
        });
    });
</script>

{#if loading}
    <div class="flex justify-center items-center p-8">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
    </div>
{:else if error}
    <div class="alert alert-error shadow-lg">
        <div>
            <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current flex-shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{error}</span>
        </div>
        {#if retry}
            <button on:click={retry} class="btn btn-sm">Retry</button>
        {/if}
    </div>
{:else}
    <slot />
{/if}
```

## 5. Python Dependencies Modernization

### 5.1 Updated Python Dependencies with Result Pattern

```python
# src/lib/result.py
from typing import TypeVar, Generic, Union, Callable, Optional
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E')
U = TypeVar('U')

@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T
    
    @property
    def is_ok(self) -> bool:
        return True
    
    @property
    def is_err(self) -> bool:
        return False

@dataclass(frozen=True)
class Err(Generic[E]):
    error: E
    
    @property
    def is_ok(self) -> bool:
        return False
    
    @property
    def is_err(self) -> bool:
        return True

Result = Union[Ok[T], Err[E]]

def ok(value: T) -> Result[T, E]:
    """Create a successful result."""
    return Ok(value)

def err(error: E) -> Result[T, E]:
    """Create an error result."""
    return Err(error)

def map_result(result: Result[T, E], fn: Callable[[T], U]) -> Result[U, E]:
    """Map a function over the success value of a Result."""
    if isinstance(result, Ok):
        return Ok(fn(result.value))
    return result

def flat_map(result: Result[T, E], fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
    """Flat map a function over the success value of a Result."""
    if isinstance(result, Ok):
        return fn(result.value)
    return result

def unwrap_or(result: Result[T, E], default: T) -> T:
    """Unwrap a Result or return a default value."""
    if isinstance(result, Ok):
        return result.value
    return default
```

### 5.2 MCP Tools with Result Pattern

```python
# src/campaign/mcp_tools.py
from typing import Dict, Any, Optional
from src.lib.result import Result, ok, err
from src.campaign.campaign_manager import CampaignManager
import logging

logger = logging.getLogger(__name__)

def register_campaign_tools(mcp_server, campaign_manager: CampaignManager):
    """Register campaign tools with the main MCP server using Result pattern."""
    
    @mcp_server.tool()
    async def create_campaign(
        name: str,
        system: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new campaign with error-as-values pattern."""
        result: Result[Dict, str] = await campaign_manager.create_campaign_safe(
            name=name,
            system=system,
            description=description
        )
        
        if result.is_ok:
            return {
                "success": True,
                "message": f"Campaign '{name}' created successfully",
                "data": result.value
            }
        else:
            logger.error(f"Failed to create campaign: {result.error}")
            return {
                "success": False,
                "error": result.error
            }
    
    @mcp_server.tool()
    async def get_campaign_data(
        campaign_id: str,
        data_type: Optional[str] = None,
        include_related: bool = False
    ) -> Dict[str, Any]:
        """Retrieve campaign data with Result pattern."""
        result = await campaign_manager.get_campaign_safe(
            campaign_id=campaign_id,
            data_type=data_type,
            include_related=include_related
        )
        
        if result.is_ok:
            return {
                "success": True,
                "query": f"campaign:{campaign_id}",
                "data": result.value,
                "metadata": {
                    "include_related": include_related,
                    "data_type": data_type
                }
            }
        else:
            logger.error(f"Failed to get campaign data: {result.error}")
            return {
                "success": False,
                "error": result.error,
                "campaign_id": campaign_id
            }
```

## 6. Migration Implementation Plan

### Phase 1: Foundation (Week 1)
- [x] Create this migration documentation
- [ ] Set up SvelteKit project structure
- [ ] Implement Result pattern utilities (TypeScript & Python)
- [ ] Create MCP server client for SSR

### Phase 2: Bridge Integration (Week 2)
- [ ] Update bridge service for SvelteKit compatibility
- [ ] Implement WebSocket store
- [ ] Create SSE endpoints
- [ ] Set up API routes for MCP tools

### Phase 3: UI Components (Weeks 3-4)
- [ ] Migrate campaign management UI
- [ ] Implement search interface
- [ ] Create session management views
- [ ] Build character generation forms

### Phase 4: State Management (Week 5)
- [ ] Implement Svelte stores for app state
- [ ] Create derived stores for computed data
- [ ] Set up persistent storage (localStorage/sessionStorage)
- [ ] Implement optimistic UI updates

### Phase 5: Testing & Optimization (Week 6)
- [ ] Write unit tests for Result pattern
- [ ] Create integration tests for MCP communication
- [ ] Implement E2E tests with Playwright
- [ ] Performance optimization and bundle size reduction

## 7. Key Architecture Benefits

### SvelteKit Advantages
1. **Simplified State Management**: Built-in stores replace Zustand
2. **Better Performance**: Smaller bundle size, faster initial load
3. **SSR by Default**: Better SEO and initial render performance
4. **Unified Routing**: File-based routing for pages and API
5. **Progressive Enhancement**: Forms work without JavaScript

### MCP Integration Benefits
1. **Type Safety**: End-to-end TypeScript with Result pattern
2. **Error Resilience**: Explicit error handling without exceptions
3. **Stream Support**: Native support for streaming responses
4. **Session Persistence**: SSR maintains MCP sessions server-side
5. **Reduced Latency**: Server-side MCP calls eliminate round trips

## 8. Security Considerations

### Authentication Flow
```typescript
// src/hooks.server.ts
import type { Handle } from '@sveltejs/kit';
import { getMCPClient } from '$lib/server/mcp-client';

export const handle: Handle = async ({ event, resolve }) => {
    // Verify authentication token
    const token = event.cookies.get('auth-token');
    
    if (token) {
        // Validate token and attach user to locals
        event.locals.user = await validateToken(token);
        
        // Initialize MCP session for authenticated user
        if (event.locals.user) {
            const client = getMCPClient();
            await client.connectForUser(event.locals.user.id);
        }
    }
    
    return resolve(event);
};
```

### Rate Limiting
```typescript
// src/lib/server/rate-limit.ts
import { RateLimiter } from 'sveltekit-rate-limiter/server';

export const limiter = new RateLimiter({
    IP: [10, 's'],      // 10 requests per second per IP
    IPUA: [5, 's'],     // 5 requests per second per IP+UserAgent
    cookie: {
        name: 'rate-limit-token',
        secret: import.meta.env.VITE_RATE_LIMIT_SECRET,
        rate: [100, 'm']  // 100 requests per minute per session
    }
});
```

## 9. Performance Optimization

### Caching Strategy
```typescript
// src/lib/server/cache.ts
import { building } from '$app/environment';
import type { Result } from '$lib/types/result';

class MCPCache {
    private cache = new Map<string, { data: any; expires: number }>();
    
    set(key: string, data: any, ttl: number = 60000) {
        if (!building) {
            this.cache.set(key, {
                data,
                expires: Date.now() + ttl
            });
        }
    }
    
    get<T>(key: string): Result<T> | null {
        const entry = this.cache.get(key);
        
        if (!entry) return null;
        
        if (Date.now() > entry.expires) {
            this.cache.delete(key);
            return null;
        }
        
        return ok(entry.data);
    }
    
    invalidate(pattern: string | RegExp) {
        for (const key of this.cache.keys()) {
            if (typeof pattern === 'string' ? key.includes(pattern) : pattern.test(key)) {
                this.cache.delete(key);
            }
        }
    }
}

export const mcpCache = new MCPCache();
```

## 10. Monitoring & Observability

### Metrics Collection
```typescript
// src/lib/server/metrics.ts
import { Counter, Histogram, register } from 'prom-client';

export const mcpRequestCounter = new Counter({
    name: 'mcp_requests_total',
    help: 'Total number of MCP requests',
    labelNames: ['method', 'status']
});

export const mcpRequestDuration = new Histogram({
    name: 'mcp_request_duration_seconds',
    help: 'MCP request duration in seconds',
    labelNames: ['method'],
    buckets: [0.1, 0.5, 1, 2, 5]
});

register.registerMetric(mcpRequestCounter);
register.registerMetric(mcpRequestDuration);

// Metrics endpoint
// src/routes/api/metrics/+server.ts
import { register } from 'prom-client';

export async function GET() {
    return new Response(await register.metrics(), {
        headers: {
            'Content-Type': register.contentType
        }
    });
}
```

## Conclusion

This migration to SvelteKit provides a more streamlined architecture for MCP integration while maintaining all protocol compliance requirements. The combination of SSR, built-in stores, and the Result pattern creates a robust foundation for reliable MCP communication with excellent error handling and performance characteristics.

The removal of the separate mobile app development phase allows focus on a truly responsive web experience that works across all devices, leveraging SvelteKit's excellent mobile performance and progressive enhancement capabilities.