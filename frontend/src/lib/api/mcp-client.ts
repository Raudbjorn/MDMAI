import type { Tool, ToolResult } from './types';
import { CacheManager } from '$lib/cache/cache-manager';
import { MetricsCollector } from '$lib/performance/metrics-collector';
import { WebSocketPool } from '$lib/performance/request-optimizer';
import { prepareForTransmission, validateApiKeyFormat } from '$lib/security/api-key-handler';

export interface MCPSession {
	id: string;
	userId: string;
	status: 'connecting' | 'connected' | 'disconnected' | 'error';
	error?: string;
}

export interface MCPMessage {
	type: 'tool_call' | 'tool_result' | 'error' | 'status' | 'collaboration';
	data: any;
	timestamp: number;
	room_id?: string;
	sender_id?: string;
}

export class MCPClient {
	private ws: WebSocket | null = null;
	private wsPool: WebSocketPool | null = null;
	private eventSource: EventSource | null = null;
	private session: MCPSession | null = null;
	private messageHandlers: Set<(msg: MCPMessage) => void> = new Set();
	private baseUrl: string;
	private cacheManager: CacheManager;
	private metricsCollector: MetricsCollector;
	private pendingRequests: Map<string, Promise<any>> = new Map();

	constructor(baseUrl: string = '') {
		this.baseUrl = baseUrl || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000');
		this.cacheManager = new CacheManager();
		this.metricsCollector = new MetricsCollector();
	}

	async connect(userId: string, provider: string, apiKey: string): Promise<MCPSession> {
		const start = performance.now();
		
		try {
			// Validate API key format before sending
			if (!validateApiKeyFormat(apiKey, provider)) {
				throw new Error(`Invalid API key format for provider: ${provider}`);
			}
			
			// Check cache for existing session
			const cacheKey = `session:${userId}:${provider}`;
			const cachedSession = await this.cacheManager.get<MCPSession>(cacheKey);
			
			if (cachedSession && cachedSession.status === 'connected') {
				this.session = cachedSession;
				this.connectWebSocket();
				this.connectSSE();
				
				this.metricsCollector.recordMetric({
					name: 'session_cache_hit',
					value: performance.now() - start,
					timestamp: Date.now(),
					unit: 'ms'
				});
				
				return cachedSession;
			}

			// Prepare API key for secure transmission
			const securePayload = prepareForTransmission(provider, apiKey);
			
			// Initialize session via HTTPS (ensure HTTPS in production)
			if (
				typeof window !== 'undefined' &&
				window.location.protocol !== 'https:' &&
				window.location.hostname !== 'localhost' &&
				window.location.hostname !== '127.0.0.1'
			) {
				console.warn('API keys should only be transmitted over HTTPS in production!');
			}
			
			const response = await fetch(`${this.baseUrl}/api/bridge/session`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'X-Request-Timestamp': securePayload.timestamp.toString(),
					'X-Request-Nonce': securePayload.nonce
				},
				body: JSON.stringify({
					user_id: userId,
					provider: securePayload.provider,
					secure_payload: securePayload.payload
				})
			});

			if (!response.ok) {
				throw new Error(`Failed to create session: ${response.statusText}`);
			}

			this.session = await response.json();
			
			// Cache the session
			await this.cacheManager.set(cacheKey, this.session, {
				ttl: 3600000, // 1 hour
				persistent: true
			});

			// Initialize WebSocket pool for better connection management
			this.wsPool = new WebSocketPool({
				maxConnections: 3,
				url: `${this.baseUrl.replace('http', 'ws')}/api/bridge/ws/${this.session!.id}`,
				reconnectDelay: 1000
			});

			// Use WebSocket pool for the primary connection
			this.connectWebSocketWithPool();

			// Connect SSE for server-pushed updates
			this.connectSSE();
			
			this.metricsCollector.recordMetric({
				name: 'session_connect_time',
				value: performance.now() - start,
				timestamp: Date.now(),
				unit: 'ms'
			});

			return this.session!;
		} catch (error) {
			this.metricsCollector.recordMetric({
				name: 'session_connect_error',
				value: 1,
				timestamp: Date.now(),
				tags: { error: error instanceof Error ? error.message : 'Unknown' }
			});
			
			console.error('Failed to connect to MCP bridge:', error);
			throw error;
		}
	}

	private async connectWebSocketWithPool() {
		if (!this.session || !this.wsPool) return;

		try {
			// Execute WebSocket operations through the pool
			await this.wsPool.execute(async (ws: WebSocket) => {
				this.ws = ws;
				
				// Set up handlers for pooled connection
				ws.onmessage = (event) => {
					try {
						const message: MCPMessage = JSON.parse(event.data);
						this.handleMessage(message);
					} catch (error) {
						console.error('Failed to parse WebSocket message:', error);
					}
				};
				
				// Keep connection alive
				await new Promise(() => {
					// This promise never resolves, keeping the connection alive
				});
			});
		} catch (error) {
			console.error('Failed to connect via WebSocket pool:', error);
			// Fall back to regular connection
			this.connectWebSocket();
		}
	}

	private connectWebSocket() {
		if (!this.session) return;

		const wsUrl = `${this.baseUrl.replace('http', 'ws')}/api/bridge/ws/${this.session.id}`;
		this.ws = new WebSocket(wsUrl);

		this.ws.onopen = () => {
			this.updateStatus('connected');
		};

		this.ws.onmessage = (event) => {
			try {
				const message: MCPMessage = JSON.parse(event.data);
				this.handleMessage(message);
			} catch (error) {
				console.error('Failed to parse WebSocket message:', error);
			}
		};

		this.ws.onerror = (error) => {
			console.error('WebSocket error:', error);
			this.updateStatus('error');
		};

		this.ws.onclose = () => {
			this.updateStatus('disconnected');
			// Use exponential backoff for reconnection
			const baseDelay = 1000;
			const maxDelay = 30000;
			const delay = Math.min(baseDelay * Math.pow(2, this.reconnectAttempts || 0), maxDelay);
			
			setTimeout(() => {
				if (this.session?.status === 'disconnected') {
					this.reconnectAttempts = (this.reconnectAttempts || 0) + 1;
					this.connectWebSocket();
				}
			}, delay);
		};
	}
	
	private reconnectAttempts = 0;

	private connectSSE() {
		if (!this.session) return;

		this.eventSource = new EventSource(`${this.baseUrl}/api/bridge/sse/${this.session.id}`);

		this.eventSource.onmessage = (event) => {
			try {
				const message: MCPMessage = JSON.parse(event.data);
				this.handleMessage(message);
			} catch (error) {
				console.error('Failed to parse SSE message:', error);
			}
		};

		this.eventSource.onerror = (error) => {
			console.error('SSE error:', error);
			// SSE will auto-reconnect
		};
	}

	async callTool(toolName: string, params: Record<string, any>): Promise<ToolResult> {
		const start = performance.now();
		const cacheKey = `tool:${toolName}:${JSON.stringify(params)}`;
		
		// Check for deduplicated request
		if (this.pendingRequests.has(cacheKey)) {
			this.metricsCollector.recordMetric({
				name: 'tool_call_deduplicated',
				value: 1,
				timestamp: Date.now(),
				tags: { tool: toolName }
			});
			return this.pendingRequests.get(cacheKey)!;
		}
		
		// Check cache for idempotent tools
		const idempotentTools = ['search_rules', 'get_character', 'get_campaign', 'list_systems'];
		if (idempotentTools.includes(toolName)) {
			const cached = await this.cacheManager.get<ToolResult>(cacheKey);
			if (cached !== null) {
				this.metricsCollector.recordMetric({
					name: 'tool_cache_hit',
					value: performance.now() - start,
					timestamp: Date.now(),
					unit: 'ms',
					tags: { tool: toolName }
				});
				return cached;
			}
		}

		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
			throw new Error('WebSocket not connected');
		}

		const promise = new Promise<ToolResult>((resolve, reject) => {
			const requestId = `${Date.now()}-${Math.random()}`;

			const handler = (msg: MCPMessage) => {
				if (msg.type === 'tool_result' && msg.data.request_id === requestId) {
					this.messageHandlers.delete(handler);
					if (msg.data.error) {
						reject(new Error(msg.data.error));
					} else {
						resolve(msg.data.result);
					}
				}
			};

			this.messageHandlers.add(handler);

			this.ws!.send(JSON.stringify({
				type: 'tool_call',
				tool: toolName,
				params,
				request_id: requestId
			}));

			// Timeout after 30 seconds
			setTimeout(() => {
				this.messageHandlers.delete(handler);
				reject(new Error('Tool call timed out'));
			}, 30000);
		}).then(async (result) => {
			// Cache result for idempotent tools
			if (idempotentTools.includes(toolName)) {
				await this.cacheManager.set(cacheKey, result, {
					ttl: 600000, // 10 minutes
					persistent: true
				});
			}
			
			this.metricsCollector.recordMetric({
				name: 'tool_call_success',
				value: performance.now() - start,
				timestamp: Date.now(),
				unit: 'ms',
				tags: { tool: toolName }
			});
			
			this.pendingRequests.delete(cacheKey);
			return result;
		}).catch((error) => {
			this.metricsCollector.recordMetric({
				name: 'tool_call_error',
				value: 1,
				timestamp: Date.now(),
				tags: { 
					tool: toolName,
					error: error instanceof Error ? error.message : 'Unknown'
				}
			});
			
			this.pendingRequests.delete(cacheKey);
			throw error;
		});
		
		// Store pending request for deduplication
		this.pendingRequests.set(cacheKey, promise);
		
		return promise;
	}

	private handleMessage(message: MCPMessage) {
		// Notify all message handlers
		this.messageHandlers.forEach(handler => handler(message));

		// Handle status updates
		if (message.type === 'status') {
			this.updateStatus(message.data.status);
		}
	}

	private updateStatus(status: MCPSession['status']) {
		if (this.session) {
			this.session.status = status;
		}
	}

	onMessage(handler: (msg: MCPMessage) => void) {
		this.messageHandlers.add(handler);
		return () => this.messageHandlers.delete(handler);
	}

	disconnect() {
		if (this.ws) {
			this.ws.close();
			this.ws = null;
		}
		if (this.wsPool) {
			this.wsPool.close();
			this.wsPool = null;
		}
		if (this.eventSource) {
			this.eventSource.close();
			this.eventSource = null;
		}
		this.session = null;
		this.messageHandlers.clear();
		this.pendingRequests.clear();
	}

	getSession(): MCPSession | null {
		return this.session;
	}
	
	/**
	 * Prefetch and cache tool results
	 */
	async prefetchTools(tools: Array<{ name: string; params: Record<string, any> }>): Promise<void> {
		const promises = tools.map(tool => 
			this.callTool(tool.name, tool.params).catch(error => {
				console.error(`Failed to prefetch ${tool.name}:`, error);
			})
		);
		
		await Promise.all(promises);
	}
	
	/**
	 * Clear cache for specific tools or all tools
	 */
	async clearCache(toolName?: string): Promise<number> {
		if (toolName) {
			return this.cacheManager.invalidate(`tool:${toolName}:*`);
		}
		return this.cacheManager.invalidate(/^tool:/);
	}
	
	/**
	 * Get performance metrics
	 */
	getMetrics() {
		return {
			performance: this.metricsCollector.generateReport(),
			cache: this.cacheManager.getStats(),
			webSocket: this.wsPool?.getStats() || null
		};
	}
	
	/**
	 * Warm cache with common queries
	 */
	async warmCache(): Promise<void> {
		const commonQueries = [
			{ name: 'list_systems', params: {} },
			{ name: 'get_campaign', params: { id: 'default' } }
		];
		
		await this.cacheManager.warmCache({
			resources: commonQueries.map(q => `tool:${q.name}:${JSON.stringify(q.params)}`),
			priority: 'high',
			schedule: 'background'
		});
	}
}