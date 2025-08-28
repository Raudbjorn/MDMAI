import { browser } from '$app/environment';

export interface SSEConfig {
	url: string;
	withCredentials?: boolean;
	reconnectDelay?: number;
	maxReconnectDelay?: number;
	reconnectDecay?: number;
	maxReconnectAttempts?: number;
	headers?: Record<string, string>;
}

export interface SSEState {
	readyState: number;
	isReconnecting: boolean;
	reconnectAttempts: number;
	lastConnectedAt: number | null;
	lastDisconnectedAt: number | null;
	lastEventId: string | null;
	lastError: Event | null;
}

export interface SSEMessage {
	id?: string;
	event?: string;
	data: any;
	retry?: number;
}

type MessageHandler = (message: SSEMessage) => void;
type EventHandler = (event: Event) => void;

/**
 * Enhanced Server-Sent Events client with automatic reconnection and SvelteKit integration
 */
export class EnhancedSSEClient {
	private eventSource: EventSource | null = null;
	private config: Required<SSEConfig>;
	private reconnectTimer: number | null = null;
	private messageHandlers = new Map<string, Set<MessageHandler>>();
	private openHandlers = new Set<EventHandler>();
	private errorHandlers = new Set<EventHandler>();
	private state: SSEState = {
		readyState: EventSource.CLOSED,
		isReconnecting: false,
		reconnectAttempts: 0,
		lastConnectedAt: null,
		lastDisconnectedAt: null,
		lastEventId: null,
		lastError: null
	};
	private reconnectAttempts = 0;
	private abortController: AbortController | null = null;

	constructor(config: SSEConfig) {
		this.config = {
			url: config.url,
			withCredentials: config.withCredentials ?? true,
			reconnectDelay: config.reconnectDelay || 1000,
			maxReconnectDelay: config.maxReconnectDelay || 30000,
			reconnectDecay: config.reconnectDecay || 1.5,
			maxReconnectAttempts: config.maxReconnectAttempts || Infinity,
			headers: config.headers || {}
		};
		
		if (browser) {
			this.connect();
		}
	}

	// Getters for state access
	get readyState() {
		return this.state.readyState;
	}

	get isConnected() {
		return this.state.readyState === EventSource.OPEN;
	}

	get isReconnecting() {
		return this.state.isReconnecting;
	}

	get lastEventId() {
		return this.state.lastEventId;
	}

	get connectionStats() {
		return {
			...this.state,
			uptime: this.state.lastConnectedAt 
				? Date.now() - this.state.lastConnectedAt 
				: 0,
			downtime: this.state.lastDisconnectedAt && !this.isConnected
				? Date.now() - this.state.lastDisconnectedAt
				: 0
		};
	}

	/**
	 * Connect to SSE endpoint
	 */
	connect(): void {
		if (this.eventSource?.readyState === EventSource.OPEN) {
			return;
		}

		try {
			// Build URL with last event ID if available
			let url = this.config.url;
			if (this.state.lastEventId) {
				const separator = url.includes('?') ? '&' : '?';
				url += `${separator}lastEventId=${encodeURIComponent(this.state.lastEventId)}`;
			}

			// Create EventSource with credentials
			this.eventSource = new EventSource(url, {
				withCredentials: this.config.withCredentials
			});
			
			this.setupEventHandlers();
		} catch (error) {
			console.error('SSE connection failed:', error);
			this.scheduleReconnect();
		}
	}

	/**
	 * Connect using fetch with custom headers (alternative method)
	 */
	async connectWithFetch(): Promise<void> {
		if (this.abortController) {
			this.abortController.abort();
		}

		this.abortController = new AbortController();

		try {
			const response = await fetch(this.config.url, {
				headers: {
					'Accept': 'text/event-stream',
					'Cache-Control': 'no-cache',
					...this.config.headers
				},
				credentials: this.config.withCredentials ? 'include' : 'same-origin',
				signal: this.abortController.signal
			});

			if (!response.ok) {
				throw new Error(`SSE connection failed: ${response.statusText}`);
			}

			if (!response.body) {
				throw new Error('No response body');
			}

			this.state.readyState = EventSource.OPEN;
			this.state.isReconnecting = false;
			this.state.reconnectAttempts = 0;
			this.state.lastConnectedAt = Date.now();
			this.reconnectAttempts = 0;

			// Process stream
			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();
				
				if (done) {
					break;
				}

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');
				buffer = lines.pop() || '';

				for (const line of lines) {
					this.processLine(line);
				}
			}
		} catch (error: any) {
			if (error.name !== 'AbortError') {
				console.error('SSE fetch connection failed:', error);
				this.state.lastError = error;
				this.scheduleReconnect();
			}
		} finally {
			this.state.readyState = EventSource.CLOSED;
			this.state.lastDisconnectedAt = Date.now();
		}
	}

	/**
	 * Process SSE line from fetch stream
	 */
	private processLine(line: string): void {
		if (line.startsWith('data: ')) {
			const data = line.slice(6);
			this.handleMessage({ data: this.parseData(data) });
		} else if (line.startsWith('event: ')) {
			// Store event type for next data
		} else if (line.startsWith('id: ')) {
			this.state.lastEventId = line.slice(4);
		} else if (line.startsWith('retry: ')) {
			const retry = parseInt(line.slice(7));
			if (!isNaN(retry)) {
				this.config.reconnectDelay = retry;
			}
		}
	}

	/**
	 * Set up EventSource event handlers
	 */
	private setupEventHandlers(): void {
		if (!this.eventSource) return;

		this.eventSource.onopen = (event) => {
			this.state.readyState = EventSource.OPEN;
			this.state.isReconnecting = false;
			this.state.reconnectAttempts = 0;
			this.state.lastConnectedAt = Date.now();
			this.reconnectAttempts = 0;
			
			// Notify handlers
			this.openHandlers.forEach(handler => handler(event));
		};

		this.eventSource.onmessage = (event) => {
			// Update last event ID
			if (event.lastEventId) {
				this.state.lastEventId = event.lastEventId;
			}
			
			// Parse and handle message
			const message: SSEMessage = {
				id: event.lastEventId,
				event: event.type,
				data: this.parseData(event.data)
			};
			
			this.handleMessage(message);
		};

		this.eventSource.onerror = (event) => {
			this.state.lastError = event;
			console.error('SSE error:', event);
			
			this.errorHandlers.forEach(handler => handler(event));
			
			// EventSource automatically reconnects, but we want to control it
			if (this.eventSource?.readyState === EventSource.CLOSED) {
				this.state.readyState = EventSource.CLOSED;
				this.state.lastDisconnectedAt = Date.now();
				
				if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
					this.scheduleReconnect();
				}
			}
		};

		// Add custom event listeners
		this.messageHandlers.forEach((handlers, eventType) => {
			if (eventType !== 'message' && this.eventSource) {
				this.eventSource.addEventListener(eventType, (event: any) => {
					const message: SSEMessage = {
						id: event.lastEventId,
						event: event.type,
						data: this.parseData(event.data)
					};
					
					handlers.forEach(handler => handler(message));
				});
			}
		});
	}

	/**
	 * Parse SSE data
	 */
	private parseData(data: string): any {
		try {
			return JSON.parse(data);
		} catch {
			return data;
		}
	}

	/**
	 * Handle incoming message
	 */
	private handleMessage(message: SSEMessage): void {
		// Handle default message event
		const handlers = this.messageHandlers.get(message.event || 'message');
		if (handlers) {
			handlers.forEach(handler => handler(message));
		}
		
		// Always notify 'all' handlers
		const allHandlers = this.messageHandlers.get('all');
		if (allHandlers) {
			allHandlers.forEach(handler => handler(message));
		}
	}

	/**
	 * Schedule reconnection with exponential backoff
	 */
	private scheduleReconnect(): void {
		if (this.reconnectTimer) {
			return;
		}

		this.state.isReconnecting = true;
		this.state.reconnectAttempts++;
		this.reconnectAttempts++;
		
		const delay = Math.min(
			this.config.reconnectDelay * Math.pow(this.config.reconnectDecay, this.reconnectAttempts - 1),
			this.config.maxReconnectDelay
		);
		
		console.log(`SSE reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
		
		this.reconnectTimer = window.setTimeout(() => {
			this.reconnectTimer = null;
			this.connect();
		}, delay);
	}

	/**
	 * Register message handler for specific event type
	 */
	onMessage(eventType: string, handler: MessageHandler): () => void {
		if (!this.messageHandlers.has(eventType)) {
			this.messageHandlers.set(eventType, new Set());
			
			// Add event listener if EventSource exists and not default message
			if (this.eventSource && eventType !== 'message' && eventType !== 'all') {
				this.eventSource.addEventListener(eventType, (event: any) => {
					const message: SSEMessage = {
						id: event.lastEventId,
						event: event.type,
						data: this.parseData(event.data)
					};
					
					const handlers = this.messageHandlers.get(eventType);
					if (handlers) {
						handlers.forEach(h => h(message));
					}
				});
			}
		}
		
		this.messageHandlers.get(eventType)!.add(handler);
		
		return () => {
			const handlers = this.messageHandlers.get(eventType);
			if (handlers) {
				handlers.delete(handler);
				if (handlers.size === 0) {
					this.messageHandlers.delete(eventType);
				}
			}
		};
	}

	/**
	 * Register open event handler
	 */
	onOpen(handler: EventHandler): () => void {
		this.openHandlers.add(handler);
		return () => this.openHandlers.delete(handler);
	}

	/**
	 * Register error event handler
	 */
	onError(handler: EventHandler): () => void {
		this.errorHandlers.add(handler);
		return () => this.errorHandlers.delete(handler);
	}

	/**
	 * Close SSE connection
	 */
	close(): void {
		// Cancel reconnection
		if (this.reconnectTimer) {
			clearTimeout(this.reconnectTimer);
			this.reconnectTimer = null;
		}
		
		this.state.isReconnecting = false;
		
		if (this.eventSource) {
			this.eventSource.close();
			this.eventSource = null;
		}
		
		if (this.abortController) {
			this.abortController.abort();
			this.abortController = null;
		}
	}

	/**
	 * Reconnect immediately
	 */
	reconnect(): void {
		this.close();
		this.reconnectAttempts = 0;
		this.connect();
	}

	/**
	 * Clear all handlers
	 */
	clearHandlers(): void {
		this.messageHandlers.clear();
		this.openHandlers.clear();
		this.errorHandlers.clear();
	}

	/**
	 * Destroy the client
	 */
	destroy(): void {
		this.close();
		this.clearHandlers();
	}
}