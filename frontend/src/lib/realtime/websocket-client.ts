import { browser } from '$app/environment';
import type { CollaborationMessage } from '$lib/types/collaboration';

export interface WebSocketConfig {
	url: string;
	protocols?: string[];
	reconnectDelay?: number;
	maxReconnectDelay?: number;
	reconnectDecay?: number;
	timeoutInterval?: number;
	maxReconnectAttempts?: number;
	binaryType?: BinaryType;
	enableHeartbeat?: boolean;
	heartbeatInterval?: number;
}

export interface WebSocketState {
	readyState: number;
	isReconnecting: boolean;
	reconnectAttempts: number;
	lastConnectedAt: number | null;
	lastDisconnectedAt: number | null;
	lastError: Event | null;
	latency: number;
}

type MessageHandler<T = any> = (data: T, event: MessageEvent) => void;
type EventHandler = (event: Event) => void;
type CloseHandler = (event: CloseEvent) => void;

/**
 * Enhanced WebSocket client with automatic reconnection, heartbeat, and SvelteKit integration
 */
export class EnhancedWebSocketClient {
	private ws: WebSocket | null = null;
	private config: Required<WebSocketConfig>;
	private reconnectTimer: number | null = null;
	private heartbeatTimer: number | null = null;
	private messageHandlers = new Set<MessageHandler>();
	private openHandlers = new Set<EventHandler>();
	private closeHandlers = new Set<CloseHandler>();
	private errorHandlers = new Set<EventHandler>();
	private messageQueue: any[] = [];
	private state = $state<WebSocketState>({
		readyState: WebSocket.CLOSED,
		isReconnecting: false,
		reconnectAttempts: 0,
		lastConnectedAt: null,
		lastDisconnectedAt: null,
		lastError: null,
		latency: 0
	});
	private pingTimestamp = 0;
	private reconnectAttempts = 0;

	constructor(config: WebSocketConfig) {
		this.config = {
			url: config.url,
			protocols: config.protocols || [],
			reconnectDelay: config.reconnectDelay || 1000,
			maxReconnectDelay: config.maxReconnectDelay || 30000,
			reconnectDecay: config.reconnectDecay || 1.5,
			timeoutInterval: config.timeoutInterval || 2000,
			maxReconnectAttempts: config.maxReconnectAttempts || Infinity,
			binaryType: config.binaryType || 'blob',
			enableHeartbeat: config.enableHeartbeat !== false,
			heartbeatInterval: config.heartbeatInterval || 30000
		};
		
		if (browser) {
			this.connect();
		}
	}

	// Reactive getters using Svelte 5 runes
	get readyState() {
		return this.state.readyState;
	}

	get isConnected() {
		return this.state.readyState === WebSocket.OPEN;
	}

	get isReconnecting() {
		return this.state.isReconnecting;
	}

	get latency() {
		return this.state.latency;
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
	 * Connect to WebSocket server
	 */
	connect(): void {
		if (this.ws?.readyState === WebSocket.OPEN) {
			return;
		}

		try {
			this.ws = new WebSocket(this.config.url, this.config.protocols);
			this.ws.binaryType = this.config.binaryType;
			
			this.setupEventHandlers();
		} catch (error) {
			console.error('WebSocket connection failed:', error);
			this.scheduleReconnect();
		}
	}

	/**
	 * Set up WebSocket event handlers
	 */
	private setupEventHandlers(): void {
		if (!this.ws) return;

		this.ws.onopen = (event) => {
			this.state.readyState = WebSocket.OPEN;
			this.state.isReconnecting = false;
			this.state.reconnectAttempts = 0;
			this.state.lastConnectedAt = Date.now();
			this.reconnectAttempts = 0;
			
			// Start heartbeat
			if (this.config.enableHeartbeat) {
				this.startHeartbeat();
			}
			
			// Process queued messages
			this.flushMessageQueue();
			
			// Notify handlers
			this.openHandlers.forEach(handler => handler(event));
		};

		this.ws.onmessage = (event) => {
			// Handle ping/pong for latency measurement
			if (event.data === 'pong' && this.pingTimestamp > 0) {
				this.state.latency = Date.now() - this.pingTimestamp;
				this.pingTimestamp = 0;
				return;
			}
			
			// Parse and handle message
			try {
				const data = typeof event.data === 'string' 
					? JSON.parse(event.data) 
					: event.data;
				
				this.messageHandlers.forEach(handler => handler(data, event));
			} catch (error) {
				console.error('Failed to parse WebSocket message:', error);
				// Still notify handlers with raw data
				this.messageHandlers.forEach(handler => handler(event.data, event));
			}
		};

		this.ws.onerror = (event) => {
			this.state.lastError = event;
			console.error('WebSocket error:', event);
			
			this.errorHandlers.forEach(handler => handler(event));
		};

		this.ws.onclose = (event) => {
			this.state.readyState = WebSocket.CLOSED;
			this.state.lastDisconnectedAt = Date.now();
			
			this.stopHeartbeat();
			
			// Notify handlers
			this.closeHandlers.forEach(handler => handler(event));
			
			// Schedule reconnection if not a normal closure
			if (!event.wasClean && this.reconnectAttempts < this.config.maxReconnectAttempts) {
				this.scheduleReconnect();
			}
		};
	}

	/**
	 * Send data through WebSocket
	 */
	send(data: any): void {
		const message = typeof data === 'string' ? data : JSON.stringify(data);
		
		if (this.ws?.readyState === WebSocket.OPEN) {
			this.ws.send(message);
		} else {
			// Queue message for later
			this.messageQueue.push(message);
			
			// Try to reconnect if not already trying
			if (!this.state.isReconnecting) {
				this.scheduleReconnect();
			}
		}
	}

	/**
	 * Send and wait for response with timeout
	 */
	async request<T = any>(
		data: any,
		options: {
			timeout?: number;
			responseType?: string;
			requestId?: string;
		} = {}
	): Promise<T> {
		const requestId = options.requestId || `${Date.now()}-${Math.random()}`;
		const timeout = options.timeout || 30000;
		
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.offMessage(handler);
				reject(new Error('Request timeout'));
			}, timeout);
			
			const handler: MessageHandler = (response) => {
				if (response.request_id === requestId) {
					clearTimeout(timer);
					this.offMessage(handler);
					
					if (response.error) {
						reject(new Error(response.error));
					} else {
						resolve(response.data || response);
					}
				}
			};
			
			this.onMessage(handler);
			this.send({ ...data, request_id: requestId });
		});
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
		
		console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
		
		this.reconnectTimer = window.setTimeout(() => {
			this.reconnectTimer = null;
			this.connect();
		}, delay);
	}

	/**
	 * Flush queued messages
	 */
	private flushMessageQueue(): void {
		while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
			const message = this.messageQueue.shift();
			this.ws.send(message);
		}
	}

	/**
	 * Start heartbeat to keep connection alive
	 */
	private startHeartbeat(): void {
		this.stopHeartbeat();
		
		this.heartbeatTimer = window.setInterval(() => {
			if (this.ws?.readyState === WebSocket.OPEN) {
				// Send ping and record timestamp
				this.pingTimestamp = Date.now();
				this.ws.send('ping');
			}
		}, this.config.heartbeatInterval);
	}

	/**
	 * Stop heartbeat
	 */
	private stopHeartbeat(): void {
		if (this.heartbeatTimer) {
			clearInterval(this.heartbeatTimer);
			this.heartbeatTimer = null;
		}
	}

	/**
	 * Register message handler
	 */
	onMessage(handler: MessageHandler): () => void {
		this.messageHandlers.add(handler);
		return () => this.messageHandlers.delete(handler);
	}

	/**
	 * Unregister message handler
	 */
	offMessage(handler: MessageHandler): void {
		this.messageHandlers.delete(handler);
	}

	/**
	 * Register open event handler
	 */
	onOpen(handler: EventHandler): () => void {
		this.openHandlers.add(handler);
		return () => this.openHandlers.delete(handler);
	}

	/**
	 * Register close event handler
	 */
	onClose(handler: CloseHandler): () => void {
		this.closeHandlers.add(handler);
		return () => this.closeHandlers.delete(handler);
	}

	/**
	 * Register error event handler
	 */
	onError(handler: EventHandler): () => void {
		this.errorHandlers.add(handler);
		return () => this.errorHandlers.delete(handler);
	}

	/**
	 * Close WebSocket connection
	 */
	close(code?: number, reason?: string): void {
		// Cancel reconnection
		if (this.reconnectTimer) {
			clearTimeout(this.reconnectTimer);
			this.reconnectTimer = null;
		}
		
		this.stopHeartbeat();
		this.state.isReconnecting = false;
		
		if (this.ws) {
			this.ws.close(code, reason);
			this.ws = null;
		}
		
		// Clear message queue
		this.messageQueue = [];
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
	 * Get current WebSocket instance
	 */
	getWebSocket(): WebSocket | null {
		return this.ws;
	}

	/**
	 * Clear all handlers
	 */
	clearHandlers(): void {
		this.messageHandlers.clear();
		this.openHandlers.clear();
		this.closeHandlers.clear();
		this.errorHandlers.clear();
	}

	/**
	 * Destroy the client
	 */
	destroy(): void {
		this.close();
		this.clearHandlers();
		this.messageQueue = [];
	}
}