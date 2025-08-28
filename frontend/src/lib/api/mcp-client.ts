import type { Tool, ToolResult } from './types';

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
	private eventSource: EventSource | null = null;
	private session: MCPSession | null = null;
	private messageHandlers: Set<(msg: MCPMessage) => void> = new Set();
	private baseUrl: string;

	constructor(baseUrl: string = '') {
		this.baseUrl = baseUrl || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000');
	}

	async connect(userId: string, provider: string, apiKey: string): Promise<MCPSession> {
		try {
			// Initialize session via HTTP
			const response = await fetch(`${this.baseUrl}/api/bridge/session`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					user_id: userId,
					provider,
					api_key: apiKey
				})
			});

			if (!response.ok) {
				throw new Error(`Failed to create session: ${response.statusText}`);
			}

			this.session = await response.json();

			// Connect WebSocket for bidirectional communication
			this.connectWebSocket();

			// Connect SSE for server-pushed updates
			this.connectSSE();

			return this.session;
		} catch (error) {
			console.error('Failed to connect to MCP bridge:', error);
			throw error;
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
			// Attempt reconnection after 3 seconds
			setTimeout(() => {
				if (this.session?.status === 'disconnected') {
					this.connectWebSocket();
				}
			}, 3000);
		};
	}

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
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
			throw new Error('WebSocket not connected');
		}

		return new Promise((resolve, reject) => {
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

			this.ws.send(JSON.stringify({
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
		});
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
		if (this.eventSource) {
			this.eventSource.close();
			this.eventSource = null;
		}
		this.session = null;
		this.messageHandlers.clear();
	}

	getSession(): MCPSession | null {
		return this.session;
	}
}