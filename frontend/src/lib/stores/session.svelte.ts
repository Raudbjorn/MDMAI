import { MCPClient } from '$lib/api/mcp-client';
import type { MCPSession, MCPMessage } from '$lib/api/mcp-client';
import type { User, Campaign, Session } from '$lib/api/types';

class SessionStore {
	private client = $state<MCPClient | null>(null);
	private _user = $state<User | null>(null);
	private _session = $state<MCPSession | null>(null);
	private _currentCampaign = $state<Campaign | null>(null);
	private _currentGameSession = $state<Session | null>(null);
	private _messages = $state<MCPMessage[]>([]);
	private messageUnsubscribe: (() => void) | null = null;

	get user() {
		return this._user;
	}

	get session() {
		return this._session;
	}

	get currentCampaign() {
		return this._currentCampaign;
	}

	get currentGameSession() {
		return this._currentGameSession;
	}

	get messages() {
		return this._messages;
	}

	get isConnected() {
		return this._session?.status === 'connected';
	}

	async connect(user: User, provider: string, apiKey: string) {
		try {
			this._user = user;
			
			// Initialize MCP client
			this.client = new MCPClient();
			this._session = await this.client.connect(user.id, provider, apiKey);

			// Subscribe to messages
			this.messageUnsubscribe = this.client.onMessage((msg) => {
				this._messages = [...this._messages, msg];
				// Keep only last 100 messages
				if (this._messages.length > 100) {
					this._messages = this._messages.slice(-100);
				}
			});

			return this._session;
		} catch (error) {
			console.error('Failed to connect session:', error);
			throw error;
		}
	}

	async callTool(toolName: string, params: Record<string, any>) {
		if (!this.client) {
			throw new Error('Client not initialized');
		}
		return await this.client.callTool(toolName, params);
	}

	setCampaign(campaign: Campaign) {
		this._currentCampaign = campaign;
	}

	setGameSession(session: Session) {
		this._currentGameSession = session;
	}

	clearMessages() {
		this._messages = [];
	}

	disconnect() {
		if (this.messageUnsubscribe) {
			this.messageUnsubscribe();
			this.messageUnsubscribe = null;
		}
		if (this.client) {
			this.client.disconnect();
			this.client = null;
		}
		this._session = null;
		this._user = null;
		this._currentCampaign = null;
		this._currentGameSession = null;
		this._messages = [];
	}
}

// Export singleton instance
export const sessionStore = new SessionStore();