import { MCPClient } from '$lib/api/mcp-client';
import type { MCPSession, MCPMessage } from '$lib/api/mcp-client';
import type { User, Campaign, Session } from '$lib/api/types';

class SessionStore {
	private client = $state<MCPClient | null>(null);
	user = $state<User | null>(null);
	session = $state<MCPSession | null>(null);
	currentCampaign = $state<Campaign | null>(null);
	currentGameSession = $state<Session | null>(null);
	messages = $state<MCPMessage[]>([]);
	private messageUnsubscribe: (() => void) | null = null;

	get isConnected() {
		return this.session?.status === 'connected';
	}

	async connect(user: User, provider: string, apiKey: string) {
		try {
			this.user = user;
			this.client = new MCPClient();
			this.session = await this.client.connect(user.id, provider, apiKey);

			this.messageUnsubscribe = this.client.onMessage((msg) => {
				this.messages = [...this.messages, msg].slice(-100); // Keep only last 100
			});

			return this.session;
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

	setCampaign = (campaign: Campaign) => this.currentCampaign = campaign;
	setGameSession = (session: Session) => this.currentGameSession = session;
	clearMessages = () => this.messages = [];

	disconnect() {
		this.messageUnsubscribe?.();
		this.messageUnsubscribe = null;
		this.client?.disconnect();
		this.client = null;
		[this.session, this.user, this.currentCampaign, this.currentGameSession, this.messages] = [null, null, null, null, []];
	}
}

// Export singleton instance
export const sessionStore = new SessionStore();