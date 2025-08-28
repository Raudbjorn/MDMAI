import { MCPClient } from '$lib/api/mcp-client';
import type {
	CollaborativeRoom,
	Participant,
	RoomInvitation,
	CollaborationMessage,
	SharedState,
	StateUpdate,
	ConflictResolution,
	ChatMessage,
	TurnState,
	DiceRoll,
	CursorPosition,
	Permission,
	ParticipantRole
} from '$lib/types/collaboration';

interface CollaborationState {
	rooms: CollaborativeRoom[];
	currentRoom: CollaborativeRoom | null;
	invitations: RoomInvitation[];
	participants: Map<string, Participant>;
	messages: ChatMessage[];
	presence: Map<string, CursorPosition>;
	pendingUpdates: StateUpdate[];
	conflicts: ConflictResolution[];
	isConnected: boolean;
	isSyncing: boolean;
	lastSyncTime: number;
}

class CollaborationStore {
	// State using Svelte 5 runes
	private state = $state<CollaborationState>({
		rooms: [],
		currentRoom: null,
		invitations: [],
		participants: new Map(),
		messages: [],
		presence: new Map(),
		pendingUpdates: [],
		conflicts: [],
		isConnected: false,
		isSyncing: false,
		lastSyncTime: Date.now()
	});

	private ws: WebSocket | null = null;
	private reconnectTimer: number | null = null;
	private reconnectAttempts: number = 0;
	private maxReconnectAttempts: number = 10;
	private heartbeatInterval: number | null = null;
	private syncInterval: number | null = null;
	private messageHandlers = new Map<string, Set<(msg: CollaborationMessage) => void>>();
	private currentUserId: string = '';

	// Getters for reactive state
	get rooms() {
		return this.state.rooms;
	}

	get currentRoom() {
		return this.state.currentRoom;
	}

	get invitations() {
		return this.state.invitations;
	}

	get participants() {
		return Array.from(this.state.participants.values());
	}

	get messages() {
		return this.state.messages;
	}

	get presence() {
		return this.state.presence;
	}

	get isConnected() {
		return this.state.isConnected;
	}

	get isSyncing() {
		return this.state.isSyncing;
	}

	get conflicts() {
		return this.state.conflicts;
	}

	get currentParticipant(): Participant | undefined {
		return this.state.participants.get(this.currentUserId);
	}

	get hasPermission() {
		return (action: string, resource: string): boolean => {
			const participant = this.currentParticipant;
			if (!participant) return false;
			
			// Host has all permissions
			if (participant.role === 'host') return true;
			
			// GM has most permissions except managing participants
			if (participant.role === 'gm' && action !== 'manage_participants') return true;
			
			// Check specific permissions
			return participant.permissions.some(
				p => p.action === action && p.resource === resource && p.granted
			);
		};
	}

	// Initialize connection
	async connect(userId: string, baseUrl: string = '') {
		this.currentUserId = userId;
		const wsUrl = `${(baseUrl || window.location.origin).replace('http', 'ws')}/api/collaboration/ws`;
		
		try {
			this.ws = new WebSocket(wsUrl);
			this.setupWebSocketHandlers();
			this.startHeartbeat();
			this.startAutoSync();
		} catch (error) {
			console.error('Failed to connect to collaboration server:', error);
			this.scheduleReconnect();
		}
	}

	private setupWebSocketHandlers() {
		if (!this.ws) return;

		this.ws.onopen = () => {
			this.state.isConnected = true;
			this.clearReconnectTimer();
			this.reconnectAttempts = 0; // Reset attempts on successful connection
			
			// Authenticate
			this.sendMessage({
				type: 'authenticate',
				room_id: '',
				sender_id: this.currentUserId,
				data: { user_id: this.currentUserId },
				timestamp: Date.now()
			});

			// Request sync if in a room
			if (this.state.currentRoom) {
				this.requestSync();
			}
		};

		this.ws.onmessage = (event) => {
			try {
				const message: CollaborationMessage = JSON.parse(event.data);
				this.handleMessage(message);
			} catch (error) {
				console.error('Failed to parse collaboration message:', error);
			}
		};

		this.ws.onerror = (error) => {
			console.error('WebSocket error:', error);
			this.state.isConnected = false;
		};

		this.ws.onclose = () => {
			this.state.isConnected = false;
			this.stopHeartbeat();
			this.stopAutoSync();
			this.scheduleReconnect();
		};
	}

	private handleMessage(message: CollaborationMessage) {
		// Notify type-specific handlers
		const handlers = this.messageHandlers.get(message.type);
		if (handlers) {
			handlers.forEach(handler => handler(message));
		}

		// Handle built-in message types
		switch (message.type) {
			case 'room_created':
				this.handleRoomCreated(message);
				break;
			case 'participant_joined':
				this.handleParticipantJoined(message);
				break;
			case 'participant_left':
				this.handleParticipantLeft(message);
				break;
			case 'participant_status_changed':
				this.handleParticipantStatusChanged(message);
				break;
			case 'state_update':
				this.handleStateUpdate(message);
				break;
			case 'cursor_move':
				this.handleCursorMove(message);
				break;
			case 'initiative_update':
				this.handleInitiativeUpdate(message);
				break;
			case 'dice_roll':
				this.handleDiceRoll(message);
				break;
			case 'chat_message':
				this.handleChatMessage(message);
				break;
			case 'turn_changed':
				this.handleTurnChanged(message);
				break;
			case 'conflict_detected':
				this.handleConflictDetected(message);
				break;
			case 'sync_response':
				this.handleSyncResponse(message);
				break;
		}
	}

	// Room management
	async createRoom(name: string, campaignId: string, settings?: Partial<CollaborativeRoom['settings']>): Promise<CollaborativeRoom> {
		const room: CollaborativeRoom = {
			id: `room-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
			name,
			campaign_id: campaignId,
			host_id: this.currentUserId,
			participants: [],
			state: {
				initiative_order: [],
				active_turn: 0,
				round_number: 1,
				shared_notes: '',
				dice_rolls: [],
				last_update: new Date().toISOString(),
				version: 1
			},
			settings: {
				max_participants: 10,
				allow_spectators: true,
				require_approval: false,
				enable_voice: false,
				enable_video: false,
				auto_save: true,
				save_interval: 60,
				...settings
			},
			created_at: new Date().toISOString(),
			updated_at: new Date().toISOString()
		};

		this.sendMessage({
			type: 'room_created',
			room_id: room.id,
			sender_id: this.currentUserId,
			data: room,
			timestamp: Date.now()
		});

		this.state.rooms = [...this.state.rooms, room];
		this.state.currentRoom = room;

		return room;
	}

	async joinRoom(roomId: string, inviteCode?: string): Promise<void> {
		this.sendMessage({
			type: 'participant_joined',
			room_id: roomId,
			sender_id: this.currentUserId,
			data: { invite_code: inviteCode },
			timestamp: Date.now()
		});
	}

	async leaveRoom(): Promise<void> {
		if (!this.state.currentRoom) return;

		this.sendMessage({
			type: 'participant_left',
			room_id: this.state.currentRoom.id,
			sender_id: this.currentUserId,
			data: {},
			timestamp: Date.now()
		});

		this.state.currentRoom = null;
		this.state.participants.clear();
		this.state.messages = [];
		this.state.presence.clear();
	}

	// Invitation management
	async createInvitation(
		roomId: string,
		role: ParticipantRole,
		userId?: string,
		expiresIn: number = 3600000 // 1 hour default
	): Promise<RoomInvitation> {
		const invitation: RoomInvitation = {
			id: `inv-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
			room_id: roomId,
			room_name: this.state.currentRoom?.name || '',
			invited_by: this.currentUserId,
			invited_user_id: userId,
			invite_code: !userId ? this.generateInviteCode() : undefined,
			role,
			expires_at: new Date(Date.now() + expiresIn).toISOString(),
			created_at: new Date().toISOString(),
			status: 'pending'
		};

		this.state.invitations = [...this.state.invitations, invitation];
		return invitation;
	}

	private generateInviteCode(): string {
		return Math.random().toString(36).substr(2, 9).toUpperCase();
	}

	// State synchronization
	async updateState(update: StateUpdate): Promise<void> {
		if (!this.state.currentRoom) return;

		// Add to pending updates
		this.state.pendingUpdates = [...this.state.pendingUpdates, update];

		// Send update
		this.sendMessage({
			type: 'state_update',
			room_id: this.state.currentRoom.id,
			sender_id: this.currentUserId,
			data: update,
			timestamp: Date.now(),
			version: update.version
		});
	}

	private requestSync() {
		if (!this.state.currentRoom) return;

		this.state.isSyncing = true;
		this.sendMessage({
			type: 'sync_request',
			room_id: this.state.currentRoom.id,
			sender_id: this.currentUserId,
			data: { last_version: this.state.currentRoom.state.version },
			timestamp: Date.now()
		});
	}

	// Initiative and turn management
	async updateInitiative(initiative: SharedState['initiative_order']): Promise<void> {
		await this.updateState({
			path: ['initiative_order'],
			value: initiative,
			operation: 'set',
			version: (this.state.currentRoom?.state.version || 0) + 1,
			previous_version: this.state.currentRoom?.state.version || 0
		});
	}

	async nextTurn(): Promise<void> {
		if (!this.state.currentRoom || !this.hasPermission('control_initiative', 'initiative')) {
			return;
		}

		// Send turn change request to server to handle atomically
		// This prevents race conditions from multiple clients
		this.sendMessage({
			type: 'next_turn_request',
			room_id: this.state.currentRoom.id,
			sender_id: this.currentUserId,
			data: {
				current_turn: this.state.currentRoom.state.active_turn,
				current_round: this.state.currentRoom.state.round_number,
				initiative_count: this.state.currentRoom.state.initiative_order.length
			},
			timestamp: Date.now()
		});
		
		// Server will broadcast the turn_changed event to all clients
		// including this one, updating the state consistently
	}

	// Dice rolling
	async rollDice(expression: string, purpose?: string): Promise<DiceRoll> {
		if (!this.state.currentRoom) {
			throw new Error('Not in a room');
		}

		// Parse and evaluate dice expression
		const results = this.evaluateDiceExpression(expression);
		const total = results.reduce((sum, val) => sum + val, 0);

		const roll: DiceRoll = {
			id: `roll-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
			player_id: this.currentUserId,
			player_name: this.currentParticipant?.username || 'Unknown',
			expression,
			results,
			total,
			timestamp: new Date().toISOString(),
			purpose
		};

		this.sendMessage({
			type: 'dice_roll',
			room_id: this.state.currentRoom.id,
			sender_id: this.currentUserId,
			data: roll,
			timestamp: Date.now()
		});

		return roll;
	}

	private evaluateDiceExpression(expression: string): number[] {
		// Enhanced dice expression parser supporting multiple dice types and operations
		const results: number[] = [];
		
		try {
			// Clean and normalize the expression
			const normalized = expression.toLowerCase().replace(/\s+/g, '');
			
			// Support multiple dice expressions (e.g., "2d6+1d4+3")
			const dicePattern = /(\d+)d(\d+)/g;
			const modifierPattern = /([+-]\d+)(?!d)/g;
			
			// Roll all dice
			let diceMatch;
			while ((diceMatch = dicePattern.exec(normalized)) !== null) {
				const count = parseInt(diceMatch[1]);
				const sides = parseInt(diceMatch[2]);
				
				// Validate dice parameters
				if (count > 0 && count <= 100 && sides > 0 && sides <= 1000) {
					for (let i = 0; i < count; i++) {
						results.push(Math.floor(Math.random() * sides) + 1);
					}
				}
			}
			
			// Add modifiers
			let modifierMatch;
			while ((modifierMatch = modifierPattern.exec(normalized)) !== null) {
				const modifier = parseInt(modifierMatch[1]);
				if (!isNaN(modifier) && Math.abs(modifier) <= 1000) {
					results.push(modifier);
				}
			}
			
			// Support advantage/disadvantage notation (e.g., "2d20kh1" - keep highest 1)
			if (normalized.includes('kh')) {
				const keepHighest = parseInt(normalized.split('kh')[1]) || 1;
				const sorted = [...results].filter(r => r > 0).sort((a, b) => b - a);
				return sorted.slice(0, keepHighest);
			}
			
			if (normalized.includes('kl')) {
				const keepLowest = parseInt(normalized.split('kl')[1]) || 1;
				const sorted = [...results].filter(r => r > 0).sort((a, b) => a - b);
				return sorted.slice(0, keepLowest);
			}
			
			// If no valid dice were found, return a single d20
			if (results.length === 0) {
				results.push(Math.floor(Math.random() * 20) + 1);
			}
		} catch (error) {
			console.error('Error parsing dice expression:', error);
			// Fallback to simple d20
			results.push(Math.floor(Math.random() * 20) + 1);
		}

		return results;
	}

	// Chat messaging
	async sendChatMessage(content: string, type: ChatMessage['type'] = 'text'): Promise<void> {
		if (!this.state.currentRoom) return;

		const message: ChatMessage = {
			id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
			room_id: this.state.currentRoom.id,
			sender_id: this.currentUserId,
			sender_name: this.currentParticipant?.username || 'Unknown',
			content,
			type,
			timestamp: new Date().toISOString()
		};

		this.sendMessage({
			type: 'chat_message',
			room_id: this.state.currentRoom.id,
			sender_id: this.currentUserId,
			data: message,
			timestamp: Date.now()
		});
	}

	// Cursor tracking
	updateCursor(x: number, y: number, element?: string) {
		if (!this.state.currentRoom) return;

		const cursor: CursorPosition = {
			x,
			y,
			element,
			timestamp: Date.now()
		};

		// Throttle cursor updates
		if (this.shouldThrottleCursor(cursor)) return;

		this.sendMessage({
			type: 'cursor_move',
			room_id: this.state.currentRoom.id,
			sender_id: this.currentUserId,
			data: cursor,
			timestamp: Date.now()
		});
	}

	private lastCursorUpdate = 0;
	private shouldThrottleCursor(cursor: CursorPosition): boolean {
		const now = Date.now();
		if (now - this.lastCursorUpdate < 50) return true; // 20 FPS max
		this.lastCursorUpdate = now;
		return false;
	}

	// Message handlers
	private handleRoomCreated(message: CollaborationMessage) {
		const room = message.data as CollaborativeRoom;
		this.state.rooms = [...this.state.rooms, room];
		if (room.host_id === this.currentUserId) {
			this.state.currentRoom = room;
		}
	}

	private handleParticipantJoined(message: CollaborationMessage) {
		const participant = message.data as Participant;
		this.state.participants.set(participant.user_id, participant);
		this.state.participants = new Map(this.state.participants);
	}

	private handleParticipantLeft(message: CollaborationMessage) {
		const { user_id } = message.data;
		this.state.participants.delete(user_id);
		this.state.presence.delete(user_id);
		this.state.participants = new Map(this.state.participants);
		this.state.presence = new Map(this.state.presence);
	}

	private handleParticipantStatusChanged(message: CollaborationMessage) {
		const { user_id, status } = message.data;
		const participant = this.state.participants.get(user_id);
		if (participant) {
			participant.status = status;
			this.state.participants = new Map(this.state.participants);
		}
	}

	private handleStateUpdate(message: CollaborationMessage) {
		const update = message.data as StateUpdate;
		if (!this.state.currentRoom) return;

		// Check for conflicts
		if (update.previous_version !== this.state.currentRoom.state.version) {
			this.handleConflict(update);
			return;
		}

		// Apply update
		this.applyStateUpdate(update);
	}

	private handleCursorMove(message: CollaborationMessage) {
		const cursor = message.data as CursorPosition;
		this.state.presence.set(message.sender_id, cursor);
		this.state.presence = new Map(this.state.presence);
	}

	private handleInitiativeUpdate(message: CollaborationMessage) {
		if (!this.state.currentRoom) return;
		this.state.currentRoom.state.initiative_order = message.data;
		this.state.currentRoom = { ...this.state.currentRoom };
	}

	private handleDiceRoll(message: CollaborationMessage) {
		const roll = message.data as DiceRoll;
		if (!this.state.currentRoom) return;
		this.state.currentRoom.state.dice_rolls = [
			...this.state.currentRoom.state.dice_rolls.slice(-9), // Keep last 10 rolls
			roll
		];
		this.state.currentRoom = { ...this.state.currentRoom };
	}

	private handleChatMessage(message: CollaborationMessage) {
		const chatMessage = message.data as ChatMessage;
		this.state.messages = [...this.state.messages.slice(-99), chatMessage]; // Keep last 100 messages
	}

	private handleTurnChanged(message: CollaborationMessage) {
		if (!this.state.currentRoom) return;
		const { turn, round } = message.data;
		this.state.currentRoom.state.active_turn = turn;
		this.state.currentRoom.state.round_number = round;
		this.state.currentRoom = { ...this.state.currentRoom };
	}

	private handleConflictDetected(message: CollaborationMessage) {
		const conflict = message.data as ConflictResolution;
		this.state.conflicts = [...this.state.conflicts, conflict];
	}

	private handleSyncResponse(message: CollaborationMessage) {
		if (!this.state.currentRoom) return;
		this.state.currentRoom = message.data.room;
		this.state.participants = new Map(
			Object.entries(message.data.participants || {})
		);
		this.state.isSyncing = false;
		this.state.lastSyncTime = Date.now();
	}

	// Conflict resolution
	private handleConflict(update: StateUpdate) {
		const conflict: ConflictResolution = {
			strategy: 'last_write_wins', // Default strategy
			conflicting_updates: [update],
			requires_user_input: false
		};

		// For now, use last-write-wins
		this.applyStateUpdate(update);
		
		this.state.conflicts = [...this.state.conflicts, conflict];
	}

	private applyStateUpdate(update: StateUpdate) {
		if (!this.state.currentRoom) return;

		// Apply the update based on the path
		let target: any = this.state.currentRoom.state;
		const path = [...update.path];
		const lastKey = path.pop();
		
		for (const key of path) {
			target = target[key];
		}

		if (lastKey) {
			switch (update.operation) {
				case 'set':
					target[lastKey] = update.value;
					break;
				case 'merge':
					target[lastKey] = { ...target[lastKey], ...update.value };
					break;
				case 'delete':
					delete target[lastKey];
					break;
			}
		}

		this.state.currentRoom.state.version = update.version;
		this.state.currentRoom = { ...this.state.currentRoom };

		// Remove from pending updates
		this.state.pendingUpdates = this.state.pendingUpdates.filter(
			u => u !== update
		);
	}

	// Reconnection logic with exponential backoff
	private scheduleReconnect() {
		this.clearReconnectTimer();
		
		if (this.reconnectAttempts >= this.maxReconnectAttempts) {
			console.error('Maximum reconnection attempts reached');
			return;
		}
		
		// Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 512s (max)
		const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 512000);
		this.reconnectAttempts++;
		
		console.log(`Reconnecting in ${delay / 1000}s (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
		
		this.reconnectTimer = window.setTimeout(() => {
			if (!this.state.isConnected) {
				this.connect(this.currentUserId);
			}
		}, delay);
	}

	private clearReconnectTimer() {
		if (this.reconnectTimer) {
			clearTimeout(this.reconnectTimer);
			this.reconnectTimer = null;
		}
	}

	// Heartbeat
	private startHeartbeat() {
		this.heartbeatInterval = window.setInterval(() => {
			if (this.ws?.readyState === WebSocket.OPEN) {
				this.sendMessage({
					type: 'heartbeat',
					room_id: this.state.currentRoom?.id || '',
					sender_id: this.currentUserId,
					data: {},
					timestamp: Date.now()
				});
			}
		}, 30000); // Every 30 seconds
	}

	private stopHeartbeat() {
		if (this.heartbeatInterval) {
			clearInterval(this.heartbeatInterval);
			this.heartbeatInterval = null;
		}
	}

	// Auto-sync
	private startAutoSync() {
		if (!this.state.currentRoom?.settings.auto_save) return;

		const interval = (this.state.currentRoom.settings.save_interval || 60) * 1000;
		this.syncInterval = window.setInterval(() => {
			if (this.state.currentRoom && this.state.pendingUpdates.length > 0) {
				this.requestSync();
			}
		}, interval);
	}

	private stopAutoSync() {
		if (this.syncInterval) {
			clearInterval(this.syncInterval);
			this.syncInterval = null;
		}
	}

	// Utility methods
	private sendMessage(message: CollaborationMessage) {
		if (this.ws?.readyState === WebSocket.OPEN) {
			this.ws.send(JSON.stringify(message));
		} else {
			console.warn('WebSocket not connected, queuing message');
			// Could implement a message queue here
		}
	}

	onMessage(type: string, handler: (msg: CollaborationMessage) => void) {
		if (!this.messageHandlers.has(type)) {
			this.messageHandlers.set(type, new Set());
		}
		this.messageHandlers.get(type)!.add(handler);
		
		return () => {
			this.messageHandlers.get(type)?.delete(handler);
		};
	}

	// Cleanup
	disconnect() {
		this.stopHeartbeat();
		this.stopAutoSync();
		this.clearReconnectTimer();
		
		if (this.ws) {
			this.ws.close();
			this.ws = null;
		}
		
		this.state = {
			rooms: [],
			currentRoom: null,
			invitations: [],
			participants: new Map(),
			messages: [],
			presence: new Map(),
			pendingUpdates: [],
			conflicts: [],
			isConnected: false,
			isSyncing: false,
			lastSyncTime: Date.now()
		};
		
		this.messageHandlers.clear();
	}
}

// Export singleton instance
export const collaborationStore = new CollaborationStore();