// Collaborative session types

export interface CollaborativeRoom {
	id: string;
	name: string;
	campaign_id: string;
	host_id: string;
	participants: Participant[];
	state: SharedState;
	settings: RoomSettings;
	created_at: string;
	updated_at: string;
}

export interface Participant {
	id: string;
	user_id: string;
	username: string;
	role: ParticipantRole;
	permissions: Permission[];
	status: 'online' | 'away' | 'offline';
	cursor?: CursorPosition;
	color: string;
	joined_at: string;
	last_activity: string;
}

export type ParticipantRole = 'host' | 'gm' | 'player' | 'spectator';

export interface Permission {
	action: PermissionAction;
	resource: PermissionResource;
	granted: boolean;
}

export type PermissionAction = 
	| 'read' 
	| 'write' 
	| 'delete' 
	| 'manage_participants'
	| 'manage_session'
	| 'control_initiative'
	| 'edit_characters'
	| 'edit_monsters';

export type PermissionResource = 
	| 'session'
	| 'characters'
	| 'monsters'
	| 'notes'
	| 'initiative'
	| 'dice'
	| 'chat';

export interface RoomSettings {
	max_participants: number;
	allow_spectators: boolean;
	require_approval: boolean;
	enable_voice: boolean;
	enable_video: boolean;
	auto_save: boolean;
	save_interval: number; // in seconds
}

export interface SharedState {
	initiative_order: InitiativeEntry[];
	active_turn: number;
	round_number: number;
	shared_notes: string;
	dice_rolls: DiceRoll[];
	last_update: string;
	version: number; // for conflict resolution
	map?: MapState; // Optional collaborative map state
	[key: string]: any; // Allow for dynamic properties
}

export interface MapState {
	tokens: Record<string, MapToken>;
	drawings: any[];
	fogOfWar?: any;
	background?: string;
	gridSize?: number;
	[key: string]: any;
}

export interface MapToken {
	id: string;
	x: number;
	y: number;
	owner_id?: string;
	locked?: boolean;
	[key: string]: any;
}

export interface InitiativeEntry {
	id: string;
	character_id?: string;
	name: string;
	initiative: number;
	is_player: boolean;
	current_hp?: number;
	max_hp?: number;
	conditions: string[];
	has_acted: boolean;
}

export interface DiceRoll {
	id: string;
	player_id: string;
	player_name: string;
	expression: string; // e.g., "2d6+3"
	results: number[];
	total: number;
	timestamp: string;
	purpose?: string; // e.g., "Attack roll", "Damage"
}

export interface CursorPosition {
	x: number;
	y: number;
	element?: string; // ID of element being interacted with
	timestamp: number;
}

export interface RoomInvitation {
	id: string;
	room_id: string;
	room_name: string;
	invited_by: string;
	invited_user_id?: string;
	invite_code?: string;
	role: ParticipantRole;
	expires_at: string;
	created_at: string;
	status: 'pending' | 'accepted' | 'declined' | 'expired';
}

// WebSocket message types for collaboration
export interface CollaborationMessage {
	type: CollaborationMessageType;
	room_id: string;
	sender_id: string;
	data: any;
	timestamp: number;
	version?: number;
}

export type CollaborationMessageType =
	| 'room_created'
	| 'participant_joined'
	| 'participant_left'
	| 'participant_status_changed'
	| 'state_update'
	| 'cursor_move'
	| 'initiative_update'
	| 'dice_roll'
	| 'chat_message'
	| 'permission_changed'
	| 'turn_changed'
	| 'next_turn_request'
	| 'conflict_detected'
	| 'sync_request'
	| 'sync_response'
	| 'authenticate'
	| 'heartbeat'
	| 'typing_indicator'
	| 'selection_change';

export interface StateUpdate {
	path: string[]; // Path to the updated property
	value: any;
	operation: 'set' | 'merge' | 'delete';
	version: number;
	previous_version: number;
}

export interface ConflictResolution {
	strategy: 'last_write_wins' | 'merge' | 'manual';
	conflicting_updates: StateUpdate[];
	resolved_value?: any;
	requires_user_input: boolean;
}

export interface ChatMessage {
	id: string;
	room_id: string;
	sender_id: string;
	sender_name: string;
	content: string;
	type: 'text' | 'roll' | 'system' | 'whisper';
	timestamp: string;
	edited?: boolean;
	edited_at?: string;
}

// Turn management
export interface TurnState {
	current_turn_index: number;
	round_number: number;
	phase: 'planning' | 'action' | 'reaction' | 'end';
	timer?: TurnTimer;
	history: TurnHistoryEntry[];
}

export interface TurnTimer {
	duration: number; // seconds
	remaining: number;
	paused: boolean;
	started_at: string;
}

export interface TurnHistoryEntry {
	round: number;
	turn: number;
	character_id: string;
	character_name: string;
	actions: string[];
	timestamp: string;
}