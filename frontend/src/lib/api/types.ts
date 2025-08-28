// Core types for TTRPG Assistant

export interface Tool {
	name: string;
	description: string;
	parameters: Record<string, any>;
}

export interface ToolResult {
	success: boolean;
	data?: any;
	error?: string;
}

export interface Campaign {
	id: string;
	name: string;
	system: string;
	description: string;
	created_at: string;
	updated_at: string;
	gm_id: string;
	players: string[];
	active: boolean;
}

export interface Character {
	id: string;
	campaign_id: string;
	name: string;
	player_id?: string;
	is_npc: boolean;
	stats: Record<string, any>;
	backstory?: string;
	notes?: string;
	created_at: string;
	updated_at: string;
}

export interface Session {
	id: string;
	campaign_id: string;
	name: string;
	date: string;
	notes: string;
	initiative_order: InitiativeEntry[];
	monsters: Monster[];
	active: boolean;
	created_at: string;
	updated_at: string;
}

export interface InitiativeEntry {
	id: string;
	name: string;
	initiative: number;
	is_player: boolean;
	current_hp?: number;
	max_hp?: number;
}

export interface Monster {
	id: string;
	name: string;
	type: string;
	current_hp: number;
	max_hp: number;
	ac: number;
	stats: Record<string, any>;
}

export interface SearchResult {
	id: string;
	content: string;
	source: string;
	page?: number;
	relevance: number;
	metadata: Record<string, any>;
}

export interface AIProvider {
	name: 'anthropic' | 'openai' | 'google';
	display_name: string;
	enabled: boolean;
	api_key?: string;
}

export interface User {
	id: string;
	username: string;
	email?: string;
	role: 'player' | 'gm' | 'admin';
	campaigns: string[];
	created_at: string;
}