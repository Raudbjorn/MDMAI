/**
 * Provider Management Types
 * Type definitions for AI provider configuration, credentials, and analytics
 */

export enum ProviderType {
	ANTHROPIC = 'anthropic',
	OPENAI = 'openai',
	GOOGLE = 'google',
	OLLAMA = 'ollama'
}

export enum ProviderStatus {
	AVAILABLE = 'available',
	UNAVAILABLE = 'unavailable',
	RATE_LIMITED = 'rate_limited',
	QUOTA_EXCEEDED = 'quota_exceeded',
	ERROR = 'error',
	MAINTENANCE = 'maintenance'
}

export enum ProviderCapability {
	TEXT_GENERATION = 'text_generation',
	TOOL_CALLING = 'tool_calling',
	VISION = 'vision',
	STREAMING = 'streaming',
	BATCH_PROCESSING = 'batch_processing',
	FINE_TUNING = 'fine_tuning'
}

export enum CostTier {
	FREE = 'free',
	LOW = 'low',
	MEDIUM = 'medium',
	HIGH = 'high',
	PREMIUM = 'premium'
}

export interface ProviderConfig {
	provider_type: ProviderType;
	api_key: string;
	base_url?: string;
	timeout: number;
	max_retries: number;
	retry_delay: number;
	rate_limit_rpm: number;
	rate_limit_tpm: number;
	budget_limit?: number;
	enabled: boolean;
	priority: number;
	metadata?: Record<string, any>;
}

export interface ModelSpec {
	model_id: string;
	provider_type: ProviderType;
	display_name: string;
	capabilities: ProviderCapability[];
	context_length: number;
	max_output_tokens: number;
	cost_per_input_token: number;
	cost_per_output_token: number;
	cost_tier: CostTier;
	supports_streaming: boolean;
	supports_tools: boolean;
	supports_vision: boolean;
	is_available: boolean;
	metadata?: Record<string, any>;
}

export interface ProviderHealth {
	provider_type: ProviderType;
	status: ProviderStatus;
	last_success?: Date;
	last_error?: Date;
	error_count: number;
	success_count: number;
	avg_latency_ms: number;
	rate_limit_remaining?: number;
	rate_limit_reset?: Date;
	quota_remaining?: number;
	uptime_percentage: number;
	updated_at: Date;
}

export interface UsageRecord {
	request_id: string;
	provider_type: ProviderType;
	session_id?: string;
	model: string;
	input_tokens: number;
	output_tokens: number;
	cost: number;
	latency_ms: number;
	timestamp: Date;
	success: boolean;
	error_message?: string;
	metadata?: Record<string, any>;
}

export interface CostBudget {
	budget_id: string;
	name: string;
	daily_limit?: number;
	monthly_limit?: number;
	provider_limits: Partial<Record<ProviderType, number>>;
	alert_thresholds: number[];
	enabled: boolean;
	created_at: Date;
}

export interface AIProviderStats {
	provider_type: ProviderType;
	total_requests: number;
	successful_requests: number;
	failed_requests: number;
	total_input_tokens: number;
	total_output_tokens: number;
	total_cost: number;
	avg_latency_ms: number;
	uptime_percentage: number;
	last_request?: Date;
	daily_usage: Record<string, number>;
	monthly_usage: Record<string, number>;
}

export interface ProviderSelection {
	required_capabilities: ProviderCapability[];
	preferred_providers: ProviderType[];
	exclude_providers: ProviderType[];
	max_cost_per_request?: number;
	max_latency_ms?: number;
	require_streaming: boolean;
	require_tools: boolean;
	cost_optimization: boolean;
}

// API Request/Response types
export interface APIRequest<T = any> {
	data: T;
	timestamp?: Date;
}

export interface APIResponse<T = any> {
	success: boolean;
	message?: string;
	data?: T;
	error?: string;
}

export interface ProviderConfigRequest extends APIRequest<{
	configs: ProviderConfig[];
	budgets?: CostBudget[];
}> {}

export interface ProviderStatsRequest extends APIRequest<{
	provider_type?: ProviderType;
	start_date?: string;
	end_date?: string;
}> {}

export interface ProviderStatsResponse extends APIResponse<{
	stats: AIProviderStats[];
	total_cost: number;
	period: { start: string; end: string };
}> {}

export interface ProviderHealthResponse extends APIResponse<{
	health: ProviderHealth[];
	overall_status: ProviderStatus;
}> {}

// Credential management types
export interface ProviderCredentials {
	provider_type: ProviderType;
	api_key: string;
	encrypted: boolean;
	last_updated: Date;
	valid_until?: Date;
}

export interface CredentialValidation {
	provider_type: ProviderType;
	is_valid: boolean;
	error_message?: string;
	tested_at: Date;
}

// Simplified base types for common patterns
export interface BaseEntity {
	id: string;
	created_at: Date;
	updated_at: Date;
}

export interface TimestampedEntity {
	timestamp: Date;
}

export interface StatusEntity {
	status: ProviderStatus;
	last_updated: Date;
}

// Result type for error handling
export type Result<T, E = Error> = 
	| { ok: true; value: T }
	| { ok: false; error: E };

// Common utility types
export type Partial<T> = { [P in keyof T]?: T[P] };
export type Required<T> = { [P in keyof T]-?: T[P] };
export type Pick<T, K extends keyof T> = { [P in K]: T[P] };
export type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

// Provider-specific utility types
export type ProviderConfigUpdate = Partial<Omit<ProviderConfig, 'provider_type'>>;
export type ModelSpecSummary = Pick<ModelSpec, 'model_id' | 'display_name' | 'is_available'>;
export type UsageSummary = Pick<UsageRecord, 'provider_type' | 'cost' | 'timestamp' | 'success'>;