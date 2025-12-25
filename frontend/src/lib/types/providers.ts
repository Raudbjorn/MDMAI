/**
 * Provider Management Types
 * Type definitions for AI provider configuration, credentials, and analytics
 */

// Use const assertions for better type inference and immutability
export const ProviderType = {
	ANTHROPIC: 'anthropic',
	OPENAI: 'openai',
	GOOGLE: 'google',
	ELEVENLABS: 'elevenlabs',
	FISH_AUDIO: 'fish_audio',
	OLLAMA_TTS: 'ollama_tts'
} as const;

export type ProviderType = typeof ProviderType[keyof typeof ProviderType];

export const ProviderStatus = {
	AVAILABLE: 'available',
	UNAVAILABLE: 'unavailable',
	RATE_LIMITED: 'rate_limited',
	QUOTA_EXCEEDED: 'quota_exceeded',
	ERROR: 'error',
	MAINTENANCE: 'maintenance'
} as const;

export type ProviderStatus = typeof ProviderStatus[keyof typeof ProviderStatus];

export const ProviderCapability = {
	TEXT_GENERATION: 'text_generation',
	TOOL_CALLING: 'tool_calling',
	VISION: 'vision',
	STREAMING: 'streaming',
	BATCH_PROCESSING: 'batch_processing',
	FINE_TUNING: 'fine_tuning'
} as const;

export type ProviderCapability = typeof ProviderCapability[keyof typeof ProviderCapability];

export const CostTier = {
	FREE: 'free',
	LOW: 'low',
	MEDIUM: 'medium',
	HIGH: 'high',
	PREMIUM: 'premium'
} as const;

export type CostTier = typeof CostTier[keyof typeof CostTier];

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

	// Voice specific
	voice_id?: string;
	// base_url already exists in interface
	cost_limit_usd?: number;
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

// API Request/Response types with better error-as-values pattern
export interface APIRequest<T = unknown> {
	data: T;
	timestamp?: Date;
}

// Use discriminated unions for better type safety
export type APIResponse<T = unknown> =
	| { ok: true; data: T; timestamp?: Date }
	| { ok: false; error: string; message?: string; timestamp?: Date };

// More type-safe request interfaces
export interface ProviderConfigRequest extends APIRequest<{
	configs: readonly ProviderConfig[];
	budgets?: readonly CostBudget[];
}> {}

export interface ProviderStatsRequest {
	provider_type?: ProviderType;
	start_date?: string;
	end_date?: string;
}

// Response types using the Result pattern
export type ProviderStatsResponse = APIResponse<{
	stats: readonly AIProviderStats[];
	total_cost: number;
	period: { readonly start: string; readonly end: string };
}>;

export type ProviderHealthResponse = APIResponse<{
	health: readonly ProviderHealth[];
	overall_status: ProviderStatus;
}>;

export type ProviderConfigResponse = APIResponse<{
	configs: readonly ProviderConfig[];
	success_count: number;
	failed_count: number;
}>;


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

// Enhanced Result type for comprehensive error handling
export type Result<T, E = string> =
	| { ok: true; value: T }
	| { ok: false; error: E; context?: Record<string, unknown> };

// Provider-specific utility types with better type constraints
export type ProviderConfigUpdate = Partial<Omit<ProviderConfig, 'provider_type'>>;
export type ModelSpecSummary = Pick<ModelSpec, 'model_id' | 'display_name' | 'is_available'>;
export type UsageSummary = Pick<UsageRecord, 'provider_type' | 'cost' | 'timestamp' | 'success'>;

// Utility types for type-safe operations
export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;
export type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// Type guards for runtime type checking
export const isProviderType = (value: string): value is ProviderType =>
	Object.values(ProviderType).includes(value as ProviderType);

export const isProviderStatus = (value: string): value is ProviderStatus =>
	Object.values(ProviderStatus).includes(value as ProviderStatus);

export const isSuccessResponse = <T>(response: APIResponse<T>): response is { ok: true; data: T; timestamp?: Date } =>
	response.ok === true;

// Helper type for debounced functions (addresses Issue 6)
export type DebouncedFunction<T extends (...args: any[]) => any> = {
	(...args: Parameters<T>): void;
	cancel: () => void;
	flush: () => void;
};
