/**
 * Provider Management API Client
 * Handles all provider-related API calls with proper error handling
 */

import type { 
	ProviderConfig, 
	CostBudget, 
	ProviderHealth,
	AIProviderStats,
	ProviderConfigRequest,
	ProviderConfigResponse,
	ProviderStatsRequest,
	ProviderStatsResponse,
	ProviderHealthResponse,
	ProviderCredentials,
	CredentialValidation,
	ProviderType,
	Result
} from '$lib/types/providers';

const API_BASE = '/api';

/**
 * Handles API errors and returns Result type
 */
async function handleResponse<T>(response: Response): Promise<Result<T>> {
	if (!response.ok) {
		const error = await response.text().catch(() => 'Unknown error');
		return {
			ok: false,
			error: new Error(`${response.status}: ${error}`)
		};
	}

	try {
		const data = await response.json();
		return { ok: true, value: data };
	} catch (error) {
		return {
			ok: false,
			error: error instanceof Error ? error : new Error('Failed to parse response')
		};
	}
}

/**
 * Handles API responses that return void
 */
async function handleVoidResponse(response: Response): Promise<Result<void>> {
	if (!response.ok) {
		const error = await response.text().catch(() => 'Unknown error');
		return {
			ok: false,
			error: new Error(`${response.status}: ${error}`)
		};
	}

	return { ok: true, value: undefined };
}

/**
 * Provider API client
 */
export class ProviderApiClient {
	private baseUrl: string;
	private headers: HeadersInit;

	constructor(baseUrl: string = API_BASE) {
		this.baseUrl = baseUrl;
		this.headers = {
			'Content-Type': 'application/json',
		};
	}

	/**
	 * Set authorization header
	 */
	setAuthToken(token: string) {
		this.headers = {
			...this.headers,
			'Authorization': `Bearer ${token}`
		};
	}

	/**
	 * Configure providers
	 */
	async configureProviders(
		configs: ProviderConfig[], 
		budgets?: CostBudget[]
	): Promise<Result<ProviderConfigResponse>> {
		const request: ProviderConfigRequest = { configs, budgets };
		
		const response = await fetch(`${this.baseUrl}/providers/configure`, {
			method: 'POST',
			headers: this.headers,
			body: JSON.stringify(request)
		});

		return handleResponse<ProviderConfigResponse>(response);
	}

	/**
	 * Get provider configurations
	 */
	async getProviderConfigs(): Promise<Result<ProviderConfig[]>> {
		const response = await fetch(`${this.baseUrl}/providers/configs`, {
			headers: this.headers
		});

		return handleResponse<ProviderConfig[]>(response);
	}

	/**
	 * Update provider configuration
	 */
	async updateProviderConfig(
		providerType: ProviderType,
		config: Partial<ProviderConfig>
	): Promise<Result<ProviderConfig>> {
		const response = await fetch(`${this.baseUrl}/providers/${providerType}/config`, {
			method: 'PATCH',
			headers: this.headers,
			body: JSON.stringify(config)
		});

		return handleResponse<ProviderConfig>(response);
	}

	/**
	 * Delete provider configuration
	 */
	async deleteProviderConfig(providerType: ProviderType): Promise<Result<void>> {
		const response = await fetch(`${this.baseUrl}/providers/${providerType}`, {
			method: 'DELETE',
			headers: this.headers
		});

		return handleVoidResponse(response);
	}

	/**
	 * Store encrypted credentials
	 */
	async storeCredentials(credentials: ProviderCredentials): Promise<Result<void>> {
		const response = await fetch(`${this.baseUrl}/providers/credentials`, {
			method: 'POST',
			headers: this.headers,
			body: JSON.stringify(credentials)
		});

		return handleVoidResponse(response);
	}

	/**
	 * Validate provider credentials
	 */
	async validateCredentials(
		providerType: ProviderType,
		apiKey: string
	): Promise<Result<CredentialValidation>> {
		const response = await fetch(`${this.baseUrl}/providers/credentials/validate`, {
			method: 'POST',
			headers: this.headers,
			body: JSON.stringify({ provider_type: providerType, api_key: apiKey })
		});

		return handleResponse<CredentialValidation>(response);
	}

	/**
	 * Get provider health status
	 */
	async getProviderHealth(): Promise<Result<ProviderHealthResponse>> {
		const response = await fetch(`${this.baseUrl}/providers/health`, {
			headers: this.headers
		});

		return handleResponse<ProviderHealthResponse>(response);
	}

	/**
	 * Get provider usage statistics
	 */
	async getProviderStats(request?: ProviderStatsRequest): Promise<Result<ProviderStatsResponse>> {
		const params = new URLSearchParams();
		if (request?.provider_type) params.append('provider_type', request.provider_type);
		if (request?.start_date) params.append('start_date', request.start_date);
		if (request?.end_date) params.append('end_date', request.end_date);

		const response = await fetch(`${this.baseUrl}/providers/stats?${params}`, {
			headers: this.headers
		});

		return handleResponse<ProviderStatsResponse>(response);
	}

	/**
	 * Get cost budgets
	 */
	async getCostBudgets(): Promise<Result<CostBudget[]>> {
		const response = await fetch(`${this.baseUrl}/providers/budgets`, {
			headers: this.headers
		});

		return handleResponse<CostBudget[]>(response);
	}

	/**
	 * Create or update cost budget
	 */
	async upsertCostBudget(budget: CostBudget): Promise<Result<CostBudget>> {
		const response = await fetch(`${this.baseUrl}/providers/budgets`, {
			method: 'PUT',
			headers: this.headers,
			body: JSON.stringify(budget)
		});

		return handleResponse<CostBudget>(response);
	}

	/**
	 * Delete cost budget
	 */
	async deleteCostBudget(budgetId: string): Promise<Result<void>> {
		const response = await fetch(`${this.baseUrl}/providers/budgets/${budgetId}`, {
			method: 'DELETE',
			headers: this.headers
		});

		return handleVoidResponse(response);
	}

	/**
	 * Switch active provider
	 */
	async switchProvider(providerType: ProviderType): Promise<Result<void>> {
		const response = await fetch(`${this.baseUrl}/providers/active`, {
			method: 'POST',
			headers: this.headers,
			body: JSON.stringify({ provider_type: providerType })
		});

		return handleVoidResponse(response);
	}

	/**
	 * Test provider connection
	 */
	async testProvider(providerType: ProviderType): Promise<Result<boolean>> {
		const response = await fetch(`${this.baseUrl}/providers/${providerType}/test`, {
			method: 'POST',
			headers: this.headers
		});

		return handleResponse<boolean>(response);
	}

	/**
	 * Get available models for a provider
	 */
	async getProviderModels(providerType: ProviderType): Promise<Result<any[]>> {
		const response = await fetch(`${this.baseUrl}/providers/${providerType}/models`, {
			headers: this.headers
		});

		return handleResponse<any[]>(response);
	}
}

// Export singleton instance
export const providerApi = new ProviderApiClient();