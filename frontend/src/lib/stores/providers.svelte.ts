/**
 * Provider Management Store
 * Manages provider configurations, credentials, and usage analytics using Svelte 5 runes
 */

import { providerApi } from '$lib/api/providers-client';
import type { 
	ProviderConfig, 
	CostBudget, 
	ProviderHealth, 
	AIProviderStats,
	ProviderType,
	ProviderStatus,
	Result,
	ProviderCredentials
} from '$lib/types/providers';

interface ProviderState {
	configs: ProviderConfig[];
	budgets: CostBudget[];
	health: ProviderHealth[];
	stats: AIProviderStats[];
	activeProvider: ProviderType | null;
	loading: boolean;
	error: string | null;
	initialized: boolean;
}

class ProviderStore {
	// State using Svelte 5 runes
	private state = $state<ProviderState>({
		configs: [],
		budgets: [],
		health: [],
		stats: [],
		activeProvider: null,
		loading: false,
		error: null,
		initialized: false
	});

	// Computed properties using $derived
	configs = $derived(this.state.configs);
	budgets = $derived(this.state.budgets);
	health = $derived(this.state.health);
	stats = $derived(this.state.stats);
	activeProvider = $derived(this.state.activeProvider);
	loading = $derived(this.state.loading);
	error = $derived(this.state.error);
	initialized = $derived(this.state.initialized);

	// Derived computations
	enabledProviders = $derived(
		this.state.configs.filter(config => config.enabled)
	);

	totalCost = $derived(
		this.state.stats.reduce((sum, stat) => sum + stat.total_cost, 0)
	);

	overallHealth = $derived<ProviderStatus>(() => {
		if (this.state.health.length === 0) return ProviderStatus.UNAVAILABLE;
		
		const statuses = this.state.health.map(h => h.status);
		if (statuses.some(s => s === ProviderStatus.ERROR)) return ProviderStatus.ERROR;
		if (statuses.some(s => s === ProviderStatus.RATE_LIMITED)) return ProviderStatus.RATE_LIMITED;
		if (statuses.some(s => s === ProviderStatus.QUOTA_EXCEEDED)) return ProviderStatus.QUOTA_EXCEEDED;
		if (statuses.every(s => s === ProviderStatus.AVAILABLE)) return ProviderStatus.AVAILABLE;
		return ProviderStatus.MAINTENANCE;
	});

	providersByPriority = $derived(
		[...this.state.configs]
			.filter(c => c.enabled)
			.sort((a, b) => b.priority - a.priority)
	);

	constructor() {
		// Initialize on creation
		$effect(() => {
			if (!this.state.initialized) {
				this.initialize();
			}
		});

		// Auto-refresh health status every 30 seconds when active
		$effect(() => {
			if (this.state.initialized) {
				const interval = setInterval(() => {
					this.refreshHealth();
				}, 30000);

				return () => clearInterval(interval);
			}
		});
	}

	/**
	 * Initialize the provider store
	 */
	async initialize() {
		this.state.loading = true;
		this.state.error = null;

		try {
			// Load configurations
			const configsResult = await providerApi.getProviderConfigs();
			if (!configsResult.ok) throw configsResult.error;
			this.state.configs = configsResult.value;

			// Load budgets
			const budgetsResult = await providerApi.getCostBudgets();
			if (!budgetsResult.ok) throw budgetsResult.error;
			this.state.budgets = budgetsResult.value;

			// Load health status
			const healthResult = await providerApi.getProviderHealth();
			if (!healthResult.ok) throw healthResult.error;
			this.state.health = healthResult.value.health;

			// Load usage stats
			const statsResult = await providerApi.getProviderStats();
			if (!statsResult.ok) throw statsResult.error;
			this.state.stats = statsResult.value.stats;

			// Set active provider from configs
			const activeConfig = this.state.configs.find(c => c.enabled && c.priority === Math.max(...this.state.configs.map(cfg => cfg.priority)));
			this.state.activeProvider = activeConfig?.provider_type || null;

			this.state.initialized = true;
		} catch (error) {
			this.state.error = error instanceof Error ? error.message : 'Failed to initialize providers';
			console.error('Provider initialization error:', error);
		} finally {
			this.state.loading = false;
		}
	}

	/**
	 * Configure a provider
	 */
	async configureProvider(config: ProviderConfig, credentials?: ProviderCredentials) {
		this.state.loading = true;
		this.state.error = null;

		try {
			// Validate credentials if provided
			if (credentials) {
				const validationResult = await providerApi.validateCredentials(
					credentials.provider_type,
					credentials.api_key
				);
				if (!validationResult.ok) throw validationResult.error;
				if (!validationResult.value.is_valid) {
					throw new Error(validationResult.value.error_message || 'Invalid credentials');
				}

				// Store encrypted credentials
				const storeResult = await providerApi.storeCredentials(credentials);
				if (!storeResult.ok) throw storeResult.error;
			}

			// Update provider configuration
			const configResult = await providerApi.updateProviderConfig(
				config.provider_type,
				config
			);
			if (!configResult.ok) throw configResult.error;

			// Update local state
			const index = this.state.configs.findIndex(c => c.provider_type === config.provider_type);
			if (index >= 0) {
				this.state.configs[index] = configResult.value;
			} else {
				this.state.configs.push(configResult.value);
			}

			// Refresh health status
			await this.refreshHealth();
		} catch (error) {
			this.state.error = error instanceof Error ? error.message : 'Failed to configure provider';
			throw error;
		} finally {
			this.state.loading = false;
		}
	}

	/**
	 * Remove a provider configuration
	 */
	async removeProvider(providerType: ProviderType) {
		this.state.loading = true;
		this.state.error = null;

		try {
			const result = await providerApi.deleteProviderConfig(providerType);
			if (!result.ok) throw result.error;

			// Update local state
			this.state.configs = this.state.configs.filter(c => c.provider_type !== providerType);
			this.state.health = this.state.health.filter(h => h.provider_type !== providerType);
			this.state.stats = this.state.stats.filter(s => s.provider_type !== providerType);

			// Update active provider if needed
			if (this.state.activeProvider === providerType) {
				const nextProvider = this.providersByPriority[0];
				this.state.activeProvider = nextProvider?.provider_type || null;
			}
		} catch (error) {
			this.state.error = error instanceof Error ? error.message : 'Failed to remove provider';
			throw error;
		} finally {
			this.state.loading = false;
		}
	}

	/**
	 * Switch active provider
	 */
	async switchProvider(providerType: ProviderType) {
		this.state.loading = true;
		this.state.error = null;

		try {
			const result = await providerApi.switchProvider(providerType);
			if (!result.ok) throw result.error;

			this.state.activeProvider = providerType;

			// Update priorities
			this.state.configs = this.state.configs.map(config => ({
				...config,
				priority: config.provider_type === providerType ? 10 : config.priority
			}));
		} catch (error) {
			this.state.error = error instanceof Error ? error.message : 'Failed to switch provider';
			throw error;
		} finally {
			this.state.loading = false;
		}
	}

	/**
	 * Configure cost budget
	 */
	async configureBudget(budget: CostBudget) {
		this.state.loading = true;
		this.state.error = null;

		try {
			const result = await providerApi.upsertCostBudget(budget);
			if (!result.ok) throw result.error;

			// Update local state
			const index = this.state.budgets.findIndex(b => b.budget_id === budget.budget_id);
			if (index >= 0) {
				this.state.budgets[index] = result.value;
			} else {
				this.state.budgets.push(result.value);
			}
		} catch (error) {
			this.state.error = error instanceof Error ? error.message : 'Failed to configure budget';
			throw error;
		} finally {
			this.state.loading = false;
		}
	}

	/**
	 * Remove cost budget
	 */
	async removeBudget(budgetId: string) {
		this.state.loading = true;
		this.state.error = null;

		try {
			const result = await providerApi.deleteCostBudget(budgetId);
			if (!result.ok) throw result.error;

			// Update local state
			this.state.budgets = this.state.budgets.filter(b => b.budget_id !== budgetId);
		} catch (error) {
			this.state.error = error instanceof Error ? error.message : 'Failed to remove budget';
			throw error;
		} finally {
			this.state.loading = false;
		}
	}

	/**
	 * Refresh health status
	 */
	async refreshHealth() {
		try {
			const result = await providerApi.getProviderHealth();
			if (result.ok) {
				this.state.health = result.value.health;
			}
		} catch (error) {
			console.error('Failed to refresh health:', error);
		}
	}

	/**
	 * Refresh usage statistics
	 */
	async refreshStats(startDate?: string, endDate?: string) {
		this.state.loading = true;
		
		try {
			const result = await providerApi.getProviderStats({ 
				start_date: startDate, 
				end_date: endDate 
			});
			if (!result.ok) throw result.error;

			this.state.stats = result.value.stats;
		} catch (error) {
			this.state.error = error instanceof Error ? error.message : 'Failed to refresh statistics';
			throw error;
		} finally {
			this.state.loading = false;
		}
	}

	/**
	 * Test provider connection
	 */
	async testProvider(providerType: ProviderType): Promise<boolean> {
		try {
			const result = await providerApi.testProvider(providerType);
			if (!result.ok) throw result.error;
			return result.value;
		} catch (error) {
			console.error('Provider test failed:', error);
			return false;
		}
	}

	/**
	 * Get provider by type
	 */
	getProvider(providerType: ProviderType): ProviderConfig | undefined {
		return this.state.configs.find(c => c.provider_type === providerType);
	}

	/**
	 * Get provider health
	 */
	getProviderHealth(providerType: ProviderType): ProviderHealth | undefined {
		return this.state.health.find(h => h.provider_type === providerType);
	}

	/**
	 * Get provider stats
	 */
	getProviderStats(providerType: ProviderType): AIProviderStats | undefined {
		return this.state.stats.find(s => s.provider_type === providerType);
	}

	/**
	 * Clear error state
	 */
	clearError() {
		this.state.error = null;
	}
}

// Export singleton instance
export const providerStore = new ProviderStore();