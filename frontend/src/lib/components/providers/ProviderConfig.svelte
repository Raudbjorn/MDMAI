<script lang="ts">
	import { providerStore } from '$lib/stores/providers.svelte';
	import { ProviderType, ProviderStatus } from '$lib/types/providers';
	import type { ProviderConfig, ProviderHealth } from '$lib/types/providers';
	import CredentialManager from './CredentialManager.svelte';
	
	interface Props {
		providerType: ProviderType;
	}

	let { providerType }: Props = $props();
	
	// Get provider data from store
	let config = $derived(providerStore.getProvider(providerType));
	let health = $derived(providerStore.getProviderHealth(providerType));
	let stats = $derived(providerStore.getProviderStats(providerType));
	
	// Local state for editing
	let isEditing = $state(false);
	let editConfig = $state<Partial<ProviderConfig>>({});
	let showAdvanced = $state(false);
	let isSaving = $state(false);
	
	// Initialize edit config when entering edit mode
	$effect(() => {
		if (isEditing && config) {
			editConfig = { ...config };
		}
	});
	
	// Status colors and icons
	const statusConfig: Record<ProviderStatus, { color: string; icon: string; text: string }> = {
		[ProviderStatus.AVAILABLE]: { 
			color: 'green', 
			icon: '✓', 
			text: 'Available' 
		},
		[ProviderStatus.UNAVAILABLE]: { 
			color: 'gray', 
			icon: '○', 
			text: 'Unavailable' 
		},
		[ProviderStatus.RATE_LIMITED]: { 
			color: 'yellow', 
			icon: '⚠', 
			text: 'Rate Limited' 
		},
		[ProviderStatus.QUOTA_EXCEEDED]: { 
			color: 'orange', 
			icon: '!', 
			text: 'Quota Exceeded' 
		},
		[ProviderStatus.ERROR]: { 
			color: 'red', 
			icon: '✕', 
			text: 'Error' 
		},
		[ProviderStatus.MAINTENANCE]: { 
			color: 'blue', 
			icon: '🔧', 
			text: 'Maintenance' 
		}
	};
	
	// Provider display names
	const providerNames: Record<ProviderType, string> = {
		[ProviderType.ANTHROPIC]: 'Anthropic',
		[ProviderType.OPENAI]: 'OpenAI',
		[ProviderType.GOOGLE]: 'Google AI'
	};
	
	// Provider logos/icons
	const providerIcons: Record<ProviderType, string> = {
		[ProviderType.ANTHROPIC]: '🤖',
		[ProviderType.OPENAI]: '🧠',
		[ProviderType.GOOGLE]: '🌐'
	};
	
	/**
	 * Save configuration changes
	 */
	async function saveConfig() {
		if (!config) return;
		
		isSaving = true;
		try {
			await providerStore.configureProvider({
				...config,
				...editConfig
			} as ProviderConfig);
			
			isEditing = false;
			editConfig = {};
		} catch (error) {
			console.error('Failed to save configuration:', error);
		} finally {
			isSaving = false;
		}
	}
	
	/**
	 * Toggle provider enabled state
	 */
	async function toggleEnabled() {
		if (!config) return;
		
		try {
			await providerStore.configureProvider({
				...config,
				enabled: !config.enabled
			});
		} catch (error) {
			console.error('Failed to toggle provider:', error);
		}
	}
	
	/**
	 * Test provider connection
	 */
	async function testConnection() {
		const success = await providerStore.testProvider(providerType);
		if (success) {
			await providerStore.refreshHealth();
		}
	}
	
	/**
	 * Format uptime percentage
	 */
	function formatUptime(percentage: number): string {
		return `${percentage.toFixed(1)}%`;
	}
	
	/**
	 * Format latency
	 */
	function formatLatency(ms: number): string {
		if (ms < 1000) return `${Math.round(ms)}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}
</script>

<div class="provider-config">
	<!-- Provider Header -->
	<div class="provider-header">
		<div class="flex items-center gap-3">
			<span class="text-2xl">{providerIcons[providerType]}</span>
			<div>
				<h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
					{providerNames[providerType]}
				</h3>
				{#if health}
					<div class="flex items-center gap-2 mt-1">
						<span class="status-badge status-{statusConfig[health.status].color}">
							{statusConfig[health.status].icon} {statusConfig[health.status].text}
						</span>
						{#if health.uptime_percentage < 100}
							<span class="text-xs text-gray-500 dark:text-gray-400">
								{formatUptime(health.uptime_percentage)} uptime
							</span>
						{/if}
					</div>
				{/if}
			</div>
		</div>
		
		<div class="flex items-center gap-2">
			<!-- Enable/Disable Toggle -->
			<label class="relative inline-flex items-center cursor-pointer">
				<input 
					type="checkbox" 
					checked={config?.enabled}
					onchange={toggleEnabled}
					class="sr-only peer"
				>
				<div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 
				            peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer 
				            dark:bg-gray-700 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full 
				            peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] 
				            after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full 
				            after:h-5 after:w-5 after:transition-all dark:border-gray-600 
				            peer-checked:bg-blue-600"></div>
			</label>
			
			<!-- Test Connection Button -->
			<button
				onclick={testConnection}
				disabled={!config?.enabled}
				class="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md 
				       hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors
				       disabled:opacity-50 disabled:cursor-not-allowed"
				title="Test Connection"
			>
				🔌 Test
			</button>
			
			<!-- Edit Button -->
			<button
				onclick={() => isEditing = !isEditing}
				class="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md 
				       hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
			>
				{isEditing ? 'Cancel' : 'Edit'}
			</button>
		</div>
	</div>
	
	<!-- Configuration Details -->
	{#if config}
		<div class="config-details mt-4">
			{#if !isEditing}
				<!-- Read-only View -->
				<div class="grid grid-cols-2 gap-4">
					<div>
						<label class="text-sm font-medium text-gray-600 dark:text-gray-400">Priority</label>
						<p class="text-gray-900 dark:text-gray-100">{config.priority}</p>
					</div>
					<div>
						<label class="text-sm font-medium text-gray-600 dark:text-gray-400">Timeout</label>
						<p class="text-gray-900 dark:text-gray-100">{config.timeout}s</p>
					</div>
					<div>
						<label class="text-sm font-medium text-gray-600 dark:text-gray-400">Rate Limit</label>
						<p class="text-gray-900 dark:text-gray-100">{config.rate_limit_rpm} req/min</p>
					</div>
					<div>
						<label class="text-sm font-medium text-gray-600 dark:text-gray-400">Token Limit</label>
						<p class="text-gray-900 dark:text-gray-100">{config.rate_limit_tpm.toLocaleString()} tokens/min</p>
					</div>
					{#if config.budget_limit}
						<div>
							<label class="text-sm font-medium text-gray-600 dark:text-gray-400">Daily Budget</label>
							<p class="text-gray-900 dark:text-gray-100">${config.budget_limit}/day</p>
						</div>
					{/if}
					{#if health}
						<div>
							<label class="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Latency</label>
							<p class="text-gray-900 dark:text-gray-100">{formatLatency(health.avg_latency_ms)}</p>
						</div>
					{/if}
				</div>
				
				<!-- Usage Stats -->
				{#if stats}
					<div class="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
						<h4 class="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Usage Statistics</h4>
						<div class="grid grid-cols-3 gap-4 text-sm">
							<div>
								<span class="text-gray-600 dark:text-gray-400">Total Requests:</span>
								<span class="ml-1 font-medium">{stats.total_requests.toLocaleString()}</span>
							</div>
							<div>
								<span class="text-gray-600 dark:text-gray-400">Success Rate:</span>
								<span class="ml-1 font-medium">
									{((stats.successful_requests / stats.total_requests) * 100).toFixed(1)}%
								</span>
							</div>
							<div>
								<span class="text-gray-600 dark:text-gray-400">Total Cost:</span>
								<span class="ml-1 font-medium">${stats.total_cost.toFixed(2)}</span>
							</div>
						</div>
					</div>
				{/if}
			{:else}
				<!-- Edit Mode -->
				<div class="space-y-4">
					<div class="grid grid-cols-2 gap-4">
						<div>
							<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
								Priority (1-10)
							</label>
							<input
								type="number"
								bind:value={editConfig.priority}
								min="1"
								max="10"
								class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
								       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
							/>
						</div>
						<div>
							<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
								Timeout (seconds)
							</label>
							<input
								type="number"
								bind:value={editConfig.timeout}
								min="1"
								max="120"
								class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
								       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
							/>
						</div>
						<div>
							<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
								Rate Limit (req/min)
							</label>
							<input
								type="number"
								bind:value={editConfig.rate_limit_rpm}
								min="1"
								class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
								       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
							/>
						</div>
						<div>
							<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
								Token Limit (tokens/min)
							</label>
							<input
								type="number"
								bind:value={editConfig.rate_limit_tpm}
								min="100"
								class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
								       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
							/>
						</div>
					</div>
					
					<!-- Advanced Settings -->
					<button
						type="button"
						onclick={() => showAdvanced = !showAdvanced}
						class="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400"
					>
						{showAdvanced ? '▼' : '▶'} Advanced Settings
					</button>
					
					{#if showAdvanced}
						<div class="grid grid-cols-2 gap-4 pt-2">
							<div>
								<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
									Max Retries
								</label>
								<input
									type="number"
									bind:value={editConfig.max_retries}
									min="0"
									max="10"
									class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
									       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
								/>
							</div>
							<div>
								<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
									Retry Delay (seconds)
								</label>
								<input
									type="number"
									bind:value={editConfig.retry_delay}
									min="0.1"
									max="10"
									step="0.1"
									class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
									       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
								/>
							</div>
							<div>
								<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
									Daily Budget (USD)
								</label>
								<input
									type="number"
									bind:value={editConfig.budget_limit}
									min="0"
									step="0.01"
									placeholder="No limit"
									class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
									       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
								/>
							</div>
							<div>
								<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
									Base URL (optional)
								</label>
								<input
									type="url"
									bind:value={editConfig.base_url}
									placeholder="Default API endpoint"
									class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
									       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
								/>
							</div>
						</div>
					{/if}
					
					<!-- Save Button -->
					<div class="flex justify-end gap-2 pt-2">
						<button
							onclick={() => isEditing = false}
							class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
							       hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
						>
							Cancel
						</button>
						<button
							onclick={saveConfig}
							disabled={isSaving}
							class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 
							       disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
						>
							{isSaving ? 'Saving...' : 'Save Changes'}
						</button>
					</div>
				</div>
			{/if}
		</div>
		
		<!-- Credentials Section -->
		{#if config.enabled && isEditing}
			<div class="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
				<CredentialManager 
					{providerType} 
					onSave={async (credentials) => {
						// Credentials are handled by the CredentialManager component
						console.log('Credentials saved for', providerType);
					}}
				/>
			</div>
		{/if}
	{:else}
		<!-- No Configuration -->
		<div class="text-center py-8">
			<p class="text-gray-500 dark:text-gray-400 mb-4">Provider not configured</p>
			<CredentialManager 
				{providerType}
				onSave={async (credentials) => {
					// Create new configuration with credentials
					await providerStore.configureProvider({
						provider_type: providerType,
						api_key: credentials.api_key,
						enabled: true,
						priority: 1,
						timeout: 30,
						max_retries: 3,
						retry_delay: 1,
						rate_limit_rpm: 60,
						rate_limit_tpm: 40000
					}, credentials);
				}}
			/>
		</div>
	{/if}
</div>

<style>
	.provider-config {
		@apply p-6 bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700;
	}
	
	.provider-header {
		@apply flex items-center justify-between pb-4 border-b border-gray-200 dark:border-gray-700;
	}
	
	.status-badge {
		@apply inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full;
	}
	
	.status-green {
		@apply bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400;
	}
	
	.status-gray {
		@apply bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400;
	}
	
	.status-yellow {
		@apply bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400;
	}
	
	.status-orange {
		@apply bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400;
	}
	
	.status-red {
		@apply bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400;
	}
	
	.status-blue {
		@apply bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400;
	}
</style>