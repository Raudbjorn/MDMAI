<script lang="ts">
	import { providerStore } from '$lib/stores/providers.svelte';
	import { ProviderType } from '$lib/types/providers';
	import ProviderConfig from '$lib/components/providers/ProviderConfig.svelte';
	import CostDashboard from '$lib/components/providers/CostDashboard.svelte';
	import UsageAnalytics from '$lib/components/providers/UsageAnalytics.svelte';
	import ProviderSwitcher from '$lib/components/providers/ProviderSwitcher.svelte';
	
	// Tab state
	let activeTab = $state<'configuration' | 'costs' | 'analytics'>('configuration');
	
	// Store state
	let loading = $derived(providerStore.loading);
	let error = $derived(providerStore.error);
	let initialized = $derived(providerStore.initialized);
	
	// Provider types
	const providers = [
		ProviderType.ANTHROPIC,
		ProviderType.OPENAI,
		ProviderType.GOOGLE
	];
</script>

<div class="max-w-7xl mx-auto px-4 py-8 space-y-6">
	<!-- Page Header -->
	<div class="flex items-start justify-between pb-6 border-b border-gray-200 dark:border-gray-700">
		<div>
			<h1 class="text-3xl font-bold text-gray-900 dark:text-gray-100">
				Provider Management
			</h1>
			<p class="text-gray-600 dark:text-gray-400 mt-2">
				Configure AI providers, manage credentials, and monitor usage
			</p>
		</div>
		
		<div class="flex items-center gap-4">
			<ProviderSwitcher />
		</div>
	</div>
	
	<!-- Error Display -->
	{#if error}
		<div class="flex items-center justify-between p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
			<div class="flex items-center gap-2">
				<span class="text-red-600 dark:text-red-400">⚠️</span>
				<span>{error}</span>
			</div>
			<button 
				onclick={() => providerStore.clearError()}
				class="text-red-600 hover:text-red-700 dark:text-red-400"
			>
				✕
			</button>
		</div>
	{/if}
	
	<!-- Loading State -->
	{#if loading && !initialized}
		<div class="flex flex-col items-center justify-center py-12">
			<svg class="animate-spin h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24">
				<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
				<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
			</svg>
			<p class="text-gray-600 dark:text-gray-400 mt-2">Loading provider configurations...</p>
		</div>
	{:else}
		<!-- Tab Navigation -->
		<div class="flex gap-2 border-b border-gray-200 dark:border-gray-700">
			<button
				onclick={() => activeTab = 'configuration'}
				class="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 border-b-2 border-transparent transition-colors {activeTab === 'configuration' ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400' : ''}"
			>
				🔧 Configuration
			</button>
			<button
				onclick={() => activeTab = 'costs'}
				class="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 border-b-2 border-transparent transition-colors {activeTab === 'costs' ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400' : ''}"
			>
				💰 Cost Management
			</button>
			<button
				onclick={() => activeTab = 'analytics'}
				class="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 border-b-2 border-transparent transition-colors {activeTab === 'analytics' ? 'text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400' : ''}"
			>
				📊 Usage Analytics
			</button>
		</div>
		
		<!-- Tab Content -->
		<div class="py-6">
			{#if activeTab === 'configuration'}
				<div class="configuration-tab">
					<div class="mb-6">
						<h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100">
							Provider Configuration
						</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
							Set up API keys, configure settings, and manage provider priorities
						</p>
					</div>
					
					<div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
						{#each providers as providerType}
							<ProviderConfig {providerType} />
						{/each}
					</div>
					
					<!-- Quick Actions -->
					<div class="p-6 bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
						<h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
							Quick Actions
						</h3>
						
						<div class="grid grid-cols-2 md:grid-cols-3 gap-4">
							<button 
								onclick={() => providerStore.refreshHealth()}
								class="flex flex-col items-center gap-2 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors border border-gray-200 dark:border-gray-700"
							>
								<span class="text-2xl">🔄</span>
								<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Refresh Health</span>
							</button>
							
							<button 
								onclick={() => providerStore.refreshStats()}
								class="flex flex-col items-center gap-2 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors border border-gray-200 dark:border-gray-700"
							>
								<span class="text-2xl">📈</span>
								<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Update Stats</span>
							</button>
							
							<a 
								href="/api/providers/export"
								class="flex flex-col items-center gap-2 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors border border-gray-200 dark:border-gray-700"
							>
								<span class="text-2xl">💾</span>
								<span class="text-sm font-medium text-gray-700 dark:text-gray-300">Export Config</span>
							</a>
						</div>
					</div>
				</div>
			{:else if activeTab === 'costs'}
				<div class="costs-tab">
					<CostDashboard />
				</div>
			{:else if activeTab === 'analytics'}
				<div class="analytics-tab">
					<UsageAnalytics />
				</div>
			{/if}
		</div>
	{/if}
</div>