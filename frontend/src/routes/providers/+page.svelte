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

<div class="providers-page">
	<!-- Page Header -->
	<div class="page-header">
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
		<div class="error-banner">
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
		<div class="loading-state">
			<svg class="animate-spin h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24">
				<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
				<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
			</svg>
			<p class="text-gray-600 dark:text-gray-400 mt-2">Loading provider configurations...</p>
		</div>
	{:else}
		<!-- Tab Navigation -->
		<div class="tab-navigation">
			<button
				onclick={() => activeTab = 'configuration'}
				class="tab-button {activeTab === 'configuration' ? 'active' : ''}"
			>
				🔧 Configuration
			</button>
			<button
				onclick={() => activeTab = 'costs'}
				class="tab-button {activeTab === 'costs' ? 'active' : ''}"
			>
				💰 Cost Management
			</button>
			<button
				onclick={() => activeTab = 'analytics'}
				class="tab-button {activeTab === 'analytics' ? 'active' : ''}"
			>
				📊 Usage Analytics
			</button>
		</div>
		
		<!-- Tab Content -->
		<div class="tab-content">
			{#if activeTab === 'configuration'}
				<div class="configuration-tab">
					<div class="section-header">
						<h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100">
							Provider Configuration
						</h2>
						<p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
							Set up API keys, configure settings, and manage provider priorities
						</p>
					</div>
					
					<div class="providers-grid">
						{#each providers as providerType}
							<ProviderConfig {providerType} />
						{/each}
					</div>
					
					<!-- Quick Actions -->
					<div class="quick-actions">
						<h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
							Quick Actions
						</h3>
						
						<div class="action-cards">
							<button 
								onclick={() => providerStore.refreshHealth()}
								class="action-card"
							>
								<span class="action-icon">🔄</span>
								<span class="action-label">Refresh Health</span>
							</button>
							
							<button 
								onclick={() => providerStore.refreshStats()}
								class="action-card"
							>
								<span class="action-icon">📈</span>
								<span class="action-label">Update Stats</span>
							</button>
							
							<a 
								href="/api/providers/export"
								class="action-card"
							>
								<span class="action-icon">💾</span>
								<span class="action-label">Export Config</span>
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

<style>
	.providers-page {
		@apply max-w-7xl mx-auto px-4 py-8 space-y-6;
	}
	
	.page-header {
		@apply flex items-start justify-between pb-6 border-b border-gray-200 dark:border-gray-700;
	}
	
	.error-banner {
		@apply flex items-center justify-between p-4 bg-red-50 dark:bg-red-900/20 
		       border border-red-200 dark:border-red-800 rounded-lg;
	}
	
	.loading-state {
		@apply flex flex-col items-center justify-center py-12;
	}
	
	.tab-navigation {
		@apply flex gap-2 border-b border-gray-200 dark:border-gray-700;
	}
	
	.tab-button {
		@apply px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 
		       dark:hover:text-gray-100 border-b-2 border-transparent transition-colors;
	}
	
	.tab-button.active {
		@apply text-blue-600 dark:text-blue-400 border-blue-600 dark:border-blue-400;
	}
	
	.tab-content {
		@apply py-6;
	}
	
	.section-header {
		@apply mb-6;
	}
	
	.providers-grid {
		@apply grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8;
	}
	
	.quick-actions {
		@apply p-6 bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700;
	}
	
	.action-cards {
		@apply grid grid-cols-2 md:grid-cols-3 gap-4;
	}
	
	.action-card {
		@apply flex flex-col items-center gap-2 p-4 bg-gray-50 dark:bg-gray-800 
		       rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors
		       border border-gray-200 dark:border-gray-700;
	}
	
	.action-icon {
		@apply text-2xl;
	}
	
	.action-label {
		@apply text-sm font-medium text-gray-700 dark:text-gray-300;
	}
</style>