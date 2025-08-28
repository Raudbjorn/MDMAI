<script lang="ts">
	import { providerStore } from '$lib/stores/providers.svelte';
	import { ProviderType, ProviderStatus } from '$lib/types/providers';
	
	// Get reactive data from store
	let activeProvider = $derived(providerStore.activeProvider);
	let enabledProviders = $derived(providerStore.enabledProviders);
	let health = $derived(providerStore.health);
	let loading = $derived(providerStore.loading);
	
	// Local state
	let isOpen = $state(false);
	let isSwitching = $state(false);
	
	// Provider display config
	const providerConfig: Record<ProviderType, { name: string; icon: string; color: string }> = {
		[ProviderType.ANTHROPIC]: { 
			name: 'Claude', 
			icon: '🤖', 
			color: 'purple' 
		},
		[ProviderType.OPENAI]: { 
			name: 'GPT', 
			icon: '🧠', 
			color: 'green' 
		},
		[ProviderType.GOOGLE]: { 
			name: 'Gemini', 
			icon: '🌐', 
			color: 'blue' 
		}
	};
	
	// Get active provider config
	let activeConfig = $derived(
		activeProvider ? providerConfig[activeProvider] : null
	);
	
	// Get provider health status
	function getProviderStatus(type: ProviderType): ProviderStatus {
		const providerHealth = health.find(h => h.provider_type === type);
		return providerHealth?.status || ProviderStatus.UNAVAILABLE;
	}
	
	// Check if provider is available
	function isProviderAvailable(type: ProviderType): boolean {
		const status = getProviderStatus(type);
		return status === ProviderStatus.AVAILABLE;
	}
	
	// Get status indicator
	function getStatusIndicator(type: ProviderType): string {
		const status = getProviderStatus(type);
		switch (status) {
			case ProviderStatus.AVAILABLE:
				return '🟢';
			case ProviderStatus.RATE_LIMITED:
				return '🟡';
			case ProviderStatus.QUOTA_EXCEEDED:
				return '🟠';
			case ProviderStatus.ERROR:
				return '🔴';
			case ProviderStatus.MAINTENANCE:
				return '🔵';
			default:
				return '⚫';
		}
	}
	
	// Switch provider
	async function switchProvider(type: ProviderType) {
		if (type === activeProvider || !isProviderAvailable(type)) return;
		
		isSwitching = true;
		try {
			await providerStore.switchProvider(type);
			isOpen = false;
		} catch (error) {
			console.error('Failed to switch provider:', error);
		} finally {
			isSwitching = false;
		}
	}
	
	// Close dropdown when clicking outside
	function handleClickOutside(event: MouseEvent) {
		const target = event.target as HTMLElement;
		if (!target.closest('.provider-switcher')) {
			isOpen = false;
		}
	}
	
	// Add/remove click outside listener
	$effect(() => {
		if (isOpen) {
			document.addEventListener('click', handleClickOutside);
			return () => document.removeEventListener('click', handleClickOutside);
		}
	});
</script>

<div class="provider-switcher relative">
	<!-- Trigger Button -->
	<button
		onclick={() => isOpen = !isOpen}
		disabled={loading || isSwitching || enabledProviders.length === 0}
		class="switcher-button"
		aria-label="Switch AI Provider"
		aria-expanded={isOpen}
	>
		{#if activeConfig}
			<span class="provider-icon">{activeConfig.icon}</span>
			<span class="provider-name">{activeConfig.name}</span>
			<svg 
				class="chevron {isOpen ? 'rotate-180' : ''}" 
				width="12" 
				height="12" 
				viewBox="0 0 12 12" 
				fill="none" 
				xmlns="http://www.w3.org/2000/svg"
			>
				<path d="M2 4L6 8L10 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
			</svg>
		{:else}
			<span class="text-gray-500">No Provider</span>
		{/if}
		
		{#if isSwitching}
			<svg class="animate-spin ml-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
				<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
				<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
			</svg>
		{/if}
	</button>
	
	<!-- Dropdown Menu -->
	{#if isOpen && !loading}
		<div class="dropdown-menu">
			<div class="dropdown-header">
				<span class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
					AI Providers
				</span>
			</div>
			
			<div class="dropdown-items">
				{#each enabledProviders as provider}
					{@const config = providerConfig[provider.provider_type]}
					{@const isActive = provider.provider_type === activeProvider}
					{@const isAvailable = isProviderAvailable(provider.provider_type)}
					
					<button
						onclick={() => switchProvider(provider.provider_type)}
						disabled={!isAvailable || isActive || isSwitching}
						class="dropdown-item {isActive ? 'active' : ''} {!isAvailable ? 'unavailable' : ''}"
					>
						<div class="flex items-center justify-between w-full">
							<div class="flex items-center gap-2">
								<span class="text-lg">{config.icon}</span>
								<div class="text-left">
									<div class="font-medium">{config.name}</div>
									{#if provider.priority > 1}
										<div class="text-xs text-gray-500 dark:text-gray-400">
											Priority: {provider.priority}
										</div>
									{/if}
								</div>
							</div>
							
							<div class="flex items-center gap-2">
								{#if isActive}
									<span class="active-badge">Active</span>
								{/if}
								<span class="status-indicator" title={getProviderStatus(provider.provider_type)}>
									{getStatusIndicator(provider.provider_type)}
								</span>
							</div>
						</div>
					</button>
				{/each}
			</div>
			
			{#if enabledProviders.length === 0}
				<div class="empty-state">
					<p class="text-sm text-gray-500 dark:text-gray-400">
						No providers configured
					</p>
					<a href="/providers" class="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400">
						Configure providers →
					</a>
				</div>
			{/if}
			
			<!-- Quick Stats -->
			{#if activeProvider}
				{@const activeHealth = health.find(h => h.provider_type === activeProvider)}
				{#if activeHealth}
					<div class="dropdown-footer">
						<div class="flex justify-between text-xs text-gray-500 dark:text-gray-400">
							<span>Latency: {Math.round(activeHealth.avg_latency_ms)}ms</span>
							{#if activeHealth.rate_limit_remaining !== undefined}
								<span>Rate: {activeHealth.rate_limit_remaining} left</span>
							{/if}
						</div>
					</div>
				{/if}
			{/if}
		</div>
	{/if}
</div>

<style>
	.provider-switcher {
		@apply inline-block;
	}
	
	.switcher-button {
		@apply flex items-center gap-2 px-3 py-2 bg-white dark:bg-gray-800 
		       border border-gray-300 dark:border-gray-600 rounded-lg
		       hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors
		       disabled:opacity-50 disabled:cursor-not-allowed
		       text-gray-900 dark:text-gray-100;
	}
	
	.provider-icon {
		@apply text-lg;
	}
	
	.provider-name {
		@apply font-medium;
	}
	
	.chevron {
		@apply transition-transform duration-200;
	}
	
	.dropdown-menu {
		@apply absolute top-full mt-2 left-0 w-64 bg-white dark:bg-gray-800
		       border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg
		       z-50 overflow-hidden;
	}
	
	.dropdown-header {
		@apply px-3 py-2 border-b border-gray-200 dark:border-gray-700;
	}
	
	.dropdown-items {
		@apply py-1;
	}
	
	.dropdown-item {
		@apply w-full px-3 py-2 text-left hover:bg-gray-50 dark:hover:bg-gray-700
		       transition-colors text-gray-900 dark:text-gray-100;
	}
	
	.dropdown-item.active {
		@apply bg-blue-50 dark:bg-blue-900/20;
	}
	
	.dropdown-item.unavailable {
		@apply opacity-50 cursor-not-allowed;
	}
	
	.active-badge {
		@apply px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 
		       dark:bg-blue-900/30 dark:text-blue-400 rounded-full;
	}
	
	.status-indicator {
		@apply text-xs;
	}
	
	.empty-state {
		@apply px-3 py-4 text-center;
	}
	
	.dropdown-footer {
		@apply px-3 py-2 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900;
	}
</style>