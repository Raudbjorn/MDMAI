<script lang="ts">
	import { providerStore } from '$lib/stores/providers.svelte';
	import { ProviderType, ProviderStatus } from '$lib/types/providers';
	import type { Result } from '$lib/api/types';
	import { toast } from 'svelte-sonner';
	
	// Get reactive data from store
	let activeProvider = $derived(providerStore.activeProvider);
	let enabledProviders = $derived(providerStore.enabledProviders);
	let health = $derived(providerStore.health);
	let loading = $derived(providerStore.loading);
	let error = $derived(providerStore.error);
	
	// Local state
	let isOpen = $state(false);
	let isSwitching = $state(false);
	let screenReaderAnnouncement = $state('');
	let switcherElement: HTMLElement;
	
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
		},
		[ProviderType.OLLAMA]: { 
			name: 'Ollama', 
			icon: '🦙', 
			color: 'orange' 
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
	
	// Enhanced switch provider with Result pattern and accessibility
	async function switchProvider(type: ProviderType): Promise<Result<void, string>> {
		if (type === activeProvider) {
			announceToScreenReader(`${providerConfig[type].name} is already active`);
			return { ok: true, value: undefined };
		}
		
		if (!isProviderAvailable(type)) {
			const errorMsg = `${providerConfig[type].name} is currently unavailable`;
			announceToScreenReader(errorMsg);
			toast.error('Provider unavailable', { description: errorMsg });
			return { ok: false, error: errorMsg };
		}
		
		isSwitching = true;
		announceToScreenReader(`Switching to ${providerConfig[type].name}...`);
		
		try {
			await providerStore.switchProvider(type);
			isOpen = false;
			
			const successMsg = `Switched to ${providerConfig[type].name}`;
			announceToScreenReader(successMsg);
			toast.success('Provider switched', { description: successMsg });
			
			return { ok: true, value: undefined };
		} catch (error) {
			const errorMsg = error instanceof Error ? error.message : 'Failed to switch provider';
			announceToScreenReader(`Failed to switch: ${errorMsg}`);
			toast.error('Switch failed', { description: errorMsg });
			return { ok: false, error: errorMsg };
		} finally {
			isSwitching = false;
		}
	}
	
	// Screen reader announcements
	function announceToScreenReader(message: string) {
		screenReaderAnnouncement = message;
		setTimeout(() => {
			screenReaderAnnouncement = '';
		}, 1000);
	}
	
	// Enhanced keyboard navigation and accessibility
	function handleKeyDown(event: KeyboardEvent) {
		if (!isOpen) {
			if (event.key === 'Enter' || event.key === ' ') {
				event.preventDefault();
				toggleDropdown();
			}
			return;
		}
		
		switch (event.key) {
			case 'Escape':
				event.preventDefault();
				closeDropdown();
				switcherElement?.focus();
				break;
			case 'ArrowDown':
				event.preventDefault();
				focusNextItem();
				break;
			case 'ArrowUp':
				event.preventDefault();
				focusPreviousItem();
				break;
			case 'Home':
				event.preventDefault();
				focusFirstItem();
				break;
			case 'End':
				event.preventDefault();
				focusLastItem();
				break;
		}
	}
	
	// Close dropdown when clicking outside
	function handleClickOutside(event: MouseEvent) {
		const target = event.target as HTMLElement;
		if (!target.closest('.provider-switcher')) {
			closeDropdown();
		}
	}
	
	// Toggle dropdown
	function toggleDropdown() {
		isOpen = !isOpen;
		if (isOpen) {
			announceToScreenReader('Provider menu opened');
		}
	}
	
	// Close dropdown
	function closeDropdown() {
		if (isOpen) {
			isOpen = false;
			announceToScreenReader('Provider menu closed');
		}
	}
	
	// Focus management for keyboard navigation
	function focusNextItem() {
		const items = switcherElement?.querySelectorAll('.dropdown-item:not([disabled])') as NodeListOf<HTMLElement>;
		if (!items || items.length === 0) return;
		
		const currentIndex = Array.from(items).indexOf(document.activeElement as HTMLElement);
		const nextIndex = currentIndex < items.length - 1 ? currentIndex + 1 : 0;
		items[nextIndex]?.focus();
	}
	
	function focusPreviousItem() {
		const items = switcherElement?.querySelectorAll('.dropdown-item:not([disabled])') as NodeListOf<HTMLElement>;
		if (!items || items.length === 0) return;
		
		const currentIndex = Array.from(items).indexOf(document.activeElement as HTMLElement);
		const prevIndex = currentIndex > 0 ? currentIndex - 1 : items.length - 1;
		items[prevIndex]?.focus();
	}
	
	function focusFirstItem() {
		const firstItem = switcherElement?.querySelector('.dropdown-item:not([disabled])') as HTMLElement;
		firstItem?.focus();
	}
	
	function focusLastItem() {
		const items = switcherElement?.querySelectorAll('.dropdown-item:not([disabled])') as NodeListOf<HTMLElement>;
		const lastItem = items?.[items.length - 1];
		lastItem?.focus();
	}
	
	// Add/remove event listeners
	$effect(() => {
		if (isOpen) {
			document.addEventListener('click', handleClickOutside);
			document.addEventListener('keydown', handleKeyDown);
			
			// Focus first available item when opening
			setTimeout(() => {
				focusFirstItem();
			}, 100);
			
			return () => {
				document.removeEventListener('click', handleClickOutside);
				document.removeEventListener('keydown', handleKeyDown);
			};
		}
	});
</script>

<!-- Screen reader announcements -->
<div aria-live="polite" aria-atomic="true" class="sr-only">
	{screenReaderAnnouncement}
</div>

<div class="provider-switcher relative" bind:this={switcherElement}>
	<!-- Trigger Button with enhanced accessibility -->
	<button
		onclick={toggleDropdown}
		disabled={loading || isSwitching || enabledProviders.length === 0}
		class="switcher-button focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
		aria-label={activeConfig ? `Current provider: ${activeConfig.name}. Click to switch provider` : 'No provider selected. Click to select provider'}
		aria-expanded={isOpen}
		aria-haspopup="menu"
		aria-describedby="provider-status"
		type="button"
		role="combobox"
	>
		{#if activeConfig}
			<span class="provider-icon" aria-hidden="true">{activeConfig.icon}</span>
			<span class="provider-name">{activeConfig.name}</span>
			<svg 
				class="chevron {isOpen ? 'rotate-180' : ''}" 
				width="12" 
				height="12" 
				viewBox="0 0 12 12" 
				fill="none" 
				xmlns="http://www.w3.org/2000/svg"
				aria-hidden="true"
			>
				<path d="M2 4L6 8L10 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
			</svg>
		{:else}
			<span class="text-gray-500 dark:text-gray-400">No Provider</span>
		{/if}
		
		{#if isSwitching}
			<svg class="animate-spin ml-2 h-4 w-4" fill="none" viewBox="0 0 24 24" aria-hidden="true">
				<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
				<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
			</svg>
			<span class="sr-only">Switching provider...</span>
		{/if}
	</button>
	
	<!-- Status indicator -->
	{#if activeProvider}
		<span id="provider-status" class="sr-only">
			Provider status: {getProviderStatus(activeProvider)}
		</span>
	{/if}
	
	<!-- Enhanced Dropdown Menu with full accessibility -->
	{#if isOpen && !loading}
		<div 
			class="dropdown-menu"
			role="menu"
			aria-labelledby="provider-switcher"
			aria-orientation="vertical"
		>
			<div class="dropdown-header">
				<h3 class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
					AI Providers
				</h3>
				{#if error}
					<div class="mt-1 text-xs text-red-600 dark:text-red-400" role="alert">
						{error}
					</div>
				{/if}
			</div>
			
			<div class="dropdown-items" role="none">
				{#each enabledProviders as provider, index}
					{@const config = providerConfig[provider.provider_type]}
					{@const isActive = provider.provider_type === activeProvider}
					{@const isAvailable = isProviderAvailable(provider.provider_type)}
					{@const status = getProviderStatus(provider.provider_type)}
					
					<button
						onclick={() => switchProvider(provider.provider_type)}
						disabled={!isAvailable || isActive || isSwitching}
						class="dropdown-item {isActive ? 'active' : ''} {!isAvailable ? 'unavailable' : ''}"
						role="menuitem"
						tabindex={isAvailable && !isActive ? 0 : -1}
						aria-label={`Switch to ${config.name}${isActive ? ' (currently active)' : ''}. Status: ${status}`}
						aria-current={isActive ? 'true' : 'false'}
					>
						<div class="flex items-center justify-between w-full">
							<div class="flex items-center gap-3">
								<span class="text-lg" aria-hidden="true">{config.icon}</span>
								<div class="text-left">
									<div class="font-medium text-gray-900 dark:text-gray-100">
										{config.name}
									</div>
									<div class="text-xs text-gray-500 dark:text-gray-400 space-x-2">
										{#if provider.priority > 1}
											<span>Priority: {provider.priority}</span>
										{/if}
										<span class="capitalize">{status.toLowerCase()}</span>
									</div>
								</div>
							</div>
							
							<div class="flex items-center gap-2">
								{#if isActive}
									<span class="active-badge" aria-hidden="true">Active</span>
								{/if}
								<span 
									class="status-indicator" 
									title={status}
									aria-label={`Status: ${status}`}
								>
									{getStatusIndicator(provider.provider_type)}
								</span>
							</div>
						</div>
					</button>
				{/each}
			</div>
			
			{#if enabledProviders.length === 0}
				<div class="empty-state" role="note">
					<svg class="w-8 h-8 text-gray-400 dark:text-gray-500 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
					</svg>
					<p class="text-sm text-gray-500 dark:text-gray-400 mb-2">
						No providers configured
					</p>
					<a 
						href="/providers" 
						class="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 rounded px-1"
						tabindex="0"
					>
						Configure providers →
					</a>
				</div>
			{/if}
			
			<!-- Enhanced Quick Stats -->
			{#if activeProvider}
				{@const activeHealth = health.find(h => h.provider_type === activeProvider)}
				{#if activeHealth}
					<div class="dropdown-footer">
						<div class="flex justify-between text-xs text-gray-500 dark:text-gray-400" role="status">
							<div class="flex items-center gap-1">
								<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
								</svg>
								<span>Latency: {Math.round(activeHealth.avg_latency_ms)}ms</span>
							</div>
							{#if activeHealth.rate_limit_remaining !== undefined}
								<div class="flex items-center gap-1">
									<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
									</svg>
									<span>Rate: {activeHealth.rate_limit_remaining} left</span>
								</div>
							{/if}
						</div>
						<div class="mt-1 text-xs text-gray-400 dark:text-gray-500">
							Last updated: {new Date(activeHealth.updated_at).toLocaleTimeString()}
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