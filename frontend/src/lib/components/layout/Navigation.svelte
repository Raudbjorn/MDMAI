<script lang="ts">
	import { page } from '$app/stores';
	import ProviderSwitcher from '$lib/components/providers/ProviderSwitcher.svelte';
	
	// Navigation items
	const navItems = [
		{ href: '/', label: 'Home', icon: '🏠' },
		{ href: '/dashboard', label: 'Dashboard', icon: '📊' },
		{ href: '/campaign', label: 'Campaigns', icon: '⚔️' },
		{ href: '/providers', label: 'Providers', icon: '🤖' },
		{ href: '/performance', label: 'Performance', icon: '⚡' }
	];
	
	// Mobile menu state
	let mobileMenuOpen = $state(false);
	
	// Current path
	let currentPath = $derived($page.url.pathname);
	
	// Check if path is active
	function isActive(href: string): boolean {
		if (href === '/') return currentPath === '/';
		return currentPath.startsWith(href);
	}
</script>

<nav class="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-40">
	<div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
		<!-- Logo/Brand -->
		<div class="flex-shrink-0">
			<a href="/" class="flex items-center gap-2">
				<span class="text-2xl">🎲</span>
				<span class="font-bold text-xl text-gray-900 dark:text-gray-100">TTRPG Assistant</span>
			</a>
		</div>
		
		<!-- Desktop Navigation -->
		<div class="hidden md:flex flex-1 ml-8">
			<div class="flex items-center gap-1">
				{#each navItems as item}
					<a 
						href={item.href}
						class="flex items-center gap-1 px-3 py-2 rounded-md text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors {isActive(item.href) ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400' : ''}"
					>
						<span class="text-lg">{item.icon}</span>
						<span>{item.label}</span>
					</a>
				{/each}
			</div>
		</div>
		
		<!-- Right Section -->
		<div class="flex items-center gap-4">
			<!-- Provider Switcher -->
			<div class="hidden md:block">
				<ProviderSwitcher />
			</div>
			
			<!-- User Menu -->
			<button class="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
				<span class="text-xl">👤</span>
			</button>
			
			<!-- Mobile Menu Toggle -->
			<button 
				class="p-2 text-gray-700 dark:text-gray-300 md:hidden"
				onclick={() => mobileMenuOpen = !mobileMenuOpen}
			>
				{#if mobileMenuOpen}
					<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
					</svg>
				{:else}
					<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
					</svg>
				{/if}
			</button>
		</div>
	</div>
	
	<!-- Mobile Menu -->
	{#if mobileMenuOpen}
		<div class="md:hidden bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
			<div class="py-2 space-y-1 px-4">
				{#each navItems as item}
					<a 
						href={item.href}
						class="flex items-center gap-2 px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors {isActive(item.href) ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400' : ''}"
						onclick={() => mobileMenuOpen = false}
					>
						<span class="text-lg">{item.icon}</span>
						<span>{item.label}</span>
					</a>
				{/each}
			</div>
			
			<div class="p-4 border-t border-gray-200 dark:border-gray-700">
				<ProviderSwitcher />
			</div>
		</div>
	{/if}
</nav>