<script lang="ts">
    import { onMount } from 'svelte';
    import { getMCPClient, mcpStatus, mcpError, mcpLoading } from '$lib/mcp-robust-client';
    import LazyLoad from '$lib/components/LazyLoad.svelte';
    import PerformanceMonitor from '$lib/components/PerformanceMonitor.svelte';
    
    let searchQuery = $state('');
    let searchResults = $state<any[]>([]);
    let searchError = $state<string | null>(null);
    
    async function handleSearch() {
        if (!searchQuery.trim()) return;
        
        const client = getMCPClient();
        const result = await client.search(searchQuery);
        
        if (result.ok) {
            searchResults = result.data.results || [];
            searchError = null;
        } else {
            searchError = result.error;
            searchResults = [];
        }
    }
    
    onMount(async () => {
        const client = getMCPClient();
        await client.connect();
    });
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Header -->
    <header class="bg-white dark:bg-gray-800 shadow">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div class="flex items-center justify-between">
                <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
                    TTRPG Assistant
                </h1>
                
                <!-- Connection Status -->
                <div class="flex items-center gap-2">
                    <span class="text-sm text-gray-600 dark:text-gray-400">
                        MCP Status:
                    </span>
                    <span class="flex items-center gap-1">
                        {#if $mcpStatus === 'connected'}
                            <span class="w-2 h-2 bg-green-500 rounded-full"></span>
                            <span class="text-sm text-green-600 dark:text-green-400">Connected</span>
                        {:else if $mcpStatus === 'connecting'}
                            <span class="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></span>
                            <span class="text-sm text-yellow-600 dark:text-yellow-400">Connecting...</span>
                        {:else if $mcpStatus === 'degraded'}
                            <span class="w-2 h-2 bg-orange-500 rounded-full"></span>
                            <span class="text-sm text-orange-600 dark:text-orange-400">Degraded</span>
                        {:else if $mcpStatus === 'error'}
                            <span class="w-2 h-2 bg-red-500 rounded-full"></span>
                            <span class="text-sm text-red-600 dark:text-red-400">Error</span>
                        {:else}
                            <span class="w-2 h-2 bg-gray-500 rounded-full"></span>
                            <span class="text-sm text-gray-600 dark:text-gray-400">Disconnected</span>
                        {/if}
                    </span>
                </div>
            </div>
        </div>
    </header>
    
    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Search Section -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
            <h2 class="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                Search Rules & Content
            </h2>
            
            <div class="flex gap-2">
                <input
                    type="text"
                    bind:value={searchQuery}
                    onkeydown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="Search for rules, spells, monsters..."
                    class="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                           bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                           focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={$mcpStatus !== 'connected' && $mcpStatus !== 'degraded'}
                />
                <button
                    onclick={handleSearch}
                    disabled={$mcpLoading || ($mcpStatus !== 'connected' && $mcpStatus !== 'degraded')}
                    class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700
                           disabled:opacity-50 disabled:cursor-not-allowed
                           transition-colors duration-200"
                >
                    {$mcpLoading ? 'Searching...' : 'Search'}
                </button>
            </div>
            
            {#if $mcpError}
                <div class="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <p class="text-sm text-red-600 dark:text-red-400">
                        {$mcpError}
                    </p>
                </div>
            {/if}
            
            {#if searchError}
                <div class="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <p class="text-sm text-red-600 dark:text-red-400">
                        Search error: {searchError}
                    </p>
                </div>
            {/if}
            
            {#if searchResults.length > 0}
                <div class="mt-6 space-y-3">
                    <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300">
                        Results ({searchResults.length})
                    </h3>
                    {#each searchResults as result}
                        <div class="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                            <h4 class="font-semibold text-gray-900 dark:text-white">
                                {result.title || 'Untitled'}
                            </h4>
                            <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
                                {result.content || result.description || 'No description'}
                            </p>
                            {#if result.source}
                                <p class="mt-2 text-xs text-gray-500 dark:text-gray-500">
                                    Source: {result.source}
                                </p>
                            {/if}
                        </div>
                    {/each}
                </div>
            {/if}
        </div>
        
        <!-- Lazy Load Example - Heavy Components -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Campaign Manager (lazy loaded) -->
            <LazyLoad
                component={() => import('$lib/components/CampaignManager.svelte')}
                fallback="Loading Campaign Manager..."
            />
            
            <!-- Dice Roller (lazy loaded) -->
            <LazyLoad
                component={() => import('$lib/components/DiceRoller.svelte')}
                fallback="Loading Dice Roller..."
                delay={100}
            />
        </div>
    </main>
    
    <!-- Performance Monitor (dev only) -->
    <PerformanceMonitor />
</div>