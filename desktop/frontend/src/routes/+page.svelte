<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { getMCPClient, mcpStatus, mcpError, mcpLoading } from '$lib/mcp-robust-client';
    import { initializeNativeFeatures } from '$lib/native-features-client';
    import LazyLoad from '$lib/components/LazyLoad.svelte';
    import PerformanceMonitor from '$lib/components/PerformanceMonitor.svelte';
    import ProcessMonitor from '$lib/components/ProcessMonitor.svelte';
    import DragDropOverlay from '$lib/components/DragDropOverlay.svelte';
    import TrayActionHandler from '$lib/components/TrayActionHandler.svelte';
    import NativeFileOperations from '$lib/components/NativeFileOperations.svelte';
    
    let searchQuery = $state('');
    let searchResults = $state<any[]>([]);
    let searchError = $state<string | null>(null);
    let showProcessMonitor = $state(false);
    let fileOperations: NativeFileOperations;
    
    async function handleSearch() {
        if (!searchQuery.trim()) return;
        
        const client = getMCPClient();
        const result = await client.search(searchQuery);
        
        if (result.ok) {
            // Handle different possible result structures
            if (Array.isArray(result.data)) {
                searchResults = result.data;
            } else if (result.data?.results) {
                searchResults = result.data.results;
            } else {
                searchResults = [];
            }
            searchError = null;
        } else {
            searchError = result.error || 'Unknown error occurred';
            searchResults = [];
        }
    }
    
    onMount(async () => {
        // Initialize native features
        initializeNativeFeatures();
        
        // Connect MCP client
        const client = getMCPClient();
        await client.connect();
    });
    
    // Handle drag and drop events
    async function handleFilesSelected(event: CustomEvent<{ files: string[]; type: string }>) {
        const { files, type } = event.detail;
        console.log(`Files selected (${type}):`, files);
        
        // Handle different file types appropriately
        switch (type) {
            case 'rulebooks':
                // Process PDF rulebooks
                console.log('Processing rulebooks:', files);
                break;
            case 'campaigns': 
                // Load campaign files
                console.log('Loading campaigns:', files);
                break;
            case 'characters':
                // Import character sheets
                console.log('Importing characters:', files);
                break;
        }
    }
    
    // Handle tray actions
    async function handleTrayServerAction(event: CustomEvent<{ action: string }>) {
        const { action } = event.detail;
        console.log('Tray server action:', action);
        // Actions are handled by the ProcessMonitor component
    }
    
    async function handleQuickCampaign() {
        console.log('Quick campaign requested');
        // Open campaign creation dialog or wizard
    }
    
    async function handleImportRulebooks(event: CustomEvent<{ files: string[] }>) {
        const { files } = event.detail;
        console.log('Import rulebooks from tray:', files);
        // Process rulebook imports
    }
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Header -->
    <header class="bg-white dark:bg-gray-800 shadow">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div class="flex items-center justify-between">
                <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
                    TTRPG Assistant
                </h1>
                
                <!-- Action Buttons -->
                <div class="flex items-center gap-2">
                    <!-- File Operations -->
                    <div class="flex items-center gap-1 mr-4">
                        <button
                            onclick={() => fileOperations?.importRulebooks()}
                            class="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded-md 
                                   transition-colors duration-200"
                        >
                            📖 Import Rulebooks
                        </button>
                        <button
                            onclick={() => fileOperations?.openCampaign()}
                            class="px-3 py-1.5 text-xs bg-green-600 hover:bg-green-700 text-white rounded-md
                                   transition-colors duration-200"
                        >
                            🗂️ Open Campaign
                        </button>
                        <button
                            onclick={() => showProcessMonitor = !showProcessMonitor}
                            class="px-3 py-1.5 text-xs bg-gray-600 hover:bg-gray-700 text-white rounded-md
                                   transition-colors duration-200"
                            class:bg-gray-800={showProcessMonitor}
                        >
                            ⚙️ {showProcessMonitor ? 'Hide' : 'Show'} Monitor
                        </button>
                    </div>
                    
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
    
    <!-- Process Monitor (toggleable) -->
    {#if showProcessMonitor}
        <div class="fixed bottom-4 right-4 w-96 max-w-[90vw] z-40">
            <ProcessMonitor />
        </div>
    {/if}
    
    <!-- Performance Monitor (dev only) -->
    <PerformanceMonitor />
    
    <!-- Native Feature Components -->
    <DragDropOverlay />
    
    <TrayActionHandler
        on:server-action={handleTrayServerAction}
        on:quick-campaign={handleQuickCampaign}
        on:import-rulebooks={handleImportRulebooks}
        on:open-settings={() => console.log('Open settings')}
        on:open-about={() => console.log('Open about')}
    />
    
    <NativeFileOperations
        bind:this={fileOperations}
        on:files-selected={handleFilesSelected}
        on:file-saved={(e) => console.log('File saved:', e.detail)}
        on:directory-selected={(e) => console.log('Directory selected:', e.detail)}
    />
</div>