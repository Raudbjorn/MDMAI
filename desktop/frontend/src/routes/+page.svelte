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
    import type { TTRPGSearchOptions } from '$lib/mcp-stdio-bridge';
    
    // Enhanced state management with Svelte 5 runes
    interface SearchState {
        query: string;
        isSearching: boolean;
        results: SearchResult[];
        error: string | null;
        lastSearchTime: number | null;
        totalResults: number;
    }
    
    interface SearchResult {
        title: string;
        content: string;
        source: string;
        relevance: number;
        page?: number;
    }
    
    interface UIState {
        showProcessMonitor: boolean;
        showAdvancedSearch: boolean;
        selectedSystem: string | null;
        searchType: TTRPGSearchOptions['type'];
    }
    
    // Reactive state using Svelte 5 runes
    let searchState = $state<SearchState>({
        query: '',
        isSearching: false,
        results: [],
        error: null,
        lastSearchTime: null,
        totalResults: 0
    });
    
    let uiState = $state<UIState>({
        showProcessMonitor: false,
        showAdvancedSearch: false,
        selectedSystem: null,
        searchType: 'all'
    });
    
    let fileOperations: NativeFileOperations;
    
    // Derived state for better UX
    let canSearch = $derived(
        searchState.query.trim().length > 0 && 
        !searchState.isSearching && 
        ($mcpStatus === 'connected' || $mcpStatus === 'degraded')
    );
    
    let connectionStatusColor = $derived({
        connected: 'text-green-500',
        connecting: 'text-yellow-500',
        degraded: 'text-orange-500',
        error: 'text-red-500',
        reconnecting: 'text-blue-500',
        disconnected: 'text-gray-500'
    }[$mcpStatus] || 'text-gray-500');
    
    let statusIndicator = $derived({
        connected: { icon: '●', text: 'Connected', animate: false },
        connecting: { icon: '◐', text: 'Connecting...', animate: true },
        degraded: { icon: '◑', text: 'Degraded', animate: false },
        error: { icon: '●', text: 'Error', animate: false },
        reconnecting: { icon: '◐', text: 'Reconnecting...', animate: true },
        disconnected: { icon: '○', text: 'Disconnected', animate: false }
    }[$mcpStatus] || { icon: '○', text: 'Unknown', animate: false });
    
    // Available systems for filtering (would come from MCP server in real app)
    const availableSystems = ['D&D 5e', 'Pathfinder 2e', 'Call of Cthulhu', 'World of Darkness'];
    
    async function handleSearch() {
        if (!canSearch) return;
        
        // Update search state
        searchState.isSearching = true;
        searchState.error = null;
        
        const client = getMCPClient();
        const searchOptions: TTRPGSearchOptions = {
            ...(uiState.searchType && { type: uiState.searchType }),
            ...(uiState.selectedSystem && { system: uiState.selectedSystem }),
            limit: 20
        };
        
        try {
            const result = await client.search(searchState.query, searchOptions);
            
            if (result.ok && result.data) {
                searchState.results = result.data.results || [];
                searchState.totalResults = result.data.total || 0;
                searchState.lastSearchTime = Date.now();
            } else {
                searchState.error = result.error || 'Search failed';
                searchState.results = [];
                searchState.totalResults = 0;
            }
        } catch (error) {
            searchState.error = error instanceof Error ? error.message : 'Unknown error occurred';
            searchState.results = [];
            searchState.totalResults = 0;
        } finally {
            searchState.isSearching = false;
        }
    }
    
    function clearSearch() {
        searchState.query = '';
        searchState.results = [];
        searchState.error = null;
        searchState.totalResults = 0;
        searchState.lastSearchTime = null;
    }
    
    function handleKeydown(event: KeyboardEvent) {
        if (event.key === 'Enter' && canSearch) {
            handleSearch();
        } else if (event.key === 'Escape') {
            clearSearch();
        }
    }
    
    onMount(async () => {
        // Initialize native features
        initializeNativeFeatures();
        
        // Connect MCP client with error handling
        try {
            const client = getMCPClient();
            const result = await client.connect();
            
            if (!result.ok) {
                console.error('Failed to connect MCP client:', result.error);
            }
        } catch (error) {
            console.error('Error initializing MCP client:', error);
        }
    });
    
    // Enhanced file handling with proper error management
    async function handleFilesSelected(event: CustomEvent<{ files: string[]; type: string }>) {
        const { files, type } = event.detail;
        console.log(`Files selected (${type}):`, files);
        
        const client = getMCPClient();
        
        try {
            switch (type) {
                case 'rulebooks':
                    await handleRulebookFiles(client, files);
                    break;
                case 'campaigns': 
                    await handleCampaignFiles(files);
                    break;
                case 'characters':
                    await handleCharacterFiles(files);
                    break;
                default:
                    console.warn(`Unknown file type: ${type}`);
            }
        } catch (error) {
            console.error(`Error handling ${type} files:`, error);
        }
    }
    
    async function handleRulebookFiles(client: ReturnType<typeof getMCPClient>, files: string[]) {
        for (const file of files) {
            const filename = file.split('/').pop() || 'Unknown';
            const system = uiState.selectedSystem || 'Unknown';
            
            console.log(`Processing rulebook: ${filename} for system: ${system}`);
            
            try {
                const result = await client.addSource(file, filename, system);
                if (result.ok) {
                    console.log(`Successfully added rulebook: ${filename}`);
                } else {
                    console.error(`Failed to add rulebook ${filename}:`, result.error);
                }
            } catch (error) {
                console.error(`Error processing rulebook ${filename}:`, error);
            }
        }
    }
    
    async function handleCampaignFiles(files: string[]) {
        // Campaign file handling logic
        console.log('Loading campaign files:', files);
    }
    
    async function handleCharacterFiles(files: string[]) {
        // Character file handling logic
        console.log('Importing character files:', files);
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
                            onclick={() => fileOperations?.['importRulebooks']?.()}
                            class="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded-md 
                                   transition-colors duration-200"
                        >
                            📖 Import Rulebooks
                        </button>
                        <button
                            onclick={() => fileOperations?.['openCampaign']?.()}
                            class="px-3 py-1.5 text-xs bg-green-600 hover:bg-green-700 text-white rounded-md
                                   transition-colors duration-200"
                        >
                            🗂️ Open Campaign
                        </button>
                        <button
                            onclick={() => uiState.showProcessMonitor = !uiState.showProcessMonitor}
                            class="px-3 py-1.5 text-xs bg-gray-600 hover:bg-gray-700 text-white rounded-md
                                   transition-colors duration-200"
                            class:bg-gray-800={uiState.showProcessMonitor}
                        >
                            ⚙️ {uiState.showProcessMonitor ? 'Hide' : 'Show'} Monitor
                        </button>
                    </div>
                    
                    <!-- Enhanced Connection Status -->
                    <div class="flex items-center gap-2">
                        <span class="text-sm text-gray-600 dark:text-gray-400">
                            MCP Status:
                        </span>
                        <span class="flex items-center gap-1">
                            <span 
                                class={`w-2 h-2 rounded-full ${connectionStatusColor.replace('text-', 'bg-')} ${statusIndicator.animate ? 'animate-pulse' : ''}`}
                            ></span>
                            <span class={`text-sm ${connectionStatusColor}`}>
                                {statusIndicator.text}
                            </span>
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </header>
    
    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Enhanced Search Section -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
            <div class="flex items-center justify-between mb-4">
                <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
                    Search Rules & Content
                </h2>
                <button
                    onclick={() => uiState.showAdvancedSearch = !uiState.showAdvancedSearch}
                    class="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200
                           flex items-center gap-1 transition-colors duration-200"
                >
                    {uiState.showAdvancedSearch ? 'Hide' : 'Show'} Filters
                    <span class={`transform transition-transform ${uiState.showAdvancedSearch ? 'rotate-180' : ''}`}>▼</span>
                </button>
            </div>
            
            <!-- Advanced Search Options -->
            {#if uiState.showAdvancedSearch}
                <div class="mb-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                System
                            </label>
                            <select 
                                bind:value={uiState.selectedSystem}
                                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md
                                       bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                                       focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                <option value={null}>All Systems</option>
                                {#each availableSystems as system}
                                    <option value={system}>{system}</option>
                                {/each}
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                Content Type
                            </label>
                            <select 
                                bind:value={uiState.searchType}
                                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md
                                       bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                                       focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                <option value="all">All Content</option>
                                <option value="rules">Rules</option>
                                <option value="spells">Spells</option>
                                <option value="monsters">Monsters</option>
                                <option value="items">Items</option>
                            </select>
                        </div>
                    </div>
                </div>
            {/if}
            
            <!-- Search Input -->
            <div class="flex gap-2">
                <div class="flex-1 relative">
                    <input
                        type="text"
                        bind:value={searchState.query}
                        onkeydown={handleKeydown}
                        placeholder="Search for rules, spells, monsters..."
                        class="w-full px-4 py-2 pr-10 border border-gray-300 dark:border-gray-600 rounded-lg
                               bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                               focus:outline-none focus:ring-2 focus:ring-blue-500
                               disabled:opacity-50 disabled:cursor-not-allowed"
                        disabled={!canSearch && !searchState.query}
                    />
                    {#if searchState.query}
                        <button
                            onclick={clearSearch}
                            class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600
                                   dark:text-gray-500 dark:hover:text-gray-300 transition-colors"
                        >
                            ✕
                        </button>
                    {/if}
                </div>
                <button
                    onclick={handleSearch}
                    disabled={!canSearch}
                    class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700
                           disabled:opacity-50 disabled:cursor-not-allowed
                           transition-colors duration-200 flex items-center gap-2"
                >
                    {#if searchState.isSearching}
                        <span class="animate-spin">⏳</span>
                        Searching...
                    {:else}
                        🔍 Search
                    {/if}
                </button>
            </div>
            
            {#if $mcpError}
                <div class="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <p class="text-sm text-red-600 dark:text-red-400">
                        {$mcpError}
                    </p>
                </div>
            {/if}
            
            <!-- Search Results -->
            {#if $mcpError}
                <div class="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <p class="text-sm text-red-600 dark:text-red-400">
                        {$mcpError}
                    </p>
                </div>
            {/if}
            
            {#if searchState.error}
                <div class="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <p class="text-sm text-red-600 dark:text-red-400">
                        Search error: {searchState.error}
                    </p>
                </div>
            {/if}
            
            {#if searchState.results.length > 0}
                <div class="mt-6 space-y-3">
                    <div class="flex items-center justify-between">
                        <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300">
                            Results ({searchState.results.length}{#if searchState.totalResults > searchState.results.length} of {searchState.totalResults}{/if})
                        </h3>
                        {#if searchState.lastSearchTime}
                            <span class="text-xs text-gray-500">
                                {new Date(searchState.lastSearchTime).toLocaleTimeString()}
                            </span>
                        {/if}
                    </div>
                    
                    <div class="grid gap-3">
                        {#each searchState.results as result, index}
                            <div class="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors">
                                <div class="flex items-start justify-between">
                                    <h4 class="font-semibold text-gray-900 dark:text-white flex-1">
                                        {result.title || 'Untitled'}
                                    </h4>
                                    {#if result.relevance}
                                        <span class="ml-2 px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded">
                                            {Math.round(result.relevance * 100)}%
                                        </span>
                                    {/if}
                                </div>
                                
                                <p class="mt-2 text-sm text-gray-600 dark:text-gray-400 line-clamp-3">
                                    {result.content || 'No description available'}
                                </p>
                                
                                <div class="mt-3 flex items-center justify-between text-xs text-gray-500">
                                    <div class="flex items-center gap-2">
                                        <span>Source: {result.source}</span>
                                        {#if result.page}
                                            <span>• Page {result.page}</span>
                                        {/if}
                                    </div>
                                    <span>#{index + 1}</span>
                                </div>
                            </div>
                        {/each}
                    </div>
                    
                    {#if searchState.totalResults > searchState.results.length}
                        <button
                            class="w-full py-2 mt-4 text-sm text-blue-600 dark:text-blue-400 
                                   hover:text-blue-800 dark:hover:text-blue-200 transition-colors"
                            onclick={() => console.log('Load more results...')}
                        >
                            Load more results... ({searchState.totalResults - searchState.results.length} remaining)
                        </button>
                    {/if}
                </div>
            {:else if searchState.lastSearchTime && !searchState.isSearching}
                <div class="mt-6 p-8 text-center text-gray-500 dark:text-gray-400">
                    <div class="text-4xl mb-2">🔍</div>
                    <p>No results found for "{searchState.query}"</p>
                    <p class="text-sm mt-1">Try adjusting your search terms or filters</p>
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
    {#if uiState.showProcessMonitor}
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
        on:file-saved={(e: CustomEvent) => console.log('File saved:', e.detail)}
        on:directory-selected={(e: CustomEvent) => console.log('Directory selected:', e.detail)}
    />
</div>

<style>
    .line-clamp-3 {
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    /* Smooth transitions for state changes */
    .process-monitor {
        transition: all 0.3s ease-in-out;
    }
    
    /* Enhanced focus styles for accessibility */
    input:focus,
    select:focus,
    button:focus {
        outline: 2px solid transparent;
        outline-offset: 2px;
    }
    
    /* Custom scrollbar for search results */
    .search-results {
        scrollbar-width: thin;
        scrollbar-color: rgba(156, 163, 175, 0.5) transparent;
    }
    
    .search-results::-webkit-scrollbar {
        width: 6px;
    }
    
    .search-results::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .search-results::-webkit-scrollbar-thumb {
        background-color: rgba(156, 163, 175, 0.5);
        border-radius: 3px;
    }
</style>