# Data Management System Integration Example

This guide shows how to integrate the new data management system into the existing TTRPG Assistant application.

## 1. Add the Dashboard to Main Page

Update `src/routes/+page.svelte` to include the data management dashboard:

```svelte
<script lang="ts">
    import { onMount } from 'svelte';
    import CampaignManager from '$lib/components/CampaignManager.svelte';
    import ProcessMonitor from '$lib/components/ProcessMonitor.svelte';
    import DataManagerDashboard from '$lib/components/DataManagerDashboard.svelte';
    import DragDropOverlay from '$lib/components/DragDropOverlay.svelte';
    import { getMCPClient } from '$lib/mcp-robust-client';

    let activeTab = $state<'campaigns' | 'process' | 'data' | 'tools'>('campaigns');
    let mcpConnected = $state(false);

    onMount(async () => {
        const client = getMCPClient();
        // Try to connect to check status
        try {
            await client.listResources();
            mcpConnected = true;
        } catch {
            mcpConnected = false;
        }
    });
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Navigation Header -->
    <div class="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center py-6">
                <div>
                    <h1 class="text-3xl font-bold text-gray-900 dark:text-white">
                        TTRPG Assistant
                    </h1>
                    <p class="text-gray-600 dark:text-gray-300">
                        Your comprehensive tabletop RPG management system
                    </p>
                </div>
                
                <div class="flex items-center space-x-4">
                    <div class={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${
                        mcpConnected 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-red-100 text-red-800'
                    }`}>
                        <div class={`w-2 h-2 rounded-full ${
                            mcpConnected ? 'bg-green-600' : 'bg-red-600'
                        }`}></div>
                        <span>{mcpConnected ? 'Connected' : 'Disconnected'}</span>
                    </div>
                </div>
            </div>
            
            <!-- Tab Navigation -->
            <nav class="-mb-px flex space-x-8">
                {#each [
                    { id: 'campaigns', label: 'Campaigns', icon: '⚔️' },
                    { id: 'data', label: 'Data Management', icon: '💾' },
                    { id: 'process', label: 'Process Monitor', icon: '📊' },
                    { id: 'tools', label: 'Tools', icon: '🔧' }
                ] as tab}
                    <button
                        onclick={() => activeTab = tab.id as any}
                        class={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                            activeTab === tab.id
                                ? 'border-purple-500 text-purple-600'
                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                        }`}
                    >
                        <span class="mr-2">{tab.icon}</span>
                        {tab.label}
                    </button>
                {/each}
            </nav>
        </div>
    </div>

    <!-- Tab Content -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {#if activeTab === 'campaigns'}
            <CampaignManager />
        {:else if activeTab === 'data'}
            <DataManagerDashboard />
        {:else if activeTab === 'process'}
            <ProcessMonitor />
        {:else if activeTab === 'tools'}
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <!-- Tools content here -->
                <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                        Coming Soon
                    </h3>
                    <p class="text-gray-600 dark:text-gray-300">
                        Additional tools and utilities will be available here.
                    </p>
                </div>
            </div>
        {/if}
    </div>
</div>

<!-- Drag & Drop Overlay -->
<DragDropOverlay />
```

## 2. Initialize Data Management in App Setup

Update your app's initialization to include data management:

```typescript
// src/lib/app-initialization.ts
import { dataManager } from '$lib/data-manager-client';

export async function initializeApp() {
    try {
        // Initialize data management system
        await dataManager.initialize();
        console.log('Data management system initialized');
        
        // Your existing MCP initialization
        // ... existing code ...
        
    } catch (error) {
        console.error('Failed to initialize app:', error);
        throw error;
    }
}
```

## 3. Update Campaign Manager to Use New Data Layer

Replace the existing CampaignManager with data manager integration:

```svelte
<!-- src/lib/components/CampaignManager.svelte -->
<script lang="ts">
    import { onMount } from 'svelte';
    import { dataManager, type Campaign, getCampaignStatusColor } from '$lib/data-manager-client';
    
    let campaigns = $state<Campaign[]>([]);
    let loading = $state(false);
    let error = $state<string | null>(null);

    onMount(async () => {
        await loadCampaigns();
    });

    async function loadCampaigns() {
        loading = true;
        error = null;
        
        try {
            const result = await dataManager.listCampaigns({ limit: 50 });
            campaigns = result.items;
        } catch (e: any) {
            error = e.message || 'Failed to load campaigns';
            // Fallback to MCP client if data manager not available
            console.warn('Data manager not available, falling back to MCP');
            // ... existing MCP code as fallback ...
        } finally {
            loading = false;
        }
    }

    async function createNewCampaign() {
        const newCampaign: Campaign = {
            id: crypto.randomUUID(),
            name: "New Campaign",
            system: "D&D 5e",
            description: "A new adventure awaits!",
            status: "planning",
            is_active: true,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            settings: {}
        };

        try {
            await dataManager.createCampaign(newCampaign);
            await loadCampaigns();
        } catch (e: any) {
            error = e.message || 'Failed to create campaign';
        }
    }
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow">
    <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div class="flex justify-between items-center">
            <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
                Campaign Manager
            </h2>
            <button
                onclick={createNewCampaign}
                disabled={loading}
                class="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700
                       disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
                New Campaign
            </button>
        </div>
    </div>
    
    <div class="p-6">
        {#if error}
            <div class="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p class="text-red-800">{error}</p>
            </div>
        {/if}

        {#if loading}
            <div class="text-center py-8">
                <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
                <p class="mt-2 text-gray-600">Loading campaigns...</p>
            </div>
        {:else if campaigns.length > 0}
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {#each campaigns as campaign}
                    <div class="border border-gray-200 dark:border-gray-600 rounded-lg p-4 hover:shadow-md transition-shadow">
                        <div class="flex justify-between items-start mb-2">
                            <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
                                {campaign.name}
                            </h3>
                            <span class={`px-2 py-1 rounded-full text-xs font-medium ${getCampaignStatusColor(campaign.status)}`}>
                                {campaign.status}
                            </span>
                        </div>
                        <p class="text-sm text-gray-600 dark:text-gray-300 mb-2">
                            {campaign.system}
                        </p>
                        {#if campaign.description}
                            <p class="text-xs text-gray-500 dark:text-gray-400">
                                {campaign.description}
                            </p>
                        {/if}
                        <div class="mt-3 text-xs text-gray-400">
                            Updated: {new Date(campaign.updated_at).toLocaleDateString()}
                        </div>
                    </div>
                {/each}
            </div>
        {:else}
            <div class="text-center py-8">
                <div class="text-6xl mb-4">⚔️</div>
                <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    No Campaigns Yet
                </h3>
                <p class="text-gray-600 dark:text-gray-300 mb-4">
                    Create your first campaign to start your adventure!
                </p>
                <button
                    onclick={createNewCampaign}
                    class="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                >
                    Create First Campaign
                </button>
            </div>
        {/if}
    </div>
</div>
```

## 4. Add Data Management Menu Item

Update your navigation or add a settings menu to access data management features:

```svelte
<!-- In your main navigation component -->
<div class="flex items-center space-x-4">
    <!-- Existing navigation items -->
    
    <!-- Data Management Dropdown -->
    <div class="relative">
        <button class="flex items-center px-3 py-2 text-gray-700 hover:text-purple-600 transition-colors">
            <span class="mr-2">💾</span>
            Data
        </button>
        <!-- Dropdown menu with backup, integrity check, etc. -->
    </div>
</div>
```

## 5. Environment Configuration

Ensure your Tauri configuration allows data management operations:

```json
// src-tauri/tauri.conf.json
{
  "tauri": {
    "allowlist": {
      "fs": {
        "all": true,
        "readFile": true,
        "writeFile": true,
        "createDir": true,
        "removeDir": true,
        "copyFile": true,
        "scope": ["$APPDATA/**", "$APPLOCALDATA/**"]
      },
      "dialog": {
        "all": true,
        "open": true,
        "save": true
      }
    }
  }
}
```

## 6. Error Handling and Fallbacks

Implement graceful fallbacks for when the data management system is not available:

```typescript
// src/lib/data-manager-fallback.ts
export class DataManagerFallback {
    static async withFallback<T>(
        dataManagerOperation: () => Promise<T>,
        mcpFallback: () => Promise<T>
    ): Promise<T> {
        try {
            return await dataManagerOperation();
        } catch (error) {
            console.warn('Data manager operation failed, falling back to MCP:', error);
            return await mcpFallback();
        }
    }
}

// Usage example:
const campaigns = await DataManagerFallback.withFallback(
    () => dataManager.listCampaigns(),
    () => getMCPClient().callWithRetry('list_campaigns', {})
);
```

## 7. Performance Monitoring

Add performance monitoring for data operations:

```typescript
// src/lib/performance-monitor.ts
export function withPerformanceMonitoring<T>(
    operation: () => Promise<T>,
    operationName: string
): Promise<T> {
    return new Promise(async (resolve, reject) => {
        const start = performance.now();
        try {
            const result = await operation();
            const duration = performance.now() - start;
            console.log(`${operationName} completed in ${duration.toFixed(2)}ms`);
            resolve(result);
        } catch (error) {
            const duration = performance.now() - start;
            console.error(`${operationName} failed after ${duration.toFixed(2)}ms:`, error);
            reject(error);
        }
    });
}
```

This integration approach provides a seamless upgrade path while maintaining backward compatibility with the existing MCP-based system. Users will benefit from improved performance, data integrity, and offline capabilities while the application gracefully handles any initialization issues.