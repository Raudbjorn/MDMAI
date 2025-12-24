<script lang="ts">
    import { getMCPClient } from '$lib/mcp-robust-client';
    
    interface Campaign {
        id: string;
        name: string;
        description?: string;
        [key: string]: any;
    }
    
    interface ListCampaignsResponse {
        campaigns: Campaign[];
    }
    
    let campaigns = $state<Campaign[]>([]);
    let loading = $state(false);
    
    async function loadCampaigns() {
        loading = true;
        const client = getMCPClient();
        const result = await client.callWithRetry('list_campaigns', {});
        
        if (result.ok && result.data) {
            const campaignData = result.data as ListCampaignsResponse;
            campaigns = campaignData.campaigns || [];
        }
        loading = false;
    }
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h2 class="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
        Campaign Manager
    </h2>
    
    <button
        onclick={loadCampaigns}
        disabled={loading}
        class="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700
               disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
    >
        {loading ? 'Loading...' : 'Load Campaigns'}
    </button>
    
    {#if campaigns.length > 0}
        <div class="mt-4 space-y-2">
            {#each campaigns as campaign}
                <div class="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                    <p class="font-medium text-gray-900 dark:text-white">
                        {campaign.name}
                    </p>
                </div>
            {/each}
        </div>
    {/if}
</div>