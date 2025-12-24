<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { getMCPClient } from '$lib/mcp-robust-client';
    import { fade } from 'svelte/transition';
    
    let metrics = $state({
        totalCalls: 0,
        successfulCalls: 0,
        failedCalls: 0,
        cacheHits: 0,
        averageLatency: 0,
        cacheHitRate: 0,
        successRate: 0,
        cacheSize: 0
    });
    
    let visible = $state(false);
    let interval: number;
    
    onMount(() => {
        // Update metrics every 2 seconds
        const updateMetrics = () => {
            const client = getMCPClient();
            if (client) {
                metrics = client.getMetrics();
            }
        };
        
        updateMetrics();
        interval = window.setInterval(updateMetrics, 2000);
    });
    
    onDestroy(() => {
        if (interval) {
            clearInterval(interval);
        }
    });
    
    function formatPercent(value: number): string {
        return `${(value * 100).toFixed(1)}%`;
    }
    
    function formatLatency(ms: number): string {
        return `${ms.toFixed(0)}ms`;
    }
</script>

{#if import.meta.env.DEV}
    <button
        class="fixed bottom-4 right-4 p-2 bg-gray-800 text-white rounded-lg shadow-lg z-50"
        onclick={() => visible = !visible}
        aria-label="Toggle performance monitor"
    >
        📊
    </button>
    
    {#if visible}
        <div 
            transition:fade={{ duration: 200 }}
            class="fixed bottom-16 right-4 p-4 bg-gray-900 text-white rounded-lg shadow-xl z-40 min-w-[250px]"
        >
            <h3 class="text-sm font-bold mb-2">MCP Performance</h3>
            
            <div class="space-y-1 text-xs">
                <div class="flex justify-between">
                    <span>Total Calls:</span>
                    <span>{metrics.totalCalls}</span>
                </div>
                
                <div class="flex justify-between">
                    <span>Success Rate:</span>
                    <span class:text-green-400={metrics.successRate > 0.9}
                          class:text-yellow-400={metrics.successRate > 0.7 && metrics.successRate <= 0.9}
                          class:text-red-400={metrics.successRate <= 0.7}>
                        {formatPercent(metrics.successRate)}
                    </span>
                </div>
                
                <div class="flex justify-between">
                    <span>Cache Hit Rate:</span>
                    <span class:text-green-400={metrics.cacheHitRate > 0.5}>
                        {formatPercent(metrics.cacheHitRate)}
                    </span>
                </div>
                
                <div class="flex justify-between">
                    <span>Avg Latency:</span>
                    <span class:text-green-400={metrics.averageLatency < 100}
                          class:text-yellow-400={metrics.averageLatency >= 100 && metrics.averageLatency < 500}
                          class:text-red-400={metrics.averageLatency >= 500}>
                        {formatLatency(metrics.averageLatency)}
                    </span>
                </div>
                
                <div class="flex justify-between">
                    <span>Cache Size:</span>
                    <span>{metrics.cacheSize}</span>
                </div>
                
                <div class="flex justify-between">
                    <span>Failed Calls:</span>
                    <span class:text-red-400={metrics.failedCalls > 0}>
                        {metrics.failedCalls}
                    </span>
                </div>
            </div>
        </div>
    {/if}
{/if}