<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { processStats, type ProcessStats } from '../process-manager-client';
  
  let stats: ProcessStats | null = null;
  let refreshInterval: number;
  
  onMount(() => {
    // Subscribe to process stats store
    const unsubscribe = processStats.subscribe(value => {
      stats = value;
    });
    
    // Start periodic refresh
    refreshInterval = setInterval(async () => {
      try {
        const client = new (await import('../process-manager-client')).default();
        await client.getProcessStats();
      } catch (error) {
        console.error('Failed to refresh process stats:', error);
      }
    }, 5000);
    
    return () => {
      unsubscribe();
      clearInterval(refreshInterval);
    };
  });
  
  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-4">
  <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-3">
    Process Monitor
  </h3>
  
  {#if stats}
    <div class="grid grid-cols-2 gap-4">
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
        <div class="text-sm text-gray-500 dark:text-gray-400">Running Processes</div>
        <div class="text-2xl font-bold text-green-600 dark:text-green-400">
          {stats.running_processes} / {stats.total_processes}
        </div>
      </div>
      
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
        <div class="text-sm text-gray-500 dark:text-gray-400">Failed Processes</div>
        <div class="text-2xl font-bold text-red-600 dark:text-red-400">
          {stats.failed_processes}
        </div>
      </div>
      
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
        <div class="text-sm text-gray-500 dark:text-gray-400">Memory Usage</div>
        <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">
          {stats.memory_usage_mb.toFixed(1)} MB
        </div>
      </div>
      
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
        <div class="text-sm text-gray-500 dark:text-gray-400">CPU Usage</div>
        <div class="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
          {stats.cpu_usage_percent.toFixed(1)}%
        </div>
      </div>
    </div>
    
    <div class="mt-4 text-sm text-gray-600 dark:text-gray-400">
      <div>Total Restarts: {stats.total_restarts}</div>
      <div>Average Uptime: {(stats.average_uptime / 60).toFixed(1)} minutes</div>
    </div>
  {:else}
    <div class="text-gray-500 dark:text-gray-400">Loading process statistics...</div>
  {/if}
</div>