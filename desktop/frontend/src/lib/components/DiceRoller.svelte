<script lang="ts">
    import { getMCPClient } from '$lib/mcp-robust-client';
    
    let notation = $state('1d20');
    let result = $state<any>(null);
    let rolling = $state(false);
    
    async function rollDice() {
        rolling = true;
        const client = getMCPClient();
        const rollResult = await client.rollDice(notation);
        
        if (rollResult.ok) {
            result = rollResult.data;
        }
        rolling = false;
    }
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h2 class="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
        Dice Roller
    </h2>
    
    <div class="flex gap-2">
        <input
            type="text"
            bind:value={notation}
            onkeydown={(e) => e.key === 'Enter' && rollDice()}
            placeholder="e.g., 1d20+5"
            class="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                   bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                   focus:outline-none focus:ring-2 focus:ring-green-500"
        />
        <button
            onclick={rollDice}
            disabled={rolling}
            class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700
                   disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
            {rolling ? 'Rolling...' : 'Roll'}
        </button>
    </div>
    
    {#if result}
        <div class="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <p class="text-2xl font-bold text-green-700 dark:text-green-300">
                {result.total || result.result || 'No result'}
            </p>
            {#if result.details}
                <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
                    {result.details}
                </p>
            {/if}
        </div>
    {/if}
</div>