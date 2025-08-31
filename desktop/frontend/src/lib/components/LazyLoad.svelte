<script lang="ts">
    import { onMount } from 'svelte';
    import type { Component } from 'svelte';
    
    interface Props {
        component: () => Promise<{ default: Component }>;
        delay?: number;
        fallback?: string;
        props?: Record<string, any>;
    }
    
    let { component, delay = 0, fallback = 'Loading...', props = {} }: Props = $props();
    
    let ComponentToRender = $state<Component | null>(null);
    let loading = $state(true);
    let error = $state<string | null>(null);
    
    onMount(async () => {
        try {
            // Optional delay for testing or UX
            if (delay > 0) {
                await new Promise(resolve => setTimeout(resolve, delay));
            }
            
            const module = await component();
            ComponentToRender = module.default;
            loading = false;
        } catch (e) {
            error = e instanceof Error ? e.message : 'Failed to load component';
            loading = false;
            console.error('Lazy loading error:', e);
        }
    });
</script>

{#if loading}
    <div class="flex items-center justify-center p-4">
        <span class="text-gray-500">{fallback}</span>
    </div>
{:else if error}
    <div class="flex items-center justify-center p-4">
        <span class="text-red-500">Error: {error}</span>
    </div>
{:else if ComponentToRender}
    <ComponentToRender {...props} />
{/if}