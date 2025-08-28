<script lang="ts">
	import type { Snippet } from 'svelte';
	import { Tabs as TabsPrimitive } from 'bits-ui';
	import { cn } from '$lib/utils/cn.js';

	type TabsTriggerProps = TabsPrimitive.TriggerProps & {
		children?: Snippet;
		class?: string;
	};

	let {
		class: className,
		children,
		value,
		disabled = false,
		...restProps
	}: TabsTriggerProps = $props();
</script>

<TabsPrimitive.Trigger
	class={cn(
		'inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm',
		className
	)}
	{value}
	{disabled}
	{...restProps}
>
	{#if children}
		{@render children()}
	{/if}
</TabsPrimitive.Trigger>