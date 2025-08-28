<script lang="ts">
	import type { Snippet } from 'svelte';
	import { Tabs as TabsPrimitive } from 'bits-ui';
	import { cn } from '$lib/utils/cn.js';

	type TabsListProps = TabsPrimitive.ListProps & {
		children?: Snippet;
		class?: string;
	};

	let {
		class: className,
		children,
		...restProps
	}: TabsListProps = $props();
</script>

<TabsPrimitive.List
	class={cn(
		'inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground',
		className
	)}
	{...restProps}
>
	{#if children}
		{@render children()}
	{/if}
</TabsPrimitive.List>