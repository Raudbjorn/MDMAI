<script lang="ts">
	import type { Snippet } from 'svelte';
	import { Tabs as TabsPrimitive } from 'bits-ui';
	import { cn } from '$lib/utils/cn.js';

	type TabsContentProps = TabsPrimitive.ContentProps & {
		children?: Snippet;
		class?: string;
	};

	let {
		class: className,
		children,
		value,
		...restProps
	}: TabsContentProps = $props();
</script>

<TabsPrimitive.Content
	class={cn(
		'mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
		className
	)}
	{value}
	{...restProps}
>
	{#if children}
		{@render children()}
	{/if}
</TabsPrimitive.Content>