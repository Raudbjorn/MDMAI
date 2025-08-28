<script lang="ts">
	import type { Snippet } from 'svelte';
	import { Tabs as TabsPrimitive } from 'bits-ui';
	import { cn } from '$lib/utils/cn.js';

	type TabsProps = TabsPrimitive.Props & {
		children?: Snippet;
		class?: string;
	};

	let {
		class: className,
		children,
		...restProps
	}: TabsProps = $props();
</script>

<TabsPrimitive.Root
	class={cn('', className)}
	{...restProps}
>
	{#if children}
		{@render children()}
	{/if}
</TabsPrimitive.Root>