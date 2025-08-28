<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import { cn } from '$lib/utils/cn.js';

	type CardContentProps = HTMLAttributes<HTMLDivElement> & {
		children?: Snippet;
	};

	let {
		class: className,
		children,
		...restProps
	}: CardContentProps = $props();
</script>

<div
	class={cn('p-6 pt-0', className)}
	{...restProps}
>
	{#if children}
		{@render children()}
	{/if}
</div>