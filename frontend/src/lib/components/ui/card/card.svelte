<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import { cn } from '$lib/utils/cn.js';

	type CardProps = HTMLAttributes<HTMLDivElement> & {
		children?: Snippet;
	};

	let {
		class: className,
		children,
		...restProps
	}: CardProps = $props();
</script>

<div
	class={cn('rounded-lg border bg-card text-card-foreground shadow-sm', className)}
	{...restProps}
>
	{#if children}
		{@render children()}
	{/if}
</div>