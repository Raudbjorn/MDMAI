<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import { cn } from '$lib/utils/cn.js';

	type CardDescriptionProps = HTMLAttributes<HTMLParagraphElement> & {
		children?: Snippet;
	};

	let {
		class: className,
		children,
		...restProps
	}: CardDescriptionProps = $props();
</script>

<p
	class={cn('text-sm text-muted-foreground', className)}
	{...restProps}
>
	{#if children}
		{@render children()}
	{/if}
</p>