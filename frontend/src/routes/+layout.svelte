<script lang="ts">
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from 'svelte-sonner';
	import Navigation from '$lib/components/layout/Navigation.svelte';
</script>

<ModeWatcher />
<Toaster richColors position="top-right" />

<div class="min-h-screen bg-gray-50 dark:bg-gray-950">
	<Navigation />
	<main class="flex-1">
		<slot />
	</main>
</div>