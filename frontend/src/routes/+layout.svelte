<script lang="ts">
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from 'svelte-sonner';
	import Navigation from '$lib/components/layout/Navigation.svelte';
</script>

<ModeWatcher />
<Toaster richColors position="top-right" />

<div class="app-layout">
	<Navigation />
	<main class="main-content">
		<slot />
	</main>
</div>

<style>
	.app-layout {
		@apply min-h-screen bg-gray-50 dark:bg-gray-950;
	}
	
	.main-content {
		@apply flex-1;
	}
</style>