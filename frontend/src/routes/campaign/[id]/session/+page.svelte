<script lang="ts">
	import { page } from '$app/stores';
	import { sessionStore } from '$lib/stores/session.svelte';
	import { CollaborativeSession } from '$lib/components/collaboration';
	import { Button } from '$lib/components/ui/button';
	import { onMount } from 'svelte';
	
	// Get campaign ID from route params
	const campaignId = $derived($page.params.id ?? '');
	const campaign = $derived(sessionStore.currentCampaign);
	const user = $derived(sessionStore.user);
	
	let loading = $state(true);
	let error = $state('');
	
	onMount(async () => {
		try {
			// Ensure user is authenticated
			if (!user) {
				error = 'Please log in to access collaborative sessions';
				loading = false;
				return;
			}
			
			// Load campaign data if needed
			if (!campaign || campaign.id !== campaignId) {
				// You would typically load campaign data here
				// For now, we'll just set loading to false
				loading = false;
			} else {
				loading = false;
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load campaign';
			loading = false;
		}
	});
</script>

<svelte:head>
	<title>{campaign?.name || 'Campaign'} - Collaborative Session</title>
</svelte:head>

{#if loading}
	<div class="flex items-center justify-center min-h-screen">
		<div class="text-center">
			<div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
			<p class="mt-4 text-gray-600">Loading session...</p>
		</div>
	</div>
{:else if error}
	<div class="container mx-auto p-4">
		<div class="bg-red-50 border border-red-200 rounded-lg p-4">
			<h2 class="text-lg font-semibold text-red-700">Error</h2>
			<p class="text-red-600">{error}</p>
			<Button 
				class="mt-4"
				onclick={() => window.history.back()}
			>
				Go Back
			</Button>
		</div>
	</div>
{:else if !user}
	<div class="container mx-auto p-4">
		<div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
			<h2 class="text-lg font-semibold text-yellow-700">Authentication Required</h2>
			<p class="text-yellow-600">Please log in to join collaborative sessions.</p>
			<Button 
				class="mt-4"
				onclick={() => window.location.href = '/login'}
			>
				Log In
			</Button>
		</div>
	</div>
{:else}
	<CollaborativeSession 
		{campaignId}
		campaignName={campaign?.name || 'Campaign'}
	/>
{/if}