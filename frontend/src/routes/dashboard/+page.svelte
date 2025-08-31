<script lang="ts">
	import { sessionStore } from '$lib/stores/session.svelte';
	import { Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui';
	import { BookOpen, Users, Brain, Search, Plus, Settings, Upload } from 'lucide-svelte';
	import { toast } from 'svelte-sonner';
	import { goto } from '$app/navigation';

	let searchQuery = $state('');
	let searchResults = $state<any[]>([]);
	let isSearching = $state(false);
	
	const recentMessages = $derived(sessionStore.messages.slice(-5));
	const stats = $derived({
		campaigns: sessionStore.user?.campaigns.length || 0,
		isConnected: sessionStore.isConnected,
		currentCampaign: sessionStore.currentCampaign?.name || 'None',
		session: sessionStore.currentGameSession?.active ? 'Active' : 'Inactive'
	});

	const quickActions = [
		{ id: 'upload_pdf', title: 'Upload PDF', desc: 'Process game documents', icon: Upload, href: '/upload' },
		{ id: 'new_session', title: 'New Session', desc: 'Start a new game session', icon: Plus },
		{ id: 'roll_dice', title: 'Quick Roll', desc: 'Roll dice quickly', icon: Brain },
		{ id: 'generate_npc', title: 'Generate NPC', desc: 'Create a new NPC', icon: Users },
		{ id: 'view_notes', title: 'Session Notes', desc: 'View recent notes', icon: BookOpen }
	];

	const handleSearch = async () => {
		if (!searchQuery.trim()) return;
		
		isSearching = true;
		try {
			const result = await sessionStore.callTool('search_rules', { query: searchQuery, limit: 10 });
			searchResults = result.data || [];
			if (!searchResults.length) toast.info('No results found');
		} catch (error) {
			toast.error(`Search failed: ${(error as Error).message}`);
		} finally {
			isSearching = false;
		}
	};

	const quickAction = async (action: string) => {
		try {
			const actions = {
				upload_pdf: () => goto('/upload'),
				roll_dice: async () => {
					const result = await sessionStore.callTool('roll_dice', { dice_notation: '1d20' });
					toast.success(`Rolled 1d20: ${result.data.result}`);
				},
				generate_npc: async () => {
					toast.info('Generating NPC...');
					const npc = await sessionStore.callTool('generate_npc', { level: 5, type: 'merchant' });
					toast.success(`Generated NPC: ${npc.data.name}`);
				},
				default: () => toast.info(`Action ${action} not yet implemented`)
			};
			await (actions[action as keyof typeof actions] || actions.default)();
		} catch (error) {
			toast.error(`Action failed: ${(error as Error).message}`);
		}
	};
</script>

<svelte:head>
	<title>Dashboard - TTRPG Assistant</title>
</svelte:head>

<div class="min-h-screen bg-background">
	<!-- Header -->
	<header class="sticky top-0 z-50 border-b bg-background/95 backdrop-blur">
		<div class="container mx-auto px-4">
			<div class="flex h-16 items-center justify-between">
				<div class="flex items-center gap-4">
					<h1 class="text-xl font-bold">TTRPG Assistant</h1>
					<span class="text-sm text-muted-foreground">Dashboard</span>
				</div>
				<div class="flex items-center gap-2">
					<Button variant="ghost" size="icon">
						<Settings class="h-5 w-5" />
					</Button>
				</div>
			</div>
		</div>
	</header>

	<div class="container mx-auto px-4 py-6">
		<!-- Status Bar -->
		<div class="mb-6 flex items-center justify-between rounded-lg bg-muted p-4">
			<div class="flex items-center gap-6 text-sm">
				<div>
					<span class="text-muted-foreground">Connection:</span>
					<span class="ml-2 font-medium {stats.isConnected ? 'text-green-600' : 'text-red-600'}">
						{stats.isConnected ? 'Connected' : 'Disconnected'}
					</span>
				</div>
				<div>
					<span class="text-muted-foreground">Campaign:</span>
					<span class="ml-2 font-medium">{stats.currentCampaign}</span>
				</div>
				<div>
					<span class="text-muted-foreground">Session:</span>
					<span class="ml-2 font-medium">{stats.session}</span>
				</div>
			</div>
		</div>

		<!-- Quick Search -->
		<Card class="mb-6">
			<CardHeader>
				<CardTitle class="flex items-center gap-2">
					<Search class="h-5 w-5" />
					Quick Search
				</CardTitle>
				<CardDescription>Search rules, spells, monsters, and more</CardDescription>
			</CardHeader>
			<CardContent>
				<div class="flex gap-2">
					<input
						type="text"
						bind:value={searchQuery}
						onkeydown={(e) => e.key === 'Enter' && handleSearch()}
						placeholder="Search for rules, spells, monsters..."
						class="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
						disabled={isSearching}
					/>
					<Button onclick={handleSearch} disabled={isSearching || !searchQuery.trim()}>
						{isSearching ? 'Searching...' : 'Search'}
					</Button>
				</div>
				
				{#if searchResults.length > 0}
					<div class="mt-4 space-y-2">
						{#each searchResults as result}
							<div class="rounded border p-3">
								<div class="font-medium">{result.title || 'Result'}</div>
								<div class="text-sm text-muted-foreground">{result.content}</div>
								{#if result.source}
									<div class="mt-1 text-xs text-muted-foreground">
										Source: {result.source} {result.page ? `- Page ${result.page}` : ''}
									</div>
								{/if}
							</div>
						{/each}
					</div>
				{/if}
			</CardContent>
		</Card>

		<!-- Quick Actions -->
		<div class="mb-6 grid gap-4 md:grid-cols-2 lg:grid-cols-5">
			{#each quickActions as { id, title, desc, icon: Icon, href }}
				<Card class="cursor-pointer hover:bg-muted/50" onclick={href ? () => goto(href) : () => quickAction(id)}>
					<CardHeader class="flex flex-row items-center justify-between space-y-0 pb-2">
						<CardTitle class="text-sm font-medium">{title}</CardTitle>
						<Icon class="h-4 w-4 text-muted-foreground" />
					</CardHeader>
					<CardContent>
						<p class="text-xs text-muted-foreground">{desc}</p>
					</CardContent>
				</Card>
			{/each}
		</div>

		<!-- Main Content Areas -->
		<div class="grid gap-6 md:grid-cols-2">
			<!-- Active Campaign -->
			<Card>
				<CardHeader>
					<CardTitle>Active Campaign</CardTitle>
					<CardDescription>Currently running campaign details</CardDescription>
				</CardHeader>
				<CardContent>
					{#if sessionStore.currentCampaign}
						<div class="space-y-2">
							<p><span class="font-medium">Name:</span> {sessionStore.currentCampaign.name}</p>
							<p><span class="font-medium">System:</span> {sessionStore.currentCampaign.system}</p>
							<p><span class="font-medium">Players:</span> {sessionStore.currentCampaign.players.length}</p>
						</div>
					{:else}
						<p class="text-muted-foreground">No active campaign selected</p>
						<Button href="/campaigns" class="mt-4" variant="outline">
							Select Campaign
						</Button>
					{/if}
				</CardContent>
			</Card>

			<!-- Recent Activity -->
			<Card>
				<CardHeader>
					<CardTitle>Recent Activity</CardTitle>
					<CardDescription>Latest actions and updates</CardDescription>
				</CardHeader>
				<CardContent>
					<div class="space-y-2">
						{#each recentMessages as message}
							<div class="flex items-center gap-2 text-sm">
								<span class="text-xs text-muted-foreground">
									{new Date(message.timestamp).toLocaleTimeString()}
								</span>
								<span>{message.type}</span>
							</div>
						{:else}
							<p class="text-muted-foreground">No recent activity</p>
						{/each}
					</div>
				</CardContent>
			</Card>
		</div>
	</div>
</div>