<script lang="ts">
	import { sessionStore } from '$lib/stores/session.svelte';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card';
	import { BookOpen, Users, Brain, Search, Plus, Settings, Upload } from 'lucide-svelte';
	import { toast } from 'svelte-sonner';
	import { goto } from '$app/navigation';
	import type { Result } from '$lib/types/providers.js';

	// Type-safe search results interface
	interface SearchResult {
		title?: string;
		content: string;
		source?: string;
		page?: number;
	}

	// Enhanced state management with proper typing
	let searchState = $state({
		query: '',
		results: [] as SearchResult[],
		isSearching: false
	});

	// Derived state for recent messages with proper typing
	const recentMessages = $derived(
		sessionStore.messages
			.slice(-5)
			.filter((msg): msg is typeof msg => msg && typeof msg === 'object')
	);

	// Improved error handling with Result pattern
	async function handleSearch(): Promise<void> {
		const query = searchState.query.trim();
		if (!query) return;

		searchState.isSearching = true;
		
		try {
			const result = await sessionStore.callTool('search_rules', {
				query,
				limit: 10
			});

			// Type-safe result handling
			if (result && 'data' in result) {
				searchState.results = Array.isArray(result.data) ? result.data : [];
				if (searchState.results.length === 0) {
					toast.info('No results found for your search');
				}
			} else {
				throw new Error('Invalid response format');
			}
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
			toast.error(`Search failed: ${errorMessage}`);
			searchState.results = [];
		} finally {
			searchState.isSearching = false;
		}
	}

	// Type-safe quick actions with better error handling
	type QuickActionType = 'upload_pdf' | 'roll_dice' | 'generate_npc' | 'new_session' | 'view_notes';
	
	const quickActionHandlers: Record<QuickActionType, () => Promise<void> | void> = {
		upload_pdf: () => goto('/upload'),
		roll_dice: async () => {
			const result = await sessionStore.callTool('roll_dice', { 
				dice_notation: '1d20' 
			});
			if (result?.data?.result) {
				toast.success(`🎲 Rolled 1d20: ${result.data.result}`);
			}
		},
		generate_npc: async () => {
			toast.info('🧙‍♂️ Generating NPC...');
			const result = await sessionStore.callTool('generate_npc', {
				level: 5,
				type: 'merchant'
			});
			if (result?.data?.name) {
				toast.success(`✨ Generated NPC: ${result.data.name}`);
			}
		},
		new_session: async () => {
			toast.info('🎮 Starting new session...');
			// Implementation would go here
		},
		view_notes: () => goto('/notes')
	};

	async function executeQuickAction(action: string): Promise<void> {
		if (!Object.hasOwnProperty.call(quickActionHandlers, action)) {
			toast.info(`Action "${action}" is not yet implemented`);
			return;
		}

		try {
			await quickActionHandlers[action as QuickActionType]();
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : 'Action failed';
			toast.error(`❌ ${errorMessage}`);
		}
	}

	// Enhanced derived stats with better type safety
	const dashboardStats = $derived(() => {
		const user = sessionStore.user;
		const currentCampaign = sessionStore.currentCampaign;
		const gameSession = sessionStore.currentGameSession;

		return {
			campaigns: user?.campaigns?.length ?? 0,
			isConnected: sessionStore.isConnected,
			currentCampaign: currentCampaign?.name ?? 'None',
			sessionStatus: gameSession?.active ? 'Active' : 'Inactive'
		} as const;
	});

	// Keyboard shortcuts for accessibility
	function handleKeydown(event: KeyboardEvent): void {
		if (event.ctrlKey || event.metaKey) {
			switch (event.key) {
				case 'k':
					event.preventDefault();
					document.querySelector<HTMLInputElement>('[data-search-input]')?.focus();
					break;
				case '/':
					event.preventDefault();
					executeQuickAction('upload_pdf');
					break;
			}
		}
	}
</script>

<svelte:head>
	<title>Dashboard - TTRPG Assistant</title>
</svelte:head>

<svelte:window on:keydown={handleKeydown} />

<div class="min-h-screen bg-background" role="main">
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
		<!-- Enhanced Status Bar with better accessibility -->
		<div class="mb-6 rounded-lg bg-muted p-4" role="status" aria-label="Dashboard status">
			<div class="flex flex-wrap items-center gap-6 text-sm">
				<div class="flex items-center gap-2">
					<div class="h-2 w-2 rounded-full {dashboardStats().isConnected ? 'bg-green-500' : 'bg-red-500'}" aria-hidden="true"></div>
					<span class="text-muted-foreground">Connection:</span>
					<span class="font-medium">
						{dashboardStats().isConnected ? 'Connected' : 'Disconnected'}
					</span>
				</div>
				<div>
					<span class="text-muted-foreground">Campaign:</span>
					<span class="ml-2 font-medium">{dashboardStats().currentCampaign}</span>
				</div>
				<div>
					<span class="text-muted-foreground">Session:</span>
					<span class="ml-2 font-medium">{dashboardStats().sessionStatus}</span>
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
				<form on:submit|preventDefault={handleSearch} class="flex gap-2">
					<input
						type="text"
						bind:value={searchState.query}
						data-search-input
						placeholder="Search for rules, spells, monsters... (Ctrl+K)"
						class="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
						disabled={searchState.isSearching}
						aria-label="Search query"
						autocomplete="off"
					/>
					<Button 
						type="submit" 
						disabled={searchState.isSearching || !searchState.query.trim()}
						aria-label={searchState.isSearching ? 'Searching' : 'Search'}
					>
						{searchState.isSearching ? 'Searching...' : 'Search'}
					</Button>
				</form>
				
				{#if searchState.results.length > 0}
					<div class="mt-4 space-y-2" role="region" aria-label="Search results">
						{#each searchState.results as result, index}
							<article 
								class="rounded border p-3 transition-colors hover:bg-muted/50" 
								tabindex="0"
								aria-label="Search result {index + 1}"
							>
								<h3 class="font-medium">{result.title ?? 'Untitled Result'}</h3>
								<p class="text-sm text-muted-foreground mt-1">{result.content}</p>
								{#if result.source}
									<footer class="mt-2 text-xs text-muted-foreground">
										Source: {result.source}{result.page ? ` • Page ${result.page}` : ''}
									</footer>
								{/if}
							</article>
						{/each}
					</div>
				{/if}
			</CardContent>
		</Card>

		<!-- Enhanced Quick Actions with better accessibility -->
		<section class="mb-6" aria-label="Quick actions">
			<div class="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
				{#each [
					{ id: 'upload_pdf', title: 'Upload PDF', desc: 'Process game documents', icon: Upload, shortcut: 'Ctrl+/' },
					{ id: 'new_session', title: 'New Session', desc: 'Start a new game session', icon: Plus },
					{ id: 'roll_dice', title: 'Quick Roll', desc: 'Roll dice quickly', icon: Brain },
					{ id: 'generate_npc', title: 'Generate NPC', desc: 'Create a new NPC', icon: Users },
					{ id: 'view_notes', title: 'Session Notes', desc: 'View recent notes', icon: BookOpen }
				] as action}
					<Card 
						class="group cursor-pointer transition-all hover:bg-muted/50 hover:scale-[1.02] focus-within:ring-2 focus-within:ring-ring" 
						onclick={() => executeQuickAction(action.id)}
						onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && executeQuickAction(action.id)}
						tabindex="0"
						role="button"
						aria-label={`${action.title}: ${action.desc}${action.shortcut ? ` (${action.shortcut})` : ''}`}
					>
						<CardHeader class="flex flex-row items-center justify-between space-y-0 pb-2">
							<CardTitle class="text-sm font-medium group-hover:text-primary transition-colors">
								{action.title}
							</CardTitle>
							<action.icon class="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" aria-hidden="true" />
						</CardHeader>
						<CardContent>
							<p class="text-xs text-muted-foreground">{action.desc}</p>
							{#if action.shortcut}
								<p class="text-[10px] text-muted-foreground/70 mt-1">{action.shortcut}</p>
							{/if}
						</CardContent>
					</Card>
				{/each}
			</div>
		</section>

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
					{#if recentMessages.length > 0}
						<ul class="space-y-3" role="list" aria-label="Recent activities">
							{#each recentMessages as message, index}
								<li class="flex items-start gap-3 text-sm">
									<div class="h-2 w-2 rounded-full bg-primary mt-2 flex-shrink-0" aria-hidden="true"></div>
									<div class="flex-1 min-w-0">
										<div class="flex items-center justify-between">
											<span class="font-medium truncate">{message.type ?? 'Activity'}</span>
											<time 
												class="text-xs text-muted-foreground flex-shrink-0 ml-2"
												datetime={new Date(message.timestamp).toISOString()}
											>
												{new Date(message.timestamp).toLocaleTimeString()}
											</time>
										</div>
										{#if message.content}
											<p class="text-xs text-muted-foreground mt-1 truncate">{message.content}</p>
										{/if}
									</div>
								</li>
							{/each}
						</ul>
					{:else}
						<div class="text-center py-8">
							<div class="text-muted-foreground mb-2">📋</div>
							<p class="text-sm text-muted-foreground">No recent activity</p>
							<p class="text-xs text-muted-foreground mt-1">Your recent actions will appear here</p>
						</div>
					{/if}
				</CardContent>
			</Card>
		</div>
	</div>
</div>