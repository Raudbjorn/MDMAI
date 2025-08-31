<script lang="ts">
	import { Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Tabs, TabsContent, TabsList, TabsTrigger } from '$lib/components/ui';
	import { BookOpen, Users, Brain, Shield, Dice1 } from 'lucide-svelte';

	const features = [
		{ icon: BookOpen, title: 'Rule & Content Search', desc: 'Quickly find rules, spells, and monsters from your TTRPG rulebooks', items: ['Semantic search across all sources', 'Page references and citations', 'Multi-rulebook support'] },
		{ icon: Users, title: 'Campaign Management', desc: 'Store and retrieve campaign-specific data with ease', items: ['NPCs and characters', 'Locations and plot points', 'Version history'] },
		{ icon: Brain, title: 'AI-Powered Generation', desc: 'Generate characters, NPCs, and backstories that fit your world', items: ['System-appropriate personalities', 'Contextual backstories', 'Balanced stat generation'] },
		{ icon: Shield, title: 'Session Tracking', desc: 'Manage your game sessions with powerful tools', items: ['Initiative tracking', 'Monster health management', 'Session notes and history'] },
		{ icon: Dice1, title: 'Real-time Collaboration', desc: 'Play together with your party, no matter where you are', items: ['Shared game state', 'Live dice rolling', 'Player presence indicators'] },
		{ icon: Brain, title: 'Multiple AI Providers', desc: 'Use your preferred AI service with automatic fallback', items: ['Anthropic Claude', 'OpenAI GPT', 'Google Gemini'] }
	];

	const steps = {
		setup: [{ title: '1. Upload Your Rulebooks', desc: 'Import your TTRPG rulebooks and source materials in PDF format' }, { title: '2. Configure AI Provider', desc: 'Connect your preferred AI service with your API credentials' }, { title: '3. Create Campaign', desc: 'Set up your campaign with characters, NPCs, and world details' }],
		play: [{ title: '1. Start Session', desc: 'Create a new session or continue an existing one' }, { title: '2. Quick Search', desc: 'Find rules, spells, and monsters instantly during gameplay' }, { title: '3. Track Everything', desc: 'Manage initiative, health, and notes automatically' }],
		manage: [{ title: '1. Review History', desc: 'Access complete session history and campaign timeline' }, { title: '2. Update Content', desc: 'Add new NPCs, locations, and plot developments' }, { title: '3. Share & Collaborate', desc: 'Invite players to join and contribute to the campaign' }]
	};
</script>

<svelte:head>
	<title>TTRPG Assistant - MCP Server</title>
	<meta name="description" content="A comprehensive assistant for Tabletop Role-Playing Games" />
</svelte:head>

<div class="min-h-screen">
	<!-- Hero Section -->
	<section class="relative bg-gradient-to-b from-primary/10 to-background py-20 px-4">
		<div class="container mx-auto max-w-6xl">
			<div class="text-center space-y-6">
				<h1 class="text-5xl md:text-7xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
					TTRPG Assistant
				</h1>
				<p class="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto">
					Your comprehensive side-car assistant for Dungeon Masters and Game Runners
				</p>
				<div class="flex gap-4 justify-center pt-4">
					<Button href="/dashboard" size="lg">
						<Dice1 class="mr-2 h-5 w-5" />
						Start Session
					</Button>
					<Button href="/campaigns" variant="outline" size="lg">
						<BookOpen class="mr-2 h-5 w-5" />
						Manage Campaigns
					</Button>
				</div>
			</div>
		</div>
	</section>

	<!-- Features Section -->
	<section class="py-20 px-4">
		<div class="container mx-auto max-w-6xl">
			<h2 class="text-3xl md:text-4xl font-bold text-center mb-12">Core Features</h2>
			<div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
				{#each features as { icon: Icon, title, desc, items }}
					<Card>
						<CardHeader>
							<Icon class="h-10 w-10 text-primary mb-2" />
							<CardTitle>{title}</CardTitle>
							<CardDescription>{desc}</CardDescription>
						</CardHeader>
						<CardContent>
							<ul class="space-y-2 text-sm text-muted-foreground">
								{#each items as item}
									<li>• {item}</li>
								{/each}
							</ul>
						</CardContent>
					</Card>
				{/each}
			</div>
		</div>
	</section>

	<!-- How It Works -->
	<section class="py-20 px-4 bg-muted/30">
		<div class="container mx-auto max-w-6xl">
			<h2 class="text-3xl md:text-4xl font-bold text-center mb-12">How It Works</h2>
			
			<Tabs defaultValue="setup" class="max-w-4xl mx-auto">
				<TabsList class="grid w-full grid-cols-3">
					<TabsTrigger value="setup">Setup</TabsTrigger>
					<TabsTrigger value="play">Play</TabsTrigger>
					<TabsTrigger value="manage">Manage</TabsTrigger>
				</TabsList>
				
				{#each Object.entries(steps) as [tab, tabSteps]}
					<TabsContent value={tab} class="space-y-4 mt-6">
						{#each tabSteps as { title, desc }}
							<Card>
								<CardHeader>
									<CardTitle>{title}</CardTitle>
									<CardDescription>{desc}</CardDescription>
								</CardHeader>
							</Card>
						{/each}
					</TabsContent>
				{/each}
			</Tabs>
		</div>
	</section>
</div>