<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card';
	import { Tabs, TabsContent, TabsList, TabsTrigger } from '$lib/components/ui/tabs';
	import { BookOpen, Users, Brain, Shield, Dice1 } from 'lucide-svelte';
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
				<Card>
					<CardHeader>
						<BookOpen class="h-10 w-10 text-primary mb-2" />
						<CardTitle>Rule & Content Search</CardTitle>
						<CardDescription>
							Quickly find rules, spells, and monsters from your TTRPG rulebooks
						</CardDescription>
					</CardHeader>
					<CardContent>
						<ul class="space-y-2 text-sm text-muted-foreground">
							<li>• Semantic search across all sources</li>
							<li>• Page references and citations</li>
							<li>• Multi-rulebook support</li>
						</ul>
					</CardContent>
				</Card>

				<Card>
					<CardHeader>
						<Users class="h-10 w-10 text-primary mb-2" />
						<CardTitle>Campaign Management</CardTitle>
						<CardDescription>
							Store and retrieve campaign-specific data with ease
						</CardDescription>
					</CardHeader>
					<CardContent>
						<ul class="space-y-2 text-sm text-muted-foreground">
							<li>• NPCs and characters</li>
							<li>• Locations and plot points</li>
							<li>• Version history</li>
						</ul>
					</CardContent>
				</Card>

				<Card>
					<CardHeader>
						<Brain class="h-10 w-10 text-primary mb-2" />
						<CardTitle>AI-Powered Generation</CardTitle>
						<CardDescription>
							Generate characters, NPCs, and backstories that fit your world
						</CardDescription>
					</CardHeader>
					<CardContent>
						<ul class="space-y-2 text-sm text-muted-foreground">
							<li>• System-appropriate personalities</li>
							<li>• Contextual backstories</li>
							<li>• Balanced stat generation</li>
						</ul>
					</CardContent>
				</Card>

				<Card>
					<CardHeader>
						<Shield class="h-10 w-10 text-primary mb-2" />
						<CardTitle>Session Tracking</CardTitle>
						<CardDescription>
							Manage your game sessions with powerful tools
						</CardDescription>
					</CardHeader>
					<CardContent>
						<ul class="space-y-2 text-sm text-muted-foreground">
							<li>• Initiative tracking</li>
							<li>• Monster health management</li>
							<li>• Session notes and history</li>
						</ul>
					</CardContent>
				</Card>

				<Card>
					<CardHeader>
						<Dice1 class="h-10 w-10 text-primary mb-2" />
						<CardTitle>Real-time Collaboration</CardTitle>
						<CardDescription>
							Play together with your party, no matter where you are
						</CardDescription>
					</CardHeader>
					<CardContent>
						<ul class="space-y-2 text-sm text-muted-foreground">
							<li>• Shared game state</li>
							<li>• Live dice rolling</li>
							<li>• Player presence indicators</li>
						</ul>
					</CardContent>
				</Card>

				<Card>
					<CardHeader>
						<Brain class="h-10 w-10 text-primary mb-2" />
						<CardTitle>Multiple AI Providers</CardTitle>
						<CardDescription>
							Use your preferred AI service with automatic fallback
						</CardDescription>
					</CardHeader>
					<CardContent>
						<ul class="space-y-2 text-sm text-muted-foreground">
							<li>• Anthropic Claude</li>
							<li>• OpenAI GPT</li>
							<li>• Google Gemini</li>
						</ul>
					</CardContent>
				</Card>
			</div>
		</div>
	</section>

	<!-- How It Works -->
	<section class="py-20 px-4 bg-muted/30">
		<div class="container mx-auto max-w-6xl">
			<h2 class="text-3xl md:text-4xl font-bold text-center mb-12">How It Works</h2>
			
			<Tabs value="setup" class="max-w-4xl mx-auto">
				<TabsList class="grid w-full grid-cols-3">
					<TabsTrigger value="setup">Setup</TabsTrigger>
					<TabsTrigger value="play">Play</TabsTrigger>
					<TabsTrigger value="manage">Manage</TabsTrigger>
				</TabsList>
				
				<TabsContent value="setup" class="space-y-4 mt-6">
					<Card>
						<CardHeader>
							<CardTitle>1. Upload Your Rulebooks</CardTitle>
							<CardDescription>
								Import your TTRPG rulebooks and source materials in PDF format
							</CardDescription>
						</CardHeader>
					</Card>
					<Card>
						<CardHeader>
							<CardTitle>2. Configure AI Provider</CardTitle>
							<CardDescription>
								Connect your preferred AI service with your API credentials
							</CardDescription>
						</CardHeader>
					</Card>
					<Card>
						<CardHeader>
							<CardTitle>3. Create Campaign</CardTitle>
							<CardDescription>
								Set up your campaign with characters, NPCs, and world details
							</CardDescription>
						</CardHeader>
					</Card>
				</TabsContent>
				
				<TabsContent value="play" class="space-y-4 mt-6">
					<Card>
						<CardHeader>
							<CardTitle>1. Start Session</CardTitle>
							<CardDescription>
								Create a new session or continue an existing one
							</CardDescription>
						</CardHeader>
					</Card>
					<Card>
						<CardHeader>
							<CardTitle>2. Quick Search</CardTitle>
							<CardDescription>
								Find rules, spells, and monsters instantly during gameplay
							</CardDescription>
						</CardHeader>
					</Card>
					<Card>
						<CardHeader>
							<CardTitle>3. Track Everything</CardTitle>
							<CardDescription>
								Manage initiative, health, and notes automatically
							</CardDescription>
						</CardHeader>
					</Card>
				</TabsContent>
				
				<TabsContent value="manage" class="space-y-4 mt-6">
					<Card>
						<CardHeader>
							<CardTitle>1. Review History</CardTitle>
							<CardDescription>
								Access complete session history and campaign timeline
							</CardDescription>
						</CardHeader>
					</Card>
					<Card>
						<CardHeader>
							<CardTitle>2. Update Content</CardTitle>
							<CardDescription>
								Add new NPCs, locations, and plot developments
							</CardDescription>
						</CardHeader>
					</Card>
					<Card>
						<CardHeader>
							<CardTitle>3. Share & Collaborate</CardTitle>
							<CardDescription>
								Invite players to join and contribute to the campaign
							</CardDescription>
						</CardHeader>
					</Card>
				</TabsContent>
			</Tabs>
		</div>
	</section>
</div>