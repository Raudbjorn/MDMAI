<script lang="ts">
	import { onMount } from 'svelte';
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import {
		CollaborativeCanvas,
		PresenceIndicator,
		ActivityFeed,
		ChatPanel,
		TurnManager
	} from '$lib/components/collaboration';
	
	// Demo user data
	const demoUserId = `user-${Math.random().toString(36).substr(2, 9)}`;
	const demoUsername = `Player${Math.floor(Math.random() * 1000)}`;
	
	let isConnected = $state(false);
	let currentRoom: any = $state(null);
	let showCanvas = $state(true);
	let showPresence = $state(true);
	let showActivity = $state(true);
	let showChat = $state(true);
	
	onMount(() => {
		// Connect to collaboration service
		const init = async () => {
			try {
				await collaborationStore.connect(demoUserId);
				
				// Create or join demo room
				currentRoom = await collaborationStore.createRoom(
					'Demo Session',
					'demo-campaign',
					{
						max_participants: 20,
						allow_spectators: true,
						enable_voice: false,
						enable_video: false,
						auto_save: true,
						save_interval: 30
					}
				);
				
				isConnected = true;
			} catch (error) {
				console.error('Failed to initialize collaboration:', error);
				isConnected = false;
			}
			
			// Simulate some activity for demo
			setTimeout(() => {
				collaborationStore.sendChatMessage('Welcome to the real-time demo!', 'system');
			}, 1000);
		};
		
		init();
		
		return () => {
			collaborationStore.leaveRoom();
			collaborationStore.disconnect();
		};
	});
	
	function toggleFeature(feature: string) {
		switch (feature) {
			case 'canvas':
				showCanvas = !showCanvas;
				break;
			case 'presence':
				showPresence = !showPresence;
				break;
			case 'activity':
				showActivity = !showActivity;
				break;
			case 'chat':
				showChat = !showChat;
				break;
		}
	}
	
	async function simulateActivity() {
		// Simulate various activities
		const activities = [
			() => collaborationStore.rollDice('2d6+3', 'Attack roll'),
			() => collaborationStore.rollDice('1d20', 'Perception check'),
			() => collaborationStore.sendChatMessage('Great roll!'),
			() => collaborationStore.nextTurn(),
			() => collaborationStore.updateState({
				path: ['shared_notes'],
				value: 'Updated notes: The party enters the dungeon...',
				operation: 'set',
				version: Date.now(),
				previous_version: Date.now() - 1
			})
		];
		
		const randomActivity = activities[Math.floor(Math.random() * activities.length)];
		await randomActivity();
	}
</script>

<div class="realtime-demo">
	<div class="demo-header">
		<h1>Real-time Features Demo</h1>
		
		<div class="connection-info">
			{#if isConnected}
				<span class="status connected">Connected</span>
				{#if currentRoom}
					<span class="room-info">Room: {currentRoom.name} ({currentRoom.id.slice(0, 8)}...)</span>
				{/if}
			{:else}
				<span class="status connecting">Connecting...</span>
			{/if}
		</div>
		
		<div class="feature-toggles">
			<button 
				class="toggle-btn"
				class:active={showCanvas}
				onclick={() => toggleFeature('canvas')}
			>
				Canvas
			</button>
			<button 
				class="toggle-btn"
				class:active={showPresence}
				onclick={() => toggleFeature('presence')}
			>
				Presence
			</button>
			<button 
				class="toggle-btn"
				class:active={showActivity}
				onclick={() => toggleFeature('activity')}
			>
				Activity
			</button>
			<button 
				class="toggle-btn"
				class:active={showChat}
				onclick={() => toggleFeature('chat')}
			>
				Chat
			</button>
		</div>
		
		<button class="simulate-btn" onclick={simulateActivity}>
			Simulate Activity
		</button>
	</div>
	
	{#if isConnected && currentRoom}
		<div class="demo-content">
			<div class="main-area">
				{#if showCanvas}
					<div class="feature-section">
						<h2>Collaborative Canvas</h2>
						<p class="feature-description">
							Draw, annotate, and collaborate in real-time. All participants can see each other's cursors and drawings.
						</p>
						<CollaborativeCanvas
							roomId={currentRoom.id}
							width={800}
							height={600}
							enableDrawing={true}
							enableAnnotations={true}
							enableGrid={true}
							gridSize={20}
						/>
					</div>
				{/if}
				
				<div class="bottom-panels">
					{#if showChat}
						<div class="feature-section chat-section">
							<h3>Chat</h3>
							<ChatPanel />
						</div>
					{/if}
					
					<div class="feature-section turn-section">
						<h3>Turn Manager</h3>
						<TurnManager />
					</div>
				</div>
			</div>
			
			<div class="sidebar">
				{#if showPresence}
					<div class="feature-section">
						<h3>Presence Indicators</h3>
						<p class="feature-description">
							See who's online and what they're doing in real-time.
						</p>
						<PresenceIndicator
							showCursors={true}
							showStatus={true}
							showActivity={true}
							compactMode={false}
						/>
						
						<div class="compact-example">
							<h4>Compact Mode:</h4>
							<PresenceIndicator
								showCursors={false}
								showStatus={true}
								showActivity={false}
								compactMode={true}
							/>
						</div>
					</div>
				{/if}
				
				{#if showActivity}
					<div class="feature-section">
						<h3>Activity Feed</h3>
						<p class="feature-description">
							Live updates of all session activities using Server-Sent Events.
						</p>
						<ActivityFeed
							roomId={currentRoom.id}
							maxItems={50}
							autoScroll={true}
							showFilters={true}
							compactMode={false}
						/>
					</div>
				{/if}
			</div>
		</div>
		
		<div class="demo-info">
			<h3>Features Demonstrated:</h3>
			<ul>
				<li>✅ WebSocket client with native SvelteKit support and auto-reconnection</li>
				<li>✅ Server-Sent Events (SSE) for unidirectional updates with auto-reconnect</li>
				<li>✅ Collaborative canvas with real-time drawing synchronization</li>
				<li>✅ Presence indicators showing online status and activity</li>
				<li>✅ Shared cursor tracking across all participants</li>
				<li>✅ Activity feed with live updates via SSE</li>
				<li>✅ Chat messaging with WebSocket</li>
				<li>✅ Turn management for gameplay</li>
			</ul>
			
			<h3>Technical Implementation:</h3>
			<ul>
				<li>🔧 Enhanced WebSocket client with exponential backoff reconnection</li>
				<li>🔧 SSE client with automatic reconnection and event filtering</li>
				<li>🔧 Svelte 5 runes for reactive state management</li>
				<li>🔧 TypeScript for full type safety</li>
				<li>🔧 Optimistic UI updates with conflict resolution</li>
				<li>🔧 Message queuing for offline resilience</li>
				<li>🔧 Heartbeat mechanism for connection monitoring</li>
				<li>🔧 Latency tracking for performance monitoring</li>
			</ul>
		</div>
	{:else}
		<div class="loading-state">
			<div class="spinner"></div>
			<p>Connecting to real-time services...</p>
		</div>
	{/if}
</div>

<style>
	.realtime-demo {
		min-height: 100vh;
		background: #f5f5f5;
		padding: 1rem;
	}
	
	.demo-header {
		background: white;
		padding: 1.5rem;
		border-radius: 8px;
		margin-bottom: 1rem;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
		display: flex;
		align-items: center;
		gap: 2rem;
		flex-wrap: wrap;
	}
	
	.demo-header h1 {
		margin: 0;
		font-size: 1.5rem;
		color: #333;
	}
	
	.connection-info {
		display: flex;
		align-items: center;
		gap: 1rem;
	}
	
	.status {
		padding: 0.375rem 0.75rem;
		border-radius: 16px;
		font-size: 0.875rem;
		font-weight: 500;
	}
	
	.status.connected {
		background: #e8f5e9;
		color: #2e7d32;
	}
	
	.status.connecting {
		background: #fff8e1;
		color: #f57c00;
	}
	
	.room-info {
		font-size: 0.875rem;
		color: #666;
	}
	
	.feature-toggles {
		display: flex;
		gap: 0.5rem;
		margin-left: auto;
	}
	
	.toggle-btn {
		padding: 0.5rem 1rem;
		background: white;
		border: 1px solid #d0d0d0;
		border-radius: 4px;
		cursor: pointer;
		font-size: 0.875rem;
		transition: all 0.2s;
	}
	
	.toggle-btn:hover {
		background: #f0f0f0;
	}
	
	.toggle-btn.active {
		background: #2196F3;
		color: white;
		border-color: #1976D2;
	}
	
	.simulate-btn {
		padding: 0.5rem 1rem;
		background: #4CAF50;
		color: white;
		border: none;
		border-radius: 4px;
		cursor: pointer;
		font-size: 0.875rem;
		font-weight: 500;
		transition: background 0.2s;
	}
	
	.simulate-btn:hover {
		background: #45a049;
	}
	
	.demo-content {
		display: grid;
		grid-template-columns: 1fr 400px;
		gap: 1rem;
		margin-bottom: 1rem;
	}
	
	.main-area {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}
	
	.bottom-panels {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1rem;
	}
	
	.sidebar {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}
	
	.feature-section {
		background: white;
		padding: 1.5rem;
		border-radius: 8px;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
	}
	
	.feature-section h2,
	.feature-section h3 {
		margin-top: 0;
		margin-bottom: 0.5rem;
		color: #333;
	}
	
	.feature-description {
		font-size: 0.875rem;
		color: #666;
		margin-bottom: 1rem;
	}
	
	.compact-example {
		margin-top: 1rem;
		padding-top: 1rem;
		border-top: 1px solid #e0e0e0;
	}
	
	.compact-example h4 {
		margin-top: 0;
		margin-bottom: 0.5rem;
		font-size: 0.875rem;
		color: #666;
	}
	
	.chat-section {
		max-height: 400px;
		overflow: hidden;
	}
	
	.turn-section {
		max-height: 400px;
		overflow: auto;
	}
	
	.demo-info {
		background: white;
		padding: 1.5rem;
		border-radius: 8px;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
	}
	
	.demo-info h3 {
		margin-top: 0;
		margin-bottom: 1rem;
		color: #333;
	}
	
	.demo-info ul {
		margin: 0 0 1.5rem 0;
		padding-left: 1.5rem;
	}
	
	.demo-info li {
		margin-bottom: 0.5rem;
		color: #666;
	}
	
	.loading-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 400px;
		background: white;
		border-radius: 8px;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
	}
	
	.spinner {
		width: 48px;
		height: 48px;
		border: 4px solid #f0f0f0;
		border-top-color: #2196F3;
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}
	
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
	
	.loading-state p {
		margin-top: 1rem;
		color: #666;
	}
	
	@media (max-width: 1200px) {
		.demo-content {
			grid-template-columns: 1fr;
		}
		
		.bottom-panels {
			grid-template-columns: 1fr;
		}
	}
</style>