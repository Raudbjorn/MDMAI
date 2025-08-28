<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { EnhancedSSEClient } from '$lib/realtime/sse-client';
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { DiceRoll, ChatMessage } from '$lib/types/collaboration';
	
	interface Props {
		roomId?: string;
		maxItems?: number;
		autoScroll?: boolean;
		showFilters?: boolean;
		compactMode?: boolean;
	}
	
	let {
		roomId,
		maxItems = 100,
		autoScroll = true,
		showFilters = true,
		compactMode = false
	}: Props = $props();
	
	interface ActivityItem {
		id: string;
		type: 'join' | 'leave' | 'chat' | 'dice' | 'turn' | 'initiative' | 'state' | 'system';
		timestamp: number;
		userId: string;
		userName: string;
		content: any;
		icon?: string;
		color?: string;
	}
	
	let activities = $state<ActivityItem[]>([]);
	let filteredActivities = $derived(
		filters.size === 0 
			? activities 
			: activities.filter(a => filters.has(a.type))
	);
	
	let filters = $state<Set<ActivityItem['type']>>(new Set());
	let sseClient: EnhancedSSEClient | null = null;
	let feedContainer: HTMLElement;
	
	// Activity type configurations
	const activityConfig: Record<ActivityItem['type'], { icon: string; color: string; label: string }> = {
		join: { icon: '👋', color: '#4CAF50', label: 'Joined' },
		leave: { icon: '👋', color: '#F44336', label: 'Left' },
		chat: { icon: '💬', color: '#2196F3', label: 'Chat' },
		dice: { icon: '🎲', color: '#9C27B0', label: 'Dice Roll' },
		turn: { icon: '⏭️', color: '#FF9800', label: 'Turn' },
		initiative: { icon: '⚔️', color: '#795548', label: 'Initiative' },
		state: { icon: '📝', color: '#607D8B', label: 'Update' },
		system: { icon: 'ℹ️', color: '#9E9E9E', label: 'System' }
	};
	
	onMount(() => {
		// Connect to SSE endpoint for real-time updates
		const baseUrl = typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000';
		const sseUrl = roomId 
			? `${baseUrl}/api/collaboration/activity/${roomId}`
			: `${baseUrl}/api/collaboration/activity`;
		
		sseClient = new EnhancedSSEClient({
			url: sseUrl,
			reconnectDelay: 1000,
			maxReconnectDelay: 10000
		});
		
		// Handle different event types
		sseClient.onMessage('activity', (message) => {
			addActivity(message.data);
		});
		
		// Subscribe to collaboration events
		const unsubscribes = [
			collaborationStore.onMessage('participant_joined', (msg) => {
				addActivity({
					id: `join-${Date.now()}`,
					type: 'join',
					timestamp: msg.timestamp,
					userId: msg.sender_id,
					userName: msg.data.username || 'Unknown',
					content: `joined the session`
				});
			}),
			
			collaborationStore.onMessage('participant_left', (msg) => {
				addActivity({
					id: `leave-${Date.now()}`,
					type: 'leave',
					timestamp: msg.timestamp,
					userId: msg.sender_id,
					userName: msg.data.username || 'Unknown',
					content: `left the session`
				});
			}),
			
			collaborationStore.onMessage('chat_message', (msg) => {
				const chatMsg = msg.data as ChatMessage;
				addActivity({
					id: chatMsg.id,
					type: 'chat',
					timestamp: msg.timestamp,
					userId: chatMsg.sender_id,
					userName: chatMsg.sender_name,
					content: chatMsg.content
				});
			}),
			
			collaborationStore.onMessage('dice_roll', (msg) => {
				const roll = msg.data as DiceRoll;
				addActivity({
					id: roll.id,
					type: 'dice',
					timestamp: msg.timestamp,
					userId: roll.player_id,
					userName: roll.player_name,
					content: {
						expression: roll.expression,
						results: roll.results,
						total: roll.total,
						purpose: roll.purpose
					}
				});
			}),
			
			collaborationStore.onMessage('turn_changed', (msg) => {
				addActivity({
					id: `turn-${Date.now()}`,
					type: 'turn',
					timestamp: msg.timestamp,
					userId: msg.sender_id,
					userName: 'System',
					content: `Turn ${msg.data.turn + 1}, Round ${msg.data.round}`
				});
			}),
			
			collaborationStore.onMessage('initiative_update', (msg) => {
				addActivity({
					id: `init-${Date.now()}`,
					type: 'initiative',
					timestamp: msg.timestamp,
					userId: msg.sender_id,
					userName: msg.data.updatedBy || 'System',
					content: 'Updated initiative order'
				});
			}),
			
			collaborationStore.onMessage('state_update', (msg) => {
				addActivity({
					id: `state-${Date.now()}`,
					type: 'state',
					timestamp: msg.timestamp,
					userId: msg.sender_id,
					userName: msg.data.updatedBy || 'Unknown',
					content: `Updated ${msg.data.path.join(' > ')}`
				});
			})
		];
		
		return () => {
			sseClient?.destroy();
			unsubscribes.forEach(unsub => unsub());
		};
	});
	
	function addActivity(activity: ActivityItem) {
		activities = [activity, ...activities].slice(0, maxItems);
		
		// Auto-scroll to latest
		if (autoScroll && feedContainer) {
			setTimeout(() => {
				feedContainer.scrollTop = 0;
			}, 0);
		}
	}
	
	function toggleFilter(type: ActivityItem['type']) {
		if (filters.has(type)) {
			filters.delete(type);
		} else {
			filters.add(type);
		}
		filters = new Set(filters);
	}
	
	function clearFilters() {
		filters = new Set();
	}
	
	function formatTime(timestamp: number): string {
		const date = new Date(timestamp);
		const now = new Date();
		const diff = now.getTime() - date.getTime();
		
		if (diff < 60000) return 'Just now';
		if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
		if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
		
		return date.toLocaleTimeString();
	}
	
	function formatDiceRoll(content: any): string {
		if (typeof content === 'object' && content.expression) {
			const purpose = content.purpose ? ` for ${content.purpose}` : '';
			const results = content.results?.join(', ') || '';
			return `rolled ${content.expression}${purpose}: [${results}] = ${content.total}`;
		}
		return 'rolled dice';
	}
</script>

<div class="activity-feed" class:compact={compactMode}>
	{#if !compactMode}
		<div class="feed-header">
			<h3>Activity Feed</h3>
			{#if sseClient?.isConnected}
				<div class="connection-status connected" title="Connected">
					<span class="status-dot"></span>
					Live
				</div>
			{:else if sseClient?.isReconnecting}
				<div class="connection-status reconnecting" title="Reconnecting...">
					<span class="status-dot"></span>
					Reconnecting
				</div>
			{:else}
				<div class="connection-status disconnected" title="Disconnected">
					<span class="status-dot"></span>
					Offline
				</div>
			{/if}
		</div>
		
		{#if showFilters}
			<div class="feed-filters">
				<button 
					class="filter-btn"
					class:active={filters.size === 0}
					onclick={clearFilters}
				>
					All
				</button>
				{#each Object.entries(activityConfig) as [type, config]}
					<button 
						class="filter-btn"
						class:active={filters.has(type)}
						onclick={() => toggleFilter(type)}
						title={config.label}
					>
						<span class="filter-icon">{config.icon}</span>
						<span class="filter-label">{config.label}</span>
					</button>
				{/each}
			</div>
		{/if}
		
		<div class="feed-container" bind:this={feedContainer}>
			{#if filteredActivities.length === 0}
				<div class="empty-state">
					<p>No activity yet</p>
					<small>Activities will appear here as they happen</small>
				</div>
			{:else}
				{#each filteredActivities as activity}
					<div class="activity-item">
						<div 
							class="activity-icon"
							style="background-color: {activityConfig[activity.type].color}"
						>
							{activityConfig[activity.type].icon}
						</div>
						
						<div class="activity-content">
							<div class="activity-header">
								<span class="activity-user">{activity.userName}</span>
								<span class="activity-time">{formatTime(activity.timestamp)}</span>
							</div>
							
							<div class="activity-message">
								{#if activity.type === 'dice'}
									{formatDiceRoll(activity.content)}
								{:else if activity.type === 'chat'}
									<span class="chat-content">{activity.content}</span>
								{:else}
									{activity.content}
								{/if}
							</div>
						</div>
					</div>
				{/each}
			{/if}
		</div>
	{:else}
		<!-- Compact mode -->
		<div class="feed-compact">
			<div class="compact-header">
				<span class="compact-title">Activity</span>
				{#if sseClient?.isConnected}
					<span class="compact-status connected"></span>
				{:else}
					<span class="compact-status disconnected"></span>
				{/if}
			</div>
			
			<div class="compact-items">
				{#each filteredActivities.slice(0, 3) as activity}
					<div class="compact-item">
						<span class="compact-icon">{activityConfig[activity.type].icon}</span>
						<span class="compact-text">
							{activity.userName}: 
							{#if activity.type === 'dice'}
								{activity.content.total}
							{:else}
								{activity.content}
							{/if}
						</span>
						<span class="compact-time">{formatTime(activity.timestamp)}</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>

<style>
	.activity-feed {
		background: white;
		border-radius: 8px;
		border: 1px solid #e0e0e0;
		display: flex;
		flex-direction: column;
		height: 100%;
		overflow: hidden;
	}
	
	.feed-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 1rem;
		background: #f5f5f5;
		border-bottom: 1px solid #e0e0e0;
	}
	
	.feed-header h3 {
		margin: 0;
		font-size: 1rem;
		font-weight: 600;
		color: #333;
	}
	
	.connection-status {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.875rem;
		padding: 0.25rem 0.75rem;
		border-radius: 12px;
		font-weight: 500;
	}
	
	.status-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		animation: pulse 2s infinite;
	}
	
	.connection-status.connected {
		background: #e8f5e9;
		color: #2e7d32;
	}
	
	.connection-status.connected .status-dot {
		background: #4caf50;
	}
	
	.connection-status.reconnecting {
		background: #fff8e1;
		color: #f57c00;
	}
	
	.connection-status.reconnecting .status-dot {
		background: #ff9800;
		animation: pulse 1s infinite;
	}
	
	.connection-status.disconnected {
		background: #ffebee;
		color: #c62828;
	}
	
	.connection-status.disconnected .status-dot {
		background: #f44336;
		animation: none;
	}
	
	@keyframes pulse {
		0% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
		100% {
			opacity: 1;
		}
	}
	
	.feed-filters {
		display: flex;
		gap: 0.5rem;
		padding: 0.75rem;
		background: #fafafa;
		border-bottom: 1px solid #e0e0e0;
		overflow-x: auto;
	}
	
	.filter-btn {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.375rem 0.75rem;
		background: white;
		border: 1px solid #d0d0d0;
		border-radius: 16px;
		cursor: pointer;
		font-size: 0.875rem;
		white-space: nowrap;
		transition: all 0.2s;
	}
	
	.filter-btn:hover {
		background: #f5f5f5;
	}
	
	.filter-btn.active {
		background: #2196F3;
		color: white;
		border-color: #1976D2;
	}
	
	.filter-icon {
		font-size: 1rem;
	}
	
	.filter-label {
		font-weight: 500;
	}
	
	.feed-container {
		flex: 1;
		overflow-y: auto;
		padding: 0.5rem;
	}
	
	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 200px;
		color: #999;
		text-align: center;
	}
	
	.empty-state p {
		margin: 0;
		font-size: 1rem;
		font-weight: 500;
	}
	
	.empty-state small {
		margin-top: 0.5rem;
		font-size: 0.875rem;
	}
	
	.activity-item {
		display: flex;
		gap: 0.75rem;
		padding: 0.75rem;
		border-radius: 6px;
		transition: background 0.2s;
	}
	
	.activity-item:hover {
		background: #f9f9f9;
	}
	
	.activity-icon {
		width: 32px;
		height: 32px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		font-size: 1rem;
		color: white;
	}
	
	.activity-content {
		flex: 1;
		min-width: 0;
	}
	
	.activity-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.25rem;
	}
	
	.activity-user {
		font-weight: 600;
		color: #333;
		font-size: 0.875rem;
	}
	
	.activity-time {
		font-size: 0.75rem;
		color: #999;
	}
	
	.activity-message {
		font-size: 0.875rem;
		color: #666;
		word-wrap: break-word;
	}
	
	.chat-content {
		background: #f5f5f5;
		padding: 0.375rem 0.625rem;
		border-radius: 12px;
		display: inline-block;
		max-width: 100%;
	}
	
	/* Compact mode */
	.feed-compact {
		padding: 0.75rem;
	}
	
	.compact-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.5rem;
	}
	
	.compact-title {
		font-weight: 600;
		font-size: 0.875rem;
		color: #333;
	}
	
	.compact-status {
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}
	
	.compact-status.connected {
		background: #4caf50;
	}
	
	.compact-status.disconnected {
		background: #f44336;
	}
	
	.compact-items {
		display: flex;
		flex-direction: column;
		gap: 0.375rem;
	}
	
	.compact-item {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.75rem;
	}
	
	.compact-icon {
		font-size: 0.875rem;
	}
	
	.compact-text {
		flex: 1;
		color: #666;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	
	.compact-time {
		color: #999;
		font-size: 0.7rem;
	}
</style>