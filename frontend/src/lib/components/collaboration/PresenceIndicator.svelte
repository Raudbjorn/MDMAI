<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { Participant, CursorPosition } from '$lib/types/collaboration';
	
	interface Props {
		showCursors?: boolean;
		showStatus?: boolean;
		showActivity?: boolean;
		compactMode?: boolean;
	}
	
	let {
		showCursors = true,
		showStatus = true,
		showActivity = true,
		compactMode = false
	}: Props = $props();
	
	let participants = $derived(collaborationStore.participants);
	let presence = $derived(collaborationStore.presence);
	let currentUser = $derived(collaborationStore.currentParticipant);
	
	// Track participant activity
	let participantActivity = $state<Map<string, {
		lastActivity: number;
		isTyping: boolean;
		isDrawing: boolean;
		isViewing: string | null;
	}>>(new Map());
	
	// Update activity tracking
	$effect(() => {
		const unsubscribe = collaborationStore.onMessage('activity_update', (msg) => {
			const { user_id, activity } = msg.data;
			participantActivity.set(user_id, {
				lastActivity: Date.now(),
				...activity
			});
			// Trigger reactivity
			participantActivity = new Map(participantActivity);
		});
		
		return unsubscribe;
	});
	
	// Calculate relative activity time
	function getActivityStatus(participant: Participant): string {
		const activity = participantActivity.get(participant.user_id);
		if (!activity) return 'Idle';
		
		if (activity.isTyping) return 'Typing...';
		if (activity.isDrawing) return 'Drawing...';
		if (activity.isViewing) return `Viewing ${activity.isViewing}`;
		
		const timeSince = Date.now() - activity.lastActivity;
		if (timeSince < 5000) return 'Active';
		if (timeSince < 60000) return 'Recently active';
		if (timeSince < 300000) return 'Away';
		
		return 'Idle';
	}
	
	// Get status color
	function getStatusColor(status: Participant['status']): string {
		switch (status) {
			case 'online':
				return '#4CAF50';
			case 'away':
				return '#FFC107';
			case 'offline':
				return '#9E9E9E';
			default:
				return '#9E9E9E';
		}
	}
	
	// Format last seen time
	function formatLastSeen(timestamp: string): string {
		const date = new Date(timestamp);
		const now = new Date();
		const diff = now.getTime() - date.getTime();
		
		if (diff < 60000) return 'Just now';
		if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
		if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours ago`;
		
		return date.toLocaleDateString();
	}
	
	// Mouse tracking for cursor display
	let cursorsContainer: HTMLElement | undefined;
	
	onMount(() => {
		if (showCursors && cursorsContainer) {
			// Track local cursor and broadcast position
			const handleMouseMove = (e: MouseEvent) => {
				if (!cursorsContainer) return;
				const rect = cursorsContainer.getBoundingClientRect();
				const x = e.clientX - rect.left;
				const y = e.clientY - rect.top;
				
				// Only send if within container bounds
				if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
					collaborationStore.updateCursor(x, y, 'presence');
				}
			};
			
			document.addEventListener('mousemove', handleMouseMove);
			
			return () => {
				document.removeEventListener('mousemove', handleMouseMove);
			};
		}
	});
</script>

{#if !compactMode}
	<div class="presence-container">
		<div class="presence-header">
			<h3>Active Participants ({participants.filter(p => p.status === 'online').length}/{participants.length})</h3>
		</div>
		
		<div class="participants-list">
			{#each participants as participant}
				<div class="participant-item" class:current={participant.user_id === currentUser?.user_id}>
					<div 
						class="participant-avatar"
						style="background-color: {participant.color}"
					>
						{participant.username.charAt(0).toUpperCase()}
						{#if showStatus}
							<div 
								class="status-indicator"
								style="background-color: {getStatusColor(participant.status)}"
								title={participant.status}
							></div>
						{/if}
					</div>
					
					<div class="participant-info">
						<div class="participant-name">
							{participant.username}
							{#if participant.user_id === currentUser?.user_id}
								<span class="you-badge">(You)</span>
							{/if}
							{#if participant.role === 'host'}
								<span class="role-badge host">Host</span>
							{:else if participant.role === 'gm'}
								<span class="role-badge gm">GM</span>
							{/if}
						</div>
						
						{#if showActivity}
							<div class="participant-activity">
								{getActivityStatus(participant)}
							</div>
						{/if}
					</div>
					
					<div class="participant-meta">
						<div class="last-seen">
							{formatLastSeen(participant.last_activity)}
						</div>
					</div>
				</div>
			{/each}
		</div>
		
		{#if showCursors}
			<div class="cursors-container" bind:this={cursorsContainer}>
				{#each Array.from(presence.entries()) as [userId, cursor]}
					{@const participant = participants.find(p => p.user_id === userId)}
					{#if participant && userId !== currentUser?.user_id}
						<div 
							class="remote-cursor"
							style="
								left: {cursor.x}px;
								top: {cursor.y}px;
								color: {participant.color};
							"
						>
							<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
								<path d="M3 3l7 14v-6h6L3 3z"/>
							</svg>
							<span class="cursor-label">{participant.username}</span>
						</div>
					{/if}
				{/each}
			</div>
		{/if}
	</div>
{:else}
	<div class="presence-compact">
		<div class="compact-avatars">
			{#each participants.filter(p => p.status === 'online').slice(0, 5) as participant}
				<div 
					class="compact-avatar"
					style="background-color: {participant.color}"
					title="{participant.username} - {getActivityStatus(participant)}"
				>
					{participant.username.charAt(0).toUpperCase()}
					<div 
						class="compact-status"
						style="background-color: {getStatusColor(participant.status)}"
					></div>
				</div>
			{/each}
			{#if participants.filter(p => p.status === 'online').length > 5}
				<div class="compact-more">
					+{participants.filter(p => p.status === 'online').length - 5}
				</div>
			{/if}
		</div>
		
		<div class="compact-summary">
			{participants.filter(p => p.status === 'online').length} online
		</div>
	</div>
{/if}

<style>
	.presence-container {
		background: white;
		border-radius: 8px;
		border: 1px solid #e0e0e0;
		overflow: hidden;
		position: relative;
	}
	
	.presence-header {
		padding: 1rem;
		background: #f5f5f5;
		border-bottom: 1px solid #e0e0e0;
	}
	
	.presence-header h3 {
		margin: 0;
		font-size: 1rem;
		font-weight: 600;
		color: #333;
	}
	
	.participants-list {
		padding: 0.5rem;
		max-height: 400px;
		overflow-y: auto;
	}
	
	.participant-item {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.75rem;
		border-radius: 6px;
		transition: background 0.2s;
	}
	
	.participant-item:hover {
		background: #f9f9f9;
	}
	
	.participant-item.current {
		background: #e3f2fd;
	}
	
	.participant-avatar {
		position: relative;
		width: 40px;
		height: 40px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		color: white;
		font-weight: bold;
		font-size: 1rem;
		flex-shrink: 0;
	}
	
	.status-indicator {
		position: absolute;
		bottom: 0;
		right: 0;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		border: 2px solid white;
	}
	
	.participant-info {
		flex: 1;
	}
	
	.participant-name {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-weight: 500;
		color: #333;
	}
	
	.you-badge {
		font-size: 0.75rem;
		color: #666;
		font-weight: normal;
	}
	
	.role-badge {
		font-size: 0.7rem;
		padding: 0.125rem 0.375rem;
		border-radius: 3px;
		font-weight: 600;
		text-transform: uppercase;
	}
	
	.role-badge.host {
		background: #ffeaa7;
		color: #d63031;
	}
	
	.role-badge.gm {
		background: #dfe6e9;
		color: #2d3436;
	}
	
	.participant-activity {
		font-size: 0.875rem;
		color: #666;
		margin-top: 0.25rem;
	}
	
	.participant-meta {
		text-align: right;
	}
	
	.last-seen {
		font-size: 0.75rem;
		color: #999;
	}
	
	.cursors-container {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		overflow: hidden;
	}
	
	.remote-cursor {
		position: absolute;
		pointer-events: none;
		transition: transform 0.1s ease-out;
		z-index: 100;
	}
	
	.remote-cursor svg {
		filter: drop-shadow(1px 1px 1px rgba(0, 0, 0, 0.3));
	}
	
	.cursor-label {
		position: absolute;
		top: 20px;
		left: 15px;
		padding: 0.25rem 0.5rem;
		background: currentColor;
		color: white;
		font-size: 0.75rem;
		border-radius: 3px;
		white-space: nowrap;
		font-weight: 500;
	}
	
	/* Compact mode styles */
	.presence-compact {
		display: flex;
		align-items: center;
		gap: 1rem;
		padding: 0.5rem;
		background: white;
		border-radius: 6px;
		border: 1px solid #e0e0e0;
	}
	
	.compact-avatars {
		display: flex;
		align-items: center;
		gap: -10px;
	}
	
	.compact-avatar {
		position: relative;
		width: 32px;
		height: 32px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		color: white;
		font-weight: bold;
		font-size: 0.875rem;
		border: 2px solid white;
		z-index: 1;
		transition: transform 0.2s;
	}
	
	.compact-avatar:hover {
		z-index: 10;
		transform: scale(1.1);
	}
	
	.compact-status {
		position: absolute;
		bottom: 0;
		right: 0;
		width: 8px;
		height: 8px;
		border-radius: 50%;
		border: 1px solid white;
	}
	
	.compact-more {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 32px;
		height: 32px;
		border-radius: 50%;
		background: #666;
		color: white;
		font-size: 0.75rem;
		font-weight: bold;
		border: 2px solid white;
		margin-left: 10px;
		z-index: 1;
	}
	
	.compact-summary {
		font-size: 0.875rem;
		color: #666;
		font-weight: 500;
	}
</style>