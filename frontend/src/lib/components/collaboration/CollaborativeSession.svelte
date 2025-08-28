<script lang="ts">
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import { sessionStore } from '$lib/stores/session.svelte';
	import RoomManager from './RoomManager.svelte';
	import ParticipantList from './ParticipantList.svelte';
	import InviteManager from './InviteManager.svelte';
	import TurnManager from './TurnManager.svelte';
	import ChatPanel from './ChatPanel.svelte';
	import type { CollaborativeRoom } from '$lib/types/collaboration';
	import { onMount, onDestroy } from 'svelte';
	
	interface Props {
		campaignId: string;
		campaignName?: string;
	}
	
	let { campaignId, campaignName = 'Campaign' }: Props = $props();
	
	const currentRoom = $derived(collaborationStore.currentRoom);
	const isConnected = $derived(collaborationStore.isConnected);
	const isSyncing = $derived(collaborationStore.isSyncing);
	const conflicts = $derived(collaborationStore.conflicts);
	const participants = $derived(collaborationStore.participants);
	
	let showCursors = $state(true);
	
	// Track mouse movements for cursor sharing
	function handleMouseMove(event: MouseEvent) {
		if (!showCursors || !currentRoom) return;
		
		collaborationStore.updateCursor(
			event.clientX,
			event.clientY,
			(event.target as HTMLElement)?.id
		);
	}
	
	// Reactive cursor data for template rendering
	const cursorsData = $derived(() => {
		if (!showCursors) return [];
		
		const cursors: Array<{
			userId: string;
			x: number;
			y: number;
			username: string;
			color?: string;
		}> = [];
		
		collaborationStore.presence.forEach((cursor, userId) => {
			if (userId === sessionStore.user?.id) return;
			
			const participant = participants.find(p => p.user_id === userId);
			cursors.push({
				userId,
				x: cursor.x,
				y: cursor.y,
				username: participant?.username || 'Unknown',
				color: participant?.color
			});
		});
		
		return cursors;
	});
	
	// Handle room events
	function handleRoomCreated(room: CollaborativeRoom) {
		console.log('Room created:', room);
	}
	
	function handleRoomJoined(room: CollaborativeRoom) {
		console.log('Joined room:', room);
	}
	
	function handleParticipantClick(participant: any) {
		console.log('Participant clicked:', participant);
	}
	
	function handleTurnChange(index: number) {
		console.log('Turn changed to:', index);
	}
	
	// Conflict resolution UI
	async function resolveConflict(conflictIndex: number, resolution: 'accept' | 'reject') {
		const conflict = conflicts[conflictIndex];
		if (!conflict) return;
		
		if (resolution === 'accept') {
			// Apply the conflicting update
			// This would be implemented based on your conflict resolution strategy
		}
		
		// Remove from conflicts list
		collaborationStore.conflicts.splice(conflictIndex, 1);
	}
	
	onMount(() => {
		// Add global mouse move listener
		document.addEventListener('mousemove', handleMouseMove);
		
		// Connect to collaboration server
		if (sessionStore.user && !isConnected) {
			collaborationStore.connect(sessionStore.user.id);
		}
	});
	
	onDestroy(() => {
		// Clean up
		document.removeEventListener('mousemove', handleMouseMove);
	});
</script>

<!-- Render cursors reactively without direct DOM manipulation -->
{#each cursorsData() as cursor (cursor.userId)}
	<div 
		class="fixed pointer-events-none z-50 transition-all duration-75 participant-cursor"
		style="transform: translate({cursor.x}px, {cursor.y}px); color: {cursor.color || '#000'}"
	>
		<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
			<path d="M5 3L17 9L11 11L9 17L3 5L5 3Z" fill="currentColor"/>
		</svg>
		<span class="absolute top-5 left-5 text-xs bg-black text-white px-1 rounded whitespace-nowrap">
			{cursor.username}
		</span>
	</div>
{/each}

<div class="container mx-auto p-4">
	<div class="mb-4">
		<h1 class="text-2xl font-bold">{campaignName} - Collaborative Session</h1>
		
		<!-- Status Bar -->
		<div class="flex items-center gap-4 mt-2 text-sm">
			<div class="flex items-center gap-2">
				<div class={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
				<span>{isConnected ? 'Connected' : 'Disconnected'}</span>
			</div>
			
			{#if isSyncing}
				<div class="flex items-center gap-2">
					<div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
					<span>Syncing...</span>
				</div>
			{/if}
			
			{#if currentRoom}
				<span class="text-gray-600">
					Room: {currentRoom.name} ({participants.length} participants)
				</span>
			{/if}
			
			<label class="flex items-center gap-2 ml-auto">
				<input 
					type="checkbox" 
					bind:checked={showCursors}
				/>
				<span>Show cursors</span>
			</label>
		</div>
	</div>
	
	<!-- Conflict Resolution Banner -->
	{#if conflicts.length > 0}
		<div class="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
			<div class="font-medium text-sm mb-2">Conflicts detected:</div>
			{#each conflicts as conflict, index}
				<div class="flex justify-between items-center p-2 bg-white rounded mb-1">
					<span class="text-sm">
						Update conflict on {conflict.conflicting_updates[0]?.path.join('.')}
					</span>
					<div class="flex gap-2">
						<button 
							class="text-xs px-2 py-1 bg-green-500 text-white rounded"
							onclick={() => resolveConflict(index, 'accept')}
						>
							Accept
						</button>
						<button 
							class="text-xs px-2 py-1 bg-red-500 text-white rounded"
							onclick={() => resolveConflict(index, 'reject')}
						>
							Reject
						</button>
					</div>
				</div>
			{/each}
		</div>
	{/if}
	
	{#if !currentRoom}
		<!-- Room Selection/Creation -->
		<RoomManager 
			{campaignId}
			onRoomCreated={handleRoomCreated}
			onRoomJoined={handleRoomJoined}
		/>
	{:else}
		<!-- Active Session Layout -->
		<div class="grid grid-cols-1 lg:grid-cols-4 gap-4">
			<!-- Left Sidebar -->
			<div class="lg:col-span-1 space-y-4">
				<ParticipantList 
					showActions={true}
					onParticipantClick={handleParticipantClick}
				/>
				
				<InviteManager 
					roomId={currentRoom.id}
					roomName={currentRoom.name}
				/>
			</div>
			
			<!-- Main Content -->
			<div class="lg:col-span-2 space-y-4">
				<TurnManager 
					editable={true}
					onTurnChange={handleTurnChange}
				/>
				
				<!-- Shared Notes Area -->
				<div class="border rounded-lg p-4">
					<h3 class="font-bold mb-2">Shared Session Notes</h3>
					<textarea
						class="w-full h-32 p-2 border rounded resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
						placeholder="Type session notes here... (visible to all participants)"
						value={currentRoom.state.shared_notes}
						oninput={(e) => {
							collaborationStore.updateState({
								path: ['shared_notes'],
								value: e.currentTarget.value,
								operation: 'set',
								version: currentRoom.state.version + 1,
								previous_version: currentRoom.state.version
							});
						}}
					></textarea>
				</div>
				
				<!-- Leave Room Button -->
				<button
					class="w-full py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
					onclick={() => collaborationStore.leaveRoom()}
				>
					Leave Session
				</button>
			</div>
			
			<!-- Right Sidebar - Chat -->
			<div class="lg:col-span-1 h-[600px]">
				<ChatPanel />
			</div>
		</div>
	{/if}
</div>

<style>
	/* Cursor animation */
	:global(.participant-cursor) {
		transition: transform 75ms linear;
	}
</style>