<script lang="ts">
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import { sessionStore } from '$lib/stores/session.svelte';
	import type { CollaborativeRoom, RoomSettings } from '$lib/types/collaboration';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card';
	
	interface Props {
		campaignId: string;
		onRoomCreated?: (room: CollaborativeRoom) => void;
		onRoomJoined?: (room: CollaborativeRoom) => void;
	}
	
	let { campaignId, onRoomCreated, onRoomJoined }: Props = $props();
	
	let createRoomDialog = $state(false);
	let joinRoomDialog = $state(false);
	let roomName = $state('');
	let inviteCode = $state('');
	let roomSettings = $state<Partial<RoomSettings>>({
		max_participants: 10,
		allow_spectators: true,
		require_approval: false,
		auto_save: true,
		save_interval: 60
	});
	
	let loading = $state(false);
	let error = $state('');
	
	const rooms = $derived(collaborationStore.rooms);
	const currentRoom = $derived(collaborationStore.currentRoom);
	const isConnected = $derived(collaborationStore.isConnected);
	
	async function createRoom() {
		if (!roomName.trim()) {
			error = 'Room name is required';
			return;
		}
		
		loading = true;
		error = '';
		
		try {
			const room = await collaborationStore.createRoom(
				roomName.trim(),
				campaignId,
				roomSettings
			);
			
			createRoomDialog = false;
			roomName = '';
			onRoomCreated?.(room);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to create room';
		} finally {
			loading = false;
		}
	}
	
	async function joinRoom(roomId: string) {
		loading = true;
		error = '';
		
		try {
			await collaborationStore.joinRoom(roomId, inviteCode || undefined);
			joinRoomDialog = false;
			inviteCode = '';
			
			if (collaborationStore.currentRoom) {
				onRoomJoined?.(collaborationStore.currentRoom);
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to join room';
		} finally {
			loading = false;
		}
	}
	
	async function leaveRoom() {
		try {
			await collaborationStore.leaveRoom();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to leave room';
		}
	}
	
	$effect(() => {
		// Connect to collaboration server when component mounts
		if (sessionStore.user && !isConnected) {
			collaborationStore.connect(sessionStore.user.id);
		}
	});
</script>

<div class="space-y-4">
	{#if currentRoom}
		<!-- Current Room Display -->
		<Card>
			<CardHeader>
				<CardTitle>Current Session: {currentRoom.name}</CardTitle>
				<CardDescription>
					Room ID: {currentRoom.id}
				</CardDescription>
			</CardHeader>
			<CardContent class="space-y-4">
				<div class="flex justify-between items-center">
					<div class="text-sm text-gray-600">
						Participants: {collaborationStore.participants.length} / {currentRoom.settings.max_participants}
					</div>
					<div class="flex gap-2">
						{#if collaborationStore.currentParticipant?.role === 'host'}
							<Button 
								variant="outline" 
								size="sm"
								onclick={() => createInviteDialog = true}
							>
								Invite Players
							</Button>
						{/if}
						<Button 
							variant="destructive" 
							size="sm"
							onclick={leaveRoom}
						>
							Leave Room
						</Button>
					</div>
				</div>
				
				<!-- Connection Status -->
				<div class="flex items-center gap-2">
					<div class={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
					<span class="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
				</div>
			</CardContent>
		</Card>
	{:else}
		<!-- Room List and Actions -->
		<Card>
			<CardHeader>
				<CardTitle>Collaborative Sessions</CardTitle>
				<CardDescription>
					Create or join a session to play together
				</CardDescription>
			</CardHeader>
			<CardContent class="space-y-4">
				{#if !isConnected}
					<div class="bg-yellow-50 border border-yellow-200 rounded p-3 text-sm">
						Connecting to collaboration server...
					</div>
				{:else}
					<div class="flex gap-2">
						<Button 
							onclick={() => createRoomDialog = true}
							disabled={loading}
						>
							Create New Room
						</Button>
						<Button 
							variant="outline"
							onclick={() => joinRoomDialog = true}
							disabled={loading}
						>
							Join with Code
						</Button>
					</div>
					
					{#if rooms.length > 0}
						<div class="space-y-2">
							<h4 class="text-sm font-medium">Available Rooms</h4>
							{#each rooms as room}
								<div class="flex justify-between items-center p-3 border rounded">
									<div>
										<div class="font-medium">{room.name}</div>
										<div class="text-sm text-gray-600">
											Host: {room.host_id} • {room.participants.length} participants
										</div>
									</div>
									<Button 
										size="sm"
										onclick={() => joinRoom(room.id)}
										disabled={loading}
									>
										Join
									</Button>
								</div>
							{/each}
						</div>
					{/if}
				{/if}
				
				{#if error}
					<div class="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-700">
						{error}
					</div>
				{/if}
			</CardContent>
		</Card>
	{/if}
	
	<!-- Create Room Dialog -->
	{#if createRoomDialog}
		<div class="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
			<Card class="w-full max-w-md">
				<CardHeader>
					<CardTitle>Create New Room</CardTitle>
				</CardHeader>
				<CardContent class="space-y-4">
					<div>
						<label for="room-name" class="block text-sm font-medium mb-1">
							Room Name
						</label>
						<input
							id="room-name"
							type="text"
							bind:value={roomName}
							class="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
							placeholder="Enter room name..."
						/>
					</div>
					
					<div class="space-y-2">
						<h4 class="text-sm font-medium">Settings</h4>
						
						<label class="flex items-center gap-2">
							<input
								type="checkbox"
								bind:checked={roomSettings.allow_spectators}
							/>
							<span class="text-sm">Allow spectators</span>
						</label>
						
						<label class="flex items-center gap-2">
							<input
								type="checkbox"
								bind:checked={roomSettings.require_approval}
							/>
							<span class="text-sm">Require approval to join</span>
						</label>
						
						<label class="flex items-center gap-2">
							<input
								type="checkbox"
								bind:checked={roomSettings.auto_save}
							/>
							<span class="text-sm">Auto-save session</span>
						</label>
						
						<div>
							<label for="max-participants" class="block text-sm mb-1">
								Max Participants
							</label>
							<input
								id="max-participants"
								type="number"
								bind:value={roomSettings.max_participants}
								min="2"
								max="20"
								class="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
							/>
						</div>
					</div>
					
					<div class="flex justify-end gap-2">
						<Button 
							variant="outline"
							onclick={() => createRoomDialog = false}
						>
							Cancel
						</Button>
						<Button 
							onclick={createRoom}
							disabled={loading || !roomName.trim()}
						>
							{loading ? 'Creating...' : 'Create Room'}
						</Button>
					</div>
				</CardContent>
			</Card>
		</div>
	{/if}
	
	<!-- Join Room Dialog -->
	{#if joinRoomDialog}
		<div class="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
			<Card class="w-full max-w-md">
				<CardHeader>
					<CardTitle>Join Room</CardTitle>
					<CardDescription>
						Enter the invite code to join a room
					</CardDescription>
				</CardHeader>
				<CardContent class="space-y-4">
					<div>
						<label for="invite-code" class="block text-sm font-medium mb-1">
							Invite Code
						</label>
						<input
							id="invite-code"
							type="text"
							bind:value={inviteCode}
							class="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500 uppercase"
							placeholder="Enter code..."
							maxlength="9"
						/>
					</div>
					
					<div class="flex justify-end gap-2">
						<Button 
							variant="outline"
							onclick={() => joinRoomDialog = false}
						>
							Cancel
						</Button>
						<Button 
							onclick={() => {
								// Generate room ID from invite code or show error
								if (!inviteCode.trim()) {
									error = 'Please enter an invite code';
									return;
								}
								// Assuming invite code maps to a room ID, you might need to resolve it
								// For now, use the invite code as the room ID placeholder
								joinRoom(`room-${inviteCode.trim()}`);
							}}
							disabled={loading || !inviteCode.trim()}
						>
							{loading ? 'Joining...' : 'Join'}
						</Button>
					</div>
				</CardContent>
			</Card>
		</div>
	{/if}
</div>