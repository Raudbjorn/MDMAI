<script lang="ts">
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { Participant, ParticipantRole, Permission } from '$lib/types/collaboration';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
	
	interface Props {
		showActions?: boolean;
		onParticipantClick?: (participant: Participant) => void;
	}
	
	let { showActions = false, onParticipantClick }: Props = $props();
	
	const participants = $derived(collaborationStore.participants);
	const currentParticipant = $derived(collaborationStore.currentParticipant);
	const presence = $derived(collaborationStore.presence);
	const isHost = $derived(currentParticipant?.role === 'host');
	const isGM = $derived(currentParticipant?.role === 'gm');
	
	let selectedParticipant = $state<Participant | null>(null);
	let showPermissionsDialog = $state(false);
	
	function getStatusColor(status: Participant['status']) {
		switch (status) {
			case 'online':
				return 'bg-green-500';
			case 'away':
				return 'bg-yellow-500';
			case 'offline':
				return 'bg-gray-400';
		}
	}
	
	function getRoleBadgeColor(role: ParticipantRole) {
		switch (role) {
			case 'host':
				return 'bg-purple-100 text-purple-800';
			case 'gm':
				return 'bg-blue-100 text-blue-800';
			case 'player':
				return 'bg-green-100 text-green-800';
			case 'spectator':
				return 'bg-gray-100 text-gray-800';
		}
	}
	
	function getInitials(name: string) {
		return name
			.split(' ')
			.map(n => n[0])
			.join('')
			.toUpperCase()
			.slice(0, 2);
	}
	
	function changeRole(participant: Participant, newRole: ParticipantRole) {
		if (!isHost && !isGM) return;
		
		// Send role change request
		// This would be implemented through the collaboration store
		console.log('Changing role for', participant.username, 'to', newRole);
	}
	
	function kickParticipant(participant: Participant) {
		if (!isHost) return;
		
		// Send kick request
		// This would be implemented through the collaboration store
		console.log('Kicking participant', participant.username);
	}
	
	function getParticipantCursor(participantId: string) {
		const cursor = presence.get(participantId);
		if (!cursor) return null;
		
		// Calculate relative time
		const age = Date.now() - cursor.timestamp;
		if (age > 5000) return null; // Hide cursor if older than 5 seconds
		
		return cursor;
	}
	
	// Color palette for participants
	const colors = [
		'#EF4444', '#F59E0B', '#10B981', '#3B82F6',
		'#8B5CF6', '#EC4899', '#14B8A6', '#F97316'
	];
	
	function getParticipantColor(index: number) {
		return colors[index % colors.length];
	}
</script>

<Card>
	<CardHeader>
		<CardTitle>Participants ({participants.length})</CardTitle>
	</CardHeader>
	<CardContent class="space-y-2">
		{#each participants as participant, index}
			{@const cursor = getParticipantCursor(participant.user_id)}
			{@const isCurrentUser = participant.user_id === currentParticipant?.user_id}
			
			<div 
				class="flex items-center justify-between p-2 rounded hover:bg-gray-50 transition-colors"
				class:bg-blue-50={isCurrentUser}
			>
				<button
					class="flex items-center gap-3 flex-1 text-left"
					onclick={() => onParticipantClick?.(participant)}
					disabled={!onParticipantClick}
				>
					<!-- Avatar with status indicator -->
					<div class="relative">
						<div 
							class="w-10 h-10 rounded-full flex items-center justify-center text-white font-medium"
							style="background-color: {participant.color || getParticipantColor(index)}"
						>
							{getInitials(participant.username)}
						</div>
						<div 
							class={`absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-white ${getStatusColor(participant.status)}`}
						></div>
					</div>
					
					<!-- Name and role -->
					<div class="flex-1">
						<div class="flex items-center gap-2">
							<span class="font-medium">
								{participant.username}
								{#if isCurrentUser}
									<span class="text-gray-500">(you)</span>
								{/if}
							</span>
							<span class={`text-xs px-2 py-0.5 rounded-full ${getRoleBadgeColor(participant.role)}`}>
								{participant.role}
							</span>
						</div>
						
						{#if cursor}
							<div class="text-xs text-gray-500">
								Active in: {cursor.element || 'session'}
							</div>
						{/if}
					</div>
				</button>
				
				<!-- Actions -->
				{#if showActions && (isHost || isGM) && !isCurrentUser}
					<div class="flex items-center gap-1">
						{#if isHost}
							<button
								class="p-1 hover:bg-gray-200 rounded"
								title="Change role"
								aria-label="Change role for {participant.username}"
								onclick={() => {
									selectedParticipant = participant;
									showPermissionsDialog = true;
								}}
							>
								<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
								</svg>
							</button>
							
							<button
								class="p-1 hover:bg-red-100 rounded text-red-600"
								title="Remove from room"
								aria-label="Remove {participant.username} from room"
								onclick={() => kickParticipant(participant)}
							>
								<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
								</svg>
							</button>
						{/if}
					</div>
				{/if}
			</div>
		{/each}
		
		{#if participants.length === 0}
			<div class="text-center py-4 text-gray-500">
				No participants yet
			</div>
		{/if}
	</CardContent>
</Card>

<!-- Permissions Dialog -->
{#if showPermissionsDialog && selectedParticipant}
	<div class="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
		<Card class="w-full max-w-md">
			<CardHeader>
				<CardTitle>Manage Permissions</CardTitle>
			</CardHeader>
			<CardContent class="space-y-4">
				<div class="font-medium">{selectedParticipant.username}</div>
				
				<div class="space-y-2">
					<label class="block text-sm font-medium mb-2">Role</label>
					<select 
						class="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
						value={selectedParticipant.role}
						onchange={(e) => changeRole(selectedParticipant!, e.currentTarget.value as ParticipantRole)}
					>
						<option value="player">Player</option>
						<option value="gm">Game Master</option>
						<option value="spectator">Spectator</option>
						{#if isHost}
							<option value="host">Host</option>
						{/if}
					</select>
				</div>
				
				<div class="space-y-2">
					<label class="block text-sm font-medium mb-2">Permissions</label>
					
					<label class="flex items-center gap-2">
						<input type="checkbox" />
						<span class="text-sm">Can edit characters</span>
					</label>
					
					<label class="flex items-center gap-2">
						<input type="checkbox" />
						<span class="text-sm">Can control initiative</span>
					</label>
					
					<label class="flex items-center gap-2">
						<input type="checkbox" />
						<span class="text-sm">Can edit monsters</span>
					</label>
					
					<label class="flex items-center gap-2">
						<input type="checkbox" />
						<span class="text-sm">Can manage session</span>
					</label>
				</div>
				
				<div class="flex justify-end gap-2">
					<Button 
						variant="outline"
						onclick={() => {
							showPermissionsDialog = false;
							selectedParticipant = null;
						}}
					>
						Cancel
					</Button>
					<Button onclick={() => {
						// Apply permissions
						showPermissionsDialog = false;
						selectedParticipant = null;
					}}>
						Apply
					</Button>
				</div>
			</CardContent>
		</Card>
	</div>
{/if}