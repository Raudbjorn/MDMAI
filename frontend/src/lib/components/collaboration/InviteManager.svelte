<script lang="ts">
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { RoomInvitation, ParticipantRole } from '$lib/types/collaboration';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card';
	
	interface Props {
		roomId: string;
		roomName: string;
	}
	
	let { roomId, roomName }: Props = $props();
	
	let showInviteDialog = $state(false);
	let inviteRole = $state<ParticipantRole>('player');
	let inviteEmail = $state('');
	let inviteExpiration = $state(3600000); // 1 hour default
	let generatedCode = $state('');
	let copySuccess = $state(false);
	
	const invitations = $derived(collaborationStore.invitations.filter(inv => inv.room_id === roomId));
	const pendingInvitations = $derived(invitations.filter(inv => inv.status === 'pending'));
	const currentParticipant = $derived(collaborationStore.currentParticipant);
	const canInvite = $derived(
		currentParticipant?.role === 'host' || 
		currentParticipant?.role === 'gm'
	);
	
	async function createInvitation() {
		try {
			const invitation = await collaborationStore.createInvitation(
				roomId,
				inviteRole,
				inviteEmail || undefined,
				inviteExpiration
			);
			
			if (invitation.invite_code) {
				generatedCode = invitation.invite_code;
			}
			
			// Reset form
			inviteEmail = '';
			inviteRole = 'player';
		} catch (error) {
			console.error('Failed to create invitation:', error);
		}
	}
	
	async function copyInviteLink() {
		const inviteUrl = `${window.location.origin}/join/${roomId}?code=${generatedCode}`;
		
		try {
			await navigator.clipboard.writeText(inviteUrl);
			copySuccess = true;
			setTimeout(() => copySuccess = false, 2000);
		} catch (error) {
			console.error('Failed to copy invite link:', error);
		}
	}
	
	function formatExpiration(ms: number) {
		const hours = Math.floor(ms / 3600000);
		const minutes = Math.floor((ms % 3600000) / 60000);
		
		if (hours > 0) {
			return `${hours} hour${hours > 1 ? 's' : ''}`;
		}
		return `${minutes} minute${minutes !== 1 ? 's' : ''}`;
	}
	
	function getInvitationStatus(invitation: RoomInvitation) {
		const now = new Date();
		const expires = new Date(invitation.expires_at);
		
		if (invitation.status === 'accepted') {
			return { text: 'Accepted', color: 'text-green-600' };
		}
		if (invitation.status === 'declined') {
			return { text: 'Declined', color: 'text-red-600' };
		}
		if (expires < now) {
			return { text: 'Expired', color: 'text-gray-600' };
		}
		return { text: 'Pending', color: 'text-yellow-600' };
	}
</script>

{#if canInvite}
	<Card>
		<CardHeader>
			<CardTitle>Invitations</CardTitle>
			<CardDescription>
				Invite players to join {roomName}
			</CardDescription>
		</CardHeader>
		<CardContent class="space-y-4">
			<Button 
				onclick={() => showInviteDialog = true}
				class="w-full"
			>
				Create Invitation
			</Button>
			
			{#if generatedCode}
				<div class="p-3 bg-blue-50 border border-blue-200 rounded">
					<div class="flex justify-between items-center">
						<div>
							<div class="font-medium text-sm">Invite Code</div>
							<div class="font-mono text-lg">{generatedCode}</div>
						</div>
						<Button 
							size="sm"
							variant="outline"
							onclick={copyInviteLink}
						>
							{copySuccess ? 'Copied!' : 'Copy Link'}
						</Button>
					</div>
				</div>
			{/if}
			
			{#if pendingInvitations.length > 0}
				<div class="space-y-2">
					<h4 class="text-sm font-medium">Pending Invitations</h4>
					{#each pendingInvitations as invitation}
						{@const status = getInvitationStatus(invitation)}
						<div class="flex justify-between items-center p-2 border rounded">
							<div>
								{#if invitation.invited_user_id}
									<div class="font-medium text-sm">User: {invitation.invited_user_id}</div>
								{:else}
									<div class="font-medium text-sm">Code: {invitation.invite_code}</div>
								{/if}
								<div class="text-xs text-gray-600">
									Role: {invitation.role} • Expires: {new Date(invitation.expires_at).toLocaleString()}
								</div>
							</div>
							<span class={`text-sm font-medium ${status.color}`}>
								{status.text}
							</span>
						</div>
					{/each}
				</div>
			{/if}
		</CardContent>
	</Card>
	
	<!-- Create Invitation Dialog -->
	{#if showInviteDialog}
		<div class="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
			<Card class="w-full max-w-md">
				<CardHeader>
					<CardTitle>Create Invitation</CardTitle>
					<CardDescription>
						Generate an invite link or send to a specific user
					</CardDescription>
				</CardHeader>
				<CardContent class="space-y-4">
					<div>
						<label for="invite-type" class="block text-sm font-medium mb-2">
							Invitation Type
						</label>
						<div class="space-y-2">
							<label class="flex items-center gap-2">
								<input 
									type="radio" 
									name="invite-type"
									checked={!inviteEmail}
									onchange={() => inviteEmail = ''}
								/>
								<span>Generate invite code</span>
							</label>
							<label class="flex items-center gap-2">
								<input 
									type="radio" 
									name="invite-type"
									checked={!!inviteEmail}
									onchange={() => inviteEmail = 'user@example.com'}
								/>
								<span>Send to specific user</span>
							</label>
						</div>
					</div>
					
					{#if inviteEmail !== ''}
						<div>
							<label for="invite-email" class="block text-sm font-medium mb-1">
								User Email
							</label>
							<input
								id="invite-email"
								type="email"
								bind:value={inviteEmail}
								class="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
								placeholder="user@example.com"
							/>
						</div>
					{/if}
					
					<div>
						<label for="invite-role" class="block text-sm font-medium mb-1">
							Role
						</label>
						<select
							id="invite-role"
							bind:value={inviteRole}
							class="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
						>
							<option value="player">Player</option>
							<option value="gm">Game Master</option>
							<option value="spectator">Spectator</option>
						</select>
					</div>
					
					<div>
						<label for="invite-expiration" class="block text-sm font-medium mb-1">
							Expires In
						</label>
						<select
							id="invite-expiration"
							bind:value={inviteExpiration}
							class="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
						>
							<option value={900000}>15 minutes</option>
							<option value={1800000}>30 minutes</option>
							<option value={3600000}>1 hour</option>
							<option value={10800000}>3 hours</option>
							<option value={86400000}>24 hours</option>
							<option value={604800000}>7 days</option>
						</select>
					</div>
					
					<div class="flex justify-end gap-2">
						<Button 
							variant="outline"
							onclick={() => showInviteDialog = false}
						>
							Cancel
						</Button>
						<Button onclick={() => {
							createInvitation();
							showInviteDialog = false;
						}}>
							Create Invitation
						</Button>
					</div>
				</CardContent>
			</Card>
		</div>
	{/if}
{/if}