<script lang="ts">
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { ChatMessage, DiceRoll } from '$lib/types/collaboration';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
	import { onMount } from 'svelte';
	
	let messageContainer: HTMLDivElement;
	let messageInput = $state('');
	let diceExpression = $state('');
	let showDiceRoller = $state(false);
	let autoScroll = $state(true);
	
	const messages = $derived(collaborationStore.messages);
	const currentRoom = $derived(collaborationStore.currentRoom);
	const diceRolls = $derived(currentRoom?.state.dice_rolls || []);
	const currentParticipant = $derived(collaborationStore.currentParticipant);
	
	async function sendMessage() {
		if (!messageInput.trim()) return;
		
		await collaborationStore.sendChatMessage(messageInput.trim(), 'text');
		messageInput = '';
	}
	
	async function rollDice() {
		if (!diceExpression.trim()) return;
		
		try {
			const roll = await collaborationStore.rollDice(diceExpression.trim());
			
			// Send roll result as chat message
			await collaborationStore.sendChatMessage(
				`rolled ${diceExpression}: **${roll.total}** (${roll.results.join(', ')})`,
				'roll'
			);
			
			diceExpression = '';
			showDiceRoller = false;
		} catch (error) {
			console.error('Failed to roll dice:', error);
		}
	}
	
	function quickRoll(expression: string) {
		diceExpression = expression;
		rollDice();
	}
	
	function formatTimestamp(timestamp: string) {
		const date = new Date(timestamp);
		const now = new Date();
		const diff = now.getTime() - date.getTime();
		
		if (diff < 60000) {
			return 'just now';
		} else if (diff < 3600000) {
			return `${Math.floor(diff / 60000)}m ago`;
		} else if (diff < 86400000) {
			return `${Math.floor(diff / 3600000)}h ago`;
		} else {
			return date.toLocaleDateString();
		}
	}
	
	function getMessageColor(message: ChatMessage) {
		if (message.type === 'roll') return 'bg-purple-50 border-purple-200';
		if (message.type === 'system') return 'bg-gray-50 border-gray-200';
		if (message.sender_id === currentParticipant?.user_id) return 'bg-blue-50 border-blue-200';
		return 'bg-white border-gray-200';
	}
	
	function parseRollMessage(content: string) {
		// Parse dice roll messages for special formatting
		const rollPattern = /rolled (.+): \*\*(\d+)\*\* \(([^)]+)\)/;
		const match = content.match(rollPattern);
		
		if (match) {
			return {
				expression: match[1],
				total: match[2],
				rolls: match[3]
			};
		}
		return null;
	}
	
	$effect(() => {
		// Auto-scroll to bottom when new messages arrive
		if (autoScroll && messageContainer && messages.length > 0) {
			requestAnimationFrame(() => {
				messageContainer.scrollTop = messageContainer.scrollHeight;
			});
		}
	});
	
	onMount(() => {
		// Check if user scrolled up
		if (messageContainer) {
			messageContainer.addEventListener('scroll', () => {
				const isAtBottom = 
					messageContainer.scrollHeight - messageContainer.scrollTop 
					<= messageContainer.clientHeight + 50;
				autoScroll = isAtBottom;
			});
		}
	});
	
	// Common dice expressions
	const quickDice = [
		{ label: 'd20', expression: '1d20' },
		{ label: '2d6', expression: '2d6' },
		{ label: 'd100', expression: '1d100' },
		{ label: '4d6', expression: '4d6' },
		{ label: 'd12', expression: '1d12' },
		{ label: '3d8', expression: '3d8' },
	];
</script>

<Card class="flex flex-col h-full">
	<CardHeader class="flex-shrink-0">
		<div class="flex justify-between items-center">
			<CardTitle>Chat</CardTitle>
			<Button 
				size="sm"
				variant="outline"
				onclick={() => showDiceRoller = !showDiceRoller}
			>
				🎲 Dice
			</Button>
		</div>
	</CardHeader>
	<CardContent class="flex-1 flex flex-col overflow-hidden p-0">
		<!-- Dice Roller -->
		{#if showDiceRoller}
			<div class="p-4 border-b bg-gray-50">
				<div class="space-y-2">
					<div class="flex gap-2">
						{#each quickDice as dice}
							<Button 
								size="sm"
								variant="outline"
								onclick={() => quickRoll(dice.expression)}
							>
								{dice.label}
							</Button>
						{/each}
					</div>
					<div class="flex gap-2">
						<input
							type="text"
							bind:value={diceExpression}
							placeholder="e.g., 2d6+3"
							class="flex-1 px-3 py-1 border rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
							onkeypress={(e) => e.key === 'Enter' && rollDice()}
						/>
						<Button 
							size="sm"
							onclick={rollDice}
							disabled={!diceExpression.trim()}
						>
							Roll
						</Button>
					</div>
				</div>
			</div>
		{/if}
		
		<!-- Messages -->
		<div 
			bind:this={messageContainer}
			class="flex-1 overflow-y-auto p-4 space-y-2"
		>
			{#each messages as message}
				{@const rollData = parseRollMessage(message.content)}
				<div 
					class={`p-2 rounded border ${getMessageColor(message)}`}
				>
					<div class="flex justify-between items-start mb-1">
						<span class="font-medium text-sm">
							{message.sender_name}
							{#if message.sender_id === currentParticipant?.user_id}
								<span class="text-gray-500">(you)</span>
							{/if}
						</span>
						<span class="text-xs text-gray-500">
							{formatTimestamp(message.timestamp)}
						</span>
					</div>
					
					{#if rollData}
						<div class="flex items-center gap-2">
							<span class="text-2xl">🎲</span>
							<div>
								<div class="font-bold text-lg text-purple-600">
									{rollData.total}
								</div>
								<div class="text-xs text-gray-600">
									{rollData.expression}: ({rollData.rolls})
								</div>
							</div>
						</div>
					{:else if message.type === 'system'}
						<div class="text-sm text-gray-600 italic">
							{message.content}
						</div>
					{:else}
						<div class="text-sm">
							{message.content}
						</div>
					{/if}
					
					{#if message.edited}
						<span class="text-xs text-gray-500 italic">
							(edited)
						</span>
					{/if}
				</div>
			{/each}
			
			{#if messages.length === 0}
				<div class="text-center py-8 text-gray-500">
					No messages yet. Start the conversation!
				</div>
			{/if}
			
			{#if !autoScroll}
				<button 
					class="sticky bottom-0 left-1/2 transform -translate-x-1/2 bg-blue-500 text-white px-3 py-1 rounded-full text-sm shadow-lg"
					onclick={() => {
						autoScroll = true;
						if (messageContainer) {
							messageContainer.scrollTop = messageContainer.scrollHeight;
						}
					}}
				>
					↓ New messages
				</button>
			{/if}
		</div>
		
		<!-- Recent Dice Rolls -->
		{#if diceRolls.length > 0}
			<div class="p-2 border-t bg-gray-50">
				<div class="text-xs text-gray-600 mb-1">Recent rolls:</div>
				<div class="flex gap-2 overflow-x-auto">
					{#each diceRolls.slice(-5) as roll}
						<div class="flex-shrink-0 px-2 py-1 bg-white border rounded text-xs">
							<span class="font-medium">{roll.player_name}:</span>
							<span class="text-purple-600 font-bold ml-1">{roll.total}</span>
							<span class="text-gray-500 ml-1">({roll.expression})</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}
		
		<!-- Message Input -->
		<div class="p-4 border-t">
			<form 
				onsubmit={(e) => {
					e.preventDefault();
					sendMessage();
				}}
				class="flex gap-2"
			>
				<input
					type="text"
					bind:value={messageInput}
					placeholder="Type a message..."
					class="flex-1 px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
				/>
				<Button 
					type="submit"
					disabled={!messageInput.trim()}
				>
					Send
				</Button>
			</form>
		</div>
	</CardContent>
</Card>