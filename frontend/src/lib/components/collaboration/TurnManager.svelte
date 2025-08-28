<script lang="ts">
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { InitiativeEntry } from '$lib/types/collaboration';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
	
	interface Props {
		editable?: boolean;
		onTurnChange?: (index: number) => void;
	}
	
	let { editable = false, onTurnChange }: Props = $props();
	
	const currentRoom = $derived(collaborationStore.currentRoom);
	const initiative = $derived(currentRoom?.state.initiative_order || []);
	const activeTurn = $derived(currentRoom?.state.active_turn || 0);
	const roundNumber = $derived(currentRoom?.state.round_number || 1);
	const hasPermission = $derived(collaborationStore.hasPermission('control_initiative', 'initiative'));
	
	let dragging = $state<number | null>(null);
	let dragOver = $state<number | null>(null);
	let addingEntry = $state(false);
	let newEntryName = $state('');
	let newEntryInitiative = $state(0);
	let newEntryIsPlayer = $state(false);
	
	async function nextTurn() {
		if (!hasPermission) return;
		await collaborationStore.nextTurn();
		onTurnChange?.(activeTurn);
	}
	
	async function previousTurn() {
		if (!hasPermission || !currentRoom) return;
		
		let prevTurn = activeTurn - 1;
		let prevRound = roundNumber;
		
		if (prevTurn < 0) {
			prevTurn = initiative.length - 1;
			prevRound = Math.max(1, roundNumber - 1);
		}
		
		await collaborationStore.updateState({
			path: ['active_turn'],
			value: prevTurn,
			operation: 'set',
			version: currentRoom.state.version + 1,
			previous_version: currentRoom.state.version
		});
		
		if (prevRound !== roundNumber) {
			await collaborationStore.updateState({
				path: ['round_number'],
				value: prevRound,
				operation: 'set',
				version: currentRoom.state.version + 2,
				previous_version: currentRoom.state.version + 1
			});
		}
		
		onTurnChange?.(prevTurn);
	}
	
	async function addEntry() {
		if (!newEntryName.trim() || !currentRoom) return;
		
		const newEntry: InitiativeEntry = {
			id: `init-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
			name: newEntryName.trim(),
			initiative: newEntryInitiative,
			is_player: newEntryIsPlayer,
			conditions: [],
			has_acted: false
		};
		
		const updatedInitiative = [...initiative, newEntry].sort((a, b) => b.initiative - a.initiative);
		
		await collaborationStore.updateInitiative(updatedInitiative);
		
		// Reset form
		newEntryName = '';
		newEntryInitiative = 0;
		newEntryIsPlayer = false;
		addingEntry = false;
	}
	
	async function removeEntry(index: number) {
		if (!hasPermission || !currentRoom) return;
		
		const updatedInitiative = initiative.filter((_, i) => i !== index);
		await collaborationStore.updateInitiative(updatedInitiative);
		
		// Adjust active turn if needed
		if (index === activeTurn && activeTurn >= updatedInitiative.length) {
			await collaborationStore.updateState({
				path: ['active_turn'],
				value: 0,
				operation: 'set',
				version: currentRoom.state.version + 2,
				previous_version: currentRoom.state.version + 1
			});
		}
	}
	
	async function updateCondition(index: number, condition: string, add: boolean) {
		if (!hasPermission || !currentRoom) return;
		
		const entry = initiative[index];
		if (!entry) return;
		
		const updatedConditions = add
			? [...entry.conditions, condition]
			: entry.conditions.filter(c => c !== condition);
		
		const updatedInitiative = [...initiative];
		updatedInitiative[index] = { ...entry, conditions: updatedConditions };
		
		await collaborationStore.updateInitiative(updatedInitiative);
	}
	
	function handleDragStart(index: number) {
		if (!hasPermission) return;
		dragging = index;
	}
	
	function handleDragEnd() {
		dragging = null;
		dragOver = null;
	}
	
	function handleDragOver(index: number) {
		if (dragging === null) return;
		dragOver = index;
	}
	
	async function handleDrop(index: number) {
		if (dragging === null || !hasPermission || !currentRoom) return;
		
		const updatedInitiative = [...initiative];
		const [draggedItem] = updatedInitiative.splice(dragging, 1);
		updatedInitiative.splice(index, 0, draggedItem);
		
		await collaborationStore.updateInitiative(updatedInitiative);
		
		dragging = null;
		dragOver = null;
	}
	
	const conditions = [
		{ name: 'Stunned', icon: '😵', color: 'text-yellow-600' },
		{ name: 'Prone', icon: '🛌', color: 'text-gray-600' },
		{ name: 'Grappled', icon: '🤝', color: 'text-purple-600' },
		{ name: 'Poisoned', icon: '🤢', color: 'text-green-600' },
		{ name: 'Invisible', icon: '👻', color: 'text-blue-600' },
		{ name: 'Blessed', icon: '✨', color: 'text-yellow-500' },
		{ name: 'Cursed', icon: '💀', color: 'text-red-600' },
	];
</script>

<Card>
	<CardHeader>
		<div class="flex justify-between items-center">
			<CardTitle>Initiative Tracker</CardTitle>
			<div class="text-sm text-gray-600">
				Round {roundNumber}
			</div>
		</div>
	</CardHeader>
	<CardContent class="space-y-4">
		{#if hasPermission}
			<div class="flex gap-2">
				<Button 
					size="sm"
					onclick={previousTurn}
					disabled={initiative.length === 0}
				>
					Previous
				</Button>
				<Button 
					size="sm"
					onclick={nextTurn}
					disabled={initiative.length === 0}
				>
					Next Turn
				</Button>
				{#if editable}
					<Button 
						size="sm"
						variant="outline"
						onclick={() => addingEntry = true}
					>
						Add Entry
					</Button>
				{/if}
			</div>
		{/if}
		
		<div class="space-y-2">
			{#each initiative as entry, index}
				<div 
					class="border rounded-lg p-3 transition-all"
					class:bg-blue-50={index === activeTurn}
					class:border-blue-400={index === activeTurn}
					class:opacity-50={dragging === index}
					class:border-dashed={dragOver === index}
					draggable={hasPermission && editable}
					ondragstart={() => handleDragStart(index)}
					ondragend={handleDragEnd}
					ondragover={(e) => {
						e.preventDefault();
						handleDragOver(index);
					}}
					ondrop={(e) => {
						e.preventDefault();
						handleDrop(index);
					}}
				>
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-3">
							{#if hasPermission && editable}
								<div class="cursor-move text-gray-400">
									<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
									</svg>
								</div>
							{/if}
							
							<div class="flex items-center gap-2">
								<span class="font-bold text-lg text-gray-700">
									{entry.initiative}
								</span>
								<div>
									<div class="font-medium">
										{entry.name}
										{#if entry.is_player}
											<span class="text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded ml-1">
												PC
											</span>
										{:else}
											<span class="text-xs bg-red-100 text-red-800 px-1.5 py-0.5 rounded ml-1">
												NPC
											</span>
										{/if}
									</div>
									{#if entry.current_hp !== undefined && entry.max_hp}
										<div class="text-xs text-gray-600">
											HP: {entry.current_hp}/{entry.max_hp}
										</div>
									{/if}
								</div>
							</div>
							
							{#if index === activeTurn}
								<span class="text-xs bg-blue-500 text-white px-2 py-1 rounded animate-pulse">
									ACTIVE
								</span>
							{/if}
						</div>
						
						<div class="flex items-center gap-2">
							{#if entry.conditions.length > 0}
								<div class="flex gap-1">
									{#each entry.conditions as condition}
										{@const conditionData = conditions.find(c => c.name === condition)}
										{#if conditionData}
											<span 
												class={`text-lg ${conditionData.color}`}
												title={conditionData.name}
											>
												{conditionData.icon}
											</span>
										{/if}
									{/each}
								</div>
							{/if}
							
							{#if hasPermission && editable}
								<button
									class="text-red-600 hover:bg-red-50 p-1 rounded"
									onclick={() => removeEntry(index)}
									title="Remove"
								>
									<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
									</svg>
								</button>
							{/if}
						</div>
					</div>
				</div>
			{/each}
			
			{#if initiative.length === 0}
				<div class="text-center py-8 text-gray-500">
					No combatants in initiative order
				</div>
			{/if}
		</div>
	</CardContent>
</Card>

<!-- Add Entry Dialog -->
{#if addingEntry}
	<div class="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
		<Card class="w-full max-w-md">
			<CardHeader>
				<CardTitle>Add to Initiative</CardTitle>
			</CardHeader>
			<CardContent class="space-y-4">
				<div>
					<label for="entry-name" class="block text-sm font-medium mb-1">
						Name
					</label>
					<input
						id="entry-name"
						type="text"
						bind:value={newEntryName}
						class="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
						placeholder="Character or monster name..."
					/>
				</div>
				
				<div>
					<label for="entry-initiative" class="block text-sm font-medium mb-1">
						Initiative Roll
					</label>
					<input
						id="entry-initiative"
						type="number"
						bind:value={newEntryInitiative}
						class="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
						placeholder="0"
					/>
				</div>
				
				<label class="flex items-center gap-2">
					<input
						type="checkbox"
						bind:checked={newEntryIsPlayer}
					/>
					<span class="text-sm">Player Character</span>
				</label>
				
				<div class="flex justify-end gap-2">
					<Button 
						variant="outline"
						onclick={() => addingEntry = false}
					>
						Cancel
					</Button>
					<Button 
						onclick={addEntry}
						disabled={!newEntryName.trim()}
					>
						Add
					</Button>
				</div>
			</CardContent>
		</Card>
	</div>
{/if}