<script lang="ts">
	import { onMount, onDestroy, tick } from 'svelte';
	import { flip } from 'svelte/animate';
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { InitiativeEntry } from '$lib/types/collaboration';
	
	interface Props {
		roomId: string;
		showHealthBars?: boolean;
		showConditions?: boolean;
		enableTimer?: boolean;
		turnDuration?: number; // seconds
	}
	
	let {
		roomId,
		showHealthBars = true,
		showConditions = true,
		enableTimer = false,
		turnDuration = 60
	}: Props = $props();
	
	// State
	let initiatives = $state<InitiativeEntry[]>([]);
	let currentTurn = $state(0);
	let currentRound = $state(1);
	let isAddingEntry = $state(false);
	let draggedItem = $state<InitiativeEntry | null>(null);
	let dragOverIndex = $state<number | null>(null);
	let turnTimer = $state<number | null>(null);
	let timerInterval: number | null = null;
	
	// New entry form
	let newEntry = $state({
		name: '',
		initiative: 0,
		is_player: true,
		current_hp: 0,
		max_hp: 0
	});
	
	// Available conditions
	const conditions = [
		'Blinded',
		'Charmed',
		'Deafened',
		'Exhaustion',
		'Frightened',
		'Grappled',
		'Incapacitated',
		'Invisible',
		'Paralyzed',
		'Petrified',
		'Poisoned',
		'Prone',
		'Restrained',
		'Stunned',
		'Unconscious'
	];
	
	let unsubscribe: (() => void) | null = null;
	
	// Permissions
	let canManageInitiative = $derived(
		collaborationStore.hasPermission('control_initiative', 'initiative')
	);
	
	let currentParticipantTurn = $derived(
		initiatives[currentTurn]?.character_id === collaborationStore.currentParticipant?.user_id
	);
	
	onMount(() => {
		// Subscribe to initiative updates
		unsubscribe = collaborationStore.onMessage('initiative_update', (msg) => {
			initiatives = msg.data;
		});
		
		collaborationStore.onMessage('turn_changed', (msg) => {
			currentTurn = msg.data.turn;
			currentRound = msg.data.round;
			
			// Reset timer if enabled
			if (enableTimer) {
				startTurnTimer();
			}
			
			// Notify if it's this user's turn
			if (currentParticipantTurn) {
				notifyUserTurn();
			}
		});
		
		// Load initial state
		loadInitiativeOrder();
	});
	
	onDestroy(() => {
		unsubscribe?.();
		if (timerInterval) {
			clearInterval(timerInterval);
		}
	});
	
	function loadInitiativeOrder() {
		const room = collaborationStore.currentRoom;
		if (room) {
			initiatives = room.state.initiative_order;
			currentTurn = room.state.active_turn;
			currentRound = room.state.round_number;
		}
	}
	
	async function addInitiativeEntry() {
		if (!newEntry.name || !canManageInitiative) return;
		
		const entry: InitiativeEntry = {
			id: `init-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
			name: newEntry.name,
			initiative: newEntry.initiative,
			is_player: newEntry.is_player,
			current_hp: newEntry.current_hp,
			max_hp: newEntry.max_hp,
			conditions: [],
			has_acted: false,
			character_id: newEntry.is_player ? collaborationStore.currentParticipant?.user_id : undefined
		};
		
		// Add to list and sort
		const updatedList = [...initiatives, entry].sort((a, b) => b.initiative - a.initiative);
		
		// Update state
		await collaborationStore.updateInitiative(updatedList);
		
		// Reset form
		newEntry = {
			name: '',
			initiative: 0,
			is_player: true,
			current_hp: 0,
			max_hp: 0
		};
		isAddingEntry = false;
	}
	
	async function removeEntry(entry: InitiativeEntry) {
		if (!canManageInitiative) return;
		
		const updatedList = initiatives.filter(e => e.id !== entry.id);
		
		// Adjust current turn if needed
		if (currentTurn >= updatedList.length && updatedList.length > 0) {
			currentTurn = 0;
		}
		
		await collaborationStore.updateInitiative(updatedList);
	}
	
	async function nextTurn() {
		if (!canManageInitiative) return;
		
		// Mark current character as having acted
		if (initiatives[currentTurn]) {
			initiatives[currentTurn].has_acted = true;
		}
		
		// Request next turn from server (handles round increment)
		await collaborationStore.nextTurn();
	}
	
	async function previousTurn() {
		if (!canManageInitiative || currentTurn === 0) return;
		
		const newTurn = currentTurn - 1;
		const newRound = newTurn === 0 && currentRound > 1 ? currentRound - 1 : currentRound;
		
		// Send turn change
		collaborationStore.sendMessage({
			type: 'turn_changed',
			room_id: roomId,
			sender_id: collaborationStore.currentParticipant?.user_id || '',
			data: { turn: newTurn, round: newRound },
			timestamp: Date.now()
		});
	}
	
	async function resetInitiative() {
		if (!canManageInitiative) return;
		
		// Reset all has_acted flags
		const resetList = initiatives.map(e => ({ ...e, has_acted: false }));
		
		await collaborationStore.updateInitiative(resetList);
		
		// Reset turn and round
		collaborationStore.sendMessage({
			type: 'turn_changed',
			room_id: roomId,
			sender_id: collaborationStore.currentParticipant?.user_id || '',
			data: { turn: 0, round: 1 },
			timestamp: Date.now()
		});
	}
	
	async function updateHP(entry: InitiativeEntry, change: number) {
		if (!canManageInitiative && entry.character_id !== collaborationStore.currentParticipant?.user_id) {
			return;
		}
		
		entry.current_hp = Math.max(0, Math.min(entry.max_hp, entry.current_hp + change));
		
		await collaborationStore.updateInitiative(initiatives);
	}
	
	async function toggleCondition(entry: InitiativeEntry, condition: string) {
		if (!canManageInitiative && entry.character_id !== collaborationStore.currentParticipant?.user_id) {
			return;
		}
		
		const index = entry.conditions.indexOf(condition);
		if (index === -1) {
			entry.conditions = [...entry.conditions, condition];
		} else {
			entry.conditions = entry.conditions.filter(c => c !== condition);
		}
		
		await collaborationStore.updateInitiative(initiatives);
	}
	
	// Drag and drop
	function handleDragStart(event: DragEvent, entry: InitiativeEntry) {
		if (!canManageInitiative) return;
		
		draggedItem = entry;
		event.dataTransfer!.effectAllowed = 'move';
	}
	
	function handleDragOver(event: DragEvent, index: number) {
		event.preventDefault();
		event.dataTransfer!.dropEffect = 'move';
		dragOverIndex = index;
	}
	
	function handleDragLeave() {
		dragOverIndex = null;
	}
	
	async function handleDrop(event: DragEvent, targetIndex: number) {
		event.preventDefault();
		
		if (!draggedItem || !canManageInitiative) return;
		
		const draggedIndex = initiatives.findIndex(e => e.id === draggedItem!.id);
		if (draggedIndex === -1 || draggedIndex === targetIndex) return;
		
		// Reorder list
		const newList = [...initiatives];
		newList.splice(draggedIndex, 1);
		newList.splice(targetIndex, 0, draggedItem);
		
		// Adjust current turn if needed
		if (currentTurn === draggedIndex) {
			currentTurn = targetIndex;
		} else if (draggedIndex < currentTurn && targetIndex >= currentTurn) {
			currentTurn--;
		} else if (draggedIndex > currentTurn && targetIndex <= currentTurn) {
			currentTurn++;
		}
		
		await collaborationStore.updateInitiative(newList);
		
		draggedItem = null;
		dragOverIndex = null;
	}
	
	// Timer functions
	function startTurnTimer() {
		if (!enableTimer) return;
		
		// Clear existing timer
		if (timerInterval) {
			clearInterval(timerInterval);
		}
		
		turnTimer = turnDuration;
		
		timerInterval = window.setInterval(() => {
			if (turnTimer !== null && turnTimer > 0) {
				turnTimer--;
				
				// Warning at 10 seconds
				if (turnTimer === 10) {
					notifyTimeWarning();
				}
				
				// Auto-advance at 0
				if (turnTimer === 0 && canManageInitiative) {
					nextTurn();
				}
			}
		}, 1000);
	}
	
	function notifyUserTurn() {
		// Could use browser notifications or sound
		if ('Notification' in window && Notification.permission === 'granted') {
			new Notification('Your Turn!', {
				body: `It's ${initiatives[currentTurn]?.name}'s turn in combat.`,
				icon: '/favicon.png'
			});
		}
	}
	
	function notifyTimeWarning() {
		if (currentParticipantTurn) {
			// Flash or sound warning
			console.log('10 seconds remaining!');
		}
	}
	
	// Utility functions
	function getHPColor(current: number, max: number): string {
		const ratio = current / max;
		if (ratio > 0.5) return '#10b981';
		if (ratio > 0.25) return '#f59e0b';
		return '#ef4444';
	}
	
	function getConditionColor(condition: string): string {
		const colors: Record<string, string> = {
			'Poisoned': '#10b981',
			'Stunned': '#f59e0b',
			'Paralyzed': '#ef4444',
			'Unconscious': '#6b7280',
			'Invisible': '#8b5cf6',
			'Charmed': '#ec4899',
			'Frightened': '#3b82f6'
		};
		return colors[condition] || '#6b7280';
	}
	
	function rollInitiative() {
		return Math.floor(Math.random() * 20) + 1;
	}
</script>

<div class="initiative-tracker">
	<div class="tracker-header">
		<div class="header-info">
			<h3>Initiative Order</h3>
			<div class="round-info">
				Round {currentRound}
			</div>
		</div>
		
		{#if enableTimer && turnTimer !== null}
			<div class="turn-timer" class:warning={turnTimer <= 10}>
				<svg class="timer-icon" viewBox="0 0 20 20" fill="currentColor">
					<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd"/>
				</svg>
				<span>{turnTimer}s</span>
			</div>
		{/if}
		
		{#if canManageInitiative}
			<div class="header-actions">
				<button 
					class="action-btn"
					onclick={() => isAddingEntry = !isAddingEntry}
					title="Add Entry"
				>
					<svg viewBox="0 0 20 20" fill="currentColor">
						<path fill-rule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clip-rule="evenodd"/>
					</svg>
				</button>
				<button 
					class="action-btn"
					onclick={resetInitiative}
					title="Reset"
				>
					<svg viewBox="0 0 20 20" fill="currentColor">
						<path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"/>
					</svg>
				</button>
			</div>
		{/if}
	</div>
	
	<!-- Add entry form -->
	{#if isAddingEntry && canManageInitiative}
		<div class="add-entry-form">
			<input 
				type="text"
				bind:value={newEntry.name}
				placeholder="Character name"
				class="entry-input"
			/>
			<input 
				type="number"
				bind:value={newEntry.initiative}
				placeholder="Init"
				class="entry-input small"
			/>
			<button 
				class="roll-btn"
				onclick={() => newEntry.initiative = rollInitiative()}
				title="Roll d20"
			>
				⚅
			</button>
			<label class="player-toggle">
				<input 
					type="checkbox"
					bind:checked={newEntry.is_player}
				/>
				Player
			</label>
			{#if showHealthBars}
				<input 
					type="number"
					bind:value={newEntry.current_hp}
					placeholder="HP"
					class="entry-input small"
				/>
				<span>/</span>
				<input 
					type="number"
					bind:value={newEntry.max_hp}
					placeholder="Max"
					class="entry-input small"
				/>
			{/if}
			<button 
				class="add-btn"
				onclick={addInitiativeEntry}
			>
				Add
			</button>
			<button 
				class="cancel-btn"
				onclick={() => isAddingEntry = false}
			>
				Cancel
			</button>
		</div>
	{/if}
	
	<!-- Initiative list -->
	<div class="initiative-list">
		{#each initiatives as entry, index (entry.id)}
			<div 
				class="initiative-entry"
				class:current={index === currentTurn}
				class:has-acted={entry.has_acted}
				class:player={entry.is_player}
				class:unconscious={entry.current_hp === 0}
				class:drag-over={dragOverIndex === index}
				draggable={canManageInitiative}
				ondragstart={(e) => handleDragStart(e, entry)}
				ondragover={(e) => handleDragOver(e, index)}
				ondragleave={handleDragLeave}
				ondrop={(e) => handleDrop(e, index)}
				animate:flip={{ duration: 300 }}
			>
				{#if index === currentTurn}
					<div class="turn-indicator">▶</div>
				{/if}
				
				<div class="entry-main">
					<div class="entry-header">
						<span class="entry-name" class:player-name={entry.is_player}>
							{entry.name}
						</span>
						<span class="entry-initiative">{entry.initiative}</span>
					</div>
					
					{#if showHealthBars && entry.max_hp > 0}
						<div class="hp-container">
							<div class="hp-bar">
								<div 
									class="hp-fill"
									style="width: {(entry.current_hp / entry.max_hp) * 100}%; background: {getHPColor(entry.current_hp, entry.max_hp)}"
								></div>
							</div>
							<div class="hp-controls">
								<button 
									class="hp-btn"
									onclick={() => updateHP(entry, -1)}
								>
									-
								</button>
								<span class="hp-text">
									{entry.current_hp}/{entry.max_hp}
								</span>
								<button 
									class="hp-btn"
									onclick={() => updateHP(entry, 1)}
								>
									+
								</button>
							</div>
						</div>
					{/if}
					
					{#if showConditions && entry.conditions.length > 0}
						<div class="conditions-list">
							{#each entry.conditions as condition}
								<span 
									class="condition-tag"
									style="background: {getConditionColor(condition)}20; border-color: {getConditionColor(condition)}"
									onclick={() => toggleCondition(entry, condition)}
								>
									{condition}
								</span>
							{/each}
						</div>
					{/if}
				</div>
				
				<div class="entry-actions">
					{#if showConditions}
						<div class="condition-menu">
							<button 
								class="condition-btn"
								title="Add Condition"
							>
								+
							</button>
							<div class="condition-dropdown">
								{#each conditions as condition}
									<button 
										class="condition-option"
										class:active={entry.conditions.includes(condition)}
										onclick={() => toggleCondition(entry, condition)}
									>
										{condition}
									</button>
								{/each}
							</div>
						</div>
					{/if}
					
					{#if canManageInitiative}
						<button 
							class="remove-btn"
							onclick={() => removeEntry(entry)}
							title="Remove"
						>
							×
						</button>
					{/if}
				</div>
			</div>
		{/each}
	</div>
	
	<!-- Turn controls -->
	{#if canManageInitiative}
		<div class="turn-controls">
			<button 
				class="turn-btn"
				onclick={previousTurn}
				disabled={currentTurn === 0 && currentRound === 1}
			>
				Previous
			</button>
			<button 
				class="turn-btn primary"
				onclick={nextTurn}
				disabled={initiatives.length === 0}
			>
				Next Turn
			</button>
		</div>
	{/if}
</div>

<style>
	.initiative-tracker {
		display: flex;
		flex-direction: column;
		height: 100%;
		background: var(--color-surface);
		border-radius: 0.5rem;
		overflow: hidden;
	}
	
	.tracker-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 1rem;
		background: var(--color-surface-secondary);
		border-bottom: 1px solid var(--color-border);
	}
	
	.header-info {
		display: flex;
		align-items: center;
		gap: 1rem;
	}
	
	.header-info h3 {
		margin: 0;
		font-size: 1.125rem;
		font-weight: 600;
	}
	
	.round-info {
		padding: 0.25rem 0.75rem;
		background: var(--color-primary);
		color: white;
		border-radius: 1rem;
		font-size: 0.875rem;
		font-weight: 600;
	}
	
	.turn-timer {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.25rem 0.75rem;
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		font-variant-numeric: tabular-nums;
		font-weight: 600;
	}
	
	.turn-timer.warning {
		background: var(--color-warning-bg);
		border-color: var(--color-warning);
		color: var(--color-warning-text);
		animation: pulse 1s infinite;
	}
	
	@keyframes pulse {
		0%, 100% { transform: scale(1); }
		50% { transform: scale(1.05); }
	}
	
	.timer-icon {
		width: 1rem;
		height: 1rem;
	}
	
	.header-actions {
		display: flex;
		gap: 0.5rem;
	}
	
	.action-btn {
		width: 2rem;
		height: 2rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.action-btn:hover {
		background: var(--color-primary);
		color: white;
		border-color: var(--color-primary);
	}
	
	.action-btn svg {
		width: 1rem;
		height: 1rem;
	}
	
	.add-entry-form {
		display: flex;
		gap: 0.5rem;
		padding: 1rem;
		background: var(--color-surface-secondary);
		border-bottom: 1px solid var(--color-border);
	}
	
	.entry-input {
		padding: 0.5rem;
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		font-size: 0.875rem;
	}
	
	.entry-input.small {
		width: 60px;
	}
	
	.roll-btn {
		padding: 0.5rem;
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		cursor: pointer;
		font-size: 1.25rem;
	}
	
	.roll-btn:hover {
		background: var(--color-primary);
		color: white;
		border-color: var(--color-primary);
	}
	
	.player-toggle {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.875rem;
	}
	
	.add-btn,
	.cancel-btn {
		padding: 0.5rem 1rem;
		border-radius: 0.375rem;
		font-size: 0.875rem;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.add-btn {
		background: var(--color-primary);
		color: white;
		border: none;
	}
	
	.add-btn:hover {
		background: var(--color-primary-hover);
	}
	
	.cancel-btn {
		background: transparent;
		color: var(--color-text-secondary);
		border: 1px solid var(--color-border);
	}
	
	.cancel-btn:hover {
		background: var(--color-surface);
	}
	
	.initiative-list {
		flex: 1;
		overflow-y: auto;
		padding: 0.5rem;
	}
	
	.initiative-entry {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.75rem;
		margin-bottom: 0.5rem;
		background: white;
		border: 2px solid transparent;
		border-radius: 0.5rem;
		position: relative;
		transition: all 0.3s;
		cursor: move;
	}
	
	.initiative-entry.current {
		border-color: var(--color-primary);
		background: var(--color-primary-alpha);
		box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
	}
	
	.initiative-entry.has-acted {
		opacity: 0.6;
	}
	
	.initiative-entry.player {
		border-left: 4px solid var(--color-success);
	}
	
	.initiative-entry.unconscious {
		background: var(--color-error-alpha);
		opacity: 0.5;
	}
	
	.initiative-entry.drag-over {
		border-color: var(--color-primary);
		background: var(--color-primary-alpha);
	}
	
	.turn-indicator {
		position: absolute;
		left: -0.75rem;
		color: var(--color-primary);
		font-size: 1.25rem;
		animation: pulse 2s infinite;
	}
	
	.entry-main {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	
	.entry-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	
	.entry-name {
		font-weight: 600;
		font-size: 0.9375rem;
	}
	
	.entry-name.player-name {
		color: var(--color-success);
	}
	
	.entry-initiative {
		padding: 0.125rem 0.5rem;
		background: var(--color-surface-secondary);
		border-radius: 0.25rem;
		font-size: 0.875rem;
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}
	
	.hp-container {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	
	.hp-bar {
		flex: 1;
		height: 0.5rem;
		background: var(--color-surface-secondary);
		border-radius: 0.25rem;
		overflow: hidden;
	}
	
	.hp-fill {
		height: 100%;
		transition: width 0.3s, background 0.3s;
	}
	
	.hp-controls {
		display: flex;
		align-items: center;
		gap: 0.25rem;
	}
	
	.hp-btn {
		width: 1.25rem;
		height: 1.25rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--color-surface-secondary);
		border: 1px solid var(--color-border);
		border-radius: 0.25rem;
		cursor: pointer;
		font-weight: 600;
		transition: all 0.2s;
	}
	
	.hp-btn:hover {
		background: var(--color-primary);
		color: white;
		border-color: var(--color-primary);
	}
	
	.hp-text {
		font-size: 0.75rem;
		font-variant-numeric: tabular-nums;
		min-width: 3rem;
		text-align: center;
	}
	
	.conditions-list {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
	}
	
	.condition-tag {
		padding: 0.125rem 0.375rem;
		border: 1px solid;
		border-radius: 0.25rem;
		font-size: 0.75rem;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.condition-tag:hover {
		transform: scale(1.05);
	}
	
	.entry-actions {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	
	.condition-menu {
		position: relative;
	}
	
	.condition-btn {
		width: 1.5rem;
		height: 1.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--color-surface-secondary);
		border: 1px solid var(--color-border);
		border-radius: 0.25rem;
		cursor: pointer;
		font-weight: 600;
		transition: all 0.2s;
	}
	
	.condition-btn:hover {
		background: var(--color-primary);
		color: white;
		border-color: var(--color-primary);
	}
	
	.condition-dropdown {
		position: absolute;
		right: 0;
		top: 100%;
		margin-top: 0.25rem;
		padding: 0.5rem;
		background: white;
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
		display: none;
		z-index: 10;
		min-width: 150px;
	}
	
	.condition-menu:hover .condition-dropdown {
		display: block;
	}
	
	.condition-option {
		display: block;
		width: 100%;
		padding: 0.25rem 0.5rem;
		background: transparent;
		border: none;
		text-align: left;
		font-size: 0.875rem;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.condition-option:hover {
		background: var(--color-surface-secondary);
	}
	
	.condition-option.active {
		background: var(--color-primary-alpha);
		color: var(--color-primary);
		font-weight: 600;
	}
	
	.remove-btn {
		width: 1.5rem;
		height: 1.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: transparent;
		border: none;
		color: var(--color-text-secondary);
		font-size: 1.25rem;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.remove-btn:hover {
		color: var(--color-error);
	}
	
	.turn-controls {
		display: flex;
		gap: 0.5rem;
		padding: 1rem;
		background: var(--color-surface-secondary);
		border-top: 1px solid var(--color-border);
	}
	
	.turn-btn {
		flex: 1;
		padding: 0.75rem;
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		font-weight: 600;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.turn-btn:hover {
		background: var(--color-surface-secondary);
	}
	
	.turn-btn.primary {
		background: var(--color-primary);
		color: white;
		border-color: var(--color-primary);
	}
	
	.turn-btn.primary:hover {
		background: var(--color-primary-hover);
	}
	
	.turn-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	
	/* Additional CSS variables */
	:global(:root) {
		--color-success: #10b981;
		--color-warning: #f59e0b;
		--color-warning-bg: #fef3c7;
		--color-warning-text: #92400e;
		--color-error: #ef4444;
		--color-error-alpha: rgba(239, 68, 68, 0.1);
	}
</style>