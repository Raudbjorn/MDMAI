<script lang="ts">
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import { sessionStore } from '$lib/stores/session.svelte';
	import RoomManager from './RoomManager.svelte';
	import ParticipantList from './ParticipantList.svelte';
	import InviteManager from './InviteManager.svelte';
	import TurnManager from './TurnManager.svelte';
	import ChatPanel from './ChatPanel.svelte';
	import type { CollaborativeRoom, ConflictResolution } from '$lib/types/collaboration';
	import type { Result } from '$lib/api/types';
	import { onMount, onDestroy } from 'svelte';
	import { toast } from 'svelte-sonner';
	
	interface Props {
		campaignId: string;
		campaignName?: string;
	}
	
	let { campaignId, campaignName = 'Campaign' }: Props = $props();
	
	// Reactive state from stores
	const currentRoom = $derived(collaborationStore.currentRoom);
	const isConnected = $derived(collaborationStore.isConnected);
	const isSyncing = $derived(collaborationStore.isSyncing);
	const conflicts = $derived(collaborationStore.conflicts);
	const participants = $derived(collaborationStore.participants);
	const currentUser = $derived(sessionStore.user);
	
	// Component state
	let showCursors = $state(true);
	let isFullscreen = $state(false);
	let keyboardShortcutsEnabled = $state(true);
	let screenReaderAnnouncements = $state('');
	let activeMobileTab = $state<'session' | 'participants' | 'chat'>('session');
	
	// TTRPG-specific keyboard shortcuts
	const keyboardShortcuts = {
		'KeyR': () => rollD20(), // R for d20 roll
		'KeyN': () => focusSharedNotes(), // N for notes
		'KeyT': () => nextTurn(), // T for turn
		'KeyI': () => focusInitiative(), // I for initiative
		'KeyC': () => focusChat(), // C for chat
		'KeyF': () => toggleFullscreen(), // F for fullscreen
		'Escape': () => handleEscape() // ESC for various cancel actions
	};

	// Track mouse movements for cursor sharing
	function handleMouseMove(event: MouseEvent) {
		if (!showCursors || !currentRoom) return;
		
		collaborationStore.updateCursor(
			event.clientX,
			event.clientY,
			(event.target as HTMLElement)?.id
		);
	}

	// Keyboard event handler for TTRPG shortcuts
	function handleKeyDown(event: KeyboardEvent) {
		if (!keyboardShortcutsEnabled || event.repeat) return;
		
		// Only trigger shortcuts when not in input/textarea
		if (event.target instanceof HTMLInputElement || 
			event.target instanceof HTMLTextAreaElement ||
			event.target instanceof HTMLSelectElement) {
			return;
		}
		
		const shortcut = keyboardShortcuts[event.code as keyof typeof keyboardShortcuts];
		if (shortcut) {
			event.preventDefault();
			shortcut();
		}
	}
	
	// TTRPG-specific functions
	async function rollD20() {
		if (!currentRoom) return;
		try {
			const roll = await collaborationStore.rollDice('1d20', 'Quick d20 roll');
			announceToScreenReader(`${currentUser?.username || 'Player'} rolled d20: ${roll.total}`);
			toast.success(`Rolled d20: ${roll.total}`, {
				description: `Result: ${roll.results.join(', ')}`
			});
		} catch (error) {
			toast.error('Failed to roll dice', {
				description: error instanceof Error ? error.message : 'Unknown error'
			});
		}
	}
	
	function focusSharedNotes() {
		const notesTextarea = document.querySelector('[data-shared-notes]') as HTMLTextAreaElement;
		if (notesTextarea) {
			notesTextarea.focus();
			announceToScreenReader('Focused shared notes');
		}
	}
	
	function nextTurn() {
		if (!currentRoom) return;
		collaborationStore.nextTurn();
		announceToScreenReader('Requested next turn');
	}
	
	function focusInitiative() {
		const initiativeElement = document.querySelector('[data-initiative]') as HTMLElement;
		if (initiativeElement) {
			initiativeElement.focus();
			announceToScreenReader('Focused initiative tracker');
		}
	}
	
	function focusChat() {
		const chatInput = document.querySelector('[data-chat-input]') as HTMLInputElement;
		if (chatInput) {
			chatInput.focus();
			announceToScreenReader('Focused chat input');
		}
	}
	
	function toggleFullscreen() {
		isFullscreen = !isFullscreen;
		announceToScreenReader(isFullscreen ? 'Entered fullscreen mode' : 'Exited fullscreen mode');
	}
	
	function handleEscape() {
		// Clear focus from inputs, close modals, etc.
		if (document.activeElement instanceof HTMLElement) {
			document.activeElement.blur();
		}
		isFullscreen = false;
	}
	
	// Screen reader announcements
	function announceToScreenReader(message: string) {
		screenReaderAnnouncements = message;
		// Clear after a delay to prevent repeated announcements
		setTimeout(() => {
			screenReaderAnnouncements = '';
		}, 1000);
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
	
	// Mobile tab navigation handlers
	function handleMobileTabClick(tab: 'session' | 'participants' | 'chat') {
		activeMobileTab = tab;
		announceToScreenReader(`Switched to ${tab} tab`);
	}
	
	function handleMobileTabKeyDown(event: KeyboardEvent, tab: 'session' | 'participants' | 'chat') {
		const tabs = ['session', 'participants', 'chat'] as const;
		const currentIndex = tabs.indexOf(tab);
		
		switch (event.key) {
			case 'ArrowLeft':
			case 'ArrowUp':
				event.preventDefault();
				const prevIndex = currentIndex > 0 ? currentIndex - 1 : tabs.length - 1;
				handleMobileTabClick(tabs[prevIndex]);
				// Focus the new active tab
				setTimeout(() => {
					const currentTarget = event.currentTarget as HTMLElement;
					const tabElement = currentTarget?.parentElement?.children[prevIndex] as HTMLElement;
					tabElement?.focus();
				}, 0);
				break;
			case 'ArrowRight':
			case 'ArrowDown':
				event.preventDefault();
				const nextIndex = currentIndex < tabs.length - 1 ? currentIndex + 1 : 0;
				handleMobileTabClick(tabs[nextIndex]);
				// Focus the new active tab
				setTimeout(() => {
					const currentTarget = event.currentTarget as HTMLElement;
					const tabElement = currentTarget?.parentElement?.children[nextIndex] as HTMLElement;
					tabElement?.focus();
				}, 0);
				break;
			case 'Home':
				event.preventDefault();
				handleMobileTabClick('session');
				setTimeout(() => {
					const currentTarget = event.currentTarget as HTMLElement;
					const tabElement = currentTarget?.parentElement?.children[0] as HTMLElement;
					tabElement?.focus();
				}, 0);
				break;
			case 'End':
				event.preventDefault();
				handleMobileTabClick('chat');
				setTimeout(() => {
					const currentTarget = event.currentTarget as HTMLElement;
					const tabElement = currentTarget?.parentElement?.children[2] as HTMLElement;
					tabElement?.focus();
				}, 0);
				break;
		}
	}
	
	// Enhanced conflict resolution with Result pattern
	async function resolveConflict(conflictIndex: number, resolution: 'accept' | 'reject'): Promise<Result<void, string>> {
		const conflict = conflicts[conflictIndex];
		if (!conflict) {
			return { ok: false, error: 'Conflict not found' };
		}
		
		try {
			if (resolution === 'accept') {
				// Apply the conflicting update with proper validation
				for (const update of conflict.conflicting_updates) {
					await collaborationStore.updateState(update);
				}
				announceToScreenReader('Conflict resolved by accepting changes');
				toast.success('Conflict resolved', {
					description: 'Changes have been applied successfully'
				});
			} else {
				announceToScreenReader('Conflict resolved by rejecting changes');
				toast.info('Conflict resolved', {
					description: 'Changes have been rejected'
				});
			}
			
			// Remove from conflicts list
			collaborationStore.conflicts.splice(conflictIndex, 1);
			
			return { ok: true, value: undefined };
		} catch (error) {
			const errorMsg = error instanceof Error ? error.message : 'Unknown error resolving conflict';
			toast.error('Failed to resolve conflict', { description: errorMsg });
			return { ok: false, error: errorMsg };
		}
	}
	
	onMount(() => {
		// Add global event listeners
		document.addEventListener('mousemove', handleMouseMove);
		document.addEventListener('keydown', handleKeyDown);
		
		// Connect to collaboration server with error handling
		if (currentUser && !isConnected) {
			collaborationStore.connect(currentUser.id).catch(error => {
				const errorMsg = error instanceof Error ? error.message : 'Failed to connect';
				toast.error('Connection failed', { description: errorMsg });
				announceToScreenReader('Failed to connect to collaborative session');
			});
		}
		
		// Announce keyboard shortcuts to screen reader
		announceToScreenReader(
			'Collaborative session loaded. Press R for dice roll, T for next turn, N for notes, C for chat, F for fullscreen.'
		);
	});
	
	onDestroy(() => {
		// Clean up event listeners
		document.removeEventListener('mousemove', handleMouseMove);
		document.removeEventListener('keydown', handleKeyDown);
		
		// Disconnect from collaboration server
		if (isConnected) {
			collaborationStore.disconnect();
		}
	});
</script>

<!-- Screen reader announcements -->
<div aria-live="polite" aria-atomic="true" class="sr-only">
	{screenReaderAnnouncements}
</div>

<!-- Render cursors reactively without direct DOM manipulation -->
{#each cursorsData() as cursor (cursor.userId)}
	<div 
		class="fixed pointer-events-none z-50 transition-all duration-75 participant-cursor"
		style="transform: translate({cursor.x}px, {cursor.y}px); color: {cursor.color || '#3b82f6'}"
		role="img"
		aria-label="Cursor position for {cursor.username}"
	>
		<svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
			<path d="M5 3L17 9L11 11L9 17L3 5L5 3Z" fill="currentColor"/>
		</svg>
		<span class="absolute top-5 left-5 text-xs bg-gray-900 text-white px-2 py-1 rounded shadow-lg whitespace-nowrap border border-gray-700">
			{cursor.username}
		</span>
	</div>
{/each}

<div class={`container mx-auto p-4 ${isFullscreen ? 'fixed inset-0 z-40 bg-white dark:bg-gray-950' : ''}`}>
	<!-- Header with improved semantics and responsive design -->
	<header class="mb-4">
		<div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
			<h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
				{campaignName} - Collaborative Session
			</h1>
			
			<!-- Control Panel -->
			<div class="flex items-center gap-3">
				<button
					class="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
					onclick={() => keyboardShortcutsEnabled = !keyboardShortcutsEnabled}
					title={keyboardShortcutsEnabled ? 'Disable keyboard shortcuts' : 'Enable keyboard shortcuts'}
					aria-label={keyboardShortcutsEnabled ? 'Disable keyboard shortcuts' : 'Enable keyboard shortcuts'}
				>
					⌨️
				</button>
				
				<button
					class="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
					onclick={rollD20}
					title="Quick d20 roll (R)"
					aria-label="Roll d20 dice"
				>
					🎲
				</button>
				
				<button
					class="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
					onclick={toggleFullscreen}
					title={isFullscreen ? 'Exit fullscreen (F)' : 'Enter fullscreen (F)'}
					aria-label={isFullscreen ? 'Exit fullscreen mode' : 'Enter fullscreen mode'}
				>
					{isFullscreen ? '⛶' : '⛶'}
				</button>
			</div>
		</div>
		
		<!-- Status Bar with improved accessibility -->
		<div class="flex flex-wrap items-center gap-4 mt-3 text-sm" role="status" aria-live="polite">
			<div class="flex items-center gap-2" aria-label="Connection status">
				<div 
					class={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}
					role="img"
					aria-label={isConnected ? 'Connected to server' : 'Disconnected from server'}
				></div>
				<span class={`font-medium ${isConnected ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}>
					{isConnected ? 'Connected' : 'Disconnected'}
				</span>
			</div>
			
			{#if isSyncing}
				<div class="flex items-center gap-2" aria-label="Synchronization status">
					<div class="w-3 h-3 bg-blue-500 rounded-full animate-pulse" role="img" aria-label="Syncing data"></div>
					<span class="text-blue-700 dark:text-blue-400 font-medium">Syncing...</span>
				</div>
			{/if}
			
			{#if currentRoom}
				<div class="flex items-center gap-2 text-gray-600 dark:text-gray-400">
					<span class="font-medium">Room:</span>
					<span>{currentRoom.name}</span>
					<span class="text-xs bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded-full">
						{participants.length} {participants.length === 1 ? 'participant' : 'participants'}
					</span>
				</div>
			{/if}
			
			<label class="flex items-center gap-2 cursor-pointer select-none">
				<input 
					type="checkbox" 
					bind:checked={showCursors}
					class="sr-only"
					aria-describedby="cursor-toggle-description"
				/>
				<div class={`relative w-10 h-6 rounded-full transition-colors ${showCursors ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'}`}>
					<div class={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${showCursors ? 'translate-x-4' : 'translate-x-0'}`}></div>
				</div>
				<span class="text-sm font-medium" id="cursor-toggle-description">
					Show cursors
				</span>
			</label>
		</div>
		
		<!-- Keyboard shortcuts hint -->
		{#if keyboardShortcutsEnabled}
			<div class="mt-2 text-xs text-gray-500 dark:text-gray-400 flex flex-wrap gap-x-4 gap-y-1">
				<span>Shortcuts:</span>
				<kbd class="px-1.5 py-0.5 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">R</kbd>
				<span>Roll d20</span>
				<kbd class="px-1.5 py-0.5 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">T</kbd>
				<span>Next turn</span>
				<kbd class="px-1.5 py-0.5 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">N</kbd>
				<span>Notes</span>
				<kbd class="px-1.5 py-0.5 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">C</kbd>
				<span>Chat</span>
			</div>
		{/if}
	</header>
	
	<!-- Enhanced Conflict Resolution Banner -->
	{#if conflicts.length > 0}
		<div 
			class="mb-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg"
			role="alert"
			aria-live="assertive"
		>
			<div class="flex items-center gap-2 font-medium text-sm mb-3 text-yellow-800 dark:text-yellow-200">
				<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
					<path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
				</svg>
				{conflicts.length} conflict{conflicts.length === 1 ? '' : 's'} detected:
			</div>
			{#each conflicts as conflict, index}
				<div class="flex flex-col sm:flex-row sm:justify-between sm:items-center p-3 bg-white dark:bg-gray-800 rounded-lg mb-2 shadow-sm">
					<div class="mb-2 sm:mb-0">
						<p class="text-sm font-medium text-gray-900 dark:text-gray-100">
							Update conflict on {conflict.conflicting_updates[0]?.path.join('.') || 'unknown field'}
						</p>
						<p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
							{conflict.conflicting_updates.length} conflicting update{conflict.conflicting_updates.length === 1 ? '' : 's'}
						</p>
					</div>
					<div class="flex gap-2">
						<button 
							class="px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-xs font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
							onclick={() => resolveConflict(index, 'accept')}
							aria-label="Accept conflicting changes"
						>
							Accept
						</button>
						<button 
							class="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
							onclick={() => resolveConflict(index, 'reject')}
							aria-label="Reject conflicting changes"
						>
							Reject
						</button>
					</div>
				</div>
			{/each}
		</div>
	{/if}
	
	{#if !currentRoom}
		<!-- Room Selection/Creation with improved accessibility -->
		<section aria-labelledby="room-selection-heading">
			<h2 id="room-selection-heading" class="sr-only">Select or create a room</h2>
			<RoomManager 
				{campaignId}
				onRoomCreated={handleRoomCreated}
				onRoomJoined={handleRoomJoined}
			/>
		</section>
	{:else}
		<!-- Active Session Layout with responsive design -->
		<main class="space-y-6">
			<!-- Mobile Tab Navigation -->
			<div class="lg:hidden">
				<div class="flex space-x-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1" role="tablist" aria-label="Mobile navigation tabs">
					<button 
						class="flex-1 py-2 px-3 text-sm font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 {activeMobileTab === 'session' ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm' : 'text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100'}"
						role="tab"
						aria-selected={activeMobileTab === 'session'}
						tabindex={activeMobileTab === 'session' ? 0 : -1}
						onclick={() => handleMobileTabClick('session')}
						onkeydown={(event) => handleMobileTabKeyDown(event, 'session')}
					>
						Session
					</button>
					<button 
						class="flex-1 py-2 px-3 text-sm font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 {activeMobileTab === 'participants' ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm' : 'text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100'}"
						role="tab"
						aria-selected={activeMobileTab === 'participants'}
						tabindex={activeMobileTab === 'participants' ? 0 : -1}
						onclick={() => handleMobileTabClick('participants')}
						onkeydown={(event) => handleMobileTabKeyDown(event, 'participants')}
					>
						Participants
					</button>
					<button 
						class="flex-1 py-2 px-3 text-sm font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 {activeMobileTab === 'chat' ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm' : 'text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100'}"
						role="tab"
						aria-selected={activeMobileTab === 'chat'}
						tabindex={activeMobileTab === 'chat' ? 0 : -1}
						onclick={() => handleMobileTabClick('chat')}
						onkeydown={(event) => handleMobileTabKeyDown(event, 'chat')}
					>
						Chat
					</button>
				</div>
			</div>

			<!-- Desktop and Mobile Layout -->
			<div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
				<!-- Left Sidebar - Participants & Invites -->
				<aside class="lg:col-span-3 space-y-4 {activeMobileTab === 'participants' ? 'block' : 'hidden lg:block'}" aria-label="Participants and room management">
					<section aria-labelledby="participants-heading">
						<h2 id="participants-heading" class="sr-only">Participants</h2>
						<ParticipantList 
							showActions={true}
							onParticipantClick={handleParticipantClick}
						/>
					</section>
					
					<section aria-labelledby="invite-heading">
						<h2 id="invite-heading" class="sr-only">Invite participants</h2>
						<InviteManager 
							roomId={currentRoom.id}
							roomName={currentRoom.name}
						/>
					</section>
				</aside>
				
				<!-- Main Content - Initiative & Notes -->
				<section class="lg:col-span-6 space-y-6 {activeMobileTab === 'session' ? 'block' : 'hidden lg:block'}" aria-label="Game session content">
					<!-- Initiative Tracker -->
					<div data-initiative tabindex="-1">
						<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">Initiative Tracker</h2>
						<TurnManager 
							editable={true}
							onTurnChange={handleTurnChange}
						/>
					</div>
					
					<!-- Shared Notes Area with enhanced accessibility -->
					<div class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-white dark:bg-gray-800">
						<label for="shared-notes" class="block text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
							Shared Session Notes
						</label>
						<textarea
							id="shared-notes"
							data-shared-notes
							class="w-full h-40 p-3 border border-gray-300 dark:border-gray-600 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-gray-100 transition-colors"
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
							aria-describedby="shared-notes-help"
						></textarea>
						<p id="shared-notes-help" class="mt-2 text-xs text-gray-500 dark:text-gray-400">
							Notes are automatically saved and shared with all participants. Press N to focus this field.
						</p>
					</div>
					
					<!-- Quick Actions Panel -->
					<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
						<h3 class="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Quick Actions</h3>
						<div class="flex flex-wrap gap-2">
							<button
								class="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
								onclick={rollD20}
								aria-describedby="roll-d20-help"
							>
								🎲 Roll d20
							</button>
							<button
								class="px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
								onclick={nextTurn}
								aria-describedby="next-turn-help"
							>
								⏭️ Next Turn
							</button>
							<button
								class="px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2"
								onclick={focusChat}
								aria-describedby="focus-chat-help"
							>
								💬 Chat
							</button>
						</div>
						<div class="mt-2 space-y-1 text-xs text-gray-500 dark:text-gray-400">
							<p id="roll-d20-help">Press R to roll d20 quickly</p>
							<p id="next-turn-help">Press T to advance turn</p>
							<p id="focus-chat-help">Press C to focus chat input</p>
						</div>
					</div>
					
					<!-- Leave Room Button -->
					<button
						class="w-full py-3 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
						onclick={() => {
							if (confirm('Are you sure you want to leave this session?')) {
								collaborationStore.leaveRoom();
								announceToScreenReader('Left collaborative session');
							}
						}}
						aria-describedby="leave-session-warning"
					>
						🚪 Leave Session
					</button>
					<p id="leave-session-warning" class="text-xs text-gray-500 dark:text-gray-400 text-center">
						You'll need to rejoin or be invited to participate again
					</p>
				</section>
				
				<!-- Right Sidebar - Chat -->
				<aside class="lg:col-span-3 {activeMobileTab === 'chat' ? 'block' : 'hidden lg:block'}" aria-label="Chat panel">
					<div class="h-[600px] flex flex-col">
						<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">Chat</h2>
						<div class="flex-1">
							<ChatPanel />
						</div>
					</div>
				</aside>
			</div>
		</main>
	{/if}
</div>

<style>
	/* Cursor animation */
	:global(.participant-cursor) {
		transition: transform 75ms linear;
	}
</style>