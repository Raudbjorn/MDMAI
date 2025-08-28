<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { StateUpdate, Participant } from '$lib/types/collaboration';
	import { debounce } from '$lib/utils/debounce';
	
	interface Props {
		roomId: string;
		initialContent?: string;
		placeholder?: string;
		maxLength?: number;
		showPresence?: boolean;
		readOnly?: boolean;
	}
	
	let { 
		roomId, 
		initialContent = '', 
		placeholder = 'Start typing your shared notes...',
		maxLength = 10000,
		showPresence = true,
		readOnly = false
	}: Props = $props();
	
	// State using Svelte 5 runes
	let content = $state(initialContent);
	let isEditing = $state(false);
	let localVersion = $state(0);
	let remoteVersion = $state(0);
	let hasConflict = $state(false);
	let editingUsers = $state<Map<string, Participant>>(new Map());
	let selectionRanges = $state<Map<string, {start: number, end: number}>>(new Map());
	let lastSyncTime = $state(Date.now());
	let isSyncing = $state(false);
	
	// References
	let textareaEl: HTMLTextAreaElement;
	let unsubscribe: (() => void) | null = null;
	
	// Derived values
	let characterCount = $derived(content.length);
	let characterLimit = $derived(maxLength - characterCount);
	let isOverLimit = $derived(characterCount > maxLength);
	
	// Get current user permissions
	let canEdit = $derived(
		!readOnly && collaborationStore.hasPermission('write', 'notes')
	);
	
	// Active editors display
	let activeEditors = $derived(
		Array.from(editingUsers.values())
			.filter(p => p.user_id !== collaborationStore.currentParticipant?.user_id)
	);
	
	onMount(() => {
		// Subscribe to state updates
		unsubscribe = collaborationStore.onMessage('state_update', (msg) => {
			if (msg.data.path[0] === 'shared_notes') {
				handleRemoteUpdate(msg.data as StateUpdate);
			}
		});
		
		// Load initial content from room state
		const room = collaborationStore.currentRoom;
		if (room) {
			content = room.state.shared_notes || initialContent;
			remoteVersion = room.state.version;
			localVersion = room.state.version;
		}
		
		// Setup selection tracking
		if (textareaEl && showPresence) {
			setupSelectionTracking();
		}
	});
	
	onDestroy(() => {
		unsubscribe?.();
		if (isEditing) {
			notifyEditingStatus(false);
		}
	});
	
	// Handle remote updates with conflict detection
	function handleRemoteUpdate(update: StateUpdate) {
		// Check if this is our own update echoing back
		if (update.version === localVersion) {
			return;
		}
		
		// Detect conflict if we have local changes
		if (localVersion > remoteVersion && isEditing) {
			hasConflict = true;
			// Store remote version for potential merge
			remoteVersion = update.version;
			return;
		}
		
		// Apply remote update
		content = update.value;
		remoteVersion = update.version;
		localVersion = update.version;
		hasConflict = false;
		lastSyncTime = Date.now();
	}
	
	// Debounced content synchronization
	const syncContent = debounce(async () => {
		if (!canEdit || isSyncing) return;
		
		isSyncing = true;
		localVersion = remoteVersion + 1;
		
		try {
			await collaborationStore.updateState({
				path: ['shared_notes'],
				value: content,
				operation: 'set',
				version: localVersion,
				previous_version: remoteVersion
			});
			
			remoteVersion = localVersion;
			hasConflict = false;
			lastSyncTime = Date.now();
		} catch (error) {
			console.error('Failed to sync notes:', error);
			hasConflict = true;
		} finally {
			isSyncing = false;
		}
	}, 500);
	
	// Handle content changes
	function handleInput(event: Event) {
		if (!canEdit) return;
		
		const target = event.target as HTMLTextAreaElement;
		content = target.value;
		
		// Enforce character limit
		if (isOverLimit) {
			content = content.slice(0, maxLength);
			target.value = content;
		}
		
		syncContent();
	}
	
	// Handle focus/blur for editing status
	function handleFocus() {
		if (!canEdit) return;
		isEditing = true;
		notifyEditingStatus(true);
	}
	
	function handleBlur() {
		isEditing = false;
		notifyEditingStatus(false);
		
		// Final sync on blur
		if (canEdit) {
			syncContent.flush();
		}
	}
	
	// Notify other users of editing status
	function notifyEditingStatus(editing: boolean) {
		collaborationStore.sendMessage({
			type: 'participant_status_changed',
			room_id: roomId,
			sender_id: collaborationStore.currentParticipant?.user_id || '',
			data: {
				user_id: collaborationStore.currentParticipant?.user_id,
				status: editing ? 'editing_notes' : 'online'
			},
			timestamp: Date.now()
		});
	}
	
	// Setup selection tracking for collaborative cursors
	function setupSelectionTracking() {
		const handleSelection = debounce(() => {
			if (!textareaEl || !showPresence) return;
			
			const start = textareaEl.selectionStart;
			const end = textareaEl.selectionEnd;
			
			collaborationStore.sendMessage({
				type: 'selection_change',
				room_id: roomId,
				sender_id: collaborationStore.currentParticipant?.user_id || '',
				data: {
					element: 'shared_notes',
					start,
					end
				},
				timestamp: Date.now()
			});
		}, 100);
		
		textareaEl.addEventListener('select', handleSelection);
		textareaEl.addEventListener('click', handleSelection);
		textareaEl.addEventListener('keyup', handleSelection);
		
		return () => {
			textareaEl.removeEventListener('select', handleSelection);
			textareaEl.removeEventListener('click', handleSelection);
			textareaEl.removeEventListener('keyup', handleSelection);
		};
	}
	
	// Resolve conflicts
	function acceptRemoteChanges() {
		content = collaborationStore.currentRoom?.state.shared_notes || '';
		localVersion = remoteVersion;
		hasConflict = false;
	}
	
	function keepLocalChanges() {
		hasConflict = false;
		syncContent.flush();
	}
	
	// Format relative time
	function formatRelativeTime(timestamp: number): string {
		const seconds = Math.floor((Date.now() - timestamp) / 1000);
		if (seconds < 60) return 'just now';
		const minutes = Math.floor(seconds / 60);
		if (minutes < 60) return `${minutes}m ago`;
		const hours = Math.floor(minutes / 60);
		if (hours < 24) return `${hours}h ago`;
		return `${Math.floor(hours / 24)}d ago`;
	}
	
	// Calculate selection position in the textarea
	function calculateSelectionPosition(charIndex: number): { x: number; y: number } {
		if (!textareaEl) return { x: 0, y: 0 };
		
		// Create a temporary element to measure text dimensions
		const temp = document.createElement('div');
		temp.style.position = 'absolute';
		temp.style.visibility = 'hidden';
		temp.style.whiteSpace = 'pre-wrap';
		temp.style.font = window.getComputedStyle(textareaEl).font;
		temp.style.width = textareaEl.clientWidth + 'px';
		temp.style.padding = window.getComputedStyle(textareaEl).padding;
		
		// Get text up to the selection point
		const textBeforeSelection = content.substring(0, charIndex);
		temp.textContent = textBeforeSelection;
		
		document.body.appendChild(temp);
		
		// Calculate position based on the temporary element
		const lines = textBeforeSelection.split('\n');
		const lastLine = lines[lines.length - 1];
		
		// Approximate position (this is simplified)
		const lineHeight = parseInt(window.getComputedStyle(textareaEl).lineHeight) || 20;
		const y = (lines.length - 1) * lineHeight;
		
		// Measure the last line width
		const tempLine = document.createElement('span');
		tempLine.style.font = window.getComputedStyle(textareaEl).font;
		tempLine.textContent = lastLine;
		document.body.appendChild(tempLine);
		const x = tempLine.offsetWidth;
		document.body.removeChild(tempLine);
		
		document.body.removeChild(temp);
		
		return { x, y };
	}
	
	// Calculate selection width
	function calculateSelectionWidth(range: { start: number; end: number }): number {
		if (!textareaEl || range.start === range.end) return 0;
		
		// Get selected text
		const selectedText = content.substring(range.start, range.end);
		
		// Create temporary element to measure width
		const temp = document.createElement('span');
		temp.style.position = 'absolute';
		temp.style.visibility = 'hidden';
		temp.style.font = window.getComputedStyle(textareaEl).font;
		temp.textContent = selectedText;
		
		document.body.appendChild(temp);
		const width = temp.offsetWidth;
		document.body.removeChild(temp);
		
		return Math.min(width, textareaEl.clientWidth);
	}
</script>

<div class="shared-notes">
	<!-- Header with status indicators -->
	<div class="notes-header">
		<div class="header-left">
			<h3 class="notes-title">Shared Notes</h3>
			{#if showPresence && activeEditors.length > 0}
				<div class="active-editors">
					{#each activeEditors as editor}
						<div 
							class="editor-indicator"
							style="background-color: {editor.color}"
							title="{editor.username} is editing"
						>
							{editor.username.charAt(0).toUpperCase()}
						</div>
					{/each}
				</div>
			{/if}
		</div>
		
		<div class="header-right">
			{#if isSyncing}
				<span class="sync-status syncing">
					<span class="sync-icon"></span>
					Saving...
				</span>
			{:else}
				<span class="sync-status saved">
					Saved {formatRelativeTime(lastSyncTime)}
				</span>
			{/if}
			
			<span class="character-count" class:over-limit={isOverLimit}>
				{characterCount} / {maxLength}
			</span>
		</div>
	</div>
	
	<!-- Conflict resolution banner -->
	{#if hasConflict}
		<div class="conflict-banner">
			<div class="conflict-message">
				<svg class="conflict-icon" viewBox="0 0 20 20" fill="currentColor">
					<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
				</svg>
				<span>Conflicting changes detected</span>
			</div>
			<div class="conflict-actions">
				<button class="conflict-btn keep-local" onclick={keepLocalChanges}>
					Keep my changes
				</button>
				<button class="conflict-btn accept-remote" onclick={acceptRemoteChanges}>
					Accept others' changes
				</button>
			</div>
		</div>
	{/if}
	
	<!-- Textarea with collaborative features -->
	<div class="notes-container">
		<textarea
			bind:this={textareaEl}
			bind:value={content}
			{placeholder}
			disabled={!canEdit}
			class="notes-textarea"
			class:editing={isEditing}
			class:read-only={!canEdit}
			oninput={handleInput}
			onfocus={handleFocus}
			onblur={handleBlur}
			spellcheck="true"
		></textarea>
		
		{#if showPresence}
			<!-- Render other users' selections as overlays -->
			{#each selectionRanges as [userId, range]}
				{#if userId !== collaborationStore.currentParticipant?.user_id}
					<div 
						class="selection-overlay"
						style="
							left: {calculateSelectionPosition(range.start).x}px;
							top: {calculateSelectionPosition(range.start).y}px;
							width: {calculateSelectionWidth(range)}px;
							background-color: {editingUsers.get(userId)?.color}20;
							border-color: {editingUsers.get(userId)?.color};
						"
					></div>
				{/if}
			{/each}
		{/if}
	</div>
	
	<!-- Read-only indicator -->
	{#if !canEdit}
		<div class="read-only-indicator">
			<svg class="lock-icon" viewBox="0 0 20 20" fill="currentColor">
				<path fill-rule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clip-rule="evenodd" />
			</svg>
			<span>View only</span>
		</div>
	{/if}
</div>

<style>
	.shared-notes {
		display: flex;
		flex-direction: column;
		height: 100%;
		background: var(--color-surface);
		border-radius: 0.5rem;
		overflow: hidden;
	}
	
	.notes-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 1rem;
		border-bottom: 1px solid var(--color-border);
		background: var(--color-surface-secondary);
	}
	
	.header-left {
		display: flex;
		align-items: center;
		gap: 1rem;
	}
	
	.notes-title {
		font-size: 1.125rem;
		font-weight: 600;
		margin: 0;
	}
	
	.active-editors {
		display: flex;
		gap: -0.5rem;
	}
	
	.editor-indicator {
		width: 1.75rem;
		height: 1.75rem;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		color: white;
		font-size: 0.75rem;
		font-weight: 600;
		border: 2px solid var(--color-surface);
		animation: pulse 2s infinite;
	}
	
	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.7; }
	}
	
	.header-right {
		display: flex;
		align-items: center;
		gap: 1rem;
	}
	
	.sync-status {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.875rem;
		color: var(--color-text-secondary);
	}
	
	.sync-status.syncing {
		color: var(--color-warning);
	}
	
	.sync-status.saved {
		color: var(--color-success);
	}
	
	.sync-icon {
		display: inline-block;
		width: 0.75rem;
		height: 0.75rem;
		border: 2px solid currentColor;
		border-right-color: transparent;
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}
	
	@keyframes spin {
		to { transform: rotate(360deg); }
	}
	
	.character-count {
		font-size: 0.875rem;
		color: var(--color-text-secondary);
		font-variant-numeric: tabular-nums;
	}
	
	.character-count.over-limit {
		color: var(--color-error);
		font-weight: 600;
	}
	
	.conflict-banner {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.75rem 1rem;
		background: var(--color-warning-bg);
		border-bottom: 1px solid var(--color-warning);
	}
	
	.conflict-message {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		color: var(--color-warning-text);
		font-size: 0.875rem;
		font-weight: 500;
	}
	
	.conflict-icon {
		width: 1.25rem;
		height: 1.25rem;
		color: var(--color-warning);
	}
	
	.conflict-actions {
		display: flex;
		gap: 0.5rem;
	}
	
	.conflict-btn {
		padding: 0.25rem 0.75rem;
		border-radius: 0.25rem;
		font-size: 0.875rem;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.keep-local {
		background: var(--color-primary);
		color: white;
		border: none;
	}
	
	.keep-local:hover {
		background: var(--color-primary-hover);
	}
	
	.accept-remote {
		background: transparent;
		color: var(--color-warning-text);
		border: 1px solid var(--color-warning);
	}
	
	.accept-remote:hover {
		background: var(--color-warning);
		color: white;
	}
	
	.notes-container {
		flex: 1;
		position: relative;
		padding: 1rem;
		overflow: hidden;
	}
	
	.notes-textarea {
		width: 100%;
		height: 100%;
		padding: 0.75rem;
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		background: var(--color-background);
		color: var(--color-text);
		font-family: inherit;
		font-size: 0.9375rem;
		line-height: 1.6;
		resize: none;
		transition: all 0.2s;
	}
	
	.notes-textarea:focus {
		outline: none;
		border-color: var(--color-primary);
		box-shadow: 0 0 0 3px var(--color-primary-alpha);
	}
	
	.notes-textarea.editing {
		border-color: var(--color-primary);
		background: white;
	}
	
	.notes-textarea.read-only {
		background: var(--color-surface-secondary);
		cursor: not-allowed;
		opacity: 0.7;
	}
	
	.selection-overlay {
		position: absolute;
		pointer-events: none;
		border-left: 2px solid;
		opacity: 0.5;
		transition: all 0.1s;
	}
	
	.read-only-indicator {
		position: absolute;
		bottom: 1rem;
		right: 1rem;
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.375rem 0.75rem;
		background: var(--color-surface-secondary);
		border-radius: 0.25rem;
		font-size: 0.75rem;
		color: var(--color-text-secondary);
	}
	
	.lock-icon {
		width: 1rem;
		height: 1rem;
	}

	/* CSS Variables (to be defined in global styles) */
	:global(:root) {
		--color-surface: #ffffff;
		--color-surface-secondary: #f9fafb;
		--color-background: #ffffff;
		--color-border: #e5e7eb;
		--color-text: #111827;
		--color-text-secondary: #6b7280;
		--color-primary: #3b82f6;
		--color-primary-hover: #2563eb;
		--color-primary-alpha: rgba(59, 130, 246, 0.1);
		--color-success: #10b981;
		--color-warning: #f59e0b;
		--color-warning-bg: #fef3c7;
		--color-warning-text: #92400e;
		--color-error: #ef4444;
	}
</style>