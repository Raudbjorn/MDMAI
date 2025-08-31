<script lang="ts">
	import { onMount, onDestroy, tick } from 'svelte';
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { ChatMessage, DiceRoll, Participant } from '$lib/types/collaboration';
	import DOMPurify from 'dompurify';
	
	interface Props {
		roomId: string;
		maxMessages?: number;
		showTypingIndicator?: boolean;
		enableCommands?: boolean;
		enableWhispers?: boolean;
	}
	
	let {
		roomId,
		maxMessages = 200,
		showTypingIndicator = true,
		enableCommands = true,
		enableWhispers = true
	}: Props = $props();
	
	// State
	let messages = $state<ChatMessage[]>([]);
	let inputValue = $state('');
	let isTyping = $state(false);
	let typingUsers = $state<Map<string, { name: string; timeout: number }>>(new Map());
	let showEmojiPicker = $state(false);
	let whisperTarget = $state<string | null>(null);
	let unreadCount = $state(0);
	let isScrolledToBottom = $state(true);
	let showUserList = $state(false);
	let mentionSuggestions = $state<Participant[]>([]);
	let commandSuggestions = $state<string[]>([]);
	let currentSuggestionIndex = $state(0);
	
	// References
	let messagesContainer: HTMLDivElement;
	let inputEl: HTMLTextAreaElement;
	let unsubscribe: (() => void)[] = [];
	
	// Commands
	const commands = [
		{ name: '/roll', description: 'Roll dice', usage: '/roll 2d6+3' },
		{ name: '/whisper', description: 'Send private message', usage: '/whisper @user message' },
		{ name: '/me', description: 'Emote action', usage: '/me does something' },
		{ name: '/ooc', description: 'Out of character', usage: '/ooc message' },
		{ name: '/clear', description: 'Clear chat', usage: '/clear' },
		{ name: '/help', description: 'Show commands', usage: '/help' }
	];
	
	// Emojis
	const emojis = ['😀', '😄', '😊', '😎', '🤔', '👍', '👎', '❤️', '🎲', '⚔️', '🛡️', '🏹', '🧙‍♂️', '🐉', '💀', '🔥'];
	
	onMount(() => {
		// Subscribe to chat messages
		unsubscribe.push(
			collaborationStore.onMessage('chat_message', (msg) => {
				handleNewMessage(msg.data as ChatMessage);
			})
		);
		
		// Subscribe to dice rolls
		unsubscribe.push(
			collaborationStore.onMessage('dice_roll', (msg) => {
				handleDiceRoll(msg.data as DiceRoll);
			})
		);
		
		// Subscribe to typing indicators
		if (showTypingIndicator) {
			unsubscribe.push(
				collaborationStore.onMessage('typing_indicator', (msg) => {
					handleTypingIndicator(msg.sender_id, msg.data.typing);
				})
			);
		}
		
		// Load initial messages
		loadMessages();
		
		// Setup keyboard shortcuts
		document.addEventListener('keydown', handleGlobalKeyDown);
	});
	
	onDestroy(() => {
		unsubscribe.forEach(unsub => unsub());
		document.removeEventListener('keydown', handleGlobalKeyDown);
		
		// Clear typing indicator
		if (isTyping) {
			sendTypingIndicator(false);
		}
	});
	
	function loadMessages() {
		messages = collaborationStore.messages.slice(-maxMessages);
		scrollToBottom();
	}
	
	function handleNewMessage(message: ChatMessage) {
		messages = [...messages.slice(-(maxMessages - 1)), message];
		
		// Check for mentions
		if (message.content.includes(`@${collaborationStore.currentParticipant?.username}`)) {
			notifyMention(message);
		}
		
		// Update unread count if not scrolled to bottom
		if (!isScrolledToBottom && message.sender_id !== collaborationStore.currentParticipant?.user_id) {
			unreadCount++;
		}
		
		// Auto-scroll if at bottom
		if (isScrolledToBottom) {
			tick().then(scrollToBottom);
		}
	}
	
	function handleDiceRoll(roll: DiceRoll) {
		// Create a special message for dice rolls
		const rollMessage: ChatMessage = {
			id: `roll-${roll.id}`,
			room_id: roomId,
			sender_id: roll.player_id,
			sender_name: roll.player_name,
			content: `🎲 Rolled ${roll.expression}${roll.purpose ? ` for ${roll.purpose}` : ''}: **${roll.total}** [${roll.results.join(', ')}]`,
			type: 'roll',
			timestamp: roll.timestamp
		};
		
		handleNewMessage(rollMessage);
	}
	
	function handleTypingIndicator(userId: string, typing: boolean) {
		if (typing) {
			const participant = collaborationStore.participants.find(p => p.user_id === userId);
			if (participant) {
				typingUsers.set(userId, {
					name: participant.username,
					timeout: window.setTimeout(() => {
						typingUsers.delete(userId);
						typingUsers = new Map(typingUsers);
					}, 3000)
				});
				typingUsers = new Map(typingUsers);
			}
		} else {
			const user = typingUsers.get(userId);
			if (user) {
				clearTimeout(user.timeout);
				typingUsers.delete(userId);
				typingUsers = new Map(typingUsers);
			}
		}
	}
	
	async function sendMessage() {
		if (!inputValue.trim()) return;
		
		// Check for commands
		if (enableCommands && inputValue.startsWith('/')) {
			await handleCommand(inputValue);
		} else {
			// Check for whisper mode
			const type = whisperTarget ? 'whisper' : 'text';
			const content = whisperTarget ? `@${whisperTarget} ${inputValue}` : inputValue;
			
			await collaborationStore.sendChatMessage(content, type);
		}
		
		inputValue = '';
		whisperTarget = null;
		sendTypingIndicator(false);
		scrollToBottom();
	}
	
	async function handleCommand(input: string) {
		const [command, ...args] = input.split(' ');
		const argString = args.join(' ');
		
		switch (command) {
			case '/roll':
				if (argString) {
					await collaborationStore.rollDice(argString);
				} else {
					addSystemMessage('Usage: /roll <dice expression> (e.g., /roll 2d6+3)');
				}
				break;
				
			case '/whisper':
			case '/w':
				const match = argString.match(/^@?(\S+)\s+(.+)$/);
				if (match) {
					const [, target, message] = match;
					whisperTarget = target;
					inputValue = message;
					await sendMessage();
				} else {
					addSystemMessage('Usage: /whisper @username message');
				}
				break;
				
			case '/me':
				if (argString) {
					await collaborationStore.sendChatMessage(
						`*${collaborationStore.currentParticipant?.username} ${argString}*`,
						'text'
					);
				}
				break;
				
			case '/ooc':
				if (argString) {
					await collaborationStore.sendChatMessage(
						`((OOC: ${argString}))`,
						'text'
					);
				}
				break;
				
			case '/clear':
				messages = [];
				addSystemMessage('Chat cleared locally');
				break;
				
			case '/help':
				showHelp();
				break;
				
			default:
				addSystemMessage(`Unknown command: ${command}`);
		}
	}
	
	function addSystemMessage(content: string) {
		const message: ChatMessage = {
			id: crypto.randomUUID(),
			room_id: roomId,
			sender_id: 'system',
			sender_name: 'System',
			content,
			type: 'system',
			timestamp: new Date().toISOString()
		};
		handleNewMessage(message);
	}
	
	function showHelp() {
		const helpText = commands.map(cmd => 
			`**${cmd.name}** - ${cmd.description}\n  Usage: \`${cmd.usage}\``
		).join('\n\n');
		
		addSystemMessage(`Available commands:\n\n${helpText}`);
	}
	
	function handleInput() {
		// Check for @ mentions
		const text = inputValue;
		const lastAtIndex = text.lastIndexOf('@');
		
		if (lastAtIndex !== -1 && lastAtIndex === text.length - 1 || 
			(lastAtIndex !== -1 && text[lastAtIndex + 1] && !text[lastAtIndex + 1].match(/\s/))) {
			const searchTerm = text.slice(lastAtIndex + 1);
			mentionSuggestions = collaborationStore.participants
				.filter(p => p.username.toLowerCase().startsWith(searchTerm.toLowerCase()))
				.slice(0, 5);
		} else {
			mentionSuggestions = [];
		}
		
		// Check for command suggestions
		if (enableCommands && text.startsWith('/')) {
			const searchTerm = text.slice(1).split(' ')[0];
			commandSuggestions = commands
				.filter(cmd => cmd.name.slice(1).startsWith(searchTerm))
				.map(cmd => cmd.name)
				.slice(0, 5);
		} else {
			commandSuggestions = [];
		}
		
		// Handle typing indicator
		if (showTypingIndicator) {
			if (!isTyping && text.length > 0) {
				isTyping = true;
				sendTypingIndicator(true);
			} else if (isTyping && text.length === 0) {
				isTyping = false;
				sendTypingIndicator(false);
			}
		}
	}
	
	function sendTypingIndicator(typing: boolean) {
		collaborationStore.sendMessage({
			type: 'typing_indicator',
			room_id: roomId,
			sender_id: collaborationStore.currentParticipant?.user_id || '',
			data: { typing },
			timestamp: Date.now()
		});
	}
	
	function insertEmoji(emoji: string) {
		const start = inputEl.selectionStart;
		const end = inputEl.selectionEnd;
		const newValue = inputValue.slice(0, start) + emoji + inputValue.slice(end);
		inputValue = newValue;
		
		// Reset cursor position
		tick().then(() => {
			inputEl.selectionStart = inputEl.selectionEnd = start + emoji.length;
			inputEl.focus();
		});
		
		showEmojiPicker = false;
	}
	
	function insertMention(participant: Participant) {
		const text = inputValue;
		const lastAtIndex = text.lastIndexOf('@');
		const newValue = text.slice(0, lastAtIndex) + `@${participant.username} `;
		inputValue = newValue;
		mentionSuggestions = [];
		inputEl.focus();
	}
	
	function insertCommand(command: string) {
		inputValue = command + ' ';
		commandSuggestions = [];
		inputEl.focus();
	}
	
	function handleKeyDown(event: KeyboardEvent) {
		// Handle suggestions navigation
		if ((mentionSuggestions.length > 0 || commandSuggestions.length > 0) && 
			(event.key === 'ArrowUp' || event.key === 'ArrowDown' || event.key === 'Enter' || event.key === 'Tab')) {
			event.preventDefault();
			
			const suggestions = mentionSuggestions.length > 0 ? mentionSuggestions : commandSuggestions;
			const maxIndex = suggestions.length - 1;
			
			switch (event.key) {
				case 'ArrowUp':
					currentSuggestionIndex = Math.max(0, currentSuggestionIndex - 1);
					break;
				case 'ArrowDown':
					currentSuggestionIndex = Math.min(maxIndex, currentSuggestionIndex + 1);
					break;
				case 'Enter':
				case 'Tab':
					if (mentionSuggestions.length > 0) {
						insertMention(mentionSuggestions[currentSuggestionIndex]);
					} else if (commandSuggestions.length > 0) {
						insertCommand(commandSuggestions[currentSuggestionIndex]);
					}
					currentSuggestionIndex = 0;
					break;
			}
			return;
		}
		
		// Send on Enter (without Shift)
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			sendMessage();
		}
	}
	
	function handleGlobalKeyDown(event: KeyboardEvent) {
		// Focus input on slash key
		if (event.key === '/' && document.activeElement !== inputEl) {
			event.preventDefault();
			inputEl.focus();
		}
	}
	
	function handleScroll() {
		if (!messagesContainer) return;
		
		const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
		isScrolledToBottom = scrollHeight - scrollTop - clientHeight < 50;
		
		// Clear unread count when scrolled to bottom
		if (isScrolledToBottom) {
			unreadCount = 0;
		}
	}
	
	function scrollToBottom() {
		if (messagesContainer) {
			messagesContainer.scrollTop = messagesContainer.scrollHeight;
			unreadCount = 0;
		}
	}
	
	function notifyMention(message: ChatMessage) {
		// Browser notification
		if ('Notification' in window && Notification.permission === 'granted') {
			new Notification(`${message.sender_name} mentioned you`, {
				body: message.content,
				icon: '/favicon.png'
			});
		}
		
		// Could also play a sound
	}
	
	function formatTime(timestamp: string): string {
		const date = new Date(timestamp);
		return date.toLocaleTimeString('en-US', { 
			hour: 'numeric', 
			minute: '2-digit',
			hour12: true 
		});
	}
	
	function formatMessage(content: string): string {
		// Escape HTML first to prevent XSS
		const div = document.createElement('div');
		div.textContent = content;
		content = div.innerHTML;
		
		// Format bold text (using escaped asterisks)
		content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
		
		// Format italic text (single asterisk, but not if it's part of bold)
		content = content.replace(/(?<!\*)\*(?!\*)(.*?)\*(?!\*)/g, '<em>$1</em>');
		
		// Format mentions
		content = content.replace(/@(\w+)/g, '<span class="mention">@$1</span>');
		
		// Format dice rolls
		content = content.replace(/\[(\d+(?:,\s*\d+)*)\]/g, '<span class="dice-results">[$1]</span>');
		
		// Configure DOMPurify to allow only safe tags and attributes
		const cleanHTML = DOMPurify.sanitize(content, {
			ALLOWED_TAGS: ['strong', 'em', 'span'],
			ALLOWED_ATTR: ['class'],
			ALLOWED_CLASSES: {
				span: ['mention', 'dice-results']
			},
			KEEP_CONTENT: true,
			RETURN_TRUSTED_TYPE: false
		});
		
		return cleanHTML;
	}
	
	function getMessageClass(message: ChatMessage): string {
		const classes = ['chat-message'];
		
		if (message.sender_id === collaborationStore.currentParticipant?.user_id) {
			classes.push('own-message');
		}
		
		if (message.type === 'system') {
			classes.push('system-message');
		} else if (message.type === 'roll') {
			classes.push('roll-message');
		} else if (message.type === 'whisper') {
			classes.push('whisper-message');
		}
		
		if (message.content.includes(`@${collaborationStore.currentParticipant?.username}`)) {
			classes.push('mentioned');
		}
		
		return classes.join(' ');
	}
</script>

<div class="enhanced-chat">
	<!-- Chat header -->
	<div class="chat-header">
		<h3>Chat</h3>
		<div class="header-actions">
			{#if unreadCount > 0}
				<span class="unread-badge">{unreadCount}</span>
			{/if}
			<button 
				class="header-btn"
				onclick={() => showUserList = !showUserList}
				title="Toggle user list"
			>
				<svg viewBox="0 0 20 20" fill="currentColor">
					<path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z"/>
				</svg>
			</button>
		</div>
	</div>
	
	<!-- Messages container -->
	<div 
		class="messages-container"
		bind:this={messagesContainer}
		onscroll={handleScroll}
	>
		{#each messages as message}
			<div class={getMessageClass(message)}>
				{#if message.type !== 'system'}
					<div class="message-header">
						<span class="sender-name" style="color: {getParticipantColor(message.sender_id)}">
							{message.sender_name}
						</span>
						<span class="message-time">{formatTime(message.timestamp)}</span>
					</div>
				{/if}
				<div class="message-content">
					{@html formatMessage(message.content)}
				</div>
			</div>
		{/each}
		
		{#if !isScrolledToBottom && messages.length > 0}
			<button 
				class="scroll-to-bottom"
				onclick={scrollToBottom}
			>
				<svg viewBox="0 0 20 20" fill="currentColor">
					<path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
				</svg>
				{#if unreadCount > 0}
					<span class="unread-count">{unreadCount}</span>
				{/if}
			</button>
		{/if}
	</div>
	
	<!-- Typing indicator -->
	{#if showTypingIndicator && typingUsers.size > 0}
		<div class="typing-indicator">
			{#if typingUsers.size === 1}
				<span>{[...typingUsers.values()][0].name} is typing...</span>
			{:else if typingUsers.size === 2}
				<span>{[...typingUsers.values()].map(u => u.name).join(' and ')} are typing...</span>
			{:else}
				<span>{typingUsers.size} people are typing...</span>
			{/if}
			<div class="typing-dots">
				<span></span>
				<span></span>
				<span></span>
			</div>
		</div>
	{/if}
	
	<!-- Input area -->
	<div class="input-area">
		{#if whisperTarget}
			<div class="whisper-indicator">
				Whispering to @{whisperTarget}
				<button onclick={() => whisperTarget = null}>×</button>
			</div>
		{/if}
		
		<!-- Suggestions -->
		{#if mentionSuggestions.length > 0}
			<div class="suggestions mention-suggestions">
				{#each mentionSuggestions as participant, index}
					<button 
						class="suggestion-item"
						class:active={index === currentSuggestionIndex}
						onclick={() => insertMention(participant)}
					>
						<span class="user-avatar" style="background: {participant.color}">
							{participant.username.charAt(0).toUpperCase()}
						</span>
						<span>{participant.username}</span>
					</button>
				{/each}
			</div>
		{/if}
		
		{#if commandSuggestions.length > 0}
			<div class="suggestions command-suggestions">
				{#each commandSuggestions as command, index}
					<button 
						class="suggestion-item"
						class:active={index === currentSuggestionIndex}
						onclick={() => insertCommand(command)}
					>
						<span class="command-icon">/</span>
						<span>{command.slice(1)}</span>
					</button>
				{/each}
			</div>
		{/if}
		
		<div class="input-row">
			<button 
				class="input-btn"
				onclick={() => showEmojiPicker = !showEmojiPicker}
				title="Emojis"
			>
				😊
			</button>
			
			<textarea
				bind:this={inputEl}
				bind:value={inputValue}
				placeholder={whisperTarget ? `Whisper to @${whisperTarget}...` : "Type a message..."}
				class="message-input"
				rows="1"
				oninput={handleInput}
				onkeydown={handleKeyDown}
			></textarea>
			
			<button 
				class="send-btn"
				onclick={sendMessage}
				disabled={!inputValue.trim()}
			>
				<svg viewBox="0 0 20 20" fill="currentColor">
					<path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"/>
				</svg>
			</button>
		</div>
		
		<!-- Emoji picker -->
		{#if showEmojiPicker}
			<div class="emoji-picker">
				{#each emojis as emoji}
					<button 
						class="emoji-btn"
						onclick={() => insertEmoji(emoji)}
					>
						{emoji}
					</button>
				{/each}
			</div>
		{/if}
	</div>
	
	<!-- User list sidebar -->
	{#if showUserList}
		<div class="user-list">
			<h4>Participants</h4>
			{#each collaborationStore.participants as participant}
				<div class="user-item">
					<span 
						class="user-status"
						class:online={participant.status === 'online'}
						class:away={participant.status === 'away'}
						class:offline={participant.status === 'offline'}
					></span>
					<span class="user-name">{participant.username}</span>
					{#if participant.role === 'host'}
						<span class="user-role">Host</span>
					{:else if participant.role === 'gm'}
						<span class="user-role">GM</span>
					{/if}
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.enhanced-chat {
		display: flex;
		flex-direction: column;
		height: 100%;
		background: var(--color-surface);
		border-radius: 0.5rem;
		overflow: hidden;
		position: relative;
	}
	
	.chat-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 1rem;
		background: var(--color-surface-secondary);
		border-bottom: 1px solid var(--color-border);
	}
	
	.chat-header h3 {
		margin: 0;
		font-size: 1.125rem;
		font-weight: 600;
	}
	
	.header-actions {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	
	.unread-badge {
		padding: 0.125rem 0.375rem;
		background: var(--color-error);
		color: white;
		border-radius: 1rem;
		font-size: 0.75rem;
		font-weight: 600;
	}
	
	.header-btn {
		width: 2rem;
		height: 2rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: transparent;
		border: none;
		color: var(--color-text-secondary);
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.header-btn:hover {
		color: var(--color-primary);
	}
	
	.header-btn svg {
		width: 1.25rem;
		height: 1.25rem;
	}
	
	.messages-container {
		flex: 1;
		overflow-y: auto;
		padding: 1rem;
		scroll-behavior: smooth;
	}
	
	.chat-message {
		margin-bottom: 1rem;
		animation: slideIn 0.3s ease-out;
	}
	
	@keyframes slideIn {
		from {
			opacity: 0;
			transform: translateY(10px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}
	
	.chat-message.own-message {
		text-align: right;
	}
	
	.chat-message.own-message .message-header {
		justify-content: flex-end;
	}
	
	.chat-message.system-message {
		text-align: center;
		font-style: italic;
		color: var(--color-text-secondary);
		font-size: 0.875rem;
	}
	
	.chat-message.roll-message .message-content {
		background: var(--color-primary-alpha);
		border-left: 3px solid var(--color-primary);
		padding: 0.5rem;
		border-radius: 0.25rem;
	}
	
	.chat-message.whisper-message {
		opacity: 0.7;
		font-style: italic;
	}
	
	.chat-message.mentioned {
		background: var(--color-warning-bg);
		padding: 0.5rem;
		border-radius: 0.25rem;
		border-left: 3px solid var(--color-warning);
	}
	
	.message-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.25rem;
	}
	
	.sender-name {
		font-weight: 600;
		font-size: 0.875rem;
	}
	
	.message-time {
		font-size: 0.75rem;
		color: var(--color-text-secondary);
	}
	
	.message-content {
		word-wrap: break-word;
		line-height: 1.5;
	}
	
	.message-content :global(strong) {
		font-weight: 600;
	}
	
	.message-content :global(em) {
		font-style: italic;
	}
	
	.message-content :global(.mention) {
		color: var(--color-primary);
		font-weight: 600;
		cursor: pointer;
	}
	
	.message-content :global(.mention:hover) {
		text-decoration: underline;
	}
	
	.message-content :global(.dice-results) {
		font-family: 'Monaco', 'Menlo', monospace;
		color: var(--color-primary);
		font-weight: 600;
	}
	
	.scroll-to-bottom {
		position: absolute;
		bottom: 7rem;
		right: 1rem;
		width: 2.5rem;
		height: 2.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--color-primary);
		color: white;
		border: none;
		border-radius: 50%;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.scroll-to-bottom:hover {
		transform: scale(1.1);
	}
	
	.scroll-to-bottom svg {
		width: 1.25rem;
		height: 1.25rem;
	}
	
	.scroll-to-bottom .unread-count {
		position: absolute;
		top: -0.25rem;
		right: -0.25rem;
		padding: 0.125rem 0.375rem;
		background: var(--color-error);
		border-radius: 1rem;
		font-size: 0.75rem;
		font-weight: 600;
	}
	
	.typing-indicator {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 1rem;
		background: var(--color-surface-secondary);
		border-top: 1px solid var(--color-border);
		font-size: 0.875rem;
		color: var(--color-text-secondary);
		font-style: italic;
	}
	
	.typing-dots {
		display: flex;
		gap: 0.25rem;
	}
	
	.typing-dots span {
		width: 0.5rem;
		height: 0.5rem;
		background: var(--color-text-secondary);
		border-radius: 50%;
		animation: bounce 1.4s infinite;
	}
	
	.typing-dots span:nth-child(2) {
		animation-delay: 0.2s;
	}
	
	.typing-dots span:nth-child(3) {
		animation-delay: 0.4s;
	}
	
	@keyframes bounce {
		0%, 60%, 100% {
			transform: translateY(0);
		}
		30% {
			transform: translateY(-0.5rem);
		}
	}
	
	.input-area {
		padding: 1rem;
		background: var(--color-surface-secondary);
		border-top: 1px solid var(--color-border);
	}
	
	.whisper-indicator {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.5rem;
		margin-bottom: 0.5rem;
		background: var(--color-primary-alpha);
		border-radius: 0.25rem;
		font-size: 0.875rem;
		color: var(--color-primary);
	}
	
	.whisper-indicator button {
		background: transparent;
		border: none;
		color: var(--color-primary);
		font-size: 1.25rem;
		cursor: pointer;
	}
	
	.suggestions {
		position: absolute;
		bottom: 100%;
		left: 1rem;
		right: 1rem;
		margin-bottom: 0.5rem;
		padding: 0.5rem;
		background: white;
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.1);
		max-height: 200px;
		overflow-y: auto;
		z-index: 10;
	}
	
	.suggestion-item {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		width: 100%;
		padding: 0.5rem;
		background: transparent;
		border: none;
		text-align: left;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.suggestion-item:hover,
	.suggestion-item.active {
		background: var(--color-surface-secondary);
	}
	
	.user-avatar {
		width: 1.5rem;
		height: 1.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		color: white;
		border-radius: 50%;
		font-size: 0.75rem;
		font-weight: 600;
	}
	
	.command-icon {
		width: 1.5rem;
		height: 1.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--color-primary);
		color: white;
		border-radius: 0.25rem;
		font-weight: 600;
	}
	
	.input-row {
		display: flex;
		gap: 0.5rem;
		align-items: flex-end;
	}
	
	.input-btn {
		width: 2.5rem;
		height: 2.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		font-size: 1.25rem;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.input-btn:hover {
		background: var(--color-primary-alpha);
		border-color: var(--color-primary);
	}
	
	.message-input {
		flex: 1;
		padding: 0.5rem;
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		font-family: inherit;
		font-size: 0.9375rem;
		resize: none;
		min-height: 2.5rem;
		max-height: 6rem;
		transition: all 0.2s;
	}
	
	.message-input:focus {
		outline: none;
		border-color: var(--color-primary);
		box-shadow: 0 0 0 3px var(--color-primary-alpha);
	}
	
	.send-btn {
		width: 2.5rem;
		height: 2.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--color-primary);
		color: white;
		border: none;
		border-radius: 0.375rem;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.send-btn:hover {
		background: var(--color-primary-hover);
		transform: scale(1.05);
	}
	
	.send-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
		transform: none;
	}
	
	.send-btn svg {
		width: 1.25rem;
		height: 1.25rem;
	}
	
	.emoji-picker {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
		margin-top: 0.5rem;
		padding: 0.5rem;
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
	}
	
	.emoji-btn {
		width: 2rem;
		height: 2rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: transparent;
		border: none;
		font-size: 1.25rem;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.emoji-btn:hover {
		background: var(--color-surface-secondary);
		border-radius: 0.25rem;
		transform: scale(1.2);
	}
	
	.user-list {
		position: absolute;
		right: 0;
		top: 3.5rem;
		bottom: 0;
		width: 200px;
		padding: 1rem;
		background: var(--color-surface-secondary);
		border-left: 1px solid var(--color-border);
		overflow-y: auto;
		z-index: 5;
	}
	
	.user-list h4 {
		margin: 0 0 1rem;
		font-size: 0.875rem;
		font-weight: 600;
		color: var(--color-text-secondary);
	}
	
	.user-item {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem;
		margin-bottom: 0.25rem;
		border-radius: 0.25rem;
		transition: all 0.2s;
	}
	
	.user-item:hover {
		background: var(--color-surface);
	}
	
	.user-status {
		width: 0.5rem;
		height: 0.5rem;
		border-radius: 50%;
		background: var(--color-text-secondary);
	}
	
	.user-status.online {
		background: var(--color-success);
	}
	
	.user-status.away {
		background: var(--color-warning);
	}
	
	.user-status.offline {
		background: var(--color-text-secondary);
	}
	
	.user-name {
		flex: 1;
		font-size: 0.875rem;
	}
	
	.user-role {
		padding: 0.125rem 0.375rem;
		background: var(--color-primary);
		color: white;
		border-radius: 0.25rem;
		font-size: 0.625rem;
		font-weight: 600;
		text-transform: uppercase;
	}
	
	/* Helper function for participant colors */
	:global(:root) {
		--color-primary-alpha: rgba(59, 130, 246, 0.1);
		--color-warning-bg: #fef3c7;
		--color-warning: #f59e0b;
		--color-warning-text: #92400e;
	}
</style>

<script module lang="ts">
	function getParticipantColor(userId: string): string {
		// Generate a consistent color based on user ID
		const colors = [
			'#3b82f6', '#10b981', '#f59e0b', '#ef4444',
			'#8b5cf6', '#ec4899', '#14b8a6', '#f97316'
		];
		const index = userId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
		return colors[index % colors.length];
	}
</script>