<script lang="ts">
		import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { DiceRoll, CollaborationMessage } from '$lib/types/collaboration';

	interface DicePreset {
		name: string;
		expression: string;
		icon: string;
		color: string;
	}

	interface Props {
		roomId: string; // Reserved for future room-specific features
		maxHistory?: number;
		showAnimation?: boolean;
		compactMode?: boolean;
	}

	let { roomId, maxHistory = 20, showAnimation = true, compactMode = false }: Props = $props();
	// Note: roomId is currently managed internally by collaborationStore
	// but kept in props for future room-specific configurations

	let customExpression = $state('');
	let selectedPreset = $state<DicePreset | null>(null);
	let rollHistory = $state<DiceRoll[]>([]);
	let isRolling = $state(false);
	let animatedResult = $state<number | null>(null);
	let showAdvanced = $state(false);
	let advantage = $state<'normal' | 'advantage' | 'disadvantage'>('normal');
	let modifier = $state(0);
	let criticalRange = $state(20);
	let rerollOnes = $state(false);
	let animatingDice = $state<{ value: number; final: number; id: string }[]>([]);

	const presets = [
		{ name: 'd4', expression: '1d4', icon: '△', color: '#f59e0b' },
		{ name: 'd6', expression: '1d6', icon: '□', color: '#10b981' },
		{ name: 'd8', expression: '1d8', icon: '◇', color: '#3b82f6' },
		{ name: 'd10', expression: '1d10', icon: '⬟', color: '#8b5cf6' },
		{ name: 'd12', expression: '1d12', icon: '⬢', color: '#ec4899' },
		{ name: 'd20', expression: '1d20', icon: '⬡', color: '#ef4444' },
		{ name: 'd100', expression: '1d100', icon: '%', color: '#6b7280' }
	];

	const purposes = ['Attack Roll', 'Damage', 'Saving Throw', 'Ability Check', 'Initiative', 'Death Save', 'Concentration', 'Custom'];
	let selectedPurpose = $state('Custom');
	$effect(() => {
		const unsubscribe = collaborationStore.onMessage('dice_roll', (msg) => {
			const roll = msg.data as DiceRoll;
			handleDiceRoll(roll);
		});
		const room = collaborationStore.currentRoom;
		if (room) rollHistory = room.state.dice_rolls.slice(-maxHistory);
		return () => unsubscribe?.();
	});
	
	function handleDiceRoll(roll: DiceRoll) {
		// Add to history
		rollHistory = [...rollHistory.slice(-(maxHistory - 1)), roll];
		
		// Trigger animation if it's our roll
		if (roll.player_id === collaborationStore.currentParticipant?.user_id && showAnimation) {
			animateRoll(roll);
		}
	}
	
	async function rollDice(expression?: string, purpose?: string) {
		if (isRolling) return;
		
		const finalExpression = expression || customExpression || selectedPreset?.expression || '1d20';
		const finalPurpose = purpose || (selectedPurpose !== 'Custom' ? selectedPurpose : undefined);
		
		// Apply advantage/disadvantage
		let modifiedExpression = finalExpression;
		if (advantage !== 'normal' && finalExpression.includes('d20')) {
			if (advantage === 'advantage') {
				modifiedExpression = finalExpression.replace(/(\d+)d20/, '2d20kh1');
			} else {
				modifiedExpression = finalExpression.replace(/(\d+)d20/, '2d20kl1');
			}
		}
		
		// Add modifier
		if (modifier !== 0) {
			modifiedExpression += modifier > 0 ? `+${modifier}` : `${modifier}`;
		}
		
		isRolling = true;
		
		try {
			const roll = await collaborationStore.rollDice(modifiedExpression, finalPurpose);
			
			// Check for critical
			if (finalExpression.includes('d20') && roll.results[0] >= criticalRange) {
				handleCritical(roll);
			}
			
			// Handle reroll ones - this creates a separate reroll entry
			if (rerollOnes && roll.results.some(r => r === 1)) {
				await rerollOnesInRoll(roll);
			}
		} catch (error) {
			// Handle roll error gracefully
			// Could show error message to user through a toast notification
		} finally {
			isRolling = false;
		}
	}
	
	function animateRoll(roll: DiceRoll) {
		if (!showAnimation) return;
		
		// Create animated dice for each result
		animatingDice = roll.results.map((final, index) => ({
			value: 0,
			final,
			id: `${roll.id}-${index}`
		}));
		
		// Animate each die
		animatingDice.forEach((die, index) => {
			animateDie(die, index * 100);
		});
		
		// Animate total
		animateTotal(roll.total);
	}
	
	function animateDie(die: { value: number; final: number; id: string }, delay: number) {
		const duration = 1000;
		const fps = 60;
		const frames = duration / (1000 / fps);
		let frame = 0;
		
		const animate = () => {
			if (frame < frames) {
				// Random value during animation
				if (frame < frames - 10) {
					die.value = Math.floor(Math.random() * 20) + 1;
				} else {
					// Slow down and show final value
					die.value = die.final;
				}
				
				frame++;
				// Trigger reactivity with minimal overhead
				animatingDice = animatingDice;
				requestAnimationFrame(animate);
			}
		};
		
		setTimeout(animate, delay);
	}
	
	function animateTotal(total: number) {
		const duration = 1500;
		const fps = 60;
		const frames = duration / (1000 / fps);
		const increment = total / frames;
		let current = 0;
		let frame = 0;
		
		const animate = () => {
			if (frame < frames) {
				current = Math.min(Math.floor(current + increment), total);
				animatedResult = current;
				frame++;
				requestAnimationFrame(animate);
			} else {
				animatedResult = total;
				// Clear animation after showing result
				setTimeout(() => {
					animatedResult = null;
					animatingDice = [];
				}, 2000);
			}
		};
		
		animate();
	}
	
	function handleCritical(roll: DiceRoll) {
			// Visual feedback for critical hits
		// Could implement confetti or other celebration animation here
		// Message would be: CRITICAL ${roll.results[0] === 20 ? 'SUCCESS' : 'HIT'}!
	}
	
	async function rerollOnesInRoll(roll: DiceRoll): Promise<boolean> {
		const hasOnes = roll.results.some(r => r === 1);
		if (!hasOnes) return false;
		
		// Count how many 1s to reroll and construct reroll expression
		const onesToReroll = roll.results.filter(r => r === 1).length;
		
		// Parse the original expression to determine die type
		const diceMatch = roll.expression.match(/(\d+)d(\d+)/);
		if (!diceMatch) return false;
		
		const [, , dieSize] = diceMatch;
		const rerollExpression = `${onesToReroll}d${dieSize}`;
		
		try {
			// Perform the actual reroll
			const rerollResult = await collaborationStore.rollDice(
				rerollExpression, 
				`Reroll 1s from ${roll.expression}`
			);
			return true;
		} catch (error) {
			// Handle reroll error gracefully
			// Could show error message to user through a toast notification
			return false;
		}
	}
	
	function quickRoll(preset: DicePreset) {
		selectedPreset = preset;
		rollDice(preset.expression);
	}
	
	function formatRollTime(timestamp: string): string {
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
	
	function getRollColor(total: number, expression: string): string {
		if (expression.includes('d20')) {
			if (total >= 20) return '#10b981'; // Critical
			if (total === 1) return '#ef4444'; // Critical fail
			if (total >= 15) return '#3b82f6'; // Good
			if (total >= 10) return '#f59e0b'; // Average
			return '#6b7280'; // Poor
		}
		return '#6b7280';
	}
	
	// Utility function for parsing dice expressions - kept for potential future use
	// Currently not used but may be useful for advanced expression validation
	function parseExpression(expression: string): { dice: string[]; modifier: number } {
		const dicePattern = /(\d+d\d+)/g;
		const modPattern = /([+-]\d+)(?!d)/g;
		
		const dice = expression.match(dicePattern) || [];
		const mods = expression.match(modPattern) || [];
		const modifier = mods.reduce((sum, mod) => sum + parseInt(mod), 0);
		
		return { dice, modifier };
	}
</script>

<div class="shared-dice-roller" class:compact={compactMode}>
	{#if !compactMode}
		<div class="roller-header">
			<h3>Dice Roller</h3>
			<button 
				class="advanced-toggle"
				onclick={() => showAdvanced = !showAdvanced}
				aria-label="Toggle advanced options"
				aria-expanded={showAdvanced}
			>
				{showAdvanced ? 'Simple' : 'Advanced'}
			</button>
		</div>
	{/if}
	
	<!-- Quick roll presets -->
	<div class="dice-presets">
		{#each presets as preset}
			<button 
				class="dice-btn"
				class:selected={selectedPreset?.name === preset.name}
				style="--dice-color: {preset.color}"
				onclick={() => quickRoll(preset)}
				disabled={isRolling}
				aria-label={`Roll ${preset.name}`}
			>
				<span class="dice-icon">{preset.icon}</span>
				<span class="dice-label">{preset.name}</span>
			</button>
		{/each}
	</div>
	
	<!-- Advanced options -->
	{#if showAdvanced && !compactMode}
		<div class="advanced-options">
			<div class="option-group">
				<label>Advantage</label>
				<div class="button-group" role="group" aria-label="Advantage selection">
					<button 
						class="option-btn"
						class:active={advantage === 'normal'}
						onclick={() => advantage = 'normal'}
						aria-label="Normal roll"
						aria-pressed={advantage === 'normal'}
					>
						Normal
					</button>
					<button 
						class="option-btn"
						class:active={advantage === 'advantage'}
						onclick={() => advantage = 'advantage'}
						aria-label="Roll with advantage"
						aria-pressed={advantage === 'advantage'}
					>
						Adv
					</button>
					<button 
						class="option-btn"
						class:active={advantage === 'disadvantage'}
						onclick={() => advantage = 'disadvantage'}
						aria-label="Roll with disadvantage"
						aria-pressed={advantage === 'disadvantage'}
					>
						Dis
					</button>
				</div>
			</div>
			
			<div class="option-group">
				<label>Modifier</label>
				<input 
					type="number" 
					bind:value={modifier}
					class="modifier-input"
					min="-20"
					max="20"
					aria-label="Dice roll modifier"
				/>
			</div>
			
			<div class="option-group">
				<label class="checkbox-label">
					<input 
						type="checkbox"
						bind:checked={rerollOnes}
						id="reroll-ones"
						class="reroll-checkbox"
					/>
					<span class="checkbox-text">Reroll 1s</span>
					<span class="checkbox-description">Automatically reroll any dice showing 1</span>
				</label>
			</div>
			
			<div class="option-group">
				<label>Crit Range</label>
				<input 
					type="number"
					bind:value={criticalRange}
					class="crit-input"
					min="1"
					max="20"
					aria-label="Critical hit range"
				/>
			</div>
		</div>
	{/if}
	
	<!-- Custom expression input -->
	<div class="custom-roll">
		<select 
			class="purpose-select"
			bind:value={selectedPurpose}
			aria-label="Select roll purpose"
		>
			{#each purposes as purpose}
				<option value={purpose}>{purpose}</option>
			{/each}
		</select>
		
		<input 
			type="text"
			bind:value={customExpression}
			placeholder="e.g. 2d6+3"
			class="expression-input"
			onkeypress={(e) => e.key === 'Enter' && rollDice()}
			aria-label="Custom dice expression"
		/>
		
		<button 
			class="roll-btn"
			onclick={() => rollDice()}
			disabled={isRolling}
			aria-label="Roll dice"
		>
			{#if isRolling}
				<span class="rolling-icon"></span>
			{:else}
				Roll
			{/if}
		</button>
	</div>
	
	<!-- Animation display -->
	{#if showAnimation && animatingDice.length > 0}
		<div class="dice-animation">
			<div class="animated-dice">
				{#each animatingDice as die (die.id)}
					<div class="animated-die" data-die-id={die.id}>
						<span class="die-value">{die.value}</span>
					</div>
				{/each}
			</div>
			{#if animatedResult !== null}
				<div class="animated-total">
					= {animatedResult}
				</div>
			{/if}
		</div>
	{/if}
	
	<!-- Roll history -->
	<div class="roll-history" class:collapsed={compactMode} role="region" aria-label="Roll history">
		<div class="history-header">
			<h4>Recent Rolls</h4>
			<span class="history-count">{rollHistory.length}</span>
		</div>
		
		<div class="history-list">
			{#each rollHistory.slice().reverse() as roll}
				<div class="roll-entry" class:own-roll={roll.player_id === collaborationStore.currentParticipant?.user_id}>
					<div class="roll-info">
						<span class="roll-player" style="color: {getRollColor(roll.total, roll.expression)}">
							{roll.player_name}
						</span>
						<span class="roll-time">{formatRollTime(roll.timestamp)}</span>
					</div>
					
					<div class="roll-details">
						<span class="roll-expression">{roll.expression}</span>
						{#if roll.purpose}
							<span class="roll-purpose">({roll.purpose})</span>
						{/if}
					</div>
					
					<div class="roll-results">
						<span class="roll-dice">
							[
							{#each roll.results as result, i}
								{#if i > 0}, {/if}
								{#if result === 20}
									<span class="crit">20</span>
								{:else if result === 1}
									<span class="fail">1</span>
								{:else}
									{result}
								{/if}
							{/each}
							]
						</span>
						<span class="roll-total" style="color: {getRollColor(roll.total, roll.expression)}">
							= {roll.total}
						</span>
					</div>
				</div>
			{/each}
		</div>
	</div>
</div>

<style>
	.shared-dice-roller {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		padding: 1rem;
		background: var(--color-surface);
		border-radius: 0.5rem;
	}
	
	.shared-dice-roller.compact {
		gap: 0.5rem;
		padding: 0.5rem;
	}
	
	.roller-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	
	.roller-header h3 {
		margin: 0;
		font-size: 1.125rem;
		font-weight: 600;
	}
	
	.advanced-toggle {
		padding: 0.25rem 0.75rem;
		background: var(--color-surface-secondary);
		border: 1px solid var(--color-border);
		border-radius: 0.25rem;
		font-size: 0.875rem;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.advanced-toggle:hover {
		background: var(--color-primary);
		color: white;
		border-color: var(--color-primary);
	}
	
	.dice-presets {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
	}
	
	.dice-btn {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.25rem;
		padding: 0.5rem;
		background: white;
		border: 2px solid var(--dice-color, #6b7280);
		border-radius: 0.5rem;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.dice-btn:hover {
		background: var(--dice-color);
		color: white;
		transform: translateY(-2px);
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
	}
	
	.dice-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	
	.dice-btn.selected {
		background: var(--dice-color);
		color: white;
	}
	
	.dice-icon {
		font-size: 1.5rem;
		line-height: 1;
	}
	
	.dice-label {
		font-size: 0.75rem;
		font-weight: 600;
	}
	
	.advanced-options {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
		gap: 1rem;
		padding: 1rem;
		background: var(--color-surface-secondary);
		border-radius: 0.375rem;
	}
	
	.option-group {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	
	.option-group label {
		font-size: 0.875rem;
		font-weight: 500;
		color: var(--color-text-secondary);
	}
	
	.button-group {
		display: flex;
		gap: 0.25rem;
	}
	
	.option-btn {
		flex: 1;
		padding: 0.25rem;
		background: white;
		border: 1px solid var(--color-border);
		border-radius: 0.25rem;
		font-size: 0.75rem;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.option-btn:hover {
		background: var(--color-surface-secondary);
	}
	
	.option-btn.active {
		background: var(--color-primary);
		color: white;
		border-color: var(--color-primary);
	}
	
	.modifier-input,
	.crit-input {
		width: 100%;
		padding: 0.25rem 0.5rem;
		border: 1px solid var(--color-border);
		border-radius: 0.25rem;
		font-size: 0.875rem;
	}

	.checkbox-label {
		display: flex;
		flex-direction: column;
		gap: 0.125rem;
		cursor: pointer;
	}

	.reroll-checkbox {
		width: 1rem;
		height: 1rem;
		margin-right: 0.5rem;
		cursor: pointer;
	}

	.checkbox-text {
		display: flex;
		align-items: center;
		font-size: 0.875rem;
		font-weight: 500;
		color: var(--color-text);
	}

	.checkbox-description {
		font-size: 0.75rem;
		color: var(--color-text-secondary);
		margin-left: 1.5rem;
		font-style: italic;
	}
	
	.custom-roll {
		display: flex;
		gap: 0.5rem;
	}
	
	.purpose-select {
		padding: 0.5rem;
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		background: white;
		font-size: 0.875rem;
	}
	
	.expression-input {
		flex: 1;
		padding: 0.5rem;
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		font-size: 0.875rem;
		font-family: 'Monaco', 'Menlo', monospace;
	}
	
	.expression-input:focus {
		outline: none;
		border-color: var(--color-primary);
		box-shadow: 0 0 0 3px var(--color-primary-alpha);
	}
	
	.roll-btn {
		padding: 0.5rem 1.5rem;
		background: var(--color-primary);
		color: white;
		border: none;
		border-radius: 0.375rem;
		font-weight: 600;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.roll-btn:hover {
		background: var(--color-primary-hover);
	}
	
	.roll-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	
	.rolling-icon {
		display: inline-block;
		width: 1rem;
		height: 1rem;
		border: 2px solid white;
		border-right-color: transparent;
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}
	
	@keyframes spin {
		to { transform: rotate(360deg); }
	}
	
	.dice-animation {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 1rem;
		padding: 2rem;
		background: var(--color-surface-secondary);
		border-radius: 0.5rem;
	}
	
	.animated-dice {
		display: flex;
		gap: 0.5rem;
	}
	
	.animated-die {
		width: 3rem;
		height: 3rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: white;
		border: 2px solid var(--color-primary);
		border-radius: 0.5rem;
		font-size: 1.5rem;
		font-weight: bold;
		animation: bounce 0.5s ease-in-out;
	}
	
	@keyframes bounce {
		0%, 100% { transform: translateY(0); }
		50% { transform: translateY(-20px); }
	}
	
	.die-value {
		animation: roll 0.1s linear infinite;
	}
	
	@keyframes roll {
		to { transform: rotateX(360deg); }
	}
	
	.animated-total {
		font-size: 2rem;
		font-weight: bold;
		color: var(--color-primary);
		animation: fadeIn 0.5s ease-in-out;
	}
	
	@keyframes fadeIn {
		from { opacity: 0; transform: scale(0.5); }
		to { opacity: 1; transform: scale(1); }
	}
	
	.roll-history {
		max-height: 300px;
		overflow-y: auto;
		background: var(--color-surface-secondary);
		border-radius: 0.375rem;
	}
	
	.roll-history.collapsed {
		max-height: 150px;
	}
	
	.history-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.75rem;
		background: var(--color-surface);
		border-bottom: 1px solid var(--color-border);
		position: sticky;
		top: 0;
		z-index: 1;
	}
	
	.history-header h4 {
		margin: 0;
		font-size: 0.875rem;
		font-weight: 600;
	}
	
	.history-count {
		padding: 0.125rem 0.5rem;
		background: var(--color-primary);
		color: white;
		border-radius: 1rem;
		font-size: 0.75rem;
		font-weight: 600;
	}
	
	.history-list {
		padding: 0.5rem;
	}
	
	.roll-entry {
		padding: 0.75rem;
		margin-bottom: 0.5rem;
		background: white;
		border-radius: 0.375rem;
		border-left: 3px solid transparent;
		transition: all 0.2s;
	}
	
	.roll-entry.own-roll {
		border-left-color: var(--color-primary);
		background: var(--color-primary-alpha);
	}
	
	.roll-entry:hover {
		transform: translateX(2px);
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
	}
	
	.roll-info {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.25rem;
	}
	
	.roll-player {
		font-weight: 600;
		font-size: 0.875rem;
	}
	
	.roll-time {
		font-size: 0.75rem;
		color: var(--color-text-secondary);
	}
	
	.roll-details {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		margin-bottom: 0.25rem;
	}
	
	.roll-expression {
		font-family: 'Monaco', 'Menlo', monospace;
		font-size: 0.875rem;
		color: var(--color-text-secondary);
	}
	
	.roll-purpose {
		font-size: 0.75rem;
		color: var(--color-text-secondary);
		font-style: italic;
	}
	
	.roll-results {
		display: flex;
		gap: 0.5rem;
		align-items: center;
	}
	
	.roll-dice {
		font-family: 'Monaco', 'Menlo', monospace;
		font-size: 0.875rem;
		color: var(--color-text-secondary);
	}
	
	.roll-dice .crit {
		color: var(--color-success);
		font-weight: bold;
	}
	
	.roll-dice .fail {
		color: var(--color-error);
		font-weight: bold;
	}
	
	.roll-total {
		font-size: 1.125rem;
		font-weight: bold;
	}
</style>