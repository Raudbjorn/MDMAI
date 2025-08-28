<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { CursorPosition } from '$lib/types/collaboration';
	
	interface Props {
		roomId: string;
		width?: number;
		height?: number;
		enableDrawing?: boolean;
		enableAnnotations?: boolean;
		enableGrid?: boolean;
		gridSize?: number;
	}
	
	let {
		roomId,
		width = 1200,
		height = 800,
		enableDrawing = true,
		enableAnnotations = true,
		enableGrid = true,
		gridSize = 20
	}: Props = $props();
	
	interface CanvasState {
		isDrawing: boolean;
		currentTool: 'pen' | 'eraser' | 'rectangle' | 'circle' | 'text' | 'select';
		strokeColor: string;
		strokeWidth: number;
		fillColor: string;
		opacity: number;
		elements: CanvasElement[];
		selectedElement: CanvasElement | null;
		history: CanvasElement[][];
		historyIndex: number;
	}
	
	interface CanvasElement {
		id: string;
		type: 'path' | 'rectangle' | 'circle' | 'text' | 'image';
		points?: { x: number; y: number }[];
		x?: number;
		y?: number;
		width?: number;
		height?: number;
		radius?: number;
		text?: string;
		strokeColor: string;
		strokeWidth: number;
		fillColor?: string;
		opacity: number;
		createdBy: string;
		createdAt: number;
	}
	
	let canvas: HTMLCanvasElement;
	let overlayCanvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D;
	let overlayCtx: CanvasRenderingContext2D;
	
	let canvasState = $state<CanvasState>({
		isDrawing: false,
		currentTool: 'pen',
		strokeColor: '#000000',
		strokeWidth: 2,
		fillColor: 'transparent',
		opacity: 1,
		elements: [],
		selectedElement: null,
		history: [],
		historyIndex: -1
	});
	
	let cursors = $derived(Array.from(collaborationStore.presence.entries()));
	let participants = $derived(collaborationStore.participants);
	
	onMount(() => {
		if (!canvas || !overlayCanvas) return;
		
		ctx = canvas.getContext('2d')!;
		overlayCtx = overlayCanvas.getContext('2d')!;
		
		// Set canvas size
		canvas.width = width;
		canvas.height = height;
		overlayCanvas.width = width;
		overlayCanvas.height = height;
		
		// Initial render
		renderCanvas();
		
		// Subscribe to canvas updates from other users
		const unsubscribe = collaborationStore.onMessage('canvas_update', (msg) => {
			if (msg.data.element) {
				canvasState.elements = [...canvasState.elements, msg.data.element];
				renderCanvas();
			}
		});
		
		return unsubscribe;
	});
	
	function renderCanvas() {
		if (!ctx) return;
		
		// Clear canvas
		ctx.clearRect(0, 0, width, height);
		
		// Draw grid if enabled
		if (enableGrid) {
			drawGrid();
		}
		
		// Draw all elements
		canvasState.elements.forEach(element => {
			drawElement(ctx, element);
		});
		
		// Draw cursors on overlay
		renderOverlay();
	}
	
	function renderOverlay() {
		if (!overlayCtx) return;
		
		// Clear overlay
		overlayCtx.clearRect(0, 0, width, height);
		
		// Draw remote cursors
		cursors.forEach(([userId, cursor]) => {
			if (userId !== collaborationStore.currentParticipant?.user_id) {
				drawCursor(overlayCtx, cursor, userId);
			}
		});
		
		// Draw selection if any
		if (canvasState.selectedElement) {
			drawSelection(overlayCtx, canvasState.selectedElement);
		}
	}
	
	function drawGrid() {
		ctx.strokeStyle = '#e0e0e0';
		ctx.lineWidth = 0.5;
		
		// Vertical lines
		for (let x = 0; x <= width; x += gridSize) {
			ctx.beginPath();
			ctx.moveTo(x, 0);
			ctx.lineTo(x, height);
			ctx.stroke();
		}
		
		// Horizontal lines
		for (let y = 0; y <= height; y += gridSize) {
			ctx.beginPath();
			ctx.moveTo(0, y);
			ctx.lineTo(width, y);
			ctx.stroke();
		}
	}
	
	function drawElement(context: CanvasRenderingContext2D, element: CanvasElement) {
		context.save();
		
		context.globalAlpha = element.opacity;
		context.strokeStyle = element.strokeColor;
		context.lineWidth = element.strokeWidth;
		if (element.fillColor) {
			context.fillStyle = element.fillColor;
		}
		
		switch (element.type) {
			case 'path':
				if (element.points && element.points.length > 0) {
					context.beginPath();
					context.moveTo(element.points[0].x, element.points[0].y);
					element.points.forEach(point => {
						context.lineTo(point.x, point.y);
					});
					context.stroke();
				}
				break;
				
			case 'rectangle':
				if (element.x !== undefined && element.y !== undefined && 
					element.width !== undefined && element.height !== undefined) {
					if (element.fillColor && element.fillColor !== 'transparent') {
						context.fillRect(element.x, element.y, element.width, element.height);
					}
					context.strokeRect(element.x, element.y, element.width, element.height);
				}
				break;
				
			case 'circle':
				if (element.x !== undefined && element.y !== undefined && element.radius !== undefined) {
					context.beginPath();
					context.arc(element.x, element.y, element.radius, 0, Math.PI * 2);
					if (element.fillColor && element.fillColor !== 'transparent') {
						context.fill();
					}
					context.stroke();
				}
				break;
				
			case 'text':
				if (element.x !== undefined && element.y !== undefined && element.text) {
					context.font = `${element.strokeWidth * 8}px sans-serif`;
					context.fillStyle = element.strokeColor;
					context.fillText(element.text, element.x, element.y);
				}
				break;
		}
		
		context.restore();
	}
	
	function drawCursor(context: CanvasRenderingContext2D, cursor: CursorPosition, userId: string) {
		const participant = participants.find(p => p.user_id === userId);
		if (!participant) return;
		
		context.save();
		
		// Draw cursor
		context.fillStyle = participant.color;
		context.strokeStyle = participant.color;
		context.lineWidth = 2;
		
		context.beginPath();
		context.moveTo(cursor.x, cursor.y);
		context.lineTo(cursor.x + 10, cursor.y + 10);
		context.lineTo(cursor.x, cursor.y + 14);
		context.closePath();
		context.fill();
		context.stroke();
		
		// Draw username
		context.fillStyle = 'white';
		context.fillRect(cursor.x + 12, cursor.y + 8, participant.username.length * 7 + 6, 18);
		context.fillStyle = participant.color;
		context.font = '12px sans-serif';
		context.fillText(participant.username, cursor.x + 15, cursor.y + 20);
		
		context.restore();
	}
	
	function drawSelection(context: CanvasRenderingContext2D, element: CanvasElement) {
		context.save();
		
		context.strokeStyle = '#0066cc';
		context.lineWidth = 2;
		context.setLineDash([5, 5]);
		
		// Calculate bounding box
		let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
		
		if (element.type === 'path' && element.points) {
			element.points.forEach(point => {
				minX = Math.min(minX, point.x);
				minY = Math.min(minY, point.y);
				maxX = Math.max(maxX, point.x);
				maxY = Math.max(maxY, point.y);
			});
		} else if (element.x !== undefined && element.y !== undefined) {
			minX = element.x;
			minY = element.y;
			if (element.width !== undefined && element.height !== undefined) {
				maxX = element.x + element.width;
				maxY = element.y + element.height;
			} else if (element.radius !== undefined) {
				minX -= element.radius;
				minY -= element.radius;
				maxX += element.radius;
				maxY += element.radius;
			}
		}
		
		if (minX !== Infinity) {
			context.strokeRect(minX - 5, minY - 5, maxX - minX + 10, maxY - minY + 10);
		}
		
		context.restore();
	}
	
	let currentPath: { x: number; y: number }[] = [];
	let startPoint: { x: number; y: number } | null = null;
	
	function handleMouseDown(e: MouseEvent) {
		if (!enableDrawing) return;
		
		const rect = canvas.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const y = e.clientY - rect.top;
		
		canvasState.isDrawing = true;
		
		switch (canvasState.currentTool) {
			case 'pen':
			case 'eraser':
				currentPath = [{ x, y }];
				break;
			case 'rectangle':
			case 'circle':
				startPoint = { x, y };
				break;
			case 'select':
				// Find element at position
				canvasState.selectedElement = findElementAt(x, y);
				renderOverlay();
				break;
		}
	}
	
	function handleMouseMove(e: MouseEvent) {
		const rect = canvas.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const y = e.clientY - rect.top;
		
		// Update cursor position for collaboration
		collaborationStore.updateCursor(x, y, 'canvas');
		
		if (!canvasState.isDrawing || !enableDrawing) return;
		
		switch (canvasState.currentTool) {
			case 'pen':
			case 'eraser':
				currentPath.push({ x, y });
				// Draw preview
				if (overlayCtx) {
					overlayCtx.clearRect(0, 0, width, height);
					renderOverlay();
					drawElement(overlayCtx, {
						id: 'preview',
						type: 'path',
						points: currentPath,
						strokeColor: canvasState.currentTool === 'eraser' ? '#ffffff' : canvasState.strokeColor,
						strokeWidth: canvasState.strokeWidth,
						opacity: canvasState.opacity,
						createdBy: collaborationStore.currentParticipant?.user_id || '',
						createdAt: Date.now()
					});
				}
				break;
				
			case 'rectangle':
				if (startPoint && overlayCtx) {
					overlayCtx.clearRect(0, 0, width, height);
					renderOverlay();
					drawElement(overlayCtx, {
						id: 'preview',
						type: 'rectangle',
						x: Math.min(startPoint.x, x),
						y: Math.min(startPoint.y, y),
						width: Math.abs(x - startPoint.x),
						height: Math.abs(y - startPoint.y),
						strokeColor: canvasState.strokeColor,
						strokeWidth: canvasState.strokeWidth,
						fillColor: canvasState.fillColor,
						opacity: canvasState.opacity,
						createdBy: collaborationStore.currentParticipant?.user_id || '',
						createdAt: Date.now()
					});
				}
				break;
				
			case 'circle':
				if (startPoint && overlayCtx) {
					overlayCtx.clearRect(0, 0, width, height);
					renderOverlay();
					const radius = Math.sqrt(Math.pow(x - startPoint.x, 2) + Math.pow(y - startPoint.y, 2));
					drawElement(overlayCtx, {
						id: 'preview',
						type: 'circle',
						x: startPoint.x,
						y: startPoint.y,
						radius,
						strokeColor: canvasState.strokeColor,
						strokeWidth: canvasState.strokeWidth,
						fillColor: canvasState.fillColor,
						opacity: canvasState.opacity,
						createdBy: collaborationStore.currentParticipant?.user_id || '',
						createdAt: Date.now()
					});
				}
				break;
		}
	}
	
	async function handleMouseUp(e: MouseEvent) {
		if (!canvasState.isDrawing || !enableDrawing) return;
		
		const rect = canvas.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const y = e.clientY - rect.top;
		
		canvasState.isDrawing = false;
		
		let newElement: CanvasElement | null = null;
		
		switch (canvasState.currentTool) {
			case 'pen':
			case 'eraser':
				if (currentPath.length > 0) {
					newElement = {
						id: generateId(),
						type: 'path',
						points: currentPath,
						strokeColor: canvasState.currentTool === 'eraser' ? '#ffffff' : canvasState.strokeColor,
						strokeWidth: canvasState.strokeWidth,
						opacity: canvasState.opacity,
						createdBy: collaborationStore.currentParticipant?.user_id || '',
						createdAt: Date.now()
					};
				}
				break;
				
			case 'rectangle':
				if (startPoint) {
					newElement = {
						id: generateId(),
						type: 'rectangle',
						x: Math.min(startPoint.x, x),
						y: Math.min(startPoint.y, y),
						width: Math.abs(x - startPoint.x),
						height: Math.abs(y - startPoint.y),
						strokeColor: canvasState.strokeColor,
						strokeWidth: canvasState.strokeWidth,
						fillColor: canvasState.fillColor,
						opacity: canvasState.opacity,
						createdBy: collaborationStore.currentParticipant?.user_id || '',
						createdAt: Date.now()
					};
				}
				break;
				
			case 'circle':
				if (startPoint) {
					const radius = Math.sqrt(Math.pow(x - startPoint.x, 2) + Math.pow(y - startPoint.y, 2));
					newElement = {
						id: generateId(),
						type: 'circle',
						x: startPoint.x,
						y: startPoint.y,
						radius,
						strokeColor: canvasState.strokeColor,
						strokeWidth: canvasState.strokeWidth,
						fillColor: canvasState.fillColor,
						opacity: canvasState.opacity,
						createdBy: collaborationStore.currentParticipant?.user_id || '',
						createdAt: Date.now()
					};
				}
				break;
		}
		
		if (newElement) {
			// Add to local state
			canvasState.elements = [...canvasState.elements, newElement];
			addToHistory();
			
			// Broadcast to other users
			await collaborationStore.updateState({
				path: ['canvas_elements'],
				value: newElement,
				operation: 'merge',
				version: canvasState.elements.length,
				previous_version: canvasState.elements.length - 1
			});
		}
		
		// Clear preview
		overlayCtx?.clearRect(0, 0, width, height);
		renderCanvas();
		
		// Reset
		currentPath = [];
		startPoint = null;
	}
	
	function findElementAt(x: number, y: number): CanvasElement | null {
		// Search in reverse order (top elements first)
		for (let i = canvasState.elements.length - 1; i >= 0; i--) {
			const element = canvasState.elements[i];
			
			switch (element.type) {
				case 'path':
					if (element.points) {
						for (const point of element.points) {
							if (Math.abs(point.x - x) < 5 && Math.abs(point.y - y) < 5) {
								return element;
							}
						}
					}
					break;
					
				case 'rectangle':
					if (element.x !== undefined && element.y !== undefined && 
						element.width !== undefined && element.height !== undefined) {
						if (x >= element.x && x <= element.x + element.width &&
							y >= element.y && y <= element.y + element.height) {
							return element;
						}
					}
					break;
					
				case 'circle':
					if (element.x !== undefined && element.y !== undefined && element.radius !== undefined) {
						const distance = Math.sqrt(Math.pow(x - element.x, 2) + Math.pow(y - element.y, 2));
						if (distance <= element.radius) {
							return element;
						}
					}
					break;
			}
		}
		
		return null;
	}
	
	function generateId(): string {
		return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
	}
	
	function addToHistory() {
		// Remove any redo history
		canvasState.history = canvasState.history.slice(0, canvasState.historyIndex + 1);
		// Add current state
		canvasState.history = [...canvasState.history, [...canvasState.elements]];
		canvasState.historyIndex++;
		// Limit history size
		if (canvasState.history.length > 50) {
			canvasState.history = canvasState.history.slice(-50);
			canvasState.historyIndex = canvasState.history.length - 1;
		}
	}
	
	function undo() {
		if (canvasState.historyIndex > 0) {
			canvasState.historyIndex--;
			canvasState.elements = [...canvasState.history[canvasState.historyIndex]];
			renderCanvas();
		}
	}
	
	function redo() {
		if (canvasState.historyIndex < canvasState.history.length - 1) {
			canvasState.historyIndex++;
			canvasState.elements = [...canvasState.history[canvasState.historyIndex]];
			renderCanvas();
		}
	}
	
	function clearCanvas() {
		canvasState.elements = [];
		addToHistory();
		renderCanvas();
	}
	
	function setTool(tool: CanvasState['currentTool']) {
		canvasState.currentTool = tool;
		canvasState.selectedElement = null;
		renderOverlay();
	}
	
	// Keyboard shortcuts
	function handleKeyDown(e: KeyboardEvent) {
		if (e.ctrlKey || e.metaKey) {
			switch (e.key) {
				case 'z':
					e.preventDefault();
					if (e.shiftKey) {
						redo();
					} else {
						undo();
					}
					break;
				case 'y':
					e.preventDefault();
					redo();
					break;
			}
		} else if (e.key === 'Delete' && canvasState.selectedElement) {
			// Remove selected element
			canvasState.elements = canvasState.elements.filter(el => el !== canvasState.selectedElement);
			canvasState.selectedElement = null;
			addToHistory();
			renderCanvas();
		}
	}
</script>

<div class="collaborative-canvas">
	<div class="toolbar">
		<div class="tool-group">
			<button
				class="tool-btn"
				class:active={canvasState.currentTool === 'select'}
				onclick={() => setTool('select')}
				title="Select (S)"
			>
				<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
					<path d="M3 3l7 14v-6h6L3 3z"/>
				</svg>
			</button>
			<button
				class="tool-btn"
				class:active={canvasState.currentTool === 'pen'}
				onclick={() => setTool('pen')}
				title="Pen (P)"
			>
				<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
					<path d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z"/>
					<path d="M3 17h14v2H3z"/>
				</svg>
			</button>
			<button
				class="tool-btn"
				class:active={canvasState.currentTool === 'eraser'}
				onclick={() => setTool('eraser')}
				title="Eraser (E)"
			>
				<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
					<path d="M8.5 8.5l-3 3L3 14h4l2-2 3-3-3.5-3.5z"/>
					<path d="M11 6L14.5 2.5a1.5 1.5 0 012.1 0l.9.9a1.5 1.5 0 010 2.1L14 9l-3-3z"/>
				</svg>
			</button>
			<button
				class="tool-btn"
				class:active={canvasState.currentTool === 'rectangle'}
				onclick={() => setTool('rectangle')}
				title="Rectangle (R)"
			>
				<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
					<rect x="3" y="5" width="14" height="10" fill="none" stroke="currentColor" stroke-width="2"/>
				</svg>
			</button>
			<button
				class="tool-btn"
				class:active={canvasState.currentTool === 'circle'}
				onclick={() => setTool('circle')}
				title="Circle (C)"
			>
				<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
					<circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="2"/>
				</svg>
			</button>
		</div>
		
		<div class="tool-group">
			<label class="tool-label">
				Stroke
				<input
					type="color"
					bind:value={canvasState.strokeColor}
					class="color-input"
				/>
			</label>
			<label class="tool-label">
				Fill
				<input
					type="color"
					bind:value={canvasState.fillColor}
					class="color-input"
				/>
			</label>
			<label class="tool-label">
				Width
				<input
					type="range"
					min="1"
					max="20"
					bind:value={canvasState.strokeWidth}
					class="range-input"
				/>
				<span class="range-value">{canvasState.strokeWidth}</span>
			</label>
			<label class="tool-label">
				Opacity
				<input
					type="range"
					min="0"
					max="1"
					step="0.1"
					bind:value={canvasState.opacity}
					class="range-input"
				/>
				<span class="range-value">{(canvasState.opacity * 100).toFixed(0)}%</span>
			</label>
		</div>
		
		<div class="tool-group">
			<button
				class="tool-btn"
				onclick={undo}
				disabled={canvasState.historyIndex <= 0}
				title="Undo (Ctrl+Z)"
			>
				<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
					<path d="M3 10h10a5 5 0 010 10h-3v-2h3a3 3 0 000-6H3v3L0 10l3-5v3z"/>
				</svg>
			</button>
			<button
				class="tool-btn"
				onclick={redo}
				disabled={canvasState.historyIndex >= canvasState.history.length - 1}
				title="Redo (Ctrl+Y)"
			>
				<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
					<path d="M17 10H7a5 5 0 000 10h3v-2H7a3 3 0 010-6h10v3l3-5-3-5v3z"/>
				</svg>
			</button>
			<button
				class="tool-btn"
				onclick={clearCanvas}
				title="Clear Canvas"
			>
				<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
					<path d="M6 2l2-2h4l2 2h4v2H2V2h4zM3 6h14l-1 14H4L3 6zm3 2v10h2V8H6zm6 0v10h2V8h-2z"/>
				</svg>
			</button>
		</div>
		
		<div class="participants-indicator">
			{#each participants.slice(0, 3) as participant}
				<div 
					class="participant-avatar"
					style="background-color: {participant.color}"
					title="{participant.username}"
				>
					{participant.username.charAt(0).toUpperCase()}
				</div>
			{/each}
			{#if participants.length > 3}
				<div class="participant-count">+{participants.length - 3}</div>
			{/if}
		</div>
	</div>
	
	<div class="canvas-container" onkeydown={handleKeyDown}>
		<canvas
			bind:this={canvas}
			class="main-canvas"
			onmousedown={handleMouseDown}
			onmousemove={handleMouseMove}
			onmouseup={handleMouseUp}
			onmouseleave={handleMouseUp}
		/>
		<canvas
			bind:this={overlayCanvas}
			class="overlay-canvas"
			onmousedown={handleMouseDown}
			onmousemove={handleMouseMove}
			onmouseup={handleMouseUp}
			onmouseleave={handleMouseUp}
		/>
	</div>
</div>

<style>
	.collaborative-canvas {
		display: flex;
		flex-direction: column;
		background: white;
		border: 1px solid #e0e0e0;
		border-radius: 8px;
		overflow: hidden;
	}
	
	.toolbar {
		display: flex;
		align-items: center;
		gap: 1rem;
		padding: 0.75rem;
		background: #f5f5f5;
		border-bottom: 1px solid #e0e0e0;
		flex-wrap: wrap;
	}
	
	.tool-group {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0 0.5rem;
		border-right: 1px solid #d0d0d0;
	}
	
	.tool-group:last-child {
		border-right: none;
	}
	
	.tool-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 36px;
		height: 36px;
		padding: 0;
		background: white;
		border: 1px solid #d0d0d0;
		border-radius: 4px;
		cursor: pointer;
		transition: all 0.2s;
		color: #666;
	}
	
	.tool-btn:hover:not(:disabled) {
		background: #f0f0f0;
		border-color: #999;
	}
	
	.tool-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	
	.tool-btn.active {
		background: #007bff;
		color: white;
		border-color: #0056b3;
	}
	
	.tool-label {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.875rem;
		color: #666;
	}
	
	.color-input {
		width: 32px;
		height: 32px;
		padding: 0;
		border: 1px solid #d0d0d0;
		border-radius: 4px;
		cursor: pointer;
	}
	
	.range-input {
		width: 80px;
	}
	
	.range-value {
		min-width: 30px;
		text-align: right;
		font-weight: 500;
	}
	
	.participants-indicator {
		display: flex;
		align-items: center;
		gap: -8px;
		margin-left: auto;
	}
	
	.participant-avatar {
		width: 32px;
		height: 32px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		color: white;
		font-weight: bold;
		font-size: 0.875rem;
		border: 2px solid white;
		position: relative;
		z-index: 1;
	}
	
	.participant-avatar:hover {
		z-index: 2;
		transform: scale(1.1);
	}
	
	.participant-count {
		width: 32px;
		height: 32px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		background: #666;
		color: white;
		font-size: 0.75rem;
		font-weight: bold;
		border: 2px solid white;
		margin-left: 8px;
	}
	
	.canvas-container {
		position: relative;
		overflow: auto;
		flex: 1;
		background: #fafafa;
	}
	
	.main-canvas {
		position: absolute;
		top: 0;
		left: 0;
		cursor: crosshair;
		background: white;
		box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
	}
	
	.overlay-canvas {
		position: absolute;
		top: 0;
		left: 0;
		pointer-events: none;
		z-index: 10;
	}
	
	.main-canvas:active {
		cursor: crosshair;
	}
</style>