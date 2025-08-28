<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { collaborationStore } from '$lib/stores/collaboration.svelte';
	import type { StateUpdate } from '$lib/types/collaboration';
	import { throttle } from '$lib/utils/debounce';
	
	interface MapToken {
		id: string;
		name: string;
		type: 'player' | 'enemy' | 'ally' | 'object';
		x: number;
		y: number;
		size: number; // Grid squares (1 = medium, 2 = large, etc.)
		color: string;
		icon?: string;
		hp?: { current: number; max: number };
		conditions?: string[];
		owner_id?: string;
		visible: boolean;
		locked: boolean;
	}
	
	interface MapState {
		tokens: Map<string, MapToken>;
		gridSize: number;
		gridVisible: boolean;
		background?: string;
		fogOfWar?: boolean[][];
		measurements: Measurement[];
		drawings: Drawing[];
	}
	
	interface Measurement {
		id: string;
		start: { x: number; y: number };
		end: { x: number; y: number };
		color: string;
		owner_id: string;
	}
	
	interface Drawing {
		id: string;
		type: 'line' | 'rectangle' | 'circle' | 'polygon';
		points: { x: number; y: number }[];
		color: string;
		strokeWidth: number;
		fill?: string;
		owner_id: string;
	}
	
	interface Props {
		roomId: string;
		width?: number;
		height?: number;
		gridSize?: number;
		showGrid?: boolean;
		enableFogOfWar?: boolean;
		readOnly?: boolean;
	}
	
	let {
		roomId,
		width = 800,
		height = 600,
		gridSize = 40,
		showGrid = true,
		enableFogOfWar = false,
		readOnly = false
	}: Props = $props();
	
	// State
	let mapState = $state<MapState>({
		tokens: new Map(),
		gridSize,
		gridVisible: showGrid,
		fogOfWar: enableFogOfWar ? Array(Math.ceil(height / gridSize)).fill(null).map(() => 
			Array(Math.ceil(width / gridSize)).fill(true)) : undefined,
		measurements: [],
		drawings: []
	});
	
	let selectedToken = $state<MapToken | null>(null);
	let isDragging = $state(false);
	let dragOffset = $state({ x: 0, y: 0 });
	let isDrawing = $state(false);
	let drawingMode = $state<'select' | 'measure' | 'draw' | 'fog'>('select');
	let currentDrawing = $state<Drawing | null>(null);
	let currentMeasurement = $state<Measurement | null>(null);
	let cursorPositions = $state<Map<string, { x: number; y: number; color: string }>>(new Map());
	let scale = $state(1);
	let panOffset = $state({ x: 0, y: 0 });
	
	// Canvas references
	let canvasEl: HTMLCanvasElement;
	let overlayEl: HTMLDivElement;
	let ctx: CanvasRenderingContext2D | null = null;
	let unsubscribe: (() => void) | null = null;
	
	// Permissions
	let canEdit = $derived(
		!readOnly && collaborationStore.hasPermission('write', 'map')
	);
	
	let canMoveAllTokens = $derived(
		collaborationStore.hasPermission('control_initiative', 'initiative')
	);
	
	onMount(() => {
		if (canvasEl) {
			ctx = canvasEl.getContext('2d');
			setupCanvas();
		}
		
		// Subscribe to map updates
		unsubscribe = collaborationStore.onMessage('state_update', (msg) => {
			if (msg.data.path[0] === 'map') {
				handleMapUpdate(msg.data as StateUpdate);
			}
		});
		
		// Subscribe to cursor positions
		collaborationStore.onMessage('cursor_move', (msg) => {
			if (msg.data.element === 'battle_map') {
				updateCursorPosition(msg.sender_id, msg.data);
			}
		});
		
		// Load initial map state
		loadMapState();
		
		// Setup keyboard shortcuts
		window.addEventListener('keydown', handleKeyDown);
		window.addEventListener('keyup', handleKeyUp);
		
		// Start render loop
		requestAnimationFrame(renderLoop);
	});
	
	onDestroy(() => {
		unsubscribe?.();
		window.removeEventListener('keydown', handleKeyDown);
		window.removeEventListener('keyup', handleKeyUp);
	});
	
	function setupCanvas() {
		if (!canvasEl || !ctx) return;
		
		// Set canvas size
		canvasEl.width = width;
		canvasEl.height = height;
		
		// Set default styles
		ctx.lineCap = 'round';
		ctx.lineJoin = 'round';
	}
	
	function loadMapState() {
		const room = collaborationStore.currentRoom;
		if (room?.state.map) {
			mapState = room.state.map;
		}
	}
	
	function handleMapUpdate(update: StateUpdate) {
		const path = update.path.slice(1); // Remove 'map' prefix
		
		if (path[0] === 'tokens' && path[1]) {
			// Token update
			const tokenId = path[1];
			if (update.operation === 'delete') {
				mapState.tokens.delete(tokenId);
			} else {
				mapState.tokens.set(tokenId, update.value);
			}
			mapState.tokens = new Map(mapState.tokens);
		} else if (path[0] === 'drawings') {
			// Drawing update
			mapState.drawings = update.value;
		} else if (path[0] === 'fogOfWar') {
			// Fog of war update
			mapState.fogOfWar = update.value;
		}
	}
	
	function renderLoop() {
		render();
		requestAnimationFrame(renderLoop);
	}
	
	function render() {
		if (!ctx || !canvasEl) return;
		
		// Clear canvas
		ctx.clearRect(0, 0, width, height);
		
		// Save context state
		ctx.save();
		
		// Apply transformations
		ctx.translate(panOffset.x, panOffset.y);
		ctx.scale(scale, scale);
		
		// Draw background
		if (mapState.background) {
			// Draw background image
		} else {
			ctx.fillStyle = '#f3f4f6';
			ctx.fillRect(0, 0, width, height);
		}
		
		// Draw grid
		if (mapState.gridVisible) {
			drawGrid();
		}
		
		// Draw fog of war
		if (enableFogOfWar && mapState.fogOfWar) {
			drawFogOfWar();
		}
		
		// Draw drawings
		mapState.drawings.forEach(drawing => {
			drawShape(drawing);
		});
		
		// Draw tokens
		mapState.tokens.forEach(token => {
			drawToken(token);
		});
		
		// Draw measurements
		mapState.measurements.forEach(measurement => {
			drawMeasurement(measurement);
		});
		
		// Draw current measurement
		if (currentMeasurement) {
			drawMeasurement(currentMeasurement);
		}
		
		// Draw current drawing
		if (currentDrawing) {
			drawShape(currentDrawing);
		}
		
		// Draw other users' cursors
		cursorPositions.forEach((pos, userId) => {
			if (userId !== collaborationStore.currentParticipant?.user_id) {
				drawCursor(pos);
			}
		});
		
		// Restore context state
		ctx.restore();
	}
	
	function drawGrid() {
		if (!ctx) return;
		
		ctx.strokeStyle = 'rgba(156, 163, 175, 0.3)';
		ctx.lineWidth = 1;
		
		// Vertical lines
		for (let x = 0; x <= width; x += mapState.gridSize) {
			ctx.beginPath();
			ctx.moveTo(x, 0);
			ctx.lineTo(x, height);
			ctx.stroke();
		}
		
		// Horizontal lines
		for (let y = 0; y <= height; y += mapState.gridSize) {
			ctx.beginPath();
			ctx.moveTo(0, y);
			ctx.lineTo(width, y);
			ctx.stroke();
		}
	}
	
	function drawFogOfWar() {
		if (!ctx || !mapState.fogOfWar) return;
		
		ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
		
		for (let y = 0; y < mapState.fogOfWar.length; y++) {
			for (let x = 0; x < mapState.fogOfWar[y].length; x++) {
				if (mapState.fogOfWar[y][x]) {
					ctx.fillRect(
						x * mapState.gridSize,
						y * mapState.gridSize,
						mapState.gridSize,
						mapState.gridSize
					);
				}
			}
		}
	}
	
	function drawToken(token: MapToken) {
		if (!ctx || !token.visible) return;
		
		const size = token.size * mapState.gridSize;
		const x = token.x * mapState.gridSize;
		const y = token.y * mapState.gridSize;
		
		// Draw token background
		ctx.fillStyle = token.color;
		ctx.strokeStyle = selectedToken?.id === token.id ? '#3b82f6' : '#000';
		ctx.lineWidth = selectedToken?.id === token.id ? 3 : 1;
		
		ctx.beginPath();
		ctx.arc(x + size / 2, y + size / 2, size / 2 - 2, 0, Math.PI * 2);
		ctx.fill();
		ctx.stroke();
		
		// Draw token label
		ctx.fillStyle = getContrastColor(token.color);
		ctx.font = `bold ${Math.max(12, size / 4)}px sans-serif`;
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		ctx.fillText(token.name.substring(0, 2).toUpperCase(), x + size / 2, y + size / 2);
		
		// Draw HP bar if present
		if (token.hp) {
			const barWidth = size - 8;
			const barHeight = 6;
			const barX = x + 4;
			const barY = y + size - 10;
			const hpPercent = token.hp.current / token.hp.max;
			
			// Background
			ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
			ctx.fillRect(barX, barY, barWidth, barHeight);
			
			// HP bar
			ctx.fillStyle = hpPercent > 0.5 ? '#10b981' : hpPercent > 0.25 ? '#f59e0b' : '#ef4444';
			ctx.fillRect(barX, barY, barWidth * hpPercent, barHeight);
		}
		
		// Draw conditions
		if (token.conditions && token.conditions.length > 0) {
			ctx.fillStyle = '#fbbf24';
			ctx.beginPath();
			ctx.arc(x + size - 8, y + 8, 4, 0, Math.PI * 2);
			ctx.fill();
		}
	}
	
	function drawMeasurement(measurement: Measurement) {
		if (!ctx) return;
		
		const distance = calculateDistance(measurement.start, measurement.end);
		const squares = Math.round(distance / mapState.gridSize);
		const feet = squares * 5; // D&D 5e standard
		
		ctx.strokeStyle = measurement.color;
		ctx.lineWidth = 2;
		ctx.setLineDash([5, 5]);
		
		ctx.beginPath();
		ctx.moveTo(measurement.start.x, measurement.start.y);
		ctx.lineTo(measurement.end.x, measurement.end.y);
		ctx.stroke();
		
		ctx.setLineDash([]);
		
		// Draw distance label
		const midX = (measurement.start.x + measurement.end.x) / 2;
		const midY = (measurement.start.y + measurement.end.y) / 2;
		
		ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
		ctx.fillRect(midX - 20, midY - 10, 40, 20);
		
		ctx.fillStyle = 'white';
		ctx.font = 'bold 12px sans-serif';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		ctx.fillText(`${feet}ft`, midX, midY);
	}
	
	function drawShape(drawing: Drawing) {
		if (!ctx) return;
		
		ctx.strokeStyle = drawing.color;
		ctx.lineWidth = drawing.strokeWidth;
		if (drawing.fill) {
			ctx.fillStyle = drawing.fill;
		}
		
		ctx.beginPath();
		
		switch (drawing.type) {
			case 'line':
				if (drawing.points.length >= 2) {
					ctx.moveTo(drawing.points[0].x, drawing.points[0].y);
					for (let i = 1; i < drawing.points.length; i++) {
						ctx.lineTo(drawing.points[i].x, drawing.points[i].y);
					}
				}
				break;
				
			case 'rectangle':
				if (drawing.points.length >= 2) {
					const width = drawing.points[1].x - drawing.points[0].x;
					const height = drawing.points[1].y - drawing.points[0].y;
					ctx.rect(drawing.points[0].x, drawing.points[0].y, width, height);
				}
				break;
				
			case 'circle':
				if (drawing.points.length >= 2) {
					const radius = calculateDistance(drawing.points[0], drawing.points[1]);
					ctx.arc(drawing.points[0].x, drawing.points[0].y, radius, 0, Math.PI * 2);
				}
				break;
				
			case 'polygon':
				if (drawing.points.length >= 3) {
					ctx.moveTo(drawing.points[0].x, drawing.points[0].y);
					for (let i = 1; i < drawing.points.length; i++) {
						ctx.lineTo(drawing.points[i].x, drawing.points[i].y);
					}
					ctx.closePath();
				}
				break;
		}
		
		ctx.stroke();
		if (drawing.fill) {
			ctx.fill();
		}
	}
	
	function drawCursor(pos: { x: number; y: number; color: string }) {
		if (!ctx) return;
		
		ctx.strokeStyle = pos.color;
		ctx.lineWidth = 2;
		
		// Draw crosshair
		ctx.beginPath();
		ctx.moveTo(pos.x - 10, pos.y);
		ctx.lineTo(pos.x + 10, pos.y);
		ctx.moveTo(pos.x, pos.y - 10);
		ctx.lineTo(pos.x, pos.y + 10);
		ctx.stroke();
		
		// Draw circle
		ctx.beginPath();
		ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
		ctx.stroke();
	}
	
	// Event handlers
	function handleMouseDown(event: MouseEvent) {
		if (!canEdit) return;
		
		const pos = getMousePosition(event);
		
		switch (drawingMode) {
			case 'select':
				handleTokenSelection(pos);
				break;
			case 'measure':
				startMeasurement(pos);
				break;
			case 'draw':
				startDrawing(pos);
				break;
			case 'fog':
				toggleFogOfWar(pos);
				break;
		}
	}
	
	function handleMouseMove(event: MouseEvent) {
		const pos = getMousePosition(event);
		
		// Update cursor position for other users
		const throttledUpdate = throttle(() => {
			collaborationStore.updateCursor(pos.x, pos.y, 'battle_map');
		}, 50);
		throttledUpdate();
		
		if (isDragging && selectedToken) {
			moveToken(selectedToken, pos);
		} else if (currentMeasurement) {
			updateMeasurement(pos);
		} else if (currentDrawing && isDrawing) {
			updateDrawing(pos);
		}
	}
	
	function handleMouseUp(event: MouseEvent) {
		if (isDragging && selectedToken) {
			finishTokenMove();
		} else if (currentMeasurement) {
			finishMeasurement();
		} else if (currentDrawing) {
			finishDrawing();
		}
		
		isDragging = false;
		isDrawing = false;
	}
	
	function handleTokenSelection(pos: { x: number; y: number }) {
		// Find token at position
		let foundToken: MapToken | null = null;
		
		mapState.tokens.forEach(token => {
			const tokenX = token.x * mapState.gridSize;
			const tokenY = token.y * mapState.gridSize;
			const tokenSize = token.size * mapState.gridSize;
			
			if (pos.x >= tokenX && pos.x <= tokenX + tokenSize &&
				pos.y >= tokenY && pos.y <= tokenY + tokenSize) {
				foundToken = token;
			}
		});
		
		if (foundToken) {
			// Check if user can move this token
			const canMove = canMoveAllTokens || 
				foundToken.owner_id === collaborationStore.currentParticipant?.user_id;
			
			if (canMove && !foundToken.locked) {
				selectedToken = foundToken;
				isDragging = true;
				dragOffset = {
					x: pos.x - foundToken.x * mapState.gridSize,
					y: pos.y - foundToken.y * mapState.gridSize
				};
			}
		} else {
			selectedToken = null;
		}
	}
	
	function moveToken(token: MapToken, pos: { x: number; y: number }) {
		// Calculate new grid position
		const newX = Math.floor((pos.x - dragOffset.x) / mapState.gridSize);
		const newY = Math.floor((pos.y - dragOffset.y) / mapState.gridSize);
		
		// Update token position locally for smooth dragging
		token.x = newX;
		token.y = newY;
	}
	
	function finishTokenMove() {
		if (!selectedToken) return;
		
		// Sync token position
		collaborationStore.updateState({
			path: ['map', 'tokens', selectedToken.id],
			value: selectedToken,
			operation: 'set',
			version: (collaborationStore.currentRoom?.state.version || 0) + 1,
			previous_version: collaborationStore.currentRoom?.state.version || 0
		});
	}
	
	function startMeasurement(pos: { x: number; y: number }) {
		currentMeasurement = {
			id: crypto.randomUUID(),
			start: pos,
			end: pos,
			color: '#3b82f6',
			owner_id: collaborationStore.currentParticipant?.user_id || ''
		};
	}
	
	function updateMeasurement(pos: { x: number; y: number }) {
		if (currentMeasurement) {
			currentMeasurement.end = pos;
		}
	}
	
	function finishMeasurement() {
		if (currentMeasurement) {
			mapState.measurements = [...mapState.measurements, currentMeasurement];
			
			// Remove measurement after 5 seconds
			setTimeout(() => {
				mapState.measurements = mapState.measurements.filter(
					m => m.id !== currentMeasurement?.id
				);
			}, 5000);
			
			currentMeasurement = null;
		}
	}
	
	function startDrawing(pos: { x: number; y: number }) {
		currentDrawing = {
			id: crypto.randomUUID(),
			type: 'line',
			points: [pos],
			color: '#ef4444',
			strokeWidth: 2,
			owner_id: collaborationStore.currentParticipant?.user_id || ''
		};
		isDrawing = true;
	}
	
	function updateDrawing(pos: { x: number; y: number }) {
		if (currentDrawing) {
			currentDrawing.points = [...currentDrawing.points, pos];
		}
	}
	
	function finishDrawing() {
		if (currentDrawing) {
			mapState.drawings = [...mapState.drawings, currentDrawing];
			
			// Sync drawing
			collaborationStore.updateState({
				path: ['map', 'drawings'],
				value: mapState.drawings,
				operation: 'set',
				version: (collaborationStore.currentRoom?.state.version || 0) + 1,
				previous_version: collaborationStore.currentRoom?.state.version || 0
			});
			
			currentDrawing = null;
		}
	}
	
	function toggleFogOfWar(pos: { x: number; y: number }) {
		if (!mapState.fogOfWar) return;
		
		const gridX = Math.floor(pos.x / mapState.gridSize);
		const gridY = Math.floor(pos.y / mapState.gridSize);
		
		if (gridY < mapState.fogOfWar.length && gridX < mapState.fogOfWar[gridY].length) {
			mapState.fogOfWar[gridY][gridX] = !mapState.fogOfWar[gridY][gridX];
			
			// Sync fog of war
			collaborationStore.updateState({
				path: ['map', 'fogOfWar'],
				value: mapState.fogOfWar,
				operation: 'set',
				version: (collaborationStore.currentRoom?.state.version || 0) + 1,
				previous_version: collaborationStore.currentRoom?.state.version || 0
			});
		}
	}
	
	function handleKeyDown(event: KeyboardEvent) {
		switch (event.key) {
			case 's':
				if (!event.ctrlKey && !event.metaKey) {
					drawingMode = 'select';
				}
				break;
			case 'm':
				drawingMode = 'measure';
				break;
			case 'd':
				drawingMode = 'draw';
				break;
			case 'f':
				if (enableFogOfWar) {
					drawingMode = 'fog';
				}
				break;
			case 'Delete':
				if (selectedToken && canEdit) {
					deleteSelectedToken();
				}
				break;
			case 'Escape':
				selectedToken = null;
				currentMeasurement = null;
				currentDrawing = null;
				isDragging = false;
				isDrawing = false;
				break;
		}
	}
	
	function handleKeyUp(event: KeyboardEvent) {
		// Handle key releases if needed
	}
	
	function handleWheel(event: WheelEvent) {
		event.preventDefault();
		
		if (event.ctrlKey || event.metaKey) {
			// Zoom
			const delta = event.deltaY > 0 ? 0.9 : 1.1;
			scale = Math.max(0.5, Math.min(3, scale * delta));
		} else {
			// Pan
			panOffset.x -= event.deltaX;
			panOffset.y -= event.deltaY;
		}
	}
	
	// Utility functions
	function getMousePosition(event: MouseEvent): { x: number; y: number } {
		const rect = canvasEl.getBoundingClientRect();
		return {
			x: (event.clientX - rect.left - panOffset.x) / scale,
			y: (event.clientY - rect.top - panOffset.y) / scale
		};
	}
	
	function calculateDistance(p1: { x: number; y: number }, p2: { x: number; y: number }): number {
		return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
	}
	
	function getContrastColor(hex: string): string {
		// Convert hex to RGB
		const r = parseInt(hex.slice(1, 3), 16);
		const g = parseInt(hex.slice(3, 5), 16);
		const b = parseInt(hex.slice(5, 7), 16);
		
		// Calculate luminance
		const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
		
		return luminance > 0.5 ? '#000000' : '#ffffff';
	}
	
	function updateCursorPosition(userId: string, data: any) {
		if (userId !== collaborationStore.currentParticipant?.user_id) {
			const participant = collaborationStore.participants.find(p => p.user_id === userId);
			if (participant) {
				cursorPositions.set(userId, {
					x: data.x,
					y: data.y,
					color: participant.color
				});
				
				// Remove cursor after inactivity
				setTimeout(() => {
					cursorPositions.delete(userId);
					cursorPositions = new Map(cursorPositions);
				}, 5000);
			}
		}
	}
	
	function deleteSelectedToken() {
		if (!selectedToken) return;
		
		mapState.tokens.delete(selectedToken.id);
		
		// Sync deletion
		collaborationStore.updateState({
			path: ['map', 'tokens', selectedToken.id],
			value: null,
			operation: 'delete',
			version: (collaborationStore.currentRoom?.state.version || 0) + 1,
			previous_version: collaborationStore.currentRoom?.state.version || 0
		});
		
		selectedToken = null;
	}
	
	// Public methods for adding tokens
	export function addToken(token: Omit<MapToken, 'id'>) {
		const newToken: MapToken = {
			...token,
			id: crypto.randomUUID()
		};
		
		mapState.tokens.set(newToken.id, newToken);
		
		// Sync new token
		collaborationStore.updateState({
			path: ['map', 'tokens', newToken.id],
			value: newToken,
			operation: 'set',
			version: (collaborationStore.currentRoom?.state.version || 0) + 1,
			previous_version: collaborationStore.currentRoom?.state.version || 0
		});
	}
	
	export function clearDrawings() {
		mapState.drawings = [];
		
		// Sync clearing
		collaborationStore.updateState({
			path: ['map', 'drawings'],
			value: [],
			operation: 'set',
			version: (collaborationStore.currentRoom?.state.version || 0) + 1,
			previous_version: collaborationStore.currentRoom?.state.version || 0
		});
	}
	
	export function resetFogOfWar() {
		if (!mapState.fogOfWar) return;
		
		mapState.fogOfWar = mapState.fogOfWar.map(row => row.map(() => true));
		
		// Sync fog reset
		collaborationStore.updateState({
			path: ['map', 'fogOfWar'],
			value: mapState.fogOfWar,
			operation: 'set',
			version: (collaborationStore.currentRoom?.state.version || 0) + 1,
			previous_version: collaborationStore.currentRoom?.state.version || 0
		});
	}
</script>

<div class="collaborative-map">
	<!-- Toolbar -->
	<div class="map-toolbar">
		<div class="tool-group">
			<button 
				class="tool-btn"
				class:active={drawingMode === 'select'}
				onclick={() => drawingMode = 'select'}
				title="Select (S)"
			>
				<svg viewBox="0 0 20 20" fill="currentColor">
					<path d="M6.684 11.316l3.465-6.93 6.93 3.465-3.465 6.93-6.93-3.465z"/>
				</svg>
			</button>
			<button 
				class="tool-btn"
				class:active={drawingMode === 'measure'}
				onclick={() => drawingMode = 'measure'}
				title="Measure (M)"
			>
				<svg viewBox="0 0 20 20" fill="currentColor">
					<path d="M5.5 16a.5.5 0 01-.5-.5V4a.5.5 0 011 0v11.5a.5.5 0 01-.5.5zm9 0a.5.5 0 01-.5-.5V4a.5.5 0 011 0v11.5a.5.5 0 01-.5.5zM3 8.5a.5.5 0 01.5-.5h13a.5.5 0 010 1h-13a.5.5 0 01-.5-.5zm0 3a.5.5 0 01.5-.5h13a.5.5 0 010 1h-13a.5.5 0 01-.5-.5z"/>
				</svg>
			</button>
			<button 
				class="tool-btn"
				class:active={drawingMode === 'draw'}
				onclick={() => drawingMode = 'draw'}
				title="Draw (D)"
			>
				<svg viewBox="0 0 20 20" fill="currentColor">
					<path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z"/>
				</svg>
			</button>
			{#if enableFogOfWar}
				<button 
					class="tool-btn"
					class:active={drawingMode === 'fog'}
					onclick={() => drawingMode = 'fog'}
					title="Fog of War (F)"
				>
					<svg viewBox="0 0 20 20" fill="currentColor">
						<path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
						<path fill-rule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd"/>
					</svg>
				</button>
			{/if}
		</div>
		
		<div class="tool-group">
			<button 
				class="tool-btn"
				onclick={() => mapState.gridVisible = !mapState.gridVisible}
				title="Toggle Grid"
			>
				<svg viewBox="0 0 20 20" fill="currentColor">
					<path d="M5 3a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2V5a2 2 0 00-2-2H5zm0 2h2v2H5V5zm0 4h2v2H5V9zm0 4h2v2H5v-2zm4-8h2v2H9V5zm0 4h2v2H9V9zm0 4h2v2H9v-2zm4-8h2v2h-2V5zm0 4h2v2h-2V9zm0 4h2v2h-2v-2z"/>
				</svg>
			</button>
			<button 
				class="tool-btn"
				onclick={clearDrawings}
				title="Clear Drawings"
			>
				<svg viewBox="0 0 20 20" fill="currentColor">
					<path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd"/>
				</svg>
			</button>
			{#if enableFogOfWar}
				<button 
					class="tool-btn"
					onclick={resetFogOfWar}
					title="Reset Fog of War"
				>
					<svg viewBox="0 0 20 20" fill="currentColor">
						<path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"/>
					</svg>
				</button>
			{/if}
		</div>
		
		<div class="tool-group">
			<div class="zoom-controls">
				<button 
					class="zoom-btn"
					onclick={() => scale = Math.max(0.5, scale - 0.1)}
				>
					-
				</button>
				<span class="zoom-level">{Math.round(scale * 100)}%</span>
				<button 
					class="zoom-btn"
					onclick={() => scale = Math.min(3, scale + 0.1)}
				>
					+
				</button>
			</div>
		</div>
	</div>
	
	<!-- Canvas Container -->
	<div 
		class="map-container"
		bind:this={overlayEl}
	>
		<canvas
			bind:this={canvasEl}
			class="battle-canvas"
			onmousedown={handleMouseDown}
			onmousemove={handleMouseMove}
			onmouseup={handleMouseUp}
			onwheel={handleWheel}
		></canvas>
		
		<!-- Mode indicator -->
		<div class="mode-indicator">
			{drawingMode === 'select' ? 'Select' : 
			 drawingMode === 'measure' ? 'Measure' :
			 drawingMode === 'draw' ? 'Draw' :
			 drawingMode === 'fog' ? 'Fog of War' : ''}
		</div>
	</div>
	
	<!-- Token palette -->
	{#if canEdit}
		<div class="token-palette">
			<h4>Add Token</h4>
			<div class="token-grid">
				<button 
					class="token-preset player"
					onclick={() => addToken({
						name: 'Player',
						type: 'player',
						x: 0,
						y: 0,
						size: 1,
						color: '#10b981',
						visible: true,
						locked: false,
						owner_id: collaborationStore.currentParticipant?.user_id
					})}
				>
					P
				</button>
				<button 
					class="token-preset enemy"
					onclick={() => addToken({
						name: 'Enemy',
						type: 'enemy',
						x: 0,
						y: 0,
						size: 1,
						color: '#ef4444',
						visible: true,
						locked: false
					})}
				>
					E
				</button>
				<button 
					class="token-preset ally"
					onclick={() => addToken({
						name: 'Ally',
						type: 'ally',
						x: 0,
						y: 0,
						size: 1,
						color: '#3b82f6',
						visible: true,
						locked: false
					})}
				>
					A
				</button>
				<button 
					class="token-preset object"
					onclick={() => addToken({
						name: 'Object',
						type: 'object',
						x: 0,
						y: 0,
						size: 1,
						color: '#6b7280',
						visible: true,
						locked: true
					})}
				>
					O
				</button>
			</div>
		</div>
	{/if}
</div>

<style>
	.collaborative-map {
		display: flex;
		flex-direction: column;
		height: 100%;
		background: var(--color-surface);
		border-radius: 0.5rem;
		overflow: hidden;
	}
	
	.map-toolbar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.5rem;
		background: var(--color-surface-secondary);
		border-bottom: 1px solid var(--color-border);
	}
	
	.tool-group {
		display: flex;
		gap: 0.25rem;
	}
	
	.tool-btn {
		width: 2.5rem;
		height: 2.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: transparent;
		border: 1px solid transparent;
		border-radius: 0.375rem;
		color: var(--color-text-secondary);
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.tool-btn:hover {
		background: var(--color-surface);
		border-color: var(--color-border);
	}
	
	.tool-btn.active {
		background: var(--color-primary);
		color: white;
		border-color: var(--color-primary);
	}
	
	.tool-btn svg {
		width: 1.25rem;
		height: 1.25rem;
	}
	
	.zoom-controls {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.25rem 0.5rem;
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
	}
	
	.zoom-btn {
		width: 1.5rem;
		height: 1.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		background: transparent;
		border: none;
		color: var(--color-text);
		font-weight: 600;
		cursor: pointer;
	}
	
	.zoom-btn:hover {
		background: var(--color-surface-secondary);
		border-radius: 0.25rem;
	}
	
	.zoom-level {
		font-size: 0.875rem;
		font-variant-numeric: tabular-nums;
		min-width: 3rem;
		text-align: center;
	}
	
	.map-container {
		flex: 1;
		position: relative;
		overflow: hidden;
		background: #f3f4f6;
	}
	
	.battle-canvas {
		cursor: crosshair;
		image-rendering: pixelated;
	}
	
	.mode-indicator {
		position: absolute;
		top: 0.5rem;
		left: 0.5rem;
		padding: 0.25rem 0.75rem;
		background: rgba(0, 0, 0, 0.7);
		color: white;
		font-size: 0.875rem;
		font-weight: 500;
		border-radius: 0.25rem;
		pointer-events: none;
	}
	
	.token-palette {
		position: absolute;
		bottom: 1rem;
		right: 1rem;
		padding: 1rem;
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		border-radius: 0.5rem;
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
	}
	
	.token-palette h4 {
		margin: 0 0 0.5rem;
		font-size: 0.875rem;
		font-weight: 600;
		color: var(--color-text-secondary);
	}
	
	.token-grid {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 0.5rem;
	}
	
	.token-preset {
		width: 2.5rem;
		height: 2.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		border: 2px solid;
		border-radius: 50%;
		font-weight: bold;
		color: white;
		cursor: pointer;
		transition: all 0.2s;
	}
	
	.token-preset:hover {
		transform: scale(1.1);
	}
	
	.token-preset.player {
		background: #10b981;
		border-color: #059669;
	}
	
	.token-preset.enemy {
		background: #ef4444;
		border-color: #dc2626;
	}
	
	.token-preset.ally {
		background: #3b82f6;
		border-color: #2563eb;
	}
	
	.token-preset.object {
		background: #6b7280;
		border-color: #4b5563;
	}
</style>