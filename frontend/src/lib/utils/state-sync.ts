// State synchronization utilities for collaborative features

import type { StateUpdate, ConflictResolution, SharedState } from '$lib/types/collaboration';

/**
 * Operational Transformation (OT) for collaborative text editing
 */
export class OperationalTransform {
	/**
	 * Transform operation A against operation B
	 * Returns transformed operation A' that can be applied after B
	 */
	static transformOperation(
		opA: StateUpdate,
		opB: StateUpdate,
		priority: 'local' | 'remote' = 'remote'
	): StateUpdate {
		// If operations are on different paths, no transformation needed
		if (!this.pathsConflict(opA.path, opB.path)) {
			return opA;
		}

		// Handle different operation types
		if (opA.operation === 'set' && opB.operation === 'set') {
			// Last-write-wins based on version or priority
			if (priority === 'remote' || opB.version > opA.version) {
				// Remote wins, discard local operation
				return { ...opA, operation: 'delete' as const, value: undefined };
			}
			// Local wins, keep operation as is
			return opA;
		}

		if (opA.operation === 'merge' && opB.operation === 'merge') {
			// Merge both objects
			const merged = { ...opB.value, ...opA.value };
			return { ...opA, value: merged };
		}

		if (opA.operation === 'delete' || opB.operation === 'delete') {
			// Delete operations take precedence
			return { ...opA, operation: 'delete' as const, value: undefined };
		}

		return opA;
	}

	/**
	 * Check if two paths conflict
	 */
	static pathsConflict(pathA: string[], pathB: string[]): boolean {
		const minLength = Math.min(pathA.length, pathB.length);
		for (let i = 0; i < minLength; i++) {
			if (pathA[i] !== pathB[i]) {
				return false;
			}
		}
		return true;
	}
}

/**
 * Vector clock for tracking causality
 */
export class VectorClock {
	private clock: Map<string, number>;

	constructor(initialClock?: Map<string, number>) {
		this.clock = new Map(initialClock || []);
	}

	/**
	 * Increment clock for a given node
	 */
	increment(nodeId: string): void {
		this.clock.set(nodeId, (this.clock.get(nodeId) || 0) + 1);
	}

	/**
	 * Update clock with another vector clock
	 */
	update(other: VectorClock): void {
		other.clock.forEach((value, key) => {
			this.clock.set(key, Math.max(this.clock.get(key) || 0, value));
		});
	}

	/**
	 * Compare with another vector clock
	 * Returns: -1 if this < other, 0 if concurrent, 1 if this > other
	 */
	compare(other: VectorClock): -1 | 0 | 1 {
		let isLess = false;
		let isGreater = false;

		// Check all keys in both clocks
		const allKeys = new Set([...this.clock.keys(), ...other.clock.keys()]);
		
		for (const key of allKeys) {
			const thisValue = this.clock.get(key) || 0;
			const otherValue = other.clock.get(key) || 0;
			
			if (thisValue < otherValue) isLess = true;
			if (thisValue > otherValue) isGreater = true;
		}

		if (isLess && !isGreater) return -1;
		if (isGreater && !isLess) return 1;
		return 0; // Concurrent
	}

	/**
	 * Check if this clock happened before another
	 */
	happenedBefore(other: VectorClock): boolean {
		return this.compare(other) === -1;
	}

	/**
	 * Clone the vector clock
	 */
	clone(): VectorClock {
		return new VectorClock(new Map(this.clock));
	}

	/**
	 * Serialize to JSON
	 */
	toJSON(): Record<string, number> {
		return Object.fromEntries(this.clock);
	}

	/**
	 * Create from JSON
	 */
	static fromJSON(json: Record<string, number>): VectorClock {
		return new VectorClock(new Map(Object.entries(json)));
	}
}

/**
 * CRDT (Conflict-free Replicated Data Type) for collaborative state
 */
export class CollaborativeCRDT {
	private state: any;
	private tombstones: Set<string>;
	private clock: VectorClock;
	private nodeId: string;

	constructor(nodeId: string, initialState: any = {}) {
		this.nodeId = nodeId;
		this.state = initialState;
		this.tombstones = new Set();
		this.clock = new VectorClock();
	}

	/**
	 * Apply a local operation
	 */
	applyLocal(operation: StateUpdate): StateUpdate {
		this.clock.increment(this.nodeId);
		
		const timestampedOp: StateUpdate = {
			...operation,
			version: Date.now(),
			previous_version: operation.previous_version || 0
		};

		this.applyOperation(timestampedOp);
		return timestampedOp;
	}

	/**
	 * Apply a remote operation
	 */
	applyRemote(operation: StateUpdate, remoteClock: VectorClock): void {
		// Update our vector clock
		this.clock.update(remoteClock);
		
		// Apply the operation
		this.applyOperation(operation);
	}

	/**
	 * Apply an operation to the state
	 */
	private applyOperation(operation: StateUpdate): void {
		let target = this.state;
		const path = [...operation.path];
		const lastKey = path.pop();

		// Navigate to the target object
		for (const key of path) {
			if (!target[key]) {
				target[key] = {};
			}
			target = target[key];
		}

		if (!lastKey) return;

		// Apply the operation
		switch (operation.operation) {
			case 'set':
				target[lastKey] = operation.value;
				break;
			case 'merge':
				if (typeof target[lastKey] === 'object' && target[lastKey] !== null) {
					target[lastKey] = { ...target[lastKey], ...operation.value };
				} else {
					target[lastKey] = operation.value;
				}
				break;
			case 'delete':
				delete target[lastKey];
				this.tombstones.add(path.concat(lastKey).join('.'));
				break;
		}
	}

	/**
	 * Get the current state
	 */
	getState(): any {
		return this.state;
	}

	/**
	 * Get the vector clock
	 */
	getClock(): VectorClock {
		return this.clock.clone();
	}

	/**
	 * Merge with another CRDT instance
	 */
	merge(other: CollaborativeCRDT): void {
		// This is a simplified merge - in production you'd want more sophisticated merging
		this.state = this.deepMerge(this.state, other.state);
		other.tombstones.forEach(t => this.tombstones.add(t));
		this.clock.update(other.clock);
	}

	/**
	 * Deep merge two objects
	 */
	private deepMerge(obj1: any, obj2: any): any {
		if (typeof obj1 !== 'object' || typeof obj2 !== 'object') {
			// For non-objects, use last-write-wins
			return obj2;
		}

		const result = { ...obj1 };

		for (const key in obj2) {
			const tombstoneKey = `${key}`; // Simplified - should use full path
			
			if (this.tombstones.has(tombstoneKey)) {
				continue; // Skip deleted items
			}

			if (key in result) {
				result[key] = this.deepMerge(result[key], obj2[key]);
			} else {
				result[key] = obj2[key];
			}
		}

		return result;
	}
}

/**
 * Debounced state synchronization
 */
export class DebouncedSync {
	private pendingUpdates: StateUpdate[] = [];
	private timer: number | null = null;
	private callback: (updates: StateUpdate[]) => void;
	private delay: number;

	constructor(callback: (updates: StateUpdate[]) => void, delay: number = 100) {
		this.callback = callback;
		this.delay = delay;
	}

	/**
	 * Add an update to the queue
	 */
	addUpdate(update: StateUpdate): void {
		this.pendingUpdates.push(update);
		this.scheduleFlush();
	}

	/**
	 * Schedule a flush of pending updates
	 */
	private scheduleFlush(): void {
		// Clear existing timer before setting a new one
		if (this.timer !== null) {
			clearTimeout(this.timer);
			this.timer = null;
		}

		this.timer = window.setTimeout(() => {
			this.flush();
		}, this.delay);
	}

	/**
	 * Flush pending updates immediately
	 */
	flush(): void {
		if (this.pendingUpdates.length === 0) return;

		const updates = [...this.pendingUpdates];
		this.pendingUpdates = [];
		
		if (this.timer !== null) {
			clearTimeout(this.timer);
			this.timer = null;
		}

		this.callback(updates);
	}

	/**
	 * Clear all pending updates
	 */
	clear(): void {
		this.pendingUpdates = [];
		if (this.timer !== null) {
			clearTimeout(this.timer);
			this.timer = null;
		}
	}
}

/**
 * Optimistic UI updates with rollback
 */
export class OptimisticUpdates<T> {
	private confirmedState: T;
	private optimisticState: T;
	private pendingOperations: Map<string, StateUpdate>;

	constructor(initialState: T) {
		this.confirmedState = initialState;
		this.optimisticState = initialState;
		this.pendingOperations = new Map();
	}

	/**
	 * Apply an optimistic update
	 */
	applyOptimistic(operationId: string, update: StateUpdate, transformer: (state: T, update: StateUpdate) => T): T {
		this.pendingOperations.set(operationId, update);
		this.optimisticState = transformer(this.optimisticState, update);
		return this.optimisticState;
	}

	/**
	 * Confirm an optimistic update
	 */
	confirm(operationId: string, finalState?: T): T {
		this.pendingOperations.delete(operationId);
		
		if (finalState) {
			this.confirmedState = finalState;
			this.optimisticState = finalState;
			
			// Reapply remaining pending operations
			this.pendingOperations.forEach((update, id) => {
				// You'd need to implement the transformer logic here
			});
		}

		return this.optimisticState;
	}

	/**
	 * Rollback an optimistic update
	 */
	rollback(operationId: string): T {
		this.pendingOperations.delete(operationId);
		
		// Reset to confirmed state and reapply remaining operations
		this.optimisticState = this.confirmedState;
		
		// Reapply remaining pending operations
		// Implementation depends on your specific needs
		
		return this.optimisticState;
	}

	/**
	 * Get the current optimistic state
	 */
	getOptimisticState(): T {
		return this.optimisticState;
	}

	/**
	 * Get the confirmed state
	 */
	getConfirmedState(): T {
		return this.confirmedState;
	}
}

/**
 * Calculate diff between two states
 */
export function calculateStateDiff(oldState: any, newState: any, path: string[] = []): StateUpdate[] {
	const updates: StateUpdate[] = [];

	// Handle primitives
	if (typeof oldState !== 'object' || typeof newState !== 'object' || 
		oldState === null || newState === null) {
		if (oldState !== newState) {
			updates.push({
				path,
				value: newState,
				operation: 'set',
				version: Date.now(),
				previous_version: 0
			});
		}
		return updates;
	}

	// Handle arrays with more efficient diffing
	if (Array.isArray(oldState) && Array.isArray(newState)) {
		// For small arrays or when structure changes significantly, replace entirely
		if (oldState.length !== newState.length || oldState.length > 100) {
			if (JSON.stringify(oldState) !== JSON.stringify(newState)) {
				updates.push({
					path,
					value: newState,
					operation: 'set',
					version: Date.now(),
					previous_version: 0
				});
			}
			return updates;
		}
		
		// For similar-sized arrays, diff individual elements
		for (let i = 0; i < Math.max(oldState.length, newState.length); i++) {
			if (i >= newState.length) {
				// Element was removed
				updates.push({
					path: [...path, String(i)],
					value: undefined,
					operation: 'delete',
					version: Date.now(),
					previous_version: 0
				});
			} else if (i >= oldState.length) {
				// Element was added
				updates.push({
					path: [...path, String(i)],
					value: newState[i],
					operation: 'set',
					version: Date.now(),
					previous_version: 0
				});
			} else if (oldState[i] !== newState[i]) {
				// Element was modified - recurse for nested structures
				if (typeof oldState[i] === 'object' && typeof newState[i] === 'object') {
					updates.push(...calculateStateDiff(oldState[i], newState[i], [...path, String(i)]));
				} else {
					updates.push({
						path: [...path, String(i)],
						value: newState[i],
						operation: 'set',
						version: Date.now(),
						previous_version: 0
					});
				}
			}
		}
		return updates;
	} else if (Array.isArray(oldState) || Array.isArray(newState)) {
		// One is array, other is not - complete replacement
		if (JSON.stringify(oldState) !== JSON.stringify(newState)) {
			updates.push({
				path,
				value: newState,
				operation: 'set',
				version: Date.now(),
				previous_version: 0
			});
		}
		return updates;
	}

	// Handle objects
	const allKeys = new Set([...Object.keys(oldState), ...Object.keys(newState)]);
	
	for (const key of allKeys) {
		const oldValue = oldState[key];
		const newValue = newState[key];
		
		if (!(key in newState)) {
			// Key was deleted
			updates.push({
				path: [...path, key],
				value: undefined,
				operation: 'delete',
				version: Date.now(),
				previous_version: 0
			});
		} else if (!(key in oldState)) {
			// Key was added
			updates.push({
				path: [...path, key],
				value: newValue,
				operation: 'set',
				version: Date.now(),
				previous_version: 0
			});
		} else if (oldValue !== newValue) {
			// Key was modified - recurse for nested objects
			updates.push(...calculateStateDiff(oldValue, newValue, [...path, key]));
		}
	}

	return updates;
}

/**
 * Apply a series of updates to a state
 */
export function applyUpdates<T>(state: T, updates: StateUpdate[]): T {
	let newState = JSON.parse(JSON.stringify(state)); // Deep clone
	
	for (const update of updates) {
		let target: any = newState;
		const path = [...update.path];
		const lastKey = path.pop();
		
		// Navigate to target
		for (const key of path) {
			if (!target[key]) {
				target[key] = {};
			}
			target = target[key];
		}
		
		if (!lastKey) continue;
		
		// Apply operation
		switch (update.operation) {
			case 'set':
				target[lastKey] = update.value;
				break;
			case 'merge':
				if (typeof target[lastKey] === 'object') {
					target[lastKey] = { ...target[lastKey], ...update.value };
				} else {
					target[lastKey] = update.value;
				}
				break;
			case 'delete':
				delete target[lastKey];
				break;
		}
	}
	
	return newState;
}