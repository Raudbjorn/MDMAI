/**
 * A debounced function with additional control methods.
 * @template T - The original function type
 */
export interface DebouncedFunction<T extends (...args: any[]) => any> {
	(...args: Parameters<T>): ReturnType<T>;
	/**
	 * Immediately invoke the debounced function and return the result.
	 */
	flush(): ReturnType<T>;
	/**
	 * Cancel the delayed function invocation.
	 */
	cancel(): void;
}

/**
 * Creates a debounced version of a function that delays invoking func until after
 * wait milliseconds have elapsed since the last time the debounced function was invoked.
 * 
 * @template T - The function type to debounce
 * @param func - The function to debounce
 * @param wait - The number of milliseconds to delay
 * @param options - Optional configuration
 * @returns A debounced function with flush and cancel methods
 */

export function debounce<T extends (...args: any[]) => any>(
	func: T,
	wait: number,
	options?: {
		/** Invoke on the leading edge of the timeout */
		leading?: boolean;
		/** Invoke on the trailing edge of the timeout */
		trailing?: boolean;
		/** Maximum time func is allowed to be delayed before it's invoked */
		maxWait?: number;
	}
): DebouncedFunction<T> {
	const leading = options?.leading !== false;
	const trailing = options?.trailing !== false;
	const hasMaxWait = typeof options?.maxWait === 'number';
	const maxWait = hasMaxWait ? Math.max(options?.maxWait ?? wait, wait) : 0;

	let timeoutId: ReturnType<typeof setTimeout> | null = null;
	let maxTimeoutId: ReturnType<typeof setTimeout> | null = null;
	let result: ReturnType<T>;
	let lastCallTime: number | null = null;
	let lastInvokeTime = 0;
	let lastArgs: Parameters<T> | null = null;
	let lastThis: any;

	function invokeFunc(time: number): ReturnType<T> {
		const args = lastArgs;
		const thisArg = lastThis;

		lastArgs = null;
		lastThis = undefined;
		lastInvokeTime = time;
		result = func.apply(thisArg, args!);
		return result;
	}

	function shouldInvoke(time: number): boolean {
		const timeSinceLastCall = time - (lastCallTime || 0);
		const timeSinceLastInvoke = time - lastInvokeTime;

		// Either this is the first call, activity has stopped and we're at the trailing
		// edge, the system time has gone backwards and we're treating it as the
		// trailing edge, or we've hit the `maxWait` limit.
		return (lastCallTime === null || (timeSinceLastCall >= wait) ||
			(timeSinceLastCall < 0) || (hasMaxWait && timeSinceLastInvoke >= maxWait));
	}

	function timerExpired() {
		const time = Date.now();
		if (shouldInvoke(time)) {
			return trailingEdge(time);
		}
		// Restart the timer.
		const timeSinceLastCall = time - (lastCallTime || 0);
		const timeSinceLastInvoke = time - lastInvokeTime;
		const timeWaiting = wait - timeSinceLastCall;

		const remainingWait = hasMaxWait
			? Math.min(timeWaiting, maxWait - timeSinceLastInvoke)
			: timeWaiting;

		timeoutId = setTimeout(timerExpired, remainingWait);
	}

	function maxTimerExpired() {
		const time = Date.now();
		trailingEdge(time);
	}

	function leadingEdge(time: number): ReturnType<T> {
		// Reset any `maxWait` timer.
		lastInvokeTime = time;
		// Start the timer for the trailing edge.
		timeoutId = setTimeout(timerExpired, wait);
		// Invoke the leading edge.
		return leading ? invokeFunc(time) : result;
	}
	function trailingEdge(time: number) {
		timeoutId = null;
		if (maxTimeoutId) {
			clearTimeout(maxTimeoutId);
			maxTimeoutId = null;
		}

		if (trailing && lastArgs) {
			return invokeFunc(time);
		}
		lastArgs = null;
		return result;
	}

	function cancel() {
		if (timeoutId !== null) {
			clearTimeout(timeoutId);
		}
		if (maxTimeoutId !== null) {
			clearTimeout(maxTimeoutId);
		}
		lastInvokeTime = 0;
		lastCallTime = null;
		timeoutId = null;
		maxTimeoutId = null;
		lastArgs = null;
		lastThis = undefined;
	}

	function flush() {
		return timeoutId === null ? result : trailingEdge(Date.now());
	}

	function debounced(this: any, ...args: Parameters<T>): ReturnType<T> {
		const time = Date.now();
		const isInvoking = shouldInvoke(time);

		lastThis = this;
		lastArgs = args;
		lastCallTime = time;

		if (isInvoking) {
			if (timeoutId === null) {
				return leadingEdge(time);
			}
			if (hasMaxWait) {
				// Handle invocations in a tight loop.
				if (maxTimeoutId === null) {
					maxTimeoutId = setTimeout(maxTimerExpired, maxWait);
				}
				timeoutId = setTimeout(timerExpired, wait);
				return invokeFunc(time);
			}
		}
		if (timeoutId === null) {
			timeoutId = setTimeout(timerExpired, wait);
		}
		return result;
	}

	debounced.cancel = cancel;
	debounced.flush = flush;

	// Type-safe return without assertions
	return debounced as DebouncedFunction<T>;
}

/**
 * Creates a throttled version of a function that only invokes func at most once per
 * every wait milliseconds.
 * 
 * @template T - The function type to throttle
 * @param func - The function to throttle
 * @param wait - The number of milliseconds to throttle invocations to
 * @param options - Optional configuration
 * @returns A throttled function with flush and cancel methods
 */
export function throttle<T extends (...args: any[]) => any>(
	func: T,
	wait: number,
	options?: {
		/** Invoke on the leading edge of the timeout */
		leading?: boolean;
		/** Invoke on the trailing edge of the timeout */
		trailing?: boolean;
	}
): DebouncedFunction<T> {
	const leading = options?.leading !== false;
	const trailing = options?.trailing !== false;

	return debounce(func, wait, { 
		leading, 
		trailing, 
		maxWait: wait 
	});
}