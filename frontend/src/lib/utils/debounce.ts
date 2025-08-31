/**
 * Simplified, efficient debounce and throttle utilities
 * 
 * Note: This is a simplified implementation without advanced options.
 * If you need features like `leading`, `maxWait`, or `trailing`, consider
 * using a more comprehensive library like lodash-es/debounce.
 * 
 * Current implementation provides:
 * - Standard debounce (wait after last call)
 * - Standard throttle (limit frequency)  
 * - cancel() method for cleanup
 */

export function debounce<T extends (...args: any[]) => any>(
	func: T,
	wait: number
): T & { cancel: () => void } {
	let timeoutId: ReturnType<typeof setTimeout> | null = null;

	const debounced = ((...args: Parameters<T>) => {
		clearTimeout(timeoutId!);
		timeoutId = setTimeout(() => func(...args), wait);
	}) as T & { cancel: () => void };

	debounced.cancel = () => {
		clearTimeout(timeoutId!);
		timeoutId = null;
	};

	return debounced;
}

export function throttle<T extends (...args: any[]) => any>(
	func: T,
	wait: number
): T & { cancel: () => void } {
	let lastCall = 0;
	let timeoutId: ReturnType<typeof setTimeout> | null = null;

	const throttled = ((...args: Parameters<T>) => {
		const now = Date.now();
		const timeSinceLastCall = now - lastCall;

		if (timeSinceLastCall >= wait) {
			lastCall = now;
			func(...args);
		} else if (!timeoutId) {
			timeoutId = setTimeout(() => {
				lastCall = Date.now();
				timeoutId = null;
				func(...args);
			}, wait - timeSinceLastCall);
		}
	}) as T & { cancel: () => void };

	throttled.cancel = () => {
		clearTimeout(timeoutId!);
		timeoutId = null;
		lastCall = 0;
	};

	return throttled;
}