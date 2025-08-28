/**
 * Client-side encryption utilities for secure credential handling
 * Uses Web Crypto API for encryption before sending to backend
 */

/**
 * Generate a random encryption key
 */
export async function generateKey(): Promise<CryptoKey> {
	return await crypto.subtle.generateKey(
		{
			name: 'AES-GCM',
			length: 256
		},
		true,
		['encrypt', 'decrypt']
	);
}

/**
 * Export a CryptoKey to base64 string
 */
export async function exportKey(key: CryptoKey): Promise<string> {
	const exported = await crypto.subtle.exportKey('raw', key);
	return btoa(String.fromCharCode(...new Uint8Array(exported)));
}

/**
 * Import a base64 key string to CryptoKey
 */
export async function importKey(keyString: string): Promise<CryptoKey> {
	const keyData = Uint8Array.from(atob(keyString), c => c.charCodeAt(0));
	return await crypto.subtle.importKey(
		'raw',
		keyData,
		'AES-GCM',
		true,
		['encrypt', 'decrypt']
	);
}

/**
 * Encrypt sensitive data
 */
export async function encrypt(text: string, key: CryptoKey): Promise<{ encrypted: string; iv: string }> {
	const encoder = new TextEncoder();
	const data = encoder.encode(text);
	
	// Generate random IV
	const iv = crypto.getRandomValues(new Uint8Array(12));
	
	// Encrypt
	const encrypted = await crypto.subtle.encrypt(
		{
			name: 'AES-GCM',
			iv: iv
		},
		key,
		data
	);
	
	// Convert to base64
	const encryptedBase64 = btoa(String.fromCharCode(...new Uint8Array(encrypted)));
	const ivBase64 = btoa(String.fromCharCode(...iv));
	
	return {
		encrypted: encryptedBase64,
		iv: ivBase64
	};
}

/**
 * Decrypt sensitive data
 */
export async function decrypt(encryptedBase64: string, ivBase64: string, key: CryptoKey): Promise<string> {
	// Convert from base64
	const encrypted = Uint8Array.from(atob(encryptedBase64), c => c.charCodeAt(0));
	const iv = Uint8Array.from(atob(ivBase64), c => c.charCodeAt(0));
	
	// Decrypt
	const decrypted = await crypto.subtle.decrypt(
		{
			name: 'AES-GCM',
			iv: iv
		},
		key,
		encrypted
	);
	
	// Convert to string
	const decoder = new TextDecoder();
	return decoder.decode(decrypted);
}

/**
 * Hash a value using SHA-256
 */
export async function hash(text: string): Promise<string> {
	const encoder = new TextEncoder();
	const data = encoder.encode(text);
	const hashBuffer = await crypto.subtle.digest('SHA-256', data);
	const hashArray = Array.from(new Uint8Array(hashBuffer));
	return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Generate a secure random token
 */
export function generateToken(length: number = 32): string {
	const array = new Uint8Array(length);
	crypto.getRandomValues(array);
	return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * Validate API key format (basic validation)
 */
export function validateApiKeyFormat(key: string, provider: 'anthropic' | 'openai' | 'google'): boolean {
	const patterns = {
		anthropic: /^sk-ant-api\d{2}-[\w-]{48,}$/,
		openai: /^sk-[a-zA-Z0-9]{48,}$/,
		google: /^AIza[a-zA-Z0-9-_]{35}$/
	};
	
	const pattern = patterns[provider];
	return pattern ? pattern.test(key) : key.length > 0;
}

/**
 * Mask sensitive data for display
 */
export function maskSensitiveData(text: string, visibleChars: number = 4): string {
	if (text.length <= visibleChars * 2) {
		return '*'.repeat(text.length);
	}
	
	const start = text.substring(0, visibleChars);
	const end = text.substring(text.length - visibleChars);
	const masked = '*'.repeat(Math.max(4, text.length - visibleChars * 2));
	
	return `${start}${masked}${end}`;
}

/**
 * Store encrypted credentials in localStorage
 */
export async function storeEncryptedCredential(
	provider: string, 
	credential: string, 
	key: CryptoKey
): Promise<void> {
	const { encrypted, iv } = await encrypt(credential, key);
	const data = {
		encrypted,
		iv,
		provider,
		timestamp: Date.now()
	};
	
	localStorage.setItem(`credential_${provider}`, JSON.stringify(data));
}

/**
 * Retrieve and decrypt credentials from localStorage
 */
export async function retrieveEncryptedCredential(
	provider: string, 
	key: CryptoKey
): Promise<string | null> {
	const stored = localStorage.getItem(`credential_${provider}`);
	if (!stored) return null;
	
	try {
		const data = JSON.parse(stored);
		return await decrypt(data.encrypted, data.iv, key);
	} catch (error) {
		console.error('Failed to decrypt credential:', error);
		return null;
	}
}

/**
 * Clear stored credentials
 */
export function clearStoredCredentials(provider?: string): void {
	if (provider) {
		localStorage.removeItem(`credential_${provider}`);
	} else {
		// Clear all credentials
		Object.keys(localStorage)
			.filter(key => key.startsWith('credential_'))
			.forEach(key => localStorage.removeItem(key));
	}
}

/**
 * Check if Web Crypto API is available
 */
export function isCryptoAvailable(): boolean {
	return typeof crypto !== 'undefined' && 
	       crypto.subtle !== undefined &&
	       typeof crypto.subtle.encrypt === 'function';
}

/**
 * Security recommendations checker
 */
export function checkSecurityRecommendations(): string[] {
	const recommendations: string[] = [];
	
	// Check HTTPS
	if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
		recommendations.push('Use HTTPS for secure credential handling');
	}
	
	// Check Web Crypto API
	if (!isCryptoAvailable()) {
		recommendations.push('Web Crypto API not available - upgrade your browser');
	}
	
	// Check for secure context
	if (!window.isSecureContext) {
		recommendations.push('Not in a secure context - some features may be limited');
	}
	
	return recommendations;
}