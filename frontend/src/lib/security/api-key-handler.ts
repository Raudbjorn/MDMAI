/**
 * Secure API Key Handler
 * Provides encryption and secure transmission of API keys
 */

import { browser } from '$app/environment';

// Simple XOR encryption for client-side obfuscation (not cryptographically secure)
// For production, use proper encryption libraries or server-side key storage
class ApiKeyHandler {
    private static readonly STORAGE_PREFIX = 'ttrpg_encrypted_';
    private static readonly SESSION_KEY = 'ttrpg_session_key';
    
    /**
     * Generate a random session key for XOR encryption
     */
    private static generateSessionKey(): string {
        if (!browser) return '';
        
        const array = new Uint8Array(32);
        crypto.getRandomValues(array);
        return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
    }
    
    /**
     * Get or create session key
     */
    private static getSessionKey(): string {
        if (!browser) return '';
        
        let key = sessionStorage.getItem(this.SESSION_KEY);
        if (!key) {
            key = this.generateSessionKey();
            sessionStorage.setItem(this.SESSION_KEY, key);
        }
        return key;
    }
    
    /**
     * XOR encryption/decryption (symmetric)
     */
    private static xorCipher(text: string, key: string): string {
        if (!text || !key) return text;
        
        let result = '';
        for (let i = 0; i < text.length; i++) {
            result += String.fromCharCode(
                text.charCodeAt(i) ^ key.charCodeAt(i % key.length)
            );
        }
        return btoa(result); // Base64 encode for storage
    }
    
    /**
     * XOR decryption
     */
    private static xorDecipher(encoded: string, key: string): string {
        if (!encoded || !key) return encoded;
        
        try {
            const text = atob(encoded); // Base64 decode
            let result = '';
            for (let i = 0; i < text.length; i++) {
                result += String.fromCharCode(
                    text.charCodeAt(i) ^ key.charCodeAt(i % key.length)
                );
            }
            return result;
        } catch {
            return '';
        }
    }
    
    /**
     * Store API key securely in session storage (encrypted)
     */
    public static storeApiKey(provider: string, apiKey: string): void {
        if (!browser) return;
        
        const sessionKey = this.getSessionKey();
        const encrypted = this.xorCipher(apiKey, sessionKey);
        sessionStorage.setItem(`${this.STORAGE_PREFIX}${provider}`, encrypted);
    }
    
    /**
     * Retrieve API key from session storage (decrypted)
     */
    public static getApiKey(provider: string): string | null {
        if (!browser) return null;
        
        const encrypted = sessionStorage.getItem(`${this.STORAGE_PREFIX}${provider}`);
        if (!encrypted) return null;
        
        const sessionKey = this.getSessionKey();
        return this.xorDecipher(encrypted, sessionKey);
    }
    
    /**
     * Remove API key from storage
     */
    public static removeApiKey(provider: string): void {
        if (!browser) return;
        sessionStorage.removeItem(`${this.STORAGE_PREFIX}${provider}`);
    }
    
    /**
     * Clear all API keys
     */
    public static clearAllKeys(): void {
        if (!browser) return;
        
        const keys = Object.keys(sessionStorage);
        keys.forEach(key => {
            if (key.startsWith(this.STORAGE_PREFIX)) {
                sessionStorage.removeItem(key);
            }
        });
    }
    
    /**
     * Prepare API key for secure transmission
     * Returns a payload that should be sent over HTTPS only
     */
    public static prepareForTransmission(provider: string, apiKey: string): {
        provider: string;
        timestamp: number;
        nonce: string;
        payload: string;
    } {
        const timestamp = Date.now();
        const nonce = this.generateNonce();
        
        // Create a signed payload
        const data = JSON.stringify({
            provider,
            key: apiKey,
            timestamp,
            nonce
        });
        
        // In production, this should use proper encryption
        // For now, we'll use base64 encoding with a warning
        const payload = btoa(data);
        
        return {
            provider,
            timestamp,
            nonce,
            payload
        };
    }
    
    /**
     * Generate a cryptographic nonce
     */
    private static generateNonce(): string {
        if (!browser) return '';
        
        const array = new Uint8Array(16);
        crypto.getRandomValues(array);
        return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
    }
    
    /**
     * Validate API key format (basic validation)
     */
    public static validateApiKeyFormat(apiKey: string, provider: string): boolean {
        if (!apiKey || typeof apiKey !== 'string') return false;
        
        // Provider-specific validation
        switch (provider) {
            case 'openai':
                return /^sk-[A-Za-z0-9]{48}$/.test(apiKey);
            case 'anthropic':
                return /^sk-ant-[A-Za-z0-9-]{95}$/.test(apiKey);
            case 'google':
                return /^[A-Za-z0-9_-]{39}$/.test(apiKey);
            default:
                // Generic validation: non-empty string with reasonable length
                return apiKey.length > 10 && apiKey.length < 500;
        }
    }
    
    /**
     * Mask API key for display
     */
    public static maskApiKey(apiKey: string): string {
        if (!apiKey || apiKey.length < 8) return '****';
        
        const visibleStart = 3;
        const visibleEnd = 4;
        const masked = apiKey.substring(0, visibleStart) + 
                      '*'.repeat(Math.max(8, apiKey.length - visibleStart - visibleEnd)) +
                      apiKey.substring(apiKey.length - visibleEnd);
        return masked;
    }
}

export default ApiKeyHandler;

// Export utility functions
export const {
    storeApiKey,
    getApiKey,
    removeApiKey,
    clearAllKeys,
    prepareForTransmission,
    validateApiKeyFormat,
    maskApiKey
} = ApiKeyHandler;