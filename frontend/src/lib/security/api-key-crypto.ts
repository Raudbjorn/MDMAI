/**
 * Secure API Key Cryptography using Web Crypto API
 * Uses AES-GCM for proper encryption instead of XOR
 */

import { browser } from '$app/environment';

class SecureApiKeyHandler {
    private static readonly STORAGE_PREFIX = 'ttrpg_secure_';
    private static readonly KEY_NAME = 'ttrpg_encryption_key';
    
    /**
     * Generate a cryptographic key for AES-GCM encryption
     */
    private static async generateKey(): Promise<CryptoKey> {
        return await crypto.subtle.generateKey(
            {
                name: 'AES-GCM',
                length: 256
            },
            true, // extractable
            ['encrypt', 'decrypt']
        );
    }
    
    /**
     * Get or create encryption key
     */
    private static async getOrCreateKey(): Promise<CryptoKey | null> {
        if (!browser) return null;
        
        try {
            // Try to get existing key from sessionStorage for better security
            const storedKey = sessionStorage.getItem(this.KEY_NAME);
            if (storedKey) {
                const keyData = JSON.parse(storedKey);
                return await crypto.subtle.importKey(
                    'jwk',
                    keyData,
                    { name: 'AES-GCM', length: 256 },
                    true,
                    ['encrypt', 'decrypt']
                );
            }
            
            // Generate new key
            const key = await this.generateKey();
            const exportedKey = await crypto.subtle.exportKey('jwk', key);
            sessionStorage.setItem(this.KEY_NAME, JSON.stringify(exportedKey));
            return key;
        } catch (error) {
            console.error('Failed to get/create encryption key:', error);
            return null;
        }
    }
    
    /**
     * Encrypt data using AES-GCM
     */
    private static async encrypt(text: string, key: CryptoKey): Promise<{ encrypted: ArrayBuffer; iv: Uint8Array }> {
        const encoder = new TextEncoder();
        const data = encoder.encode(text);
        
        // Generate random IV
        const iv = crypto.getRandomValues(new Uint8Array(12));
        
        const encrypted = await crypto.subtle.encrypt(
            {
                name: 'AES-GCM',
                iv: iv as BufferSource
            },
            key,
            data
        );
        
        return { encrypted, iv };
    }
    
    /**
     * Decrypt data using AES-GCM
     */
    private static async decrypt(encrypted: ArrayBuffer, iv: Uint8Array, key: CryptoKey): Promise<string> {
        const decrypted = await crypto.subtle.decrypt(
            {
                name: 'AES-GCM',
                iv: iv as BufferSource
            },
            key,
            encrypted
        );
        
        const decoder = new TextDecoder();
        return decoder.decode(decrypted);
    }
    
    /**
     * Store API key securely using AES-GCM encryption
     */
    public static async storeApiKey(provider: string, apiKey: string): Promise<void> {
        if (!browser) return;
        
        try {
            const key = await this.getOrCreateKey();
            if (!key) throw new Error('Failed to get encryption key');
            
            const { encrypted, iv } = await this.encrypt(apiKey, key);
            
            // Convert to base64 for storage
            const encryptedArray = new Uint8Array(encrypted);
            const combined = new Uint8Array(iv.length + encryptedArray.length);
            combined.set(iv);
            combined.set(encryptedArray, iv.length);
            
            const base64 = btoa(String.fromCharCode(...combined));
            sessionStorage.setItem(`${this.STORAGE_PREFIX}${provider}`, base64);
        } catch (error) {
            console.error('Failed to store API key securely:', error);
            throw error;
        }
    }
    
    /**
     * Retrieve API key from secure storage
     */
    public static async getApiKey(provider: string): Promise<string | null> {
        if (!browser) return null;
        
        try {
            const stored = sessionStorage.getItem(`${this.STORAGE_PREFIX}${provider}`);
            if (!stored) return null;
            
            const key = await this.getOrCreateKey();
            if (!key) throw new Error('Failed to get encryption key');
            
            // Decode from base64
            const combined = Uint8Array.from(atob(stored), c => c.charCodeAt(0));
            const iv = combined.slice(0, 12);
            const encrypted = combined.slice(12);
            
            return await this.decrypt(encrypted.buffer, iv, key);
        } catch (error) {
            console.error('Failed to retrieve API key:', error);
            return null;
        }
    }
    
    /**
     * Remove API key from storage
     */
    public static removeApiKey(provider: string): void {
        if (!browser) return;
        sessionStorage.removeItem(`${this.STORAGE_PREFIX}${provider}`);
    }
    
    /**
     * Clear all API keys and encryption key
     */
    public static clearAllKeys(): void {
        if (!browser) return;
        
        // Clear all encrypted keys
        const keys = Object.keys(sessionStorage);
        keys.forEach(key => {
            if (key.startsWith(this.STORAGE_PREFIX)) {
                sessionStorage.removeItem(key);
            }
        });
        
        // Clear the encryption key from sessionStorage
        sessionStorage.removeItem(this.KEY_NAME);
    }
    
    /**
     * Validate API key format (reuse from original)
     */
    public static validateApiKeyFormat(apiKey: string, provider: string): boolean {
        if (!apiKey || typeof apiKey !== 'string') return false;
        
        switch (provider) {
            case 'openai':
                return /^sk-[A-Za-z0-9]{48}$/.test(apiKey);
            case 'anthropic':
                return /^sk-ant-[A-Za-z0-9-]{95}$/.test(apiKey);
            case 'google':
                return /^[A-Za-z0-9_-]{39}$/.test(apiKey);
            default:
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

export default SecureApiKeyHandler;

// Export utility functions
export const {
    storeApiKey,
    getApiKey,
    removeApiKey,
    clearAllKeys,
    validateApiKeyFormat,
    maskApiKey
} = SecureApiKeyHandler;