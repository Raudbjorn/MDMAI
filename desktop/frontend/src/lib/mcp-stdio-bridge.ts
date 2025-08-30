/**
 * MCP Stdio Bridge Client for Tauri Desktop App
 * Communicates with Python MCP server via Rust stdio bridge
 */

import { invoke } from '@tauri-apps/api/tauri';
import { writable, type Writable } from 'svelte/store';

// Result type for error handling (error-as-values pattern)
export type Result<T> = { ok: true; data: T } | { ok: false; error: string };

// Connection status
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export class MCPStdioBridge {
    private initialized = false;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 3;
    private healthCheckInterval: number | null = null;
    
    // Stores
    public status: Writable<ConnectionStatus> = writable('disconnected');
    public lastError: Writable<string | null> = writable(null);
    
    /**
     * Connect to the MCP server via stdio
     */
    async connect(): Promise<Result<void>> {
        try {
            this.status.set('connecting');
            
            // Start the Python MCP backend process
            await invoke('start_mcp_backend');
            
            // Verify connection with server info
            const result = await this.call('server_info', {});
            
            if (result.ok) {
                this.initialized = true;
                this.status.set('connected');
                this.reconnectAttempts = 0;
                this.startHealthCheck();
                return { ok: true, data: undefined };
            }
            
            this.status.set('error');
            this.lastError.set(result.error);
            return result;
            
        } catch (e) {
            const error = String(e);
            this.status.set('error');
            this.lastError.set(error);
            return { ok: false, error };
        }
    }
    
    /**
     * Disconnect from the MCP server
     */
    async disconnect(): Promise<void> {
        try {
            this.stopHealthCheck();
            await invoke('stop_mcp_backend');
            this.initialized = false;
            this.status.set('disconnected');
        } catch (e) {
            console.error('Error stopping MCP backend:', e);
        }
    }
    
    /**
     * Call an MCP tool/method
     */
    async call<T>(method: string, params: any = {}): Promise<Result<T>> {
        if (!this.initialized && method !== 'server_info') {
            // Try to reconnect if not initialized
            const connectResult = await this.connect();
            if (!connectResult.ok) {
                return { ok: false, error: 'Not connected to MCP server' };
            }
        }
        
        try {
            const result = await invoke<T>('mcp_call', { method, params });
            return { ok: true, data: result };
        } catch (e) {
            const error = String(e);
            
            // Check if process died and attempt reconnect
            if (error.includes('not running') || error.includes('disconnected')) {
                return this.handleDisconnection(method, params);
            }
            
            return { ok: false, error };
        }
    }
    
    /**
     * Handle disconnection and attempt reconnection
     */
    private async handleDisconnection<T>(method: string, params: any): Promise<Result<T>> {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.status.set('error');
            const error = 'MCP server disconnected - max reconnection attempts reached';
            this.lastError.set(error);
            return { ok: false, error };
        }
        
        this.reconnectAttempts++;
        console.log(`Attempting reconnection (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
        
        const connectResult = await this.connect();
        if (connectResult.ok) {
            // Retry the original call
            return this.call(method, params);
        }
        
        return { ok: false, error: connectResult.error };
    }
    
    /**
     * Start health check monitoring
     */
    private startHealthCheck(): void {
        this.healthCheckInterval = window.setInterval(async () => {
            try {
                const isHealthy = await invoke<boolean>('check_mcp_health');
                
                if (!isHealthy && this.initialized) {
                    console.warn('MCP server health check failed');
                    this.status.set('error');
                    
                    // Attempt to reconnect
                    await this.connect();
                }
            } catch (e) {
                console.error('Health check error:', e);
            }
        }, 30000); // Every 30 seconds
    }
    
    /**
     * Stop health check monitoring
     */
    private stopHealthCheck(): void {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
            this.healthCheckInterval = null;
        }
    }
    
    // Convenience methods for common MCP tools
    
    async search(query: string, options: any = {}): Promise<Result<any>> {
        return this.call('search', { query, ...options });
    }
    
    async addSource(pdfPath: string, rulebookName: string, system: string): Promise<Result<any>> {
        return this.call('add_source', {
            pdf_path: pdfPath,
            rulebook_name: rulebookName,
            system
        });
    }
    
    async listSources(): Promise<Result<any>> {
        return this.call('list_sources', {});
    }
    
    async createCampaign(name: string, system: string, description?: string): Promise<Result<any>> {
        return this.call('create_campaign', {
            name,
            system,
            description
        });
    }
    
    async getCampaignData(campaignId: string): Promise<Result<any>> {
        return this.call('get_campaign_data', {
            campaign_id: campaignId
        });
    }
    
    async updateCampaignData(campaignId: string, data: any): Promise<Result<any>> {
        return this.call('update_campaign_data', {
            campaign_id: campaignId,
            ...data
        });
    }
    
    async startSession(campaignId: string, title?: string): Promise<Result<any>> {
        return this.call('start_session', {
            campaign_id: campaignId,
            title
        });
    }
    
    async rollDice(notation: string): Promise<Result<any>> {
        return this.call('roll_dice', {
            notation
        });
    }
    
    async generateCharacter(options: any = {}): Promise<Result<any>> {
        return this.call('generate_character', options);
    }
    
    async generateNPC(options: any = {}): Promise<Result<any>> {
        return this.call('generate_npc', options);
    }
}

// Singleton instance
export const mcpBridge = new MCPStdioBridge();

// Convenience stores
export const mcpStatus = mcpBridge.status;
export const mcpError = mcpBridge.lastError;

// Auto-connect on app start
if (typeof window !== 'undefined') {
    // Only run in browser environment
    mcpBridge.connect().then(result => {
        if (!result.ok) {
            console.error('Failed to connect to MCP server:', result.error);
        }
    });
}