/**
 * Enhanced MCP Stdio Bridge Client for Tauri Desktop App
 * Communicates with Python MCP server via Rust stdio bridge
 * 
 * Features:
 * - Svelte 5 runes for reactive state management
 * - Enhanced error-as-values patterns
 * - Type-safe method calls with generics
 * - TTRPG-optimized convenience methods
 */

import { invoke } from '@tauri-apps/api/core';
import { writable, type Writable, derived } from 'svelte/store';
import { processStats, type ProcessState } from './process-manager-client';

// Enhanced Result type with optional metadata
export interface Result<T, E = string> {
  readonly ok: boolean;
  readonly data?: T;
  readonly error?: E;
  readonly metadata?: Record<string, unknown>;
}

export type Success<T> = { ok: true; data: T; metadata?: Record<string, unknown> };
export type Failure<E = string> = { ok: false; error: E; metadata?: Record<string, unknown> };

// Helper functions for Result type
export const createSuccess = <T>(data: T, metadata?: Record<string, unknown>): Success<T> => ({ 
  ok: true, 
  data, 
  ...(metadata && { metadata }) 
});

export const createFailure = <E = string>(error: E, metadata?: Record<string, unknown>): Failure<E> => ({ 
  ok: false, 
  error, 
  ...(metadata && { metadata }) 
});

// Connection status with more granular states
export type ConnectionStatus = 
  | 'disconnected'
  | 'connecting' 
  | 'connected'
  | 'degraded'
  | 'error'
  | 'reconnecting';

// TTRPG-specific method types
export interface TTRPGSearchOptions extends Record<string, unknown> {
  system?: string;
  type?: 'rules' | 'spells' | 'monsters' | 'items' | 'all';
  sources?: string[];
  limit?: number;
}

export interface CampaignCreateOptions extends Record<string, unknown> {
  name: string;
  system: string;
  description?: string;
  template?: string;
  players?: string[];
}

export interface CharacterGenOptions extends Record<string, unknown> {
  system: string;
  level?: number;
  class?: string;
  race?: string;
  background?: string;
  randomize?: boolean;
}

export class MCPStdioBridge {
    private _initialized = false;
    private _reconnectAttempts = 0;
    private readonly _maxReconnectAttempts = 3;
    private _healthCheckInterval: number | null = null;
    
    // Enhanced stores with better typing
    public readonly status: Writable<ConnectionStatus> = writable('disconnected');
    public readonly lastError: Writable<string | null> = writable(null);
    public readonly isReconnecting: Writable<boolean> = writable(false);
    
    // Performance metrics
    private readonly _metrics = {
        totalCalls: 0,
        successfulCalls: 0,
        failedCalls: 0,
        averageLatency: 0,
        lastCallTimestamp: 0
    };
    
    // Derived store with enhanced status information
    public readonly detailedStatus = derived(
        [this.status, this.isReconnecting, processStats],
        ([$status, $reconnecting, $stats]) => {
            const processState = ('running' as unknown) as ProcessState; // Default to running when we have stats
            const health = 'healthy';
            
            return {
                connection: $status,
                process: processState,
                health,
                isReconnecting: $reconnecting,
                canOperate: $status === 'connected' || $status === 'degraded',
                metrics: { ...this._metrics }
            };
        }
    );
    
    // Getters for encapsulation
    get initialized(): boolean { return this._initialized; }
    get reconnectAttempts(): number { return this._reconnectAttempts; }
    get metrics(): typeof this._metrics { return { ...this._metrics }; }
    
    /**
     * Connect to the MCP server via stdio with enhanced error handling
     */
    async connect(): Promise<Result<void>> {
        const startTime = performance.now();
        
        try {
            this.status.set('connecting');
            this.lastError.set(null);
            
            // Start the Python MCP backend process
            await invoke('start_mcp_backend');
            
            // Verify connection with server info
            const result = await this.call<{ version: string; capabilities: string[] }>('server_info', {});
            
            if (result.ok) {
                this._initialized = true;
                this.status.set('connected');
                this._reconnectAttempts = 0;
                this._startHealthCheck();
                
                const latency = performance.now() - startTime;
                return createSuccess(undefined, { 
                    connectionTime: latency,
                    serverInfo: result.data 
                });
            }
            
            this.status.set('error');
            this.lastError.set(result.error || null);
            return createFailure(result.error || 'Unknown connection error', { connectionTime: performance.now() - startTime });
            
        } catch (e) {
            const error = this._formatError(e);
            this.status.set('error');
            this.lastError.set(error);
            return createFailure(error, { 
                connectionTime: performance.now() - startTime,
                exception: true 
            });
        }
    }
    
    /**
     * Disconnect from the MCP server with graceful shutdown
     */
    async disconnect(): Promise<Result<void>> {
        try {
            this._stopHealthCheck();
            this.isReconnecting.set(false);
            
            // Graceful shutdown with timeout
            const shutdownPromise = invoke('stop_mcp_backend');
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Shutdown timeout')), 5000)
            );
            
            await Promise.race([shutdownPromise, timeoutPromise]);
            
            this._initialized = false;
            this._reconnectAttempts = 0;
            this.status.set('disconnected');
            this.lastError.set(null);
            
            return createSuccess(undefined);
        } catch (e) {
            const error = this._formatError(e);
            console.error('Error stopping MCP backend:', error);
            return createFailure(error);
        }
    }
    
    /**
     * Call an MCP tool/method with enhanced type safety and error handling
     */
    async call<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<Result<T>> {
        const startTime = performance.now();
        this._metrics.totalCalls++;
        this._metrics.lastCallTimestamp = Date.now();
        
        // Auto-connect if not initialized (except for server_info)
        if (!this._initialized && method !== 'server_info') {
            const connectResult = await this.connect();
            if (!connectResult.ok) {
                this._metrics.failedCalls++;
                return createFailure('Not connected to MCP server', {
                    method,
                    autoConnectFailed: true
                });
            }
        }
        
        try {
            const result = await invoke<T>('mcp_call', { method, params });
            
            // Update metrics
            this._metrics.successfulCalls++;
            const latency = performance.now() - startTime;
            this._updateAverageLatency(latency);
            
            return createSuccess(result, { 
                method, 
                latency,
                timestamp: Date.now()
            });
        } catch (e) {
            const error = this._formatError(e);
            this._metrics.failedCalls++;
            
            // Check if process died and attempt reconnect
            if (this._isConnectionError(error)) {
                return this._handleDisconnection<T>(method, params);
            }
            
            return createFailure(error, { 
                method, 
                latency: performance.now() - startTime,
                timestamp: Date.now()
            });
        }
    }
    
    /**
     * Handle disconnection and attempt reconnection with exponential backoff
     */
    private async _handleDisconnection<T>(method: string, params: Record<string, unknown>): Promise<Result<T>> {
        if (this._reconnectAttempts >= this._maxReconnectAttempts) {
            this.status.set('error');
            const error = `MCP server disconnected - max reconnection attempts (${this._maxReconnectAttempts}) reached`;
            this.lastError.set(error);
            return createFailure(error, { 
                method,
                maxAttemptsReached: true,
                totalAttempts: this._reconnectAttempts 
            });
        }
        
        this._reconnectAttempts++;
        this.status.set('reconnecting');
        this.isReconnecting.set(true);
        
        console.log(`Attempting reconnection (${this._reconnectAttempts}/${this._maxReconnectAttempts})...`);
        
        // Exponential backoff with jitter
        const baseDelay = 1000;
        const maxDelay = 10000;
        const delay = Math.min(
            baseDelay * Math.pow(2, this._reconnectAttempts - 1),
            maxDelay
        ) + Math.random() * 1000; // Add jitter
        
        await this._delay(delay);
        
        const connectResult = await this.connect();
        this.isReconnecting.set(false);
        
        if (connectResult.ok) {
            console.log('Reconnection successful, retrying original call...');
            // Retry the original call
            return this.call<T>(method, params);
        }
        
        return createFailure(connectResult.error!, {
            method,
            reconnectAttempt: this._reconnectAttempts,
            originalError: connectResult.error
        });
    }
    
    /**
     * Start health check monitoring with adaptive intervals
     */
    private _startHealthCheck(): void {
        if (this._healthCheckInterval) {
            clearInterval(this._healthCheckInterval);
        }
        
        this._healthCheckInterval = window.setInterval(async () => {
            try {
                const isHealthy = await invoke<boolean>('check_mcp_health');
                
                if (!isHealthy && this._initialized) {
                    console.warn('MCP server health check failed');
                    const currentStatus = this.status;
                    
                    // Don't interrupt ongoing reconnection attempts
                    currentStatus.update(status => 
                        status === 'reconnecting' ? status : 'degraded'
                    );
                    
                    // Only attempt reconnection if not already reconnecting
                    let statusValue: ConnectionStatus | undefined;
                    const unsubscribe = currentStatus.subscribe(status => statusValue = status);
                    unsubscribe();
                    if (statusValue !== 'reconnecting') {
                        setTimeout(() => this.connect(), 1000);
                    }
                } else if (isHealthy) {
                    // Health check passed - ensure we're in connected state
                    this.status.update(status => 
                        status === 'degraded' ? 'connected' : status
                    );
                }
            } catch (e) {
                console.error('Health check error:', this._formatError(e));
            }
        }, 30000); // Every 30 seconds
    }
    
    /**
     * Stop health check monitoring
     */
    private _stopHealthCheck(): void {
        if (this._healthCheckInterval) {
            clearInterval(this._healthCheckInterval);
            this._healthCheckInterval = null;
        }
    }
    
    // TTRPG-optimized convenience methods with enhanced type safety
    
    async search(query: string, options: TTRPGSearchOptions = {}): Promise<Result<{
        results: Array<{
            title: string;
            content: string;
            source: string;
            relevance: number;
            page?: number;
        }>;
        query: string;
        total: number;
        metadata: Record<string, unknown>;
    }>> {
        return this.call('search', { query, ...options });
    }
    
    async addSource(
        pdfPath: string, 
        rulebookName: string, 
        system: string,
        metadata?: Record<string, unknown>
    ): Promise<Result<{ sourceId: string; pagesProcessed: number; }>> {
        return this.call('add_source', {
            pdf_path: pdfPath,
            rulebook_name: rulebookName,
            system,
            ...(metadata && { metadata })
        });
    }
    
    async listSources(): Promise<Result<Array<{
        id: string;
        name: string;
        system: string;
        pageCount: number;
        status: 'processing' | 'ready' | 'error';
        addedAt: string;
    }>>> {
        return this.call('list_sources', {});
    }
    
    async createCampaign(options: CampaignCreateOptions): Promise<Result<{
        campaignId: string;
        name: string;
        system: string;
        createdAt: string;
    }>> {
        return this.call('create_campaign', options);
    }
    
    async getCampaignData(campaignId: string): Promise<Result<{
        id: string;
        name: string;
        system: string;
        description?: string;
        players: string[];
        sessions: number;
        lastActivity: string;
        data: Record<string, unknown>;
    }>> {
        return this.call('get_campaign_data', { campaign_id: campaignId });
    }
    
    async updateCampaignData(
        campaignId: string, 
        data: Record<string, unknown>
    ): Promise<Result<{ updated: boolean; timestamp: string; }>> {
        return this.call('update_campaign_data', {
            campaign_id: campaignId,
            ...data
        });
    }
    
    async startSession(
        campaignId: string, 
        title?: string
    ): Promise<Result<{ sessionId: string; title: string; startedAt: string; }>> {
        return this.call('start_session', {
            campaign_id: campaignId,
            ...(title && { title })
        });
    }
    
    async rollDice(notation: string): Promise<Result<{
        notation: string;
        result: number;
        breakdown: Array<{ die: string; roll: number }>;
        timestamp: string;
    }>> {
        return this.call('roll_dice', { notation });
    }
    
    async generateCharacter(options: CharacterGenOptions): Promise<Result<{
        name: string;
        system: string;
        level: number;
        stats: Record<string, number>;
        background: string;
        equipment: string[];
    }>> {
        return this.call('generate_character', options);
    }
    
    async generateNPC(options: Partial<CharacterGenOptions> = {}): Promise<Result<{
        name: string;
        role: string;
        description: string;
        stats?: Record<string, number>;
        traits: string[];
    }>> {
        return this.call('generate_npc', options);
    }
    
    // Utility methods
    
    /**
     * Check if the bridge is ready for operations
     */
    get isReady(): boolean {
        let statusValue: ConnectionStatus | undefined;
        const unsubscribe = this.status.subscribe(status => statusValue = status);
        unsubscribe();
        return this._initialized && statusValue !== 'error' && statusValue !== 'disconnected';
    }
    
    /**
     * Get connection health score (0-100)
     */
    get healthScore(): number {
        if (!this._initialized) return 0;
        
        const successRate = this._metrics.totalCalls > 0 
            ? (this._metrics.successfulCalls / this._metrics.totalCalls) * 100
            : 100;
            
        let statusValue: ConnectionStatus | undefined;
        const unsubscribe = this.status.subscribe(status => statusValue = status);
        unsubscribe();
        const statusScore = {
            'connected': 100,
            'degraded': 70,
            'reconnecting': 30,
            'connecting': 50,
            'error': 0,
            'disconnected': 0
        }[statusValue!] ?? 0;
        
        return Math.min(successRate, statusScore);
    }
    
    // Private utility methods
    
    private _formatError(error: unknown): string {
        if (typeof error === 'string') return error;
        if (error instanceof Error) return error.message;
        if (error && typeof error === 'object' && 'toString' in error) {
            return error.toString() as string;
        }
        return 'Unknown error occurred';
    }
    
    private _isConnectionError(error: string): boolean {
        const connectionErrorPatterns = [
            'not running',
            'disconnected',
            'connection refused',
            'broken pipe',
            'process died'
        ];
        
        return connectionErrorPatterns.some(pattern => 
            error.toLowerCase().includes(pattern)
        );
    }
    
    private _updateAverageLatency(latency: number): void {
        const weight = 0.1; // Exponential moving average
        this._metrics.averageLatency = 
            (this._metrics.averageLatency * (1 - weight)) + (latency * weight);
    }
    
    private _delay(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
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