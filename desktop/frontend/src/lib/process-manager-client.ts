/**
 * Process Manager Client
 * 
 * TypeScript client for interfacing with the Rust process manager
 */

import { invoke } from '@tauri-apps/api/core';
import { writable, type Writable } from 'svelte/store';

export interface ProcessState {
  id: string;
  status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping';
  pid?: number;
  uptime?: number;
  memory_usage?: number;
  cpu_usage?: number;
  last_health_check?: string;
  restart_count?: number;
}

export interface ProcessStats {
  total_processes: number;
  running_processes: number;
  failed_processes: number;
  total_restarts: number;
  average_uptime: number;
  memory_usage_mb: number;
  cpu_usage_percent: number;
}

// Store for process statistics
export const processStats: Writable<ProcessStats | null> = writable(null);

// Store for individual process states
export const processStates: Writable<Record<string, ProcessState>> = writable({});

export class ProcessManagerClient {
  async getProcessStats(): Promise<ProcessStats> {
    try {
      const stats = await invoke<ProcessStats>('get_process_stats');
      processStats.set(stats);
      return stats;
    } catch (error) {
      console.error('Failed to get process stats:', error);
      throw error;
    }
  }

  async getProcessState(processId: string): Promise<ProcessState> {
    try {
      return await invoke<ProcessState>('get_process_state', { processId });
    } catch (error) {
      console.error(`Failed to get process state for ${processId}:`, error);
      throw error;
    }
  }

  async startProcess(processId: string): Promise<void> {
    try {
      await invoke('start_process', { processId });
    } catch (error) {
      console.error(`Failed to start process ${processId}:`, error);
      throw error;
    }
  }

  async stopProcess(processId: string): Promise<void> {
    try {
      await invoke('stop_process', { processId });
    } catch (error) {
      console.error(`Failed to stop process ${processId}:`, error);
      throw error;
    }
  }

  async restartProcess(processId: string): Promise<void> {
    try {
      await invoke('restart_process', { processId });
    } catch (error) {
      console.error(`Failed to restart process ${processId}:`, error);
      throw error;
    }
  }
}

export default ProcessManagerClient;