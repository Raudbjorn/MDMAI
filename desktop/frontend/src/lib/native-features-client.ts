/**
 * Native Features Client
 * 
 * TypeScript client for native desktop features via Tauri
 */

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export interface NativeFeatures {
  system_tray: boolean;
  file_dialogs: boolean;
  notifications: boolean;
  drag_drop: boolean;
  file_associations: boolean;
}

export class NativeFeaturesClient {
  async initializeNativeFeatures(): Promise<NativeFeatures> {
    try {
      return await invoke<NativeFeatures>('initialize_native_features');
    } catch (error) {
      console.error('Failed to initialize native features:', error);
      return {
        system_tray: false,
        file_dialogs: false,
        notifications: false,
        drag_drop: false,
        file_associations: false,
      };
    }
  }

  async showFileDialog(options: {
    title?: string;
    filters?: Array<{ name: string; extensions: string[] }>;
    directory?: boolean;
    multiple?: boolean;
  }): Promise<string[] | null> {
    try {
      return await invoke<string[] | null>('show_file_dialog', { options });
    } catch (error) {
      console.error('Failed to show file dialog:', error);
      return null;
    }
  }

  async showNotification(title: string, message: string): Promise<void> {
    try {
      await invoke('show_notification', { title, message });
    } catch (error) {
      console.error('Failed to show notification:', error);
    }
  }

  async setupDragDrop(): Promise<void> {
    try {
      await invoke('setup_drag_drop');
      
      // Listen for drag and drop events
      await listen('drag-drop-files', (event) => {
        console.log('Files dropped:', event.payload);
        // Dispatch custom event for components to handle
        window.dispatchEvent(new CustomEvent('native-files-dropped', {
          detail: event.payload
        }));
      });
    } catch (error) {
      console.error('Failed to setup drag and drop:', error);
    }
  }

  async setupSystemTray(): Promise<void> {
    try {
      await invoke('setup_system_tray');
    } catch (error) {
      console.error('Failed to setup system tray:', error);
    }
  }
}

// Initialize native features client
export const initializeNativeFeatures = async (): Promise<NativeFeatures> => {
  const client = new NativeFeaturesClient();
  const features = await client.initializeNativeFeatures();
  
  if (features.drag_drop) {
    await client.setupDragDrop();
  }

  if (features.system_tray) {
    await client.setupSystemTray();
  }
  
  return features;
};

export default NativeFeaturesClient;