<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  const dispatch = createEventDispatcher<{
    'files-selected': { files: string[]; type: string };
    'file-saved': { path: string };
    'directory-selected': { path: string };
  }>();
  
  export let showOpenButton = true;
  export let showSaveButton = true;
  export let showDirectoryButton = true;
  
  async function openFileDialog() {
    try {
      const files = await invoke<string[] | null>('show_open_dialog', {
        filters: [
          { name: 'PDF Documents', extensions: ['pdf'] },
          { name: 'Text Files', extensions: ['txt', 'md'] },
          { name: 'All Files', extensions: ['*'] }
        ],
        multiple: true
      });
      
      if (files && files.length > 0) {
        dispatch('files-selected', { files, type: 'file-dialog' });
      }
    } catch (error) {
      console.error('Failed to open file dialog:', error);
    }
  }
  
  async function saveFileDialog() {
    try {
      const path = await invoke<string | null>('show_save_dialog', {
        filters: [
          { name: 'JSON Files', extensions: ['json'] },
          { name: 'Text Files', extensions: ['txt'] },
          { name: 'All Files', extensions: ['*'] }
        ],
        defaultPath: 'export.json'
      });
      
      if (path) {
        dispatch('file-saved', { path });
      }
    } catch (error) {
      console.error('Failed to open save dialog:', error);
    }
  }
  
  async function selectDirectoryDialog() {
    try {
      const path = await invoke<string | null>('show_directory_dialog');
      
      if (path) {
        dispatch('directory-selected', { path });
      }
    } catch (error) {
      console.error('Failed to open directory dialog:', error);
    }
  }
  
  // Additional methods for compatibility
  export async function importRulebooks() {
    await openFileDialog();
  }
  
  export async function openCampaign() {
    await selectDirectoryDialog();
  }
</script>

<div class="flex gap-2">
  {#if showOpenButton}
    <button
      on:click={openFileDialog}
      class="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
    >
      Open Files
    </button>
  {/if}
  
  {#if showSaveButton}
    <button
      on:click={saveFileDialog}
      class="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors"
    >
      Save As...
    </button>
  {/if}
  
  {#if showDirectoryButton}
    <button
      on:click={selectDirectoryDialog}
      class="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
    >
      Select Directory
    </button>
  {/if}
</div>