<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  
  const dispatch = createEventDispatcher<{
    'files-dropped': { files: string[]; type: string }
  }>();
  
  let isDragging = false;
  let dragCounter = 0;
  
  onMount(() => {
    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      dragCounter++;
      isDragging = true;
    };
    
    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        isDragging = false;
        dragCounter = 0;
      }
    };
    
    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
    };
    
    const handleDrop = (e: DragEvent) => {
      e.preventDefault();
      isDragging = false;
      dragCounter = 0;
      
      const files = Array.from(e.dataTransfer?.files || []);
      if (files.length > 0) {
        const filePaths = files.map(file => file.name); // In real implementation, would get actual paths
        dispatch('files-dropped', { files: filePaths, type: 'drag-drop' });
      }
    };
    
    // Add event listeners to window
    window.addEventListener('dragenter', handleDragEnter);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('drop', handleDrop);
    
    return () => {
      window.removeEventListener('dragenter', handleDragEnter);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('drop', handleDrop);
    };
  });
</script>

{#if isDragging}
  <div class="fixed inset-0 z-50 bg-blue-500 bg-opacity-20 flex items-center justify-center">
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 border-2 border-dashed border-blue-500">
      <div class="text-center">
        <div class="text-4xl mb-4">📁</div>
        <div class="text-xl font-semibold text-gray-900 dark:text-white mb-2">
          Drop Files Here
        </div>
        <div class="text-gray-600 dark:text-gray-400">
          Supported formats: PDF, TXT, DOC, DOCX
        </div>
      </div>
    </div>
  </div>
{/if}