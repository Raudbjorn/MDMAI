import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
    plugins: [sveltekit()],
    
    // Optimizations for desktop application
    build: {
        // Single chunk strategy for desktop (reduces HTTP requests)
        rollupOptions: {
            output: {
                manualChunks: undefined, // Single chunk for desktop
                inlineDynamicImports: true // Inline dynamic imports
            }
        },
        
        // Inline small assets to reduce file operations
        assetsInlineLimit: 8192, // 8KB
        
        // Optimize chunk size for desktop
        chunkSizeWarningLimit: 2000, // 2MB is fine for desktop
        
        // Enable minification
        minify: 'terser',
        terserOptions: {
            compress: {
                drop_console: true, // Remove console logs in production
                drop_debugger: true
            }
        },
        
        // Source maps for debugging
        sourcemap: process.env.NODE_ENV === 'development'
    },
    
    // Optimize dependencies
    optimizeDeps: {
        include: [
            '@tauri-apps/api',
            '@tauri-apps/api/tauri',
            '@tauri-apps/api/window',
            '@tauri-apps/api/dialog',
            '@tauri-apps/api/fs',
            '@tauri-apps/api/notification'
        ],
        exclude: ['@tauri-apps/cli']
    },
    
    // Server configuration for development
    server: {
        port: 5173,
        strictPort: true,
        fs: {
            allow: ['..']
        }
    },
    
    // Clear screen in dev mode
    clearScreen: false
});