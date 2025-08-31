import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    // Preprocessors
    preprocess: vitePreprocess(),

    kit: {
        // Static adapter for desktop application
        adapter: adapter({
            // Required for SPA mode
            fallback: 'index.html',
            pages: 'build',
            assets: 'build',
            precompress: false
        }),

        // CSP configuration for desktop security
        csp: {
            mode: 'hash',  // Use hash-based CSP for inline scripts
            directives: {
                'default-src': ['self', 'tauri:'],
                'script-src': ['self', 'tauri:'],  // Removed unsafe-inline
                'style-src': ['self'],  // Removed unsafe-inline, hashes will be added automatically
                'connect-src': ['self', 'tauri:', 'ipc:', 'asset:'],  // Removed broad https/ws/wss
                'img-src': ['self', 'data:', 'blob:', 'tauri:', 'asset:'],
                'font-src': ['self', 'data:'],
                'media-src': ['self', 'tauri:', 'asset:'],
                'frame-src': ['none'],
                'object-src': ['none'],
                'base-uri': ['self'],
                'form-action': ['none'],  // Prevent form submissions
                'upgrade-insecure-requests': true  // Force HTTPS where possible
            }
        },

        // Prerender settings
        prerender: {
            entries: [] // No prerendering for desktop SPA
        },

        // Version for cache busting
        version: {
            name: process.env.npm_package_version || '1.0.0'
        },

        // Service worker disabled for desktop
        serviceWorker: {
            register: false
        }
    }
};

export default config;