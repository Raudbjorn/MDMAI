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
            mode: 'auto',
            directives: {
                'default-src': ['self', 'tauri:'],
                'script-src': ['self', 'unsafe-inline', 'tauri:'],
                'style-src': ['self', 'unsafe-inline'],
                'connect-src': ['self', 'tauri:', 'ipc:', 'https:', 'ws:', 'wss:'],
                'img-src': ['self', 'data:', 'blob:', 'tauri:', 'asset:'],
                'font-src': ['self', 'data:'],
                'media-src': ['self', 'tauri:', 'asset:'],
                'frame-src': ['none'],
                'object-src': ['none'],
                'base-uri': ['self']
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