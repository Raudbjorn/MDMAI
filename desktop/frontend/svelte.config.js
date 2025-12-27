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
        // Note: CSP is less relevant for Tauri desktop apps since content is local
        // Keeping minimal CSP for defense-in-depth
        csp: {
            mode: 'auto',
            directives: {
                'default-src': ['self', 'tauri:', 'asset:'],
                'script-src': ['self', 'tauri:', 'asset:', 'unsafe-inline'],
                'style-src': ['self', 'unsafe-inline'],
                'connect-src': ['self', 'tauri:', 'ipc:', 'asset:', 'http://localhost:*', 'ws://localhost:*'],
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
        },

        // Path aliases (recommended approach instead of tsconfig paths)
        alias: {
            $lib: 'src/lib',
            $components: 'src/lib/components',
            $types: 'src/lib/types',
            $utils: 'src/lib/utils'
        }
    }
};

export default config;