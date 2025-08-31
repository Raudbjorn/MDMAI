// Disable SSR for desktop application (required for Tauri)
export const ssr = false;

// Disable prerendering for SPA mode
export const prerender = false;

// Enable client-side routing
export const csr = true;

// Load function for app initialization
export async function load() {
    // Check if running in Tauri context
    const isTauri = typeof window !== 'undefined' && window.__TAURI__;
    
    return {
        isTauri,
        platform: isTauri ? 'desktop' : 'web'
    };
}