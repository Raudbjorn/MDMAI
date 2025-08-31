

export const index = 0;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/fallbacks/layout.svelte.js')).default;
export const universal = {
  "ssr": false,
  "prerender": false,
  "csr": true
};
export const universal_id = "src/routes/+layout.ts";
export const imports = ["_app/immutable/nodes/0.DSCe5O4w.js","_app/immutable/chunks/DBavZ5B7.js","_app/immutable/chunks/Bb6-my-h.js"];
export const stylesheets = [];
export const fonts = [];
