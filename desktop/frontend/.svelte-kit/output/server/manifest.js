export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.BMQy6PvV.js",app:"_app/immutable/entry/app.CoRnixRy.js",imports:["_app/immutable/entry/start.BMQy6PvV.js","_app/immutable/chunks/BvsdbkWI.js","_app/immutable/chunks/T1mY9Bsz.js","_app/immutable/chunks/Bb6-my-h.js","_app/immutable/entry/app.CoRnixRy.js","_app/immutable/chunks/B_oE_bSN.js","_app/immutable/chunks/Bb6-my-h.js","_app/immutable/chunks/T1mY9Bsz.js","_app/immutable/chunks/DBavZ5B7.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
