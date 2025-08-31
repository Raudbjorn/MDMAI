
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * Environment variables [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env`. Like [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), this module cannot be imported into client-side code. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * _Unlike_ [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), the values exported from this module are statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * ```ts
 * import { API_KEY } from '$env/static/private';
 * ```
 * 
 * Note that all environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * 
 * ```
 * MY_FEATURE_FLAG=""
 * ```
 * 
 * You can override `.env` values from the command line like so:
 * 
 * ```sh
 * MY_FEATURE_FLAG="enabled" npm run dev
 * ```
 */
declare module '$env/static/private' {
	export const SHELL: string;
	export const npm_command: string;
	export const COREPACK_ENABLE_AUTO_PIN: string;
	export const npm_config_userconfig: string;
	export const CSF_MDTVTexturesDirectory: string;
	export const GDK_DPI_SCALE: string;
	export const npm_config_cache: string;
	export const WSL2_GUI_APPS_ENABLED: string;
	export const WSL_DISTRO_NAME: string;
	export const CSF_DrawPluginDefaults: string;
	export const WT_SESSION: string;
	export const NODE: string;
	export const CSF_LANGUAGE: string;
	export const CSF_MIGRATION_TYPES: string;
	export const ANTHROPIC_API_KEY: string;
	export const GRADLE_HOME: string;
	export const MESA_LOADER_DRIVER_OVERRIDE: string;
	export const GEMINI_API_KEY: string;
	export const COLOR: string;
	export const npm_config_local_prefix: string;
	export const GALLIUM_DRIVER: string;
	export const CSF_OCCTResourcePath: string;
	export const NO_AT_BRIDGE: string;
	export const npm_config_globalconfig: string;
	export const CSF_STEPDefaults: string;
	export const EDITOR: string;
	export const ANDROID_NDK: string;
	export const NAME: string;
	export const PWD: string;
	export const LOGNAME: string;
	export const DRAWHOME: string;
	export const npm_config_init_module: string;
	export const VIPSHOME: string;
	export const _: string;
	export const PAT: string;
	export const CSF_StandardLiteDefaults: string;
	export const RAWWAVE_PATH: string;
	export const CLAUDECODE: string;
	export const HOME: string;
	export const LANG: string;
	export const WSL_INTEROP: string;
	export const npm_package_version: string;
	export const WAYLAND_DISPLAY: string;
	export const GITHUB_PERSONAL_ACCESS_TOKEN: string;
	export const INIT_CWD: string;
	export const ANDROID_NDK_HOME: string;
	export const CSF_ShadersDirectory: string;
	export const CSF_EXCEPTION_PROMPT: string;
	export const CSF_XmlOcafResource: string;
	export const npm_lifecycle_script: string;
	export const NVM_DIR: string;
	export const CSF_SHMessage: string;
	export const npm_config_npm_version: string;
	export const OPCODE6DIR: string;
	export const ANDROID_HOME: string;
	export const TERM: string;
	export const npm_package_name: string;
	export const npm_config_prefix: string;
	export const LESSOPEN: string;
	export const USER: string;
	export const CSF_StandardDefaults: string;
	export const CSF_IGESDefaults: string;
	export const CSSTRNGS: string;
	export const DISPLAY: string;
	export const CSF_XCAFDefaults: string;
	export const npm_lifecycle_event: string;
	export const SHLVL: string;
	export const NVM_CD_FLAGS: string;
	export const BW_SESSION: string;
	export const GIT_EDITOR: string;
	export const ANDROID_SDK_ROOT: string;
	export const CSF_PluginDefaults: string;
	export const CSF_TObjMessage: string;
	export const LIBGL_ALWAYS_SOFTWARE: string;
	export const npm_config_user_agent: string;
	export const CASROOT: string;
	export const OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE: string;
	export const npm_execpath: string;
	export const XDG_RUNTIME_DIR: string;
	export const CLAUDE_CODE_ENTRYPOINT: string;
	export const MKLROOT: string;
	export const DEBUGINFOD_URLS: string;
	export const npm_package_json: string;
	export const WSLENV: string;
	export const BUN_INSTALL: string;
	export const ANDROID_NDK_ROOT: string;
	export const QT_AUTO_SCREEN_SCALE_FACTOR: string;
	export const CSF_XSMessage: string;
	export const MMGT_CLEAR: string;
	export const XDG_DATA_DIRS: string;
	export const npm_config_noproxy: string;
	export const PATH: string;
	export const CSF_TObjDefaults: string;
	export const GDK_SCALE: string;
	export const npm_config_node_gyp: string;
	export const DBUS_SESSION_BUS_ADDRESS: string;
	export const npm_config_global_prefix: string;
	export const HG: string;
	export const HOSTTYPE: string;
	export const QT_SCALE_FACTOR: string;
	export const DRAWDEFAULT: string;
	export const PULSE_SERVER: string;
	export const WT_PROFILE_ID: string;
	export const npm_node_execpath: string;
	export const OLDPWD: string;
	export const npm_package_engines_node: string;
	export const NODE_ENV: string;
}

/**
 * Similar to [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private), except that it only includes environment variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Values are replaced statically at build time.
 * 
 * ```ts
 * import { PUBLIC_BASE_URL } from '$env/static/public';
 * ```
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to runtime environment variables, as defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * This module cannot be imported into client-side code.
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * console.log(env.DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` always includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 */
declare module '$env/dynamic/private' {
	export const env: {
		SHELL: string;
		npm_command: string;
		COREPACK_ENABLE_AUTO_PIN: string;
		npm_config_userconfig: string;
		CSF_MDTVTexturesDirectory: string;
		GDK_DPI_SCALE: string;
		npm_config_cache: string;
		WSL2_GUI_APPS_ENABLED: string;
		WSL_DISTRO_NAME: string;
		CSF_DrawPluginDefaults: string;
		WT_SESSION: string;
		NODE: string;
		CSF_LANGUAGE: string;
		CSF_MIGRATION_TYPES: string;
		ANTHROPIC_API_KEY: string;
		GRADLE_HOME: string;
		MESA_LOADER_DRIVER_OVERRIDE: string;
		GEMINI_API_KEY: string;
		COLOR: string;
		npm_config_local_prefix: string;
		GALLIUM_DRIVER: string;
		CSF_OCCTResourcePath: string;
		NO_AT_BRIDGE: string;
		npm_config_globalconfig: string;
		CSF_STEPDefaults: string;
		EDITOR: string;
		ANDROID_NDK: string;
		NAME: string;
		PWD: string;
		LOGNAME: string;
		DRAWHOME: string;
		npm_config_init_module: string;
		VIPSHOME: string;
		_: string;
		PAT: string;
		CSF_StandardLiteDefaults: string;
		RAWWAVE_PATH: string;
		CLAUDECODE: string;
		HOME: string;
		LANG: string;
		WSL_INTEROP: string;
		npm_package_version: string;
		WAYLAND_DISPLAY: string;
		GITHUB_PERSONAL_ACCESS_TOKEN: string;
		INIT_CWD: string;
		ANDROID_NDK_HOME: string;
		CSF_ShadersDirectory: string;
		CSF_EXCEPTION_PROMPT: string;
		CSF_XmlOcafResource: string;
		npm_lifecycle_script: string;
		NVM_DIR: string;
		CSF_SHMessage: string;
		npm_config_npm_version: string;
		OPCODE6DIR: string;
		ANDROID_HOME: string;
		TERM: string;
		npm_package_name: string;
		npm_config_prefix: string;
		LESSOPEN: string;
		USER: string;
		CSF_StandardDefaults: string;
		CSF_IGESDefaults: string;
		CSSTRNGS: string;
		DISPLAY: string;
		CSF_XCAFDefaults: string;
		npm_lifecycle_event: string;
		SHLVL: string;
		NVM_CD_FLAGS: string;
		BW_SESSION: string;
		GIT_EDITOR: string;
		ANDROID_SDK_ROOT: string;
		CSF_PluginDefaults: string;
		CSF_TObjMessage: string;
		LIBGL_ALWAYS_SOFTWARE: string;
		npm_config_user_agent: string;
		CASROOT: string;
		OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE: string;
		npm_execpath: string;
		XDG_RUNTIME_DIR: string;
		CLAUDE_CODE_ENTRYPOINT: string;
		MKLROOT: string;
		DEBUGINFOD_URLS: string;
		npm_package_json: string;
		WSLENV: string;
		BUN_INSTALL: string;
		ANDROID_NDK_ROOT: string;
		QT_AUTO_SCREEN_SCALE_FACTOR: string;
		CSF_XSMessage: string;
		MMGT_CLEAR: string;
		XDG_DATA_DIRS: string;
		npm_config_noproxy: string;
		PATH: string;
		CSF_TObjDefaults: string;
		GDK_SCALE: string;
		npm_config_node_gyp: string;
		DBUS_SESSION_BUS_ADDRESS: string;
		npm_config_global_prefix: string;
		HG: string;
		HOSTTYPE: string;
		QT_SCALE_FACTOR: string;
		DRAWDEFAULT: string;
		PULSE_SERVER: string;
		WT_PROFILE_ID: string;
		npm_node_execpath: string;
		OLDPWD: string;
		npm_package_engines_node: string;
		NODE_ENV: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * Similar to [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), but only includes variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Note that public dynamic environment variables must all be sent from the server to the client, causing larger network requests — when possible, use `$env/static/public` instead.
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.PUBLIC_DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
