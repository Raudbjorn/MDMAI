<script lang="ts">
	import { providerStore } from '$lib/stores/providers.svelte';
	import { providerApi } from '$lib/api/providers-client';
	import { ProviderType } from '$lib/types/providers';
	import type { ProviderCredentials } from '$lib/types/providers';
	
	interface Props {
		providerType: ProviderType;
		onSave?: (credentials: ProviderCredentials) => void;
	}

	let { providerType, onSave }: Props = $props();
	
	// Local state
	let apiKey = $state('');
	let showKey = $state(false);
	let isValidating = $state(false);
	let validationError = $state<string | null>(null);
	let validationSuccess = $state(false);
	
	// Provider display names
	const providerNames: Record<ProviderType, string> = {
		[ProviderType.ANTHROPIC]: 'Anthropic (Claude)',
		[ProviderType.OPENAI]: 'OpenAI (GPT)',
		[ProviderType.GOOGLE]: 'Google AI (Gemini)'
	};
	
	// API key patterns for basic validation
	const apiKeyPatterns: Record<ProviderType, RegExp> = {
		[ProviderType.ANTHROPIC]: /^sk-ant-api\d{2}-[\w-]{48,}$/,
		[ProviderType.OPENAI]: /^sk-[a-zA-Z0-9]{48,}$/,
		[ProviderType.GOOGLE]: /^AIza[a-zA-Z0-9-_]{35}$/
	};
	
	// API key documentation URLs
	const docUrls: Record<ProviderType, string> = {
		[ProviderType.ANTHROPIC]: 'https://console.anthropic.com/account/keys',
		[ProviderType.OPENAI]: 'https://platform.openai.com/api-keys',
		[ProviderType.GOOGLE]: 'https://makersuite.google.com/app/apikey'
	};
	
	/**
	 * Basic client-side validation
	 */
	function validateKeyFormat(key: string): boolean {
		const pattern = apiKeyPatterns[providerType];
		if (!pattern) return key.length > 0;
		return pattern.test(key);
	}
	
	/**
	 * Mask API key for display
	 */
	function maskApiKey(key: string): string {
		if (key.length <= 8) return '•'.repeat(key.length);
		return key.substring(0, 4) + '•'.repeat(key.length - 8) + key.substring(key.length - 4);
	}
	
	/**
	 * Validate API key with backend
	 */
	async function validateApiKey() {
		if (!apiKey) {
			validationError = 'Please enter an API key';
			return;
		}
		
		if (!validateKeyFormat(apiKey)) {
			validationError = `Invalid ${providerNames[providerType]} API key format`;
			return;
		}
		
		isValidating = true;
		validationError = null;
		validationSuccess = false;
		
		try {
			// Use providerApi directly to validate the new API key
			const validationResult = await providerApi.validateCredentials(providerType, apiKey);
			
			if (validationResult.ok && validationResult.value.is_valid) {
				validationSuccess = true;
				validationError = null;
			} else {
				validationError = validationResult.ok 
					? (validationResult.value.error_message || 'API key validation failed. Please check your key and try again.')
					: 'API key validation failed. Please check your key and try again.';
			}
		} catch (error) {
			validationError = error instanceof Error ? error.message : 'Validation failed';
		} finally {
			isValidating = false;
		}
	}
	
	/**
	 * Save credentials
	 */
	async function saveCredentials() {
		if (!apiKey || !validationSuccess) {
			validationError = 'Please validate your API key first';
			return;
		}
		
		const credentials: ProviderCredentials = {
			provider_type: providerType,
			api_key: apiKey,
			encrypted: true,
			last_updated: new Date()
		};
		
		try {
			if (onSave) {
				onSave(credentials);
			}
			
			// Clear sensitive data
			apiKey = '';
			showKey = false;
			validationSuccess = false;
		} catch (error) {
			validationError = error instanceof Error ? error.message : 'Failed to save credentials';
		}
	}
	
	/**
	 * Copy API key to clipboard
	 */
	async function copyToClipboard() {
		if (!apiKey) return;
		
		try {
			await navigator.clipboard.writeText(apiKey);
			// Show brief success feedback
			const originalKey = apiKey;
			apiKey = 'Copied!';
			setTimeout(() => {
				apiKey = originalKey;
			}, 1000);
		} catch (error) {
			validationError = 'Failed to copy to clipboard';
		}
	}
	
	/**
	 * Clear form
	 */
	function clearForm() {
		apiKey = '';
		showKey = false;
		validationError = null;
		validationSuccess = false;
	}
</script>

<div class="credential-manager">
	<div class="provider-header">
		<h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
			{providerNames[providerType]}
		</h3>
		<a 
			href={docUrls[providerType]}
			target="_blank"
			rel="noopener noreferrer"
			class="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
		>
			Get API Key →
		</a>
	</div>
	
	<div class="mt-4 space-y-4">
		<!-- API Key Input -->
		<div class="relative">
			<label for="api-key-{providerType}" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
				API Key
			</label>
			<div class="relative">
				<input
					id="api-key-{providerType}"
					type={showKey ? 'text' : 'password'}
					bind:value={apiKey}
					placeholder="Enter your API key"
					class="w-full px-3 py-2 pr-20 border border-gray-300 dark:border-gray-600 rounded-md 
					       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
					       focus:ring-2 focus:ring-blue-500 focus:border-blue-500
					       disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:cursor-not-allowed"
					disabled={isValidating}
					autocomplete="off"
					spellcheck="false"
				/>
				
				<!-- Show/Hide Toggle -->
				<button
					type="button"
					onclick={() => showKey = !showKey}
					class="absolute right-10 top-1/2 -translate-y-1/2 p-1 text-gray-500 hover:text-gray-700 
					       dark:text-gray-400 dark:hover:text-gray-200"
					title={showKey ? 'Hide' : 'Show'}
				>
					{#if showKey}
						<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
								  d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
						</svg>
					{:else}
						<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
								  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
								  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
						</svg>
					{/if}
				</button>
				
				<!-- Copy Button -->
				<button
					type="button"
					onclick={copyToClipboard}
					class="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-500 hover:text-gray-700 
					       dark:text-gray-400 dark:hover:text-gray-200"
					title="Copy"
					disabled={!apiKey}
				>
					<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
							  d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
					</svg>
				</button>
			</div>
		</div>
		
		<!-- Validation Status -->
		{#if validationError}
			<div class="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
				<p class="text-sm text-red-600 dark:text-red-400">{validationError}</p>
			</div>
		{/if}
		
		{#if validationSuccess}
			<div class="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md">
				<p class="text-sm text-green-600 dark:text-green-400">✓ API key validated successfully</p>
			</div>
		{/if}
		
		<!-- Action Buttons -->
		<div class="flex gap-2">
			<button
				type="button"
				onclick={validateApiKey}
				disabled={isValidating || !apiKey}
				class="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 
				       disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors
				       dark:bg-blue-500 dark:hover:bg-blue-600 dark:disabled:bg-gray-600"
			>
				{#if isValidating}
					<span class="flex items-center justify-center">
						<svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
							<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
							<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
						</svg>
						Validating...
					</span>
				{:else}
					Validate
				{/if}
			</button>
			
			<button
				type="button"
				onclick={saveCredentials}
				disabled={!validationSuccess}
				class="flex-1 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 
				       disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors
				       dark:bg-green-500 dark:hover:bg-green-600 dark:disabled:bg-gray-600"
			>
				Save Credentials
			</button>
			
			<button
				type="button"
				onclick={clearForm}
				class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
				       hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors
				       text-gray-700 dark:text-gray-300"
			>
				Clear
			</button>
		</div>
		
		<!-- Security Notice -->
		<div class="text-xs text-gray-500 dark:text-gray-400 mt-2">
			<p>🔒 Your API key will be encrypted and stored securely. Never share your API keys publicly.</p>
		</div>
	</div>
</div>

<style>
	.credential-manager {
		@apply p-4 bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700;
	}
	
	.provider-header {
		@apply flex items-center justify-between pb-2 border-b border-gray-200 dark:border-gray-700;
	}
</style>