<script lang="ts">
	import PDFUpload from '$lib/components/upload/PDFUpload.svelte';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';
	import type { Result } from '$lib/types/providers.js';

	// Enhanced type definitions
	interface RecentUpload {
		readonly id: string;
		readonly filename: string;
		readonly model: string;
		readonly timestamp: Date;
		readonly status: 'success' | 'error';
		readonly message?: string;
	}

	interface UploadState {
		count: number;
		current: { file: File; model: string } | null;
		history: readonly RecentUpload[];
	}

	// Enhanced state management with better organization
	let uploadState = $state<UploadState>({
		count: 0,
		current: null,
		history: []
	});

	// Type-safe localStorage operations
	const STORAGE_KEY = 'recent_uploads' as const;
	const MAX_HISTORY_SIZE = 5 as const;

	// Utility functions with better error handling
	const generateUUID = (): string => 
		crypto.randomUUID?.() ?? 
		`${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

	const parseStoredUploads = (data: string): Result<RecentUpload[]> => {
		try {
			const parsed = JSON.parse(data) as unknown;
			
			if (!Array.isArray(parsed)) {
				return { ok: false, error: 'Invalid data format' };
			}

			const uploads = parsed
				.filter((item): item is Record<string, unknown> => 
					item && typeof item === 'object' && item !== null
				)
				.map((item): RecentUpload => ({
					id: typeof item.id === 'string' ? item.id : generateUUID(),
					filename: typeof item.filename === 'string' ? item.filename : 'Unknown File',
					model: typeof item.model === 'string' ? item.model : 'Unknown Model',
					timestamp: new Date(item.timestamp as string),
					status: (item.status === 'success' || item.status === 'error') ? item.status : 'success',
					message: typeof item.message === 'string' ? item.message : undefined
				}))
				.filter(upload => upload.timestamp instanceof Date && !isNaN(upload.timestamp.getTime()));

			return { ok: true, value: uploads };
		} catch (error) {
			return { 
				ok: false, 
				error: error instanceof Error ? error.message : 'Parse error',
				context: { originalData: data }
			};
		}
	};

	const saveUploadsToStorage = (uploads: readonly RecentUpload[]): void => {
		try {
			const serializable = uploads.map(upload => ({
				...upload,
				timestamp: upload.timestamp.toISOString()
			}));
			localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
		} catch (error) {
			console.warn('Failed to save uploads to localStorage:', error);
		}
	};

	// Enhanced mount function with better error handling
	onMount(() => {
		const stored = localStorage.getItem(STORAGE_KEY);
		if (!stored) return;

		const result = parseStoredUploads(stored);
		if (result.ok) {
			uploadState.history = result.value;
		} else {
			console.warn('Failed to load recent uploads:', result.error);
			// Clear corrupted data
			localStorage.removeItem(STORAGE_KEY);
		}
	});

	// More robust upload management
	const addRecentUpload = (status: RecentUpload['status'], message: string): void => {
		if (!uploadState.current) {
			console.warn('No current upload to record');
			return;
		}

		const newUpload: RecentUpload = {
			id: generateUUID(),
			filename: uploadState.current.file.name,
			model: uploadState.current.model,
			timestamp: new Date(),
			status,
			message
		};

		const updatedHistory = [newUpload, ...uploadState.history].slice(0, MAX_HISTORY_SIZE);
		uploadState = {
			...uploadState,
			history: updatedHistory,
			current: null
		};
		
		saveUploadsToStorage(updatedHistory);
	};

	// Type-safe event handlers
	const handleUploadStart = (event: CustomEvent<{ file: File; model: string }>): void => {
		uploadState.current = event.detail;
	};

	const handleUploadSuccess = (event: CustomEvent<{ message: string }>): void => {
		uploadState.count++;
		addRecentUpload('success', event.detail.message);
	};

	const handleUploadError = (event: CustomEvent<{ error: string }>): void => {
		addRecentUpload('error', event.detail.error);
	};

	// Enhanced timestamp formatting with better UX
	const formatTimestamp = (date: Date): string => {
		const now = new Date();
		const diff = now.getTime() - date.getTime();
		const minutes = Math.floor(diff / 60000);
		const hours = Math.floor(diff / 3600000);
		const days = Math.floor(diff / 86400000);

		if (minutes < 1) return 'Just now';
		if (minutes === 1) return '1 minute ago';
		if (minutes < 60) return `${minutes} minutes ago`;
		if (hours === 1) return '1 hour ago';
		if (hours < 24) return `${hours} hours ago`;
		if (days === 1) return 'Yesterday';
		if (days < 7) return `${days} days ago`;
		
		// Use relative date for older items
		return new Intl.RelativeTimeFormat('en', { numeric: 'auto' })
			.format(-Math.floor(diff / 86400000), 'day');
	};

	const clearRecentUploads = (): void => {
		uploadState.history = [];
		localStorage.removeItem(STORAGE_KEY);
	};

	// Enhanced derived state with type safety
	const uploadStats = $derived(() => {
		const successful = uploadState.history.filter(u => u.status === 'success').length;
		const failed = uploadState.history.filter(u => u.status === 'error').length;
		
		return {
			total: uploadState.count,
			successful,
			failed,
			successRate: uploadState.history.length > 0 
				? Math.round((successful / uploadState.history.length) * 100)
				: 0
		} as const;
	});

	// Keyboard navigation support
	const handleKeydown = (event: KeyboardEvent): void => {
		if (event.ctrlKey || event.metaKey) {
			switch (event.key) {
				case 'Escape':
					event.preventDefault();
					goto('/dashboard');
					break;
			}
		}
	};
</script>

<svelte:head>
	<title>Upload PDF - TTRPG Assistant</title>
	<meta name="description" content="Upload and process PDF documents for your TTRPG campaigns" />
</svelte:head>

<svelte:window on:keydown={handleKeydown} />

<div class="upload-page" role="main">
	<div class="page-header">
		<div class="header-content">
			<h1 class="page-title">Upload PDF Document</h1>
			<p class="page-description">
				Process your TTRPG rulebooks, adventures, and campaign materials for intelligent search and analysis
			</p>
		</div>
		<div class="header-actions">
			<button
				onclick={() => goto('/dashboard')}
				class="back-button"
				type="button"
				aria-label="Return to dashboard (Escape key)"
			>
				<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/>
				</svg>
				Back to Dashboard
				<span class="sr-only">(Esc)</span>
			</button>
		</div>
	</div>

	<div class="content-grid">
		<!-- Main Upload Section -->
		<div class="upload-section">
			<section class="upload-card" aria-label="Document upload">
				<h2 class="section-title">Select Document</h2>
				<PDFUpload
					maxFileSize={100}
					on:upload={handleUploadStart}
					on:success={handleUploadSuccess}
					on:error={handleUploadError}
				/>
			</section>

			<!-- Enhanced Tips Section -->
			<aside class="tips-card" aria-label="Upload tips">
				<h3 class="tips-title">
					<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
						<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
					</svg>
					Tips for Best Results
				</h3>
				<ul class="tips-list" role="list">
					<li>📄 Ensure PDFs contain searchable text (not just scanned images)</li>
					<li>⏱️ Larger documents may take longer to process</li>
					<li>🔍 Embedding models are optimized for semantic search</li>
					<li>⚡ Use Ollama models for quality, Sentence Transformers for speed</li>
					<li>📊 Monitor processing statistics in the sidebar</li>
				</ul>
			</aside>
		</div>

		<!-- Sidebar -->
		<aside class="sidebar">
			<!-- Enhanced Statistics -->
			<section class="stats-card" aria-label="Upload statistics">
				<h3 class="card-title">Processing Statistics</h3>
				<div class="stats-grid">
					<div class="stat">
						<span class="stat-value" aria-label="{uploadStats().total} documents processed today">{uploadStats().total}</span>
						<span class="stat-label">Documents Today</span>
					</div>
					<div class="stat">
						<span class="stat-value" aria-label="{uploadStats().successful} successful uploads">{uploadStats().successful}</span>
						<span class="stat-label">Successful</span>
					</div>
					<div class="stat">
						<span class="stat-value" aria-label="{uploadStats().successRate}% success rate">{uploadStats().successRate}%</span>
						<span class="stat-label">Success Rate</span>
					</div>
					{#if uploadStats().failed > 0}
						<div class="stat">
							<span class="stat-value text-red-600" aria-label="{uploadStats().failed} failed uploads">{uploadStats().failed}</span>
							<span class="stat-label">Failed</span>
						</div>
					{/if}
				</div>
			</section>

			<!-- Enhanced Recent Uploads -->
			{#if uploadState.history.length > 0}
				<section class="recent-card" aria-label="Recent uploads history">
					<div class="card-header">
						<h3 class="card-title">Recent Uploads</h3>
						<button
							onclick={clearRecentUploads}
							type="button"
							class="clear-button"
							aria-label="Clear upload history"
						>
							Clear
						</button>
					</div>
					<ul class="recent-list" role="list">
						{#each uploadState.history as upload, index}
							<li class="recent-item" role="listitem">
								<div 
									class="item-icon {upload.status}" 
									aria-label="{upload.status === 'success' ? 'Success' : 'Failed'}"
								>
									{#if upload.status === 'success'}
										<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
											<path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
										</svg>
									{:else}
										<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
											<path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
										</svg>
									{/if}
								</div>
								<div class="item-content">
									<h4 class="item-name" title={upload.filename}>{upload.filename}</h4>
									<p class="item-meta">
										<span class="model-tag">{upload.model}</span>
										<span aria-hidden="true"> • </span>
										<time datetime={upload.timestamp.toISOString()}>
											{formatTimestamp(upload.timestamp)}
										</time>
									</p>
									{#if upload.message}
										<p class="item-message {upload.status}" role="status">
											{upload.message}
										</p>
									{/if}
								</div>
							</li>
						{/each}
					</ul>
				</section>
			{:else}
				<div class="empty-state">
					<div class="empty-icon">📄</div>
					<h3 class="empty-title">No uploads yet</h3>
					<p class="empty-text">Your upload history will appear here after you process your first document.</p>
				</div>
			{/if}

			<!-- Enhanced Help Section -->
			<aside class="help-card" aria-label="Help and documentation">
				<h3 class="card-title">Need Help?</h3>
				<p class="help-text">
					Check our documentation for detailed guides on PDF processing, model selection, and troubleshooting.
				</p>
				<div class="help-links">
					<a href="/docs" class="help-link" aria-label="View documentation">
						📚 Documentation
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
						</svg>
					</a>
					<a href="/troubleshooting" class="help-link" aria-label="View troubleshooting guide">
						🔧 Troubleshooting
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
						</svg>
					</a>
				</div>
			</aside>
		</aside>
	</div>
</div>

<style>
	.upload-page {
		@apply min-h-screen bg-gray-50 dark:bg-gray-900;
	}

	.page-header {
		@apply bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700;
		@apply px-4 sm:px-6 lg:px-8 py-6;
	}

	.header-content {
		@apply max-w-7xl mx-auto;
	}

	.page-title {
		@apply text-2xl font-bold text-gray-900 dark:text-gray-100;
	}

	.page-description {
		@apply mt-2 text-sm text-gray-600 dark:text-gray-400;
	}

	.header-actions {
		@apply max-w-7xl mx-auto mt-4;
	}

	.back-button {
		@apply inline-flex items-center gap-2 px-4 py-2;
		@apply text-sm font-medium text-gray-700 dark:text-gray-300;
		@apply bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600;
		@apply rounded-md hover:bg-gray-50 dark:hover:bg-gray-600;
		@apply transition-colors;
	}

	.content-grid {
		@apply max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8;
		@apply grid grid-cols-1 lg:grid-cols-3 gap-8;
	}

	.upload-section {
		@apply lg:col-span-2 space-y-6;
	}

	.upload-card {
		@apply bg-white dark:bg-gray-800 rounded-lg shadow p-6;
	}

	.section-title {
		@apply text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4;
	}

	.tips-card {
		@apply bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4;
	}

	.tips-title {
		@apply flex items-center gap-2 text-sm font-medium text-blue-900 dark:text-blue-300 mb-3;
	}

	.tips-list {
		@apply space-y-2 text-sm text-blue-800 dark:text-blue-400;
	}

	.tips-list li {
		@apply flex items-start;
	}

	.tips-list li::before {
		content: "•";
		@apply mr-2 font-bold;
	}

	.sidebar {
		@apply space-y-6;
	}

	.stats-card, .recent-card, .help-card {
		@apply bg-white dark:bg-gray-800 rounded-lg shadow p-4;
	}

	.card-title {
		@apply text-sm font-medium text-gray-900 dark:text-gray-100 mb-3;
	}

	.card-header {
		@apply flex items-center justify-between mb-3;
	}

	.clear-button {
		@apply text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200;
	}

	.stats-grid {
		@apply grid grid-cols-2 lg:grid-cols-3 gap-4;
	}

	.stat {
		@apply text-center;
	}

	.stat-value {
		@apply block text-2xl font-bold text-gray-900 dark:text-gray-100;
	}

	.stat-label {
		@apply block text-xs text-gray-500 dark:text-gray-400 mt-1;
	}

	.recent-list {
		@apply space-y-3;
	}

	.recent-item {
		@apply flex gap-3;
	}

	.item-icon {
		@apply w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0;
	}

	.item-icon.success {
		@apply bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400;
	}

	.item-icon.error {
		@apply bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400;
	}

	.item-content {
		@apply flex-1 min-w-0;
	}

	.item-name {
		@apply text-sm font-medium text-gray-900 dark:text-gray-100 truncate;
	}

	.item-meta {
		@apply text-xs text-gray-500 dark:text-gray-400;
	}

	.item-message {
		@apply text-xs mt-1;
	}

	.item-message.success {
		@apply text-green-600 dark:text-green-400;
	}

	.item-message.error {
		@apply text-red-600 dark:text-red-400;
	}

	.help-text {
		@apply text-sm text-gray-600 dark:text-gray-400 mb-3;
	}

	.help-links {
		@apply space-y-2;
	}

	.help-link {
		@apply flex items-center justify-between w-full p-2 text-sm font-medium text-blue-600 dark:text-blue-400;
		@apply bg-blue-50 dark:bg-blue-900/20 rounded-md hover:bg-blue-100 dark:hover:bg-blue-900/30;
		@apply transition-colors;
	}

	.model-tag {
		@apply inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium;
		@apply bg-primary/10 text-primary;
	}

	.empty-state {
		@apply text-center py-8 px-4 bg-white dark:bg-gray-800 rounded-lg shadow;
	}

	.empty-icon {
		@apply text-4xl mb-3;
	}

	.empty-title {
		@apply text-lg font-medium text-gray-900 dark:text-gray-100 mb-2;
	}

	.empty-text {
		@apply text-sm text-muted-foreground;
	}

	.sr-only {
		@apply absolute w-px h-px p-0 -m-px overflow-hidden whitespace-nowrap;
		clip: rect(0, 0, 0, 0);
		border: 0;
	}
</style>