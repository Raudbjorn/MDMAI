<script lang="ts">
	import PDFUpload from '$lib/components/upload/PDFUpload.svelte';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';

	// Local state
	let uploadCount = $state(0);
	let currentUpload = $state<{file: File; model: string} | null>(null);
	let recentUploads = $state<Array<RecentUpload>>([]);
	
	type RecentUpload = {
		id: string;
		filename: string;
		model: string;
		timestamp: Date;
		status: 'success' | 'error';
		message?: string;
	};

	onMount(() => {
		const saved = localStorage.getItem('recent_uploads');
		if (saved) {
			try {
				const parsed = JSON.parse(saved);
				recentUploads = parsed.map((u: any) => ({
					...u,
					timestamp: new Date(u.timestamp)
				}));
			} catch (e) {
				console.error('Failed to load recent uploads:', e);
			}
		}
	});

	const generateUUID = () => crypto.randomUUID?.() || 
		'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
			const r = Math.random() * 16 | 0;
			return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
		});

	const addRecentUpload = (status: 'success' | 'error', message: string) => {
		const upload: RecentUpload = {
			id: generateUUID(),
			filename: currentUpload?.file.name || 'document.pdf',
			model: currentUpload?.model || 'Unknown Model',
			timestamp: new Date(),
			status,
			message
		};

		recentUploads = [upload, ...recentUploads].slice(0, 5);
		localStorage.setItem('recent_uploads', JSON.stringify(recentUploads));
		currentUpload = null;
	};

	const handleUploadStart = (event: CustomEvent<{ file: File; model: string }>) => {
		currentUpload = event.detail;
	};

	const handleUploadSuccess = (event: CustomEvent<{ message: string }>) => {
		uploadCount++;
		addRecentUpload('success', event.detail.message);
	};

	const handleUploadError = (event: CustomEvent<{ error: string }>) => {
		addRecentUpload('error', event.detail.error);
	};

	const formatTimestamp = (date: Date): string => {
		const diff = Date.now() - date.getTime();
		const minutes = Math.floor(diff / 60000);
		const hours = Math.floor(diff / 3600000);
		const days = Math.floor(diff / 86400000);

		if (minutes < 1) return 'Just now';
		if (minutes < 60) return `${minutes}m ago`;
		if (hours < 24) return `${hours}h ago`;
		if (days < 7) return `${days}d ago`;
		return date.toLocaleDateString();
	};

	const clearRecentUploads = () => {
		recentUploads = [];
		localStorage.removeItem('recent_uploads');
	};

	// Derived state for statistics
	const successfulUploads = $derived(recentUploads.filter(u => u.status === 'success').length);
</script>

<svelte:head>
	<title>Upload PDF - TTRPG Assistant</title>
	<meta name="description" content="Upload and process PDF documents for your TTRPG campaigns" />
</svelte:head>

<div class="upload-page">
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
			>
				<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/>
				</svg>
				Back to Dashboard
			</button>
		</div>
	</div>

	<div class="content-grid">
		<!-- Main Upload Section -->
		<div class="upload-section">
			<div class="upload-card">
				<h2 class="section-title">Select Document</h2>
				<PDFUpload
					maxFileSize={100}
					onupload={handleUploadStart}
					onsuccess={handleUploadSuccess}
					onerror={handleUploadError}
				/>
			</div>

			<!-- Tips Section -->
			<div class="tips-card">
				<h3 class="tips-title">
					<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
						<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
					</svg>
					Tips for Best Results
				</h3>
				<ul class="tips-list">
					<li>Ensure PDFs contain searchable text (not just scanned images)</li>
					<li>Larger documents may take longer to process</li>
					<li>Embedding models are optimized for semantic search</li>
					<li>Use Ollama models for better quality, Sentence Transformers for speed</li>
				</ul>
			</div>
		</div>

		<!-- Sidebar -->
		<aside class="sidebar">
			<!-- Statistics -->
			<div class="stats-card">
				<h3 class="card-title">Processing Statistics</h3>
				<div class="stats-grid">
					<div class="stat">
						<span class="stat-value">{uploadCount}</span>
						<span class="stat-label">Documents Today</span>
					</div>
					<div class="stat">
						<span class="stat-value">{successfulUploads}</span>
						<span class="stat-label">Successful</span>
					</div>
				</div>
			</div>

			<!-- Recent Uploads -->
			{#if recentUploads.length > 0}
				<div class="recent-card">
					<div class="card-header">
						<h3 class="card-title">Recent Uploads</h3>
						<button
							onclick={clearRecentUploads}
							class="clear-button"
							aria-label="Clear history"
						>
							Clear
						</button>
					</div>
					<ul class="recent-list">
						{#each recentUploads as upload}
							<li class="recent-item">
								<div class="item-icon {upload.status}">
									{#if upload.status === 'success'}
										<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
											<path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
										</svg>
									{:else}
										<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
											<path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
										</svg>
									{/if}
								</div>
								<div class="item-content">
									<p class="item-name">{upload.filename}</p>
									<p class="item-meta">
										{upload.model} • {formatTimestamp(upload.timestamp)}
									</p>
									{#if upload.message}
										<p class="item-message {upload.status}">
											{upload.message}
										</p>
									{/if}
								</div>
							</li>
						{/each}
					</ul>
				</div>
			{/if}

			<!-- Help Section -->
			<div class="help-card">
				<h3 class="card-title">Need Help?</h3>
				<p class="help-text">
					Check our documentation for detailed guides on PDF processing and model selection.
				</p>
				<a href="/docs" class="help-link">
					View Documentation
					<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
					</svg>
				</a>
			</div>
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
		@apply grid grid-cols-2 gap-4;
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

	.help-link {
		@apply inline-flex items-center gap-1 text-sm font-medium text-blue-600 dark:text-blue-400;
		@apply hover:text-blue-700 dark:hover:text-blue-300;
	}
</style>