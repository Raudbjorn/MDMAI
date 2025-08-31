<script lang="ts">
	import { onMount } from 'svelte';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card';
	import { Upload, FileText, AlertCircle, CheckCircle, X } from 'lucide-svelte';
	import { 
		type UploadResult, 
		type UploadProgress, 
		type RecentUpload 
	} from '$lib/types/upload';
	import { 
		addRecentUpload, 
		getRecentUploads,
		removeRecentUpload 
	} from '$lib/utils/upload';

	// Svelte 5 runes for reactive state
	let dragOver = $state(false);
	let uploading = $state(false);
	let uploadProgress = $state<UploadProgress[]>([]);
	let recentUploads = $state<RecentUpload[]>([]);
	let selectedFiles = $state<FileList | null>(null);

	// Load recent uploads on component mount
	onMount(() => {
		const uploadData = getRecentUploads();
		recentUploads = uploadData.recentUploads;
	});

	/**
	 * Handles successful upload completion
	 * Uses shared addRecentUpload helper to eliminate code duplication
	 */
	function handleUploadSuccess(uploadResult: UploadResult): void {
		// Use actual upload data instead of hardcoded placeholder values
		const displayName = uploadResult.filename;
		const tags = ['upload', uploadResult.type.split('/')[0]]; // e.g., ['upload', 'application'] for PDFs
		
		// Add to recent uploads using shared helper
		addRecentUpload(uploadResult, displayName, tags);
		
		// Update local state to reflect the change
		const uploadData = getRecentUploads();
		recentUploads = uploadData.recentUploads;
		
		// Update upload progress state
		const progressIndex = uploadProgress.findIndex(p => p.file.name === uploadResult.filename);
		if (progressIndex !== -1) {
			uploadProgress[progressIndex] = {
				...uploadProgress[progressIndex],
				status: 'success',
				progress: 100,
				result: uploadResult
			};
		}
		
		console.log('Upload successful:', uploadResult);
	}

	/**
	 * Handles upload errors
	 * Uses shared addRecentUpload helper to eliminate code duplication
	 */
	function handleUploadError(file: File, errorMessage: string): void {
		// Create error upload result with actual file data instead of placeholder values
		const errorUploadResult: UploadResult = {
			id: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
			filename: file.name,
			size: file.size,
			type: file.type,
			uploadedAt: new Date(),
			status: 'error',
			errorMessage
		};
		
		const displayName = file.name;
		const tags = ['upload', 'error', file.type.split('/')[0]];
		
		// Add to recent uploads using shared helper
		addRecentUpload(errorUploadResult, displayName, tags);
		
		// Update local state to reflect the change
		const uploadData = getRecentUploads();
		recentUploads = uploadData.recentUploads;
		
		// Update upload progress state
		const progressIndex = uploadProgress.findIndex(p => p.file.name === file.name);
		if (progressIndex !== -1) {
			uploadProgress[progressIndex] = {
				...uploadProgress[progressIndex],
				status: 'error',
				errorMessage,
				result: errorUploadResult
			};
		}
		
		console.error('Upload failed:', errorMessage, file);
	}

	/**
	 * Simulates file upload process
	 * In a real implementation, this would call the actual upload API
	 */
	async function uploadFile(file: File): Promise<void> {
		const progress: UploadProgress = {
			file,
			progress: 0,
			status: 'uploading'
		};
		
		uploadProgress = [...uploadProgress, progress];
		
		try {
			// Simulate upload progress
			for (let i = 0; i <= 100; i += 10) {
				await new Promise(resolve => setTimeout(resolve, 100));
				const progressIndex = uploadProgress.findIndex(p => p.file === file);
				if (progressIndex !== -1) {
					uploadProgress[progressIndex] = { ...uploadProgress[progressIndex], progress: i };
				}
			}
			
			// Simulate successful upload result with actual file data
			const uploadResult: UploadResult = {
				id: `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
				filename: file.name,
				size: file.size,
				type: file.type,
				uploadedAt: new Date(),
				status: 'success',
				url: `/uploads/${file.name}`, // Use actual file name instead of placeholder
				progress: 100
			};
			
			handleUploadSuccess(uploadResult);
			
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : 'Upload failed';
			handleUploadError(file, errorMessage);
		}
	}

	/**
	 * Handles file selection from input or drag-and-drop
	 */
	async function handleFiles(files: FileList | null): Promise<void> {
		if (!files || files.length === 0) return;
		
		uploading = true;
		
		try {
			const uploadPromises = Array.from(files).map(file => uploadFile(file));
			await Promise.all(uploadPromises);
		} finally {
			uploading = false;
		}
	}

	/**
	 * Handles file input change
	 */
	function handleFileInput(event: Event): void {
		const target = event.target as HTMLInputElement;
		selectedFiles = target.files;
		handleFiles(selectedFiles);
	}

	/**
	 * Handles drag and drop events
	 */
	function handleDrop(event: DragEvent): void {
		event.preventDefault();
		dragOver = false;
		
		const files = event.dataTransfer?.files;
		handleFiles(files);
	}

	function handleDragOver(event: DragEvent): void {
		event.preventDefault();
		dragOver = true;
	}

	function handleDragLeave(): void {
		dragOver = false;
	}

	/**
	 * Removes an upload from recent uploads
	 */
	function removeUpload(uploadId: string): void {
		removeRecentUpload(uploadId);
		const uploadData = getRecentUploads();
		recentUploads = uploadData.recentUploads;
	}

	/**
	 * Formats file size for display
	 */
	function formatFileSize(bytes: number): string {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}

	/**
	 * Formats date for display
	 */
	function formatDate(date: Date): string {
		return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { 
			hour: '2-digit', 
			minute: '2-digit' 
		});
	}
</script>

<svelte:head>
	<title>Upload Files - TTRPG Assistant</title>
	<meta name="description" content="Upload TTRPG rulebooks and documents" />
</svelte:head>

<div class="container mx-auto max-w-4xl py-8 px-4">
	<div class="space-y-6">
		<!-- Page Header -->
		<div class="text-center space-y-2">
			<h1 class="text-3xl font-bold">Upload Files</h1>
			<p class="text-muted-foreground">
				Upload your TTRPG rulebooks, character sheets, and campaign documents
			</p>
		</div>

		<!-- Upload Area -->
		<Card>
			<CardHeader>
				<CardTitle>Select Files</CardTitle>
				<CardDescription>
					Choose files to upload or drag and drop them below
				</CardDescription>
			</CardHeader>
			<CardContent>
				<div
					class="relative border-2 border-dashed rounded-lg p-8 text-center transition-colors {dragOver ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'}"
					ondrop={handleDrop}
					ondragover={handleDragOver}
					ondragleave={handleDragLeave}
					role="button"
					tabindex="0"
				>
					<Upload class="mx-auto h-12 w-12 text-muted-foreground mb-4" />
					<div class="space-y-2">
						<p class="text-lg font-medium">
							{dragOver ? 'Drop files here' : 'Drop files here or click to browse'}
						</p>
						<p class="text-sm text-muted-foreground">
							Supports PDF, DOC, DOCX, TXT files up to 100MB each
						</p>
					</div>
					<input
						type="file"
						multiple
						accept=".pdf,.doc,.docx,.txt"
						class="absolute inset-0 opacity-0 cursor-pointer"
						onchange={handleFileInput}
						disabled={uploading}
					/>
				</div>
			</CardContent>
		</Card>

		<!-- Upload Progress -->
		{#if uploadProgress.length > 0}
			<Card>
				<CardHeader>
					<CardTitle>Upload Progress</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="space-y-3">
						{#each uploadProgress as progress}
							<div class="flex items-center space-x-3">
								<div class="flex-shrink-0">
									{#if progress.status === 'uploading'}
										<div class="animate-spin rounded-full h-5 w-5 border-2 border-primary border-t-transparent"></div>
									{:else if progress.status === 'success'}
										<CheckCircle class="h-5 w-5 text-green-500" />
									{:else if progress.status === 'error'}
										<AlertCircle class="h-5 w-5 text-red-500" />
									{/if}
								</div>
								<div class="flex-1 min-w-0">
									<p class="text-sm font-medium truncate">{progress.file.name}</p>
									<div class="flex items-center space-x-2">
										<div class="flex-1 bg-muted rounded-full h-2">
											<div
												class="bg-primary h-2 rounded-full transition-all {progress.status === 'error' ? 'bg-red-500' : ''}"
												style="width: {progress.progress}%"
											></div>
										</div>
										<span class="text-xs text-muted-foreground">
											{progress.progress}%
										</span>
									</div>
									{#if progress.errorMessage}
										<p class="text-xs text-red-500 mt-1">{progress.errorMessage}</p>
									{/if}
								</div>
							</div>
						{/each}
					</div>
				</CardContent>
			</Card>
		{/if}

		<!-- Recent Uploads -->
		{#if recentUploads.length > 0}
			<Card>
				<CardHeader>
					<CardTitle>Recent Uploads</CardTitle>
					<CardDescription>
						Your recently uploaded files
					</CardDescription>
				</CardHeader>
				<CardContent>
					<div class="space-y-3">
						{#each recentUploads as recent}
							<div class="flex items-start space-x-3 p-3 bg-muted/20 rounded-lg">
								<div class="flex-shrink-0 mt-1">
									{#if recent.upload.status === 'success'}
										<CheckCircle class="h-5 w-5 text-green-500" />
									{:else if recent.upload.status === 'error'}
										<AlertCircle class="h-5 w-5 text-red-500" />
									{:else}
										<FileText class="h-5 w-5 text-blue-500" />
									{/if}
								</div>
								<div class="flex-1 min-w-0">
									<div class="flex items-center justify-between">
										<p class="text-sm font-medium truncate">{recent.displayName}</p>
										<Button
											variant="ghost"
											size="sm"
											onclick={() => removeUpload(recent.upload.id)}
											class="flex-shrink-0 h-6 w-6 p-0"
										>
											<X class="h-4 w-4" />
										</Button>
									</div>
									<div class="flex items-center space-x-4 mt-1">
										<span class="text-xs text-muted-foreground">
											{formatFileSize(recent.upload.size)}
										</span>
										<span class="text-xs text-muted-foreground">
											{formatDate(recent.upload.uploadedAt)}
										</span>
										<span class="text-xs px-2 py-1 rounded-md {
											recent.upload.status === 'success' ? 'bg-green-100 text-green-700' :
											recent.upload.status === 'error' ? 'bg-red-100 text-red-700' :
											'bg-blue-100 text-blue-700'
										}">
											{recent.upload.status}
										</span>
									</div>
									{#if recent.upload.errorMessage}
										<p class="text-xs text-red-500 mt-1">{recent.upload.errorMessage}</p>
									{/if}
									{#if recent.tags.length > 0}
										<div class="flex flex-wrap gap-1 mt-2">
											{#each recent.tags as tag}
												<span class="text-xs px-2 py-1 bg-muted rounded-md">
													{tag}
												</span>
											{/each}
										</div>
									{/if}
								</div>
							</div>
						{/each}
					</div>
				</CardContent>
			</Card>
		{/if}
	</div>
</div>