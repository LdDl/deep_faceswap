<script>
	import { request } from '$lib/api.js';
	import { FileEntry } from '$lib/file_entry.js';

	let { open, startPath = '/', filterExtensions, onSelect, onClose, saveMode = false, defaultFilename = '' } = $props();

	let currentPath = $state('/');
	let entries = $state([]);
	let loading = $state(false);
	let error = $state('');
	let history = $state([]);
	let historyIndex = $state(-1);
	let editingPath = $state(false);
	let pathInputValue = $state('');
	let cancelledEdit = false;
	let filename = $state('');

	/** @type {HTMLDivElement|undefined} */
	let dialogRef = $state();

	// Initialize filename when opening in save mode
	$effect(() => {
		if (open && saveMode && defaultFilename && !filename) {
			filename = defaultFilename;
		}
	});

	// Focus the dialog when it opens
	$effect(() => {
		if (open && dialogRef) {
			dialogRef.focus();
		}
	});

	// Check if filename already exists in current directory
	let filenameExists = $derived(
		saveMode && filename && entries.some((e) => !e.is_dir && e.name === filename)
	);

	$effect(() => {
		if (open) {
			// Reset stale state from previous browse session
			history = [];
			historyIndex = -1;
			error = '';
			editingPath = false;
			entries = [];
			loadDir(startPath || '/');
		}
	});

	/**
	 * @param {string} path
	 * @param {boolean} [addToHistory]
	 */
	async function loadDir(path, addToHistory = true) {
		loading = true;
		error = '';
		try {
			const res = await request('GET', `/files?path=${encodeURIComponent(path)}`);
			currentPath = res.path;
			entries = res.entries.map(FileEntry.fromJSON);
			if (addToHistory) {
				history = [...history.slice(0, historyIndex + 1), res.path];
				historyIndex = history.length - 1;
			}
		} catch (e) {
			error = e.error_text || e.message || 'Failed to load directory';
		} finally {
			loading = false;
		}
	}

	function goUp() {
		const parent = currentPath.replace(/\/[^/]+\/?$/, '') || '/';
		loadDir(parent);
	}

	function goBack() {
		if (historyIndex > 0) {
			historyIndex--;
			loadDir(history[historyIndex], false);
		}
	}

	function goForward() {
		if (historyIndex < history.length - 1) {
			historyIndex++;
			loadDir(history[historyIndex], false);
		}
	}

	function startEditPath() {
		pathInputValue = currentPath;
		editingPath = true;
	}

	function submitPath() {
		if (cancelledEdit) {
			cancelledEdit = false;
			return;
		}
		editingPath = false;
		const val = pathInputValue.trim();
		if (val && val !== currentPath) {
			loadDir(val);
		}
	}

	function cancelEditPath() {
		cancelledEdit = true;
		editingPath = false;
	}

	/** @param {object} entry */
	function handleClick(entry) {
		if (entry.is_dir) {
			if (entry.name === '..') {
				const parent = currentPath.replace(/\/[^/]+\/?$/, '') || '/';
				loadDir(parent);
			} else {
				loadDir(currentPath.replace(/\/$/, '') + '/' + entry.name);
			}
		} else if (saveMode) {
			filename = entry.name;
		} else {
			const fullPath = currentPath.replace(/\/$/, '') + '/' + entry.name;
			onSelect(fullPath);
		}
	}

	function handleSave() {
		if (!filename.trim()) return;
		const fullPath = currentPath.replace(/\/$/, '') + '/' + filename.trim();
		onSelect(fullPath);
	}

	/** @param {KeyboardEvent} e */
	function handleDialogKeydown(e) {
		if (e.key === 'Escape') {
			e.preventDefault();
			onClose();
			return;
		}

		// Simple focus trap: keep Tab within the dialog
		if (e.key === 'Tab' && dialogRef) {
			const focusable = dialogRef.querySelectorAll(
				'button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])'
			);
			if (focusable.length === 0) return;

			const first = /** @type {HTMLElement} */ (focusable[0]);
			const last = /** @type {HTMLElement} */ (focusable[focusable.length - 1]);

			if (e.shiftKey && document.activeElement === first) {
				e.preventDefault();
				last.focus();
			} else if (!e.shiftKey && document.activeElement === last) {
				e.preventDefault();
				first.focus();
			}
		}
	}

	/** @param {MouseEvent} e */
	function handleBackdropClick(e) {
		if (e.target === e.currentTarget) {
			onClose();
		}
	}

	/**
	 * @param {string} name
	 * @returns {boolean}
	 */
	function matchesFilter(name) {
		if (!filterExtensions || filterExtensions.length === 0) return true;
		const ext = name.split('.').pop()?.toLowerCase() || '';
		return filterExtensions.includes(ext);
	}

	/**
	 * @param {number} [bytes]
	 * @returns {string}
	 */
	function formatSize(bytes) {
		if (bytes === undefined) return '';
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
		return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
	}
</script>

{#if open}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="fixed inset-0 bg-black/60 z-50 flex items-end sm:items-center justify-center sm:p-4"
		role="dialog"
		aria-modal="true"
		aria-label="Browse files"
		onkeydown={handleDialogKeydown}
		onclick={handleBackdropClick}
		bind:this={dialogRef}
		tabindex="-1"
	>
		<!-- Full-screen on mobile, centered card on sm+ -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			class="bg-surface-1 w-full h-[92vh] sm:h-auto sm:rounded-xl shadow-2xl sm:max-w-[600px] sm:max-h-[min(500px,85vh)]
				   flex flex-col border-t sm:border border-border rounded-t-xl"
			onclick={(e) => e.stopPropagation()}
			onkeydown={() => {}}
		>
			<!-- Header -->
			<div class="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
				<h3 class="text-sm font-semibold text-text-primary">Browse files</h3>
				<button
					type="button"
					class="w-10 h-10 sm:w-8 sm:h-8 flex items-center justify-center text-text-muted hover:text-text-primary
						   hover:bg-surface-3 rounded-lg transition-colors
						   focus-visible:ring-2 focus-visible:ring-accent/50"
					onclick={onClose}
					aria-label="Close"
				>&times;</button>
			</div>

			<!-- Navigation bar -->
			<div class="flex items-center gap-1 px-4 py-2 border-b border-border shrink-0">
				<button
					type="button"
					class="w-10 h-10 sm:w-8 sm:h-8 flex items-center justify-center text-sm rounded-lg
						   enabled:hover:bg-surface-3 disabled:text-text-muted text-text-secondary transition-colors
						   focus-visible:ring-2 focus-visible:ring-accent/50"
					disabled={historyIndex <= 0}
					onclick={goBack}
					title="Back"
					aria-label="Go back"
				>&larr;</button>
				<button
					type="button"
					class="w-10 h-10 sm:w-8 sm:h-8 flex items-center justify-center text-sm rounded-lg
						   enabled:hover:bg-surface-3 disabled:text-text-muted text-text-secondary transition-colors
						   focus-visible:ring-2 focus-visible:ring-accent/50"
					disabled={historyIndex >= history.length - 1}
					onclick={goForward}
					title="Forward"
					aria-label="Go forward"
				>&rarr;</button>
				<button
					type="button"
					class="w-10 h-10 sm:w-8 sm:h-8 flex items-center justify-center text-sm rounded-lg
						   enabled:hover:bg-surface-3 disabled:text-text-muted text-text-secondary transition-colors
						   focus-visible:ring-2 focus-visible:ring-accent/50"
					disabled={currentPath === '/'}
					onclick={goUp}
					title="Up"
					aria-label="Go up one directory"
				>&uarr;</button>
				{#if editingPath}
					<input
						type="text"
						class="flex-1 ml-2 bg-surface-2 text-text-primary text-xs px-2 py-1.5 rounded-lg border border-border
							   focus:border-border-focus focus:ring-1 focus:ring-border-focus/30 focus:outline-none font-mono"
						bind:value={pathInputValue}
						onkeydown={(e) => { if (e.key === 'Enter') submitPath(); if (e.key === 'Escape') cancelEditPath(); }}
						onblur={submitPath}
						autofocus
					/>
				{:else}
					<button
						type="button"
						class="flex-1 ml-2 text-left truncate rounded-md focus-visible:ring-2 focus-visible:ring-accent/50"
						onclick={startEditPath}
						title="Click to edit path"
					>
						<code class="text-xs text-text-secondary hover:text-text-primary transition-colors">{currentPath}</code>
					</button>
				{/if}
			</div>

			<!-- Error -->
			{#if error}
				<div class="px-4 py-2 text-sm text-danger shrink-0">{error}</div>
			{/if}

			<!-- File list -->
			<div class="flex-1 overflow-y-auto min-h-0">
				{#if loading}
					<div class="px-4 py-8 text-center text-text-muted text-sm">Loading...</div>
				{:else if entries.length === 0}
					<div class="px-4 py-8 text-center text-text-muted text-sm">Empty directory</div>
				{:else}
					{#each entries as entry}
						{#if entry.is_dir || matchesFilter(entry.name)}
							<button
								type="button"
								class="w-full text-left px-4 py-2.5 hover:bg-surface-2 flex items-center gap-2 text-sm transition-colors
								focus-visible:bg-surface-2 focus-visible:outline-none"
								onclick={() => handleClick(entry)}
							>
								<span class="w-4 text-center text-text-muted shrink-0">
									{entry.is_dir ? '/' : ' '}
								</span>
								<span class="truncate {entry.is_dir ? 'text-accent' : 'text-text-primary'}">
									{entry.name}
								</span>
								{#if !entry.is_dir}
									<span class="ml-auto text-xs text-text-muted shrink-0 tabular-nums">{formatSize(entry.size)}</span>
								{/if}
							</button>
						{/if}
					{/each}
				{/if}
			</div>

			<!-- Save mode footer -->
			{#if saveMode}
				<div class="px-4 py-3 border-t border-border flex flex-col gap-2 shrink-0">
					<div class="flex items-center gap-2">
						<span class="text-xs text-text-secondary shrink-0">File name:</span>
						<input
							type="text"
							class="flex-1 min-w-0 bg-surface-2 text-text-primary text-sm px-2.5 py-1.5 rounded-lg border border-border
								   focus:border-border-focus focus:ring-1 focus:ring-border-focus/30 focus:outline-none"
							bind:value={filename}
							onkeydown={(e) => { if (e.key === 'Enter') handleSave(); }}
						/>
						<button
							type="button"
							class="px-3 py-1.5 bg-accent hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed
								   text-sm text-white rounded-lg font-medium transition-colors shrink-0
								   focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1"
							disabled={!filename.trim()}
							onclick={handleSave}
						>
							Save
						</button>
					</div>
					{#if filenameExists}
						<div class="text-xs text-warning">File already exists and will be overwritten</div>
					{/if}
				</div>
			{/if}
		</div>
	</div>
{/if}
