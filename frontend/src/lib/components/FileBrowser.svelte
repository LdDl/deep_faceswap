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
	let filename = $state('');

	// Initialize filename when opening in save mode
	$effect(() => {
		if (open && saveMode && defaultFilename && !filename) {
			filename = defaultFilename;
		}
	});

	// Check if filename already exists in current directory
	let filenameExists = $derived(
		saveMode && filename && entries.some((e) => !e.is_dir && e.name === filename)
	);

	$effect(() => {
		if (open) {
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
		editingPath = false;
		const val = pathInputValue.trim();
		if (val && val !== currentPath) {
			loadDir(val);
		}
	}

	function cancelEditPath() {
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
	<div class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center" role="dialog">
		<div class="bg-gray-800 rounded-lg shadow-xl w-[600px] max-h-[500px] flex flex-col">
			<div class="flex items-center justify-between px-4 py-3 border-b border-gray-700">
				<h3 class="text-sm font-medium text-gray-200">Browse files</h3>
				<button class="text-gray-400 hover:text-gray-200" onclick={onClose}>&times;</button>
			</div>

			<div class="flex items-center gap-1 px-4 py-2 border-b border-gray-700">
				<button
					class="px-2 py-1 text-xs rounded enabled:hover:bg-gray-700 disabled:text-gray-600 text-gray-400"
					disabled={historyIndex <= 0}
					onclick={goBack}
					title="Back"
				>&larr;</button>
				<button
					class="px-2 py-1 text-xs rounded enabled:hover:bg-gray-700 disabled:text-gray-600 text-gray-400"
					disabled={historyIndex >= history.length - 1}
					onclick={goForward}
					title="Forward"
				>&rarr;</button>
				<button
					class="px-2 py-1 text-xs rounded enabled:hover:bg-gray-700 disabled:text-gray-600 text-gray-400"
					disabled={currentPath === '/'}
					onclick={goUp}
					title="Up"
				>&uarr;</button>
				{#if editingPath}
					<input
						type="text"
						class="flex-1 ml-2 bg-gray-700 text-gray-200 text-xs px-2 py-1 rounded border border-gray-500 focus:border-blue-500 focus:outline-none font-mono"
						bind:value={pathInputValue}
						onkeydown={(e) => { if (e.key === 'Enter') submitPath(); if (e.key === 'Escape') cancelEditPath(); }}
						onblur={submitPath}
						autofocus
					/>
				{:else}
					<button
						class="flex-1 ml-2 text-left truncate"
						onclick={startEditPath}
						title="Click to edit path"
					>
						<code class="text-xs text-gray-400 hover:text-gray-200">{currentPath}</code>
					</button>
				{/if}
			</div>

			{#if error}
				<div class="px-4 py-2 text-sm text-red-400">{error}</div>
			{/if}

			<div class="flex-1 overflow-y-auto">
				{#if loading}
					<div class="px-4 py-8 text-center text-gray-500">Loading...</div>
				{:else}
					{#each entries as entry}
						{#if entry.is_dir || matchesFilter(entry.name)}
							<button
								class="w-full text-left px-4 py-1.5 hover:bg-gray-700 flex items-center gap-2 text-sm"
								onclick={() => handleClick(entry)}
							>
								<span class="w-4 text-center text-gray-500">
									{entry.is_dir ? '/' : ' '}
								</span>
								<span class={entry.is_dir ? 'text-blue-400' : 'text-gray-300'}>
									{entry.name}
								</span>
								{#if !entry.is_dir}
									<span class="ml-auto text-xs text-gray-600">{formatSize(entry.size)}</span>
								{/if}
							</button>
						{/if}
					{/each}
				{/if}
			</div>

			{#if saveMode}
				<div class="px-4 py-3 border-t border-gray-700 flex flex-col gap-2">
					<div class="flex items-center gap-2">
						<span class="text-xs text-gray-400 shrink-0">File name:</span>
						<input
							type="text"
							class="flex-1 bg-gray-700 text-gray-200 text-sm px-2 py-1 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
							bind:value={filename}
							onkeydown={(e) => { if (e.key === 'Enter') handleSave(); }}
						/>
						<button
							class="px-3 py-1 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:text-gray-400 text-sm text-white rounded"
							disabled={!filename.trim()}
							onclick={handleSave}
						>
							Save
						</button>
					</div>
					{#if filenameExists}
						<div class="text-xs text-yellow-400">File already exists and will be overwritten</div>
					{/if}
				</div>
			{/if}
		</div>
	</div>
{/if}
