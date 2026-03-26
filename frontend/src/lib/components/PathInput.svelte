<script>
	import FileBrowser from './FileBrowser.svelte';

	let { label, value, filterExtensions = [], storageKey = '', onchange, saveMode = false, defaultFilename = '' } = $props();

	let browserOpen = $state(false);

	/** @param {string} path */
	function handleSelect(path) {
		if (storageKey) {
			const dir = path.replace(/\/[^/]+$/, '') || '/';
			localStorage.setItem(`browse_dir_${storageKey}`, dir);
		}
		onchange(path);
		browserOpen = false;
	}

	/** @returns {string} */
	function startPath() {
		if (value) {
			const dir = value.replace(/\/[^/]+$/, '');
			return dir || '/';
		}
		if (storageKey) {
			return localStorage.getItem(`browse_dir_${storageKey}`) || '/';
		}
		return '/';
	}
</script>

<div class="flex flex-col gap-1">
	{#if label}<span class="text-xs text-gray-400">{label}</span>{/if}
	<div class="flex gap-2">
		<input
			type="text"
			class="flex-1 bg-gray-700 text-gray-200 text-sm px-3 py-1.5 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
			{value}
			oninput={(e) => onchange(e.currentTarget.value)}
		/>
		<button
			class="px-3 py-1.5 bg-gray-600 hover:bg-gray-500 text-sm text-gray-200 rounded"
			onclick={() => (browserOpen = true)}
		>
			Browse
		</button>
	</div>
</div>

<FileBrowser
	open={browserOpen}
	startPath={startPath()}
	{filterExtensions}
	{saveMode}
	{defaultFilename}
	onSelect={handleSelect}
	onClose={() => (browserOpen = false)}
/>
