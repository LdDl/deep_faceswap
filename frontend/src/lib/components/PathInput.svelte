<script>
	import FileBrowser from './FileBrowser.svelte';

	let {
		label = '',
		value,
		filterExtensions = [],
		storageKey = '',
		onchange,
		saveMode = false,
		defaultFilename = '',
		placeholder = '',
		onRemove = undefined,
		removeDisabled = false
	} = $props();

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

<div class="flex flex-col gap-1.5">
	{#if label}<span class="text-xs font-medium text-text-secondary">{label}</span>{/if}
	<div class="flex gap-2">
		<input
			type="text"
			class="flex-1 min-w-0 bg-surface-2 text-text-primary text-sm px-3 py-2 rounded-lg border border-border
				   focus:border-border-focus focus:ring-1 focus:ring-border-focus/30 focus:outline-none
				   placeholder:text-text-muted transition-colors"
			aria-label={label || placeholder || 'File path'}
			{value}
			{placeholder}
			oninput={(e) => onchange(e.currentTarget.value)}
		/>
		<button
			class="px-3 py-2 bg-surface-2 hover:bg-surface-3 text-sm text-text-secondary hover:text-text-primary
				   rounded-lg border border-border transition-colors shrink-0
				   focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1"
			onclick={() => (browserOpen = true)}
		>
			Browse
		</button>
		{#if onRemove}
			<button
				class="px-2 py-2 text-sm rounded-lg border border-border transition-colors shrink-0
					   {removeDisabled
						? 'text-text-muted bg-surface-2 opacity-40 cursor-not-allowed'
						: 'text-danger/70 hover:text-danger hover:bg-danger/10 bg-surface-2 cursor-pointer'}
					   focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1"
				onclick={onRemove}
				disabled={removeDisabled}
				aria-label="Remove"
			>&times;</button>
		{/if}
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
