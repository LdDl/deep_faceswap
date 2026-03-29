<script>
	import PathInput from './PathInput.svelte';
	import Lightbox from './Lightbox.svelte';
	import { fileUrl } from '$lib/api.js';

	let { paths = $bindable(['']), filterExtensions = [], storageKey = 'source' } = $props();

	let lightboxSrc = $state('');

	function addPath() {
		paths = [...paths, ''];
	}

	/** @param {number} index */
	function removePath(index) {
		paths = paths.filter((_, i) => i !== index);
	}

	/**
	 * @param {number} index
	 * @param {string} value
	 */
	function updatePath(index, value) {
		paths[index] = value;
	}

	let canRemove = $derived(paths.length > 1);

	/**
	 * Track which thumbnails failed to load.
	 * @type {Record<number, boolean>}
	 */
	let thumbErrors = $state({});

	/** @param {number} index */
	function onThumbError(index) {
		thumbErrors = { ...thumbErrors, [index]: true };
	}

	// Reset error state when path changes
	$effect(() => {
		paths.forEach((p, i) => {
			if (p) {
				thumbErrors = { ...thumbErrors, [i]: false };
			}
		});
	});
</script>

<div class="flex flex-col gap-2">
	<div class="max-h-[40vh] sm:max-h-[280px] overflow-y-auto flex flex-col gap-2">
		{#each paths as path, i}
			<div class="flex items-end gap-2">
				<!-- Inline thumbnail — click to enlarge -->
				<button
					class="w-10 h-10 rounded-md border border-border shrink-0 overflow-hidden bg-surface-2 self-end
						   {path.trim() && !thumbErrors[i] ? 'cursor-zoom-in' : 'cursor-default'}
						   focus-visible:ring-2 focus-visible:ring-accent/50"
					onclick={() => { if (path.trim() && !thumbErrors[i]) lightboxSrc = fileUrl(path); }}
					tabindex={path.trim() && !thumbErrors[i] ? 0 : -1}
					aria-label={path.trim() && !thumbErrors[i] ? 'View full source image' : undefined}
				>
					{#if path.trim() && !thumbErrors[i]}
						<img
							src={fileUrl(path)}
							alt=""
							class="w-full h-full object-cover"
							onerror={() => onThumbError(i)}
						/>
					{:else}
						<div class="w-full h-full flex items-center justify-center text-text-muted text-xs">
							{#if path.trim() && thumbErrors[i]}!{/if}
						</div>
					{/if}
				</button>
				<!-- Input + Browse + optional × -->
				<div class="flex-1 min-w-0">
					<PathInput
						label={i === 0 ? 'Source image(s)' : ''}
						value={path}
						{filterExtensions}
						{storageKey}
						onchange={(/** @type {string} */ v) => updatePath(i, v)}
						onRemove={() => removePath(i)}
						removeDisabled={!canRemove}
					/>
				</div>
			</div>
		{/each}
	</div>

	<button
		class="px-3 py-1.5 text-xs font-medium text-accent border border-dashed border-accent/40
			   hover:border-accent hover:bg-accent/5 rounded-lg transition-colors self-start
		   focus-visible:ring-2 focus-visible:ring-accent/50"
		onclick={addPath}
	>
		+ Add source image
	</button>
</div>

<Lightbox src={lightboxSrc} alt="Source image" onClose={() => (lightboxSrc = '')} />
