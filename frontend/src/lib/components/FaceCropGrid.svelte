<script>
	import { cropUrl } from '$lib/api.js';

	let { label, items, selectedIndex = null, onSelect } = $props();

	/** @param {object} item */
	function isCluster(item) {
		return 'cluster_id' in item;
	}

	/** @param {object} item */
	function getIndex(item) {
		return isCluster(item) ? item.cluster_id : item.index;
	}

	/** @param {object} item */
	function getLabel(item) {
		if (isCluster(item)) {
			return `Cluster ${item.cluster_id} (${item.frame_count} frames)`;
		}
		let text = `Face ${item.index}`;
		if (item.source_filename) text += ` (${item.source_filename})`;
		return text;
	}

	/** @param {object} item */
	function getScore(item) {
		if (isCluster(item)) return '';
		return item.det_score.toFixed(2);
	}
</script>

<div class="flex flex-col gap-2">
	<h3 class="text-sm font-medium text-gray-300">{label}</h3>
	{#if items.length === 0}
		<p class="text-xs text-gray-500">No faces detected</p>
	{:else}
		<div class="flex flex-wrap gap-3">
			{#each items as item}
				{@const idx = getIndex(item)}
				<button
					class="flex flex-col items-center gap-1 p-1.5 rounded border-2 transition-colors {selectedIndex === idx ? 'border-blue-500 bg-blue-500/10' : 'border-transparent hover:border-gray-600'}"
					onclick={() => onSelect?.(idx)}
				>
					<img
						src={cropUrl(item.crop_url)}
						alt={getLabel(item)}
						class="w-20 h-20 object-cover rounded"
					/>
					<span class="text-xs text-gray-400">{getLabel(item)}</span>
					{#if getScore(item)}
						<span class="text-xs text-gray-600">{getScore(item)}</span>
					{/if}
				</button>
			{/each}
		</div>
	{/if}
</div>
