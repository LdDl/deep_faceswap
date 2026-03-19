<script>
	import { cropUrl } from '$lib/api.js';
	import { FaceMapping } from '$lib/face_mapping.js';
	import { ClusterMapping } from '$lib/cluster_mapping.js';

	let { sourceFaces, targetItems, mode, onMappingsChange } = $props();

	let selectedSource = $state(null);
	let mappings = $state([]);

	/** @param {object} item */
	function isCluster(item) {
		return 'cluster_id' in item;
	}

	/** @param {object} item */
	function getTargetId(item) {
		return isCluster(item) ? item.cluster_id : item.index;
	}

	/** @param {object} item */
	function getTargetLabel(item) {
		if (isCluster(item)) return `Cluster ${item.cluster_id}`;
		return `Face ${item.index}`;
	}

	/** @param {number} idx */
	function handleSourceClick(idx) {
		selectedSource = selectedSource === idx ? null : idx;
	}

	/** @param {number} targetIdx */
	function handleTargetClick(targetIdx) {
		if (selectedSource === null) return;

		mappings = mappings.filter((m) => m.target !== targetIdx);
		mappings = [...mappings, { source: selectedSource, target: targetIdx }];
		selectedSource = null;

		emitMappings();
	}

	/** @param {number} index */
	function removeMapping(index) {
		mappings = mappings.filter((_, i) => i !== index);
		emitMappings();
	}

	function emitMappings() {
		if (mode === 'image') {
			onMappingsChange(
				mappings.map((m) => new FaceMapping(m.source, m.target))
			);
		} else {
			onMappingsChange(
				mappings.map((m) => new ClusterMapping(m.source, m.target))
			);
		}
	}

	function autoMap() {
		const count = Math.min(sourceFaces.length, targetItems.length);
		mappings = [];
		for (let i = 0; i < count; i++) {
			mappings.push({ source: i, target: getTargetId(targetItems[i]) });
		}
		emitMappings();
	}

	/** @param {number} sourceIdx */
	function getMappedTarget(sourceIdx) {
		return mappings.find((m) => m.source === sourceIdx)?.target;
	}

	/** @param {number} targetIdx */
	function getMappedSource(targetIdx) {
		return mappings.find((m) => m.target === targetIdx)?.source;
	}
</script>

<div class="flex flex-col gap-3">
	<div class="flex items-center justify-between">
		<h3 class="text-sm font-medium text-gray-300">Face mapping</h3>
		<div class="flex gap-2">
			<button
				class="px-2 py-1 text-xs bg-gray-600 hover:bg-gray-500 text-gray-200 rounded"
				onclick={autoMap}
			>
				Auto-map
			</button>
			<button
				class="px-2 py-1 text-xs bg-gray-600 hover:bg-gray-500 text-gray-200 rounded"
				onclick={() => { mappings = []; emitMappings(); }}
			>
				Clear
			</button>
		</div>
	</div>

	{#if selectedSource !== null}
		<p class="text-xs text-blue-400">
			Source face {selectedSource} selected. Click a target to map.
		</p>
	{:else}
		<p class="text-xs text-gray-500">Click a source face, then click a target to create a mapping.</p>
	{/if}

	<div class="flex gap-6">
		<!-- Source faces -->
		<div class="flex flex-col gap-2">
			<span class="text-xs text-gray-500">Source</span>
			{#each sourceFaces as face}
				{@const mapped = getMappedTarget(face.index)}
				<button
					class="flex items-center gap-2 p-1.5 rounded border-2 transition-colors {selectedSource === face.index ? 'border-blue-500' : mapped !== undefined ? 'border-green-600' : 'border-gray-700 hover:border-gray-500'}"
					onclick={() => handleSourceClick(face.index)}
				>
					<img src={cropUrl(face.crop_url)} alt="Source {face.index}" class="w-14 h-14 object-cover rounded" />
					<span class="text-xs text-gray-400">#{face.index}</span>
				</button>
			{/each}
		</div>

		<!-- Mapping lines -->
		<div class="flex flex-col justify-center gap-1 min-w-[60px]">
			{#each mappings as mapping, i}
				<div class="flex items-center gap-1 text-xs text-gray-400">
					<span>{mapping.source}</span>
					<span class="text-gray-600">-></span>
					<span>{mapping.target}</span>
					<button class="ml-1 text-red-500 hover:text-red-400" onclick={() => removeMapping(i)}>&times;</button>
				</div>
			{/each}
		</div>

		<!-- Target faces / clusters -->
		<div class="flex flex-col gap-2">
			<span class="text-xs text-gray-500">{mode === 'video' ? 'Clusters' : 'Target'}</span>
			{#each targetItems as item}
				{@const tid = getTargetId(item)}
				{@const mapped = getMappedSource(tid)}
				<button
					class="flex items-center gap-2 p-1.5 rounded border-2 transition-colors {mapped !== undefined ? 'border-green-600' : 'border-gray-700 hover:border-gray-500'}"
					onclick={() => handleTargetClick(tid)}
				>
					<img src={cropUrl(item.crop_url)} alt={getTargetLabel(item)} class="w-14 h-14 object-cover rounded" />
					<span class="text-xs text-gray-400">{getTargetLabel(item)}</span>
				</button>
			{/each}
		</div>
	</div>
</div>
