<script>
	import { cropUrl } from '$lib/api.js';
	import { FaceMapping } from '$lib/face_mapping.js';
	import { ClusterMapping } from '$lib/cluster_mapping.js';
	import Lightbox from './Lightbox.svelte';

	import { untrack } from 'svelte';

	let { sourceFaces, targetItems, mode, onMappingsChange, initialAutoMap = false } = $props();

	let selectedSource = $state(null);
	/** @type {{ source: number, target: number }[]} */
	let mappings = $state([]);
	let lightboxSrc = $state('');

	// Reset mappings when source/target props change; optionally auto-map
	let prevSourceLen = -1;
	let prevTargetLen = -1;
	$effect(() => {
		const sLen = sourceFaces.length;
		const tLen = targetItems.length;
		if (sLen !== prevSourceLen || tLen !== prevTargetLen) {
			prevSourceLen = sLen;
			prevTargetLen = tLen;
			untrack(() => {
				mappings = [];
				selectedSource = null;
				if (initialAutoMap && sLen > 0 && tLen > 0) {
					autoMap();
				} else {
					emitMappings();
				}
			});
		}
	});

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

	/** @param {object} item */
	function getTargetSublabel(item) {
		if (isCluster(item)) return `${item.frame_count} frames`;
		return '';
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
		<h3 class="text-sm font-semibold text-text-primary">Face mapping</h3>
		<div class="flex gap-2">
			<button
				type="button"
				class="px-2.5 py-1 text-xs font-medium bg-surface-2 hover:bg-surface-3
					   text-text-secondary hover:text-text-primary rounded-md border border-border transition-colors
					   focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:ring-offset-1 focus-visible:ring-offset-surface-1"
				onclick={autoMap}
			>
				Auto-map
			</button>
			<button
				type="button"
				class="px-2.5 py-1 text-xs font-medium bg-surface-2 hover:bg-surface-3
					   text-text-secondary hover:text-text-primary rounded-md border border-border transition-colors
					   focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:ring-offset-1 focus-visible:ring-offset-surface-1"
				onclick={() => { mappings = []; emitMappings(); }}
			>
				Clear
			</button>
		</div>
	</div>

	<p class="text-xs text-text-muted">
		{#if selectedSource !== null}
			<span class="text-accent">Source face {selectedSource} selected.</span> Click a target to map.
		{:else}
			Click a source face, then click a target to create a mapping.
		{/if}
	</p>

	<!-- Responsive: stack on mobile, row on sm+ -->
	<div class="flex flex-col sm:flex-row gap-4 sm:gap-6">
		<!-- Source faces -->
		<div class="flex flex-col gap-2 min-w-0 flex-1">
			<span class="text-xs font-medium text-text-secondary uppercase tracking-wider">Source</span>
			{#if sourceFaces.length === 0}
				<div class="py-6 text-center text-xs text-text-muted rounded-lg border border-dashed border-border">
					No source faces detected
				</div>
			{:else}
				<div class="flex flex-row sm:flex-col gap-2 overflow-x-auto sm:overflow-x-visible pb-2 sm:pb-0">
					{#each sourceFaces as face}
						{@const mapped = getMappedTarget(face.index)}
						<button
							type="button"
							class="flex items-center gap-2 p-1.5 rounded-lg border-2 transition-colors shrink-0
								   focus-visible:ring-2 focus-visible:ring-accent/50
								   {selectedSource === face.index
									? 'border-accent bg-accent/10'
									: mapped !== undefined
										? 'border-success/60 bg-success/5'
										: 'border-border hover:border-text-muted'}"
							onclick={() => handleSourceClick(face.index)}
						>
							<!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
							<!-- svelte-ignore a11y_click_events_have_key_events -->
							<img
								src={cropUrl(face.crop_url)}
								alt="Source face {face.index}"
								width="64" height="64" class="w-16 h-16 object-cover rounded-md cursor-zoom-in"
								onclick={(e) => { e.stopPropagation(); lightboxSrc = cropUrl(face.crop_url); }}
								role="button"
								tabindex="-1"
							/>
							<div class="flex flex-col items-start">
								<span class="text-xs font-mono tabular-nums text-text-secondary">#{face.index}</span>
								{#if face.source_filename}
									<span class="text-xs text-text-muted truncate max-w-[80px]">{face.source_filename}</span>
								{/if}
							</div>
						</button>
					{/each}
				</div>
			{/if}
		</div>

		<!-- Mapping display -->
		<div class="flex flex-row sm:flex-col justify-center items-center gap-1.5 shrink-0 min-w-[76px]">
			{#if mappings.length === 0}
				<span class="text-xs text-text-muted">No mappings</span>
			{:else}
				{#each mappings as mapping, i}
					<div class="flex items-center gap-1.5 text-xs text-text-secondary bg-surface-2 px-2.5 py-1 rounded-md border border-border">
						<span class="font-mono tabular-nums">{mapping.source}</span>
						<span class="text-text-muted">&rarr;</span>
						<span class="font-mono tabular-nums">{mapping.target}</span>
						<button
							type="button"
							class="ml-0.5 w-6 h-6 sm:w-5 sm:h-5 inline-flex items-center justify-center rounded-sm
								   text-danger/70 hover:text-danger hover:bg-danger/10 transition-colors
								   focus-visible:ring-2 focus-visible:ring-danger/50"
							onclick={() => removeMapping(i)}
							aria-label="Remove mapping {mapping.source} to {mapping.target}"
						>&times;</button>
					</div>
				{/each}
			{/if}
		</div>

		<!-- Target faces / clusters -->
		<div class="flex flex-col gap-2 min-w-0 flex-1">
			<span class="text-xs font-medium text-text-secondary uppercase tracking-wider">
				{mode === 'video' ? 'Clusters' : 'Target'}
			</span>
			{#if targetItems.length === 0}
				<div class="py-6 text-center text-xs text-text-muted rounded-lg border border-dashed border-border">
					No {mode === 'video' ? 'clusters' : 'target faces'} detected
				</div>
			{:else}
				<div class="flex flex-row sm:flex-col gap-2 overflow-x-auto sm:overflow-x-visible pb-2 sm:pb-0">
					{#each targetItems as item}
						{@const tid = getTargetId(item)}
						{@const mapped = getMappedSource(tid)}
						<button
							type="button"
							class="flex items-center gap-2 p-1.5 rounded-lg border-2 transition-colors shrink-0
								   focus-visible:ring-2 focus-visible:ring-accent/50
								   {mapped !== undefined
									? 'border-success/60 bg-success/5'
									: 'border-border hover:border-text-muted'}"
							onclick={() => handleTargetClick(tid)}
						>
							<!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
							<!-- svelte-ignore a11y_click_events_have_key_events -->
							<img
								src={cropUrl(item.crop_url)}
								alt={getTargetLabel(item)}
								width="64" height="64" class="w-16 h-16 object-cover rounded-md cursor-zoom-in"
								onclick={(e) => { e.stopPropagation(); lightboxSrc = cropUrl(item.crop_url); }}
								role="button"
								tabindex="-1"
							/>
							<div class="flex flex-col items-start">
								<span class="text-xs text-text-secondary">{getTargetLabel(item)}</span>
								{#if getTargetSublabel(item)}
									<span class="text-xs text-text-muted">{getTargetSublabel(item)}</span>
								{/if}
							</div>
						</button>
					{/each}
				</div>
			{/if}
		</div>
	</div>
</div>

<Lightbox src={lightboxSrc} alt="Face crop" onClose={() => (lightboxSrc = '')} />
