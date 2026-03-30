<script>
	let { progress } = $props();

	/** @type {Record<string, string>} */
	const stageLabels = {
		queued: 'Queued',
		initializing: 'Initializing',
		// image detect
		detecting_faces: 'Detecting faces',
		// analyze stages
		detecting_source_faces: 'Detecting source faces',
		extracting_frames: 'Extracting frames',
		scanning_faces: 'Scanning faces',
		clustering: 'Clustering faces',
		// swap stages
		processing_frames: 'Processing frames',
		encoding: 'Encoding video',
		// terminal
		completed: 'Completed',
		failed: 'Failed',
	};

	let stageLabel = $derived(stageLabels[progress.stage] || progress.stage);

	let percent = $derived(
		progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0
	);

	let isIndeterminate = $derived(progress.total === 0);
</script>

<div class="flex flex-col gap-1.5">
	<div class="flex justify-between text-xs text-text-secondary">
		<span>{stageLabel}</span>
		{#if !isIndeterminate}
			<span class="tabular-nums font-mono">{progress.current}/{progress.total} ({percent}%)</span>
		{/if}
	</div>
	<div
		class="w-full h-2 bg-surface-2 rounded-full overflow-hidden"
		role="progressbar"
		aria-label={stageLabel}
		aria-valuenow={isIndeterminate ? undefined : percent}
		aria-valuemin={0}
		aria-valuemax={100}
	>
		{#if isIndeterminate}
			<div class="h-full w-1/3 bg-accent/70 rounded-full animate-[indeterminate_1.5s_ease-in-out_infinite]"></div>
		{:else}
			<div
				class="h-full bg-accent rounded-full transition-all duration-300 ease-out"
				style="width: {percent}%"
			></div>
		{/if}
	</div>
</div>
