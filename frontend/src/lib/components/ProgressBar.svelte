<script>
	let { progress } = $props();

	let percent = $derived(
		progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0
	);
</script>

<div class="flex flex-col gap-1">
	<div class="flex justify-between text-xs text-gray-400">
		<span>{progress.stage}</span>
		<span>
			{#if progress.total > 0}
				{progress.current} / {progress.total} ({percent}%)
			{/if}
		</span>
	</div>
	<div class="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
		<div
			class="h-full bg-blue-500 transition-all duration-300"
			style="width: {percent}%"
		></div>
	</div>
</div>
