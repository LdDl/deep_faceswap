<script>
	import { fileUrl } from '$lib/api.js';

	let { path, type = 'image' } = $props();

	const videoExtensions = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'm4v'];
	const browserPlayable = ['mp4', 'webm', 'mov'];

	let isVideo = $derived(
		type === 'video' ||
		videoExtensions.some((ext) => path.toLowerCase().endsWith('.' + ext))
	);

	let canPlay = $derived(
		browserPlayable.some((ext) => path.toLowerCase().endsWith('.' + ext))
	);

	let extension = $derived(path.split('.').pop()?.toLowerCase() || '');
	let loaded = $state(false);

	// Reset loaded state when path changes
	$effect(() => {
		path;
		loaded = false;
	});
</script>

{#if path}
	<div class="w-full rounded-lg overflow-hidden border border-border bg-surface-2">
		{#if isVideo}
			{#if canPlay}
				<div class="aspect-video bg-black">
					<video
						src={fileUrl(path)}
						controls
						preload="metadata"
						class="w-full h-full object-contain"
					>
						<track kind="captions" />
					</video>
				</div>
			{:else}
				<div class="aspect-video flex items-center justify-center text-sm text-text-muted bg-surface-2">
					.{extension} preview not supported in browser
				</div>
			{/if}
		{:else}
			<div class="aspect-video flex items-center justify-center bg-black relative">
				{#if !loaded}
					<div class="absolute inset-0 bg-surface-2 animate-pulse"></div>
				{/if}
				<img
					src={fileUrl(path)}
					alt="Preview"
					loading="lazy"
					class="max-w-full max-h-full object-contain"
					onload={() => loaded = true}
					onerror={() => loaded = true}
				/>
			</div>
		{/if}
	</div>
{/if}
