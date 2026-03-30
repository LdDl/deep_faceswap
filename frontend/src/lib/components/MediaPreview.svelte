<script>
	import { fileUrl } from '$lib/api.js';
	import Lightbox from './Lightbox.svelte';

	let { path, type = 'image' } = $props();

	let lightboxSrc = $state('');

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
	let loadError = $state(false);

	// Reset state when path changes
	$effect(() => {
		path;
		loaded = false;
		loadError = false;
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
			<button
				type="button"
				class="aspect-video w-full flex items-center justify-center bg-black relative
					   {loaded && !loadError ? 'cursor-zoom-in' : 'cursor-default'}
					   focus-visible:ring-2 focus-visible:ring-accent/50"
				onclick={() => { if (loaded && !loadError) lightboxSrc = fileUrl(path); }}
				tabindex={loaded && !loadError ? 0 : -1}
				aria-label={loaded && !loadError ? 'View full image' : undefined}
			>
				{#if !loaded && !loadError}
					<div class="absolute inset-0 bg-surface-2 animate-pulse"></div>
				{/if}
				{#if loadError}
					<div class="text-sm text-text-muted">Could not load image</div>
				{:else}
					<img
						src={fileUrl(path)}
						alt="Preview"
						loading="lazy"
						class="max-w-full max-h-full object-contain"
						onload={() => loaded = true}
						onerror={() => { loadError = true; loaded = true; }}
					/>
				{/if}
			</button>
		{/if}
	</div>
{/if}

<Lightbox src={lightboxSrc} alt="Preview" onClose={() => (lightboxSrc = '')} />
