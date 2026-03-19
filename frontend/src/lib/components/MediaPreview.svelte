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
</script>

{#if path}
	<div class="w-full max-w-sm rounded overflow-hidden border border-gray-700 bg-gray-900">
		{#if isVideo}
			{#if canPlay}
				<video
					src={fileUrl(path)}
					controls
					preload="metadata"
					class="w-full max-h-72 object-contain"
				>
					<track kind="captions" />
				</video>
			{:else}
				<div class="flex items-center justify-center h-24 text-sm text-gray-500">
					.{extension} preview not supported in browser
				</div>
			{/if}
		{:else}
			<img
				src={fileUrl(path)}
				alt="Preview"
				class="w-full max-h-72 object-contain"
			/>
		{/if}
	</div>
{/if}
