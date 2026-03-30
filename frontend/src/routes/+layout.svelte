<script>
	import '../app.css';
	import { page } from '$app/stores';

	let { children } = $props();

	let isImage = $derived($page.url.pathname === '/image' || $page.url.pathname === '/');
	let isVideo = $derived($page.url.pathname === '/video');
</script>

<div class="min-h-screen bg-surface-0 text-text-primary">
	<nav class="sticky top-0 z-40 border-b border-border bg-surface-0/80 backdrop-blur-sm">
		<div class="max-w-5xl mx-auto px-3 sm:px-6 py-3 flex items-center gap-4 sm:gap-6">
			<span class="text-base font-semibold tracking-tight text-text-primary shrink-0">Deep FaceSwap</span>
			<div class="flex gap-0.5 bg-surface-2 rounded-lg p-1">
				<a
					href="/image"
					class="px-4 py-2 rounded-md text-sm font-medium transition-all duration-150
						   focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:outline-none
						   {isImage ? 'bg-accent text-white shadow-sm' : 'text-text-secondary hover:text-text-primary'}"
				>
					Image
				</a>
				<a
					href="/video"
					class="px-4 py-2 rounded-md text-sm font-medium transition-all duration-150
						   focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:outline-none
						   {isVideo ? 'bg-accent text-white shadow-sm' : 'text-text-secondary hover:text-text-primary'}"
				>
					Video
				</a>
			</div>
		</div>
	</nav>

	<main class="max-w-5xl mx-auto px-3 sm:px-6 py-4 sm:py-8">
		{@render children()}
	</main>
</div>
