<script>
	import '../app.css';
	import { page } from '$app/stores';

	let { children } = $props();
</script>

<div class="min-h-screen bg-gray-900 text-gray-100">
	<nav class="border-b border-gray-800 px-6 py-3 flex items-center gap-6">
		<span class="text-lg font-semibold text-gray-200">Deep FaceSwap</span>
		<div class="flex gap-1">
			<a
				href="/image"
				class="px-3 py-1.5 rounded text-sm transition-colors {$page.url.pathname === '/image' || $page.url.pathname === '/' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-gray-200'}"
			>
				Image
			</a>
			<a
				href="/video"
				class="px-3 py-1.5 rounded text-sm transition-colors {$page.url.pathname === '/video' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-gray-200'}"
			>
				Video
			</a>
		</div>
	</nav>

	<main class="max-w-4xl mx-auto px-6 py-6">
		{@render children()}
	</main>
</div>
