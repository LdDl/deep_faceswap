<script>
	let { src = '', alt = '', onClose } = $props();

	/** @type {HTMLDivElement|undefined} */
	let overlayRef = $state();
	let loaded = $state(false);

	// Reset loaded + auto-focus overlay when src changes
	$effect(() => {
		if (src) {
			loaded = false;
			if (overlayRef) overlayRef.focus();
		}
	});

	/** @param {KeyboardEvent} e */
	function handleKeydown(e) {
		if (e.key === 'Escape') {
			e.preventDefault();
			onClose();
		}
		// Simple focus trap — only one interactive element (close button)
		if (e.key === 'Tab') {
			e.preventDefault();
		}
	}

	/** @param {MouseEvent} e */
	function handleBackdrop(e) {
		if (e.target === e.currentTarget) onClose();
	}
</script>

{#if src}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="fixed inset-0 z-[60] bg-black/80 flex items-center justify-center p-4 sm:p-8"
		role="dialog"
		aria-modal="true"
		aria-label={alt || 'Image preview'}
		tabindex="-1"
		bind:this={overlayRef}
		onclick={handleBackdrop}
		onkeydown={handleKeydown}
	>
		<button
			class="absolute top-3 right-3 w-10 h-10 flex items-center justify-center
				   text-white/70 hover:text-white bg-white/10 hover:bg-white/20
				   rounded-full transition-colors text-lg
				   focus-visible:ring-2 focus-visible:ring-white/50"
			onclick={onClose}
			aria-label="Close preview"
		>&times;</button>
		{#if !loaded}
			<div class="absolute inset-0 flex items-center justify-center pointer-events-none">
				<div class="w-8 h-8 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
			</div>
		{/if}
		<img
			{src}
			{alt}
			class="max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl transition-opacity duration-150
				   {loaded ? 'opacity-100' : 'opacity-0'}"
			onload={() => (loaded = true)}
			onerror={() => (loaded = true)}
		/>
	</div>
{/if}
