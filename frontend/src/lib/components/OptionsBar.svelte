<script>
	let { enhance, mouthMask, onEnhanceChange, onMouthMaskChange } = $props();
</script>

<div class="flex flex-wrap gap-x-6 gap-y-3 items-center">
	<label class="flex items-center gap-2.5 cursor-pointer select-none group" title="Improve face quality with GFPGAN enhancement">
		<span class="relative inline-flex items-center justify-center w-4 h-4">
			<input
				type="checkbox"
				class="peer appearance-none w-4 h-4 rounded border border-border bg-surface-2
					   checked:bg-accent checked:border-accent transition-colors cursor-pointer
					   focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1 focus-visible:ring-offset-surface-1"
				checked={enhance}
				onchange={(e) => onEnhanceChange(e.currentTarget.checked)}
			/>
			<svg class="absolute w-2.5 h-2.5 text-white pointer-events-none opacity-0 peer-checked:opacity-100 transition-opacity" viewBox="0 0 12 12" fill="none">
				<path d="M2 6l3 3 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
			</svg>
		</span>
		<span class="text-sm text-text-primary">Enhance (GFPGAN)</span>
	</label>

	<label class="flex items-center gap-2.5 cursor-pointer select-none group" title="Preserve original mouth to avoid artifacts on lip movement">
		<span class="relative inline-flex items-center justify-center w-4 h-4">
			<input
				type="checkbox"
				class="peer appearance-none w-4 h-4 rounded border border-border bg-surface-2
					   checked:bg-accent checked:border-accent transition-colors cursor-pointer
					   focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1 focus-visible:ring-offset-surface-1"
				checked={mouthMask}
				onchange={(e) => onMouthMaskChange(e.currentTarget.checked)}
			/>
			<svg class="absolute w-2.5 h-2.5 text-white pointer-events-none opacity-0 peer-checked:opacity-100 transition-opacity" viewBox="0 0 12 12" fill="none">
				<path d="M2 6l3 3 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
			</svg>
		</span>
		<span class="text-sm text-text-primary">Mouth mask</span>
	</label>
</div>
