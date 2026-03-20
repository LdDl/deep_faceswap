<script>
	let { enhance, mouthMask, onEnhanceChange, onMouthMaskChange } = $props();
</script>

<div class="flex gap-6 items-center">
	<label class="flex items-center gap-2 cursor-pointer">
		<input
			type="checkbox"
			class="w-4 h-4 rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-gray-800"
			checked={enhance}
			onchange={(e) => onEnhanceChange(e.currentTarget.checked)}
		/>
		<span class="text-sm text-gray-300">Enhance (GFPGAN)</span>
	</label>

	<label class="flex items-center gap-2 cursor-pointer">
		<input
			type="checkbox"
			class="w-4 h-4 rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-gray-800"
			checked={mouthMask}
			onchange={(e) => onMouthMaskChange(e.currentTarget.checked)}
		/>
		<span class="text-sm text-gray-300">Mouth mask</span>
	</label>
</div>
