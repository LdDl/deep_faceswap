<script>
	import PathInput from '$lib/components/PathInput.svelte';
	import FaceMapper from '$lib/components/FaceMapper.svelte';
	import OptionsBar from '$lib/components/OptionsBar.svelte';
	import { detectFaces } from '$lib/stores/detection.js';
	import { swapImage } from '$lib/stores/swap.js';
	import { FaceMapping } from '$lib/face_mapping.js';
	import { DetectionResponse } from '$lib/stores/detection.js';
	import { SwapImageResponse } from '$lib/stores/swap.js';

	let sourcePaths = $state(['']);
	let targetPath = $state('');
	let outputPath = $state('');
	let enhance = $state(true);
	let mouthMask = $state(false);

	/** @type {DetectionResponse|null} */
	let detection = $state(null);
	/** @type {FaceMapping[]} */
	let mappings = $state([]);
	/** @type {SwapImageResponse|null} */
	let swapResult = $state(null);

	let detecting = $state(false);
	let swapping = $state(false);
	let error = $state('');

	function addSourcePath() {
		sourcePaths = [...sourcePaths, ''];
	}

	/** @param {number} index */
	function removeSourcePath(index) {
		sourcePaths = sourcePaths.filter((_, i) => i !== index);
	}

	/**
	 * @param {number} index
	 * @param {string} value
	 */
	function updateSourcePath(index, value) {
		sourcePaths[index] = value;
	}

	async function handleDetect() {
		const paths = sourcePaths.filter((p) => p.trim());
		if (paths.length === 0 || !targetPath.trim()) {
			error = 'Provide at least one source path and a target path';
			return;
		}

		error = '';
		detecting = true;
		detection = null;
		swapResult = null;
		mappings = [];

		try {
			detection = await detectFaces(paths, targetPath);

			if (detection.source_faces.length === 1 && detection.target_faces.length === 1) {
				mappings = [new FaceMapping(0, 0)];
			}
		} catch (e) {
			error = e.error_text || e.message || 'Detection failed';
		} finally {
			detecting = false;
		}
	}

	async function handleSwap() {
		if (!detection || mappings.length === 0) {
			error = 'Detect faces and create mappings first';
			return;
		}
		if (!outputPath.trim()) {
			error = 'Provide an output path';
			return;
		}

		error = '';
		swapping = true;
		swapResult = null;

		try {
			swapResult = await swapImage(
				sourcePaths.filter((p) => p.trim()),
				targetPath,
				outputPath,
				mappings,
				enhance,
				mouthMask
			);
		} catch (e) {
			error = e.error_text || e.message || 'Swap failed';
		} finally {
			swapping = false;
		}
	}

	const imageExtensions = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];
</script>

<div class="flex flex-col gap-6">
	<h2 class="text-xl font-semibold">Image swap</h2>

	<!-- Source paths -->
	<div class="flex flex-col gap-3">
		{#each sourcePaths as path, i}
			<div class="flex items-end gap-2">
				<div class="flex-1">
					<PathInput
						label={i === 0 ? 'Source image(s)' : ''}
						value={path}
						filterExtensions={imageExtensions}
						onchange={(v) => updateSourcePath(i, v)}
					/>
				</div>
				{#if sourcePaths.length > 1}
					<button
						class="px-2 py-1.5 text-sm text-red-400 hover:text-red-300"
						onclick={() => removeSourcePath(i)}
					>
						Remove
					</button>
				{/if}
			</div>
		{/each}
		<button
			class="text-xs text-blue-400 hover:text-blue-300 self-start"
			onclick={addSourcePath}
		>
			+ Add source image
		</button>
	</div>

	<!-- Target path -->
	<PathInput
		label="Target image"
		value={targetPath}
		filterExtensions={imageExtensions}
		onchange={(v) => (targetPath = v)}
	/>

	<!-- Output path -->
	<PathInput
		label="Output path"
		value={outputPath}
		filterExtensions={imageExtensions}
		onchange={(v) => (outputPath = v)}
	/>

	<!-- Options -->
	<OptionsBar
		{enhance}
		{mouthMask}
		onEnhanceChange={(v) => (enhance = v)}
		onMouthMaskChange={(v) => (mouthMask = v)}
	/>

	<!-- Detect button -->
	<button
		class="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:text-gray-400 text-white rounded text-sm font-medium self-start"
		onclick={handleDetect}
		disabled={detecting}
	>
		{detecting ? 'Detecting...' : 'Detect faces'}
	</button>

	<!-- Error -->
	{#if error}
		<div class="text-sm text-red-400 bg-red-900/20 px-4 py-2 rounded">{error}</div>
	{/if}

	<!-- Face mapper -->
	{#if detection}
		<div class="border-t border-gray-800 pt-4">
			<FaceMapper
				sourceFaces={detection.source_faces}
				targetItems={detection.target_faces}
				mode="image"
				onMappingsChange={(m) => (mappings = m)}
			/>
		</div>

		<!-- Swap button -->
		<button
			class="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 disabled:text-gray-400 text-white rounded text-sm font-medium self-start"
			onclick={handleSwap}
			disabled={swapping || mappings.length === 0}
		>
			{swapping ? 'Swapping...' : 'Swap faces'}
		</button>
	{/if}

	<!-- Result -->
	{#if swapResult}
		<div class="border-t border-gray-800 pt-4">
			<div class="text-sm text-green-400">
				Swap completed in {swapResult.elapsed_s.toFixed(1)}s. {swapResult.faces_swapped} face(s)
				swapped.
			</div>
			<div class="text-xs text-gray-500 mt-1">Output: {swapResult.output_path}</div>
		</div>
	{/if}
</div>
