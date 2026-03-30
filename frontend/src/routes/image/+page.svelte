<script>
	import { tick } from 'svelte';
	import SourcePathList from '$lib/components/SourcePathList.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import MediaPreview from '$lib/components/MediaPreview.svelte';
	import FaceMapper from '$lib/components/FaceMapper.svelte';
	import OptionsBar from '$lib/components/OptionsBar.svelte';
	import ProgressBar from '$lib/components/ProgressBar.svelte';
	import StepIndicator from '$lib/components/StepIndicator.svelte';
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

	/** @type {HTMLElement|undefined} */
	let mappingSection = $state();
	/** @type {HTMLElement|undefined} */
	let resultSection = $state();

	const steps = [
		{ label: 'Setup' },
		{ label: 'Detect' },
		{ label: 'Map' },
		{ label: 'Swap' },
		{ label: 'Result' }
	];

	let currentStep = $derived.by(() => {
		if (swapResult) return 4;
		if (swapping) return 3;
		if (detection && mappings.length > 0) return 3;
		if (detection) return 2;
		if (detecting) return 1;
		return 0;
	});

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

			await tick();
			mappingSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
		} catch (/** @type {any} */ e) {
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

			await tick();
			resultSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
		} catch (/** @type {any} */ e) {
			error = e.error_text || e.message || 'Swap failed';
		} finally {
			swapping = false;
		}
	}

	const imageExtensions = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];
</script>

<div class="flex flex-col gap-5">
	<!-- Step indicator -->
	<StepIndicator {steps} {currentStep} />

	<!-- Input configuration card -->
	<section class="rounded-xl border border-border bg-surface-1 p-4 sm:p-5 flex flex-col" aria-busy={detecting}>
		<!-- SOURCE sub-section -->
		<div class="flex flex-col gap-3 pb-4">
			<span class="text-xs font-semibold uppercase tracking-wider text-text-muted">Source</span>
			<SourcePathList bind:paths={sourcePaths} filterExtensions={imageExtensions} storageKey="source" />
		</div>

		<!-- TARGET sub-section -->
		<div class="flex flex-col gap-3 border-t border-border pt-4 pb-4">
			<span class="text-xs font-semibold uppercase tracking-wider text-text-muted">Target</span>
			<PathInput
				value={targetPath}
				placeholder="/path/to/target.jpg"
				filterExtensions={imageExtensions}
				storageKey="target"
				onchange={(/** @type {string} */ v) => (targetPath = v)}
			/>
			<!-- Target preview — animate in with grid-expand -->
			<div class="grid-expand {targetPath.trim() ? 'open' : ''}">
				<div>
					{#if targetPath.trim()}
						<div class="w-full max-w-sm pt-1">
							<MediaPreview path={targetPath} />
						</div>
					{/if}
				</div>
			</div>
		</div>

		<!-- OUTPUT sub-section -->
		<div class="flex flex-col gap-3 border-t border-border pt-4 pb-4">
			<span class="text-xs font-semibold uppercase tracking-wider text-text-muted">Output</span>
			<PathInput
				value={outputPath}
				placeholder="/path/to/output.jpg"
				filterExtensions={imageExtensions}
				storageKey="output"
				saveMode={true}
				defaultFilename="out.jpg"
				onchange={(/** @type {string} */ v) => (outputPath = v)}
			/>
		</div>

		<!-- OPTIONS sub-section -->
		<div class="flex flex-col gap-3 border-t border-border pt-4 pb-4">
			<span class="text-xs font-semibold uppercase tracking-wider text-text-muted">Options</span>
			<OptionsBar
				{enhance}
				{mouthMask}
				onEnhanceChange={(/** @type {boolean} */ v) => (enhance = v)}
				onMouthMaskChange={(/** @type {boolean} */ v) => (mouthMask = v)}
			/>
		</div>

		<!-- Detect button — inside card -->
		<div class="border-t border-border pt-4">
			<button
				type="button"
				class="w-full sm:w-auto px-5 py-2.5 bg-accent hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed
					   text-white rounded-lg text-sm font-medium transition-colors shadow-sm
					   focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1"
				onclick={handleDetect}
				disabled={detecting}
			>
				{detecting ? 'Detecting...' : 'Detect faces'}
			</button>
		</div>
	</section>

	<!-- Detect progress -->
	<!-- Detect progress -->
	<div class="grid-expand {detecting ? 'open' : ''}">
		<div>
			{#if detecting}
				<section class="rounded-xl border border-border bg-surface-1 p-4 sm:p-5">
					<ProgressBar progress={{ stage: 'detecting_faces', current: 0, total: 0 }} />
				</section>
			{/if}
		</div>
	</div>

	<!-- Error -->
	<div class="grid-expand {error ? 'open' : ''}">
		<div>
			{#if error}
				<div role="alert" aria-live="assertive" class="flex items-start gap-3 text-sm text-danger bg-danger/10 border border-danger/20 px-4 py-3 rounded-lg">
					<span class="shrink-0 mt-0.5 font-bold">!</span>
					<span class="flex-1 min-w-0">{error}</span>
					<button
						type="button"
						class="shrink-0 w-10 h-10 sm:w-8 sm:h-8 flex items-center justify-center text-danger/60 hover:text-danger
							   hover:bg-danger/10 rounded-md transition-colors
							   focus-visible:ring-2 focus-visible:ring-danger/50"
						onclick={() => (error = '')}
						aria-label="Dismiss error"
					>&times;</button>
				</div>
			{/if}
		</div>
	</div>

	<!-- Face mapping card -->
	<div class="grid-expand {detection ? 'open' : ''}">
		<div>
			{#if detection}
				<section
					bind:this={mappingSection}
					class="rounded-xl border border-border bg-surface-1 p-4 sm:p-5 flex flex-col gap-4"
					aria-busy={swapping}
				>
					<FaceMapper
						sourceFaces={detection.source_faces}
						targetItems={detection.target_faces}
						mode="image"
						initialAutoMap={true}
						onMappingsChange={(/** @type {FaceMapping[]} */ m) => (mappings = m)}
					/>

					<!-- Swap button — inside mapping card -->
					<div class="border-t border-border pt-4">
						<button
							type="button"
							class="w-full sm:w-auto px-5 py-2.5 bg-success hover:bg-success-hover disabled:opacity-50 disabled:cursor-not-allowed
								   text-white rounded-lg text-sm font-medium transition-colors shadow-sm
								   focus-visible:ring-2 focus-visible:ring-success/50 focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1"
							onclick={handleSwap}
							disabled={swapping || mappings.length === 0}
						>
							{swapping ? 'Swapping...' : 'Swap faces'}
						</button>
					</div>
				</section>
			{/if}
		</div>
	</div>

	<!-- Result card -->
	<div class="grid-expand {swapResult ? 'open' : ''}">
		<div>
			{#if swapResult}
				<section
					bind:this={resultSection}
					class="rounded-xl border border-success/30 bg-surface-1 p-4 sm:p-5 flex flex-col gap-3"
				>
					<div class="flex items-center gap-2 text-sm text-success font-medium">
						Swap completed in {swapResult.elapsed_s.toFixed(1)}s — {swapResult.faces_swapped} face(s) swapped
					</div>
					<div class="text-xs text-text-muted break-all">Output: {swapResult.output_path}</div>
					<div class="w-full max-w-sm">
						<MediaPreview path={swapResult.output_path} />
					</div>
				</section>
			{/if}
		</div>
	</div>
</div>
