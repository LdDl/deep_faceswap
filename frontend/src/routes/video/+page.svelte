<script>
	import { tick, onMount, onDestroy } from 'svelte';
	import SourcePathList from '$lib/components/SourcePathList.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import MediaPreview from '$lib/components/MediaPreview.svelte';
	import FaceMapper from '$lib/components/FaceMapper.svelte';
	import OptionsBar from '$lib/components/OptionsBar.svelte';
	import ProgressBar from '$lib/components/ProgressBar.svelte';
	import StepIndicator from '$lib/components/StepIndicator.svelte';
	import { analyzeVideo, swapVideo } from '$lib/stores/video.js';
	import { getJobStatus } from '$lib/stores/jobs.js';
	import { ClusterMapping } from '$lib/cluster_mapping.js';
	import { VideoAnalyzeResponse } from '$lib/stores/video.js';
	import { JobState } from '$lib/stores/jobs.js';

	let sourcePaths = $state(['']);
	let targetVideoPath = $state('');
	let outputPath = $state('');
	let tmpDir = $state('');
	let enhance = $state(true);
	let mouthMask = $state(false);
	let advancedOpen = $state(false);

	// Auto-generate tmp_dir from target video directory
	$effect(() => {
		if (targetVideoPath.trim()) {
			const lastSlash = targetVideoPath.lastIndexOf('/');
			const dir = lastSlash > 0 ? targetVideoPath.substring(0, lastSlash) : '.';
			tmpDir = dir + '/tmp_frames';
		}
	});

	/** @type {VideoAnalyzeResponse|null} */
	let analysis = $state(null);
	/** @type {ClusterMapping[]} */
	let clusterMappings = $state([]);
	/** @type {JobState|null} */
	let job = $state(null);

	let analyzing = $state(false);
	let starting = $state(false);
	let error = $state('');

	/** @type {ReturnType<typeof setInterval>|null} */
	let pollTimer = null;

	/** @type {HTMLElement|undefined} */
	let mappingSection = $state();
	/** @type {HTMLElement|undefined} */
	let resultSection = $state();

	const ACTIVE_JOB_KEY = 'video_active_job';

	const steps = [
		{ label: 'Setup' },
		{ label: 'Analyze' },
		{ label: 'Map' },
		{ label: 'Swap' },
		{ label: 'Result' }
	];

	let currentStep = $derived.by(() => {
		if (job?.status === 'completed') return 4;
		if (job?.status === 'running' || job?.status === 'queued' || starting) return 3;
		if (analysis && clusterMappings.length > 0) return 3;
		if (analysis) return 2;
		if (analyzing) return 1;
		return 0;
	});

	/**
	 * Save active job info to localStorage for session recovery.
	 * @param {string} jobId
	 * @param {string} outPath
	 */
	function saveActiveJob(jobId, outPath) {
		localStorage.setItem(
			ACTIVE_JOB_KEY,
			JSON.stringify({
				job_id: jobId,
				output_path: outPath
			})
		);
	}

	function clearActiveJob() {
		localStorage.removeItem(ACTIVE_JOB_KEY);
	}

	// Recover a previously active job on page load
	async function tryRecoverJob() {
		const raw = localStorage.getItem(ACTIVE_JOB_KEY);
		if (!raw) return;

		try {
			const saved = JSON.parse(raw);
			if (!saved.job_id) return;

			if (saved.output_path && !outputPath.trim()) {
				outputPath = saved.output_path;
			}

			const state = await getJobStatus(saved.job_id);
			job = state;

			if (state.status === 'completed' || state.status === 'failed') {
				clearActiveJob();
			} else {
				startPolling(saved.job_id);
			}
		} catch {
			clearActiveJob();
		}
	}

	$effect(() => {
		if (job && (job.status === 'completed' || job.status === 'failed')) {
			clearActiveJob();
		}
	});

	// Auto-scroll to result when job completes
	$effect(() => {
		if (job?.status === 'completed') {
			tick().then(() => {
				resultSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
			});
		}
	});

	onMount(() => {
		tryRecoverJob();
	});

	onDestroy(() => {
		if (pollTimer) {
			clearInterval(pollTimer);
			pollTimer = null;
		}
	});

	async function handleAnalyze() {
		const paths = sourcePaths.filter((p) => p.trim());
		if (paths.length === 0 || !targetVideoPath.trim()) {
			error = 'Provide at least one source path and a target video path';
			return;
		}

		error = '';
		analyzing = true;
		analysis = null;
		job = null;
		clusterMappings = [];

		try {
			analysis = await analyzeVideo(paths, targetVideoPath, tmpDir || undefined);

			await tick();
			mappingSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
		} catch (/** @type {any} */ e) {
			error = e.error_text || e.message || 'Analysis failed';
		} finally {
			analyzing = false;
		}
	}

	async function handleStartSwap() {
		if (!analysis || clusterMappings.length === 0) {
			error = 'Analyze video and create mappings first';
			return;
		}
		if (!outputPath.trim()) {
			error = 'Provide an output path';
			return;
		}

		error = '';
		starting = true;

		try {
			const res = await swapVideo(
				analysis.session_id,
				sourcePaths.filter((p) => p.trim()),
				targetVideoPath,
				outputPath,
				clusterMappings,
				enhance,
				mouthMask,
				tmpDir || undefined
			);

			job = new JobState(res.job_id, 'queued', { stage: 'queued', current: 0, total: 0 });

			startPolling(res.job_id);
			saveActiveJob(res.job_id, outputPath);
		} catch (/** @type {any} */ e) {
			error = e.error_text || e.message || 'Failed to start swap';
		} finally {
			starting = false;
		}
	}

	/** @param {string} jobId */
	function startPolling(jobId) {
		if (pollTimer) clearInterval(pollTimer);
		pollTimer = setInterval(async () => {
			try {
				job = await getJobStatus(jobId);
				if (job.status === 'completed' || job.status === 'failed') {
					if (pollTimer) {
						clearInterval(pollTimer);
						pollTimer = null;
					}
				}
			} catch {
				// Ignore polling errors
			}
		}, 2000);
	}

	const imageExtensions = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];
	const videoExtensions = ['mp4', 'avi', 'mkv', 'mov', 'webm'];
</script>

<div class="flex flex-col gap-5">
	<!-- Step indicator -->
	<StepIndicator {steps} {currentStep} />

	<!-- Input configuration card -->
	<section class="rounded-xl border border-border bg-surface-1 p-4 sm:p-5 flex flex-col" aria-busy={analyzing}>
		<!-- SOURCE sub-section -->
		<div class="flex flex-col gap-3 pb-4">
			<span class="text-xs font-semibold uppercase tracking-wider text-text-muted">Source</span>
			<SourcePathList bind:paths={sourcePaths} filterExtensions={imageExtensions} storageKey="source" />
		</div>

		<!-- TARGET sub-section -->
		<div class="flex flex-col gap-3 border-t border-border pt-4 pb-4">
			<span class="text-xs font-semibold uppercase tracking-wider text-text-muted">Target</span>
			<PathInput
				value={targetVideoPath}
				placeholder="/path/to/video.mp4"
				filterExtensions={videoExtensions}
				storageKey="target_video"
				onchange={(/** @type {string} */ v) => (targetVideoPath = v)}
			/>
			<!-- Target preview — animate in with grid-expand -->
			<div class="grid-expand {targetVideoPath.trim() ? 'open' : ''}">
				<div>
					{#if targetVideoPath.trim()}
						<div class="w-full max-w-sm pt-1">
							<MediaPreview path={targetVideoPath} type="video" />
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
				placeholder="/path/to/output.mp4"
				filterExtensions={videoExtensions}
				storageKey="output_video"
				saveMode={true}
				defaultFilename="out.mp4"
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

			<!-- Advanced toggle for temp directory -->
			<button
				class="flex items-center gap-1.5 py-2 text-xs text-text-muted hover:text-text-secondary transition-colors self-start
				   focus-visible:ring-2 focus-visible:ring-accent/50 rounded-md"
				onclick={() => (advancedOpen = !advancedOpen)}
			>
				<svg
					class="w-3 h-3 transition-transform duration-150 {advancedOpen ? 'rotate-90' : ''}"
					fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
				</svg>
				Advanced
			</button>
			<div class="grid-expand {advancedOpen ? 'open' : ''}">
				<div>
					<div class="pt-2">
						<PathInput
							label="Temp directory (frames)"
							value={tmpDir}
							placeholder="/path/to/tmp_frames"
							storageKey="tmp_dir_video"
							onchange={(/** @type {string} */ v) => (tmpDir = v)}
						/>
					</div>
				</div>
			</div>
		</div>

		<!-- Analyze button — inside card -->
		<div class="border-t border-border pt-4">
			<button
				class="w-full sm:w-auto px-5 py-2.5 bg-accent hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed
					   text-white rounded-lg text-sm font-medium transition-colors shadow-sm
					   focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1"
				onclick={handleAnalyze}
				disabled={analyzing}
			>
				{analyzing ? 'Analyzing...' : 'Analyze video'}
			</button>
		</div>
	</section>

	<!-- Error -->
	{#if error}
		<div class="flex items-start gap-3 text-sm text-danger bg-danger/10 border border-danger/20 px-4 py-3 rounded-lg section-enter">
			<span class="shrink-0 mt-0.5 font-bold">!</span>
			<span class="flex-1 min-w-0">{error}</span>
			<button
				class="shrink-0 w-10 h-10 sm:w-8 sm:h-8 flex items-center justify-center text-danger/60 hover:text-danger
					   hover:bg-danger/10 rounded-md transition-colors
					   focus-visible:ring-2 focus-visible:ring-danger/50"
				onclick={() => (error = '')}
				aria-label="Dismiss error"
			>&times;</button>
		</div>
	{/if}

	<!-- Analysis results + mapper card -->
	{#if analysis}
		<section
			bind:this={mappingSection}
			class="rounded-xl border border-border bg-surface-1 p-4 sm:p-5 flex flex-col gap-4 section-enter"
			aria-busy={starting}
		>
			<div class="text-sm text-text-secondary">
				{analysis.total_frames} frames, analyzed in {analysis.elapsed_s.toFixed(1)}s —
				{analysis.clusters.length} cluster(s) found
			</div>

			<FaceMapper
				sourceFaces={analysis.source_faces}
				targetItems={analysis.clusters}
				mode="video"
				onMappingsChange={(/** @type {ClusterMapping[]} */ m) => (clusterMappings = m)}
			/>

			{#if !job}
				<!-- Start swap button — inside mapping card -->
				<div class="border-t border-border pt-4">
					<button
						class="w-full sm:w-auto px-5 py-2.5 bg-success hover:bg-success-hover disabled:opacity-50 disabled:cursor-not-allowed
							   text-white rounded-lg text-sm font-medium transition-colors shadow-sm
							   focus-visible:ring-2 focus-visible:ring-success/50 focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1"
						onclick={handleStartSwap}
						disabled={starting || clusterMappings.length === 0}
					>
						{starting ? 'Starting...' : 'Start swap'}
					</button>
				</div>
			{/if}
		</section>
	{/if}

	<!-- Job progress / result card -->
	{#if job}
		<section
			bind:this={resultSection}
			class="rounded-xl border {job.status === 'completed' ? 'border-success/30' : job.status === 'failed' ? 'border-danger/30' : 'border-border'}
				   bg-surface-1 p-4 sm:p-5 flex flex-col gap-3 section-enter"
		>
			{#if job.status === 'queued' || job.status === 'running'}
				<ProgressBar progress={job.progress} />
			{/if}

			{#if job.status === 'completed'}
				<div class="flex items-center gap-2 text-sm text-success font-medium">
					Video swap completed
				</div>
				{#if job.result}
					<div class="text-xs text-text-muted break-all">Output: {job.result.output_path}</div>
					<div class="w-full max-w-sm">
						<MediaPreview path={job.result.output_path} type="video" />
					</div>
				{/if}
			{/if}

			{#if job.status === 'failed'}
				<div class="text-sm text-danger">
					Video swap failed{job.error ? `: ${job.error}` : ''}
				</div>
			{/if}
		</section>
	{/if}
</div>
