<script>
	import { tick, onMount, onDestroy } from 'svelte';
	import SourcePathList from '$lib/components/SourcePathList.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import MediaPreview from '$lib/components/MediaPreview.svelte';
	import FaceMapper from '$lib/components/FaceMapper.svelte';
	import OptionsBar from '$lib/components/OptionsBar.svelte';
	import ProgressBar from '$lib/components/ProgressBar.svelte';
	import StepIndicator from '$lib/components/StepIndicator.svelte';
	import { analyzeVideo, parseAnalyzeResult, swapVideo } from '$lib/stores/video.js';
	import { getJobStatus } from '$lib/stores/jobs.js';
	import { ClusterMapping } from '$lib/cluster_mapping.js';
	import { VideoAnalyzeResponse } from '$lib/stores/video.js';
	import { JobState } from '$lib/stores/jobs.js';

	const FORM_KEY = 'video_form';

	// Restore form state from localStorage
	function loadForm() {
		try {
			const raw = localStorage.getItem(FORM_KEY);
			if (!raw) return;
			const f = JSON.parse(raw);
			if (f.sourcePaths?.length) sourcePaths = f.sourcePaths;
			if (f.targetVideoPath) targetVideoPath = f.targetVideoPath;
			if (f.outputPath) outputPath = f.outputPath;
			if (f.tmpDir) tmpDir = f.tmpDir;
			if (f.enhance !== undefined) enhance = f.enhance;
			if (f.mouthMask !== undefined) mouthMask = f.mouthMask;
		} catch {
			// ignore
		}
	}

	function saveForm() {
		localStorage.setItem(FORM_KEY, JSON.stringify({
			sourcePaths, targetVideoPath, outputPath, tmpDir, enhance, mouthMask
		}));
	}

	let sourcePaths = $state(['']);
	let targetVideoPath = $state('');
	let outputPath = $state('');
	let tmpDir = $state('');
	let enhance = $state(true);
	let mouthMask = $state(false);
	let advancedOpen = $state(false);
	let formLoaded = $state(false);

	// Auto-generate tmp_dir when target video path changes (user action, not restore)
	let prevTargetVideo = '';
	$effect(() => {
		if (!formLoaded) return;
		if (targetVideoPath.trim() && targetVideoPath !== prevTargetVideo) {
			prevTargetVideo = targetVideoPath;
			const lastSlash = targetVideoPath.lastIndexOf('/');
			const dir = lastSlash > 0 ? targetVideoPath.substring(0, lastSlash) : '.';
			tmpDir = dir + '/tmp_frames';
		}
	});

	// Persist form whenever values change (after initial load), debounced
	/** @type {ReturnType<typeof setTimeout>|null} */
	let saveTimer = null;
	$effect(() => {
		if (!formLoaded) return;
		// Access all reactive values to track them
		sourcePaths; targetVideoPath; outputPath; tmpDir; enhance; mouthMask;
		if (saveTimer) clearTimeout(saveTimer);
		saveTimer = setTimeout(saveForm, 500);
	});

	/** @type {VideoAnalyzeResponse|null} */
	let analysis = $state(null);
	/** @type {ClusterMapping[]} */
	let clusterMappings = $state([]);

	/** @type {JobState|null} — analyze job */
	let analyzeJob = $state(null);
	/** @type {JobState|null} — swap job */
	let swapJob = $state(null);

	let starting = $state(false);
	let error = $state('');

	/** @type {ReturnType<typeof setInterval>|null} */
	let analyzePollTimer = null;
	/** @type {ReturnType<typeof setInterval>|null} */
	let swapPollTimer = null;

	/** @type {HTMLElement|undefined} */
	let mappingSection = $state();
	/** @type {HTMLElement|undefined} */
	let resultSection = $state();

	const ANALYZE_JOB_KEY = 'video_analyze_job';
	const SWAP_JOB_KEY = 'video_swap_job';

	const steps = [
		{ label: 'Setup' },
		{ label: 'Analyze' },
		{ label: 'Map' },
		{ label: 'Swap' },
		{ label: 'Result' }
	];

	let analyzing = $derived(
		analyzeJob != null &&
			(analyzeJob.status === 'queued' || analyzeJob.status === 'running')
	);

	let currentStep = $derived.by(() => {
		if (swapJob?.status === 'completed') return 4;
		if (swapJob?.status === 'running' || swapJob?.status === 'queued' || starting) return 3;
		if (analysis && clusterMappings.length > 0) return 3;
		if (analysis) return 2;
		if (analyzing) return 1;
		return 0;
	});

	// --- localStorage helpers ---

	/**
	 * @param {string} key
	 * @param {string} jobId
	 * @param {object} [extra]
	 */
	function saveJob(key, jobId, extra) {
		localStorage.setItem(key, JSON.stringify({ job_id: jobId, ...extra }));
	}

	/** @param {string} key */
	function clearJob(key) {
		localStorage.removeItem(key);
	}

	const MAX_POLL_FAILURES = 3;

	/**
	 * @param {string} jobId
	 * @param {'analyze'|'swap'} type
	 */
	function startPolling(jobId, type) {
		const isAnalyze = type === 'analyze';
		// Clear existing timer for this type
		if (isAnalyze && analyzePollTimer) clearInterval(analyzePollTimer);
		if (!isAnalyze && swapPollTimer) clearInterval(swapPollTimer);

		let failures = 0;
		const timer = setInterval(async () => {
			try {
				const state = await getJobStatus(jobId);
				failures = 0;
				if (isAnalyze) {
					analyzeJob = state;
				} else {
					swapJob = state;
				}
				if (state.status === 'completed' || state.status === 'failed') {
					clearInterval(timer);
					if (isAnalyze) analyzePollTimer = null;
					else swapPollTimer = null;
				}
			} catch {
				failures++;
				if (failures >= MAX_POLL_FAILURES) {
					clearInterval(timer);
					if (isAnalyze) analyzePollTimer = null;
					else swapPollTimer = null;
					error = 'Lost connection to server. Please check the backend and retry.';
				}
			}
		}, 2000);

		if (isAnalyze) analyzePollTimer = timer;
		else swapPollTimer = timer;
	}

	$effect(() => {
		if (analyzeJob?.status === 'completed' && analyzeJob.result?.data && !analysis) {
			try {
				analysis = parseAnalyzeResult(analyzeJob.result.data);
				clearJob(ANALYZE_JOB_KEY);
				tick().then(() => {
					mappingSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
				});
			} catch (/** @type {any} */ e) {
				error = 'Failed to parse analysis result: ' + (e.message || e);
			}
		}
	});

	$effect(() => {
		if (analyzeJob?.status === 'failed') {
			error = analyzeJob.error || 'Analysis failed';
			clearJob(ANALYZE_JOB_KEY);
			if (!localStorage.getItem(SWAP_JOB_KEY)) {
				localStorage.removeItem(FORM_KEY);
			}
		}
	});

	$effect(() => {
		if (swapJob && (swapJob.status === 'completed' || swapJob.status === 'failed')) {
			clearJob(SWAP_JOB_KEY);
			// Clean up form persistence when all jobs are done
			if (!localStorage.getItem(ANALYZE_JOB_KEY)) {
				localStorage.removeItem(FORM_KEY);
			}
		}
	});

	// Auto-scroll to result when swap job completes
	$effect(() => {
		if (swapJob?.status === 'completed') {
			tick().then(() => {
				resultSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
			});
		}
	});

	async function tryRecoverJobs() {
		let anyRecovered = false;

		// Recover analyze job
		const analyzeRaw = localStorage.getItem(ANALYZE_JOB_KEY);
		if (analyzeRaw) {
			try {
				const saved = JSON.parse(analyzeRaw);
				if (saved.job_id) {
					const state = await getJobStatus(saved.job_id);
					analyzeJob = state;
					anyRecovered = true;
					if (state.status !== 'completed' && state.status !== 'failed') {
						startPolling(saved.job_id, 'analyze');
					}
				}
			} catch {
				clearJob(ANALYZE_JOB_KEY);
			}
		}

		// Recover swap job
		const swapRaw = localStorage.getItem(SWAP_JOB_KEY);
		if (swapRaw) {
			try {
				const saved = JSON.parse(swapRaw);
				if (saved.job_id) {
					if (saved.output_path && !outputPath.trim()) {
						outputPath = saved.output_path;
					}
					const state = await getJobStatus(saved.job_id);
					swapJob = state;
					anyRecovered = true;
					if (state.status !== 'completed' && state.status !== 'failed') {
						startPolling(saved.job_id, 'swap');
					}
				}
			} catch {
				clearJob(SWAP_JOB_KEY);
			}
		}

		// If no jobs were recovered (server restarted, jobs lost), reset form to clean state
		if (!anyRecovered && (analyzeRaw || swapRaw)) {
			localStorage.removeItem(FORM_KEY);
			sourcePaths = [''];
			targetVideoPath = '';
			outputPath = '';
			tmpDir = '';
			enhance = true;
			mouthMask = false;
		}
	}

	onMount(() => {
		// Only restore form if there's an active job to recover
		const hasActiveJob =
			localStorage.getItem(ANALYZE_JOB_KEY) || localStorage.getItem(SWAP_JOB_KEY);
		if (hasActiveJob) {
			loadForm();
			prevTargetVideo = targetVideoPath;
		}
		formLoaded = true;
		tryRecoverJobs();
	});

	onDestroy(() => {
		if (analyzePollTimer) {
			clearInterval(analyzePollTimer);
			analyzePollTimer = null;
		}
		if (swapPollTimer) {
			clearInterval(swapPollTimer);
			swapPollTimer = null;
		}
	});

	async function handleAnalyze() {
		const paths = sourcePaths.filter((p) => p.trim());
		if (paths.length === 0 || !targetVideoPath.trim()) {
			error = 'Provide at least one source path and a target video path';
			return;
		}

		error = '';
		analysis = null;
		swapJob = null;
		clusterMappings = [];

		try {
			const res = await analyzeVideo(paths, targetVideoPath, tmpDir || undefined);
			analyzeJob = new JobState(res.job_id, 'queued', { stage: 'queued', current: 0, total: 0 });
			startPolling(res.job_id, 'analyze');
			saveJob(ANALYZE_JOB_KEY, res.job_id);
		} catch (/** @type {any} */ e) {
			error = e.error_text || e.message || 'Failed to start analysis';
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

			swapJob = new JobState(res.job_id, 'queued', { stage: 'queued', current: 0, total: 0 });
			startPolling(res.job_id, 'swap');
			saveJob(SWAP_JOB_KEY, res.job_id, { output_path: outputPath });
		} catch (/** @type {any} */ e) {
			error = e.error_text || e.message || 'Failed to start swap';
		} finally {
			starting = false;
		}
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
				type="button"
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
				type="button"
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

	<!-- Analyze progress card -->
	<div class="grid-expand {analyzing && analyzeJob ? 'open' : ''}">
		<div>
			{#if analyzing && analyzeJob}
				<section class="rounded-xl border border-border bg-surface-1 p-4 sm:p-5 flex flex-col gap-3">
					<ProgressBar progress={analyzeJob.progress} />
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

	<!-- Analysis results + mapper card -->
	<div class="grid-expand {analysis ? 'open' : ''}">
		<div>
			{#if analysis}
				<section
					bind:this={mappingSection}
					class="rounded-xl border border-border bg-surface-1 p-4 sm:p-5 flex flex-col gap-4"
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

					{#if !swapJob || swapJob.status === 'failed'}
						<!-- Start swap button — inside mapping card -->
						<div class="border-t border-border pt-4">
							<button
								type="button"
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
		</div>
	</div>

	<!-- Swap job progress / result card -->
	<div class="grid-expand {swapJob ? 'open' : ''}">
		<div>
			{#if swapJob}
				<section
					bind:this={resultSection}
					class="rounded-xl border {swapJob.status === 'completed' ? 'border-success/30' : swapJob.status === 'failed' ? 'border-danger/30' : 'border-border'}
						   bg-surface-1 p-4 sm:p-5 flex flex-col gap-3"
				>
					{#if swapJob.status === 'queued' || swapJob.status === 'running'}
						<ProgressBar progress={swapJob.progress} />
					{/if}

					{#if swapJob.status === 'completed'}
						<div class="flex items-center gap-2 text-sm text-success font-medium">
							Video swap completed
						</div>
						{#if swapJob.result?.output_path}
							<div class="text-xs text-text-muted break-all">Output: {swapJob.result.output_path}</div>
							<div class="w-full max-w-sm">
								<MediaPreview path={swapJob.result.output_path} type="video" />
							</div>
						{/if}
					{/if}

					{#if swapJob.status === 'failed'}
						<div class="text-sm text-danger">
							Video swap failed{swapJob.error ? `: ${swapJob.error}` : ''}
						</div>
					{/if}
				</section>
			{/if}
		</div>
	</div>
</div>
