<script>
	import PathInput from '$lib/components/PathInput.svelte';
	import FaceMapper from '$lib/components/FaceMapper.svelte';
	import OptionsBar from '$lib/components/OptionsBar.svelte';
	import ProgressBar from '$lib/components/ProgressBar.svelte';
	import { analyzeVideo, swapVideo } from '$lib/stores/video.js';
	import { getJobStatus } from '$lib/stores/jobs.js';
	import { ClusterMapping } from '$lib/cluster_mapping.js';
	import { VideoAnalyzeResponse } from '$lib/stores/video.js';
	import { JobState } from '$lib/stores/jobs.js';

	let sourcePaths = $state(['']);
	let targetVideoPath = $state('');
	let outputPath = $state('');
	let enhance = $state(true);
	let mouthMask = $state(false);

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
			analysis = await analyzeVideo(paths, targetVideoPath);
		} catch (e) {
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
				mouthMask
			);

			job = {
				job_id: res.job_id,
				status: 'queued',
				progress: { stage: 'queued', current: 0, total: 0 }
			};

			startPolling(res.job_id);
		} catch (e) {
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

<div class="flex flex-col gap-6">
	<h2 class="text-xl font-semibold">Video swap</h2>

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

	<!-- Target video path -->
	<PathInput
		label="Target video"
		value={targetVideoPath}
		filterExtensions={videoExtensions}
		onchange={(v) => (targetVideoPath = v)}
	/>

	<!-- Output path -->
	<PathInput
		label="Output path"
		value={outputPath}
		filterExtensions={videoExtensions}
		onchange={(v) => (outputPath = v)}
	/>

	<!-- Options -->
	<OptionsBar
		{enhance}
		{mouthMask}
		onEnhanceChange={(v) => (enhance = v)}
		onMouthMaskChange={(v) => (mouthMask = v)}
	/>

	<!-- Analyze button -->
	<button
		class="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:text-gray-400 text-white rounded text-sm font-medium self-start"
		onclick={handleAnalyze}
		disabled={analyzing}
	>
		{analyzing ? 'Analyzing...' : 'Analyze video'}
	</button>

	<!-- Error -->
	{#if error}
		<div class="text-sm text-red-400 bg-red-900/20 px-4 py-2 rounded">{error}</div>
	{/if}

	<!-- Analysis results + mapper -->
	{#if analysis}
		<div class="border-t border-gray-800 pt-4 flex flex-col gap-3">
			<div class="text-sm text-gray-400">
				{analysis.total_frames} frames, analyzed in {analysis.elapsed_s.toFixed(1)}s,
				{analysis.clusters.length} cluster(s) found
			</div>

			<FaceMapper
				sourceFaces={analysis.source_faces}
				targetItems={analysis.clusters}
				mode="video"
				onMappingsChange={(m) => (clusterMappings = m)}
			/>

			<!-- Start swap button -->
			{#if !job}
				<button
					class="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 disabled:text-gray-400 text-white rounded text-sm font-medium self-start"
					onclick={handleStartSwap}
					disabled={starting || clusterMappings.length === 0}
				>
					{starting ? 'Starting...' : 'Start swap'}
				</button>
			{/if}
		</div>
	{/if}

	<!-- Job progress -->
	{#if job}
		<div class="border-t border-gray-800 pt-4 flex flex-col gap-3">
			{#if job.status === 'queued' || job.status === 'running'}
				<ProgressBar progress={job.progress} />
			{/if}

			{#if job.status === 'completed'}
				<div class="text-sm text-green-400">Video swap completed.</div>
				{#if job.result}
					<div class="text-xs text-gray-500">Output: {job.result.output_path}</div>
				{/if}
			{/if}

			{#if job.status === 'failed'}
				<div class="text-sm text-red-400">
					Video swap failed{job.error ? `: ${job.error}` : ''}
				</div>
			{/if}
		</div>
	{/if}
</div>
