<script>
	import { onMount } from 'svelte';
	import { fetchHealth, healthData } from '$lib/stores/health.js';

	/**
	 * @typedef {object} Props
	 * @property {boolean} enhance
	 * @property {boolean} mouthMask
	 * @property {(value: boolean) => void} onEnhanceChange
	 * @property {(value: boolean) => void} onMouthMaskChange
	 * @property {boolean} [preserveForeground]
	 * @property {number} [foregroundMargin]
	 * @property {number} [foregroundBlur]
	 * @property {(value: boolean) => void} onPreserveForegroundChange
	 * @property {(value: number) => void} onForegroundMarginChange
	 * @property {(value: number) => void} onForegroundBlurChange
	 */
	/** @type {Props} */
	let {
		enhance, mouthMask, onEnhanceChange, onMouthMaskChange,
		preserveForeground = false, foregroundMargin = 2, foregroundBlur = 3,
		onPreserveForegroundChange, onForegroundMarginChange, onForegroundBlurChange
	} = $props();

	let foregroundSettingsOpen = $state(false);
	let foregroundAvailable = $derived($healthData?.foreground_available === true);

	onMount(() => {
		fetchHealth();
	});
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
	<label class="flex items-center gap-2.5 select-none group {foregroundAvailable || preserveForeground ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}" title="Keep hands, food and other objects in front of the face">
		<span class="relative inline-flex items-center justify-center w-4 h-4">
			<input
				type="checkbox"
				class="peer appearance-none w-4 h-4 rounded border border-border bg-surface-2
					   checked:bg-accent checked:border-accent transition-colors cursor-pointer disabled:cursor-not-allowed
					   focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1 focus-visible:ring-offset-surface-1"
				checked={preserveForeground}
				disabled={!foregroundAvailable && !preserveForeground}
				onchange={(e) => onPreserveForegroundChange(e.currentTarget.checked)}
			/>
			<svg class="absolute w-2.5 h-2.5 text-white pointer-events-none opacity-0 peer-checked:opacity-100 transition-opacity" viewBox="0 0 12 12" fill="none">
				<path d="M2 6l3 3 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
			</svg>
		</span>
		<span class="text-sm text-text-primary">Preserve foreground objects</span>
	</label>
</div>

{#if !foregroundAvailable}
	<p class="text-xs text-text-muted" role="status">
		{!$healthData ? 'Checking foreground preservation availability...' : $healthData.status === 'unavailable' ? 'Could not check foreground preservation availability. Reload to retry.' : 'Foreground preservation is unavailable on this server.'}
	</p>
{/if}

{#if preserveForeground && foregroundAvailable}
	<button
		type="button"
		class="flex items-center gap-1.5 py-2 text-xs text-text-muted hover:text-text-secondary transition-colors self-start
			   focus-visible:ring-2 focus-visible:ring-accent/50 rounded-md"
		aria-expanded={foregroundSettingsOpen}
		onclick={() => (foregroundSettingsOpen = !foregroundSettingsOpen)}
	>
		<svg class="w-3 h-3 transition-transform duration-150 {foregroundSettingsOpen ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
			<path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
		</svg>
		Foreground settings
	</button>
	<div class="grid-expand {foregroundSettingsOpen ? 'open' : ''}">
		<div inert={!foregroundSettingsOpen}>
			<div class="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
				<label class="flex flex-col gap-2 text-xs text-text-secondary" title="Increase to protect a wider area around objects">
					<span class="flex justify-between gap-3">Protection margin <span class="font-mono tabular-nums">{foregroundMargin}</span></span>
					<input type="range" min="0" max="16" step="1" value={foregroundMargin} class="w-full accent-accent focus-visible:ring-2 focus-visible:ring-accent/50 rounded-md" oninput={(e) => onForegroundMarginChange(e.currentTarget.valueAsNumber)} />
				</label>
				<label class="flex flex-col gap-2 text-xs text-text-secondary" title="Soften the transition between the preserved object and the swapped face">
					<span class="flex justify-between gap-3">Edge softness <span class="font-mono tabular-nums">{foregroundBlur}</span></span>
					<input type="range" min="0" max="16" step="1" value={foregroundBlur} class="w-full accent-accent focus-visible:ring-2 focus-visible:ring-accent/50 rounded-md" oninput={(e) => onForegroundBlurChange(e.currentTarget.valueAsNumber)} />
				</label>
			</div>
		</div>
	</div>
{/if}
