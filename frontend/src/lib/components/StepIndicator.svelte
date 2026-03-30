<script>
	/**
	 * @type {{ steps: {label: string}[], currentStep: number }}
	 */
	let { steps, currentStep } = $props();
</script>

<div class="flex items-center w-full" role="navigation" aria-label="Workflow progress">
	{#each steps as step, i}
		{@const isActive = i <= currentStep}
		{@const isCurrent = i === currentStep}
		<!-- Step dot + label -->
		<div class="flex flex-col items-center gap-1 shrink-0">
			<div
				class="w-2.5 h-2.5 rounded-full transition-colors duration-200
					   {isCurrent ? 'bg-accent ring-4 ring-accent/20' : isActive ? 'bg-accent' : 'bg-surface-3'}"
			></div>
			<span
				class="text-[10px] sm:text-xs transition-colors duration-200
					   {isCurrent ? 'text-accent font-medium' : isActive ? 'text-text-secondary' : 'text-text-muted'}"
			>{step.label}</span>
		</div>
		<!-- Connector line -->
		{#if i < steps.length - 1}
			<div class="flex-1 h-px mx-1 sm:mx-2 transition-colors duration-200 {i < currentStep ? 'bg-accent' : 'bg-surface-3'}"></div>
		{/if}
	{/each}
</div>
