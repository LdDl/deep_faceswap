<script>
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';

	onMount(() => {
		goto('/images', { replaceState: true });
	});
</script>
