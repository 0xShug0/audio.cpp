<script lang="ts">
  import { Users, UserRound } from '@lucide/svelte';
  import type { SavedVoice } from '$lib/library';
  import VoiceLibrary from './VoiceLibrary.svelte';
  import SpeakerLibrary from '$lib/apps/tts/SpeakerLibrary.svelte';
  export let onchange: () => void | Promise<void>;
  export let ontext: (voice: SavedVoice) => void | Promise<void>;
  let section: 'voices' | 'speakers' = 'voices';
  let voices: VoiceLibrary | null = null;
  let speakers: SpeakerLibrary | null = null;
  export function canClose() {
    return section === 'voices' ? voices?.canClose() ?? true : speakers?.canClose() ?? true;
  }
  function switchSection(next: typeof section) {
    if (section !== next && canClose()) section = next;
  }
</script>

<nav class="library-tabs" aria-label="Library collections">
  <button class:active={section === 'voices'} aria-pressed={section === 'voices'} on:click={() => switchSection('voices')}><Users size={16} />Voices</button>
  <button class:active={section === 'speakers'} aria-pressed={section === 'speakers'} on:click={() => switchSection('speakers')}><UserRound size={16} />Speakers</button>
</nav>
{#if section === 'voices'}
  <VoiceLibrary bind:this={voices} {onchange} {ontext} />
{:else}
  <SpeakerLibrary bind:this={speakers} />
{/if}

<style>
  .library-tabs { display: flex; gap: 8px; padding-bottom: 16px; margin-bottom: 16px; border-bottom: 1px solid var(--line); }
  button { display: inline-flex; align-items: center; gap: 8px; min-height: 38px; }
  button.active { color: var(--cyan); border-color: var(--cyan); background: var(--active-bg); }
</style>
