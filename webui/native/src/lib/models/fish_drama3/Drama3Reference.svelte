<script lang="ts">
  import { FileAudio, X } from '@lucide/svelte';
  import MediaPreview from '$lib/MediaPreview.svelte';
  import type { SavedVoice } from '$lib/voices';

  export let id: string;
  export let file: File | null = null;
  export let transcript = '';
  export let busy = false;
  export let savedVoices: SavedVoice[] = [];
  let savedVoiceId = '';

  function useSavedVoice(value: string) {
    savedVoiceId = value;
    const voice = savedVoices.find((entry) => entry.id === value);
    if (!voice) return;
    file = new File([voice.audio], `${voice.name}.wav`, { type: voice.audio.type || 'audio/wav' });
    transcript = voice.transcript;
  }
</script>

<div class="reference-fields">
  {#if savedVoices.length}
    <label for={`${id}-library`}>Saved voice</label>
    <select id={`${id}-library`} value={savedVoiceId} disabled={busy}
      on:change={(event) => useSavedVoice(event.currentTarget.value)}>
      <option value="">Choose from Voices</option>
      {#each savedVoices as voice}<option value={voice.id}>{voice.name}</option>{/each}
    </select>
  {/if}
  <div class="recording-label"><label for={`${id}-audio`}>Reference recording</label>
    {#if file}<button class="icon-button" type="button" title="Clear recording" aria-label="Clear recording" disabled={busy}
      on:click={() => { file = null; savedVoiceId = ''; }}><X size={14} /></button>{/if}
  </div>
  <input class="file-native" id={`${id}-audio`} type="file" accept="audio/*" disabled={busy}
    on:change={(event) => { file = event.currentTarget.files?.[0] || null; savedVoiceId = ''; event.currentTarget.value = ''; }} />
  <label class="file-picker" for={`${id}-audio`}><FileAudio size={17} /><strong>Choose audio</strong><span>{file?.name || 'No recording selected'}</span></label>
  <MediaPreview {file} kind="audio" label="Reference preview" />
  <label for={`${id}-transcript`}>Words spoken in the recording <span>Required with a recording</span></label>
  <textarea id={`${id}-transcript`} rows="3" bind:value={transcript} disabled={busy}
    placeholder="Enter the exact reference transcript."></textarea>
</div>

<style>
  .reference-fields { display: grid; gap: 8px; min-width: 0; }
  .recording-label { display: flex; align-items: center; justify-content: space-between; min-height: 24px; }
  label { margin: 0; font-size: 12px; }
  label span { color: var(--muted); font-weight: normal; margin-left: 4px; }
  textarea, select { width: 100%; box-sizing: border-box; }
  .file-picker { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .file-picker span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .icon-button { display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px; padding: 0; }
</style>
