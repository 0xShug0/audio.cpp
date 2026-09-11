<script lang="ts">
  import { onDestroy } from 'svelte';

  export let file: File | null = null;
  export let src = '';
  export let name = '';
  export let kind: 'audio' | 'video' = 'audio';
  export let label = 'Preview';

  let objectUrl = '';
  let lastFile: File | null = null;
  $: previewUrl = objectUrl || src;
  $: previewName = file?.name || name;

  function clearObjectUrl() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = '';
    }
  }

  $: if (file !== lastFile) {
    clearObjectUrl();
    lastFile = file;
    if (file) objectUrl = URL.createObjectURL(file);
  }

  onDestroy(clearObjectUrl);
</script>

{#if previewUrl}
  <div class="media-preview">
    <div class="media-preview-header">
      <strong>{label}</strong>
      {#if previewName}<span>{previewName}</span>{/if}
    </div>
    {#if kind === 'video'}
      <!-- svelte-ignore a11y_media_has_caption -->
      <video controls preload="metadata" src={previewUrl}></video>
    {:else}
      <audio controls src={previewUrl}></audio>
    {/if}
  </div>
{/if}

<style>
  .media-preview {
    display: grid;
    gap: 7px;
    margin-top: 8px;
    padding: 8px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--control-bg);
  }

  .media-preview-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    font-size: 11px;
  }

  .media-preview-header strong {
    flex: 0 0 auto;
    color: var(--text);
  }

  .media-preview-header span {
    min-width: 0;
    overflow: hidden;
    color: var(--muted);
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  audio {
    width: 100%;
    height: 38px;
  }

  video {
    width: 100%;
    max-height: 220px;
    border-radius: 4px;
    background: #000;
  }
</style>
