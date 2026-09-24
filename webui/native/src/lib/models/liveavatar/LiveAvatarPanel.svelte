<script lang="ts">
  import { onDestroy } from 'svelte';
  import { uploadFile } from '$lib/api';
  import MediaPreview from '$lib/MediaPreview.svelte';
  import type { ParamSpec } from '$lib/types';

  export let busy = false;
  export let loraUploading = false;
  export let paramSpecs: ParamSpec[] = [];
  export let advancedValues: Record<string, unknown> = {};
  export let setParameterValue: (spec: ParamSpec, value: unknown) => void = () => {};
  export let sourceFile: File | null = null;
  export let sourceRecording = false;
  export let sourceRecordingBlocked = false;
  export let setSourceFile: (file: File | null) => void = () => {};
  export let startSourceRecording: () => void = () => {};
  export let stopSourceRecording: () => void = () => {};

  const generationNames = [
    'generation_mode', 'height', 'width', 'num_frames', 'num_clips',
    'num_inference_steps', 'guidance_scale', 'shift', 'negative_prompt'
  ];
  const memoryNames = [
    'sage_attention', 'memory_saver', 'fused_cfg', 'denoiser_layerwise',
    'denoiser_layerwise_batch', 'target_cache_blocks', 'vae_cache_f16',
    'vae_encoder_chunk_size', 'vae_decoder_tile_size', 'denoiser_weight_streaming'
  ];

  let referenceImage: File | null = null;
  let referenceImageInput: HTMLInputElement | null = null;
  let sourceAudioInput: HTMLInputElement | null = null;
  let referenceImageUrl = '';
  let imageError = '';
  let upload: AbortController | null = null;

  $: generationSpecs = generationNames
    .map((name) => paramSpecs.find((spec) => spec.name === name))
    .filter((spec): spec is ParamSpec => spec !== undefined);
  $: memorySpecs = memoryNames
    .map((name) => paramSpecs.find((spec) => spec.name === name))
    .filter((spec): spec is ParamSpec => spec !== undefined);

  function setNamedParameter(name: string, value: unknown) {
    const spec = paramSpecs.find((candidate) => candidate.name === name);
    if (spec) setParameterValue(spec, value);
  }

  function clearReferenceImage() {
    upload?.abort();
    upload = null;
    loraUploading = false;
    referenceImage = null;
    imageError = '';
    if (referenceImageUrl) URL.revokeObjectURL(referenceImageUrl);
    referenceImageUrl = '';
    if (referenceImageInput) referenceImageInput.value = '';
    setNamedParameter('reference_image_path', '');
  }

  async function selectReferenceImage(file: File | null) {
    if (!file) return;
    upload?.abort();
    imageError = '';
    referenceImage = file;
    if (referenceImageUrl) URL.revokeObjectURL(referenceImageUrl);
    referenceImageUrl = URL.createObjectURL(file);
    loraUploading = true;
    upload = new AbortController();
    try {
      const path = await uploadFile(file, upload.signal);
      setNamedParameter('reference_image_path', path);
    } catch (error) {
      if (!upload.signal.aborted) {
        imageError = error instanceof Error ? error.message : String(error);
        setNamedParameter('reference_image_path', '');
      }
    } finally {
      loraUploading = false;
      upload = null;
    }
  }

  function selectSourceAudio(file: File | null) {
    setSourceFile(file);
  }

  function clearSourceAudio() {
    if (sourceAudioInput) sourceAudioInput.value = '';
    setSourceFile(null);
  }

  onDestroy(() => {
    upload?.abort();
    if (referenceImageUrl) URL.revokeObjectURL(referenceImageUrl);
  });
</script>

<div class="liveavatar-form">
  <div class="reference-image-field">
    <label for="liveavatar-reference-image">Reference image <span>required</span></label>
    <input bind:this={referenceImageInput} id="liveavatar-reference-image" class="file-native" type="file"
      accept="image/png,image/jpeg,image/webp" disabled={busy || loraUploading}
      on:change={(event) => selectReferenceImage(event.currentTarget.files?.[0] || null)} />
    <div class="reference-image-actions">
      <label class="file-picker" for="liveavatar-reference-image">Choose image</label>
      <span>{referenceImage?.name || 'No image selected'}</span>
      {#if referenceImage}<button type="button" disabled={busy} on:click={clearReferenceImage}>Clear</button>{/if}
    </div>
    {#if referenceImageUrl}<img src={referenceImageUrl} alt="LiveAvatar reference" />{/if}
    {#if loraUploading}<small>Uploading reference image…</small>{/if}
    {#if imageError}<small class="error">{imageError}</small>{/if}
  </div>

  <div class="source-audio-field">
    <label for="liveavatar-source-audio">Driving audio <span>required</span></label>
    <input bind:this={sourceAudioInput} id="liveavatar-source-audio" class="file-native" type="file"
      accept="audio/*" disabled={busy || sourceRecording}
      on:change={(event) => selectSourceAudio(event.currentTarget.files?.[0] || null)} />
    <label class="file-picker" for="liveavatar-source-audio">
      <strong>Choose file</strong><span>{sourceFile?.name || 'No file chosen'}</span>
    </label>
    <div class="media-actions">
      {#if sourceRecording}
        <button class="danger" type="button" on:click={stopSourceRecording}>Stop recording</button>
        <span class="recording-dot">Recording microphone…</span>
      {:else}
        <button type="button" disabled={sourceRecordingBlocked} on:click={startSourceRecording}>Record microphone</button>
        <button type="button" disabled={!sourceFile} on:click={clearSourceAudio}>Clear file</button>
        {#if sourceFile}<span>{sourceFile.name}</span>{/if}
      {/if}
    </div>
    <MediaPreview file={sourceFile} kind="audio" label="Audio preview" />
  </div>

  <div class="control-grid">
    {#each generationSpecs as spec}
      <div class:wide={spec.type === 'text'} class="field">
        <label for={'liveavatar-' + spec.name}>{spec.label || spec.name.replace(/_/g, ' ')}</label>
        {#if spec.type === 'choice'}
          <select id={'liveavatar-' + spec.name} value={String(advancedValues[spec.name] ?? spec.default ?? '')}
            on:change={(event) => setParameterValue(spec, event.currentTarget.value)}>
            {#each spec.choices || [] as choice}<option value={choice}>{choice}</option>{/each}
          </select>
        {:else if spec.type === 'slider'}
          <div class="range-control">
            <input id={'liveavatar-' + spec.name} type="range" min={spec.minimum} max={spec.maximum} step={spec.step}
              value={Number(advancedValues[spec.name] ?? spec.default)}
              on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
            <output>{String(advancedValues[spec.name])}</output>
          </div>
        {:else}
          <input id={'liveavatar-' + spec.name} type={spec.type === 'number' ? 'number' : 'text'}
            min={spec.minimum} max={spec.maximum} step={spec.step}
            value={String(advancedValues[spec.name] ?? spec.default ?? '')}
            on:input={(event) => setParameterValue(spec,
              spec.type === 'number' ? event.currentTarget.valueAsNumber : event.currentTarget.value)} />
        {/if}
      </div>
    {/each}
  </div>

  <details>
    <summary>Memory and execution</summary>
    <div class="control-grid memory-grid">
      {#each memorySpecs as spec}
        <div class="field">
          {#if spec.type === 'bool'}
            <label class="toggle" for={'liveavatar-' + spec.name}>
              <input id={'liveavatar-' + spec.name} type="checkbox"
                checked={Boolean(advancedValues[spec.name])}
                on:change={(event) => setParameterValue(spec, event.currentTarget.checked)} />
              <span></span>{spec.label || spec.name.replace(/_/g, ' ')}
            </label>
          {:else}
            <label for={'liveavatar-' + spec.name}>{spec.label || spec.name.replace(/_/g, ' ')}</label>
            <input id={'liveavatar-' + spec.name} type="number" min={spec.minimum} max={spec.maximum} step={spec.step}
              value={Number(advancedValues[spec.name] ?? spec.default)}
              on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
          {/if}
        </div>
      {/each}
    </div>
  </details>
</div>

<style>
  .liveavatar-form { display: grid; gap: 14px; }
  .reference-image-field, .source-audio-field { display: grid; gap: 8px; }
  .reference-image-field > label, .source-audio-field > label, .field > label { color: var(--muted); font-size: 11px; font-weight: 700; }
  .reference-image-field > label span, .source-audio-field > label span { font-weight: 500; }
  .file-native { display: none; }
  .reference-image-actions { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .reference-image-actions span { min-width: 0; overflow: hidden; color: var(--muted); font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
  .reference-image-actions button { margin-left: auto; }
  .reference-image-field img { width: min(100%, 360px); max-height: 260px; object-fit: contain; object-position: left center; border: 1px solid var(--line); border-radius: 6px; background: #000; }
  .reference-image-field small { color: var(--muted); }
  .reference-image-field small.error { color: var(--danger); }
  .control-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
  .field { display: grid; align-content: start; gap: 7px; min-width: 0; }
  .field.wide { grid-column: 1 / -1; }
  .field input, .field select { width: 100%; }
  .range-control { display: grid; grid-template-columns: minmax(0, 1fr) 48px; align-items: center; gap: 8px; }
  .range-control output { color: var(--text); font-family: var(--mono); font-size: 12px; text-align: right; }
  details { margin: 0; }
  details > summary { cursor: pointer; color: var(--text); font-size: 12px; font-weight: 700; }
  .memory-grid { margin-top: 12px; }
  @media (max-width: 780px) { .control-grid { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 520px) { .control-grid { grid-template-columns: 1fr; } .field.wide { grid-column: auto; } }
</style>
