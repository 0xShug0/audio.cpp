<script lang="ts">
  import { onMount } from 'svelte';
  import { browserDecodeToWav } from '$lib/audio';
  import { base64AudioBlob, runTask, uploadFile, uploadWav } from '$lib/api';
  import MediaPreview from '$lib/MediaPreview.svelte';
  import type { PanelRunContext, PanelRunOutcome, RegisterPanelRunner } from '$lib/models/panel_runner';

  export let busy = false;
  export let setPanelRunner: RegisterPanelRunner = () => () => {};

  let reference: File | null = null;
  let referenceMetadata: File | null = null;
  let targetMetadata: File | null = null;

  async function run(context: PanelRunContext): Promise<PanelRunOutcome> {
    if (!reference) throw context.warning('Choose a reference singing recording.');
    if (!referenceMetadata || !targetMetadata) {
      throw context.warning('Choose both reference and target metadata JSON files.');
    }
    const recording = reference;
    const referenceFile = referenceMetadata;
    const targetFile = targetMetadata;
    await context.ensureLoaded('offline');
    context.setStatus('Uploading reference singing and metadata…');
    const voiceRef = await uploadWav(await browserDecodeToWav(recording, 24000), context.signal);
    const referencePath = await uploadFile(referenceFile, context.signal);
    const targetPath = await uploadFile(targetFile, context.signal);
    context.setStatus('Synthesizing singing…');
    const result = await runTask({
      model: context.modelId,
      request: {
        voice_ref: voiceRef,
        options: { ...context.options, seed: context.seed,
          reference_metadata_file: referencePath, target_metadata_file: targetPath }
      }
    }, context.signal);
    if (typeof result.audio !== 'string') throw new Error('SoulX-Singer returned no audio.');
    return {
      audio: base64AudioBlob(result.audio),
      text: typeof result.text === 'string' ? result.text : '',
      json: JSON.stringify(result, (key, value) => key === 'audio' && typeof value === 'string'
        ? `<base64 audio: ${value.length} chars>` : value, 2)
    };
  }

  onMount(() => setPanelRunner(run));
</script>

<div class="singing-inputs">
  <label for="singer-reference">Reference singing <span>required</span></label>
  <input id="singer-reference" class="file-native" type="file" accept="audio/*" disabled={busy}
    on:change={(event) => reference = event.currentTarget.files?.[0] || null} />
  <label class="file-picker" for="singer-reference"><strong>Choose audio</strong><span>{reference?.name || 'No recording selected'}</span></label>
  <MediaPreview file={reference} kind="audio" label="Reference singing preview" />
  <label for="singer-reference-metadata">Reference metadata <span>JSON, required</span></label>
  <input id="singer-reference-metadata" class="file-native" type="file" accept=".json,application/json" disabled={busy}
    on:change={(event) => referenceMetadata = event.currentTarget.files?.[0] || null} />
  <label class="file-picker" for="singer-reference-metadata"><strong>Choose JSON</strong><span>{referenceMetadata?.name || 'No metadata selected'}</span></label>
  <label for="singer-target-metadata">Target metadata <span>JSON, required</span></label>
  <input id="singer-target-metadata" class="file-native" type="file" accept=".json,application/json" disabled={busy}
    on:change={(event) => targetMetadata = event.currentTarget.files?.[0] || null} />
  <label class="file-picker" for="singer-target-metadata"><strong>Choose JSON</strong><span>{targetMetadata?.name || 'No metadata selected'}</span></label>
  <small>Metadata defines the phonemes, notes and timing. Use matching reference metadata and
    choose score or melody control under Model parameters.
    <a href="https://github.com/Soul-AILab/SoulX-Singer/tree/main/example/audio" target="_blank" rel="noreferrer">Example files</a>
  </small>
</div>

<style>
  .singing-inputs { display: grid; gap: 8px; min-width: 0; }
  label { color: var(--muted); font-size: 12px; font-weight: 600; }
  label span { font-weight: 400; }
  small { color: var(--muted); }
  a { color: var(--cyan); }
</style>
