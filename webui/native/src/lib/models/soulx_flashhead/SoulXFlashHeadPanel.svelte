<script lang="ts">
  import { onMount } from 'svelte';
  import { browserDecodeToWav } from '$lib/audio';
  import { runTask, uploadFile, uploadWav } from '$lib/api';
  import MediaPreview from '$lib/MediaPreview.svelte';
  import { prepareRgb24Output } from '$lib/media/rgb24_video';
  import type { PanelRunContext, PanelRunOutcome, RegisterPanelRunner } from '$lib/models/panel_runner';

  export let busy = false;
  export let sourceFile: File | null = null;
  export let setPanelRunner: RegisterPanelRunner = () => () => {};

  let portrait: File | null = null;
  let input: HTMLInputElement | null = null;

  async function run(context: PanelRunContext): Promise<PanelRunOutcome> {
    if (!portrait) throw context.warning('Choose a portrait image.');
    if (!sourceFile) throw context.warning('Choose the speech audio that will animate the portrait.');
    const image = portrait;
    const drivingAudio = sourceFile;
    await context.ensureLoaded('offline');
    context.setStatus('Uploading portrait and speech audio...');
    const imagePath = await uploadFile(image, context.signal);
    const audio = await uploadWav(await browserDecodeToWav(drivingAudio, 16000), context.signal);
    context.setStatus('Animating portrait...');
    const result = await runTask({
      model: context.modelId,
      request: { audio, options: { ...context.options, seed: context.seed, reference_image_file: imagePath } }
    }, context.signal);
    context.setStatus('Encoding MP4...');
    const prepared = await prepareRgb24Output(result, drivingAudio, 'SoulX-FlashHead', 'soulx_flashhead', 'opus');
    return {
      audio: null,
      text: prepared.warning,
      artifacts: prepared.artifacts,
      json: JSON.stringify(result, (key, value) => key === 'payload' && typeof value === 'string'
        ? `<base64 data: ${value.length} chars>` : value, 2)
    };
  }

  onMount(() => setPanelRunner(run));
</script>

<div class="portrait-input">
  <label for="flashhead-portrait">Portrait image <span>required</span></label>
  <input bind:this={input} id="flashhead-portrait" class="file-native" type="file" accept="image/png,image/jpeg,image/bmp"
    disabled={busy} on:change={(event) => portrait = event.currentTarget.files?.[0] || null} />
  <label class="file-picker" for="flashhead-portrait"><strong>Choose portrait</strong><span>{portrait?.name || 'No portrait selected'}</span></label>
  <button type="button" disabled={!portrait || busy} on:click={() => { portrait = null; if (input) input.value = ''; }}>Clear portrait</button>
  <MediaPreview file={portrait} kind="image" label="Portrait preview" />
  <p>The model center-crops the image to the selected output dimensions. A centered, front-facing portrait works best.</p>
</div>

<style>
  .portrait-input { display: grid; gap: .5rem; min-width: 0; }
  .portrait-input > label:first-child { color: var(--muted); font-size: .75rem; font-weight: 600; }
  .portrait-input > label span { font-weight: 400; }
  .portrait-input button { justify-self: start; }
  .portrait-input p { color: var(--muted); font-size: .75rem; margin: 0; }
</style>
