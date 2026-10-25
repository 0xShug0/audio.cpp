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

  let video: File | null = null;
  let input: HTMLInputElement | null = null;

  async function run(context: PanelRunContext): Promise<PanelRunOutcome> {
    if (!video) throw context.warning('Choose a face video.');
    if (!sourceFile) throw context.warning('Choose the speech audio that will drive the lips.');
    const referenceVideo = video;
    const drivingAudio = sourceFile;
    await context.ensureLoaded('offline');
    context.setStatus('Uploading video and speech audio...');
    const videoPath = await uploadFile(referenceVideo, context.signal);
    const audio = await uploadWav(await browserDecodeToWav(drivingAudio, 16000), context.signal);
    context.setStatus('Synchronizing lip motion...');
    const result = await runTask({
      model: context.modelId,
      request: { audio, options: { ...context.options, seed: context.seed, reference_video_file: videoPath } }
    }, context.signal);
    context.setStatus('Encoding MP4...');
    const prepared = await prepareRgb24Output(result, drivingAudio, 'LatentSync', 'latentsync', 'opus');
    if (prepared.warning || !prepared.artifacts.some((artifact) => artifact.mime === 'video/mp4')) {
      for (const artifact of prepared.artifacts) URL.revokeObjectURL(artifact.url);
      throw new Error(prepared.warning);
    }
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

<div class="latentsync-video">
  <label for="latentsync-video">Face video <span>required</span></label>
  <input bind:this={input} id="latentsync-video" class="file-native" type="file" accept="video/*"
    disabled={busy} on:change={(event) => video = event.currentTarget.files?.[0] || null} />
  <label class="file-picker" for="latentsync-video"><strong>Choose video</strong><span>{video?.name || 'No video selected'}</span></label>
  <button type="button" disabled={!video || busy} on:click={() => { video = null; if (input) input.value = ''; }}>Clear video</button>
  <MediaPreview file={video} kind="video" label="Face video preview" />
</div>

<style>
  .latentsync-video { display: grid; gap: 8px; min-width: 0; }
  .latentsync-video > label:first-child { color: var(--muted); font-size: 12px; font-weight: 600; }
  .latentsync-video > label span { font-weight: 400; }
  .latentsync-video button { justify-self: start; }
</style>
