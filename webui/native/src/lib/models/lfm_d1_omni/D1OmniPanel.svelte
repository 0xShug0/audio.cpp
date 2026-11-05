<script lang="ts">
  import { onMount } from 'svelte';
  import { browserDecodeToWav } from '$lib/audio';
  import { runTask, uploadWav } from '$lib/api';
  import type { PanelRunContext, PanelRunOutcome, RegisterPanelRunner } from '$lib/models/panel_runner';

  export let sourceFile: File | null = null;
  export let setPanelRunner: RegisterPanelRunner = () => () => {};

  async function run(context: PanelRunContext): Promise<PanelRunOutcome> {
    if (!sourceFile) throw context.warning('Choose an audio recording to analyze.');
    const questions = JSON.parse(String(context.options.questions || ''));
    if (!questions || Array.isArray(questions) || typeof questions !== 'object' || !Object.keys(questions).length) {
      throw context.warning('Questions JSON must be a nonempty object of named questions.');
    }
    const recording = sourceFile;
    await context.ensureLoaded('offline');
    context.setStatus('Analyzing audio...');
    const audio = await uploadWav(await browserDecodeToWav(recording, 16000), context.signal);
    const result = await runTask({ model: context.modelId, request: { audio, options: context.options } }, context.signal);
    const output = result.custom_schema_output as { schema?: string; data?: { answers?: unknown } } | undefined;
    if (output?.schema !== 'audio_decisions' || !output.data?.answers) {
      throw new Error('d1-Omni returned no audio decisions.');
    }
    return { audio: null, text: JSON.stringify(output.data.answers, null, 2), json: JSON.stringify(result, null, 2) };
  }

  onMount(() => setPanelRunner(run));
</script>

<p>Edit Questions JSON under Model parameters to choose the decisions returned for this recording.</p>
