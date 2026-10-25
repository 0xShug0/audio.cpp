<script lang="ts">
  import { runTask } from '$lib/api';
  import type { Translator } from '$lib/i18n';
  import type { ParamSpec } from '$lib/types';
  import LoraGroup from '$lib/LoraGroup.svelte';

  export let modelId: string;
  export let text = '';
  export let lyrics = '';
  export let language = '';
  export let duration = 30;
  export let seed = -1;
  export let busy = false;
  export let adapterEditable = false;
  export let rewritingCaption = false;
  export let advancedValues: Record<string, unknown> = {};
  export let paramSpecs: ParamSpec[] = [];
  export let outputText = '';
  export let outputJson = '';
  export let status = '';
  export let warningStatus = '';
  export let errorStatus = '';
  export let ensureLoaded: () => Promise<void>;
  export let requestOptions: () => Record<string, unknown>;
  export let resolveRequestSeed: (value: number) => number;
  export let setParameterValue: (spec: ParamSpec, value: unknown) => void;
  export let onBrowseAdapter: (spec: ParamSpec) => void;
  export let log: (message: string) => void;
  export let tr: Translator;

  $: loraSpec = paramSpecs.find((spec) => spec.name === 'lora');
  $: scaleSpec = paramSpecs.find((spec) => spec.name === 'lora_scale');

  function planFromResult(result: Record<string, unknown>): Record<string, unknown> {
    if (Array.isArray(result.artifacts)) {
      const artifact = result.artifacts.find((entry): entry is { id: string; payload: string } =>
        typeof entry === 'object' && entry !== null &&
          (entry as { id?: unknown }).id === 'ace_step_caption_plan' &&
          typeof (entry as { payload?: unknown }).payload === 'string');
      if (artifact) {
        const bytes = Uint8Array.from(atob(artifact.payload), (character) => character.charCodeAt(0));
        return JSON.parse(new TextDecoder().decode(bytes));
      }
    }
    if (typeof result.text === 'string') return { caption: result.text };
    return {};
  }

  async function rewriteCaption() {
    if (busy || rewritingCaption) return;
    if (!text.trim() && !lyrics.trim()) {
      status = 'Enter a caption or lyrics to rewrite.';
      warningStatus = status;
      errorStatus = '';
      return;
    }
    rewritingCaption = true;
    warningStatus = '';
    errorStatus = '';
    status = tr('request.rewritingCaption');
    try {
      await ensureLoaded();
      const request: Record<string, unknown> = {
        text,
        seed: resolveRequestSeed(seed),
        duration_seconds: duration,
        options: { ...requestOptions(), rewrite_caption: true }
      };
      if (language.trim()) request.language = language;
      if (lyrics.trim()) request.lyrics = lyrics;
      const result = await runTask({ model: modelId, request });
      const plan = planFromResult(result);
      if (typeof plan.caption === 'string' && plan.caption.trim()) text = plan.caption;
      if (typeof plan.language === 'string' && plan.language.trim()) language = plan.language;
      if (typeof plan.duration_seconds === 'number' && Number.isFinite(plan.duration_seconds) && plan.duration_seconds > 0) {
        duration = plan.duration_seconds;
      }
      const nextAdvanced = { ...advancedValues };
      if (typeof plan.bpm === 'number' && Number.isFinite(plan.bpm)) nextAdvanced.bpm = plan.bpm;
      if (typeof plan.keyscale === 'string') nextAdvanced.keyscale = plan.keyscale;
      if (typeof plan.timesignature === 'string') nextAdvanced.timesignature = plan.timesignature;
      advancedValues = nextAdvanced;
      outputText = typeof result.text === 'string' ? result.text : '';
      outputJson = JSON.stringify(plan, null, 2);
      status = 'Caption rewritten.';
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      errorStatus = status;
      log(`Caption rewrite failed: ${status}`);
    } finally {
      rewritingCaption = false;
    }
  }
</script>

<label for="lyrics">{tr('request.lyrics')} <span>{tr('request.optional')}</span></label>
<textarea id="lyrics" rows="3" bind:value={lyrics} placeholder="[Verse]..."></textarea>
<div class="media-actions">
  <button type="button" disabled={busy || rewritingCaption || (!text.trim() && !lyrics.trim())}
    on:click={rewriteCaption}>
    {rewritingCaption ? tr('request.rewritingCaption') : tr('request.rewriteCaption')}
  </button>
</div>

{#if loraSpec && scaleSpec}
  <LoraGroup selected={advancedValues.lora ? 1 : 0}>
    <div class="lora-fields">
      <div>
        <label for="ace-lora">Adapter path <span>{tr('request.optional')}</span></label>
        <div class="lora-path">
          <input id="ace-lora" type="text" disabled={busy || rewritingCaption || !adapterEditable}
            title={adapterEditable ? '' : 'Set adapter session options in the server configuration.'}
            value={String(advancedValues.lora ?? '')} placeholder="Server adapter file or PEFT directory"
            on:input={(event) => setParameterValue(loraSpec, event.currentTarget.value.trim())} />
          <button type="button" disabled={busy || rewritingCaption || !adapterEditable}
            on:click={() => onBrowseAdapter(loraSpec)}>Browse folder</button>
        </div>
      </div>
      <div>
        <label for="ace-lora-scale">LoRA strength</label>
        <input id="ace-lora-scale" type="number" step="0.1" disabled={busy || rewritingCaption || !adapterEditable || !advancedValues.lora}
          value={Number(advancedValues.lora_scale ?? 1)}
          on:input={(event) => {
            if (Number.isFinite(event.currentTarget.valueAsNumber)) setParameterValue(scaleSpec, event.currentTarget.valueAsNumber);
          }} />
      </div>
    </div>
    <button type="button" disabled={busy || rewritingCaption || !adapterEditable || !advancedValues.lora}
      on:click={() => setParameterValue(loraSpec, '')}>Clear adapter</button>
  </LoraGroup>
{/if}
