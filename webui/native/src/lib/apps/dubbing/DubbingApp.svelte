<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { AudioLines, Download, FileText, Languages, Play, RefreshCw, Square, Upload } from '@lucide/svelte';
  import { base64AudioUrl, jsonRequest, uploadWav } from '$lib/api';
  import { browserDecodeToWav } from '$lib/audio';
  import { catalog, modelCapabilities } from '$lib/catalog';
  import { loadModelChoices } from '$lib/model-settings/choices';
  import { ModelRuntime } from '$lib/model-settings/runtime';
  import type { ModelSettings } from '$lib/model-settings/types';
  import ModelChoice from '../tts/ModelChoice.svelte';
  import MediaPreview from '$lib/MediaPreview.svelte';

  export let busy = false;
  export let active = true;
  let root: HTMLElement;
  let choices: ModelSettings[] = [];
  let translator: ModelSettings | null = null;
  let tts: ModelSettings | null = null;
  let loading = true;
  let source: File | null = null;
  let sourceLanguage = '';
  let targetLanguage = 'en';
  let reference: File | null = null;
  let referenceText = '';
  let status = '';
  let error = '';
  let aborter: AbortController | null = null;
  let outputUrl = '';
  let translatedText = '';
  let translationUrl = '';
  let timing: { wall_ms?: number; audio_duration_ms?: number; rtf?: number } = {};
  let outputSeed: number | null = null;

  const translatorFamilies = [...new Set(catalog.filter(entry =>
    entry.task === 'asr' && entry.request_options?.includes('target_language')
  ).map(entry => entry.family))];
  $: speechEntry = tts && (catalog.find(entry => entry.id === tts!.model.id) ||
    catalog.find(entry => entry.family === tts!.model.family && entry.task === 'tts'));
  $: builtinVoices = speechEntry?.builtin_voices || [];
  $: acceptsReference = Boolean(tts && modelCapabilities[tts.model.family]?.includes('speaker_reference'));
  $: if (!active && root) root.querySelectorAll('audio').forEach(audio => audio.pause());

  async function refreshChoices() {
    if (busy) return;
    loading = true;
    error = '';
    try {
      choices = await loadModelChoices();
      if (translator && !choices.some(choice => choice.id === translator!.id)) translator = null;
      if (tts && !choices.some(choice => choice.id === tts!.id)) tts = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }

  function changeSpeechModel(kind?: 'model' | 'settings') {
    if (kind === 'model') {
      reference = null;
      referenceText = '';
    }
  }

  async function generate() {
    if (busy || loading) return;
    if (!source || !translator || !tts || !targetLanguage.trim()) {
      error = 'Choose a recording, translation model, speech model, and target language.';
      return;
    }
    busy = true;
    error = '';
    const controller = new AbortController();
    aborter = controller;
    // Freeze both roles and inputs for the whole recipe, independent of Studio.
    const input = source;
    const voiceReference = acceptsReference ? reference : null;
    const transcript = referenceText.trim();
    const sourceCode = sourceLanguage.trim();
    const targetCode = targetLanguage.trim();
    const translation = structuredClone(translator);
    const synthesis = structuredClone(tts);
    try {
      const signal = controller.signal;
      status = 'Preparing models...';
      await new ModelRuntime(translation, signal).prepare();
      const speechRuntime = new ModelRuntime(synthesis, signal);
      await speechRuntime.prepare();
      status = 'Uploading recording...';
      const audio = await uploadWav(await browserDecodeToWav(input), signal);
      signal.throwIfAborted();
      const options: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(translation.request.options)) options[`translator.${key}`] = value;
      for (const [key, value] of Object.entries(synthesis.request.options)) options[`tts.${key}`] = value;
      const language = sourceCode || translation.request.language?.trim();
      if (language) options['translator.language'] = language;
      options['translator.target_language'] = targetCode;
      if (synthesis.request.language) options['tts.language'] = synthesis.request.language;
      if (synthesis.request.max_tokens !== undefined) options['tts.max_tokens'] = synthesis.request.max_tokens;
      options['tts.seed'] = speechRuntime.resolvedSeed;
      const request: Record<string, unknown> = { audio, options };
      if (voiceReference) {
        request.voice_ref = await uploadWav(await browserDecodeToWav(voiceReference), signal);
        if (transcript) options['tts.reference_text'] = transcript;
      } else if (synthesis.request.voice) {
        request.voice_id = synthesis.request.voice;
      }
      signal.throwIfAborted();
      status = 'Translating and generating speech...';
      const result = await jsonRequest<{
        text?: string;
        audio?: string;
        timing?: typeof timing;
      }>('/v1/recipes/run', {
        method: 'POST',
        body: JSON.stringify({ recipe: 'cross-language-dubbing',
          models: { translator: translation.model.id, tts: synthesis.model.id }, request })
      }, signal);
      signal.throwIfAborted();
      if (!result.audio || !result.text?.trim()) throw new Error('The recipe returned no dubbed audio or translation.');
      const nextAudio = base64AudioUrl(result.audio);
      const nextText = URL.createObjectURL(new Blob([result.text], { type: 'text/plain;charset=utf-8' }));
      if (outputUrl) URL.revokeObjectURL(outputUrl);
      if (translationUrl) URL.revokeObjectURL(translationUrl);
      outputUrl = nextAudio;
      translationUrl = nextText;
      translatedText = result.text;
      timing = result.timing || {};
      outputSeed = speechRuntime.resolvedSeed;
      status = 'Dub ready';
    } catch (e) {
      if (controller.signal.aborted) status = 'Cancelled';
      else {
        error = e instanceof Error ? e.message : String(e);
        status = '';
      }
    } finally {
      busy = false;
      aborter = null;
    }
  }

  onMount(refreshChoices);
  onDestroy(() => {
    aborter?.abort();
    if (outputUrl) URL.revokeObjectURL(outputUrl);
    if (translationUrl) URL.revokeObjectURL(translationUrl);
  });
</script>

<section class="dubbing" bind:this={root}>
  <header>
    <h1><Languages size={24} />Cross-language dubbing</h1>
    <button type="button" disabled={busy || loading} on:click={refreshChoices} title="Refresh installed models" aria-label="Refresh installed models"><RefreshCw size={17} /></button>
  </header>
  <div class="workspace">
    <div class="inputs">
      <section class="source">
        <h2><Upload size={17} />Source recording</h2>
        <label>Audio file<input type="file" accept="audio/*" disabled={busy} on:change={event => source = event.currentTarget.files?.[0] || null} /></label>
        <MediaPreview file={source} label="Original audio" />
      </section>
      <section>
        <h2><Languages size={17} />Translation</h2>
        <ModelChoice label="Translation model" {choices} tasks={['asr']} families={translatorFamilies}
          bind:value={translator} disabled={busy || loading} onchange={() => {}} />
        <div class="language-fields">
          <label>Source language<input aria-label="Source language" bind:value={sourceLanguage} disabled={busy} placeholder="Model default" /></label>
          <label>Target language<input aria-label="Target language" bind:value={targetLanguage} disabled={busy} placeholder="en" required /></label>
        </div>
      </section>
      <section>
        <h2><AudioLines size={17} />Speech</h2>
        <ModelChoice label="Speech model" {choices} tasks={['tts']} bind:value={tts}
          disabled={busy || loading} onchange={changeSpeechModel} />
        {#if tts && builtinVoices.length}
          <label class="voice">Voice<select aria-label="Dub voice" bind:value={tts.request.voice} disabled={busy}>
            {#each builtinVoices as voice}<option value={voice}>{voice}</option>{/each}
          </select></label>
        {/if}
        {#if acceptsReference}
          <details>
            <summary>Reference voice</summary>
            <label>Reference audio<input type="file" accept="audio/*" disabled={busy} on:change={event => reference = event.currentTarget.files?.[0] || null} /></label>
            <MediaPreview file={reference} label="Reference voice" />
            <label>Reference transcript<textarea rows="2" bind:value={referenceText} disabled={busy}></textarea></label>
            {#if reference}<button type="button" disabled={busy} on:click={() => { reference = null; referenceText = ''; }}>Clear reference</button>{/if}
          </details>
        {/if}
      </section>
      <div class="actions">
        <button class="run" type="button" disabled={busy || loading || !source || !translator || !tts || !targetLanguage.trim()} on:click={generate}><Play size={16} />Generate dub</button>
        {#if busy}<button type="button" on:click={() => aborter?.abort()}><Square size={15} />Cancel</button>{/if}
        <span role="status">{loading ? 'Loading installed models...' : status}</span>
      </div>
      {#if error}<p role="alert">{error}</p>{/if}
    </div>
    <section class="result" aria-label="Dub result">
      <h2><FileText size={17} />Translation and audio</h2>
      {#if outputUrl}
        <p class="translation">{translatedText}</p>
        <audio controls src={outputUrl}></audio>
        <dl>
          {#if timing.wall_ms !== undefined}<div><dt>Generation</dt><dd>{(timing.wall_ms / 1000).toFixed(2)} s</dd></div>{/if}
          {#if timing.audio_duration_ms !== undefined}<div><dt>Audio</dt><dd>{(timing.audio_duration_ms / 1000).toFixed(2)} s</dd></div>{/if}
          {#if timing.rtf !== undefined}<div><dt>RTF</dt><dd>{timing.rtf.toFixed(3)}</dd></div>{/if}
          <div><dt>Seed</dt><dd>{outputSeed}</dd></div>
        </dl>
        <div class="downloads">
          <a href={outputUrl} download="dub.wav"><Download size={15} />WAV</a>
          <a href={translationUrl} download="translation.txt"><Download size={15} />Translation</a>
        </div>
      {:else}
        <p class="empty">No dub generated</p>
      {/if}
    </section>
  </div>
</section>

<style>
  .dubbing { max-width: 1200px; margin: 20px auto; }
  header, h1, h2, .actions, .downloads, .downloads a { display: flex; align-items: center; gap: 10px; }
  header { justify-content: space-between; margin-bottom: 24px; }
  h1 { margin: 0; font-size: 24px; }
  h2 { margin: 0 0 16px; font-size: 15px; }
  header button { width: 36px; height: 36px; padding: 0; flex-shrink: 0; justify-content: center; }
  .workspace { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 32px; }
  .inputs > section { border-bottom: 1px solid var(--line); padding: 0 0 24px; margin-bottom: 24px; }
  .result { border-left: 1px solid var(--line); padding-left: 32px; min-width: 0; }
  label { display: grid; gap: 8px; margin: 0; font-size: 13px; color: var(--muted); min-width: 0; }
  input, select, textarea { width: 100%; min-width: 0; }
  input[type='file'] { font-size: 12px; }
  .language-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
  .voice, details { margin-top: 16px; }
  details label { margin: 12px 0; }
  summary { cursor: pointer; font-size: 13px; color: var(--muted); }
  .actions, .downloads { flex-wrap: wrap; }
  .actions span, .empty, dt { color: var(--muted); font-size: 12px; }
  [role='alert'] { color: var(--danger); overflow-wrap: anywhere; }
  .translation { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.7; margin: 0 0 24px; }
  audio { width: 100%; }
  dl { display: flex; flex-wrap: wrap; gap: 24px; margin: 24px 0; }
  dd { margin: 6px 0 0; font-size: 14px; font-variant-numeric: tabular-nums; }
  .downloads a { color: var(--cyan); font-size: 13px; padding: 8px 0; }
  @media (max-width: 760px) {
    .workspace { grid-template-columns: minmax(0, 1fr); gap: 24px; }
    .result { border-left: 0; border-top: 1px solid var(--line); padding: 24px 0 0; }
    h1 { font-size: 20px; }
  }
</style>
