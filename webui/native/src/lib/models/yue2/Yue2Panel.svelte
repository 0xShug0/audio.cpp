<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { loadModel, models, runTask, unloadModel, uploadFile } from '$lib/api';
  import MediaPreview from '$lib/MediaPreview.svelte';
  import type { Translator } from '$lib/i18n';
  import type { CatalogEntry, LoadedModel, ParamSpec, ServerHealth } from '$lib/types';

  export let lyrics = '';
  export let seed = 1234;
  export let paramSpecs: ParamSpec[] = [];
  export let advancedValues: Record<string, unknown> = {};
  export let catalogEntries: CatalogEntry[] = [];
  export let loadedModels: LoadedModel[] = [];
  export let server: ServerHealth | null = null;
  export let modelPathFor: (entry: CatalogEntry) => string = (entry) => entry.path;
  export let sessionOptionsFor: (entry: CatalogEntry) => Record<string, string> = (entry) => entry.session_options || {};
  export let refreshModels: () => Promise<void> = async () => {};
  export let log: (message: string) => void = () => {};
  export let tr: Translator = (key, _values, fallback = key) => fallback;
  export let localizedParameterText:
    (spec: ParamSpec, field: 'label' | 'info' | 'placeholder', translate?: Translator) => string =
      (spec, field) => field === 'label' ? spec.name : '';
  export let setParameterValue: (spec: ParamSpec, value: unknown) => void = () => {};

  const componentParamNames = ['main_gguf', 'vae_gguf'];
  const coreParamNames = ['style', 'cot', 'cfg_scale', 'num_inference_steps'];
  const abcParamNames = ['abc', 'abc_file'];
  const semanticParamNames = [
    'semantic_temperature',
    'semantic_top_p',
    'semantic_top_k',
    'semantic_repetition_penalty',
    'semantic_penalty_window',
    'semantic_min_tokens',
    'semantic_max_tokens'
  ];
  const plannerParamNames = [
    'abc_temperature',
    'abc_top_p',
    'abc_top_k',
    'abc_repetition_penalty',
    'abc_penalty_window',
    'abc_min_tokens',
    'abc_max_tokens'
  ];

  let coverAudioFile: File | null = null;
  let coverAudioInput: HTMLInputElement | null = null;
  let coverRunning = false;
  const unloadSettingKey = 'audiocpp.ui.yue2.unloadSheetSageAfterConversion';
  let unloadAfterConversion = true;
  onMount(() => {
    unloadAfterConversion = localStorage.getItem(unloadSettingKey) !== 'false';
  });
  let coverStatus = '';
  let coverError = '';
  let abcDraft = '';
  let abcPreviewElement: HTMLDivElement | null = null;
  let lastRenderedAbc = '';
  let abcRenderError = '';
  let abcRender: ((target: HTMLElement, abc: string, options?: Record<string, unknown>) => unknown) | null = null;

  function specsByName(names: string[], specs: ParamSpec[]) {
    return names
      .map((name) => specs.find((spec) => spec.name === name))
      .filter((spec): spec is ParamSpec => spec !== undefined);
  }

  function specByName(name: string) {
    return paramSpecs.find((spec) => spec.name === name);
  }

  function setNamedParameter(name: string, value: unknown) {
    const spec = specByName(name);
    if (spec) setParameterValue(spec, value);
  }

  function decodedArtifactPayload(payload: string) {
    try {
      return atob(payload);
    } catch {
      return payload;
    }
  }

  function abcFromResult(result: Record<string, unknown>) {
    if (typeof result.text === 'string' && result.text.trim()) return result.text;
    const artifacts = Array.isArray(result.artifacts) ? result.artifacts : [];
    for (const artifact of artifacts) {
      if (!artifact || typeof artifact !== 'object') continue;
      const entry = artifact as { id?: unknown; payload?: unknown; meta?: Record<string, unknown> };
      const format = String(entry.meta?.format || entry.meta?.extension || entry.id || '');
      if (/abc|score/i.test(format) && typeof entry.payload === 'string') {
        return decodedArtifactPayload(entry.payload);
      }
    }
    return '';
  }

  function sheetSageModel() {
    return catalogEntries.find((entry) => entry.family === 'sheetsage2') || null;
  }

  async function ensureSheetSageLoaded(entry: CatalogEntry) {
    if (loadedModels.some((model) => model.id === entry.id && model.loaded)) return;
    const current = await models();
    const resident = current.find((model) => model.id === entry.id && model.loaded);
    if (resident) return;
    if (!server?.ui_management) {
      throw new Error('SheetSage2 is not loaded. Start the server with UI management or add SheetSage2 to server config.');
    }
    await loadModel({
      id: entry.id,
      path: modelPathFor(entry),
      family: entry.family,
      task: entry.task,
      mode: entry.mode || 'offline',
      load_options: entry.load_options || {},
      session_options: sessionOptionsFor(entry)
    });
  }

  async function transcribeCoverScore() {
    if (!coverAudioFile || coverRunning) return;
    coverRunning = true;
    coverError = '';
    coverStatus = 'Preparing SheetSage2 cover score transcription...';
    const shouldUnload = unloadAfterConversion && server?.ui_management === true;
    let cleanupModelId: string | null = null;
    try {
      const entry = sheetSageModel();
      if (!entry) throw new Error('SheetSage2 is not available in the model catalog.');
      await ensureSheetSageLoaded(entry);
      cleanupModelId = entry.id;
      await refreshModels();
      coverStatus = 'Uploading source song...';
      const audio = await uploadFile(coverAudioFile);
      coverStatus = 'Transcribing source song to ABC with SheetSage2...';
      const result = await runTask({
        model: entry.id,
        request: {
          audio,
          options: {}
        }
      });
      const abc = abcFromResult(result);
      if (!abc.trim()) throw new Error('SheetSage2 did not return an ABC score.');
      abcDraft = abc;
      setNamedParameter('abc', abc);
      setNamedParameter('abc_file', '');
      setNamedParameter('cot', 'melody');
      coverStatus = 'ABC score imported. Review/edit the sheet before generating the cover.';
      log('SheetSage2 cover score imported into Yue2 ABC conditioning.');
    } catch (error) {
      coverError = error instanceof Error ? error.message : String(error);
      coverStatus = '';
      log(`SheetSage2 cover transcription failed: ${coverError}`);
    } finally {
      if (shouldUnload && cleanupModelId) {
        try {
          await unloadModel(cleanupModelId);
          log('SheetSage2 unloaded after cover transcription.');
        } catch (error) {
          const message = `SheetSage2 unload failed: ${error instanceof Error ? error.message : String(error)}`;
          coverError = [coverError, message].filter(Boolean).join(' ');
          log(message);
        }
        try {
          await refreshModels();
        } catch (error) {
          const message = `Model status refresh failed: ${error instanceof Error ? error.message : String(error)}`;
          coverError = [coverError, message].filter(Boolean).join(' ');
          log(message);
        }
      }
      coverRunning = false;
    }
  }

  function clearCoverAudio() {
    coverAudioFile = null;
    if (coverAudioInput) coverAudioInput.value = '';
  }

  function updateAbc(value: string) {
    abcDraft = value;
    setNamedParameter('abc', value);
    if (value.trim()) setNamedParameter('abc_file', '');
  }

  async function renderAbcPreview(value: string) {
    const abc = value.trim();
    if (!abcPreviewElement) return;
    if (!abc) {
      abcPreviewElement.replaceChildren();
      lastRenderedAbc = '';
      abcRenderError = '';
      return;
    }
    if (abc === lastRenderedAbc) return;
    await tick();
    if (!abcPreviewElement) return;
    try {
      if (!abcRender) {
        const mod = await import('abcjs');
        abcRender = mod.renderAbc;
      }
      abcPreviewElement.replaceChildren();
      abcRender(abcPreviewElement, abc, {
        add_classes: true,
        responsive: 'resize'
      });
      lastRenderedAbc = abc;
      abcRenderError = '';
    } catch (error) {
      abcPreviewElement.replaceChildren();
      lastRenderedAbc = '';
      abcRenderError = error instanceof Error ? error.message : String(error);
    }
  }

  $: componentSpecs = specsByName(componentParamNames, paramSpecs);
  $: coreSpecs = specsByName(coreParamNames, paramSpecs);
  $: abcSpecs = specsByName(abcParamNames, paramSpecs);
  $: semanticSpecs = specsByName(semanticParamNames, paramSpecs);
  $: plannerSpecs = specsByName(plannerParamNames, paramSpecs);
  $: if (String(advancedValues.abc || '') !== abcDraft && !coverRunning) {
    abcDraft = String(advancedValues.abc || '');
  }
  $: void renderAbcPreview(abcDraft);
</script>

<div class="model-form yue2-form">
  <div class="yue2-field wide">
    <label for="lyrics">{tr('request.lyrics')} <span>{tr('voice.required')}</span></label>
    <textarea id="lyrics" rows="5" bind:value={lyrics} required aria-required="true"
      placeholder="[Verse]&#10;...&#10;[Chorus]&#10;..."></textarea>
  </div>

  <div class="yue2-grid yue2-grid-components">
    <div class="yue2-field">
      <label for="seed">{tr('request.seed')} <span>{tr('request.randomSeed')}</span></label>
      <input id="seed" type="number" min="-1" max="4294967295" step="1" bind:value={seed} />
    </div>
    {#each componentSpecs as spec}
      <div class="yue2-field">
        <label for={'param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
        <select id={'param-' + spec.name} value={String(advancedValues[spec.name] ?? '')}
          on:change={(event) => setParameterValue(spec, event.currentTarget.value)}>
          {#each spec.choices || [] as choice}<option value={choice}>{choice}</option>{/each}
        </select>
        {#if localizedParameterText(spec, 'info', tr)}<small>{localizedParameterText(spec, 'info', tr)}</small>{/if}
      </div>
    {/each}
  </div>

  <div class="yue2-grid yue2-grid-core">
    {#each coreSpecs as spec}
      <div class="yue2-field" class:wide={spec.type === 'text'}>
        <label for={'param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
        {#if spec.type === 'choice'}
          <select id={'param-' + spec.name} value={String(advancedValues[spec.name] ?? '')}
            on:change={(event) => setParameterValue(spec, event.currentTarget.value)}>
            {#each spec.choices || [] as choice}<option value={choice}>{choice}</option>{/each}
          </select>
        {:else if spec.type === 'slider'}
          <div class="yue2-range">
            <input id={'param-' + spec.name} type="range" min={spec.minimum} max={spec.maximum} step={spec.step}
              value={Number(advancedValues[spec.name] ?? spec.default)}
              on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
            <output>{String(advancedValues[spec.name])}</output>
          </div>
        {:else}
          <input id={'param-' + spec.name} type={spec.type === 'number' ? 'number' : 'text'}
            min={spec.minimum} max={spec.maximum} step={spec.step}
            value={String(advancedValues[spec.name] ?? '')}
            placeholder={localizedParameterText(spec, 'placeholder', tr)}
            on:input={(event) => setParameterValue(spec,
              spec.type === 'number' ? event.currentTarget.valueAsNumber : event.currentTarget.value)} />
        {/if}
        {#if localizedParameterText(spec, 'info', tr)}<small>{localizedParameterText(spec, 'info', tr)}</small>{/if}
      </div>
    {/each}
  </div>

  <details class="yue2-details">
    <summary>Yue2 ABC conditioning <span>{abcSpecs.length}</span></summary>
    <div class="yue2-cover">
      <section class="yue2-cover-card">
        <div class="yue2-cover-head">
          <div>
            <strong>Cover source</strong>
            <small>Use SheetSage2 + MERT2 to extract an editable ABC score from a song.</small>
          </div>
          <button type="button" disabled={!coverAudioFile || coverRunning} on:click={transcribeCoverScore}>
            {coverRunning ? 'Transcribing...' : 'Extract ABC'}
          </button>
        </div>
        <label class="yue2-unload-toggle"
          title={server?.ui_management ? undefined : 'Requires server UI model management'}>
          <input type="checkbox" role="switch"
            checked={unloadAfterConversion && server?.ui_management === true}
            disabled={coverRunning || !server?.ui_management}
            on:change={(event) => {
              unloadAfterConversion = event.currentTarget.checked;
              localStorage.setItem(unloadSettingKey, String(unloadAfterConversion));
            }} />
          <span>Unload SheetSage2 after conversion</span>
        </label>
        <small class="yue2-error">Warning: VRAM may remain in use after unloading.</small>
        <input id="yue2-cover-audio" class="file file-native" type="file" accept="audio/*"
          bind:this={coverAudioInput}
          on:change={(event) => coverAudioFile = event.currentTarget.files?.[0] || null} />
        <label class="file-picker yue2-cover-picker" for="yue2-cover-audio">
          <strong>Choose source song</strong>
          <span>{coverAudioFile?.name || tr('file.none')}</span>
        </label>
        <div class="media-actions">
          <button type="button" disabled={!coverAudioFile || coverRunning} on:click={clearCoverAudio}>Clear</button>
          {#if coverStatus}<span>{coverStatus}</span>{/if}
          {#if coverError}<span class="yue2-error">{coverError}</span>{/if}
        </div>
        <MediaPreview file={coverAudioFile} kind="audio" label={tr('file.preview')} />
      </section>

      <section class="yue2-cover-card">
        <div class="yue2-cover-head">
          <div>
            <strong>ABC score editor</strong>
            <small>Edit the extracted score here. The sheet preview updates from this ABC.</small>
          </div>
        </div>
        <textarea id="param-abc" rows="8"
          value={abcDraft}
          placeholder="X:1&#10;T:Cover melody&#10;M:4/4&#10;L:1/8&#10;K:C&#10;C D E F | G A B c |"
          on:input={(event) => updateAbc(event.currentTarget.value)}></textarea>
        {#if specByName('abc_file')}
          <div class="yue2-field">
            <label for="param-abc_file">{localizedParameterText(specByName('abc_file')!, 'label', tr)}</label>
            <input id="param-abc_file" type="text"
              value={String(advancedValues.abc_file ?? '')}
              placeholder={localizedParameterText(specByName('abc_file')!, 'placeholder', tr)}
              on:input={(event) => setNamedParameter('abc_file', event.currentTarget.value)} />
          </div>
        {/if}
      </section>

      <section class="yue2-cover-card yue2-sheet-card">
        <div class="yue2-cover-head">
          <div>
            <strong>Sheet preview</strong>
            <small>Rendered from the editable ABC score.</small>
          </div>
        </div>
        <div class="yue2-sheet-preview" bind:this={abcPreviewElement}>
          {#if !abcDraft.trim()}<span>Extract or paste ABC to preview the score.</span>{/if}
        </div>
        {#if abcRenderError}<span class="yue2-error">{abcRenderError}</span>{/if}
      </section>
    </div>
  </details>

  <details class="yue2-details">
    <summary>Yue2 semantic sampling <span>{semanticSpecs.length}</span></summary>
    <div class="yue2-grid">
      {#each semanticSpecs as spec}
        <div class="yue2-field">
          <label for={'param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
          {#if spec.type === 'slider'}
            <div class="yue2-range">
              <input id={'param-' + spec.name} type="range" min={spec.minimum} max={spec.maximum} step={spec.step}
                value={Number(advancedValues[spec.name] ?? spec.default)}
                on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
              <output>{String(advancedValues[spec.name])}</output>
            </div>
          {:else}
            <input id={'param-' + spec.name} type="number" min={spec.minimum} max={spec.maximum} step={spec.step}
              value={String(advancedValues[spec.name] ?? '')}
              on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
          {/if}
        </div>
      {/each}
    </div>
  </details>

  <details class="yue2-details">
    <summary>Yue2 ABC planner sampling <span>{plannerSpecs.length}</span></summary>
    <div class="yue2-grid">
      {#each plannerSpecs as spec}
        <div class="yue2-field">
          <label for={'param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
          {#if spec.type === 'slider'}
            <div class="yue2-range">
              <input id={'param-' + spec.name} type="range" min={spec.minimum} max={spec.maximum} step={spec.step}
                value={Number(advancedValues[spec.name] ?? spec.default)}
                on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
              <output>{String(advancedValues[spec.name])}</output>
            </div>
          {:else}
            <input id={'param-' + spec.name} type="number" min={spec.minimum} max={spec.maximum} step={spec.step}
              value={String(advancedValues[spec.name] ?? '')}
              on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
          {/if}
        </div>
      {/each}
    </div>
  </details>
</div>

<style>
  .yue2-form {
    display: grid;
    gap: 14px;
  }

  .yue2-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px 12px;
    align-items: start;
  }

  .yue2-grid-core,
  .yue2-grid-components {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .yue2-field {
    min-width: 0;
  }

  .yue2-field.wide {
    grid-column: 1 / -1;
  }

  .yue2-field label {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
    min-height: 16px;
    margin: 0 0 5px;
    color: var(--ink);
    font-size: 11px;
    font-weight: 700;
    line-height: 1.3;
  }

  .yue2-field label span,
  .yue2-field small {
    color: var(--muted);
    font-size: 10px;
    font-weight: 500;
    line-height: 1.35;
  }

  .yue2-field small {
    display: block;
    min-height: 14px;
    margin-top: 5px;
  }

  .yue2-form :global(input),
  .yue2-form :global(select),
  .yue2-form :global(textarea) {
    width: 100%;
    min-width: 0;
    font-size: 12px;
    line-height: 1.35;
  }

  .yue2-form :global(textarea) {
    resize: vertical;
  }

  .yue2-range {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 62px;
    gap: 10px;
    align-items: center;
  }

  .yue2-range :global(input) {
    padding: 0;
    accent-color: var(--cyan);
  }

  .yue2-range output {
    color: var(--cyan);
    font: 12px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    text-align: right;
  }

  .yue2-details {
    margin: 0;
  }

  .yue2-details > .yue2-grid {
    padding: 0 10px 10px;
  }

  .yue2-cover {
    display: grid;
    gap: 12px;
    padding: 0 10px 10px;
  }

  .yue2-cover-card {
    display: grid;
    gap: 10px;
    min-width: 0;
    padding: 10px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--control-bg);
  }

  .yue2-sheet-card {
    overflow: hidden;
  }

  .yue2-sheet-preview {
    min-height: 180px;
    max-height: 420px;
    overflow: auto;
    padding: 10px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #f8fbff;
    color: #0e1726;
  }

  .yue2-sheet-preview span {
    color: var(--muted);
    font-size: 12px;
  }

  .yue2-sheet-preview :global(svg) {
    display: block;
    max-width: 100%;
  }

  .yue2-cover-head {
    display: flex;
    align-items: start;
    justify-content: space-between;
    gap: 12px;
  }

  .yue2-cover-head strong {
    display: block;
    color: var(--text-strong);
    font-size: 12px;
    line-height: 1.3;
  }

  .yue2-cover-head small {
    display: block;
    margin-top: 3px;
    color: var(--muted);
    font-size: 10px;
    line-height: 1.35;
  }

  .yue2-cover-head button {
    width: auto;
    min-width: 110px;
  }

  .yue2-cover-picker {
    margin: 0;
  }

  .yue2-unload-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    line-height: 1.4;
  }

  .yue2-unload-toggle input {
    appearance: none;
    position: relative;
    flex: 0 0 34px;
    width: 34px;
    height: 20px;
    padding: 0;
    margin: 0;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: var(--muted);
    cursor: pointer;
  }

  .yue2-unload-toggle input::before {
    content: '';
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: white;
  }

  .yue2-unload-toggle input:checked {
    background: var(--cyan);
  }

  .yue2-unload-toggle input:checked::before {
    transform: translateX(14px);
  }

  .yue2-unload-toggle input:focus-visible {
    outline: 2px solid var(--cyan);
    outline-offset: 2px;
  }

  .yue2-unload-toggle input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .yue2-error {
    color: var(--danger);
  }

  .yue2-details > summary {
    min-height: 42px;
    font-size: 11px;
    line-height: 1.3;
  }

  @media (max-width: 760px) {
    .yue2-grid,
    .yue2-grid-components,
    .yue2-grid-core {
      grid-template-columns: 1fr;
    }
  }
</style>
