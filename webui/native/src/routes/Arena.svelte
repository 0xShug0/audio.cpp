<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { browserDecodeToWav } from '$lib/audio';
  import {
    availableVoices,
    base64AudioUrl,
    loadModel,
    runTask,
    speech,
    transcription,
    unloadModel,
    uploadWav
  } from '$lib/api';
  import type { CatalogEntry, InstallPackageChoice, LoadedModel, ServerHealth } from '$lib/types';

  export let activeCatalog: CatalogEntry[] = [];
  export let loadedModels: LoadedModel[] = [];
  export let server: ServerHealth | null = null;
  export let modelsFolder = '';
  export let maxTokens = 1024;
  export let entrySelectable: (entry: CatalogEntry) => boolean = () => true;
  export let studioPackageSlots: (entry: CatalogEntry) =>
    Array<{ key: string; label: string; choice?: InstallPackageChoice }> = () => [];
  export let packageIsAvailable: (entry: CatalogEntry, choice: InstallPackageChoice) => boolean = () => false;
  export let packageSessionOptionsMatch:
    (entry: CatalogEntry, choice: InstallPackageChoice, model: LoadedModel) => boolean = () => true;
  export let supportsMaxTokens: (entry: CatalogEntry) => boolean = () => false;
  export let supportsRequestOption: (entry: CatalogEntry, option: string) => boolean = () => false;
  export let requiresRequestOption: (entry: CatalogEntry, option: string) => boolean = () => false;
  export let refresh: () => Promise<void> = async () => {};
  export let log: (message: string) => void = () => {};

  type ArenaMode = 'tts' | 'vc' | 'asr';
  type ArenaVoiceMode = 'default' | 'builtin' | 'reference';
  type ArenaItemStatus = 'queued' | 'loading' | 'running' | 'done' | 'failed' | 'skipped';

  interface ArenaItem {
    id: string;
    entryId: string;
    packageId?: string;
    label: string;
    packageLabel: string;
    status: ArenaItemStatus;
    note: string;
    error: string;
    outputUrl: string;
    outputText: string;
    wallMs: string;
    rtf: string;
    wer: string;
  }

  class StatusWarning extends Error {}

  let arenaMode: ArenaMode = 'tts';
  let arenaModelId = '';
  let arenaPackageId = '';
  let arenaText = '';
  let arenaLanguage = '';
  let arenaSeed = 1234;
  let arenaSourceFile: File | null = null;
  let arenaGroundTruthText = '';
  let arenaVoiceMode: ArenaVoiceMode = 'default';
  let arenaBuiltinVoice = '';
  let arenaVoiceFile: File | null = null;
  let arenaVoiceInput: HTMLInputElement | null = null;
  let arenaReferenceText = '';
  let arenaOptionsJson = '{}';
  let arenaItems: ArenaItem[] = [];
  let running = false;
  let aborter: AbortController | null = null;
  let status = 'Ready';

  let arenaCatalog: CatalogEntry[] = [];
  let arenaEntry: CatalogEntry | undefined;
  let arenaPackageChoices: InstallPackageChoice[] = [];
  let arenaResultItems: ArenaItem[] = [];
  let serverVoices: string[] = [];
  let builtinVoices: string[] = [];

  const demoVoiceSources: Record<string, string> = {
    demo_1_man: 'demo_1_man',
    demo_2_man: 'demo_2_man',
    demo_3_woman: 'demo_3_woman',
    demo_4_woman: 'demo_4_woman'
  };

  $: arenaCatalog = activeCatalog.filter((entry) =>
    arenaMode === 'tts'
      ? ['tts', 'clon'].includes(entry.task)
      : arenaMode === 'vc'
        ? entry.task === 'vc'
        : entry.task === 'asr');
  $: if (!arenaModelId || !arenaCatalog.some((entry) => entry.id === arenaModelId)) {
    arenaModelId = arenaCatalog[0]?.id || '';
  }
  $: arenaEntry = arenaCatalog.find((entry) => entry.id === arenaModelId) || arenaCatalog[0];
  $: arenaPackageChoices = arenaEntry ? studioPackageSlots(arenaEntry)
    .map((slot) => slot.choice)
    .filter((choice): choice is InstallPackageChoice => Boolean(choice)) : [];
  $: if (arenaPackageChoices.length &&
      !arenaPackageChoices.some((choice) => choice.id === arenaPackageId)) {
    arenaPackageId = arenaPackageChoices[0].id;
  }
  $: builtinVoices = [...serverVoices]
    .sort((left, right) => left.localeCompare(right, 'en', { sensitivity: 'base', numeric: true }));
  $: if (arenaVoiceMode === 'builtin' && !builtinVoices.includes(arenaBuiltinVoice)) {
    arenaBuiltinVoice = builtinVoices[0] || '';
  }
  $: if (arenaVoiceMode === 'builtin' && !builtinVoices.length) {
    arenaVoiceMode = 'default';
  }
  $: if (arenaMode === 'vc' && arenaVoiceMode !== 'reference') {
    arenaVoiceMode = 'reference';
    arenaBuiltinVoice = '';
  }
  $: arenaResultItems = arenaItems.every((item) =>
    ['done', 'failed', 'skipped'].includes(item.status))
    ? [...arenaItems].sort(compareArenaResultItems)
    : arenaItems;

  async function refreshArenaVoices() {
    try {
      serverVoices = await availableVoices();
    } catch (error) {
      serverVoices = [];
      log(`Arena voices unavailable: ${error instanceof Error ? error.message : error}`);
    }
  }

  function resolveCatalogPath(path: string) {
    if (!modelsFolder) return path;
    const normalized = path.replace(/\\/g, '/');
    if (normalized === 'models') return modelsFolder;
    if (!normalized.startsWith('models/')) return path;
    const relative = normalized.slice('models/'.length);
    const separator = modelsFolder.includes('\\') ? '\\' : '/';
    return `${modelsFolder.replace(/[\\/]+$/, '')}${separator}${relative.replace(/\//g, separator)}`;
  }

  function comparablePath(path: string) {
    return path.replace(/\\/g, '/').replace(/\/$/, '').toLowerCase();
  }

  function modelPathForChoice(entry: CatalogEntry, choice?: InstallPackageChoice) {
    if (server && !server.ui_management) return entry.path;
    return resolveCatalogPath(choice?.path || entry.path);
  }

  function mergedSessionOptionsForChoice(entry: CatalogEntry, choice?: InstallPackageChoice) {
    return { ...(entry.session_options || {}), ...(choice?.session_options || {}) };
  }

  function resolveRequestSeed(value: number) {
    if (!Number.isInteger(value) || value < -1 || value > 0xffffffff) {
      throw new Error('Seed must be -1 or an unsigned 32-bit integer (0 to 4294967295).');
    }
    if (value >= 0) return value;
    const random = new Uint32Array(1);
    globalThis.crypto.getRandomValues(random);
    return random[0];
  }

  function arenaEntryAcceptsVoice(entry: CatalogEntry) {
    return (['clon', 'vc', 'svc'].includes(entry.task) && entry.family !== 'rvc') ||
      (entry.task === 's2s' && entry.family === 'personaplex') ||
      (entry.task === 'tts' && !['supertonic'].includes(entry.family));
  }

  function arenaEntryRequiresVoice(entry: CatalogEntry) {
    return ['clon', 'vc', 'svc'].includes(entry.task) && entry.family !== 'rvc';
  }

  function isQwenBaseTts(entry: CatalogEntry) {
    return entry.task === 'tts' && entry.family === 'qwen3_tts' && !entry.id.includes('custom');
  }

  function arenaEntryRequiresReferenceText(entry: CatalogEntry, usingReferenceAudio: boolean) {
    return requiresRequestOption(entry, 'reference_text') ||
      (usingReferenceAudio && isQwenBaseTts(entry));
  }

  function arenaEntryRequiresTargetVoice(entry: CatalogEntry) {
    return arenaMode === 'vc' && entry.task === 'vc' && entry.family !== 'rvc';
  }

  function selectedBuiltinVoice() {
    return arenaMode === 'tts' && arenaVoiceMode === 'builtin' && arenaBuiltinVoice
      ? demoVoiceSources[arenaBuiltinVoice] || arenaBuiltinVoice
      : '';
  }

  function normalizedWords(text: string) {
    return text
      .normalize('NFKC')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}']+/gu, ' ')
      .trim()
      .split(/\s+/)
      .filter(Boolean);
  }

  function wordErrorRate(reference: string, hypothesis: string) {
    const ref = normalizedWords(reference);
    const hyp = normalizedWords(hypothesis);
    if (!ref.length) return '';
    const previous = Array.from({ length: hyp.length + 1 }, (_, index) => index);
    const current = new Array<number>(hyp.length + 1);
    for (let refIndex = 1; refIndex <= ref.length; refIndex += 1) {
      current[0] = refIndex;
      for (let hypIndex = 1; hypIndex <= hyp.length; hypIndex += 1) {
        const substitution = previous[hypIndex - 1] + (ref[refIndex - 1] === hyp[hypIndex - 1] ? 0 : 1);
        const deletion = previous[hypIndex] + 1;
        const insertion = current[hypIndex - 1] + 1;
        current[hypIndex] = Math.min(substitution, deletion, insertion);
      }
      for (let index = 0; index <= hyp.length; index += 1) previous[index] = current[index];
    }
    return `${((previous[hyp.length] / ref.length) * 100).toFixed(2)}%`;
  }

  function numericMetric(value: string) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
  }

  function compareArenaResultItems(left: ArenaItem, right: ArenaItem) {
    const leftDone = left.status === 'done';
    const rightDone = right.status === 'done';
    if (leftDone !== rightDone) return leftDone ? -1 : 1;
    if (leftDone && rightDone) {
      const rtfDelta = numericMetric(left.rtf) - numericMetric(right.rtf);
      if (rtfDelta !== 0) return rtfDelta;
    }
    return arenaItems.indexOf(left) - arenaItems.indexOf(right);
  }

  function chooseArenaVoiceMode(mode: ArenaVoiceMode) {
    arenaVoiceMode = mode;
    if (mode !== 'reference') {
      arenaVoiceFile = null;
      arenaReferenceText = '';
      if (arenaVoiceInput) arenaVoiceInput.value = '';
    }
    if (mode !== 'builtin') {
      arenaBuiltinVoice = '';
    }
  }

  function chooseArenaReferenceVoice(file: File | null) {
    arenaVoiceFile = file;
    if (file) arenaVoiceMode = 'reference';
  }

  function clearArenaReference() {
    arenaVoiceMode = 'default';
    arenaBuiltinVoice = '';
    arenaVoiceFile = null;
    arenaReferenceText = '';
    if (arenaVoiceInput) arenaVoiceInput.value = '';
  }

  function arenaItemEntry(item: ArenaItem) {
    return activeCatalog.find((entry) => entry.id === item.entryId);
  }

  function arenaItemPackage(entry: CatalogEntry, item: ArenaItem) {
    return (entry.install_packages || []).find((choice) => choice.id === item.packageId);
  }

  function updateArenaItem(id: string, patch: Partial<ArenaItem>) {
    arenaItems = arenaItems.map((item) => item.id === id ? { ...item, ...patch } : item);
  }

  function clearArenaOutputs() {
    for (const item of arenaItems) {
      if (item.outputUrl) URL.revokeObjectURL(item.outputUrl);
    }
    arenaItems = arenaItems.map((item) => ({
      ...item,
      status: 'queued',
      note: '',
      error: '',
      outputUrl: '',
      outputText: '',
      wallMs: '',
      rtf: '',
      wer: ''
    }));
  }

  function resetArena() {
    for (const item of arenaItems) {
      if (item.outputUrl) URL.revokeObjectURL(item.outputUrl);
    }
    arenaItems = [];
  }

  function addArenaItem() {
    if (!arenaEntry) return;
    const choice = arenaPackageChoices.find((candidate) => candidate.id === arenaPackageId);
    const exists = arenaItems.some((item) =>
      item.entryId === arenaEntry?.id && item.packageId === choice?.id);
    if (exists) {
      status = `${arenaEntry.display_name} ${choice?.label || ''} is already in the arena.`;
      return;
    }
    arenaItems = [...arenaItems, {
      id: crypto.randomUUID(),
      entryId: arenaEntry.id,
      packageId: choice?.id,
      label: arenaEntry.display_name,
      packageLabel: choice?.label || 'Configured',
      status: 'queued',
      note: '',
      error: '',
      outputUrl: '',
      outputText: '',
      wallMs: '',
      rtf: '',
      wer: ''
    }];
  }

  function removeArenaItem(id: string) {
    const item = arenaItems.find((candidate) => candidate.id === id);
    if (item?.outputUrl) URL.revokeObjectURL(item.outputUrl);
    arenaItems = arenaItems.filter((candidate) => candidate.id !== id);
  }

  function parseArenaOptions() {
    try {
      const raw = JSON.parse(arenaOptionsJson || '{}');
      if (Array.isArray(raw) || raw === null) throw new Error('must be an object');
      return raw as Record<string, unknown>;
    } catch (error) {
      throw new Error(`Arena options JSON is invalid: ${error instanceof Error ? error.message : error}`);
    }
  }

  async function arenaAudioPath(file: File | null) {
    if (!file) return undefined;
    const wav = await browserDecodeToWav(file);
    return uploadWav(wav, aborter?.signal);
  }

  async function ensureArenaItemLoaded(entry: CatalogEntry, choice?: InstallPackageChoice) {
    if (!server?.ui_management) {
      if (!loadedModels.some((model) => model.id === entry.id && model.loaded)) {
        throw new Error('Configured model is not registered by this server.');
      }
      return;
    }
    if (choice && !packageIsAvailable(entry, choice)) {
      throw new StatusWarning(`${choice.label} is not downloaded.`);
    }
    const path = modelPathForChoice(entry, choice);
    const resident = loadedModels.find((model) => model.id === entry.id && model.loaded &&
      comparablePath(model.path) === comparablePath(path) &&
      (!choice || packageSessionOptionsMatch(entry, choice, model)));
    if (resident) return;

    const replaced = loadedModels.filter((model) => model.loaded &&
      (model.id !== entry.id || comparablePath(model.path) !== comparablePath(path)));
    for (const model of replaced) {
      await unloadModel(model.id);
    }
    if (replaced.length) await refresh();
    await loadModel({
      id: entry.id,
      path,
      family: entry.family,
      task: entry.task,
      mode: entry.mode || 'offline',
      load_options: entry.load_options || {},
      session_options: mergedSessionOptionsForChoice(entry, choice)
    });
    await refresh();
    if (!loadedModels.some((model) => model.id === entry.id && model.loaded &&
      comparablePath(model.path) === comparablePath(path) &&
      (!choice || packageSessionOptionsMatch(entry, choice, model)))) {
      throw new Error('Model did not load.');
    }
  }

  async function runArenaItem(item: ArenaItem, sharedOptions: Record<string, unknown>) {
    const entry = arenaItemEntry(item);
    if (!entry) {
      updateArenaItem(item.id, { status: 'failed', error: 'Model is no longer in the catalog.' });
      return;
    }
    const choice = arenaItemPackage(entry, item);
    const started = performance.now();
    try {
      updateArenaItem(item.id, { status: 'loading', note: 'Loading model', error: '' });
      await ensureArenaItemLoaded(entry, choice);
      updateArenaItem(item.id, { status: 'running', note: 'Running request' });

      const options = { ...(entry.default_options || {}), ...sharedOptions };
      const builtinVoice = selectedBuiltinVoice();
      const usingReferenceAudio = arenaVoiceMode === 'reference' && Boolean(arenaVoiceFile);
      if (arenaEntryRequiresTargetVoice(entry) && !usingReferenceAudio) {
        updateArenaItem(item.id, {
          status: 'skipped',
          note: 'Skipped because this VC model requires target speaker audio.',
          error: ''
        });
        return;
      }
      const needsReferenceText = arenaMode === 'tts' &&
        arenaEntryRequiresReferenceText(entry, usingReferenceAudio);
      if (needsReferenceText && !arenaReferenceText.trim()) {
        updateArenaItem(item.id, {
          status: 'skipped',
          note: 'Skipped because this model requires reference text for the selected voice input.',
          error: ''
        });
        return;
      }
      const voiceRef = usingReferenceAudio && arenaEntryAcceptsVoice(entry)
        ? await arenaAudioPath(arenaVoiceFile)
        : undefined;
      let note = '';
      if (builtinVoice) note = `Used built-in voice: ${arenaBuiltinVoice}.`;
      else if (usingReferenceAudio && voiceRef) note = 'Used reference voice.';
      else if (usingReferenceAudio && !arenaEntryAcceptsVoice(entry)) note = 'Used default voice; reference voice is not supported.';
      else if (entry.default_voice) note = `Used default voice: ${entry.default_voice}.`;
      else if (arenaEntryRequiresVoice(entry) && !builtinVoice) {
        updateArenaItem(item.id, {
          status: 'skipped',
          note: 'Skipped because this model requires a reference voice.',
          error: ''
        });
        return;
      }

      if (arenaMode === 'tts') {
        if (!arenaText.trim()) throw new StatusWarning('Enter text for the TTS arena.');
        const body: Record<string, unknown> = {
          model: entry.id,
          input: arenaText,
          language: arenaLanguage,
          seed: resolveRequestSeed(arenaSeed),
          options
        };
        if (supportsMaxTokens(entry)) body.max_tokens = maxTokens;
        if (voiceRef) body.voice_ref = voiceRef;
        else if (builtinVoice) body.voice = builtinVoice;
        else if (entry.default_voice) body.voice = entry.default_voice;
        if (usingReferenceAudio && arenaReferenceText.trim() &&
            supportsRequestOption(entry, 'reference_text')) {
          body.reference_text = arenaReferenceText;
        }
        const result = await speech(body, aborter?.signal);
        updateArenaItem(item.id, {
          status: 'done',
          note,
          outputUrl: URL.createObjectURL(result.blob),
          wallMs: result.wallMs || `${(performance.now() - started).toFixed(1)}`,
          rtf: result.rtf || ''
        });
        return;
      }

      if (arenaMode === 'asr') {
        const source = await arenaAudioPath(arenaSourceFile);
        if (!source) throw new StatusWarning('Choose source audio for the ASR arena.');
        const result = await transcription({
          model: entry.id,
          audio: source,
          language: arenaLanguage,
          options
        }, aborter?.signal);
        const text = typeof result.text === 'string' ? result.text : '';
        const timing = result.timing as Record<string, unknown> | undefined;
        updateArenaItem(item.id, {
          status: 'done',
          note,
          outputText: text,
          wallMs: typeof timing?.wall_ms === 'number' ? String(timing.wall_ms) : '',
          rtf: typeof timing?.rtf === 'number' ? String(timing.rtf) : '',
          wer: arenaGroundTruthText.trim() ? wordErrorRate(arenaGroundTruthText, text) : ''
        });
        return;
      }

      const source = await arenaAudioPath(arenaSourceFile);
      if (!source) throw new StatusWarning('Choose source audio for the VC arena.');
      const request: Record<string, unknown> = {
        audio: source,
        seed: resolveRequestSeed(arenaSeed),
        options
      };
      if (arenaText.trim()) request.text = arenaText;
      if (arenaLanguage.trim()) request.language = arenaLanguage;
      if (voiceRef) request.voice_ref = voiceRef;
      else if (builtinVoice) request.voice_id = builtinVoice;
      const result = await runTask({ model: entry.id, request }, aborter?.signal);
      const audio = typeof result.audio === 'string'
        ? result.audio
        : Array.isArray(result.named_audio_outputs) &&
          typeof result.named_audio_outputs[0]?.audio === 'string'
          ? result.named_audio_outputs[0].audio
          : '';
      if (!audio) throw new Error('Response did not include audio.');
      const timing = result.timing as Record<string, unknown> | undefined;
      updateArenaItem(item.id, {
        status: 'done',
        note,
        outputUrl: base64AudioUrl(audio),
        wallMs: typeof timing?.wall_ms === 'number' ? String(timing.wall_ms) : '',
        rtf: typeof timing?.rtf === 'number' ? String(timing.rtf) : ''
      });
    } catch (error) {
      updateArenaItem(item.id, {
        status: error instanceof StatusWarning ? 'skipped' : 'failed',
        error: error instanceof Error ? error.message : String(error),
        note: ''
      });
      log(`Arena row failed: ${error instanceof Error ? error.message : error}`);
    }
  }

  export async function runArena() {
    if (running) return;
    if (!arenaItems.length) {
      status = 'Add at least one model to the arena.';
      return;
    }
    running = true;
    aborter = new AbortController();
    clearArenaOutputs();
    status = `Running ${arenaMode === 'tts' ? 'TTS' : arenaMode === 'vc' ? 'voice conversion' : 'ASR'} arena...`;
    log(status);
    try {
      const sharedOptions = parseArenaOptions();
      for (const item of arenaItems) {
        if (aborter?.signal.aborted) break;
        await runArenaItem(item, sharedOptions);
      }
      status = 'Arena run complete.';
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    } finally {
      running = false;
      aborter = null;
    }
  }

  function cancel() {
    aborter?.abort();
  }

  onDestroy(() => {
    aborter?.abort();
    for (const item of arenaItems) {
      if (item.outputUrl) URL.revokeObjectURL(item.outputUrl);
    }
  });

  onMount(refreshArenaVoices);
</script>

<section class="hero arena-hero">
  <div>
    <p class="eyebrow">ARENA</p>
    <h1>{arenaMode === 'tts' ? 'TTS comparison' : arenaMode === 'vc' ? 'Voice conversion comparison' : 'ASR comparison'}</h1>
    <p>Run the same input through selected installed models and package variants, one after another.</p>
  </div>
  <div class="arena-mode-switch">
    <button class:active={arenaMode === 'tts'} on:click={() => arenaMode = 'tts'}>TTS</button>
    <button class:active={arenaMode === 'vc'} on:click={() => arenaMode = 'vc'}>Voice conversion</button>
    <button class:active={arenaMode === 'asr'} on:click={() => arenaMode = 'asr'}>ASR</button>
  </div>
</section>

<div class="arena-grid">
  <section class="panel arena-inputs">
    <div class="section-title">
      <div><span>Input</span><h2>Shared request</h2></div>
      <span class="task-chip">{arenaMode}</span>
    </div>

    {#if arenaMode === 'tts'}
      <label for="arena-text">Text</label>
      <textarea id="arena-text" rows="5" bind:value={arenaText}
        placeholder="Enter one prompt to compare across TTS models"></textarea>
    {:else}
      <label for="arena-source">Source audio</label>
      <input id="arena-source" class="file file-native" type="file" accept="audio/*"
        on:change={(event) => arenaSourceFile = event.currentTarget.files?.[0] || null} />
      <label class="file-picker" for="arena-source"><strong>Choose</strong><span>{arenaSourceFile?.name || 'No file'}</span></label>
    {/if}

    {#if arenaMode === 'asr'}
      <label for="arena-ground-truth">Ground truth text <span>optional</span></label>
      <textarea id="arena-ground-truth" rows="4" bind:value={arenaGroundTruthText}
        placeholder="Paste the expected transcript to calculate WER"></textarea>
    {/if}

    <div class="field-grid">
      <div>
        <label for="arena-language">Language <span>optional</span></label>
        <input id="arena-language" bind:value={arenaLanguage} placeholder="auto" />
      </div>
      {#if arenaMode !== 'asr'}
        <div>
          <label for="arena-seed">Seed</label>
          <input id="arena-seed" type="number" min="-1" max="4294967295" step="1" bind:value={arenaSeed} />
        </div>
      {/if}
    </div>

    {#if arenaMode === 'tts'}
      <div class="reference-input-grid">
        <div>
          <label for="arena-voice-mode">Voice <span>shared</span></label>
          <select id="arena-voice-mode" value={arenaVoiceMode}
            on:change={(event) => chooseArenaVoiceMode(event.currentTarget.value as ArenaVoiceMode)}>
            <option value="default">Model default</option>
            {#if builtinVoices.length}<option value="builtin">Built-in demo voice</option>{/if}
            <option value="reference">Reference audio</option>
          </select>
        </div>
        <div>
          <label for="arena-reference-text">Reference text <span>optional</span></label>
          <textarea id="arena-reference-text" rows="2" bind:value={arenaReferenceText}></textarea>
        </div>
      </div>
    {/if}

    {#if arenaVoiceMode === 'builtin'}
      <label for="arena-builtin-voice">Built-in demo voice</label>
      <select id="arena-builtin-voice" bind:value={arenaBuiltinVoice}>
        {#each builtinVoices as voice}
          <option value={voice}>{voice}</option>
        {/each}
      </select>
      <div class="quick-voice-note">
        Built-in voices are passed as voice IDs for each model. Reference text is not required for this voice source.
      </div>
    {:else if arenaMode === 'vc' || arenaVoiceMode === 'reference'}
      <label for="arena-voice">{arenaMode === 'vc' ? 'Target speaker audio' : 'Reference voice'} <span>{arenaMode === 'vc' ? 'required' : 'optional'}</span></label>
      <input id="arena-voice" class="file file-native" type="file" accept="audio/*"
        bind:this={arenaVoiceInput}
        on:change={(event) => chooseArenaReferenceVoice(event.currentTarget.files?.[0] || null)} />
      <label class="file-picker" for="arena-voice"><strong>Choose</strong><span>{arenaVoiceFile?.name || 'No file'}</span></label>
    {/if}
    {#if arenaMode !== 'asr'}
      <div class="media-actions">
        <button type="button"
          disabled={arenaVoiceMode === 'default' && !arenaVoiceFile && !arenaReferenceText.trim()}
          on:click={clearArenaReference}>{arenaMode === 'vc' ? 'Clear target' : 'Clear reference'}</button>
        {#if arenaVoiceFile}<span>{arenaVoiceFile.name}</span>{/if}
      </div>
    {/if}

    <details>
      <summary>Shared options <span>JSON</span></summary>
      <textarea class="code" rows="3" bind:value={arenaOptionsJson}></textarea>
    </details>
  </section>

  <section class="panel arena-queue">
    <div class="section-title">
      <div><span>Queue</span><h2>Models</h2></div>
      <span class="task-chip">{arenaItems.length}</span>
    </div>

    <div class="arena-add-row">
      <div>
        <label for="arena-model">Model</label>
        <select id="arena-model" bind:value={arenaModelId}
          on:change={() => arenaPackageId = ''}>
          {#each arenaCatalog as entry}
            <option value={entry.id} disabled={!entrySelectable(entry)}>{entry.display_name}</option>
          {/each}
        </select>
      </div>
      <div>
        <label for="arena-package">Package</label>
        <select id="arena-package" bind:value={arenaPackageId}>
          {#each arenaPackageChoices as choice}
            <option value={choice.id} disabled={arenaEntry ? !packageIsAvailable(arenaEntry, choice) : true}>
              {choice.label}{arenaEntry && packageIsAvailable(arenaEntry, choice) ? '' : ' · not downloaded'}
            </option>
          {/each}
          {#if !arenaPackageChoices.length && arenaEntry}
            <option value="">Configured</option>
          {/if}
        </select>
      </div>
      <button type="button" disabled={!arenaEntry} on:click={addArenaItem}>Add</button>
    </div>

    {#if arenaItems.length}
      <div class="arena-items">
        {#each arenaItems as item}
          <article class:done={item.status === 'done'} class:failed={item.status === 'failed'} class:skipped={item.status === 'skipped'}>
            <div>
              <strong>{item.label}</strong>
              <span>{item.packageLabel}</span>
            </div>
            <span class="arena-status">{item.status}</span>
            <button type="button" disabled={running} on:click={() => removeArenaItem(item.id)}>Remove</button>
          </article>
        {/each}
      </div>
    {:else}
      <div class="empty-output"><p>Add installed models or dtype packages to compare.</p></div>
    {/if}

    <div class="runbar arena-runbar">
      <button class="run" disabled={running || !arenaItems.length} on:click={runArena}>
        <span>{running ? 'Working' : 'Run arena'}</span>
        <kbd>Ctrl Enter</kbd>
      </button>
      <button disabled={!running} on:click={cancel}>Cancel</button>
      <button disabled={running || !arenaItems.length} on:click={resetArena}>Clear</button>
    </div>
    <div class="status" class:busy={running}>{status}</div>
  </section>

  <section class="panel arena-results">
    <div class="section-title">
      <div><span>Results</span><h2>Compare outputs</h2></div>
      <span class="task-chip">{arenaItems.filter((item) => item.status === 'done').length}</span>
    </div>

    {#if arenaItems.length}
      <div class="arena-result-list">
        {#each arenaResultItems as item}
          <article class:done={item.status === 'done'} class:failed={item.status === 'failed'} class:skipped={item.status === 'skipped'}>
            <header>
              <div>
                <strong>{item.label}</strong>
                <span>{item.packageLabel}</span>
              </div>
              <span>{item.status}</span>
            </header>
            {#if item.outputUrl}
              <audio controls src={item.outputUrl}></audio>
            {/if}
            {#if item.outputText}
              <pre class="arena-text-output">{item.outputText}</pre>
            {/if}
            {#if item.outputUrl || item.outputText || item.wallMs || item.rtf || item.wer}
              <div class="arena-metrics">
                <span>wall {item.wallMs || '?'}</span>
                <span>rtf {item.rtf || '?'}</span>
                {#if item.wer}<span>wer {item.wer}</span>{/if}
              </div>
            {/if}
            {#if item.note}<p>{item.note}</p>{/if}
            {#if item.error}<p class="arena-error">{item.error}</p>{/if}
          </article>
        {/each}
      </div>
    {:else}
      <div class="empty-output"><div class="wave">~</div><p>No arena results yet.</p></div>
    {/if}
  </section>
</div>
