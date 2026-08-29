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
    voicePreviewUrl,
    uploadWav
  } from '$lib/api';
  import type { Translator } from '$lib/i18n';
  import MediaPreview from '$lib/MediaPreview.svelte';
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
  export let tr: Translator = (key, _values, fallback = key) => fallback;

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
  let arenaSourceInput: HTMLInputElement | null = null;
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
  let status = '';

  let arenaCatalog: CatalogEntry[] = [];
  let arenaEntry: CatalogEntry | undefined;
  let arenaPackageChoices: InstallPackageChoice[] = [];
  let arenaResultItems: ArenaItem[] = [];
  let serverVoices: string[] = [];
  let builtinVoices: string[] = [];
  let arenaTitle = '';
  let arenaSubtitle = '';
  let arenaModeText = '';

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
  $: arenaBuiltinVoicePreview = arenaVoiceMode === 'builtin' && arenaBuiltinVoice
    ? voicePreviewUrl(arenaBuiltinVoice)
    : '';
  $: if (arenaMode === 'vc' && arenaVoiceMode !== 'reference') {
    arenaVoiceMode = 'reference';
    arenaBuiltinVoice = '';
  }
  $: arenaTitle = arenaMode === 'tts'
    ? tr('arena.title.tts')
    : arenaMode === 'vc'
      ? tr('arena.title.vc')
      : tr('arena.title.asr');
  $: arenaSubtitle = arenaMode === 'tts'
    ? tr('arena.subtitle.tts')
    : arenaMode === 'vc'
      ? tr('arena.subtitle.vc')
      : tr('arena.subtitle.asr');
  $: arenaModeText = arenaMode === 'tts'
    ? tr('arena.mode.tts')
    : arenaMode === 'vc'
      ? tr('arena.mode.vc')
      : tr('arena.mode.asr');
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
      throw new Error(tr('arena.error.seed'));
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

  function arenaStatusLabel(status: ArenaItemStatus) {
    return tr(`arena.itemStatus.${status}`, {}, status);
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

  function clearArenaSource() {
    arenaSourceFile = null;
    if (arenaSourceInput) arenaSourceInput.value = '';
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
      status = tr('arena.status.duplicate', {
        model: arenaEntry.display_name,
        package: choice?.label || ''
      });
      return;
    }
    arenaItems = [...arenaItems, {
      id: crypto.randomUUID(),
      entryId: arenaEntry.id,
      packageId: choice?.id,
      label: arenaEntry.display_name,
      packageLabel: choice?.label || tr('arena.package.configured'),
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
      if (Array.isArray(raw) || raw === null) throw new Error(tr('arena.error.jsonObject'));
      return raw as Record<string, unknown>;
    } catch (error) {
      throw new Error(tr('arena.error.invalidJson', {
        error: error instanceof Error ? error.message : String(error)
      }));
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
        throw new Error(tr('arena.error.unregistered'));
      }
      return;
    }
    if (choice && !packageIsAvailable(entry, choice)) {
      throw new StatusWarning(tr('arena.error.notDownloaded', { package: choice.label }));
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
      throw new Error(tr('arena.error.loadFailed'));
    }
  }

  async function runArenaItem(item: ArenaItem, sharedOptions: Record<string, unknown>) {
    const entry = arenaItemEntry(item);
    if (!entry) {
      updateArenaItem(item.id, { status: 'failed', error: tr('arena.error.missingCatalog') });
      return;
    }
    const choice = arenaItemPackage(entry, item);
    const started = performance.now();
    try {
      updateArenaItem(item.id, { status: 'loading', note: tr('arena.status.loadingModel'), error: '' });
      await ensureArenaItemLoaded(entry, choice);
      updateArenaItem(item.id, { status: 'running', note: tr('arena.status.runningRequest') });

      const options = { ...(entry.default_options || {}), ...sharedOptions };
      const builtinVoice = selectedBuiltinVoice();
      const usingReferenceAudio = arenaVoiceMode === 'reference' && Boolean(arenaVoiceFile);
      if (arenaEntryRequiresTargetVoice(entry) && !usingReferenceAudio) {
        updateArenaItem(item.id, {
          status: 'skipped',
          note: tr('arena.note.skippedTargetVoice'),
          error: ''
        });
        return;
      }
      const needsReferenceText = arenaMode === 'tts' &&
        arenaEntryRequiresReferenceText(entry, usingReferenceAudio);
      if (needsReferenceText && !arenaReferenceText.trim()) {
        updateArenaItem(item.id, {
          status: 'skipped',
          note: tr('arena.note.skippedReferenceText'),
          error: ''
        });
        return;
      }
      const voiceRef = usingReferenceAudio && arenaEntryAcceptsVoice(entry)
        ? await arenaAudioPath(arenaVoiceFile)
        : undefined;
      let note = '';
      if (builtinVoice) note = tr('arena.note.builtinVoice', { voice: arenaBuiltinVoice });
      else if (usingReferenceAudio && voiceRef) note = tr('arena.note.referenceVoice');
      else if (usingReferenceAudio && !arenaEntryAcceptsVoice(entry)) note = tr('arena.note.referenceUnsupported');
      else if (entry.default_voice) note = tr('arena.note.defaultVoice', { voice: entry.default_voice });
      else if (arenaEntryRequiresVoice(entry) && !builtinVoice) {
        updateArenaItem(item.id, {
          status: 'skipped',
          note: tr('arena.note.skippedReferenceVoice'),
          error: ''
        });
        return;
      }

      if (arenaMode === 'tts') {
        if (!arenaText.trim()) throw new StatusWarning(tr('arena.error.enterTtsText'));
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
        if (!source) throw new StatusWarning(tr('arena.error.chooseAsrSource'));
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
      if (!source) throw new StatusWarning(tr('arena.error.chooseVcSource'));
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
      if (!audio) throw new Error(tr('arena.error.noAudio'));
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
      status = tr('arena.status.addModel');
      return;
    }
    running = true;
    aborter = new AbortController();
    clearArenaOutputs();
    status = tr('arena.status.running', { mode: arenaModeText });
    log(status);
    try {
      const sharedOptions = parseArenaOptions();
      for (const item of arenaItems) {
        if (aborter?.signal.aborted) break;
        await runArenaItem(item, sharedOptions);
      }
      status = tr('arena.status.complete');
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
    <p class="eyebrow">{tr('arena.eyebrow')}</p>
    <h1>{arenaTitle}</h1>
    <p>{arenaSubtitle}</p>
  </div>
  <div class="arena-mode-switch">
    <button class:active={arenaMode === 'tts'} on:click={() => arenaMode = 'tts'}>{tr('arena.mode.tts')}</button>
    <button class:active={arenaMode === 'vc'} on:click={() => arenaMode = 'vc'}>{tr('arena.mode.vc')}</button>
    <button class:active={arenaMode === 'asr'} on:click={() => arenaMode = 'asr'}>{tr('arena.mode.asr')}</button>
  </div>
</section>

<div class="arena-grid">
  <section class="panel arena-inputs">
    <div class="section-title">
      <div><span>{tr('arena.input.label')}</span><h2>{tr('arena.input.title')}</h2></div>
      <span class="task-chip">{arenaModeText}</span>
    </div>

    {#if arenaMode === 'tts'}
      <label for="arena-text">{tr('request.text')}</label>
      <textarea id="arena-text" rows="5" bind:value={arenaText}
        placeholder={tr('arena.input.textPlaceholder')}></textarea>
    {:else}
      <label for="arena-source">{tr('request.sourceAudio')}</label>
      <input id="arena-source" class="file file-native" type="file" accept="audio/*"
        bind:this={arenaSourceInput}
        on:change={(event) => arenaSourceFile = event.currentTarget.files?.[0] || null} />
      <label class="file-picker" for="arena-source"><strong>{tr('file.choose')}</strong><span>{arenaSourceFile?.name || tr('file.none')}</span></label>
      <div class="media-actions">
        <button type="button" disabled={!arenaSourceFile} on:click={clearArenaSource}>{tr('file.clear')}</button>
        {#if arenaSourceFile}<span>{arenaSourceFile.name}</span>{/if}
      </div>
      <MediaPreview file={arenaSourceFile} kind="audio" label={tr('file.preview')} />
    {/if}

    {#if arenaMode === 'asr'}
      <label for="arena-ground-truth">{tr('arena.input.groundTruth')} <span>{tr('request.optional')}</span></label>
      <textarea id="arena-ground-truth" rows="4" bind:value={arenaGroundTruthText}
        placeholder={tr('arena.input.groundTruthPlaceholder')}></textarea>
    {/if}

    <div class="field-grid">
      <div>
        <label for="arena-language">{tr('request.language')} <span>{tr('request.optional')}</span></label>
        <input id="arena-language" bind:value={arenaLanguage} placeholder={tr('request.autoLanguage')} />
      </div>
      {#if arenaMode !== 'asr'}
        <div>
          <label for="arena-seed">{tr('request.seed')}</label>
          <input id="arena-seed" type="number" min="-1" max="4294967295" step="1" bind:value={arenaSeed} />
        </div>
      {/if}
    </div>

    {#if arenaMode === 'tts'}
      <div class="reference-input-grid">
        <div>
          <label for="arena-voice-mode">{tr('arena.voice.label')} <span>{tr('arena.shared')}</span></label>
          <select id="arena-voice-mode" value={arenaVoiceMode}
            on:change={(event) => chooseArenaVoiceMode(event.currentTarget.value as ArenaVoiceMode)}>
            <option value="default">{tr('arena.voice.modelDefault')}</option>
            {#if builtinVoices.length}<option value="builtin">{tr('arena.voice.builtin')}</option>{/if}
            <option value="reference">{tr('arena.voice.reference')}</option>
          </select>
        </div>
        <div>
          <label for="arena-reference-text">{tr('voice.referenceText')} <span>{tr('request.optional')}</span></label>
          <textarea id="arena-reference-text" rows="2" bind:value={arenaReferenceText}></textarea>
        </div>
      </div>
    {/if}

    {#if arenaVoiceMode === 'builtin'}
      <label for="arena-builtin-voice">{tr('arena.voice.builtin')}</label>
      <select id="arena-builtin-voice" bind:value={arenaBuiltinVoice}>
        {#each builtinVoices as voice}
          <option value={voice}>{voice}</option>
        {/each}
      </select>
      <div class="quick-voice-note">
        {tr('arena.voice.builtinNote')}
      </div>
      <MediaPreview src={arenaBuiltinVoicePreview} name={arenaBuiltinVoice} kind="audio" label={tr('file.preview')} />
    {:else if arenaMode === 'vc' || arenaVoiceMode === 'reference'}
      <label for="arena-voice">{arenaMode === 'vc' ? tr('arena.voice.targetSpeaker') : tr('voice.reference')} <span>{arenaMode === 'vc' ? tr('voice.required') : tr('voice.optional')}</span></label>
      <input id="arena-voice" class="file file-native" type="file" accept="audio/*"
        bind:this={arenaVoiceInput}
        on:change={(event) => chooseArenaReferenceVoice(event.currentTarget.files?.[0] || null)} />
      <label class="file-picker" for="arena-voice"><strong>{tr('file.choose')}</strong><span>{arenaVoiceFile?.name || tr('file.none')}</span></label>
      <MediaPreview file={arenaVoiceFile} kind="audio" label={tr('file.preview')} />
    {/if}
    {#if arenaMode !== 'asr'}
      <div class="media-actions">
        <button type="button"
          disabled={arenaVoiceMode === 'default' && !arenaVoiceFile && !arenaReferenceText.trim()}
          on:click={clearArenaReference}>{arenaMode === 'vc' ? tr('arena.voice.clearTarget') : tr('arena.voice.clearReference')}</button>
        {#if arenaVoiceFile}<span>{arenaVoiceFile.name}</span>{/if}
      </div>
    {/if}

    <details>
      <summary>{tr('arena.options.shared')} <span>JSON</span></summary>
      <textarea class="code" rows="3" bind:value={arenaOptionsJson}></textarea>
    </details>
  </section>

  <section class="panel arena-queue">
    <div class="section-title">
      <div><span>{tr('arena.queue.label')}</span><h2>{tr('arena.queue.title')}</h2></div>
      <span class="task-chip">{arenaItems.length}</span>
    </div>

    <div class="arena-add-row">
      <div>
        <label for="arena-model">{tr('studio.model')}</label>
        <select id="arena-model" bind:value={arenaModelId}
          on:change={() => arenaPackageId = ''}>
          {#each arenaCatalog as entry}
            <option value={entry.id} disabled={!entrySelectable(entry)}>{entry.display_name}</option>
          {/each}
        </select>
      </div>
      <div>
        <label for="arena-package">{tr('arena.package.label')}</label>
        <select id="arena-package" bind:value={arenaPackageId}>
          {#each arenaPackageChoices as choice}
            <option value={choice.id} disabled={arenaEntry ? !packageIsAvailable(arenaEntry, choice) : true}>
              {choice.label}{arenaEntry && packageIsAvailable(arenaEntry, choice) ? '' : ` · ${tr('studio.notDownloaded')}`}
            </option>
          {/each}
          {#if !arenaPackageChoices.length && arenaEntry}
            <option value="">{tr('arena.package.configured')}</option>
          {/if}
        </select>
      </div>
      <button type="button" disabled={!arenaEntry} on:click={addArenaItem}>{tr('arena.queue.add')}</button>
    </div>

    {#if arenaItems.length}
      <div class="arena-items">
        {#each arenaItems as item}
          <article class:done={item.status === 'done'} class:failed={item.status === 'failed'} class:skipped={item.status === 'skipped'}>
            <div>
              <strong>{item.label}</strong>
              <span>{item.packageLabel}</span>
            </div>
            <span class="arena-status">{arenaStatusLabel(item.status)}</span>
            <button type="button" disabled={running} on:click={() => removeArenaItem(item.id)}>{tr('arena.queue.remove')}</button>
          </article>
        {/each}
      </div>
    {:else}
      <div class="empty-output"><p>{tr('arena.queue.empty')}</p></div>
    {/if}

    <div class="runbar arena-runbar">
      <button class="run" disabled={running || !arenaItems.length} on:click={runArena}>
        <span>{running ? tr('run.working') : tr('arena.run')}</span>
        <kbd>Ctrl Enter</kbd>
      </button>
      <button disabled={!running} on:click={cancel}>{tr('run.cancel')}</button>
      <button disabled={running || !arenaItems.length} on:click={resetArena}>{tr('arena.queue.clear')}</button>
    </div>
    <div class="status" class:busy={running}>{status || tr('status.ready')}</div>
  </section>

  <section class="panel arena-results">
    <div class="section-title">
      <div><span>{tr('result.label')}</span><h2>{tr('arena.results.title')}</h2></div>
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
              <span>{arenaStatusLabel(item.status)}</span>
            </header>
            {#if item.outputUrl}
              <audio controls src={item.outputUrl}></audio>
            {/if}
            {#if item.outputText}
              <pre class="arena-text-output">{item.outputText}</pre>
            {/if}
            {#if item.outputUrl || item.outputText || item.wallMs || item.rtf || item.wer}
              <div class="arena-metrics">
                <span>{tr('arena.metric.wall')} {item.wallMs || '?'}</span>
                <span>{tr('arena.metric.rtf')} {item.rtf || '?'}</span>
                {#if item.wer}<span>{tr('arena.metric.wer')} {item.wer}</span>{/if}
              </div>
            {/if}
            {#if item.note}<p>{item.note}</p>{/if}
            {#if item.error}<p class="arena-error">{item.error}</p>{/if}
          </article>
        {/each}
      </div>
    {:else}
      <div class="empty-output"><div class="wave">~</div><p>{tr('arena.results.empty')}</p></div>
    {/if}
  </section>
</div>
