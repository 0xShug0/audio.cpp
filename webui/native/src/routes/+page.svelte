<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { browserDecodeToWav, concatenateAudioBlobs } from '$lib/audio';
  import {
    base64AudioUrl,
    availableVoices,
    browseDirectories,
    cleanPartialModelInstall,
    deleteModelPackage,
    getModelsRoot,
    health,
    installModelPackage,
    loadModel,
    modelInstallJobs,
    modelPackageSizes,
    models,
    pathStatus,
    runTask,
    setModelsRoot,
    speech,
    stopModelInstall,
    transcription,
    unloadModel,
    uploadWav,
    type ModelInstallJob,
    type ModelPackageSize,
    type DirectoryBrowserResponse
  } from '$lib/api';
  import { catalog, parameterCatalog, taskLabels } from '$lib/catalog';
  import { createTranslator, resolveUiLanguage, uiLanguages } from '$lib/i18n';
  import { defaultChunkBudget, splitTtsChunks } from '$lib/text';
  import type {
    AudioOutput,
    CatalogEntry,
    InstallPackageChoice,
    LoadedModel,
    ParamSpec,
    ServerHealth
  } from '$lib/types';
  import {
    deleteVoice as deleteSavedVoice,
    listVoices,
    saveVoice,
    type SavedVoice
  } from '$lib/voices';
  import '../app.css';

  let tab: 'studio' | 'models' | 'logs' = 'studio';
  let selectedId = catalog[0]?.id || '';
  let selected: CatalogEntry = catalog[0];
  let modelPath = selected?.path || '';
  let loadedModels: LoadedModel[] = [];
  let server: ServerHealth | null = null;
  let installed: boolean | null = null;
  let loadingModel = false;
  let running = false;
  let status = 'Ready';
  let warningStatus = '';
  let errorStatus = '';
  let text = '';
  let language = '';
  let context = '';
  let referenceText = '';
  let instructions = '';
  let lyrics = '';
  let duration = 30;
  let seed = 1234;
  let maxTokens = 1024;
  let sourceFile: File | null = null;
  let voiceFile: File | null = null;
  let vibeVoiceSpeakerFiles: Array<File | null> = [null, null, null, null];
  let voiceInput: HTMLInputElement | null = null;
  let referenceTextFile: File | null = null;
  let referenceTextInput: HTMLInputElement | null = null;
  let advancedJson = '{}';
  let advancedValues: Record<string, unknown> = {};
  let paramSpecs: ParamSpec[] = [];
  let outputAudio: AudioOutput[] = [];
  let outputArtifacts: Array<{ id: string; url: string; extension: string }> = [];
  let outputText = '';
  let outputJson = '';
  let logs: string[] = [];
  let aborter: AbortController | null = null;
  let longText = true;
  let chunkBudget = defaultChunkBudget(selected?.family || '');
  let savedVoices: SavedVoice[] = [];
  let savedVoiceId = '';
  let voiceName = '';
  let recorder: MediaRecorder | null = null;
  let recordingTarget: 'source' | 'voice' | null = null;
  let recordingStream: MediaStream | null = null;
  let liveStream: MediaStream | null = null;
  let liveRecorder: MediaRecorder | null = null;
  let liveRecording = false;
  let liveStopRequested = false;
  let liveQueue: Promise<void> = Promise.resolve();
  let liveChunkNumber = 0;
  let installJobs: Record<string, ModelInstallJob> = {};
  let modelsFolder = '';
  let modelsFolderInput = '';
  let defaultModelsFolder = '';
  let modelsFolderIsDefault = true;
  let applyingModelsFolder = false;
  let folderBrowserOpen = false;
  let folderBrowserLoading = false;
  let folderBrowserError = '';
  let folderBrowser: DirectoryBrowserResponse | null = null;
  let installPoll: number | null = null;
  let selectedPackageIds: Record<string, string> = {};
  let packageSizes: Record<string, ModelPackageSize> = {};
  let packageSizeState: 'idle' | 'running' | 'complete' | 'failed' = 'idle';
  let packageSizePoll: number | null = null;
  let packageSizeRefreshInFlight = false;
  let refreshedInstallFinishes: Record<string, number> = {};
  let quickStartVoices: string[] = [];
  let configuredVoices: string[] = [];
  let bundledVoices: string[] = [];
  let quickStartVoice = '';
  let uiLanguage = 'en';
  let tr = createTranslator(uiLanguage);
  $: tr = createTranslator(uiLanguage);

  const demoVoiceSources: Record<string, string> = {
    demo_1_man: 'demo_1_man',
    demo_2_man: 'demo_2_man',
    demo_3_woman: 'demo_3_woman',
    demo_4_woman: 'demo_4_woman'
  };

  function chooseUiLanguage(code: string) {
    uiLanguage = resolveUiLanguage([code]);
    localStorage.setItem('audiocpp.ui.language', uiLanguage);
    document.documentElement.lang = uiLanguage;
  }

  async function clearLegacyUiCaches() {
    // This server commonly reuses localhost:8080. Remove workers and Cache
    // Storage left by an older application on that origin before Native Studio
    // starts making requests. Do not clear localStorage or IndexedDB: they hold
    // saved voices, model-folder selection, and UI preferences.
    try {
      if ('serviceWorker' in navigator) {
        const registrations = await navigator.serviceWorker.getRegistrations();
        await Promise.all(registrations.map((registration) => registration.unregister()));
      }
      if ('caches' in window) {
        const cacheNames = await window.caches.keys();
        await Promise.all(cacheNames.map((name) => window.caches.delete(name)));
      }
    } catch (error) {
      // Cache cleanup must not prevent an offline/local UI from starting when a
      // browser restricts either API. The no-store response headers still apply.
      console.warn('Unable to clear legacy WebUI caches:', error);
    }
  }

  function workflowLabel(id: string, fallback: string, translate = tr) {
    const translationId = id === 'conversion' ? 'vc' : id === 'separation' ? 'sep' : id;
    return translate(`workflow.${translationId}`, {}, fallback);
  }

  function localizedTaskLabel(task: string | undefined, translate = tr) {
    if (!task) return translate('studio.title');
    return translate(`task.${task}`, {}, taskLabels[task] || task);
  }

  function localizedParameterText(
    spec: ParamSpec,
    field: 'label' | 'info' | 'placeholder',
    translate = tr
  ) {
    const fallback = field === 'label'
      ? (spec.label_en || spec.label || spec.name.replace(/_/g, ' '))
      : field === 'info'
        ? (spec.info_en || spec.info || '')
        : (spec.placeholder_en || spec.placeholder || '');
    return translate(`param.${selected?.family}.${spec.name}.${field}`, {}, fallback);
  }

  function localizedModelHint(translate = tr) {
    const fallback = selected?.input_hint_en || selected?.input_hint || '';
    return translate(`model.${selected?.family}.hint`, {}, fallback);
  }

  // MiniMax-H3 accepts frame counts with the 17n+3 temporal alignment used by
  // its joint audio/video DiT. Keep the user-facing duration and expert frame
  // control synchronized so duration_seconds actually changes output length.
  function miniMaxFramesForDuration(seconds: number) {
    const target = Math.max(1, Number.isFinite(seconds) ? seconds : 1) * 24;
    return Math.max(5, Math.round((target - 3) / 17) * 17 + 3);
  }

  function setDuration(value: number) {
    duration = Math.max(1, Number.isFinite(value) ? value : 1);
    if (selected?.family === 'minimax_h3') {
      advancedValues = { ...advancedValues, num_frames: miniMaxFramesForDuration(duration) };
    }
  }

  function setParameterValue(spec: ParamSpec, value: unknown) {
    advancedValues = { ...advancedValues, [spec.name]: value };
    if (selected?.family === 'minimax_h3' && spec.name === 'num_frames') {
      const frames = Number(value);
      if (Number.isFinite(frames) && frames > 0) duration = frames / 24;
    }
  }

  const workflowTabs = [
    { id: 'tts', label: 'Text to speech', filterLabel: 'TTS', tasks: ['tts', 'clon'] },
    { id: 'asr', label: 'ASR / Transcription', filterLabel: 'ASR', tasks: ['asr'] },
    { id: 'music', label: 'Music generation', filterLabel: 'Music', tasks: ['gen'] },
    { id: 'conversion', label: 'Voice conversion', filterLabel: 'Voice conversion', tasks: ['vc', 'svc', 's2s'] },
    { id: 'separation', label: 'Source separation', filterLabel: 'Separation', tasks: ['sep'] },
    { id: 'analysis', label: 'Audio analysis', filterLabel: 'Analysis', tasks: ['vad', 'diar', 'align', 'spk', 'midi'] },
    { id: 'design', label: 'Voice design', filterLabel: 'Voice design', tasks: ['vdes'] }
  ] as const;

  type WorkflowId = typeof workflowTabs[number]['id'];

  interface ModelGroup {
    family: string;
    label: string;
    entries: CatalogEntry[];
  }

  const familyLabels: Record<string, string> = {
    qwen3_tts: 'Qwen3-TTS',
    irodori_tts: 'Irodori-TTS',
    chatterbox: 'Chatterbox',
    stable_audio: 'Stable Audio 3',
    qwen3_asr: 'Qwen3-ASR',
    vevo2: 'Vevo2',
    seed_vc: 'Seed-VC',
    magpie_tts: 'MagpieTTS',
    meanvc2: 'MeanVC2',
    personaplex: 'PersonaPlex'
  };

  function pathVariantLabel(path: string) {
    const normalized = path.replace(/\\/g, '/');
    const filename = normalized.split('/').filter(Boolean).pop() || '';
    const match = filename.match(/(?:^|[-_])(\d+(?:\.\d+)?[bm])(?:[-_]|$)/i);
    return match ? match[1].toUpperCase() : '';
  }

  function catalogPathMatches(expectedPath: string, actualPath: string) {
    const actual = comparablePath(actualPath);
    const expected = comparablePath(resolveCatalogPath(expectedPath));
    if (actual === expected) return true;
    const relative = comparablePath(expectedPath).replace(/^models\//, '');
    return actual === relative || actual.endsWith(`/${relative}`);
  }

  function catalogEntryMatchesLoadedModel(entry: CatalogEntry, model: LoadedModel) {
    if (entry.family !== model.family || entry.task !== model.task) return false;
    if (catalogPathMatches(entry.path, model.path)) return true;
    return Boolean((entry.install_packages || []).some((choice) =>
      catalogPathMatches(choice.path, model.path)));
  }

  function loadedCatalogEntry(model: LoadedModel) {
    const exact = catalog.find((entry) => entry.id === model.id);
    if (exact && catalogEntryMatchesLoadedModel(exact, model)) return exact;
    return catalog.find((entry) => catalogEntryMatchesLoadedModel(entry, model));
  }

  function inferredLoadedModelName(model: LoadedModel, base?: CatalogEntry) {
    const variant = pathVariantLabel(model.path);
    const familyName = familyLabels[model.family] || base?.display_name || model.family;
    return variant && !familyName.toLowerCase().includes(variant.toLowerCase())
      ? `${familyName} ${variant}`
      : familyName;
  }

  function loadedModelName(model: LoadedModel) {
    return loadedCatalogEntry(model)?.display_name || inferredLoadedModelName(model);
  }

  function compareModelNames(left: string, right: string) {
    return left.localeCompare(right, 'en', { sensitivity: 'base', numeric: true });
  }

  function configuredCatalogEntries() {
    return loadedModels.map((model) => {
      const exact = loadedCatalogEntry(model);
      const familyMatch = catalog.find((entry) =>
        entry.family === model.family && entry.task === model.task);
      const base = exact || familyMatch;
      return {
        ...(base || {
          id: model.id,
          display_name: model.id,
          family: model.family,
          path: model.path,
          task: model.task,
          mode: model.mode
        }),
        id: model.id,
        display_name: exact ? exact.display_name : inferredLoadedModelName(model, base),
        display_name_en: exact ? exact.display_name_en : base?.display_name_en,
        family: model.family,
        path: model.path,
        task: model.task,
        mode: model.mode,
        install_packages: []
      } as CatalogEntry;
    });
  }

  function groupCatalog(entries: CatalogEntry[]): ModelGroup[] {
    return Array.from(
      entries.reduce((groups, entry) => {
        const existing = groups.get(entry.family) || [];
        existing.push(entry);
        groups.set(entry.family, existing);
        return groups;
      }, new Map<string, CatalogEntry[]>())
    ).map(([family, entries]) => ({
      family,
      entries: [...entries].sort((left, right) => compareModelNames(left.display_name, right.display_name)),
      label: entries.length > 1 ? (familyLabels[family] || entries[0].display_name) : entries[0].display_name
    })).sort((left, right) => compareModelNames(left.label, right.label));
  }

  let activeWorkflow: WorkflowId = 'tts';
  let workflowSelections: Partial<Record<WorkflowId, string>> = {};
  let modelWorkflowFilters: WorkflowId[] = workflowTabs.map((workflow) => workflow.id);
  let activeCatalog: CatalogEntry[] = catalog;
  let modelGroups: ModelGroup[] = groupCatalog(catalog);

  class StatusWarning extends Error {}

  $: activeCatalog = server && !server.ui_management ? configuredCatalogEntries() : catalog;
  $: modelGroups = groupCatalog(activeCatalog);
  $: selected = activeCatalog.find((entry) => entry.id === selectedId) || activeCatalog[0] || catalog[0];
  $: activeWorkflowSpec = workflowTabs.find((workflow) => workflow.id === activeWorkflow) || workflowTabs[0];
  $: workflowModels = activeCatalog.filter((entry) =>
    activeWorkflowSpec.tasks.some((task) => task === entry.task));
  $: filteredModelGroups = modelGroups.map((group) => ({
    ...group,
    entries: group.entries.filter((entry) => {
      const workflow = workflowTabs.find((candidate) => candidate.tasks.some((task) => task === entry.task));
      return Boolean(workflow && modelWorkflowFilters.includes(workflow.id));
    })
  })).filter((group) => group.entries.length > 0);
  $: isLoaded = loadedModels.some((model) => model.id === selectedId && model.loaded &&
    comparablePath(model.path) === comparablePath(modelPath));
  $: needsSource = ['asr', 'vc', 'svc', 's2s', 'sep', 'vad', 'diar', 'align', 'midi'].includes(selected?.task);
  $: acceptsSource = needsSource || selected?.task === 'gen';
  $: needsVoice = (['clon', 'vc', 'svc'].includes(selected?.task) && selected?.family !== 'rvc') ||
    (selected?.task === 's2s' && selected?.family === 'personaplex') ||
    (selected?.task === 'tts' && !['supertonic'].includes(selected?.family));
  $: usesVibeVoiceSpeakerFiles = selected?.family === 'vibevoice';
  $: isQwenBase = selected?.task === 'tts' && selected?.family === 'qwen3_tts' &&
    !selected?.id.includes('custom');
  $: allowsQuickStartVoice = ['tts', 'clon'].includes(selected?.task);
  $: referenceVoiceRequired = !(allowsQuickStartVoice && quickStartVoice) && (
    (['clon', 'vc', 'svc'].includes(selected?.task) && selected?.family !== 'rvc') || isQwenBase);
  $: lyricsRequired = requiresRequestOption(selected, 'lyrics');
  $: referenceTextRequired = requiresRequestOption(selected, 'reference_text') ||
    (Boolean(voiceFile) && isQwenBase);
  $: quickStartVoices = server && !server.ui_management
    ? configuredVoices
    : Object.entries(demoVoiceSources)
      .filter(([, source]) => bundledVoices.includes(source))
      .map(([voice]) => voice);
  $: showsText = ['tts', 'clon', 'gen', 's2s', 'align', 'vdes'].includes(selected?.task);
  $: supportsLiveAsr = selected?.task === 'asr' &&
    ['voxtral_realtime', 'nemotron_asr', 'higgs_audio_stt', 'sense_asr'].includes(selected?.family);
  $: modelInventoryLoading = server === null ||
    (Boolean(server.ui_management) && Object.keys(packageSizes).length === 0 && packageSizeState !== 'failed');
  $: selectableModelIds = new Set(activeCatalog.filter((entry) => {
    if (server && !server.ui_management) return true;
    if (loadedModels.some((model) => model.id === entry.id && model.loaded)) return true;
    if (entry.id === selectedId && installed === true) return true;
    const choices = entry.install_packages || [];
    if (!choices.length || choices.some((choice) => packageSizes[choice.id] === undefined)) return true;
    return choices.some((choice) => packageSizes[choice.id]?.installed);
  }).map((entry) => entry.id));

  function log(message: string) {
    const line = `${new Date().toLocaleTimeString()}  ${message}`;
    logs = [line, ...logs].slice(0, 200);
  }

  function formatBytes(bytes: number) {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / 1024 ** index).toFixed(index < 2 ? 0 : 1)} ${units[index]}`;
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

  function selectedPackageChoice(entry: CatalogEntry) {
    const choices = entry.install_packages || [];
    return choices.find((choice) => choice.id === selectedPackageIds[entry.id]) || choices[0];
  }

  function packageIsSelected(entry: CatalogEntry, choice: InstallPackageChoice) {
    return selectedPackageChoice(entry)?.id === choice.id;
  }

  function selectedModelPath(entry: CatalogEntry) {
    if (server && !server.ui_management) return entry.path;
    return resolveCatalogPath(selectedPackageChoice(entry)?.path || entry.path);
  }

  function comparablePath(path: string) {
    return path.replace(/\\/g, '/').replace(/\/$/, '').toLowerCase();
  }

  function packagePathMatches(choice: InstallPackageChoice, path: string) {
    return catalogPathMatches(choice.path, path);
  }

  function residentModel(entry: CatalogEntry, models = loadedModels) {
    return models.find((model) => model.id === entry.id && model.loaded);
  }

  function packageIsResident(entry: CatalogEntry, choice: InstallPackageChoice, models = loadedModels) {
    const resident = residentModel(entry, models);
    return Boolean(resident && packagePathMatches(choice, resident.path));
  }

  function packageIsAvailable(
    entry: CatalogEntry,
    choice: InstallPackageChoice,
    models = loadedModels,
    sizes = packageSizes
  ) {
    return packageIsResident(entry, choice, models) || sizes[choice.id]?.installed === true;
  }

  function studioPackageSlots(entry: CatalogEntry) {
    const choices = entry.install_packages || [];
    if (entry.family === 'ace_step') {
      return choices.map((choice) => ({ key: choice.id, label: choice.label, choice }));
    }
    const q8 = choices.find((choice) => choice.format === 'gguf' &&
      ['q8', 'q8_0'].includes(choice.precision));
    const fp16 = choices.find((choice) => choice.format === 'gguf' &&
      ['f16', 'fp16', 'bf16'].includes(choice.precision));
    if (!q8 && !fp16) {
      return choices.map((choice) => ({ key: choice.id, label: choice.label, choice }));
    }
    return [
      {
        key: 'q8',
        label: q8?.label || 'GGUF Q8',
        choice: q8
      },
      {
        key: 'fp16',
        label: fp16?.label || 'GGUF FP16',
        choice: fp16
      },
    ];
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

  function chunkSeed(base: number, index: number) {
    return (base + index) % 0x100000000;
  }

  function fileStem(name: string) {
    return name.replace(/\.[^.]+$/, '').trim().toLowerCase();
  }

  function chooseVoiceReference(file: File | null) {
    const changed = Boolean(file && (quickStartVoice || savedVoiceId ||
      (voiceFile && voiceFile.name !== file.name)));
    if (file) {
      quickStartVoice = '';
      savedVoiceId = '';
    }
    voiceFile = file;
    if (changed) {
      referenceTextFile = null;
      referenceText = '';
      if (referenceTextInput) referenceTextInput.value = '';
      status = 'Reference voice changed. Choose or enter its matching transcript.';
      warningStatus = status;
    }
  }

  function chooseVibeVoiceSpeaker(index: number, file: File | null) {
    vibeVoiceSpeakerFiles = vibeVoiceSpeakerFiles.map((current, currentIndex) =>
      currentIndex === index ? file : current);
  }

  function chooseQuickStartVoice(voice: string) {
    quickStartVoice = voice;
    if (!voice) return;
    savedVoiceId = '';
    voiceFile = null;
    voiceName = '';
    referenceTextFile = null;
    referenceText = '';
    if (voiceInput) voiceInput.value = '';
    if (referenceTextInput) referenceTextInput.value = '';
  }

  async function chooseReferenceText(file: File | null) {
    referenceTextFile = file;
    if (!file) return;
    try {
      const transcript = (await file.text()).replace(/^\uFEFF/, '').trim();
      if (!transcript) throw new Error('The selected reference text file is empty.');
      referenceText = transcript;
      const matchesVoice = !voiceFile || fileStem(file.name) === fileStem(voiceFile.name);
      status = matchesVoice
        ? `Loaded reference transcript from ${file.name}.`
        : `Loaded ${file.name}. Its name does not match ${voiceFile?.name}; verify that it is the correct transcript.`;
      warningStatus = matchesVoice ? '' : status;
    } catch (error) {
      referenceTextFile = null;
      referenceText = '';
      if (referenceTextInput) referenceTextInput.value = '';
      status = error instanceof Error ? error.message : String(error);
      warningStatus = '';
    }
  }

  function installPercent(job: ModelInstallJob) {
    if (job.state === 'complete' || job.state === 'cleaned') return 100;
    if (job.progress_percent >= 0) return Math.min(100, Math.max(0, job.progress_percent));
    return 0;
  }

  function installProgressLabel(job: ModelInstallJob) {
    const percent = installPercent(job);
    if (job.total_bytes > 0) {
      return `${percent}% · ${formatBytes(job.downloaded_bytes)} / ${formatBytes(job.total_bytes)}`;
    }
    if (job.downloaded_bytes > 0) return `${formatBytes(job.downloaded_bytes)} downloaded`;
    if (job.state === 'failed') return 'Download failed';
    if (job.state === 'cleaned') return 'Partial files cleaned';
    if (job.state === 'complete') return '100% · complete';
    return job.state === 'queued' ? '0% · queued' : 'Connecting and checking package files…';
  }

  function entryInstallJobs(entry: CatalogEntry, jobs: Record<string, ModelInstallJob>) {
    return (entry.install_packages || [])
      .map((choice) => jobs[choice.id])
      .filter((job): job is ModelInstallJob => job !== undefined);
  }

  function uniquePackagesForEntry(group: ModelGroup, entry: CatalogEntry) {
    const index = group.entries.findIndex((candidate) => candidate.id === entry.id);
    const previousIds = new Set(group.entries.slice(0, Math.max(0, index))
      .flatMap((candidate) => (candidate.install_packages || []).map((choice) => choice.id)));
    return (entry.install_packages || []).filter((choice) => !previousIds.has(choice.id));
  }

  function displayInstallJobForChoices(
    choices: InstallPackageChoice[],
    installState: Record<string, ModelInstallJob>
  ) {
    const jobs = choices.map((choice) => installState[choice.id])
      .filter((job): job is ModelInstallJob => job !== undefined);
    return jobs.find((job) => ['running', 'queued', 'cancelling'].includes(job.state)) ||
      [...jobs].sort((left, right) => right.finished_at_ms - left.finished_at_ms)[0];
  }

  function displayInstallJob(entry: CatalogEntry, installState: Record<string, ModelInstallJob>) {
    const jobs = entryInstallJobs(entry, installState);
    return jobs.find((job) => job.state === 'running' || job.state === 'queued' || job.state === 'cancelling') ||
      [...jobs].sort((left, right) => right.finished_at_ms - left.finished_at_ms)[0];
  }

  function entryInstallBusy(entry: CatalogEntry, installState: Record<string, ModelInstallJob>) {
    return entryInstallJobs(entry, installState).some((job) =>
      job.state === 'running' || job.state === 'queued' || job.state === 'cancelling');
  }

  function groupInstallBusy(group: ModelGroup, installState: Record<string, ModelInstallJob>) {
    return group.entries.some((entry) => entryInstallBusy(entry, installState));
  }

  function supportsMaxTokens(entry: CatalogEntry) {
    if (entry.request_options !== undefined) return entry.request_options.includes('max_tokens');
    return ['tts', 'clon', 'gen', 's2s', 'vdes'].includes(entry.task);
  }

  function supportsRequestOption(entry: CatalogEntry, option: string) {
    // Specs that publish request metadata are authoritative. Older specs
    // without that metadata keep the legacy UI behavior until migrated.
    return entry.request_options === undefined || entry.request_options.includes(option);
  }

  function requiresRequestOption(entry: CatalogEntry, option: string) {
    return entry.required_request_options?.includes(option) === true;
  }

  function packageVersionLabel(size: ModelPackageSize | undefined, translate = tr) {
    if (!size?.installed) return '';
    if (size.version_state === 'up_to_date') return translate('models.upToDate');
    if (size.version_state === 'update_available') return translate('models.updateAvailable');
    return translate('models.versionUnknown');
  }

  function entrySelectable(entry: CatalogEntry) {
    return selectableModelIds.has(entry.id);
  }

  function workflowForTask(task: string | undefined): WorkflowId {
    return (workflowTabs.find((workflow) => workflow.tasks.some((candidate) => candidate === task))?.id || 'tts') as WorkflowId;
  }

  function installButtonLabel(
    choice: InstallPackageChoice,
    job: ModelInstallJob | undefined,
    translate = tr
  ) {
    if (job?.state === 'running') return `${choice.label}…`;
    if (job?.state === 'queued') return `${choice.label} ${translate('models.queued')}`;
    if (job?.state === 'cancelling') return `${choice.label} ${translate('models.stopping')}`;
    return choice.label;
  }

  function packageSizeLabel(
    size: ModelPackageSize | undefined,
    sizeState: 'idle' | 'running' | 'complete' | 'failed',
    selected: boolean,
    translate = tr
  ) {
    const bytes = size?.size_bytes !== null && size?.size_bytes !== undefined
      ? formatBytes(size.size_bytes)
      : '';
    if (size?.installed) {
      const version = packageVersionLabel(size, translate);
      return `${selected ? translate('models.selected') : translate('models.downloaded')}${version ? ` · ${version}` : ''}${bytes ? ` · ${bytes}` : ''}`;
    }
    if (bytes) return bytes;
    if (size?.state === 'pending') return translate('models.checkingSize');
    if (size?.state === 'gated') return translate('models.hfAccess');
    if (size?.state === 'error' || size?.state === 'unknown') return translate('models.sizeUnavailable');
    return sizeState === 'running' ? translate('models.checkingSize') : '';
  }

  function localizedStatus(value: string, translate = tr) {
    return value === 'Ready' ? translate('status.ready') : value;
  }

  async function refreshPackageSizes() {
    if (!server?.ui_management || packageSizeRefreshInFlight) return;
    packageSizeRefreshInFlight = true;
    try {
      const response = await modelPackageSizes();
      packageSizeState = response.state;
      if (response.data.length) {
        packageSizes = Object.fromEntries(response.data.map((size) => [size.id, size]));
        if (response.state === 'complete') reconcileSelectedPackageChoices(packageSizes);
      }
      const inventoryPending = response.state === 'idle' || response.state === 'running' || response.data.length === 0;
      if (inventoryPending && packageSizePoll === null) {
        packageSizePoll = window.setInterval(refreshPackageSizes, 1000);
      } else if (!inventoryPending && packageSizePoll !== null) {
        window.clearInterval(packageSizePoll);
        packageSizePoll = null;
      }
    } catch (error) {
      packageSizeState = 'failed';
      log(`Package sizes unavailable: ${error instanceof Error ? error.message : error}`);
    } finally {
      packageSizeRefreshInFlight = false;
    }
  }

  function openModelsPage() {
    if (!server?.ui_management) {
      tab = 'studio';
      status = 'Model management is disabled for this configured server.';
      warningStatus = status;
      errorStatus = '';
      return;
    }
    tab = 'models';
    if (packageSizeState === 'idle') packageSizeState = 'running';
    refreshPackageSizes();
  }

  function rememberPackageChoice(entry: CatalogEntry, choice: InstallPackageChoice) {
    selectedPackageIds = { ...selectedPackageIds, [entry.id]: choice.id };
    localStorage.setItem('audiocpp.ui.packageIds', JSON.stringify(selectedPackageIds));
    if (entry.id === selectedId) modelPath = resolveCatalogPath(choice.path);
  }

  function reconcileSelectedPackageChoices(sizes = packageSizes) {
    const nextIds = { ...selectedPackageIds };
    let changed = false;

    for (const entry of catalog) {
      const choices = entry.install_packages || [];
      if (!choices.length) continue;
      const current = choices.find((choice) => choice.id === nextIds[entry.id]);
      const currentJob = current ? installJobs[current.id] : undefined;
      const currentBusy = currentJob && ['queued', 'running', 'cancelling'].includes(currentJob.state);
      if (currentBusy || (current && sizes[current.id]?.installed)) continue;

      const replacement = choices.find((choice) => sizes[choice.id]?.installed);
      if (replacement) {
        if (nextIds[entry.id] !== replacement.id) {
          nextIds[entry.id] = replacement.id;
          changed = true;
        }
      } else if (nextIds[entry.id] !== undefined) {
        delete nextIds[entry.id];
        changed = true;
      }
    }

    if (!changed) return false;
    selectedPackageIds = nextIds;
    localStorage.setItem('audiocpp.ui.packageIds', JSON.stringify(selectedPackageIds));
    if (selectedId) modelPath = selectedModelPath(selected);
    return true;
  }

  function reconcileResidentPackageChoices(models = loadedModels) {
    const nextIds = { ...selectedPackageIds };
    let changed = false;

    for (const entry of catalog) {
      const resident = residentModel(entry, models);
      if (!resident) continue;
      const choice = (entry.install_packages || []).find((candidate) =>
        packagePathMatches(candidate, resident.path));
      if (choice && nextIds[entry.id] !== choice.id) {
        nextIds[entry.id] = choice.id;
        changed = true;
      }
    }

    if (!changed) return false;
    selectedPackageIds = nextIds;
    localStorage.setItem('audiocpp.ui.packageIds', JSON.stringify(selectedPackageIds));
    if (selectedId) modelPath = selectedModelPath(selected);
    return true;
  }

  async function unloadRemovedResidentPackages(sizes = packageSizes) {
    const staleEntries = catalog.filter((entry) => {
      const resident = residentModel(entry);
      if (!resident) return false;
      const residentPath = comparablePath(resident.path);
      const residentChoice = (entry.install_packages || []).find((choice) =>
        comparablePath(resolveCatalogPath(choice.path)) === residentPath);
      return Boolean(residentChoice && sizes[residentChoice.id]?.installed === false);
    });
    if (!staleEntries.length) return false;

    for (const entry of staleEntries) await unloadModel(entry.id);
    await refresh();
    return true;
  }

  async function openStudioPage() {
    tab = 'studio';
    await refresh();
    if (server?.ui_management) {
      await refreshPackageSizes();
      await unloadRemovedResidentPackages();
      const selectionChanged = reconcileSelectedPackageChoices();
      if (selectionChanged && selectedId) await inspectPath();
    }
  }

  function acceptModelsRoot(root: Awaited<ReturnType<typeof getModelsRoot>>) {
    modelsFolder = root.models_root;
    modelsFolderInput = root.models_root;
    defaultModelsFolder = root.default_models_root;
    modelsFolderIsDefault = root.is_default;
    modelPath = selectedModelPath(selected);
  }

  function clearModelSelection() {
    selectedId = '';
    quickStartVoice = '';
    configuredVoices = [];
    modelPath = '';
    installed = null;
    paramSpecs = [];
    advancedValues = {};
    advancedJson = '{}';
    localStorage.removeItem('audiocpp.ui.model');
    clearOutput();
  }

  function clearUnavailableModelSelection(ignoreResident = false) {
    if (server && !server.ui_management) return false;
    if (!selectedId || (!ignoreResident && isLoaded) || installed !== false) return false;
    clearModelSelection();
    return true;
  }

  async function applyModelsFolder(useDefault = false) {
    if (!server?.ui_management || applyingModelsFolder) return;
    applyingModelsFolder = true;
    warningStatus = '';
    errorStatus = '';
    status = useDefault ? 'Restoring the default models folderâ€¦' : 'Changing models folderâ€¦';
    try {
      const root = await setModelsRoot(useDefault ? '' : modelsFolderInput.trim());
      acceptModelsRoot(root);
      if (root.is_default) localStorage.removeItem('audiocpp.ui.modelsFolder');
      else localStorage.setItem('audiocpp.ui.modelsFolder', root.models_root);
      installJobs = {};
      packageSizes = {};
      packageSizeState = 'idle';
      refreshedInstallFinishes = {};
      if (installPoll !== null) {
        window.clearInterval(installPoll);
        installPoll = null;
      }
      if (packageSizePoll !== null) {
        window.clearInterval(packageSizePoll);
        packageSizePoll = null;
      }
      await refreshPackageSizes();
      await inspectPath();
      const selectionCleared = clearUnavailableModelSelection();
      status = selectionCleared
        ? `Models folder: ${root.models_root}. No installed model is selected.`
        : `Models folder: ${root.models_root}`;
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      errorStatus = status;
      log(`Models folder change failed: ${status}`);
    } finally {
      applyingModelsFolder = false;
    }
  }

  async function openFolderBrowser(path = '') {
    folderBrowserOpen = true;
    folderBrowserLoading = true;
    folderBrowserError = '';
    try {
      folderBrowser = await browseDirectories(path || modelsFolderInput.trim() || modelsFolder);
    } catch (error) {
      folderBrowserError = error instanceof Error ? error.message : String(error);
    } finally {
      folderBrowserLoading = false;
    }
  }

  function selectBrowsedFolder() {
    if (!folderBrowser) return;
    modelsFolderInput = folderBrowser.current;
    folderBrowserOpen = false;
  }

  function resetParams() {
    const byId = parameterCatalog[selected?.id] || parameterCatalog[selected?.family] || [];
    paramSpecs = selected?.family === 'vibevoice'
      ? byId.filter((spec) => spec.name !== 'voice_samples')
      : byId;
    advancedValues = Object.fromEntries(byId.map((spec) => [spec.name, spec.default ?? '']));
    if (selected?.family === 'minimax_h3') {
      duration = 15;
      advancedValues = { ...advancedValues, num_frames: miniMaxFramesForDuration(duration), dit_acceleration: 'none' };
    } else if (selected?.task === 'gen') {
      duration = 30;
    }
    advancedJson = '{}';
  }

  function syncConfiguredSelection() {
    if (!server || server.ui_management) return;
    const entries = configuredCatalogEntries();
    if (!entries.length) {
      clearModelSelection();
      return;
    }
    const current = selectedId ? entries.find((entry) => entry.id === selectedId) : undefined;
    const next = current || entries[0];
    if (selectedId !== next.id) {
      quickStartVoice = '';
      configuredVoices = [];
    }
    selectedId = next.id;
    selected = next;
    activeWorkflow = workflowForTask(next.task);
    workflowSelections = { ...workflowSelections, [activeWorkflow]: next.id };
    modelPath = next.path;
    installed = true;
    chunkBudget = defaultChunkBudget(next.family);
    localStorage.setItem('audiocpp.ui.model', next.id);
    resetParams();
  }

  async function refresh() {
    try {
      [server, loadedModels] = await Promise.all([health(), models()]);
      if (server.ui_management) reconcileResidentPackageChoices();
      else syncConfiguredSelection();
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    }
  }

  async function inspectPath() {
    if (!selectedId || !modelPath.trim()) {
      installed = null;
      return;
    }
    if (server && !server.ui_management) {
      installed = loadedModels.some((model) => model.id === selectedId);
      return;
    }
    installed = null;
    try {
      installed = (await pathStatus(modelPath)).exists;
    } catch {
      installed = null;
    }
  }

  function chooseModel(id: string) {
    if (!id) {
      clearModelSelection();
      status = 'No model selected. Choose an installed model or download one from the Models tab.';
      return;
    }
    const next = activeCatalog.find((entry) => entry.id === id);
    if (!next || !entrySelectable(next)) return;
    selectedId = id;
    selected = next;
    quickStartVoice = '';
    configuredVoices = [];
    activeWorkflow = workflowForTask(next.task);
    workflowSelections = { ...workflowSelections, [activeWorkflow]: id };
    modelPath = selectedModelPath(next);
    chunkBudget = defaultChunkBudget(next.family);
    localStorage.setItem('audiocpp.ui.model', id);
    resetParams();
    inspectPath();
    refreshConfiguredVoices();
  }

  function chooseWorkflow(id: WorkflowId) {
    const workflow = workflowTabs.find((entry) => entry.id === id);
    if (!workflow) return;
    activeWorkflow = id;
    if (selectedId && workflow.tasks.some((task) => task === selected?.task)) return;

    const rememberedId = workflowSelections[id];
    const remembered = rememberedId
      ? activeCatalog.find((entry) => entry.id === rememberedId &&
          workflow.tasks.some((task) => task === entry.task) && entrySelectable(entry))
      : undefined;
    const next = remembered || activeCatalog.find((entry) =>
      workflow.tasks.some((task) => task === entry.task) && entrySelectable(entry));
    if (next) {
      chooseModel(next.id);
      return;
    }

    clearModelSelection();
    status = `No installed models are available for ${workflow.label}. Install one from the Models tab.`;
  }

  function toggleModelWorkflowFilter(id: WorkflowId, enabled: boolean) {
    modelWorkflowFilters = enabled
      ? [...modelWorkflowFilters, id].filter((value, index, all) => all.indexOf(value) === index)
      : modelWorkflowFilters.filter((value) => value !== id);
  }

  async function doLoad(modeOverride?: string) {
    if (!selectedId) {
      status = 'Choose an installed model before loading.';
      warningStatus = status;
      errorStatus = '';
      return;
    }
    if (!server?.ui_management) {
      status = 'This server was not started with UI management enabled.';
      warningStatus = status;
      errorStatus = '';
      return;
    }
    loadingModel = true;
    warningStatus = '';
    errorStatus = '';
    status = `Loading ${selected.display_name}…`;
    log(status);
    try {
      const targetPath = comparablePath(modelPath);
      const replaced = loadedModels.filter((model) => model.loaded &&
        (model.id !== selected.id || comparablePath(model.path) !== targetPath));
      for (const model of replaced) {
        log(`Unloading ${loadedModelName(model)} before loading ${selected.display_name}.`);
        await unloadModel(model.id);
      }
      if (replaced.length) await refresh();
      await loadModel({
        id: selected.id,
        path: modelPath,
        family: selected.family,
        task: selected.task,
        mode: modeOverride || selected.mode || 'offline',
        load_options: selected.load_options || {},
        session_options: selected.session_options || {}
      });
      await refresh();
      status = tr('status.modelReady', { model: selected.display_name });
      errorStatus = '';
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      errorStatus = status;
      log(`Load failed: ${status}`);
      throw error;
    } finally {
      loadingModel = false;
    }
  }

  async function doUnload() {
    if (!server?.ui_management) {
      status = 'Model unload is disabled for this configured server.';
      warningStatus = status;
      errorStatus = '';
      return;
    }
    if (!selectedId) return;
    const modelName = selected.display_name;
    loadingModel = true;
    try {
      await unloadModel(selected.id);
      await refresh();
      const selectionCleared = clearUnavailableModelSelection(true);
      status = selectionCleared
        ? `${modelName} unloaded. No installed model is selected.`
        : `${modelName} unloaded.`;
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    } finally {
      loadingModel = false;
    }
  }

  async function toggleStudioPackage(choice: InstallPackageChoice) {
    if (loadingModel || !packageIsAvailable(selected, choice)) return;
    if (packageIsResident(selected, choice)) {
      await doUnload();
      return;
    }

    rememberPackageChoice(selected, choice);
    await inspectPath();
    if (installed !== true) {
      status = `${selected.display_name} ${choice.label} is not available at the expected path.`;
      errorStatus = status;
      return;
    }
    await doLoad();
  }

  async function toggleSingleModel() {
    if (loadingModel || !selectedId || installed === false) return;
    if (isLoaded) await doUnload();
    else await doLoad();
  }

  async function stagedPath(file: File | null): Promise<string | undefined> {
    if (!file) return undefined;
    const targetSampleRate = selected.task === 'sep'
      ? 44100
      : ['asr', 'vad', 'diar', 'align', 'midi'].includes(selected.task) ? 16000 : undefined;
    const wav = await browserDecodeToWav(file, targetSampleRate);
    return uploadWav(wav, aborter?.signal);
  }

  async function vibeVoiceSamplePaths(): Promise<string | undefined> {
    const firstEmpty = vibeVoiceSpeakerFiles.findIndex((file) => !file);
    const hasLaterFile = firstEmpty >= 0 && vibeVoiceSpeakerFiles.slice(firstEmpty + 1).some(Boolean);
    if (hasLaterFile) {
      throw new StatusWarning('VibeVoice speaker references must be filled from Speaker 1 without gaps.');
    }
    const files = vibeVoiceSpeakerFiles.filter((file): file is File => Boolean(file));
    if (!files.length) return undefined;
    const paths = await Promise.all(files.map((file) => stagedPath(file)));
    return paths.filter((path): path is string => Boolean(path)).join(',');
  }

  function requestOptions() {
    let raw: Record<string, unknown> = {};
    try {
      raw = JSON.parse(advancedJson || '{}');
      if (Array.isArray(raw) || raw === null) throw new Error('must be an object');
    } catch (error) {
      throw new Error(`Advanced JSON is invalid: ${error instanceof Error ? error.message : error}`);
    }
    const defaults = selected.default_options || {};
    return { ...defaults, ...advancedValues, ...raw };
  }

  function clearOutput() {
    for (const output of outputAudio) URL.revokeObjectURL(output.url);
    outputAudio = [];
    outputArtifacts = [];
    outputText = '';
    outputJson = '';
  }

  async function ensureLoaded() {
    if (!server?.ui_management) {
      if (!loadedModels.some((model) => model.id === selectedId)) {
        throw new Error('Configured model is not registered by this server.');
      }
      return;
    }
    if (!isLoaded) {
      await doLoad();
      await refresh();
      if (!loadedModels.some((model) => model.id === selectedId && model.loaded)) {
        throw new Error('Model did not load.');
      }
    }
  }

  async function ensureLoadedMode(mode: string) {
    if (!server?.ui_management) {
      if (!loadedModels.some((model) => model.id === selectedId && model.mode === mode)) {
        throw new Error(`Configured model is not registered in ${mode} mode.`);
      }
      return;
    }
    const loaded = loadedModels.find((model) => model.id === selectedId && model.loaded);
    if (!loaded || loaded.mode !== mode) {
      await doLoad(mode);
      await refresh();
    }
    if (!loadedModels.some((model) =>
      model.id === selectedId && model.loaded && model.mode === mode)) {
      throw new Error(`Model did not load in ${mode} mode.`);
    }
  }

  function recordingMimeType(): string | undefined {
    for (const type of ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']) {
      if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return undefined;
  }

  async function startRecording(target: 'source' | 'voice') {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      status = 'Microphone recording is not supported by this browser.';
      return;
    }
    if (recorder || liveRecording) return;
    try {
      recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks: Blob[] = [];
      const mimeType = recordingMimeType();
      recorder = new MediaRecorder(recordingStream, mimeType ? { mimeType } : undefined);
      recordingTarget = target;
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: recorder?.mimeType || mimeType || 'audio/webm' });
        const file = new File([blob], `recording-${Date.now()}.webm`, { type: blob.type });
        if (target === 'source') sourceFile = file;
        else {
          quickStartVoice = '';
          savedVoiceId = '';
          voiceFile = file;
          if (voiceInput) voiceInput.value = '';
        }
        recordingStream?.getTracks().forEach((track) => track.stop());
        recordingStream = null;
        recorder = null;
        recordingTarget = null;
        status = `${target === 'voice' ? 'Voice reference' : 'Source audio'} recording captured.`;
      };
      recorder.start();
      status = `Recording ${target === 'voice' ? 'voice reference' : 'source audio'}…`;
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      recordingStream?.getTracks().forEach((track) => track.stop());
      recordingStream = null;
      recorder = null;
      recordingTarget = null;
    }
  }

  function stopRecording() {
    if (recorder?.state === 'recording') recorder.stop();
  }

  async function refreshVoices() {
    try {
      savedVoices = await listVoices();
    } catch (error) {
      log(`Voice library unavailable: ${error instanceof Error ? error.message : error}`);
    }
  }

  async function refreshBundledVoices() {
    try {
      bundledVoices = await availableVoices();
    } catch (error) {
      log(`Quick-start voices unavailable: ${error instanceof Error ? error.message : error}`);
    }
  }

  async function refreshConfiguredVoices() {
    if (!selectedId || server?.ui_management !== false) {
      configuredVoices = [];
      return;
    }
    try {
      configuredVoices = await availableVoices(selectedId);
      if (quickStartVoice && !configuredVoices.includes(quickStartVoice)) quickStartVoice = '';
    } catch (error) {
      configuredVoices = [];
      log(`Configured voices unavailable: ${error instanceof Error ? error.message : error}`);
    }
  }

  async function storeCurrentVoice() {
    if (!voiceFile) {
      status = 'Choose or record a voice reference first.';
      return;
    }
    const name = voiceName.trim() || voiceFile.name.replace(/\.[^.]+$/, '');
    const wav = await browserDecodeToWav(voiceFile);
    const id = crypto.randomUUID();
    await saveVoice({
      id,
      name,
      transcript: referenceText,
      audio: wav,
      createdAt: Date.now()
    });
    await refreshVoices();
    savedVoiceId = id;
    voiceName = name;
    status = `Saved voice “${name}” in this browser.`;
  }

  function chooseSavedVoice(id: string) {
    savedVoiceId = id;
    const voice = savedVoices.find((entry) => entry.id === id);
    if (!voice) return;
    quickStartVoice = '';
    voiceFile = new File([voice.audio], `${voice.name}.wav`, { type: 'audio/wav' });
    referenceTextFile = null;
    if (voiceInput) voiceInput.value = '';
    if (referenceTextInput) referenceTextInput.value = '';
    referenceText = voice.transcript;
    voiceName = voice.name;
    status = `Selected saved voice “${voice.name}”.`;
  }

  async function removeCurrentVoice() {
    if (!savedVoiceId) return;
    const voice = savedVoices.find((entry) => entry.id === savedVoiceId);
    await deleteSavedVoice(savedVoiceId);
    savedVoiceId = '';
    await refreshVoices();
    status = `Deleted saved voice “${voice?.name || ''}”.`;
  }

  async function transcribeLiveSlice(blob: Blob) {
    if (!blob.size) return;
    const file = new File([blob], `live-${liveChunkNumber}.webm`, { type: blob.type });
    const wav = await browserDecodeToWav(file, 16000);
    const audio = await uploadWav(wav);
    const result = await transcription({
      model: selected.id,
      audio,
      language,
      text: context,
      options: requestOptions()
    });
    const chunkText = String(result.text || '').trim();
    if (chunkText) outputText = [outputText, chunkText].filter(Boolean).join(' ');
    liveChunkNumber += 1;
    status = `Listening… ${liveChunkNumber} chunk${liveChunkNumber === 1 ? '' : 's'} transcribed.`;
  }

  function captureLiveSlice() {
    if (!liveStream || liveStopRequested) return;
    const chunks: Blob[] = [];
    const mimeType = recordingMimeType();
    liveRecorder = new MediaRecorder(liveStream, mimeType ? { mimeType } : undefined);
    liveRecorder.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    liveRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: liveRecorder?.mimeType || mimeType || 'audio/webm' });
      liveQueue = liveQueue
        .then(() => transcribeLiveSlice(blob))
        .catch((error) => {
          status = error instanceof Error ? error.message : String(error);
          log(`Live transcription failed: ${status}`);
        });
      liveRecorder = null;
      if (!liveStopRequested) captureLiveSlice();
    };
    liveRecorder.start();
    window.setTimeout(() => {
      if (liveRecorder?.state === 'recording') liveRecorder.stop();
    }, 4000);
  }

  async function startLiveTranscription() {
    if (!supportsLiveAsr) return;
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      status = 'Live microphone transcription is not supported by this browser.';
      return;
    }
    clearOutput();
    liveRecording = true;
    liveStopRequested = false;
    liveChunkNumber = 0;
    try {
      await ensureLoadedMode('streaming');
      liveStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      status = 'Listening… speech is transcribed in four-second native streaming requests.';
      captureLiveSlice();
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      stopLiveTranscription();
    }
  }

  function stopLiveTranscription() {
    liveStopRequested = true;
    if (liveRecorder?.state === 'recording') liveRecorder.stop();
    liveStream?.getTracks().forEach((track) => track.stop());
    liveStream = null;
    liveRecording = false;
    status = liveChunkNumber ? `Live transcription stopped after ${liveChunkNumber} chunks.` : 'Live transcription stopped.';
  }

  async function run() {
    if (running) return;
    if (!selectedId) {
      status = 'Choose an installed model before running a request.';
      warningStatus = status;
      errorStatus = '';
      return;
    }
    if (!isLoaded && installed === false) {
      status = `${selected.display_name} is not downloaded. Install a model package from the Models tab first.`;
      warningStatus = status;
      errorStatus = '';
      return;
    }
    clearOutput();
    running = true;
    aborter = new AbortController();
    const started = performance.now();
    warningStatus = '';
    errorStatus = '';
    status = tr('status.runningTask', { task: localizedTaskLabel(selected.task) });
    log(status);
    try {
      const resolvedSeed = resolveRequestSeed(seed);
      if (referenceVoiceRequired && !voiceFile) {
        throw new StatusWarning(`${selected.display_name_en || selected.display_name} requires a reference voice.`);
      }
      if (referenceTextRequired && !referenceText.trim()) {
        const prefix = isQwenBase ? 'Qwen3-TTS Base voice cloning' : (selected.display_name_en || selected.display_name);
        throw new StatusWarning(`${prefix} requires a reference transcript. Choose a matching .txt file or enter the transcript.`);
      }
      if (lyricsRequired && !lyrics.trim()) {
        throw new StatusWarning(`${selected.display_name_en || selected.display_name} requires lyrics.`);
      }
      await ensureLoaded();
      const options = requestOptions();
      if (usesVibeVoiceSpeakerFiles) {
        const samples = await vibeVoiceSamplePaths();
        if (samples) options.voice_samples = samples;
      }
      const audio = acceptsSource ? await stagedPath(sourceFile) : undefined;
      const voiceRef = needsVoice && !usesVibeVoiceSpeakerFiles ? await stagedPath(voiceFile) : undefined;

      if (['tts', 'clon', 'vdes'].includes(selected.task)) {
        if (!text.trim()) throw new StatusWarning('Enter text to generate.');
        const chunks = longText && selected.task !== 'vdes'
          ? splitTtsChunks(text, Math.max(40, chunkBudget))
          : [text];
        const audioChunks: Blob[] = [];
        const timings: Array<Record<string, unknown>> = [];
        for (let index = 0; index < chunks.length; index += 1) {
          status = chunks.length > 1
            ? `Synthesizing chunk ${index + 1} of ${chunks.length}…`
            : tr('status.runningTask', { task: localizedTaskLabel(selected.task) });
          const body: Record<string, unknown> = {
            model: selected.id,
            input: chunks[index],
            language,
            seed: chunkSeed(resolvedSeed, index),
            options
          };
          if (supportsMaxTokens(selected)) body.max_tokens = maxTokens;
          if (voiceRef) body.voice_ref = voiceRef;
          else if (quickStartVoice) body.voice = demoVoiceSources[quickStartVoice] || quickStartVoice;
          else if (selected.default_voice) body.voice = selected.default_voice;
          if (referenceText.trim() && supportsRequestOption(selected, 'reference_text')) {
            body.reference_text = referenceText;
          }
          if (selected.task === 'vdes' && instructions.trim()) body.instructions = instructions;
          const result = await speech(body, aborter.signal);
          audioChunks.push(result.blob);
          timings.push({
            chunk: index + 1,
            characters: chunks[index].length,
            wall_ms: result.wallMs,
            rtf: result.rtf
          });
        }
        const merged = await concatenateAudioBlobs(audioChunks);
        outputAudio = [{ id: chunks.length > 1 ? 'merged' : 'output', url: URL.createObjectURL(merged) }];
        outputJson = JSON.stringify({
          seed: resolvedSeed,
          chunks: chunks.length,
          characters: text.length,
          chunk_budget: chunkBudget,
          timings
        }, null, 2);
      } else if (selected.task === 'asr') {
        if (!audio) throw new StatusWarning('Choose an audio file.');
        const result = await transcription({
          model: selected.id,
          audio,
          language,
          text: context,
          options
        }, aborter.signal);
        outputText = String(result.text || '');
        outputJson = JSON.stringify(result, null, 2);
      } else {
        if (needsSource && !audio) throw new StatusWarning('Choose a source audio file.');
        const request: Record<string, unknown> = { options };
        if (['gen', 's2s', 'align'].includes(selected.task) && text.trim()) request.text = text;
        if (['gen', 's2s', 'align'].includes(selected.task) && language.trim()) request.language = language;
        if (selected.task === 'gen') {
          if (lyrics.trim()) request.lyrics = lyrics;
          request.duration_seconds = duration;
          request.seed = resolvedSeed;
          request.max_tokens = maxTokens;
        } else if (selected.task === 's2s') {
          request.seed = resolvedSeed;
          if (supportsMaxTokens(selected)) request.max_tokens = maxTokens;
        }
        if (audio) request.audio = audio;
        if (voiceRef) request.voice_ref = voiceRef;
        if (referenceText.trim() && supportsRequestOption(selected, 'reference_text')) {
          request.reference_text = referenceText;
        }
        const result = await runTask({ model: selected.id, request }, aborter.signal);
        if (typeof result.audio === 'string') {
          outputAudio = [{ id: 'output', url: base64AudioUrl(result.audio) }];
        }
        if (Array.isArray(result.named_audio_outputs)) {
          outputAudio = result.named_audio_outputs
            .filter((entry): entry is { id: string; audio: string } =>
              typeof entry?.id === 'string' && typeof entry?.audio === 'string')
            .map((entry) => ({ id: entry.id, url: base64AudioUrl(entry.audio) }));
        }
        if (Array.isArray(result.artifacts)) {
          outputArtifacts = result.artifacts
            .filter((entry): entry is { id: string; payload: string; meta?: Record<string, string> } =>
              typeof entry?.id === 'string' && typeof entry?.payload === 'string')
            .map((entry) => ({
              id: entry.id,
              extension: entry.meta?.extension || (entry.meta?.format === 'midi' ? 'mid' : 'bin'),
              url: `data:${entry.meta?.mime || 'application/octet-stream'};base64,${entry.payload}`
            }));
        }
        outputText = typeof result.text === 'string' ? result.text : '';
        outputJson = JSON.stringify(result, (key, value) =>
          (key === 'audio' || key === 'payload') && typeof value === 'string'
            ? `<base64 data: ${value.length} chars>` : value, 2);
      }
      const elapsed = ((performance.now() - started) / 1000).toFixed(2);
      warningStatus = '';
      errorStatus = '';
      status = tr('status.completeIn', { seconds: elapsed });
      log(status);
    } catch (error) {
      if ((error as Error)?.name === 'AbortError') {
        status = 'Cancelled.';
        warningStatus = '';
        errorStatus = '';
      } else {
        status = error instanceof Error ? error.message : String(error);
        warningStatus = error instanceof StatusWarning ? status : '';
        errorStatus = error instanceof StatusWarning ? '' : status;
        log(`Request failed: ${status}`);
      }
    } finally {
      running = false;
      aborter = null;
    }
  }

  function cancel() {
    aborter?.abort();
  }

  function handleShortcut(event: KeyboardEvent) {
    if (event.key === 'Escape' && folderBrowserOpen) {
      folderBrowserOpen = false;
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !running) {
      event.preventDefault();
      run();
    }
  }

  async function refreshInstallJobs() {
    if (!server?.ui_management) return;
    try {
      const jobs = await modelInstallJobs();
      const incoming = Object.fromEntries(jobs.map((job) => {
        const local = installJobs[job.id];
        return [job.id, {
          ...job,
          total_bytes: job.total_bytes || local?.total_bytes || 0
        }];
      }));
      // Preserve a just-created local job if a status request races the server's
      // worker registration. This keeps the progress row visible from the click
      // until the authoritative job appears in a later poll.
      installJobs = { ...installJobs, ...incoming };
      let refreshInventory = false;
      for (const entry of catalog) {
        const completedJobs = entryInstallJobs(entry, installJobs).filter((job) => job.state === 'complete');
        for (const job of completedJobs) {
          const known = packageSizes[job.id];
          if (known && !known.installed) {
            packageSizes = { ...packageSizes, [job.id]: { ...known, installed: true } };
          }
          if (job.finished_at_ms > (refreshedInstallFinishes[job.id] || 0)) {
            refreshedInstallFinishes = {
              ...refreshedInstallFinishes,
              [job.id]: job.finished_at_ms
            };
            refreshInventory = true;
          }
        }
        const complete = completedJobs.length > 0;
        if (complete && entry.id === selectedId) await inspectPath();
      }
      if (refreshInventory) {
        packageSizeState = 'idle';
        await refreshPackageSizes();
      }
      const active = Object.values(installJobs).some((job) =>
        job.state === 'queued' || job.state === 'running' || job.state === 'cancelling');
      if (active && installPoll === null) {
        installPoll = window.setInterval(refreshInstallJobs, 1500);
      } else if (!active && installPoll !== null) {
        window.clearInterval(installPoll);
        installPoll = null;
      }
    } catch (error) {
      log(`Installer status unavailable: ${error instanceof Error ? error.message : error}`);
    }
  }

  async function installPackage(entry: CatalogEntry, choice: InstallPackageChoice, overwrite = false) {
    if (packageSizes[choice.id]?.installed && !overwrite) return;
    const existing = installJobs[choice.id];
    if (existing && ['queued', 'running', 'cancelling'].includes(existing.state)) return;
    const expectedSize = packageSizes[choice.id]?.size_bytes;
    const verb = overwrite ? 'Update' : 'Download';
    const confirmed = window.confirm(
      `${verb} ${entry.display_name} ${choice.label}` +
      `${expectedSize ? ` (${formatBytes(expectedSize)})` : ''}?`
    );
    if (!confirmed) return;
    rememberPackageChoice(entry, choice);
    status = `Starting ${choice.label} installation for ${entry.display_name}...`;
    const expectedBytes = packageSizes[choice.id]?.size_bytes || 0;
    installJobs = {
      ...installJobs,
      [choice.id]: {
        id: choice.id,
        state: 'queued',
        message: 'Sending installation request…',
        exit_code: -1,
        downloaded_bytes: 0,
        total_bytes: expectedBytes,
        progress_percent: 0,
        started_at_ms: 0,
        finished_at_ms: 0
      }
    };
    try {
      if (installPoll === null) {
        installPoll = window.setInterval(refreshInstallJobs, 1000);
      }
      const job = await installModelPackage({ id: choice.id, overwrite });
      installJobs = {
        ...installJobs,
        [job.id]: { ...job, total_bytes: job.total_bytes || expectedBytes }
      };
      await refreshInstallJobs();
      status = `${entry.display_name} ${choice.label} installation is running in the background.`;
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      installJobs = {
        ...installJobs,
        [choice.id]: {
          id: choice.id,
          state: 'failed',
          message: status,
          exit_code: -1,
          downloaded_bytes: 0,
          total_bytes: 0,
          progress_percent: -1,
          started_at_ms: 0,
          finished_at_ms: Date.now()
        }
      };
      log(`Installer failed to start: ${status}`);
    }
  }

  function useOrInstallPackage(entry: CatalogEntry, choice: InstallPackageChoice) {
    if (packageSizes[choice.id]?.installed) {
      rememberPackageChoice(entry, choice);
      status = tr('status.packageAvailable', { model: entry.display_name, format: choice.label });
      log(status);
      return;
    }
    installPackage(entry, choice);
  }

  async function stopPackageDownload(entry: CatalogEntry, job: ModelInstallJob) {
    if (!['queued', 'running', 'cancelling'].includes(job.state)) return;
    status = `Stopping ${entry.display_name} download...`;
    try {
      const stopped = await stopModelInstall(job.id);
      installJobs = { ...installJobs, [stopped.id]: stopped };
      if (installPoll === null) installPoll = window.setInterval(refreshInstallJobs, 500);
      status = `${entry.display_name} download is stopping. Staging files will be removed automatically.`;
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      errorStatus = status;
      installJobs = {
        ...installJobs,
        [job.id]: {
          ...job,
          state: 'failed',
          message: status,
          finished_at_ms: Date.now()
        }
      };
    }
  }

  async function cleanPartialDownload(entry: CatalogEntry, job: ModelInstallJob) {
    if (['queued', 'running', 'cancelling'].includes(job.state)) return;
    status = `Cleaning partial ${entry.display_name} download...`;
    try {
      const result = await cleanPartialModelInstall(job.id);
      installJobs = {
        ...installJobs,
        [job.id]: {
          ...job,
          state: 'cleaned',
          message: result.message,
          downloaded_bytes: 0,
          total_bytes: 0,
          progress_percent: 100,
          finished_at_ms: Date.now()
        }
      };
      status = result.message;
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      errorStatus = status;
      installJobs = {
        ...installJobs,
        [job.id]: {
          ...job,
          state: 'failed',
          message: status,
          finished_at_ms: Date.now()
        }
      };
    }
  }

  async function removePackage(entry: CatalogEntry, choice: InstallPackageChoice) {
    if (!packageSizes[choice.id]?.installed) return;
    const confirmed = window.confirm(
      `Delete ${entry.display_name} ${choice.label}?\n\nOnly this package precision will be removed.`
    );
    if (!confirmed) return;
    status = `Deleting ${entry.display_name} ${choice.label}...`;
    try {
      if (packageIsResident(entry, choice)) {
        status = `Unloading ${entry.display_name} ${choice.label} before deletion...`;
        await unloadModel(entry.id);
        await refresh();
      }
      const result = await deleteModelPackage(choice.id);
      packageSizes = {
        ...packageSizes,
        [choice.id]: { ...packageSizes[choice.id], installed: false }
      };
      const nextJobs = { ...installJobs };
      delete nextJobs[choice.id];
      installJobs = nextJobs;

      if (packageIsSelected(entry, choice)) {
        const replacement = (entry.install_packages || []).find((candidate) =>
          candidate.id !== choice.id && packageSizes[candidate.id]?.installed);
        const nextIds = { ...selectedPackageIds };
        if (replacement) nextIds[entry.id] = replacement.id;
        else delete nextIds[entry.id];
        selectedPackageIds = nextIds;
        localStorage.setItem('audiocpp.ui.packageIds', JSON.stringify(selectedPackageIds));
        if (entry.id === selectedId) modelPath = resolveCatalogPath(replacement?.path || entry.path);
      }
      // Keep Studio synchronized even when its saved preference was stale or the
      // removed package shared a directory with another precision.
      reconcileSelectedPackageChoices(packageSizes);

      packageSizeState = 'idle';
      await refreshPackageSizes();
      if (entry.id === selectedId) {
        await inspectPath();
        clearUnavailableModelSelection();
      }
      status = result.message || `${entry.display_name} ${choice.label} deleted.`;
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      log(`Package deletion failed: ${status}`);
    }
  }

  onMount(async () => {
    await clearLegacyUiCaches();
    const savedLanguage = localStorage.getItem('audiocpp.ui.language');
    uiLanguage = resolveUiLanguage(savedLanguage ? [savedLanguage] : navigator.languages);
    document.documentElement.lang = uiLanguage;
    try {
      selectedPackageIds = JSON.parse(localStorage.getItem('audiocpp.ui.packageIds') || '{}');
    } catch {
      selectedPackageIds = {};
    }
    // Path-based preferences could select several packages sharing one directory.
    // Package IDs are unambiguous; discard the legacy preference after migration.
    localStorage.removeItem('audiocpp.ui.packagePaths');
    const stored = localStorage.getItem('audiocpp.ui.model');
    if (stored) selectedId = stored;
    selected = activeCatalog.find((entry) => entry.id === selectedId) || activeCatalog[0] || catalog[0];
    quickStartVoice = '';
    configuredVoices = [];
    activeWorkflow = workflowForTask(selected.task);
    if (selectedId) workflowSelections = { ...workflowSelections, [activeWorkflow]: selectedId };
    resetParams();
    await refresh();
    if (server?.ui_management) {
      try {
        let root = await getModelsRoot();
        const storedModelsFolder = localStorage.getItem('audiocpp.ui.modelsFolder');
        if (storedModelsFolder && storedModelsFolder !== root.models_root) {
          root = await setModelsRoot(storedModelsFolder);
        }
        acceptModelsRoot(root);
      } catch (error) {
        status = error instanceof Error ? error.message : String(error);
        errorStatus = status;
        log(`Models folder unavailable: ${status}`);
      }
    }
    modelPath = selectedModelPath(selected);
    if (server?.ui_management) {
      await refreshPackageSizes();
      await inspectPath();
      if (clearUnavailableModelSelection()) {
        status = 'No installed model is selected. Choose a downloaded model or install one from the Models tab.';
      }
    } else {
      installed = Boolean(selectedId);
    }
    await refreshVoices();
    await refreshBundledVoices();
    await refreshConfiguredVoices();
    await refreshInstallJobs();
  });

  onDestroy(() => {
    aborter?.abort();
    recorder?.state === 'recording' && recorder.stop();
    liveStopRequested = true;
    liveRecorder?.state === 'recording' && liveRecorder.stop();
    recordingStream?.getTracks().forEach((track) => track.stop());
    liveStream?.getTracks().forEach((track) => track.stop());
    for (const output of outputAudio) URL.revokeObjectURL(output.url);
    if (installPoll !== null) window.clearInterval(installPoll);
    if (packageSizePoll !== null) window.clearInterval(packageSizePoll);
  });
</script>

<svelte:head><title>audio.cpp · Native Studio</title></svelte:head>
<svelte:window on:keydown={handleShortcut} />

<header class="topbar">
  <div class="brand">
    <div class="mark">A</div>
    <div>
      <strong>audio.cpp</strong>
      <span>{tr('app.nativeStudio')}</span>
    </div>
  </div>
  <nav aria-label={tr('nav.primary')}>
    <button class:active={tab === 'studio'} on:click={openStudioPage}>{tr('nav.studio')}</button>
    {#if server?.ui_management !== false}
      <button class:active={tab === 'models'} on:click={openModelsPage}>{tr('nav.models')}</button>
    {/if}
    <button class:active={tab === 'logs'} on:click={() => tab = 'logs'}>{tr('nav.runtime')}</button>
  </nav>
  <label class="language-picker">
    <span>{tr('language.label')}</span>
    <select value={uiLanguage} aria-label={tr('language.label')}
      on:change={(event) => chooseUiLanguage(event.currentTarget.value)}>
      {#each uiLanguages as language}
        <option value={language.code}>{language.name}</option>
      {/each}
    </select>
  </label>
  <div class="server-pill" class:online={server?.status === 'ok'}>
    <i></i>{server?.backend || 'offline'}
  </div>
</header>

<main>
  {#if tab === 'studio'}
    <nav class="workflow-tabs" aria-label={tr('nav.workflows')}>
      {#each workflowTabs as workflow}
        <button class:active={activeWorkflow === workflow.id}
          on:click={() => chooseWorkflow(workflow.id)}>
          {workflowLabel(workflow.id, workflow.label, tr)}
          <small>{activeCatalog.filter((entry) => workflow.tasks.some((task) => task === entry.task)).length}</small>
        </button>
      {/each}
    </nav>

    <section class="hero">
      <div>
        <p class="eyebrow">{tr('studio.eyebrow')}</p>
        <h1>{selectedId ? localizedTaskLabel(selected?.task, tr) : tr('studio.title')}</h1>
        <p>{tr(`studio.subtitle.${activeWorkflow}`)}</p>
      </div>
      <div class="hero-stat">
        <span>{tr('studio.model')}</span>
        <strong>{selectedId ? selected.display_name : tr('studio.noModel')}</strong>
        <small class:ready={isLoaded}>{selectedId ? (isLoaded ? tr('studio.resident') : installed === false ? tr('studio.notInstalled') : tr('studio.available')) : tr('studio.chooseInstalled')}</small>
      </div>
    </section>

    <div class="studio-grid">
      <aside class="panel model-rail">
        <label for="model">{tr('studio.model')}</label>
        <select id="model" bind:value={selectedId} disabled={modelInventoryLoading}
          on:change={(event) => chooseModel(event.currentTarget.value)}>
          <option value="">{tr('studio.noModel')}</option>
          {#each activeWorkflowSpec.tasks as task}
            {@const entries = workflowModels.filter((entry) => entry.task === task)}
            {#if entries.length}
              <optgroup label={localizedTaskLabel(task, tr)}>
                {#each entries as entry}
                  <option value={entry.id} disabled={!selectableModelIds.has(entry.id)}>
                    {entry.display_name}{selectableModelIds.has(entry.id) ? '' : ` — ${tr('studio.notDownloaded')}`}
                  </option>
                {/each}
              </optgroup>
            {/if}
          {/each}
        </select>

        <div class="path-state">
          <span class:good={installed === true} class:bad={installed === false}>
            {!selectedId ? tr('studio.noModel') : installed === true ? tr('studio.pathFound') : installed === false ? tr('studio.pathMissing') : tr('studio.pathUnknown')}
          </span>
          <span>{selectedId ? tr('studio.estimatedVram', { value: selected?.min_vram_gb || '?' }) : tr('studio.vram')}</span>
        </div>

        {#if selectedId && (selected.install_packages || []).length}
          <div class="studio-package-buttons" aria-label="Model format">
            {#each studioPackageSlots(selected) as slot}
              {@const choice = slot.choice}
              {@const available = Boolean(choice && packageIsAvailable(selected, choice, loadedModels, packageSizes))}
              {@const resident = Boolean(choice && packageIsResident(selected, choice, loadedModels))}
              <button class:resident class:selected-package={Boolean(choice && available &&
                  packageIsSelected(selected, choice))}
                disabled={loadingModel || !available}
                title={resident ? `Unload ${choice?.label}` : available ? `Load ${choice?.label}` :
                  `${choice?.label || slot.label} is not downloaded`}
                on:click={() => choice && toggleStudioPackage(choice)}>
                {choice?.label || slot.label}
              </button>
            {/each}
          </div>
        {:else}
          <button class="single-model-toggle" class:resident={isLoaded}
            disabled={!selectedId || loadingModel || installed === false || !server?.ui_management}
            title={!server?.ui_management ? 'Configured by server config' : isLoaded ? tr('studio.unload') : tr('studio.load')}
            on:click={toggleSingleModel}>
            {!server?.ui_management ? (isLoaded ? tr('studio.bundledLoaded') : 'Configured') :
              loadingModel ? tr('studio.working') : isLoaded ? tr('studio.bundledLoaded') : tr('studio.load')}
          </button>
        {/if}

        {#if selectedId && (selected?.input_hint_en || selected?.input_hint)}
          <div class="hint">{localizedModelHint(tr)}</div>
        {/if}
      </aside>

      <section class="panel controls">
        {#if selectedId}
        <div class="section-title">
          <div><span>{tr('request.label')}</span><h2>{tr('request.title')}</h2></div>
          <span class="task-chip">{selected?.task}</span>
        </div>

        {#if showsText}
          <label for="text">{selected.task === 'gen' ? tr('request.prompt') : selected.task === 'align' ? tr('request.alignmentText') : tr('request.text')}</label>
          <textarea id="text" rows={selected.task === 'gen' ? 3 : 4} bind:value={text}
            placeholder={selected.task === 'gen' ? tr('request.soundPlaceholder') : tr('request.textPlaceholder')}></textarea>
        {/if}

        {#if ['tts', 'clon'].includes(selected.task)}
          <div class="long-text-row">
            <label class="toggle">
              <input type="checkbox" bind:checked={longText} />
              <span></span>{tr('request.splitLongText')}
            </label>
            <div>
              <label for="chunk-budget">{tr('request.charactersPerChunk')}</label>
              <input id="chunk-budget" type="number" min="40" max="10000" bind:value={chunkBudget}
                disabled={!longText} />
            </div>
          </div>
        {/if}

        {#if selected.task === 'gen'}
          <label for="lyrics">{tr('request.lyrics')} <span>{lyricsRequired ? tr('voice.required') : tr('request.optional')}</span></label>
          <textarea id="lyrics" rows="3" bind:value={lyrics} required={lyricsRequired}
            aria-required={lyricsRequired} placeholder="[Verse]…"></textarea>
        {/if}

        {#if selected.task === 'asr'}
          <label for="context">{tr('request.context')} <span>{tr('request.contextHint')}</span></label>
          <textarea id="context" rows="2" bind:value={context}></textarea>
        {/if}

        {#if selected.task === 'vdes'}
          <label for="instructions">{tr('request.voiceDescription')}</label>
          <textarea id="instructions" rows="2" bind:value={instructions}
            placeholder={tr('request.voiceDescriptionPlaceholder')}></textarea>
        {/if}

        <div class="field-grid">
          {#if ['tts', 'clon', 'asr', 'gen', 's2s', 'align', 'vdes'].includes(selected.task)}
            <div>
              <label for="language">{tr('request.language')} <span>{tr('request.autoLanguage')}</span></label>
              <input id="language" bind:value={language} placeholder="auto" />
            </div>
          {/if}
          {#if ['tts', 'clon', 'gen', 's2s', 'vdes'].includes(selected.task)}
            <div>
              <label for="seed">{tr('request.seed')} <span>{tr('request.randomSeed')}</span></label>
              <input id="seed" type="number" min="-1" max="4294967295" step="1" bind:value={seed} />
            </div>
          {/if}
          {#if supportsMaxTokens(selected)}
            <div>
              <label for="tokens">{tr('request.maxTokens')}</label>
              <input id="tokens" type="number" min="1" bind:value={maxTokens} />
            </div>
          {/if}
          {#if selected.task === 'gen'}
            <div>
              <label for="duration">{tr('request.duration')}</label>
              <input id="duration" type="number" min="1" step="0.1" value={duration}
                on:input={(event) => setDuration(event.currentTarget.valueAsNumber)} />
              {#if selected.family === 'minimax_h3'}
                <small>{tr('request.minimaxFrames', { frames: Number(advancedValues.num_frames || 0) })}</small>
              {/if}
            </div>
          {/if}
        </div>

        {#if acceptsSource}
          <label for="source">{tr('request.sourceAudio')} {needsSource ? '' : `(${tr('request.optional')})`}</label>
          <input id="source" class="file file-native" type="file" accept="audio/*"
            on:change={(event) => sourceFile = event.currentTarget.files?.[0] || null} />
          <label class="file-picker" for="source"><strong>{tr('file.choose')}</strong><span>{sourceFile?.name || tr('file.none')}</span></label>
          <div class="media-actions">
            {#if recordingTarget === 'source'}
              <button class="danger" type="button" on:click={stopRecording}>{tr('request.stopRecording')}</button>
              <span class="recording-dot">{tr('request.recordingMicrophone')}</span>
            {:else}
              <button type="button" disabled={Boolean(recorder) || liveRecording}
                on:click={() => startRecording('source')}>{tr('request.recordMicrophone')}</button>
              {#if sourceFile}<span>{sourceFile.name}</span>{/if}
            {/if}
          </div>
          {#if supportsLiveAsr}
            <div class="live-card">
              <div>
                <strong>{tr('request.liveTitle')}</strong>
                <small>{tr('request.liveDescription')}</small>
              </div>
              {#if liveRecording}
                <button class="danger" type="button" on:click={stopLiveTranscription}>{tr('request.stopLive')}</button>
              {:else}
                <button type="button" disabled={running || Boolean(recorder)}
                  on:click={startLiveTranscription}>{tr('request.startLive')}</button>
              {/if}
            </div>
          {/if}
        {/if}

        {#if needsVoice && !usesVibeVoiceSpeakerFiles}
          {#if allowsQuickStartVoice && quickStartVoices.length}
            <label for="quick-start-voice">{server?.ui_management === false ? tr('voice.configured') : tr('voice.quickStart')}</label>
            <select id="quick-start-voice" value={quickStartVoice}
              on:change={(event) => chooseQuickStartVoice(event.currentTarget.value)}>
              <option value="">{tr('voice.useReference')}</option>
              {#each quickStartVoices as voice}<option value={voice}>{voice}</option>{/each}
            </select>
            {#if quickStartVoice}
              <div class="quick-voice-note">
                {tr('voice.bundledNote')}
              </div>
            {/if}
          {/if}
          <div class="reference-input-grid">
            <div>
              <label for="voice">{tr('voice.reference')} <span>{referenceVoiceRequired ? tr('voice.required') : tr('voice.optional')}</span></label>
              <input id="voice" class="file file-native" type="file" accept="audio/*"
                bind:this={voiceInput}
                on:change={(event) => chooseVoiceReference(event.currentTarget.files?.[0] || null)} />
              <label class="file-picker" for="voice"><strong>{tr('file.choose')}</strong><span>{voiceFile?.name || tr('file.none')}</span></label>
            </div>
            <div>
              <label for="reference-file">{tr('voice.referenceText')} <span>.txt</span></label>
              <input id="reference-file" class="file file-native" type="file" accept=".txt,text/plain"
                bind:this={referenceTextInput}
                on:change={(event) => chooseReferenceText(event.currentTarget.files?.[0] || null)} />
              <label class="file-picker" for="reference-file"><strong>{tr('file.choose')}</strong><span>{referenceTextFile?.name || tr('file.none')}</span></label>
            </div>
          </div>
          <div class="media-actions">
            {#if recordingTarget === 'voice'}
              <button class="danger" type="button" on:click={stopRecording}>{tr('request.stopRecording')}</button>
              <span class="recording-dot">{tr('voice.recording')}</span>
            {:else}
              <button type="button" disabled={Boolean(recorder) || liveRecording}
                on:click={() => startRecording('voice')}>{tr('request.recordMicrophone')}</button>
              {#if voiceFile}<span>{voiceFile.name}</span>{/if}
            {/if}
          </div>
          <label for="reference">{tr('voice.transcript')}
            <span>{referenceTextRequired ? tr('voice.requiredClone') : tr('voice.recommendedClone')}</span>
          </label>
          <textarea id="reference" rows="2" bind:value={referenceText}
            placeholder={tr('voice.transcriptPlaceholder')}></textarea>
          <!--
            Saved voices keep a named reference recording and transcript for reuse. They are persisted only
            in this browser's IndexedDB, are never uploaded until the user runs a request, do not sync to
            another browser/device, and are removed if this site's browser data is cleared.
          -->
          <div class="voice-library">
            <div>
              <label for="saved-voice">{tr('voice.saved')} <span>{tr('voice.browserOnly')}</span></label>
              <select id="saved-voice" value={savedVoiceId}
                on:change={(event) => chooseSavedVoice(event.currentTarget.value)}>
                <option value="">{tr('voice.chooseSaved')}</option>
                {#each savedVoices as voice}<option value={voice.id}>{voice.name}</option>{/each}
              </select>
            </div>
            <div>
              <label for="voice-name">{tr('voice.libraryName')}</label>
              <input id="voice-name" bind:value={voiceName} placeholder={tr('voice.namePlaceholder')} />
            </div>
            <div class="library-actions">
              <button type="button" disabled={!voiceFile} on:click={storeCurrentVoice}>{tr('voice.save')}</button>
              <button class="danger" type="button" disabled={!savedVoiceId}
                on:click={removeCurrentVoice}>{tr('common.delete')}</button>
            </div>
          </div>
        {/if}

        {#if usesVibeVoiceSpeakerFiles}
          <div class="vibevoice-speakers">
            <div class="field-label">Speaker references <span>optional, up to 4</span></div>
            <div class="reference-input-grid">
              {#each [0, 1, 2, 3] as speaker}
                <div>
                  <label for={'vibevoice-speaker-' + speaker}>Speaker {speaker + 1}</label>
                  <input id={'vibevoice-speaker-' + speaker} class="file file-native" type="file" accept="audio/*"
                    on:change={(event) => chooseVibeVoiceSpeaker(speaker, event.currentTarget.files?.[0] || null)} />
                  <label class="file-picker" for={'vibevoice-speaker-' + speaker}>
                    <strong>{tr('file.choose')}</strong>
                    <span>{vibeVoiceSpeakerFiles[speaker]?.name || tr('file.none')}</span>
                  </label>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        {#if paramSpecs.length}
          <details>
            <summary>{tr('options.modelParameters')} <span>{paramSpecs.length}</span></summary>
            <div class="parameter-grid">
              {#each paramSpecs as spec}
                <div class:wide={spec.type === 'text'}>
                  <label for={'param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
                  {#if spec.type === 'bool'}
                    <label class="toggle">
                      <input id={'param-' + spec.name} type="checkbox"
                        checked={Boolean(advancedValues[spec.name])}
                        on:change={(event) => setParameterValue(spec, event.currentTarget.checked)} />
                      <span></span>{advancedValues[spec.name] ? tr('common.enabled') : tr('common.disabled')}
                    </label>
                  {:else if spec.type === 'choice'}
                    <select id={'param-' + spec.name} value={String(advancedValues[spec.name] ?? '')}
                      on:change={(event) => setParameterValue(spec, event.currentTarget.value)}>
                      {#each spec.choices || [] as choice}<option value={choice}>{choice}</option>{/each}
                    </select>
                  {:else if spec.type === 'slider'}
                    <div class="range">
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
          </details>
        {/if}

        <details>
          <summary>{tr('options.additional')} <span>JSON</span></summary>
          <textarea class="code" rows="3" bind:value={advancedJson}></textarea>
        </details>

        <div class="runbar">
          <button class="run" disabled={!selectedId || running || (!isLoaded && installed === false)} on:click={run}
            title={!selectedId ? 'Choose an installed model first' : !isLoaded && installed === false ? 'Install this model from the Models tab first' : ''}>
            <span>{running ? tr('run.working') : tr('run.run')}</span>
            <kbd>Ctrl ↵</kbd>
          </button>
          <button disabled={!running} on:click={cancel}>{tr('run.cancel')}</button>
          <div class="status" class:busy={running}
            class:warning={!running && status === warningStatus}
            class:error={!running && status === errorStatus}>{localizedStatus(status, tr)}</div>
        </div>
        {:else}
          <div class="section-title">
            <div><span>{tr('request.label')}</span><h2>{tr('studio.noModel')}</h2></div>
          </div>
          <div class="empty-output">
            <p>{tr('studio.chooseInstalled')}</p>
          </div>
        {/if}
      </section>

      <section class="panel output">
        <div class="section-title">
          <div><span>{tr('result.label')}</span><h2>{tr('result.title')}</h2></div>
          {#if outputAudio.length}<span class="task-chip">{outputAudio.length} {outputAudio.length === 1 ? tr('result.track') : tr('result.tracks')}</span>{/if}
        </div>
        {#if outputAudio.length}
          <div class="audio-list">
            {#each outputAudio as output}
              <article>
                <div><strong>{output.id}</strong><a href={output.url} download={`${selected.id}-${output.id}.wav`}>{tr('result.saveWav')}</a></div>
                <audio controls src={output.url}></audio>
              </article>
            {/each}
          </div>
        {/if}
        {#if outputArtifacts.length}
          <div class="audio-list">
            {#each outputArtifacts as artifact}
              <article>
                <div><strong>{artifact.id}</strong><a href={artifact.url} download={`${selected.id}-${artifact.id}.${artifact.extension}`}>Save {artifact.extension.toUpperCase()}</a></div>
              </article>
            {/each}
          </div>
        {/if}
        {#if !outputAudio.length && !outputArtifacts.length}
          <div class="empty-output"><div class="wave">∿</div><p>{tr('result.empty')}</p></div>
        {/if}
        {#if outputText}<textarea class="transcript" readonly rows="7" value={outputText}></textarea>{/if}
        {#if outputJson}<pre>{outputJson}</pre>{/if}
      </section>
    </div>
  {:else if tab === 'models'}
    <section class="page-head">
      <p class="eyebrow">{tr('models.eyebrow')}</p><h1>{tr('models.title')}</h1>
      <p>{tr('models.subtitle')}</p>
    </section>
    <section class="panel models-folder-options">
      <div class="models-folder-controls">
      <div class="models-folder-field">
        <label for="models-folder">{tr('models.folder')} <span>{tr('models.folderHint')}</span></label>
        <input id="models-folder" bind:value={modelsFolderInput}
          placeholder={defaultModelsFolder || tr('models.folderPlaceholder')} />
        {#if defaultModelsFolder}<small>{tr('models.default')}: {defaultModelsFolder}</small>{/if}
      </div>
      <button disabled={applyingModelsFolder} on:click={() => openFolderBrowser()}>{tr('common.browse')}</button>
      <button disabled={applyingModelsFolder || !modelsFolderInput.trim() || modelsFolderInput.trim() === modelsFolder}
        on:click={() => applyModelsFolder(false)}>{applyingModelsFolder ? tr('common.applying') : tr('common.apply')}</button>
      <button disabled={applyingModelsFolder || modelsFolderIsDefault}
        on:click={() => applyModelsFolder(true)}>{tr('models.useDefault')}</button>
      </div>
      <fieldset class="model-type-filters">
        <legend>{tr('models.showTypes')}</legend>
        <div>
          {#each workflowTabs as workflow}
            <label>
              <input type="checkbox" checked={modelWorkflowFilters.includes(workflow.id)}
                on:change={(event) => toggleModelWorkflowFilter(workflow.id, event.currentTarget.checked)} />
              <span>{workflowLabel(workflow.id, workflow.filterLabel, tr)}</span>
            </label>
          {/each}
        </div>
      </fieldset>
    </section>
    <section class="model-grid">
      {#each [0, 1] as column}
        <div class="model-column">
        {#each filteredModelGroups as group, groupIndex}
          {#if groupIndex % 2 === column}
        <article class="model-family-card" style={`--model-order: ${groupIndex}`}
          class:selected={group.entries.some((entry) => entry.id === selectedId)}>
          <div class="model-icon">{group.entries[0].task.toUpperCase()}</div>
          <div class="model-copy family-copy">
            <span>{group.entries.length} {group.entries.length === 1 ? tr('models.model') : tr('models.variants')}</span>
            <h3>{group.label}</h3>
            <p>{group.entries.map((entry) => localizedTaskLabel(entry.task, tr)).filter((value, index, all) => all.indexOf(value) === index).join(' · ')}</p>
          </div>
          <div class="model-variant-list">
            {#each group.entries as entry}
              {@const packageChoices = uniquePackagesForEntry(group, entry)}
              {@const installJob = displayInstallJobForChoices(packageChoices, installJobs)}
              <section class="model-variant" class:selected-variant={entry.id === selectedId}>
                <div class="variant-copy">
                  <strong>{entry.display_name}</strong>
                  <span>{localizedTaskLabel(entry.task, tr)} · VRAM ~{entry.min_vram_gb || '?'} GB</span>
                </div>
                <div class="model-actions">
                  {#if packageChoices.length}
                    <div class={`package-buttons${packageChoices.length > 3 ? ' wide-package-set' : ''}`}>
                      {#each packageChoices as choice}
                        <div class="package-choice">
                          <button class="package-install"
                            class:preferred={packageIsSelected(entry, choice)}
                            class:downloaded={packageSizes[choice.id]?.installed}
                            aria-pressed={packageIsSelected(entry, choice)}
                            disabled={groupInstallBusy(group, installJobs) ||
                              (packageSizeState === 'running' && Object.keys(packageSizes).length === 0)}
                            title={`${choice.format.toUpperCase()} ${choice.precision}: ${resolveCatalogPath(choice.path)}`}
                            on:click={() => useOrInstallPackage(entry, choice)}>
                            <span>{installButtonLabel(choice, installJobs[choice.id], tr)}</span>
                            {#if packageSizeLabel(packageSizes[choice.id], packageSizeState,
                              packageIsSelected(entry, choice), tr)}
                              <span class="package-size">{packageSizeLabel(packageSizes[choice.id], packageSizeState,
                                packageIsSelected(entry, choice), tr)}</span>
                            {/if}
                          </button>
                          {#if packageSizes[choice.id]?.installed &&
                            packageSizes[choice.id]?.version_state === 'update_available'}
                            <button class="package-update"
                              title={`Update ${choice.label}`}
                              aria-label={`Update ${entry.display_name} ${choice.label}`}
                              disabled={groupInstallBusy(group, installJobs)}
                              on:click={() => installPackage(entry, choice, true)}>
                              {tr('models.update')}
                            </button>
                          {/if}
                          {#if packageSizes[choice.id]?.installed}
                            <button class="package-delete"
                              title={`Delete ${choice.label}`} aria-label={`Delete ${entry.display_name} ${choice.label}`}
                              on:click={() => removePackage(entry, choice)}>
                              <svg viewBox="0 0 24 24" aria-hidden="true">
                                <path d="M9 3h6l1 2h4v2H4V5h4l1-2Zm-2 6h10l-1 11H8L7 9Zm3 2v7h2v-7h-2Zm4 0v7h2v-7h-2Z" />
                              </svg>
                            </button>
                          {/if}
                        </div>
                      {/each}
                    </div>
                    {#if installJob && installJob.state !== 'complete'}
                      <div class:failed={installJob.state === 'failed'}
                        class:cancelled={installJob.state === 'cancelled'}
                        class:cleaned={installJob.state === 'cleaned'}
                        class="install-progress" title={installJob.message}>
                        <div class="install-progress-head">
                          <strong>{installJob.state}</strong>
                          <span>{installProgressLabel(installJob)}</span>
                        </div>
                        <div class:indeterminate={['running', 'cancelling'].includes(installJob.state) && installJob.progress_percent < 0}
                          class="install-progress-track" role="progressbar"
                          aria-label={`${entry.display_name} download progress`}
                          aria-valuemin="0" aria-valuemax="100" aria-valuenow={installPercent(installJob)}>
                          <span style={`width: ${installPercent(installJob)}%`}></span>
                        </div>
                        <div class="install-status">{installJob.message}</div>
                        <div class="install-actions">
                          {#if ['queued', 'running', 'cancelling'].includes(installJob.state)}
                            <button class="danger" disabled={installJob.state === 'cancelling'}
                              on:click={() => stopPackageDownload(entry, installJob)}>{tr('models.stopDownload')}</button>
                          {:else if ['failed', 'cancelled'].includes(installJob.state)}
                            <button on:click={() => cleanPartialDownload(entry, installJob)}>{tr('models.cleanPartial')}</button>
                          {/if}
                        </div>
                      </div>
                    {/if}
                  {:else if (entry.install_packages || []).length}
                    <div class="shared-package-note">{tr('models.sharedPackage', { name: group.label })}</div>
                  {/if}
                </div>
              </section>
            {/each}
          </div>
        </article>
          {/if}
        {/each}
        </div>
      {/each}
    </section>
  {:else}
    <section class="page-head"><p class="eyebrow">{tr('runtime.eyebrow')}</p><h1>{tr('runtime.title')}</h1><p>{tr('runtime.subtitle')}</p></section>
    <section class="panel log-panel">
      <div class="runtime-cards">
        <div><span>{tr('runtime.status')}</span><strong>{server?.status || 'offline'}</strong></div>
        <div><span>{tr('runtime.backend')}</span><strong>{server?.backend || '—'}</strong></div>
        <div><span>{tr('runtime.registered')}</span><strong>{loadedModels.length}</strong></div>
        <div><span>{tr('runtime.resident')}</span><strong>{loadedModels.filter((model) => model.loaded).length}</strong></div>
      </div>
      <pre class="logs">{logs.length ? logs.join('\n') : tr('runtime.noEvents')}</pre>
    </section>
  {/if}
</main>

{#if folderBrowserOpen}
  <div class="folder-browser-backdrop">
    <div class="panel folder-browser-dialog" role="dialog" aria-modal="true" aria-labelledby="folder-browser-title">
      <header>
        <div><span>{tr('folder.eyebrow')}</span><h2 id="folder-browser-title">{tr('folder.title')}</h2></div>
        <button aria-label={tr('folder.closeLabel')} title={tr('common.close')} on:click={() => folderBrowserOpen = false}>{tr('common.close')}</button>
      </header>
      <div class="folder-browser-location">{folderBrowser?.current || tr('folder.loading')}</div>
      {#if folderBrowser?.roots.length}
        <div class="folder-browser-roots">
          {#each folderBrowser.roots as root}
            <button on:click={() => openFolderBrowser(root)}>{root}</button>
          {/each}
        </div>
      {/if}
      <div class="folder-browser-toolbar">
        <button disabled={!folderBrowser?.parent || folderBrowserLoading}
          on:click={() => openFolderBrowser(folderBrowser?.parent || '')}>{tr('folder.up')}</button>
        <button disabled={folderBrowserLoading}
          on:click={() => openFolderBrowser(folderBrowser?.current || '')}>{tr('common.refresh')}</button>
      </div>
      {#if folderBrowserError}
        <div class="folder-browser-error">{folderBrowserError}</div>
      {:else if folderBrowserLoading}
        <div class="folder-browser-empty">{tr('folder.loadingFolders')}</div>
      {:else if folderBrowser?.directories.length}
        <div class="folder-browser-list">
          {#each folderBrowser.directories as directory}
            <button title={directory.path} on:click={() => openFolderBrowser(directory.path)}>
              <span class="folder-icon">&gt;</span><span>{directory.name}</span>
            </button>
          {/each}
        </div>
      {:else}
        <div class="folder-browser-empty">{tr('folder.empty')}</div>
      {/if}
      <footer>
        <button on:click={() => folderBrowserOpen = false}>{tr('common.cancel')}</button>
        <button class="primary" disabled={!folderBrowser || folderBrowserLoading}
          on:click={selectBrowsedFolder}>{tr('folder.select')}</button>
      </footer>
    </div>
  </div>
{/if}

<footer><span>audio.cpp native WebUI</span><span>{tr('footer.embedded')}</span></footer>
