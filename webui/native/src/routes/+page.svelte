<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { browserDecodeToWav, concatenateAudioBlobs } from '$lib/audio';
  import {
    base64AudioUrl,
    browseDirectories,
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
    transcription,
    unloadModel,
    uploadWav,
    type ModelInstallJob,
    type ModelPackageSize,
    type DirectoryBrowserResponse
  } from '$lib/api';
  import { catalog, parameterCatalog, taskLabels } from '$lib/catalog';
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
  let referenceTextFile: File | null = null;
  let referenceTextInput: HTMLInputElement | null = null;
  let advancedJson = '{}';
  let advancedValues: Record<string, unknown> = {};
  let paramSpecs: ParamSpec[] = [];
  let outputAudio: AudioOutput[] = [];
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
  let selectedPackagePaths: Record<string, string> = {};
  let packageSizes: Record<string, ModelPackageSize> = {};
  let packageSizeState: 'idle' | 'running' | 'complete' | 'failed' = 'idle';
  let packageSizePoll: number | null = null;
  let packageSizeRefreshInFlight = false;
  let refreshedInstallFinishes: Record<string, number> = {};

  const workflowTabs = [
    { id: 'tts', label: 'Text to speech', tasks: ['tts', 'clon'] },
    { id: 'asr', label: 'ASR / Transcription', tasks: ['asr'] },
    { id: 'music', label: 'Music generation', tasks: ['gen'] },
    { id: 'conversion', label: 'Voice conversion', tasks: ['vc', 'svc', 's2s'] },
    { id: 'separation', label: 'Source separation', tasks: ['sep'] },
    { id: 'analysis', label: 'Audio analysis', tasks: ['vad', 'diar', 'align', 'spk'] },
    { id: 'design', label: 'Voice design', tasks: ['vdes'] }
  ] as const;

  type WorkflowId = typeof workflowTabs[number]['id'];

  let activeWorkflow: WorkflowId = 'tts';
  let workflowSelections: Partial<Record<WorkflowId, string>> = {};

  class StatusWarning extends Error {}

  $: selected = catalog.find((entry) => entry.id === selectedId) || catalog[0];
  $: activeWorkflowSpec = workflowTabs.find((workflow) => workflow.id === activeWorkflow) || workflowTabs[0];
  $: workflowModels = catalog.filter((entry) =>
    activeWorkflowSpec.tasks.some((task) => task === entry.task));
  $: isLoaded = loadedModels.some((model) => model.id === selectedId && model.loaded);
  $: needsSource = ['asr', 'vc', 'svc', 's2s', 'sep', 'vad', 'diar', 'align'].includes(selected?.task);
  $: acceptsSource = needsSource || selected?.task === 'gen';
  $: needsVoice = ['clon', 'vc', 'svc'].includes(selected?.task) ||
    (selected?.task === 'tts' && !['supertonic'].includes(selected?.family));
  $: isQwenBase = selected?.task === 'tts' && selected?.family === 'qwen3_tts' &&
    !selected?.id.includes('custom');
  $: referenceVoiceRequired = ['clon', 'vc', 'svc'].includes(selected?.task) ||
    isQwenBase || (selected?.task === 'tts' && selected?.family === 'pocket_tts');
  $: referenceTextRequired = Boolean(voiceFile) && isQwenBase;
  $: showsText = ['tts', 'clon', 'gen', 's2s', 'align', 'vdes'].includes(selected?.task);
  $: supportsLiveAsr = selected?.task === 'asr' &&
    ['voxtral_realtime', 'nemotron_asr', 'higgs_audio_stt'].includes(selected?.family);
  $: modelInventoryLoading = server === null ||
    (Boolean(server.ui_management) && Object.keys(packageSizes).length === 0 && packageSizeState !== 'failed');
  $: selectableModelIds = new Set(catalog.filter((entry) => {
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

  function selectedModelPath(entry: CatalogEntry) {
    return resolveCatalogPath(selectedPackagePaths[entry.id] || entry.path);
  }

  function comparablePath(path: string) {
    return path.replace(/\\/g, '/').replace(/\/$/, '').toLowerCase();
  }

  function residentModel(entry: CatalogEntry, models = loadedModels) {
    return models.find((model) => model.id === entry.id && model.loaded);
  }

  function packageIsResident(entry: CatalogEntry, choice: InstallPackageChoice, models = loadedModels) {
    const resident = residentModel(entry, models);
    const selectedChoicePath = selectedPackagePaths[entry.id] || entry.path;
    const expectedPath = entry.id === selectedId && choice.path === selectedChoicePath
      ? modelPath
      : resolveCatalogPath(choice.path);
    return Boolean(resident && comparablePath(resident.path) === comparablePath(expectedPath));
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
    return [
      {
        key: 'q8',
        label: 'GGUF Q8',
        choice: choices.find((choice) => choice.format === 'gguf' && ['q8', 'q8_0'].includes(choice.precision))
      },
      {
        key: 'fp16',
        label: 'GGUF FP16',
        choice: choices.find((choice) => choice.format === 'gguf' &&
          ['f16', 'fp16', 'bf16'].includes(choice.precision))
      },
      {
        key: 'safetensors',
        label: 'Safetensors',
        choice: choices.find((choice) => choice.format === 'safetensors')
      }
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
    const changed = Boolean(voiceFile && file && voiceFile.name !== file.name);
    voiceFile = file;
    if (changed) {
      referenceTextFile = null;
      referenceText = '';
      if (referenceTextInput) referenceTextInput.value = '';
      status = 'Reference voice changed. Choose or enter its matching transcript.';
      warningStatus = status;
    }
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
    if (job.state === 'complete') return 100;
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
    if (job.state === 'complete') return '100% · complete';
    return job.state === 'queued' ? '0% · queued' : 'Connecting and checking package files…';
  }

  function entryInstallJobs(entry: CatalogEntry, jobs: Record<string, ModelInstallJob>) {
    return (entry.install_packages || [])
      .map((choice) => jobs[choice.id])
      .filter((job): job is ModelInstallJob => job !== undefined);
  }

  function displayInstallJob(entry: CatalogEntry, installState: Record<string, ModelInstallJob>) {
    const jobs = entryInstallJobs(entry, installState);
    return jobs.find((job) => job.state === 'running' || job.state === 'queued') ||
      [...jobs].sort((left, right) => right.finished_at_ms - left.finished_at_ms)[0];
  }

  function entryInstallBusy(entry: CatalogEntry, installState: Record<string, ModelInstallJob>) {
    return entryInstallJobs(entry, installState).some((job) =>
      job.state === 'running' || job.state === 'queued');
  }

  function entrySelectable(entry: CatalogEntry) {
    return selectableModelIds.has(entry.id);
  }

  function workflowForTask(task: string | undefined): WorkflowId {
    return (workflowTabs.find((workflow) => workflow.tasks.some((candidate) => candidate === task))?.id || 'tts') as WorkflowId;
  }

  function installButtonLabel(
    choice: InstallPackageChoice,
    job: ModelInstallJob | undefined
  ) {
    if (job?.state === 'running') return `${choice.label}…`;
    if (job?.state === 'queued') return `${choice.label} queued`;
    return choice.label;
  }

  function packageSizeLabel(
    size: ModelPackageSize | undefined,
    sizeState: 'idle' | 'running' | 'complete' | 'failed',
    selected: boolean
  ) {
    const bytes = size?.size_bytes !== null && size?.size_bytes !== undefined
      ? formatBytes(size.size_bytes)
      : '';
    if (size?.installed) return `${selected ? 'Selected' : 'Downloaded'}${bytes ? ` · ${bytes}` : ''}`;
    if (bytes) return bytes;
    if (size?.state === 'pending') return 'checking size...';
    if (size?.state === 'gated') return 'HF access required';
    if (size?.state === 'error' || size?.state === 'unknown') return 'size unavailable';
    return sizeState === 'running' ? 'checking size…' : '';
  }

  async function refreshPackageSizes() {
    if (!server?.ui_management || packageSizeRefreshInFlight) return;
    packageSizeRefreshInFlight = true;
    try {
      const response = await modelPackageSizes();
      packageSizeState = response.state;
      if (response.data.length) {
        packageSizes = Object.fromEntries(response.data.map((size) => [size.id, size]));
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
    tab = 'models';
    if (packageSizeState === 'idle') packageSizeState = 'running';
    refreshPackageSizes();
  }

  function rememberPackagePath(entry: CatalogEntry, choice: InstallPackageChoice) {
    selectedPackagePaths = { ...selectedPackagePaths, [entry.id]: choice.path };
    localStorage.setItem('audiocpp.ui.packagePaths', JSON.stringify(selectedPackagePaths));
    if (entry.id === selectedId) modelPath = resolveCatalogPath(choice.path);
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
    modelPath = '';
    installed = null;
    paramSpecs = [];
    advancedValues = {};
    advancedJson = '{}';
    localStorage.removeItem('audiocpp.ui.model');
    clearOutput();
  }

  function clearUnavailableModelSelection(ignoreResident = false) {
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
    paramSpecs = byId;
    advancedValues = Object.fromEntries(byId.map((spec) => [spec.name, spec.default ?? '']));
    advancedJson = '{}';
  }

  async function refresh() {
    try {
      [server, loadedModels] = await Promise.all([health(), models()]);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    }
  }

  async function inspectPath() {
    if (!selectedId || !modelPath.trim()) {
      installed = null;
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
    const next = catalog.find((entry) => entry.id === id);
    if (!next || !entrySelectable(next)) return;
    selectedId = id;
    selected = next;
    activeWorkflow = workflowForTask(next.task);
    workflowSelections = { ...workflowSelections, [activeWorkflow]: id };
    modelPath = selectedModelPath(next);
    chunkBudget = defaultChunkBudget(next.family);
    localStorage.setItem('audiocpp.ui.model', id);
    resetParams();
    inspectPath();
  }

  function chooseWorkflow(id: WorkflowId) {
    const workflow = workflowTabs.find((entry) => entry.id === id);
    if (!workflow) return;
    activeWorkflow = id;
    if (selectedId && workflow.tasks.some((task) => task === selected?.task)) return;

    const rememberedId = workflowSelections[id];
    const remembered = rememberedId
      ? catalog.find((entry) => entry.id === rememberedId &&
          workflow.tasks.some((task) => task === entry.task) && entrySelectable(entry))
      : undefined;
    const next = remembered || catalog.find((entry) =>
      workflow.tasks.some((task) => task === entry.task) && entrySelectable(entry));
    if (next) {
      chooseModel(next.id);
      return;
    }

    clearModelSelection();
    status = `No installed models are available for ${workflow.label}. Install one from the Models tab.`;
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
      status = `${selected.display_name} is resident and ready.`;
      errorStatus = '';
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      errorStatus = status;
      log(`Load failed: ${status}`);
    } finally {
      loadingModel = false;
    }
  }

  async function doUnload() {
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

    rememberPackagePath(selected, choice);
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
      : ['asr', 'vad', 'diar', 'align'].includes(selected.task) ? 16000 : undefined;
    const wav = await browserDecodeToWav(file, targetSampleRate);
    return uploadWav(wav, file.name.replace(/\.[^.]+$/, '') + '.wav', aborter?.signal);
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
    outputText = '';
    outputJson = '';
  }

  async function ensureLoaded() {
    if (!isLoaded) {
      await doLoad();
      await refresh();
      if (!loadedModels.some((model) => model.id === selectedId && model.loaded)) {
        throw new Error('Model did not load.');
      }
    }
  }

  async function ensureLoadedMode(mode: string) {
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
        else voiceFile = file;
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
    voiceFile = new File([voice.audio], `${voice.name}.wav`, { type: 'audio/wav' });
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
    const audio = await uploadWav(wav, `live-${liveChunkNumber}.wav`);
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
    status = `Running ${taskLabels[selected.task] || selected.task}…`;
    log(status);
    try {
      const resolvedSeed = resolveRequestSeed(seed);
      if (referenceVoiceRequired && !voiceFile) {
        throw new StatusWarning(`${selected.display_name_en || selected.display_name} requires a reference voice.`);
      }
      if (referenceTextRequired && !referenceText.trim()) {
        throw new StatusWarning('Qwen3-TTS Base voice cloning requires a reference transcript. Choose a matching .txt file or enter the transcript.');
      }
      await ensureLoaded();
      const options = requestOptions();
      const audio = acceptsSource ? await stagedPath(sourceFile) : undefined;
      const voiceRef = needsVoice ? await stagedPath(voiceFile) : undefined;

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
            : `Running ${taskLabels[selected.task] || selected.task}…`;
          const body: Record<string, unknown> = {
            model: selected.id,
            input: chunks[index],
            language,
            seed: chunkSeed(resolvedSeed, index),
            max_tokens: maxTokens,
            options
          };
          if (voiceRef) body.voice_ref = voiceRef;
          if (referenceText.trim()) body.reference_text = referenceText;
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
          request.max_tokens = maxTokens;
        }
        if (audio) request.audio = audio;
        if (voiceRef) request.voice_ref = voiceRef;
        if (referenceText.trim()) request.reference_text = referenceText;
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
        outputText = typeof result.text === 'string' ? result.text : '';
        outputJson = JSON.stringify(result, (key, value) =>
          key === 'audio' && typeof value === 'string' ? `<base64 audio: ${value.length} chars>` : value, 2);
      }
      const elapsed = ((performance.now() - started) / 1000).toFixed(2);
      warningStatus = '';
      errorStatus = '';
      status = `Complete in ${elapsed}s.`;
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
        job.state === 'queued' || job.state === 'running');
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

  async function installPackage(entry: CatalogEntry, choice: InstallPackageChoice) {
    if (packageSizes[choice.id]?.installed) return;
    const existing = installJobs[choice.id];
    if (existing?.state === 'queued' || existing?.state === 'running') return;
    rememberPackagePath(entry, choice);
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
      const job = await installModelPackage({ id: choice.id });
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
      rememberPackagePath(entry, choice);
      status = `${entry.display_name} will use ${choice.label}. Press Open to continue.`;
      log(status);
      return;
    }
    installPackage(entry, choice);
  }

  async function removePackage(entry: CatalogEntry, choice: InstallPackageChoice) {
    if (!packageSizes[choice.id]?.installed) return;
    const confirmed = window.confirm(
      `Delete ${entry.display_name} ${choice.label}?\n\nOnly this package precision will be removed.`
    );
    if (!confirmed) return;
    status = `Deleting ${entry.display_name} ${choice.label}...`;
    try {
      const result = await deleteModelPackage(choice.id);
      packageSizes = {
        ...packageSizes,
        [choice.id]: { ...packageSizes[choice.id], installed: false }
      };
      const nextJobs = { ...installJobs };
      delete nextJobs[choice.id];
      installJobs = nextJobs;

      if (selectedPackagePaths[entry.id] === choice.path) {
        const replacement = (entry.install_packages || []).find((candidate) =>
          candidate.id !== choice.id && packageSizes[candidate.id]?.installed);
        const nextPaths = { ...selectedPackagePaths };
        if (replacement) nextPaths[entry.id] = replacement.path;
        else delete nextPaths[entry.id];
        selectedPackagePaths = nextPaths;
        localStorage.setItem('audiocpp.ui.packagePaths', JSON.stringify(selectedPackagePaths));
        if (entry.id === selectedId) modelPath = resolveCatalogPath(replacement?.path || entry.path);
      }

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
    try {
      selectedPackagePaths = JSON.parse(localStorage.getItem('audiocpp.ui.packagePaths') || '{}');
    } catch {
      selectedPackagePaths = {};
    }
    const stored = localStorage.getItem('audiocpp.ui.model');
    if (stored && catalog.some((entry) => entry.id === stored)) selectedId = stored;
    selected = catalog.find((entry) => entry.id === selectedId) || catalog[0];
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
    await refreshPackageSizes();
    await inspectPath();
    if (clearUnavailableModelSelection()) {
      status = 'No installed model is selected. Choose a downloaded model or install one from the Models tab.';
    }
    await refreshVoices();
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
      <span>Native Studio</span>
    </div>
  </div>
  <nav aria-label="Primary navigation">
    <button class:active={tab === 'studio'} on:click={() => tab = 'studio'}>Studio</button>
    <button class:active={tab === 'models'} on:click={openModelsPage}>Models</button>
    <button class:active={tab === 'logs'} on:click={() => tab = 'logs'}>Runtime</button>
  </nav>
  <div class="server-pill" class:online={server?.status === 'ok'}>
    <i></i>{server?.backend || 'offline'}
  </div>
</header>

<main>
  {#if tab === 'studio'}
    <nav class="workflow-tabs" aria-label="Audio workflows">
      {#each workflowTabs as workflow}
        <button class:active={activeWorkflow === workflow.id}
          on:click={() => chooseWorkflow(workflow.id)}>
          {workflow.label}
          <small>{catalog.filter((entry) => workflow.tasks.some((task) => task === entry.task)).length}</small>
        </button>
      {/each}
    </nav>

    <section class="hero">
      <div>
        <p class="eyebrow">LOCAL AUDIO INTELLIGENCE</p>
        <h1>{selectedId ? taskLabels[selected?.task] : 'Audio studio'}</h1>
        <p>One native server, one embedded interface, no Python between your browser and the model.</p>
      </div>
      <div class="hero-stat">
        <span>Model</span>
        <strong>{selectedId ? selected.display_name : 'No model selected'}</strong>
        <small class:ready={isLoaded}>{selectedId ? (isLoaded ? 'Resident' : installed === false ? 'Not installed' : 'Available') : 'Choose an installed model'}</small>
      </div>
    </section>

    <div class="studio-grid">
      <aside class="panel model-rail">
        <label for="model">Model</label>
        <select id="model" bind:value={selectedId} disabled={modelInventoryLoading}
          on:change={(event) => chooseModel(event.currentTarget.value)}>
          <option value="">No model selected</option>
          {#each activeWorkflowSpec.tasks as task}
            {@const entries = workflowModels.filter((entry) => entry.task === task)}
            {#if entries.length}
              <optgroup label={taskLabels[task]}>
                {#each entries as entry}
                  <option value={entry.id} disabled={!selectableModelIds.has(entry.id)}>
                    {entry.display_name}{selectableModelIds.has(entry.id) ? '' : ' — not downloaded'}
                  </option>
                {/each}
              </optgroup>
            {/if}
          {/each}
        </select>

        <div class="path-state">
          <span class:good={installed === true} class:bad={installed === false}>
            {!selectedId ? 'No model selected' : installed === true ? 'Path found' : installed === false ? 'Path missing' : 'Path not inspected'}
          </span>
          <span>{selectedId ? `${selected?.min_vram_gb || '?'} GB estimated VRAM` : 'VRAM —'}</span>
        </div>

        {#if selectedId && (selected.install_packages || []).length}
          <div class="studio-package-buttons" aria-label="Model format">
            {#each studioPackageSlots(selected) as slot}
              {@const choice = slot.choice}
              {@const available = Boolean(choice && packageIsAvailable(selected, choice, loadedModels, packageSizes))}
              {@const resident = Boolean(choice && packageIsResident(selected, choice, loadedModels))}
              <button class:resident class:selected-package={Boolean(choice &&
                  choice.path === (selectedPackagePaths[selected.id] || selected.path))}
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
            disabled={!selectedId || loadingModel || installed === false}
            title={isLoaded ? 'Unload model' : 'Load model'} on:click={toggleSingleModel}>
            {loadingModel ? 'Working…' : isLoaded ? 'Bundled · loaded' : 'Load model'}
          </button>
        {/if}

        {#if selectedId && (selected?.input_hint_en || selected?.input_hint)}
          <div class="hint">{selected.input_hint_en || selected.input_hint}</div>
        {/if}
      </aside>

      <section class="panel controls">
        {#if selectedId}
        <div class="section-title">
          <div><span>REQUEST</span><h2>Input & controls</h2></div>
          <span class="task-chip">{selected?.task}</span>
        </div>

        {#if showsText}
          <label for="text">{selected.task === 'gen' ? 'Prompt' : selected.task === 'align' ? 'Alignment text' : 'Text'}</label>
          <textarea id="text" rows={selected.task === 'gen' ? 3 : 4} bind:value={text}
            placeholder={selected.task === 'gen' ? 'Describe the sound or music…' : 'Enter the text…'}></textarea>
        {/if}

        {#if ['tts', 'clon'].includes(selected.task)}
          <div class="long-text-row">
            <label class="toggle">
              <input type="checkbox" bind:checked={longText} />
              <span></span>Split and merge long text
            </label>
            <div>
              <label for="chunk-budget">Characters per chunk</label>
              <input id="chunk-budget" type="number" min="40" max="10000" bind:value={chunkBudget}
                disabled={!longText} />
            </div>
          </div>
        {/if}

        {#if selected.task === 'gen'}
          <label for="lyrics">Lyrics <span>optional</span></label>
          <textarea id="lyrics" rows="3" bind:value={lyrics} placeholder="[Verse]…"></textarea>
        {/if}

        {#if selected.task === 'asr'}
          <label for="context">Context prompt <span>optional terminology or names</span></label>
          <textarea id="context" rows="2" bind:value={context}></textarea>
        {/if}

        {#if selected.task === 'vdes'}
          <label for="instructions">Voice description</label>
          <textarea id="instructions" rows="2" bind:value={instructions}
            placeholder="A warm, calm voice with measured pacing…"></textarea>
        {/if}

        <div class="field-grid">
          {#if ['tts', 'clon', 'asr', 'gen', 's2s', 'align', 'vdes'].includes(selected.task)}
            <div>
              <label for="language">Language <span>blank = auto</span></label>
              <input id="language" bind:value={language} placeholder="auto" />
            </div>
          {/if}
          {#if ['tts', 'clon', 'gen', 's2s', 'vdes'].includes(selected.task)}
            <div>
              <label for="seed">Seed <span>-1 = random</span></label>
              <input id="seed" type="number" min="-1" max="4294967295" step="1" bind:value={seed} />
            </div>
            <div>
              <label for="tokens">Maximum tokens</label>
              <input id="tokens" type="number" min="1" bind:value={maxTokens} />
            </div>
          {/if}
          {#if selected.task === 'gen'}
            <div>
              <label for="duration">Duration seconds</label>
              <input id="duration" type="number" min="1" bind:value={duration} />
            </div>
          {/if}
        </div>

        {#if acceptsSource}
          <label for="source">Source audio {needsSource ? '' : '(optional)'}</label>
          <input id="source" class="file" type="file" accept="audio/*"
            on:change={(event) => sourceFile = event.currentTarget.files?.[0] || null} />
          <div class="media-actions">
            {#if recordingTarget === 'source'}
              <button class="danger" type="button" on:click={stopRecording}>Stop recording</button>
              <span class="recording-dot">Recording microphone</span>
            {:else}
              <button type="button" disabled={Boolean(recorder) || liveRecording}
                on:click={() => startRecording('source')}>Record microphone</button>
              {#if sourceFile}<span>{sourceFile.name}</span>{/if}
            {/if}
          </div>
          {#if supportsLiveAsr}
            <div class="live-card">
              <div>
                <strong>Live microphone transcription</strong>
                <small>Processes consecutive four-second requests using the model's streaming mode.</small>
              </div>
              {#if liveRecording}
                <button class="danger" type="button" on:click={stopLiveTranscription}>Stop live</button>
              {:else}
                <button type="button" disabled={running || Boolean(recorder)}
                  on:click={startLiveTranscription}>Start live</button>
              {/if}
            </div>
          {/if}
        {/if}

        {#if needsVoice}
          <div class="reference-input-grid">
            <div>
              <label for="voice">Reference voice <span>{referenceVoiceRequired ? 'required' : 'optional'}</span></label>
              <input id="voice" class="file" type="file" accept="audio/*"
                on:change={(event) => chooseVoiceReference(event.currentTarget.files?.[0] || null)} />
            </div>
            <div>
              <label for="reference-file">Reference text <span>.txt</span></label>
              <input id="reference-file" class="file" type="file" accept=".txt,text/plain"
                bind:this={referenceTextInput}
                on:change={(event) => chooseReferenceText(event.currentTarget.files?.[0] || null)} />
            </div>
          </div>
          <div class="media-actions">
            {#if recordingTarget === 'voice'}
              <button class="danger" type="button" on:click={stopRecording}>Stop recording</button>
              <span class="recording-dot">Recording voice reference</span>
            {:else}
              <button type="button" disabled={Boolean(recorder) || liveRecording}
                on:click={() => startRecording('voice')}>Record microphone</button>
              {#if voiceFile}<span>{voiceFile.name}</span>{/if}
            {/if}
          </div>
          <label for="reference">Reference transcript
            <span>{referenceTextRequired ? 'required for this voice clone' : 'recommended for cloning'}</span>
          </label>
          <textarea id="reference" rows="2" bind:value={referenceText}
            placeholder="Type the exact words spoken in the reference audio, or load a matching .txt file above."></textarea>
          <div class="voice-library">
            <div>
              <label for="saved-voice">Saved voices <span>stored only in this browser</span></label>
              <select id="saved-voice" value={savedVoiceId}
                on:change={(event) => chooseSavedVoice(event.currentTarget.value)}>
                <option value="">Choose a saved voice...</option>
                {#each savedVoices as voice}<option value={voice.id}>{voice.name}</option>{/each}
              </select>
            </div>
            <div>
              <label for="voice-name">Library name</label>
              <input id="voice-name" bind:value={voiceName} placeholder="My reference voice" />
            </div>
            <div class="library-actions">
              <button type="button" disabled={!voiceFile} on:click={storeCurrentVoice}>Save voice</button>
              <button class="danger" type="button" disabled={!savedVoiceId}
                on:click={removeCurrentVoice}>Delete</button>
            </div>
          </div>
        {/if}

        {#if paramSpecs.length}
          <details>
            <summary>Model parameters <span>{paramSpecs.length}</span></summary>
            <div class="parameter-grid">
              {#each paramSpecs as spec}
                <div class:wide={spec.type === 'text'}>
                  <label for={'param-' + spec.name}>{spec.label_en || spec.label}</label>
                  {#if spec.type === 'bool'}
                    <label class="toggle">
                      <input id={'param-' + spec.name} type="checkbox"
                        checked={Boolean(advancedValues[spec.name])}
                        on:change={(event) => advancedValues = {...advancedValues, [spec.name]: event.currentTarget.checked}} />
                      <span></span>{advancedValues[spec.name] ? 'Enabled' : 'Disabled'}
                    </label>
                  {:else if spec.type === 'choice'}
                    <select id={'param-' + spec.name} value={String(advancedValues[spec.name] ?? '')}
                      on:change={(event) => advancedValues = {...advancedValues, [spec.name]: event.currentTarget.value}}>
                      {#each spec.choices || [] as choice}<option value={choice}>{choice}</option>{/each}
                    </select>
                  {:else if spec.type === 'slider'}
                    <div class="range">
                      <input id={'param-' + spec.name} type="range" min={spec.minimum} max={spec.maximum} step={spec.step}
                        value={Number(advancedValues[spec.name] ?? spec.default)}
                        on:input={(event) => advancedValues = {...advancedValues, [spec.name]: event.currentTarget.valueAsNumber}} />
                      <output>{String(advancedValues[spec.name])}</output>
                    </div>
                  {:else}
                    <input id={'param-' + spec.name} type={spec.type === 'number' ? 'number' : 'text'}
                      min={spec.minimum} max={spec.maximum} step={spec.step}
                      value={String(advancedValues[spec.name] ?? '')}
                      placeholder={spec.placeholder_en || spec.placeholder || ''}
                      on:input={(event) => advancedValues = {...advancedValues,
                        [spec.name]: spec.type === 'number' ? event.currentTarget.valueAsNumber : event.currentTarget.value}} />
                  {/if}
                  {#if spec.info_en || spec.info}<small>{spec.info_en || spec.info}</small>{/if}
                </div>
              {/each}
            </div>
          </details>
        {/if}

        <details>
          <summary>Additional options <span>JSON</span></summary>
          <textarea class="code" rows="3" bind:value={advancedJson}></textarea>
        </details>

        <div class="runbar">
          <button class="run" disabled={!selectedId || running || (!isLoaded && installed === false)} on:click={run}
            title={!selectedId ? 'Choose an installed model first' : !isLoaded && installed === false ? 'Install this model from the Models tab first' : ''}>
            <span>{running ? 'Working…' : 'Run'}</span>
            <kbd>Ctrl ↵</kbd>
          </button>
          <button disabled={!running} on:click={cancel}>Cancel</button>
          <div class="status" class:busy={running}
            class:warning={!running && status === warningStatus}
            class:error={!running && status === errorStatus}>{status}</div>
        </div>
        {:else}
          <div class="section-title">
            <div><span>REQUEST</span><h2>No model selected</h2></div>
          </div>
          <div class="empty-output">
            <p>Choose a downloaded model from the Model menu, or install one from the Models tab.</p>
          </div>
        {/if}
      </section>

      <section class="panel output">
        <div class="section-title">
          <div><span>RESULT</span><h2>Output</h2></div>
          {#if outputAudio.length}<span class="task-chip">{outputAudio.length} track{outputAudio.length === 1 ? '' : 's'}</span>{/if}
        </div>
        {#if outputAudio.length}
          <div class="audio-list">
            {#each outputAudio as output}
              <article>
                <div><strong>{output.id}</strong><a href={output.url} download={`${selected.id}-${output.id}.wav`}>Save WAV</a></div>
                <audio controls src={output.url}></audio>
              </article>
            {/each}
          </div>
        {:else}
          <div class="empty-output"><div class="wave">∿</div><p>Generated audio and structured results appear here.</p></div>
        {/if}
        {#if outputText}<textarea class="transcript" readonly rows="7" value={outputText}></textarea>{/if}
        {#if outputJson}<pre>{outputJson}</pre>{/if}
      </section>
    </div>
  {:else if tab === 'models'}
    <section class="page-head">
      <p class="eyebrow">MODEL LIBRARY</p><h1>Local packages</h1>
      <p>Download and manage model packages without leaving the native interface.</p>
    </section>
    <section class="panel models-folder-options">
      <div class="models-folder-field">
        <label for="models-folder">Models folder <span>downloads, local detection, and model loading</span></label>
        <input id="models-folder" bind:value={modelsFolderInput}
          placeholder={defaultModelsFolder || 'models folder beside audiocpp_server'} />
        {#if defaultModelsFolder}<small>Default: {defaultModelsFolder}</small>{/if}
      </div>
      <button disabled={applyingModelsFolder} on:click={() => openFolderBrowser()}>Browse</button>
      <button disabled={applyingModelsFolder || !modelsFolderInput.trim() || modelsFolderInput.trim() === modelsFolder}
        on:click={() => applyModelsFolder(false)}>{applyingModelsFolder ? 'Applyingâ€¦' : 'Apply'}</button>
      <button disabled={applyingModelsFolder || modelsFolderIsDefault}
        on:click={() => applyModelsFolder(true)}>Use default</button>
    </section>
    <section class="model-grid">
      {#each catalog as entry}
        {@const installJob = displayInstallJob(entry, installJobs)}
        {@const packageChoices = entry.install_packages || []}
        <article class:selected={entry.id === selectedId}>
          <div class="model-icon">{entry.task.toUpperCase()}</div>
          <div class="model-copy">
            <span>{taskLabels[entry.task] || entry.task}</span>
            <h3>{entry.display_name}</h3>
            <p>{selectedModelPath(entry)}</p>
          </div>
          <div class="model-actions">
            <small>VRAM ~{entry.min_vram_gb || '?'} GB</small>
            {#if packageChoices.length}
              <div class="package-buttons">
                {#each packageChoices as choice}
                  <div class="package-choice">
                    <button class="package-install"
                      class:preferred={choice.path === (selectedPackagePaths[entry.id] || entry.path)}
                      class:downloaded={packageSizes[choice.id]?.installed}
                      aria-pressed={choice.path === (selectedPackagePaths[entry.id] || entry.path)}
                      disabled={entryInstallBusy(entry, installJobs) ||
                        (packageSizeState === 'running' && Object.keys(packageSizes).length === 0)}
                      title={`${choice.format.toUpperCase()} ${choice.precision}: ${resolveCatalogPath(choice.path)}`}
                      on:click={() => useOrInstallPackage(entry, choice)}>
                      <span>{installButtonLabel(choice, installJobs[choice.id])}</span>
                      {#if packageSizeLabel(packageSizes[choice.id], packageSizeState,
                        choice.path === (selectedPackagePaths[entry.id] || entry.path))}
                        <span class="package-size">{packageSizeLabel(packageSizes[choice.id], packageSizeState,
                          choice.path === (selectedPackagePaths[entry.id] || entry.path))}</span>
                      {/if}
                    </button>
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
                  class="install-progress" title={installJob.message}>
                  <div class="install-progress-head">
                    <strong>{installJob.state}</strong>
                    <span>{installProgressLabel(installJob)}</span>
                  </div>
                  <div class:indeterminate={installJob.state === 'running' && installJob.progress_percent < 0}
                    class="install-progress-track" role="progressbar"
                    aria-label={`${entry.display_name} download progress`}
                    aria-valuemin="0" aria-valuemax="100" aria-valuenow={installPercent(installJob)}>
                    <span style={`width: ${installPercent(installJob)}%`}></span>
                  </div>
                  <div class="install-status">{installJob.message}</div>
                </div>
              {/if}
            {/if}
          </div>
        </article>
      {/each}
    </section>
  {:else}
    <section class="page-head"><p class="eyebrow">RUNTIME</p><h1>Session log</h1><p>Browser-side lifecycle and request events.</p></section>
    <section class="panel log-panel">
      <div class="runtime-cards">
        <div><span>Status</span><strong>{server?.status || 'offline'}</strong></div>
        <div><span>Backend</span><strong>{server?.backend || '—'}</strong></div>
        <div><span>Registered</span><strong>{loadedModels.length}</strong></div>
        <div><span>Resident</span><strong>{loadedModels.filter((model) => model.loaded).length}</strong></div>
      </div>
      <pre class="logs">{logs.length ? logs.join('\n') : 'No events yet.'}</pre>
    </section>
  {/if}
</main>

{#if folderBrowserOpen}
  <div class="folder-browser-backdrop">
    <div class="panel folder-browser-dialog" role="dialog" aria-modal="true" aria-labelledby="folder-browser-title">
      <header>
        <div><span>MODELS FOLDER</span><h2 id="folder-browser-title">Choose a folder</h2></div>
        <button aria-label="Close folder browser" title="Close" on:click={() => folderBrowserOpen = false}>Close</button>
      </header>
      <div class="folder-browser-location">{folderBrowser?.current || 'Loading...'}</div>
      {#if folderBrowser?.roots.length}
        <div class="folder-browser-roots">
          {#each folderBrowser.roots as root}
            <button on:click={() => openFolderBrowser(root)}>{root}</button>
          {/each}
        </div>
      {/if}
      <div class="folder-browser-toolbar">
        <button disabled={!folderBrowser?.parent || folderBrowserLoading}
          on:click={() => openFolderBrowser(folderBrowser?.parent || '')}>Up one level</button>
        <button disabled={folderBrowserLoading}
          on:click={() => openFolderBrowser(folderBrowser?.current || '')}>Refresh</button>
      </div>
      {#if folderBrowserError}
        <div class="folder-browser-error">{folderBrowserError}</div>
      {:else if folderBrowserLoading}
        <div class="folder-browser-empty">Loading folders...</div>
      {:else if folderBrowser?.directories.length}
        <div class="folder-browser-list">
          {#each folderBrowser.directories as directory}
            <button title={directory.path} on:click={() => openFolderBrowser(directory.path)}>
              <span class="folder-icon">&gt;</span><span>{directory.name}</span>
            </button>
          {/each}
        </div>
      {:else}
        <div class="folder-browser-empty">This folder has no subfolders.</div>
      {/if}
      <footer>
        <button on:click={() => folderBrowserOpen = false}>Cancel</button>
        <button class="primary" disabled={!folderBrowser || folderBrowserLoading}
          on:click={selectBrowsedFolder}>Select this folder</button>
      </footer>
    </div>
  </div>
{/if}

<footer><span>audio.cpp native WebUI</span><span>SvelteKit · embedded in audiocpp_server</span></footer>
