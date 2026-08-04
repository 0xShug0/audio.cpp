<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { browserDecodeToWav, concatenateAudioBlobs } from '$lib/audio';
  import {
    base64AudioUrl,
    health,
    installModelPackage,
    loadModel,
    modelInstallJobs,
    models,
    pathStatus,
    runTask,
    speech,
    transcription,
    unloadModel,
    uploadWav,
    type ModelInstallJob
  } from '$lib/api';
  import { catalog, parameterCatalog, taskLabels } from '$lib/catalog';
  import { defaultChunkBudget, splitTtsChunks } from '$lib/text';
  import type { AudioOutput, CatalogEntry, LoadedModel, ParamSpec, ServerHealth } from '$lib/types';
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
  let text = '';
  let language = '';
  let context = '';
  let referenceText = '';
  let instructions = '';
  let lyrics = '';
  let duration = 30;
  let seed = -1;
  let maxTokens = 1024;
  let sourceFile: File | null = null;
  let voiceFile: File | null = null;
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
  let installSourceDirectory = '';
  let installSourceFile = '';
  let installOutputFile = '';
  let installVariant = '';
  let installOverwrite = false;
  let installPoll: number | null = null;

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

  $: selected = catalog.find((entry) => entry.id === selectedId) || catalog[0];
  $: activeWorkflow = (workflowTabs.find((workflow) =>
    workflow.tasks.some((task) => task === selected?.task))?.id || 'tts') as WorkflowId;
  $: activeWorkflowSpec = workflowTabs.find((workflow) => workflow.id === activeWorkflow) || workflowTabs[0];
  $: workflowModels = catalog.filter((entry) =>
    activeWorkflowSpec.tasks.some((task) => task === entry.task));
  $: isLoaded = loadedModels.some((model) => model.id === selectedId && model.loaded);
  $: needsSource = ['asr', 'vc', 'svc', 's2s', 'sep', 'vad', 'diar', 'align'].includes(selected?.task);
  $: acceptsSource = needsSource || selected?.task === 'gen';
  $: needsVoice = ['clon', 'vc', 'svc'].includes(selected?.task) ||
    (selected?.task === 'tts' && !['supertonic'].includes(selected?.family));
  $: showsText = ['tts', 'clon', 'gen', 's2s', 'align', 'vdes'].includes(selected?.task);
  $: supportsLiveAsr = selected?.task === 'asr' &&
    ['voxtral_realtime', 'nemotron_asr', 'higgs_audio_stt'].includes(selected?.family);

  function log(message: string) {
    const line = `${new Date().toLocaleTimeString()}  ${message}`;
    logs = [line, ...logs].slice(0, 200);
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
    installed = null;
    try {
      installed = (await pathStatus(modelPath)).exists;
    } catch {
      installed = null;
    }
  }

  function chooseModel(id: string) {
    const next = catalog.find((entry) => entry.id === id);
    if (!next) return;
    selectedId = id;
    selected = next;
    modelPath = next.path;
    chunkBudget = defaultChunkBudget(next.family);
    localStorage.setItem('audiocpp.ui.model', id);
    resetParams();
    inspectPath();
  }

  function chooseWorkflow(id: WorkflowId) {
    const workflow = workflowTabs.find((entry) => entry.id === id);
    if (!workflow || workflow.tasks.some((task) => task === selected?.task)) return;
    const next = catalog.find((entry) => workflow.tasks.some((task) => task === entry.task));
    if (next) chooseModel(next.id);
  }

  async function doLoad(modeOverride?: string) {
    if (!server?.ui_management) {
      status = 'This server was not started with UI management enabled.';
      return;
    }
    loadingModel = true;
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
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      log(`Load failed: ${status}`);
    } finally {
      loadingModel = false;
    }
  }

  async function doUnload() {
    loadingModel = true;
    try {
      await unloadModel(selected.id);
      await refresh();
      status = `${selected.display_name} unloaded.`;
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    } finally {
      loadingModel = false;
    }
  }

  async function stagedPath(file: File | null): Promise<string | undefined> {
    if (!file) return undefined;
    const targetSampleRate = ['asr', 'vad', 'diar', 'align'].includes(selected.task) ? 16000 : undefined;
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
    clearOutput();
    running = true;
    aborter = new AbortController();
    const started = performance.now();
    status = `Running ${taskLabels[selected.task] || selected.task}…`;
    log(status);
    try {
      await ensureLoaded();
      const options = requestOptions();
      const audio = acceptsSource ? await stagedPath(sourceFile) : undefined;
      const voiceRef = needsVoice ? await stagedPath(voiceFile) : undefined;

      if (['tts', 'clon', 'vdes'].includes(selected.task)) {
        if (!text.trim()) throw new Error('Enter text to generate.');
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
            seed: seed < 0 ? seed : seed + index,
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
          chunks: chunks.length,
          characters: text.length,
          chunk_budget: chunkBudget,
          timings
        }, null, 2);
      } else if (selected.task === 'asr') {
        if (!audio) throw new Error('Choose an audio file.');
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
        if (needsSource && !audio) throw new Error('Choose a source audio file.');
        const request: Record<string, unknown> = {
          text,
          language,
          lyrics,
          duration_seconds: duration,
          seed,
          max_tokens: maxTokens,
          options
        };
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
      status = `Complete in ${elapsed}s.`;
      log(status);
    } catch (error) {
      if ((error as Error)?.name === 'AbortError') {
        status = 'Cancelled.';
      } else {
        status = error instanceof Error ? error.message : String(error);
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
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !running) {
      event.preventDefault();
      run();
    }
  }

  async function refreshInstallJobs() {
    if (!server?.ui_management) return;
    try {
      const jobs = await modelInstallJobs();
      installJobs = Object.fromEntries(jobs.map((job) => [job.id, job]));
      for (const entry of catalog) {
        const job = entry.download_id ? installJobs[entry.download_id] : undefined;
        if (job?.state === 'complete' && entry.id === selectedId) await inspectPath();
      }
      const active = jobs.some((job) => job.state === 'queued' || job.state === 'running');
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

  async function installPackage(entry: CatalogEntry) {
    if (!entry.download_id) return;
    const existing = installJobs[entry.download_id];
    if (existing?.state === 'queued' || existing?.state === 'running') return;
    status = `Starting installation for ${entry.display_name}...`;
    try {
      const job = await installModelPackage({
        id: entry.download_id,
        source_file: installSourceFile.trim() || undefined,
        output_file: installOutputFile.trim() || undefined,
        source_directory: installSourceDirectory.trim() || undefined,
        variant: installVariant.trim() || undefined,
        overwrite: installOverwrite
      });
      installJobs = { ...installJobs, [job.id]: job };
      await refreshInstallJobs();
      status = `${entry.display_name} installation is running in the background.`;
      log(status);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
      log(`Installer failed to start: ${status}`);
    }
  }

  onMount(async () => {
    const stored = localStorage.getItem('audiocpp.ui.model');
    if (stored && catalog.some((entry) => entry.id === stored)) selectedId = stored;
    selected = catalog.find((entry) => entry.id === selectedId) || catalog[0];
    modelPath = selected.path;
    resetParams();
    await refresh();
    await inspectPath();
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
    <button class:active={tab === 'models'} on:click={() => tab = 'models'}>Models</button>
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
        <h1>{taskLabels[selected?.task] || 'Audio studio'}</h1>
        <p>One native server, one embedded interface, no Python between your browser and the model.</p>
      </div>
      <div class="hero-stat">
        <span>Model</span>
        <strong>{selected?.display_name}</strong>
        <small class:ready={isLoaded}>{isLoaded ? 'Resident' : installed === false ? 'Not installed' : 'Available'}</small>
      </div>
    </section>

    <div class="studio-grid">
      <aside class="panel model-rail">
        <label for="model">Model</label>
        <select id="model" bind:value={selectedId} on:change={(event) => chooseModel(event.currentTarget.value)}>
          {#each activeWorkflowSpec.tasks as task}
            {@const entries = workflowModels.filter((entry) => entry.task === task)}
            {#if entries.length}
              <optgroup label={taskLabels[task]}>
                {#each entries as entry}<option value={entry.id}>{entry.display_name}</option>{/each}
              </optgroup>
            {/if}
          {/each}
        </select>

        <label for="path">Model path</label>
        <input id="path" bind:value={modelPath} on:change={inspectPath} />
        <div class="path-state">
          <span class:good={installed === true} class:bad={installed === false}>
            {installed === true ? 'Path found' : installed === false ? 'Path missing' : 'Path not inspected'}
          </span>
          <span>{selected?.min_vram_gb || '?'} GB estimated VRAM</span>
        </div>

        <div class="button-row">
          <button class="primary" disabled={loadingModel || isLoaded} on:click={() => doLoad()}>
            {loadingModel ? 'Loading…' : isLoaded ? 'Loaded' : 'Load model'}
          </button>
          <button disabled={loadingModel || !isLoaded} on:click={doUnload}>Unload</button>
        </div>

        {#if selected?.input_hint_en || selected?.input_hint}
          <div class="hint">{selected.input_hint_en || selected.input_hint}</div>
        {/if}
      </aside>

      <section class="panel controls">
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
          <div>
            <label for="language">Language <span>blank = auto</span></label>
            <input id="language" bind:value={language} placeholder="auto" />
          </div>
          <div>
            <label for="seed">Seed <span>-1 = random</span></label>
            <input id="seed" type="number" bind:value={seed} />
          </div>
          <div>
            <label for="tokens">Maximum tokens</label>
            <input id="tokens" type="number" min="1" bind:value={maxTokens} />
          </div>
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
          <label for="voice">Reference voice <span>{selected.task === 'tts' ? 'optional' : 'required'}</span></label>
          <input id="voice" class="file" type="file" accept="audio/*"
            on:change={(event) => voiceFile = event.currentTarget.files?.[0] || null} />
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
          <label for="reference">Reference transcript <span>recommended for cloning</span></label>
          <textarea id="reference" rows="2" bind:value={referenceText}></textarea>
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
          <button class="run" disabled={running} on:click={run}>
            <span>{running ? 'Working…' : 'Run'}</span>
            <kbd>Ctrl ↵</kbd>
          </button>
          <button disabled={!running} on:click={cancel}>Cancel</button>
          <div class="status" class:busy={running}>{status}</div>
        </div>
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
      <p>Download and prepare packages without leaving the native interface. Normal downloads use model_manager_v2; specialized converter inputs use the deprecated manager until those workflows migrate.</p>
    </section>
    <section class="panel installer-options">
      <div>
        <label for="install-source">Preparation source directory <span>optional, for local converter inputs</span></label>
        <input id="install-source" bind:value={installSourceDirectory} placeholder="Path to an existing checkpoint directory" />
      </div>
      <div>
        <label for="install-source-file">Source checkpoint <span>optional utility input</span></label>
        <input id="install-source-file" bind:value={installSourceFile} placeholder="Path to a source checkpoint" />
      </div>
      <div>
        <label for="install-output-file">Converted output <span>optional utility output</span></label>
        <input id="install-output-file" bind:value={installOutputFile} placeholder="Path to converted output" />
      </div>
      <div>
        <label for="install-variant">Variant <span>optional, e.g. htdemucs</span></label>
        <input id="install-variant" bind:value={installVariant} />
      </div>
      <label class="toggle overwrite-toggle">
        <input type="checkbox" bind:checked={installOverwrite} />
        <span></span>Overwrite an existing package
      </label>
    </section>
    <section class="model-grid">
      {#each catalog as entry}
        {@const installJob = entry.download_id ? installJobs[entry.download_id] : undefined}
        <article class:selected={entry.id === selectedId}>
          <div class="model-icon">{entry.task.toUpperCase()}</div>
          <div class="model-copy">
            <span>{taskLabels[entry.task] || entry.task}</span>
            <h3>{entry.display_name}</h3>
            <p>{entry.path}</p>
          </div>
          <div class="model-actions">
            <small>{entry.min_vram_gb || '?'} GB</small>
            <button on:click={() => { chooseModel(entry.id); tab = 'studio'; }}>Open</button>
            {#if entry.download_id}
              <button disabled={installJob?.state === 'queued' || installJob?.state === 'running'}
                on:click={() => installPackage(entry)}>
                {installJob?.state === 'running' ? 'Installing...' :
                  installJob?.state === 'queued' ? 'Queued' :
                  installJob?.state === 'complete' ? 'Install again' :
                  installJob?.state === 'failed' ? 'Retry install' : 'Install / prepare'}
              </button>
              {#if installJob}
                <div class:failed={installJob.state === 'failed'} class:complete={installJob.state === 'complete'}
                  class="install-status" title={installJob.message}>
                  <strong>{installJob.state}</strong> {installJob.message}
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

<footer><span>audio.cpp native WebUI</span><span>SvelteKit · embedded in audiocpp_server</span></footer>
