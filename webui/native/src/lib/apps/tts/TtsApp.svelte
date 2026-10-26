<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import {
    FileText,
    AudioLines,
    SlidersHorizontal,
    Plus,
    Play,
    RotateCcw,
    Scissors,
    ChevronRight,
    ChevronLeft,
    ChevronDown,
    Download,
    Check,
    Square,
    ArrowUp,
    ArrowDown,
    Merge,
    Trash2,
    FolderOpen,
    List
  } from '@lucide/svelte';
  import {
    listAppProjects,
    getAppProject,
    saveAppProject,
    deleteAppProject,
    listVoices,
    type SavedVoice
  } from '$lib/library';
  import { loadModelChoices } from '$lib/model-settings/choices';
  import type { ModelSettings } from '$lib/model-settings/types';
  import { ModelRuntime } from '$lib/model-settings/runtime';
  import { uploadWav } from '$lib/api';
  import { splitTtsChunks } from '$lib/text';
  import { modelCapabilities } from '$lib/catalog';
  import { assembleAudio, finishAudio, exportAudio } from './audio';
  import {
    newProject,
    newChunk,
    selectedTake,
    assignedSpeaker,
    ensureSpeakers,
    chunkConfiguration,
    chunkReference,
    speechSignature,
    chunkReady,
    assemblySignature,
    assemblyCurrent,
    finishSignature,
    type TtsProject,
    type Chunk,
    type Take,
    type VoiceAssignment
  } from './project';
  import ModelChoice from './ModelChoice.svelte';
  import SpeakerPanel from './SpeakerPanel.svelte';
  import PassageNavigator from './PassageNavigator.svelte';
  import BoundaryEditor from './BoundaryEditor.svelte';
  import WaveformEditor from './WaveformEditor.svelte';
  export let busy = false;
  export let active = true;
  let project: TtsProject | null = null;
  let projects: Array<{ id: string; title: string; updatedAt: number }> = [];
  let choices: ModelSettings[] = [];
  let voices: SavedVoice[] = [];
  let stage: 'script' | 'speech' | 'assemble' | 'finish' = 'script';
  let selectedId = '';
  let navigatorPage = 0;
  const navigatorPageSize = 40;
  let jumpNumber = 1;
  let assemblyPage = 0;
  let assemblySearch = '';
  let assemblyOnlyAttention = false;
  let archiveOpen = false;
  let archivePage = 0;
  $: archivePages = Math.max(1, Math.ceil((project?.archivedChunks.length || 0) / navigatorPageSize));
  $: if (archivePage >= archivePages) archivePage = archivePages - 1;
  let checked = new Set<string>();
  let filter = 'all';
  let search = '';
  let preview: string[] | null = null;
  let takingId = '';
  let audioEditorOpen = false;
  let comparisonOpen = false;
  let comparedTakeId = '';
  let previewIndex = 0;
  let mobilePane: 'passages' | 'editor' | 'settings' = 'editor';
  let workspaceTop = 120;
  let speakerPanel: SpeakerPanel | undefined;
  $: currentTake = selectedChunk && selectedTake(selectedChunk);
  $: comparedTake = selectedChunk?.takes.find((take) => take.id === comparedTakeId) || selectedChunk?.takes.at(-1);
  $: if (takingId && selectedChunk?.id === takingId && !busy) {
    comparedTakeId = selectedChunk.takes.at(-1)?.id || '';
    comparisonOpen = true;
    takingId = '';
  }
  function openDialog(node: HTMLDialogElement) {
    node.showModal();
    return {
      destroy() {
        node.close();
      }
    };
  }
  function measureWorkspace(node: HTMLElement) {
    const update = () => {
      workspaceTop = Math.max(0, Math.round(node.getBoundingClientRect().top + window.scrollY));
    };
    const observer = new ResizeObserver(update);
    const topbar = document.querySelector('.topbar');
    if (topbar) observer.observe(topbar);
    window.addEventListener('resize', update);
    update();
    return {
      destroy() {
        observer.disconnect();
        window.removeEventListener('resize', update);
      }
    };
  }
  function addPassage() {
    if (!project || busy) return;
    const chunk = newChunk('');
    project.chunks = [...project.chunks, chunk];
    selectChunk(chunk.id);
    touch();
    goToStage('speech');
  }
  function openSpeakers() {
    mobilePane = 'settings';
    void tick().then(() => speakerPanel?.addSpeaker());
  }
  function openComparison() {
    if (!selectedChunk || !project) return;
    comparedTakeId =
      [...selectedChunk.takes].reverse().find((take) => take.signature === speechSignature(project!, selectedChunk!))
        ?.id ||
      selectedChunk.takes.at(-1)?.id ||
      '';
    comparisonOpen = true;
  }

  let workingId = '';
  let status = '';
  let error = '';
  let loading = true;
  let wasActive = active;
  $: if (active !== wasActive) {
    wasActive = active;
    if (active && !loading) refreshChoices();
  }
  let dirty = false;
  let saving = false;
  let saveTimer: ReturnType<typeof setTimeout> | undefined;
  let savePromise: Promise<void> | null = null;
  let aborter: AbortController | null = null;
  let exportLink = '';
  let exportName = '';
  let exportedSignature = '';
  let editors: Record<string, HTMLTextAreaElement> = {};
  let urls = new Map<Blob, string>();
  let player: HTMLAudioElement | undefined;
  $: if (!active) {
    player?.pause();
    if (typeof document !== 'undefined')
      document.querySelectorAll<HTMLAudioElement>('.tts-app audio').forEach((a) => a.pause());
  }
  $: ready = project ? project.chunks.filter((c) => chunkReady(project!, c)).length : 0;
  $: needsGeneration =
    project?.chunks.filter((c) => !c.takes.some((t) => t.signature === speechSignature(project!, c))) || [];
  $: needsSelection =
    project?.chunks.filter(
      (c) => !chunkReady(project!, c) && c.takes.some((t) => t.signature === speechSignature(project!, c))
    ) || [];
  $: visible =
    project?.chunks.filter(
      (c) =>
        c.text.toLowerCase().includes(search.toLowerCase()) &&
        (filter === 'all' ||
          (filter === 'missing' && !selectedTake(c)) ||
          (filter === 'changed' && !!selectedTake(c) && !chunkReady(project!, c)))
    ) || [];
  $: selectedChunk = project?.chunks.find((c) => c.id === selectedId);
  $: selectedIndex = project?.chunks.findIndex((c) => c.id === selectedId) ?? -1;
  $: if (selectedIndex >= 0) jumpNumber = selectedIndex + 1;
  $: navigatorPages = Math.max(1, Math.ceil(visible.length / navigatorPageSize));
  $: if (navigatorPage >= navigatorPages) navigatorPage = navigatorPages - 1;
  $: navigatorRows = visible.slice(navigatorPage * navigatorPageSize, (navigatorPage + 1) * navigatorPageSize);
  $: assemblyMatches =
    project?.chunks.filter(
      (c) =>
        c.text.toLowerCase().includes(assemblySearch.toLowerCase()) &&
        (!assemblyOnlyAttention || !chunkReady(project!, c))
    ) || [];
  $: assemblyPages = Math.max(1, Math.ceil(assemblyMatches.length / navigatorPageSize));
  $: if (assemblyPage >= assemblyPages) assemblyPage = assemblyPages - 1;
  $: assemblyRows = assemblyMatches.slice(assemblyPage * navigatorPageSize, (assemblyPage + 1) * navigatorPageSize);
  function selectChunk(id: string) {
    if (!project) return;
    document.querySelectorAll<HTMLAudioElement>('.tts-app audio').forEach((a) => a.pause());
    selectedId = id;
    takingId = '';
    audioEditorOpen = false;
    comparisonOpen = false;
    mobilePane = 'editor';
    localStorage.setItem(`audiocpp.tts.chunk:${project.id}`, id);
    const at = (search || filter !== 'all' ? visible : project.chunks).findIndex((c) => c.id === id);
    if (at >= 0) navigatorPage = Math.floor(at / navigatorPageSize);
    status = '';
  }
  function navigateAssembly(index: number) {
    if (project?.chunks[index]) selectChunk(project.chunks[index].id);
  }
  function jumpToChunk() {
    if (!project) return;
    if (!Number.isInteger(jumpNumber) || jumpNumber < 1 || jumpNumber > project.chunks.length) {
      error = `Enter a chunk number between 1 and ${project.chunks.length}.`;
      return;
    }
    if (stage === 'assemble') {
      navigateAssembly(jumpNumber - 1);
      error = '';
      return;
    }
    search = '';
    filter = 'all';
    navigatorPage = Math.floor((jumpNumber - 1) / navigatorPageSize);
    selectChunk(project.chunks[jumpNumber - 1].id);
    error = '';
  }
  function nextAttention() {
    if (!project) return;
    const ordered = [...project.chunks.slice(selectedIndex + 1), ...project.chunks.slice(0, selectedIndex + 1)];
    const next = ordered.find((c) => !chunkReady(project!, c));
    if (next && stage === 'assemble') {
      navigateAssembly(project.chunks.indexOf(next));
      return;
    }
    if (next) {
      search = '';
      filter = 'all';
      selectChunk(next.id);
      navigatorPage = Math.floor(project.chunks.indexOf(next) / navigatorPageSize);
    }
  }
  $: finalCurrent =
    !!project?.finished && assemblyCurrent(project) && project.finished.signature === finishSignature(project);
  $: assemblyIsCurrent = project ? assemblyCurrent(project) : false;
  type Stage = 'script' | 'speech' | 'assemble' | 'finish';
  const steps: Array<{ id: Stage; label: string }> = [
    { id: 'script', label: 'Script' },
    { id: 'speech', label: 'Speakers' },
    { id: 'assemble', label: 'Arrange' },
    { id: 'finish', label: 'Export' }
  ];
  function resumeStage(value: TtsProject): Stage {
    const saved = localStorage.getItem(`audiocpp.tts.stage:${value.id}`) as Stage | null;
    if (saved === 'finish' && assemblyCurrent(value)) return saved;
    if (saved === 'assemble' && value.chunks.length && value.chunks.every((c) => chunkReady(value, c))) return saved;
    if (saved === 'script' || (saved === 'speech' && value.chunks.length)) return saved;
    return value.chunks.length ? 'speech' : 'script';
  }
  $: if (project) localStorage.setItem(`audiocpp.tts.stage:${project.id}`, stage);
  function revealStage() {
    mobilePane = 'editor';
    void tick().then(() => {
      document.querySelector<HTMLElement>('.stage-title')?.focus({ preventScroll: true });
      window.scrollTo({ top: 0 });
    });
  }
  function goToStage(next: Stage) {
    if (busy) return;
    stage = next;
    status = '';
    if (next === 'assemble' && selectedIndex >= 0) navigateAssembly(selectedIndex);
    revealStage();
  }
  $: generationConfigured =
    !!project && needsGeneration.every((c) => !!chunkConfiguration(project!, c) && !!c.text.trim());
  $: primaryDisabled =
    busy ||
    (stage === 'script'
      ? !project?.source.trim()
      : stage === 'speech'
        ? !project?.chunks.length || needsGeneration.some((c) => !c.text.trim())
        : stage === 'assemble'
          ? !project?.chunks.length || ready !== project.chunks.length
          : !assemblyIsCurrent);
  $: primaryLabel =
    stage === 'script'
      ? preview?.length
        ? 'Continue to speakers'
        : 'Review passages'
      : stage === 'speech'
        ? needsGeneration.length
          ? !generationConfigured && !needsGeneration.some((c) => !c.text.trim())
            ? 'Add speaker'
            : `Generate narration${needsGeneration.length > 1 ? ` (${needsGeneration.length})` : ''}`
          : needsSelection.length
            ? 'Review updated takes'
            : 'Continue to arrange'
        : stage === 'assemble'
          ? assemblyIsCurrent
            ? 'Continue to export'
            : 'Combine narration'
          : 'Download';
  $: nextTitle =
    stage === 'script'
      ? preview?.length
        ? `${preview.length} passages ready to voice`
        : 'Your words come first'
      : stage === 'speech'
        ? needsGeneration.some((c) => !c.text.trim())
          ? 'Add text to your empty passages'
          : !generationConfigured && needsGeneration.length
            ? 'Create a speaker to continue'
            : needsGeneration.length
              ? `${needsGeneration.length} passages need audio`
              : needsSelection.length
                ? `${needsSelection.length} updated takes need your choice`
                : 'Every passage is ready'
        : stage === 'assemble'
          ? ready !== project?.chunks.length
            ? 'Review the passages needing audio'
            : assemblyIsCurrent
              ? 'Your narration is combined'
              : 'Bring your passages together'
          : 'Save your finished narration';
  $: nextHint =
    stage === 'script'
      ? preview?.length
        ? 'Next, create a speaker for these passages.'
        : 'Paste text or import a TXT or Markdown file.'
      : stage === 'speech'
        ? needsGeneration.some((c) => !c.text.trim())
          ? 'Write a passage or remove it using Edit passage.'
          : !generationConfigured && needsGeneration.length
            ? 'A speaker keeps the model and voice together.'
            : needsGeneration.length
              ? 'Generate the missing audio, then listen and refine.'
              : needsSelection.length
                ? 'Compare the recordings and choose the take to keep.'
                : 'Next, adjust pauses and combine the audio.'
        : stage === 'assemble'
          ? assemblyIsCurrent
            ? 'Continue to choose a format and download.'
            : 'Chosen takes will be combined in the order above.'
          : 'Your processing choices are applied automatically.';
  function primaryAction() {
    if (primaryDisabled || !project) return;
    if (stage === 'script') {
      if (preview?.length) applySplit();
      else previewSplit();
    } else if (stage === 'speech') {
      if (needsGeneration.length && !generationConfigured) openSpeakers();
      else if (needsGeneration.length) void generate(needsGeneration.map((c) => c.id));
      else if (needsSelection.length) {
        selectChunk(needsSelection[0].id);
        openComparison();
      } else goToStage('assemble');
    } else if (stage === 'assemble') {
      if (assemblyIsCurrent) goToStage('finish');
      else void assemble();
    } else void downloadNarration();
  }
  $: enhancementFamilies = Object.keys(modelCapabilities).filter((family) =>
    modelCapabilities[family].includes('audio_enhancement')
  );
  function urlFor(audio: Blob) {
    let url = urls.get(audio);
    if (!url) {
      url = URL.createObjectURL(audio);
      urls.set(audio, url);
    }
    return url;
  }
  function release(audio: Blob | undefined) {
    if (!audio) return;
    const url = urls.get(audio);
    if (url) {
      URL.revokeObjectURL(url);
      urls.delete(audio);
    }
  }
  function clearUrls() {
    for (const url of urls.values()) URL.revokeObjectURL(url);
    urls.clear();
  }
  function touch() {
    if (!project) return;
    project = {
      ...project,
      updatedAt: Math.max(Date.now(), project.updatedAt + 1)
    };
    dirty = true;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      flush().catch((e) => (error = `Could not save project: ${String(e)}`));
    }, 450);
  }
  export async function saveBeforeClose() {
    await flush();
  }
  async function flush() {
    clearTimeout(saveTimer);
    if (savePromise) await savePromise;
    if (!project || !dirty) return;
    const snapshot = structuredClone(project);
    saving = true;
    savePromise = saveAppProject({
      id: snapshot.id,
      appId: 'tts',
      title: snapshot.title,
      updatedAt: snapshot.updatedAt,
      data: snapshot
    })
      .then(() => {
        if (project?.id === snapshot.id && project.updatedAt === snapshot.updatedAt) dirty = false;
        projects = [
          {
            id: snapshot.id,
            title: snapshot.title,
            updatedAt: snapshot.updatedAt
          },
          ...projects.filter((p) => p.id !== snapshot.id)
        ].sort((a, b) => b.updatedAt - a.updatedAt);
      })
      .finally(() => {
        saving = false;
        savePromise = null;
      });
    await savePromise;
    if (dirty) await flush();
  }
  async function selectProject(id: string) {
    if (busy) return;
    try {
      await flush();
      const record = await getAppProject(id);
      if (!record || ![1, 2].includes((record.data as TtsProject).version))
        throw new Error('Unsupported or missing TTS project.');
      clearUrls();
      project = record.data as TtsProject;
      const migrated = ensureSpeakers(project);
      if (migrated) touch();
      localStorage.setItem('audiocpp.tts.project', project.id);
      stage = resumeStage(project);
      selectedId =
        project.chunks.find((c) => c.id === localStorage.getItem(`audiocpp.tts.chunk:${project!.id}`))?.id ||
        project.chunks[0]?.id ||
        '';
      search = '';
      filter = 'all';
      navigatorPage = Math.max(0, Math.floor(project.chunks.findIndex((c) => c.id === selectedId) / navigatorPageSize));
      assemblyPage = 0;
      assemblySearch = '';
      assemblyOnlyAttention = false;
      archivePage = 0;
      preview = null;
      checked = new Set();
      error = '';
      status = '';
      exportLink = '';
      document.querySelector<HTMLDetailsElement>('.project-menu')?.removeAttribute('open');
      revealStage();
    } catch (e) {
      error = String(e);
    }
  }
  async function createProject() {
    if (busy) return;
    try {
      await flush();
      clearUrls();
      project = newProject();
      localStorage.setItem('audiocpp.tts.project', project.id);
      stage = 'script';
      selectedId = '';
      search = '';
      filter = 'all';
      navigatorPage = 0;
      assemblyPage = 0;
      assemblySearch = '';
      assemblyOnlyAttention = false;
      archivePage = 0;
      checked = new Set();
      preview = null;
      error = '';
      status = '';
      exportLink = '';
      touch();
      document.querySelector<HTMLDetailsElement>('.project-menu')?.removeAttribute('open');
      revealStage();
    } catch (e) {
      error = String(e);
    }
  }
  async function removeProject() {
    if (!project || busy || !confirm(`Delete project "${project.title}" and its audio?`)) return;
    try {
      await flush();
      const id = project.id;
      await deleteAppProject(id);
      projects = projects.filter((p) => p.id !== id);
      clearUrls();
      project = null;
      dirty = false;
      if (projects.length) await selectProject(projects[0].id);
      else await createProject();
    } catch (e) {
      error = String(e);
    }
  }
  async function refreshChoices() {
    try {
      [choices, voices] = await Promise.all([loadModelChoices(), listVoices()]);
      error = '';
    } catch (e) {
      error = `Could not refresh model choices: ${String(e)}`;
    }
  }
  function previewSplit() {
    if (!project) return;
    error = '';
    if (!project.source.trim()) {
      error = 'Paste text or import a TXT/Markdown file first.';
      return;
    }
    if (!Number.isInteger(project.split.characters) || project.split.characters < 40) {
      error = 'Chunk size must be at least 40 characters.';
      return;
    }
    previewIndex = 0;
    const text = project.source.trim();
    preview =
      project.split.method === 'sentence'
        ? Array.from(new Intl.Segmenter(undefined, { granularity: 'sentence' }).segment(text), (s) => s.segment.trim())
            .filter(Boolean)
            .flatMap((s) => splitTtsChunks(s, project!.split.characters))
        : project.split.method === 'paragraph'
          ? text
              .split(/\n\s*\n/)
              .filter((s) => s.trim())
              .flatMap((s) => splitTtsChunks(s, project!.split.characters))
          : splitTtsChunks(text, project.split.characters);
  }
  function applySplit() {
    if (!project || !preview?.length) return;
    if (
      project.chunks.length &&
      !confirm('Replace the current chunks? Existing chunks and takes will remain in the project archive.')
    )
      return;
    project.archivedChunks.push(...project.chunks);
    project.chunks = preview.map((text) => newChunk(text));
    selectedId = project.chunks[0].id;
    preview = null;
    checked = new Set();
    search = '';
    filter = 'all';
    navigatorPage = 0;
    assemblyPage = 0;
    assemblySearch = '';
    assemblyOnlyAttention = false;
    archivePage = 0;
    touch();
    goToStage('speech');
  }
  async function importText(file: File | undefined) {
    if (!file || !project) return;
    error = '';
    if (!/\.(txt|md|markdown)$/i.test(file.name)) {
      error = 'Choose a TXT or Markdown file.';
      return;
    }
    try {
      project.source = await file.text();
      if (project.title === 'Untitled narration') project.title = file.name.replace(/\.[^.]+$/, '');
      preview = null;
      touch();
    } catch (e) {
      error = String(e);
    }
  }
  function splitChunk(chunk: Chunk) {
    if (!project || busy) return;
    const at = editors[chunk.id]?.selectionStart;
    if (at === undefined || !chunk.text.slice(0, at).trim() || !chunk.text.slice(at).trim()) {
      error = 'Place the text cursor where you want to split this chunk.';
      return;
    }
    const index = project.chunks.indexOf(chunk);
    project.archivedChunks.push(structuredClone(chunk));
    const parts = [
      newChunk(chunk.text.slice(0, at).trim(), chunk.override),
      newChunk(chunk.text.slice(at).trim(), chunk.override)
    ];
    for (const part of parts) {
      part.voice = structuredClone(chunk.voice);
      part.speakerId = chunk.speakerId;
    }
    project.chunks.splice(index, 1, ...parts);
    selectChunk(parts[0].id);
    error = '';
    touch();
  }
  function mergeChunk(chunk: Chunk) {
    if (!project || busy) return;
    const index = project.chunks.indexOf(chunk);
    const next = project.chunks[index + 1];
    if (!next) return;
    if (
      JSON.stringify([chunkConfiguration(project, chunk), chunkReference(project, chunk)?.id]) !==
        JSON.stringify([chunkConfiguration(project, next), chunkReference(project, next)?.id]) &&
      !confirm('These passages use different speakers. Joining them uses the first passage’s speaker. Continue?')
    )
      return;
    project.archivedChunks.push(structuredClone(chunk), structuredClone(next));
    const merged = newChunk(`${chunk.text}\n${next.text}`, chunk.override);
    merged.voice = structuredClone(chunk.voice);
    merged.speakerId = chunk.speakerId;
    project.chunks.splice(index, 2, merged);
    selectChunk(merged.id);
    touch();
  }
  function moveChunk(chunk: Chunk, offset: number) {
    if (!project || busy) return;
    const i = project.chunks.indexOf(chunk);
    const j = i + offset;
    if (j < 0 || j >= project.chunks.length) return;
    [project.chunks[i], project.chunks[j]] = [project.chunks[j], project.chunks[i]];
    touch();
  }
  function archiveChunk(chunk: Chunk) {
    if (!project || busy) return;
    project.archivedChunks.push(chunk);
    const index = project.chunks.indexOf(chunk);
    project.chunks = project.chunks.filter((c) => c.id !== chunk.id);
    checked = new Set([...checked].filter((id) => id !== chunk.id));
    if (selectedId === chunk.id) selectChunk(project.chunks[Math.min(index, project.chunks.length - 1)]?.id || '');
    touch();
  }
  function useUpdatedTakes() {
    if (!project) return;
    for (const chunk of needsSelection)
      chunk.takeId = [...chunk.takes].reverse().find((t) => t.signature === speechSignature(project!, chunk))!.id;
    touch();
  }
  function useTake(chunk: Chunk, take: Take) {
    chunk.takeId = take.id;
    touch();
  }
  async function generate(ids: string[]) {
    if (!project || busy || !ids.length) return;
    const jobs = project.chunks
      .filter((c) => ids.includes(c.id))
      .map((c) => ({
        id: c.id,
        text: c.text,
        configuration: structuredClone(chunkConfiguration(project!, c)),
        reference: structuredClone(chunkReference(project!, c)),
        signature: speechSignature(project!, c),
        sequence: project!.chunks.indexOf(c) + c.takes.length
      }));
    if (jobs.some((job) => !job.configuration)) {
      error = 'Add a speaker and assign it to these passages.';
      return;
    }
    if (jobs.some((job) => !job.text.trim())) {
      error = 'Remove empty chunks or enter text before generation.';
      return;
    }
    if (
      jobs.some(
        (job) =>
          job.reference &&
          job.configuration!.model.task !== 'clon' &&
          !modelCapabilities[job.configuration!.model.family]?.includes('speaker_reference')
      )
    ) {
      error =
        'A selected speech model does not support a reference recording. Remove the reference or choose a voice-cloning model.';
      return;
    }
    busy = true;
    error = '';
    aborter = new AbortController();
    const signal = aborter.signal;
    let runtime: ModelRuntime | undefined;
    let key = '';
    const references = new Map<string, string>();
    try {
      for (let i = 0; i < jobs.length; i++) {
        signal.throwIfAborted();
        const job = jobs[i];
        workingId = job.id;
        status = `Generating passage ${i + 1} of ${jobs.length}…`;
        const nextKey = JSON.stringify([job.configuration!.model, job.configuration!.request]);
        if (!runtime || key !== nextKey) {
          runtime = new ModelRuntime(job.configuration!, signal);
          key = nextKey;
        }
        const reference = job.reference;
        if (reference && !references.has(reference.id))
          references.set(reference.id, await uploadWav(reference.audio, signal));
        const voice_ref = reference ? references.get(reference.id) : undefined;
        const result = await runtime.run(
          { text: job.text, voice_ref, reference_text: reference?.transcript },
          job.sequence
        );
        if (result.kind !== 'audio') throw new Error('The speech model did not return audio.');
        signal.throwIfAborted();
        const context = new AudioContext();
        let duration: number;
        try {
          duration = (await context.decodeAudioData(await result.blob.arrayBuffer())).duration;
        } finally {
          await context.close();
        }
        const chunk = project.chunks.find((c) => c.id === job.id)!;
        const take: Take = {
          id: crypto.randomUUID(),
          text: job.text,
          configuration: job.configuration!,
          reference: reference || undefined,
          signature: job.signature,
          seed: (runtime.resolvedSeed + job.sequence) % 0x100000000,
          audio: result.blob,
          duration,
          createdAt: Date.now()
        };
        chunk.takes = [...chunk.takes, take];
        if (!chunk.takeId) chunk.takeId = take.id;
        if (chunk.id === selectedId && chunk.takes.length > 1) takingId = chunk.id;
        touch();
      }
      status = 'Audio generated. The first take is selected automatically; use Compare takes to choose a replacement.';
    } catch (e) {
      status = signal.aborted
        ? 'Cancelled. Completed takes are saved.'
        : 'Generation stopped. Completed takes are saved.';
      if (!signal.aborted) error = String(e);
    } finally {
      busy = false;
      workingId = '';
      aborter = null;
      await flush().catch((e) => (error = `Could not save takes: ${String(e)}`));
    }
  }
  async function assemble() {
    if (!project || busy) return;
    if (!project.chunks.length || project.chunks.some((c) => !chunkReady(project!, c))) {
      error = 'Choose an up-to-date take for every passage before assembly.';
      return;
    }
    busy = true;
    error = '';
    aborter = new AbortController();
    status = 'Assembling chosen takes…';
    try {
      const audio = await assembleAudio(
        project.chunks.map((c) => ({
          audio: selectedTake(c)!.rendered?.audio || selectedTake(c)!.audio,
          pause: c.pauseAfter ?? project!.assembly.pause
        })),
        project.assembly.sampleRate,
        aborter.signal
      );
      aborter.signal.throwIfAborted();
      release(project.assembled?.audio);
      project.assembled = { audio, signature: assemblySignature(project) };
      touch();
      status = 'Narration assembled. Listen across chunk boundaries before finishing.';
    } catch (e) {
      error = aborter.signal.aborted ? 'Cancelled.' : String(e);
    } finally {
      busy = false;
      aborter = null;
    }
  }
  async function renderFinish(): Promise<boolean> {
    if (!project || !assemblyCurrent(project) || !project.assembled || busy) {
      error = 'Combine the chosen chunk audio in step 3 first.';
      return false;
    }
    busy = true;
    error = '';
    aborter = new AbortController();
    status = 'Processing final audio…';
    const signal = aborter.signal;
    try {
      let audio = project.assembled.audio;
      if (project.enhancement) {
        status = `Processing with ${project.enhancement.name}…`;
        const runtime = new ModelRuntime(project.enhancement, signal);
        await runtime.prepare();
        const path = await uploadWav(audio, signal);
        const result = await runtime.run({ audio: path });
        if (result.kind !== 'audio') throw new Error('The processing model returned no audio.');
        audio = result.blob;
      }
      audio = await finishAudio(audio, project.finish, project.assembly.sampleRate, signal);
      signal.throwIfAborted();
      release(project.finished?.audio);
      project.finished = { audio, signature: finishSignature(project) };
      exportLink = '';
      touch();
      status = 'Processing preview ready. Listen before and after, or download your narration.';
      return true;
    } catch (e) {
      if (signal.aborted) status = 'Processing cancelled.';
      else error = String(e);
      return false;
    } finally {
      busy = false;
      aborter = null;
    }
  }
  async function prepareExport(): Promise<boolean> {
    if (!project?.finished || !finalCurrent || busy) return false;
    busy = true;
    error = '';
    aborter = new AbortController();
    status = 'Preparing download…';
    try {
      const audio = await exportAudio(project.finished.audio, project.finish.format, aborter.signal);
      aborter.signal.throwIfAborted();
      exportLink = urlFor(audio);
      exportName = `${project.title.replace(/[\\/:*?"<>|]/g, '_') || 'narration'}.${project.finish.format}`;
      exportedSignature = finishSignature(project) + project.finish.format;
      status = 'Download ready.';
      return true;
    } catch (e) {
      if (aborter.signal.aborted) status = 'Export cancelled.';
      else error = String(e);
      return false;
    } finally {
      busy = false;
      aborter = null;
    }
  }
  async function downloadNarration() {
    if (!project || busy) return;
    if (!finalCurrent && !(await renderFinish())) return;
    if (!(await prepareExport())) return;
    const link = document.createElement('a');
    link.href = exportLink;
    link.download = exportName;
    document.body.append(link);
    link.click();
    link.remove();
  }
  function playback(event: Event) {
    const target = event.currentTarget;
    document.querySelectorAll<HTMLAudioElement>('.tts-app audio').forEach((a) => {
      if (a !== target) a.pause();
    });
  }
  onMount(async () => {
    try {
      const [records, available, savedVoices] = await Promise.all([
        listAppProjects('tts'),
        loadModelChoices().catch((e) => {
          error = `Models are unavailable. You can still prepare your script. ${String(e)}`;
          return [];
        }),
        listVoices()
      ]);
      choices = available;
      voices = savedVoices;
      projects = records.map(({ id, title, updatedAt }) => ({
        id,
        title,
        updatedAt
      }));
      if (records.length) {
        const chosen = records.find((r) => r.id === localStorage.getItem('audiocpp.tts.project')) || records[0];
        const data = chosen.data as TtsProject;
        if (![1, 2].includes(data.version)) throw new Error('Unsupported TTS project version.');
        project = data;
        if (ensureSpeakers(project)) touch();
        stage = resumeStage(project);
        selectedId =
          project.chunks.find((c) => c.id === localStorage.getItem(`audiocpp.tts.chunk:${project!.id}`))?.id ||
          project.chunks[0]?.id ||
          '';
        search = '';
        filter = 'all';
        navigatorPage = Math.max(
          0,
          Math.floor(project.chunks.findIndex((c) => c.id === selectedId) / navigatorPageSize)
        );
        assemblyPage = 0;
        assemblySearch = '';
        assemblyOnlyAttention = false;
        archivePage = 0;
      } else {
        project = newProject();
        localStorage.setItem('audiocpp.tts.project', project.id);
        touch();
      }
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  });
  onDestroy(() => {
    aborter?.abort();
    clearTimeout(saveTimer);
    clearUrls();
  });
</script>

<svelte:window
  on:beforeunload={(event) => {
    if (dirty || saving || busy) {
      event.preventDefault();
      event.returnValue = '';
    }
  }}
/>
<section
  class="tts-app"
  data-active={active}
  use:measureWorkspace
  style:--tts-top={`${workspaceTop}px`}
  aria-label="Narration workspace"
>
  <header class="project-bar">
    <div class="project-heading">
      <span class="eyebrow">TEXT TO SPEECH</span>{#if project}<input
          class="project-title"
          aria-label="Project title"
          bind:value={project.title}
          on:input={touch}
          disabled={busy}
        />{:else}<h2>Narration</h2>{/if}
    </div>
    {#if project}<nav class="pipeline" aria-label="Narration steps">
        {#each steps as step, i}{@const complete =
            i === 0
              ? project.chunks.length > 0
              : i === 1
                ? ready > 0 && ready === project.chunks.length
                : i === 2
                  ? assemblyIsCurrent
                  : !!exportLink && finalCurrent}<button
            class:active={stage === step.id}
            aria-current={stage === step.id ? 'step' : undefined}
            disabled={busy ||
              (i > 0 && !project.chunks.length) ||
              (i > 1 && ready !== project.chunks.length) ||
              (i > 2 && !assemblyIsCurrent)}
            on:click={() => goToStage(step.id)}
            ><span class="step-number"
              >{#if complete && stage !== step.id}<Check size={12} />{:else}{i + 1}{/if}</span
            >{step.label}</button
          >{/each}
      </nav>{/if}
    <span class="save-status">{loading ? 'Opening…' : saving || dirty ? 'Saving…' : 'Saved'}</span>
    <details class="project-menu">
      <summary aria-label="Projects"><FolderOpen size={16} /><ChevronDown size={12} /></summary>
      <div class="project-menu-content">
        <label
          >Open project<select
            aria-label="Open project"
            value={project?.id || ''}
            disabled={busy || loading}
            on:change={(e) => selectProject(e.currentTarget.value)}
            >{#if project && !projects.some((p) => p.id === project!.id)}<option value={project.id}
                >{project.title}</option
              >{/if}{#each projects as p}<option value={p.id}>{p.title}</option>{/each}</select
          ></label
        ><button disabled={busy || loading} on:click={createProject}><Plus size={14} />New project</button><button
          class="danger"
          disabled={busy || loading || !project}
          on:click={removeProject}><Trash2 size={14} />Delete this project</button
        ><small
          >Projects save automatically in this browser.</small
        >
      </div>
    </details>
  </header>
  {#if loading}<p class="notice" role="status">Opening your projects…</p>{/if}
  {#if error}<div class="error" role="alert">
      <span>{error}</span><button aria-label="Dismiss error" on:click={() => (error = '')}>×</button>
    </div>{/if}
  {#if project}
    <div class="stage-toolbar">
      <div>
        <h2 class="stage-title" tabindex="-1">
          {stage === 'script'
            ? preview
              ? 'Review your passages'
              : 'Write your script'
            : stage === 'speech'
              ? 'Speakers & recordings'
              : stage === 'assemble'
                ? 'Arrange your narration'
                : 'Export your narration'}
        </h2>
        <p>
          {stage === 'script'
            ? preview
              ? 'Choose a passage on the left to check the split.'
              : 'Start with your words. Review the passages when you’re ready.'
            : stage === 'speech'
              ? 'Choose a passage on the left. Assign its speaker on the right.'
              : stage === 'assemble'
                ? 'Choose a join on the left, then listen and adjust the pause.'
                : 'Listen once more and download your finished audio.'}
        </p>
      </div>
      {#if stage !== 'script'}<span class="project-progress">{ready}/{project.chunks.length} ready</span>{/if}
    </div>

    <nav class="pane-tabs" aria-label="Workspace panes">
      <button aria-pressed={mobilePane === 'passages'} on:click={() => (mobilePane = 'passages')}
        ><List size={14} />Passages</button
      ><button aria-pressed={mobilePane === 'editor'} on:click={() => (mobilePane = 'editor')}
        ><FileText size={14} />Editor</button
      ><button aria-pressed={mobilePane === 'settings'} on:click={() => (mobilePane = 'settings')}
        ><SlidersHorizontal size={14} />{stage === 'speech' ? 'Speakers' : 'Settings'}</button
      >
    </nav>
    <div class="tts-workbench">
      <aside class="passage-rail" class:mobile-visible={mobilePane === 'passages'} aria-label="Passage navigation">
        <PassageNavigator
          {project}
          {selectedId}
          disabled={busy}
          preview={stage === 'script' ? preview : null}
          {previewIndex}
          bind:checked
          onselect={(id) => {
            selectChunk(id);
            if (stage === 'script') goToStage('speech');
          }}
          onpreview={(index) => {
            previewIndex = index;
            mobilePane = 'editor';
          }}
          onadd={addPassage}
          ongenerate={generate}
        />{#if project.archivedChunks.length}<button class="archive-trigger" on:click={() => (archiveOpen = true)}
            >Archive · {project.archivedChunks.length}</button
          >{/if}
      </aside>
      <section class="editor-pane" class:mobile-visible={mobilePane === 'editor'} aria-label="Passage editor">
        {#if stage === 'script'}
          <div class="editor-heading">
            <h3>{preview ? `Passage ${previewIndex + 1} of ${preview.length}` : 'Your script'}</h3>
            {#if preview}<button class="text-button" disabled={busy} on:click={() => (preview = null)}
                >Edit full script</button
              >{:else}<label class="import-button"
                ><FileText size={14} />Import text<input
                  aria-label="Import text file"
                  type="file"
                  accept=".txt,.md,.markdown,text/plain,text/markdown"
                  disabled={busy}
                  on:change={(e) => {
                    void importText(e.currentTarget.files?.[0]);
                    e.currentTarget.value = '';
                  }}
                /></label
              >{/if}
          </div>
          {#if preview}<textarea
              class="writing-surface"
              aria-label={`Preview passage ${previewIndex + 1}`}
              readonly
              value={preview[previewIndex] || ''}></textarea>
            <div class="editor-meta">
              <span>{preview.length} passages · ready to assign speakers</span><span>Preview</span>
            </div>{:else}<textarea
              class="writing-surface"
              aria-label="Script text"
              bind:value={project.source}
              on:input={() => {
                preview = null;
                touch();
              }}
              disabled={busy}
              placeholder={'Every narration starts with a few words.\n\nPaste your story, script, or article here…'}
            ></textarea>
            <div class="editor-meta">
              <span>{project.source.length.toLocaleString()} characters</span><span>TXT & Markdown</span>
            </div>{/if}
        {:else if stage === 'speech'}
          {#if selectedChunk}{@const chunk = selectedChunk}{@const index = selectedIndex}{@const take =
              selectedTake(chunk)}{@const current = chunkReady(project, chunk)}
            <div class="editor-heading">
              <div class="passage-heading">
                <span class="eyebrow">PASSAGE {index + 1} OF {project.chunks.length}</span>
                <h3>{assignedSpeaker(project, chunk)?.name || 'Choose a speaker'}</h3>
              </div>
              <div class="passage-traverse">
                <button
                  aria-label="Previous passage"
                  disabled={busy || index === 0}
                  on:click={() => selectChunk(project!.chunks[index - 1].id)}><ChevronLeft size={16} /></button
                ><button
                  aria-label="Next passage"
                  disabled={busy || index === project.chunks.length - 1}
                  on:click={() => selectChunk(project!.chunks[index + 1].id)}><ChevronRight size={16} /></button
                >
              </div>
            </div>
            <label class="sr-only" for={`text-${chunk.id}`}>Text to read</label><textarea
              id={`text-${chunk.id}`}
              class="writing-surface passage-text"
              aria-label={`Chunk ${index + 1} text`}
              bind:this={editors[chunk.id]}
              bind:value={chunk.text}
              disabled={busy}
              on:input={touch}></textarea>
            <div class="editor-meta">
              <span class="chunk-badge" class:ready={current}
                >{workingId === chunk.id
                  ? 'Generating…'
                  : !take
                    ? 'No audio yet'
                    : current
                      ? 'Ready'
                      : chunk.takes.some((t) => t.signature === speechSignature(project!, chunk))
                        ? 'Choose an updated take'
                        : 'Audio needs updating'}</span
              ><span>{chunk.text.length} characters</span>
              <details class="passage-actions">
                <summary>Edit passage<ChevronDown size={12} /></summary>
                <div>
                  <button disabled={busy} on:click={() => splitChunk(chunk)}
                    ><Scissors size={13} />Split at cursor</button
                  ><button disabled={busy || index === project.chunks.length - 1} on:click={() => mergeChunk(chunk)}
                    ><Merge size={13} />Join with next</button
                  ><button class="danger" disabled={busy} on:click={() => archiveChunk(chunk)}
                    ><Trash2 size={13} />Remove passage</button
                  >
                </div>
              </details>
            </div>
            <section class="audio-dock" aria-label="Passage audio">
              <div class="audio-dock-heading">
                <h4>{take ? `Take ${chunk.takes.indexOf(take) + 1}` : 'Ready when you are'}</h4>
                <span
                  >{take
                    ? `${(take.rendered?.duration ?? take.duration).toFixed(1)} seconds`
                    : 'Generate this passage to hear it.'}</span
                >{#if take && !current}<span class="attention"
                    >{chunk.takes.some((t) => t.signature === speechSignature(project!, chunk))
                      ? 'Choose an updated take.'
                      : 'Text or speaker changed.'}</span
                  >{/if}
              </div>
              {#if take}<audio
                  controls
                  aria-label="Selected take playback"
                  src={urlFor(take.rendered?.audio || take.audio)}
                  on:play={playback}
                ></audio>{/if}
              <div class="local-actions">
                <button
                  disabled={busy || !chunkConfiguration(project, chunk) || !chunk.text.trim()}
                  on:click={() => generate([chunk.id])}
                  ><RotateCcw size={14} />{take ? 'Try another take' : 'Generate this passage'}</button
                >{#if chunk.takes.length}<button disabled={busy} on:click={openComparison}
                    >Compare takes ({chunk.takes.length})</button
                  ><button disabled={busy} on:click={() => (audioEditorOpen = true)}
                    ><Scissors size={14} />Edit audio</button
                  >{/if}
              </div>
            </section>
          {:else}<div class="empty-state">
              <FileText size={30} />
              <h3>Add your script first</h3>
              <button on:click={() => goToStage('script')}>Go to script</button>
            </div>{/if}
        {:else if stage === 'assemble'}
          <div class="editor-heading">
            <h3>Listen to the join</h3>
            <div class="passage-traverse">
              <button
                aria-label="Previous passage"
                disabled={busy || selectedIndex <= 0}
                on:click={() => navigateAssembly(selectedIndex - 1)}><ChevronLeft size={16} /></button
              ><button
                aria-label="Next passage"
                disabled={busy || selectedIndex === project.chunks.length - 1}
                on:click={() => navigateAssembly(selectedIndex + 1)}><ChevronRight size={16} /></button
              >
            </div>
          </div>
          <div class="arrange-content">
            {#if ready !== project.chunks.length}<div class="notice">
                {project.chunks.length - ready} passages need current audio.<button
                  disabled={busy}
                  on:click={() => goToStage('speech')}>Review recordings</button
                >
              </div>{/if}{#if selectedChunk}<BoundaryEditor
                {project}
                chunk={selectedChunk}
                next={project.chunks[selectedIndex + 1]}
                index={selectedIndex}
                increment={project.pauseIncrement ?? 0.1}
                disabled={busy}
                active={active && !busy}
                onchange={touch}
              />{/if}
          </div>
          <div class="local-actions order-actions">
            <button
              disabled={busy || selectedIndex <= 0}
              on:click={() => {
                const at = selectedIndex - 1;
                moveChunk(selectedChunk!, -1);
                navigateAssembly(at);
              }}><ArrowUp size={14} />Move earlier</button
            ><button
              disabled={busy || selectedIndex === project.chunks.length - 1}
              on:click={() => {
                const at = selectedIndex + 1;
                moveChunk(selectedChunk!, 1);
                navigateAssembly(at);
              }}><ArrowDown size={14} />Move later</button
            ><button disabled={busy} on:click={() => goToStage('speech')}>Edit passage</button>
          </div>
          {#if project.assembled}<div class="assembly-dock">
              <h4>{assemblyIsCurrent ? 'Full narration' : 'Previous narration · needs updating'}</h4>
              <audio
                controls
                aria-label="Combined narration playback"
                src={urlFor(project.assembled.audio)}
                bind:this={player}
                on:play={playback}
              ></audio>
            </div>{/if}
        {:else}
          <div class="editor-heading">
            <h3>One final listen</h3>
            <span class="eyebrow">{project.chunks.length} PASSAGES</span>
          </div>
          <div class="final-preview">
            <AudioLines size={38} />
            <h3>{project.title}</h3>
            <p>{project.assembly.sampleRate.toLocaleString()} Hz · {project.finish.format.toUpperCase()}</p>
            {#if project.assembled}<audio
                controls
                aria-label="Full narration playback"
                src={urlFor(project.assembled.audio)}
                on:play={playback}
              ></audio>{/if}{#if project.finished}<div class="processed-preview">
                <h4>{finalCurrent ? 'Processed preview' : 'Previous processing · needs updating'}</h4>
                <audio
                  controls
                  aria-label="Processed narration playback"
                  src={urlFor(project.finished.audio)}
                  on:play={playback}
                ></audio>
              </div>{/if}
          </div>
        {/if}
      </section>
      <aside
        class="settings-rail"
        class:mobile-visible={mobilePane === 'settings'}
        aria-label={stage === 'speech' ? 'Speaker controls' : 'Project settings'}
      >
        {#if stage === 'script'}<div class="settings-content">
            <h3>Script preparation</h3>
            <p>Each passage gets its own recording. Paragraphs are a good place to start.</p>
            <label
              >Split by<select
                aria-label="Split method"
                bind:value={project.split.method}
                disabled={busy}
                on:change={() => {
                  preview = null;
                  touch();
                }}
                ><option value="paragraph">Paragraphs</option><option value="sentence">Sentences</option><option
                  value="size">Text length</option
                ></select
              ></label
            ><label
              >Maximum characters per passage<input
                aria-label="Chunk size"
                type="number"
                min="40"
                step="10"
                bind:value={project.split.characters}
                disabled={busy}
                on:input={() => {
                  preview = null;
                  touch();
                }}
              /></label
            >
            <div class="rail-note">
              <h4>Next: speakers</h4>
              <p>Create a named speaker by choosing a model and voice. Reuse that speaker across your passages.</p>
            </div>
          </div>
        {:else if stage === 'speech'}<SpeakerPanel
            bind:this={speakerPanel}
            {project}
            chunk={selectedChunk}
            {choices}
            {voices}
            disabled={busy}
            onchange={() => {
              touch();
              mobilePane = 'editor';
            }}
            onbusy={(value) => (busy = value)}
            onerror={(message) => (error = message)}
          />
        {:else if stage === 'assemble'}<div class="settings-content">
            <h3>Pauses & audio</h3>
            <label
              >Default pause (seconds)<input
                aria-label="Default pause"
                type="number"
                min="0"
                step="0.05"
                bind:value={project.assembly.pause}
                disabled={busy}
                on:input={touch}
              /></label
            >
            <p>Applies between passages. The center preview lets you override one join.</p>
            <label
              >Pause adjustment step<select
                aria-label="Pause increment"
                value={project.pauseIncrement ?? 0.1}
                disabled={busy}
                on:change={(e) => {
                  project!.pauseIncrement = Number(e.currentTarget.value);
                  touch();
                }}
                >{#each [0.05, 0.1, 0.25, 0.5, 1] as seconds}<option value={seconds}
                    >{seconds.toFixed(2)} seconds</option
                  >{/each}</select
              ></label
            ><label
              >Output sample rate<select
                aria-label="Output sample rate"
                bind:value={project.assembly.sampleRate}
                disabled={busy}
                on:change={touch}
                >{#each [16000, 22050, 24000, 44100, 48000] as rate}<option value={rate}
                    >{rate.toLocaleString()} Hz</option
                  >{/each}</select
              ></label
            >
          </div>
        {:else}<div class="settings-content">
            <h3>Download settings</h3>
            <label
              >Audio format<select
                aria-label="Export format"
                bind:value={project.finish.format}
                disabled={busy}
                on:change={() => {
                  exportLink = '';
                  touch();
                }}><option value="wav">WAV · uncompressed</option><option value="webm">WebM · Opus</option></select
              ></label
            >
            <p>
              {project.finish.format === 'wav'
                ? 'Full-quality PCM 16-bit audio, ready for your editor.'
                : 'A smaller audio file, encoded using your browser.'}
            </p>
            <details class="processing">
              <summary>Polish audio <span>Optional</span></summary><ModelChoice
                label="Enhancement model"
                emptyLabel="Keep original audio"
                {choices}
                tasks={['s2s', 'gen']}
                families={enhancementFamilies}
                bind:value={project.enhancement}
                disabled={busy}
                onchange={touch}
              /><label class="check-label"
                ><input
                  aria-label="Peak normalization"
                  type="checkbox"
                  bind:checked={project.finish.normalize}
                  disabled={busy}
                  on:change={touch}
                />Normalize peak volume</label
              >{#if project.finish.normalize}<label
                  >Maximum peak (dBFS)<input
                    aria-label="Target peak"
                    type="number"
                    min="-60"
                    max="0"
                    step="0.5"
                    bind:value={project.finish.peakDb}
                    disabled={busy}
                    on:input={touch}
                  /></label
                >{:else}<label
                  >Volume adjustment (dB)<input
                    aria-label="Audio gain"
                    type="number"
                    min="-60"
                    max="24"
                    step="0.5"
                    bind:value={project.finish.gainDb}
                    disabled={busy}
                    on:input={touch}
                  /></label
                >{/if}<button disabled={busy || !assemblyIsCurrent} on:click={renderFinish}>Preview processing</button>
            </details>
            <div class="rail-note">
              <Download size={16} />
              <p>Projects save in this browser. Download a copy to keep.</p>
            </div>
            {#if exportLink && finalCurrent && exportedSignature === finishSignature(project) + project.finish.format}<a
                href={exportLink}
                download={exportName}>Download again</a
              >{/if}
          </div>{/if}
      </aside>
    </div>
    <footer class="workflow-footer" aria-label="Next action">
      {#if stage !== 'script'}<button
          class="back-button"
          disabled={busy}
          on:click={() => goToStage(steps[steps.findIndex((s) => s.id === stage) - 1].id)}
          ><ChevronLeft size={15} />Back</button
        >{/if}
      <div class="next-guidance" role="status" aria-live="polite">
        <strong>{busy ? status : status || nextTitle}</strong><span
          >{busy ? 'Completed work is saved as you go.' : nextHint}</span
        >
      </div>
      {#if busy}{#if aborter}<button on:click={() => aborter?.abort()}><Square size={14} />Cancel</button
          >{/if}{:else}<button class="primary" disabled={primaryDisabled} on:click={primaryAction}
          >{stage === 'finish'
            ? 'Download ' + project.finish.format.toUpperCase()
            : primaryLabel}{#if stage === 'finish'}<Download
              size={16}
            />{:else if stage === 'speech' && needsGeneration.length && generationConfigured}<Play
              size={15}
            />{:else}<ChevronRight size={16} />{/if}</button
        >{/if}
    </footer>
    {#if comparisonOpen && selectedChunk && comparedTake}<dialog
        class="tool-dialog compare-dialog"
        use:openDialog
        aria-labelledby="compare-title"
        on:cancel={() => (comparisonOpen = false)}
      >
        <header>
          <div>
            <h2 id="compare-title">Choose a take</h2>
            <p>Passage {selectedIndex + 1} · {assignedSpeaker(project, selectedChunk)?.name || 'Speaker'}</p>
          </div>
          <button on:click={() => (comparisonOpen = false)}>Close</button>
        </header>
        <div class="comparison-body">
          <div class="take-list" aria-label="Take choices">
            {#each selectedChunk.takes as candidate, i}<button
                class:chosen={candidate.id === comparedTake.id}
                aria-pressed={candidate.id === comparedTake.id}
                on:click={() => (comparedTakeId = candidate.id)}
                ><strong>Take {i + 1}</strong><small
                  >{candidate.id === selectedChunk.takeId
                    ? 'Used in narration'
                    : candidate.signature === speechSignature(project, selectedChunk)
                      ? 'Matches current settings'
                      : 'Earlier text or speaker'}</small
                ><span>{(candidate.rendered?.duration ?? candidate.duration).toFixed(1)} s</span></button
              >{/each}
          </div>
          <div class="take-listen">
            <h3>Take {selectedChunk.takes.indexOf(comparedTake) + 1}</h3>
            <p>{comparedTake.text}</p>
            {#key comparedTake.id}<audio
                controls
                aria-label="Compare take playback"
                src={urlFor(comparedTake.rendered?.audio || comparedTake.audio)}
                on:play={playback}
              ></audio>{/key}
          </div>
        </div>
        <footer>
          <span>Trying another take keeps your current choice.</span><button
            disabled={busy || comparedTake.id === selectedChunk.takeId}
            on:click={() => {
              useTake(selectedChunk!, comparedTake!);
              comparisonOpen = false;
            }}>{comparedTake.id === selectedChunk.takeId ? 'Selected' : 'Use this take'}</button
          >
        </footer>
      </dialog>{/if}
    {#if audioEditorOpen && currentTake}<dialog
        class="tool-dialog audio-dialog"
        use:openDialog
        aria-labelledby="audio-editor-title"
        on:cancel={() => (audioEditorOpen = false)}
      >
        <header>
          <div>
            <h2 id="audio-editor-title">Edit passage audio</h2>
            <p>Passage {selectedIndex + 1} · original audio is preserved</p>
          </div>
          <button on:click={() => (audioEditorOpen = false)}>Done</button>
        </header>
        <div class="audio-editor-body">
          <WaveformEditor
            take={currentTake}
            sampleRate={project.assembly.sampleRate}
            increment={project.pauseIncrement ?? 0.1}
            disabled={busy}
            {active}
            onchange={(previous) => {
              release(previous);
              touch();
            }}
            onincrement={(seconds) => {
              project!.pauseIncrement = seconds;
              touch();
            }}
          />
        </div>
      </dialog>{/if}
    {#if archiveOpen}<dialog
        class="tool-dialog archive-dialog"
        use:openDialog
        aria-labelledby="archive-title"
        on:cancel={() => (archiveOpen = false)}
      >
        <header>
          <h2 id="archive-title">Passage archive</h2>
          <button on:click={() => (archiveOpen = false)}>Close</button>
        </header>
        <div class="archive-body">
          {#each project.archivedChunks.slice(archivePage * navigatorPageSize, (archivePage + 1) * navigatorPageSize) as chunk, index}<div
              class="archive-row"
            >
              <p>{chunk.text}<small>{chunk.takes.length} recordings</small></p>
              <button
                disabled={busy}
                on:click={() => {
                  project!.chunks = [...project!.chunks, chunk];
                  project!.archivedChunks.splice(archivePage * navigatorPageSize + index, 1);
                  touch();
                }}>Restore</button
              >
            </div>{/each}{#if archivePages > 1}<div class="local-actions">
              <button disabled={archivePage === 0} on:click={() => archivePage--}>Previous</button><span
                >{archivePage + 1} / {archivePages}</span
              ><button disabled={archivePage === archivePages - 1} on:click={() => archivePage++}>Next</button>
            </div>{/if}
        </div>
      </dialog>{/if}
  {/if}
</section>

<style>
  :global(body:has(.tts-app[data-active='true'])) {
    overflow: hidden;
  }
  :global(main:has(.tts-app[data-active='true'])) {
    padding-bottom: 0;
  }
  :global(main:has(.tts-app[data-active='true']) ~ footer) {
    display: none;
  }
  .tts-app {
    --tts-accent: var(--cyan);
    display: flex;
    flex-direction: column;
    height: calc(100dvh - var(--tts-top) - 12px);
    min-height: 320px;
    width: 100%;
    max-width: none;
    margin: 0;
  }
  .tts-app :global(*) {
    box-sizing: border-box;
  }
  h2,
  h3,
  h4,
  p {
    margin: 0;
  }
  h2 {
    font-size: 17px;
  }
  h3 {
    font-size: 15px;
    font-weight: 600;
  }
  h4 {
    font-size: 12px;
    font-weight: 600;
  }
  p {
    font-size: 12px;
    line-height: 1.65;
    color: var(--muted);
  }
  button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    min-height: 34px;
    font-size: 12px;
  }
  button:disabled {
    cursor: not-allowed;
  }
  button:focus-visible,
  summary:focus-visible,
  a:focus-visible {
    outline: 2px solid var(--cyan);
    outline-offset: 2px;
  }
  a {
    color: var(--cyan);
  }
  label {
    display: grid;
    gap: 8px;
    color: var(--muted);
    font-size: 12px;
  }
  input,
  select,
  textarea {
    min-width: 0;
    width: 100%;
    font: inherit;
  }
  .eyebrow {
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 1.3px;
    font-weight: 500;
  }
  .project-bar {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 20px;
    min-height: 60px;
    padding: 0 0 14px;
    border-bottom: 1px solid var(--line);
  }
  .project-heading {
    display: grid;
    gap: 4px;
    flex: 1;
    min-width: 100px;
  }
  .project-title {
    background: transparent;
    border: 0;
    border-radius: 0;
    padding: 0;
    box-shadow: none;
    font-size: 18px;
    font-weight: 600;
    width: 100%;
  }
  .save-status {
    font-size: 10px;
    color: var(--muted);
    white-space: nowrap;
  }
  .pipeline {
    height: auto;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .pipeline button {
    background: transparent;
    border: 0;
    border-radius: 6px;
    color: var(--muted);
    font-size: 11px;
    min-height: 34px;
    padding: 6px 10px;
  }
  .pipeline button.active {
    color: var(--text);
    background: var(--active-bg);
  }
  .step-number {
    width: 19px;
    height: 19px;
    display: grid;
    place-items: center;
    border: 1px solid var(--line);
    border-radius: 50%;
    font-size: 9px;
  }
  .active .step-number {
    background: var(--cyan);
    color: var(--text-invert);
    border-color: var(--cyan);
  }
  summary {
    display: flex;
    align-items: center;
    gap: 7px;
    list-style: none;
    cursor: pointer;
    background: transparent;
    font-size: 12px;
    font-weight: 500;
  }
  summary::-webkit-details-marker {
    display: none;
  }
  .project-menu {
    position: relative;
    flex-shrink: 0;
    margin: 0;
    border: 0;
    overflow: visible;
  }
  .project-menu > summary {
    padding: 8px;
    border: 1px solid var(--line);
    border-radius: 6px;
  }
  .project-menu-content {
    position: absolute;
    top: 40px;
    right: 0;
    z-index: 25;
    display: grid;
    gap: 16px;
    width: 280px;
    padding: 20px;
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 10px;
    box-shadow: 0 12px 36px var(--panel-shadow);
  }
  .project-menu-content small {
    font-size: 10px;
    color: var(--muted);
    line-height: 1.5;
  }
  .stage-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 18px;
    flex-shrink: 0;
    padding: 15px 0 16px;
  }
  .stage-title {
    font-size: 17px;
    font-weight: 600;
    margin-bottom: 4px;
  }
  .stage-title:focus {
    outline: none;
  }
  .stage-toolbar p {
    font-size: 11px;
  }
  .project-progress {
    font-size: 11px;
    color: var(--muted);
    white-space: nowrap;
  }
  .tts-workbench {
    display: grid;
    grid-template-columns: 220px minmax(0, 1fr) 286px;
    flex: 1;
    min-height: 0;
    overflow: hidden;
    border: 1px solid var(--line);
    border-bottom: 0;
    border-radius: 10px 10px 0 0;
    background: var(--panel);
  }
  .passage-rail,
  .settings-rail {
    padding: 20px 16px;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
  }
  .passage-rail {
    display: flex;
    flex-direction: column;
  }
  .passage-rail :global(.passage-navigator) {
    flex: 1;
    height: auto;
    min-height: 0;
  }
  .settings-rail {
    overflow-y: auto;
    scrollbar-width: thin;
  }
  .editor-pane {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    border-inline: 1px solid var(--line);
    padding: 22px 26px;
    overflow-y: auto;
    scrollbar-width: thin;
  }
  .editor-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding-bottom: 18px;
    flex-shrink: 0;
  }
  .editor-heading h3 {
    font-size: 15px;
  }
  .passage-heading {
    display: grid;
    gap: 7px;
  }
  .passage-heading h3 {
    font-size: 17px;
  }
  .passage-traverse {
    display: flex;
    gap: 4px;
  }
  .passage-traverse > button {
    background: transparent;
    border: 0;
    padding: 7px;
    min-height: 30px;
  }
  .writing-surface {
    flex: 1;
    resize: none;
    min-height: 90px;
    display: block;
    border: 0;
    border-radius: 0;
    background: transparent;
    font-size: 18px;
    line-height: 1.9;
    padding: 10px 2px;
    color: var(--text);
    box-shadow: none;
  }
  .writing-surface:focus {
    box-shadow: none;
    outline: 1px solid var(--cyan);
    outline-offset: 5px;
  }
  .writing-surface[readonly]:focus {
    outline: none;
  }
  .editor-meta {
    display: flex;
    align-items: center;
    gap: 14px;
    color: var(--muted);
    font-size: 10px;
    padding-top: 14px;
    margin-top: 12px;
    border-top: 1px solid var(--line);
    flex-shrink: 0;
  }
  .editor-meta > span:nth-child(2) {
    margin-left: auto;
  }
  .chunk-badge {
    font-size: 10px;
    color: var(--muted);
  }
  .chunk-badge.ready {
    color: var(--cyan);
  }
  .passage-actions {
    position: relative;
    border: 0;
    border-radius: 0;
    overflow: visible;
    margin: 0;
  }
  .passage-actions > summary {
    padding: 6px 0;
    font-size: 10px;
    color: var(--muted);
  }
  .passage-actions > div {
    position: absolute;
    bottom: 32px;
    right: 0;
    z-index: 5;
    display: grid;
    gap: 8px;
    min-width: 180px;
    padding: 12px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: 0 8px 20px var(--panel-shadow);
  }
  .passage-actions button {
    justify-content: flex-start;
    font-size: 11px;
  }
  .audio-dock {
    flex-shrink: 0;
    padding-top: 18px;
  }
  .audio-dock-heading {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 12px;
  }
  .audio-dock-heading > span {
    font-size: 10px;
    color: var(--muted);
  }
  .audio-dock-heading .attention {
    color: var(--cyan);
    margin-left: auto;
  }
  audio {
    display: block;
    width: 100%;
    height: 36px;
  }
  .local-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
  }
  .audio-dock .local-actions {
    margin-top: 14px;
  }
  .local-actions button {
    font-size: 11px;
    min-height: 32px;
  }
  .local-actions button:not(:first-child) {
    background: transparent;
  }
  .settings-content {
    display: grid;
    gap: 20px;
  }
  .settings-content > h3 {
    font-size: 15px;
  }
  .settings-content p {
    font-size: 11px;
  }
  .settings-content label {
    font-size: 11px;
  }
  .rail-note {
    display: grid;
    gap: 10px;
    padding-top: 20px;
    border-top: 1px solid var(--line);
    color: var(--muted);
  }
  .rail-note h4 {
    font-size: 12px;
  }
  .rail-note p {
    font-size: 11px;
  }
  .import-button {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--line);
    padding: 7px 10px;
    font-size: 11px;
    border-radius: 6px;
    cursor: pointer;
    overflow: hidden;
    color: var(--text);
  }
  .import-button input {
    opacity: 0;
    position: absolute;
    inset: 0;
    height: 100%;
    width: 100%;
    cursor: pointer;
  }
  .import-button:focus-within {
    outline: 2px solid var(--cyan);
    outline-offset: 2px;
  }
  .text-button {
    background: transparent;
    border: 0;
    font-size: 11px;
    color: var(--muted);
  }
  .arrange-content {
    flex: 1;
    min-height: 0;
    overflow: auto;
    margin: 0 -6px 0 0;
    padding-right: 6px;
  }
  .arrange-content :global(.boundary-editor) {
    padding: 0;
    border: 0;
    border-radius: 0;
  }
  .arrange-content :global(.boundary-editor .passages) {
    margin-top: 0;
  }
  .arrange-content :global(.boundary-editor .passages p) {
    max-height: 70px;
    font-size: 12px;
  }
  .arrange-content :global(.boundary-editor .wave-hint),
  .arrange-content :global(.boundary-editor .click-hint) {
    font-size: 10px;
  }
  .order-actions {
    padding-top: 16px;
    flex-shrink: 0;
  }
  .assembly-dock {
    border-top: 1px solid var(--line);
    padding-top: 16px;
    margin-top: 16px;
    flex-shrink: 0;
  }
  .assembly-dock h4 {
    margin-bottom: 10px;
  }
  .final-preview {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 16px;
    text-align: center;
    min-height: 0;
    overflow: auto;
  }
  .final-preview > :global(svg) {
    color: var(--cyan);
    flex-shrink: 0;
  }
  .final-preview h3 {
    font-size: 21px;
    overflow-wrap: anywhere;
  }
  .final-preview p {
    font-size: 11px;
  }
  .final-preview audio {
    margin-top: 10px;
    flex-shrink: 0;
  }
  .processed-preview {
    width: 100%;
    border-top: 1px solid var(--line);
    padding-top: 20px;
  }
  .processed-preview h4 {
    text-align: left;
    margin-bottom: 10px;
  }
  .processing {
    border: 0;
    border-radius: 0;
    border-block: 1px solid var(--line);
    margin: 0;
  }
  .processing > summary {
    padding: 14px 0;
    display: flex;
    gap: 8px;
    font-size: 12px;
  }
  .processing > summary span {
    margin-left: auto;
    font-size: 10px;
    color: var(--muted);
  }
  .processing[open] {
    display: grid;
    gap: 18px;
    padding-bottom: 18px;
  }
  .check-label {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .check-label input {
    width: 15px;
    height: 15px;
  }
  .workflow-footer {
    position: static;
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 14px 18px;
    border: 1px solid var(--line);
    border-radius: 0 0 10px 10px;
    background: var(--panel);
    margin: 0;
    flex-shrink: 0;
  }
  .next-guidance {
    display: grid;
    gap: 5px;
    flex: 1;
    min-width: 0;
  }
  .next-guidance strong {
    font-size: 12px;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .next-guidance span {
    font-size: 10px;
    color: var(--muted);
    line-height: 1.5;
  }
  .workflow-footer > .primary {
    background: var(--cyan);
    color: var(--text-invert);
    border: 1px solid var(--cyan);
    min-height: 40px;
    padding: 8px 18px;
    font-size: 12px;
    white-space: nowrap;
  }
  .back-button {
    background: transparent;
    font-size: 11px;
  }
  .pane-tabs {
    width: 100%;
    margin: 0;
    height: auto;
    display: none;
  }
  .archive-trigger {
    font-size: 10px;
    color: var(--muted);
    border: 0;
    border-top: 1px solid var(--line);
    border-radius: 0;
    background: transparent;
    margin-top: 8px;
  }
  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 18px;
    color: var(--muted);
  }
  .error,
  .notice {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    margin: 8px 0;
    border: 1px solid var(--line);
    border-radius: 6px;
    font-size: 12px;
    flex-shrink: 0;
  }
  .error {
    color: var(--danger);
    border-color: var(--danger);
  }
  .error > span {
    flex: 1;
  }
  .error button {
    background: transparent;
    border: 0;
    font-size: 18px;
    min-height: 28px;
  }
  .danger {
    color: var(--danger);
  }
  .tool-dialog {
    padding: 0;
    width: min(860px, calc(100vw - 28px));
    max-height: calc(100dvh - 40px);
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 12px;
  }
  .tool-dialog::backdrop {
    background: #05060db3;
    backdrop-filter: blur(3px);
  }
  .tool-dialog > header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 22px 24px;
    border-bottom: 1px solid var(--line);
  }
  .tool-dialog h2 {
    font-size: 18px;
    margin-bottom: 5px;
  }
  .tool-dialog > footer {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 16px;
    padding: 18px 24px;
    border-top: 1px solid var(--line);
  }
  .tool-dialog > footer span {
    margin-right: auto;
    color: var(--muted);
    font-size: 11px;
  }
  .comparison-body {
    display: grid;
    grid-template-columns: 200px minmax(0, 1fr);
    min-height: 250px;
    max-height: 60dvh;
  }
  .take-list {
    display: flex;
    flex-direction: column;
    overflow: auto;
    border-right: 1px solid var(--line);
    padding: 14px;
    gap: 8px;
  }
  .take-list > button {
    display: grid;
    text-align: left;
    justify-content: stretch;
    gap: 5px;
    padding: 12px;
    background: transparent;
    border-color: transparent;
  }
  .take-list > button.chosen {
    background: var(--active-bg);
    border-color: var(--cyan);
  }
  .take-list strong {
    font-size: 12px;
  }
  .take-list small,
  .take-list span {
    font-size: 10px;
    color: var(--muted);
    line-height: 1.5;
  }
  .take-listen {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 28px;
    min-width: 0;
    overflow: auto;
  }
  .take-listen p {
    flex: 1;
    font-size: 15px;
    color: var(--text);
    line-height: 1.8;
    white-space: pre-wrap;
  }
  .take-listen audio {
    flex-shrink: 0;
  }
  .audio-editor-body {
    max-height: calc(100dvh - 150px);
    overflow: auto;
  }
  .archive-body {
    max-height: 65dvh;
    overflow: auto;
    padding: 20px;
  }
  .archive-row {
    display: flex;
    gap: 16px;
    padding: 12px 0;
    border-bottom: 1px solid var(--line);
    align-items: center;
  }
  .archive-row p {
    flex: 1;
  }
  .archive-row small {
    display: block;
    margin-top: 6px;
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
  @media (min-width: 1700px) {
    .tts-workbench {
      grid-template-columns: 240px minmax(0, 1fr) 310px;
    }
  }
  @media (max-width: 1150px) and (min-width: 1001px) {
    .tts-workbench {
      grid-template-columns: 190px minmax(0, 1fr) 248px;
    }
    .editor-pane {
      padding: 18px;
    }
    .passage-rail,
    .settings-rail {
      padding: 18px 12px;
    }
    .pipeline button {
      padding: 6px;
    }
    .project-bar {
      gap: 12px;
    }
  }
  @media (max-width: 1000px) {
    .tts-workbench {
      display: flex;
      flex-direction: column;
    }
    .passage-rail,
    .editor-pane,
    .settings-rail {
      display: none;
      flex: 1;
      min-height: 0;
      border: 0;
    }
    .mobile-visible {
      display: flex;
    }
    .settings-rail.mobile-visible {
      display: block;
    }
    .pane-tabs {
      display: flex;
      gap: 4px;
      padding: 5px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-bottom: 0;
      border-radius: 8px 8px 0 0;
      flex-shrink: 0;
    }
    .pane-tabs button {
      flex: 1;
      background: transparent;
      border: 0;
      font-size: 11px;
    }
    .pane-tabs button[aria-pressed='true'] {
      background: var(--active-bg);
      color: var(--cyan);
    }
    .tts-workbench {
      border-radius: 0;
    }
    .project-bar {
      flex-wrap: wrap;
      gap: 10px;
      padding-bottom: 10px;
    }
    .project-heading {
      flex: 1;
    }
    .pipeline {
      order: 4;
      width: 100%;
      justify-content: space-between;
    }
    .pipeline button {
      flex: 1;
    }
    .stage-toolbar {
      padding: 12px 0;
    }
    .stage-toolbar p {
      font-size: 10px;
    }
    .project-title {
      font-size: 16px;
    }
    .editor-pane {
      padding: 20px;
    }
    .settings-rail,
    .passage-rail {
      padding: 18px 20px;
    }
    .writing-surface {
      font-size: 17px;
    }
    .workflow-footer {
      gap: 12px;
      padding: 12px;
    }
    .next-guidance span {
      display: none;
    }
    .next-guidance strong {
      font-size: 10px;
    }
    .workflow-footer > .primary {
      padding: 8px 12px;
      font-size: 11px;
    }
  }
  @media (max-width: 520px) {
    .pipeline button {
      min-width: 0;
      padding: 6px 3px;
      gap: 4px;
      font-size: 10px;
    }
    .step-number {
      width: 16px;
      height: 16px;
      flex-shrink: 0;
      font-size: 8px;
    }
    .tts-app {
      height: calc(100dvh - var(--tts-top) - 6px);
      min-height: 280px;
    }
    .stage-toolbar h2 {
      font-size: 15px;
    }
    .stage-toolbar p {
      max-width: 300px;
    }
    .project-progress {
      display: none;
    }
    .editor-pane {
      padding: 16px;
    }
    .editor-heading {
      padding-bottom: 10px;
    }
    .writing-surface {
      font-size: 16px;
      min-height: 75px;
      line-height: 1.7;
    }
    .audio-dock {
      padding-top: 12px;
    }
    .local-actions {
      gap: 6px;
    }
    .local-actions button {
      font-size: 10px;
      padding: 5px 7px;
      min-height: 30px;
    }
    .editor-meta {
      gap: 8px;
      font-size: 9px;
      margin-top: 8px;
      padding-top: 8px;
    }
    .editor-meta > span:nth-child(2) {
      display: none;
    }
    .passage-actions {
      margin-left: auto;
    }
    .next-guidance {
      display: none;
    }
    .workflow-footer > .primary {
      flex: 1;
    }
    .back-button {
      padding: 6px 8px;
    }
    .audio-dock-heading {
      gap: 7px;
      margin-bottom: 8px;
    }
    .comparison-body {
      grid-template-columns: 1fr;
      max-height: 65dvh;
    }
    .take-list {
      max-height: 160px;
      border: 0;
      border-bottom: 1px solid var(--line);
    }
    .take-listen {
      padding: 20px;
    }
    .tool-dialog > footer {
      flex-wrap: wrap;
    }
    .tool-dialog > footer span {
      flex-basis: 100%;
    }
    .tool-dialog > header {
      padding: 18px;
    }
  }
</style>
