import type { ModelSettings } from '$lib/model-settings/types';

export interface ReferenceVoice {
  id: string;
  name: string;
  audio: Blob;
  transcript: string;
}
export type VoiceAssignment = { kind: 'builtin'; id: string } | { kind: 'reference'; data: ReferenceVoice };
export interface Speaker {
  id: string;
  name: string;
  configuration: ModelSettings;
  voice: VoiceAssignment;
}
export interface SilenceInsertion {
  id: string;
  at: number;
  seconds: number;
}
export interface AudioCut {
  start: number;
  end: number;
}
export interface TakeAudioEdits {
  silences: SilenceInsertion[];
  removed?: AudioCut[];
}
export type AudioEditSnapshot = TakeAudioEdits | SilenceInsertion[];
export interface Take {
  id: string;
  text: string;
  configuration: ModelSettings;
  reference?: ReferenceVoice;
  signature: string;
  seed: number;
  audio: Blob;
  duration: number;
  createdAt: number;
  editing?: TakeAudioEdits & { undo: AudioEditSnapshot[]; redo: AudioEditSnapshot[] };
  rendered?: { audio: Blob; duration: number };
}
export interface Chunk {
  id: string;
  text: string;
  takes: Take[];
  takeId: string;
  override: ModelSettings | null;
  voice?: VoiceAssignment;
  speakerId?: string | null;
  pauseAfter: number | null;
}
export interface TtsProject {
  version: 1 | 2;
  id: string;
  title: string;
  updatedAt: number;
  pauseIncrement?: number;
  speakers?: Speaker[];
  defaultSpeakerId?: string;
  source: string;
  chunks: Chunk[];
  archivedChunks: Chunk[];
  reference: ReferenceVoice | null;
  speech: ModelSettings | null;
  transcription: ModelSettings | null;
  enhancement: ModelSettings | null;
  split: { method: 'paragraph' | 'sentence' | 'size'; characters: number };
  assembly: { pause: number; sampleRate: number };
  finish: { normalize: boolean; peakDb: number; gainDb: number; format: 'wav' | 'webm' };
  assembled: { audio: Blob; signature: string } | null;
  finished: { audio: Blob; signature: string } | null;
}
export function newProject(): TtsProject {
  return {
    version: 2,
    id: crypto.randomUUID(),
    title: 'Untitled narration',
    updatedAt: Date.now(),
    speakers: [],
    defaultSpeakerId: '',
    pauseIncrement: 0.1,
    source: '',
    chunks: [],
    archivedChunks: [],
    reference: null,
    speech: null,
    transcription: null,
    enhancement: null,
    split: { method: 'paragraph', characters: 400 },
    assembly: { pause: 0.25, sampleRate: 24000 },
    finish: { normalize: false, peakDb: -1, gainDb: 0, format: 'wav' },
    assembled: null,
    finished: null
  };
}
export function newChunk(text: string, override: ModelSettings | null = null): Chunk {
  return { id: crypto.randomUUID(), text, speakerId: null, takes: [], takeId: '', override, pauseAfter: null };
}
export function selectedTake(chunk: Chunk) {
  return chunk.takes.find((take) => take.id === chunk.takeId);
}
export function chunkConfiguration(project: TtsProject, chunk: Chunk): ModelSettings | null {
  const speaker = assignedSpeaker(project, chunk);
  const preset = speaker?.configuration || chunk.override || project.speech;
  const voice = speaker?.voice || chunk.voice;
  if (!preset || voice?.kind !== 'builtin') return preset;
  const request = { ...preset.request };
  if (voice.id) request.voice = voice.id;
  else delete request.voice;
  return { ...preset, request };
}
export function chunkReference(project: TtsProject, chunk: Chunk): ReferenceVoice | null {
  const speaker = assignedSpeaker(project, chunk);
  if (speaker) return speaker.voice.kind === 'reference' ? speaker.voice.data : null;
  if (chunk.voice?.kind === 'builtin') return null;
  return chunk.voice?.kind === 'reference' ? chunk.voice.data : project.reference;
}
export function speechSignature(project: TtsProject, chunk: Chunk) {
  const preset = chunkConfiguration(project, chunk);
  const reference = chunkReference(project, chunk);
  return JSON.stringify([chunk.text, preset?.model, preset?.request, reference?.id, reference?.transcript]);
}
export function chunkReady(project: TtsProject, chunk: Chunk) {
  const take = selectedTake(chunk);
  return !!take && take.signature === speechSignature(project, chunk);
}
export function assemblySignature(project: TtsProject) {
  return JSON.stringify([
    project.chunks.map((c) => {
      const entry: unknown[] = [c.id, c.takeId, c.pauseAfter];
      const edits = selectedTake(c)?.editing;
      if (edits && (edits.silences.length || edits.removed?.length)) {
        entry.push(edits.silences);
        if (edits.removed?.length) entry.push(edits.removed);
      }
      return entry;
    }),
    project.assembly
  ]);
}
export function assemblyCurrent(project: TtsProject) {
  return (
    project.chunks.length > 0 &&
    project.chunks.every((c) => chunkReady(project, c)) &&
    project.assembled?.signature === assemblySignature(project)
  );
}
export function finishSignature(project: TtsProject) {
  const { format: _format, ...processing } = project.finish;
  return JSON.stringify([
    project.assembled?.signature,
    processing,
    project.enhancement?.model,
    project.enhancement?.request
  ]);
}

// Project speakers are snapshots: reusing a saved speaker never mutates another project.
export function assignedSpeaker(project: TtsProject, chunk?: Chunk): Speaker | undefined {
  const id = chunk?.speakerId || project.defaultSpeakerId;
  return project.speakers?.find((speaker) => speaker.id === id);
}
export function speakerVoiceLabel(speaker: Speaker): string {
  return speaker.voice.kind === 'reference'
    ? speaker.voice.data.name
    : speaker.voice.id.replace(/^[a-z]{2}_/, '').replace(/_/g, ' ') || 'Model voice';
}
export function ensureSpeakers(project: TtsProject): boolean {
  if (project.speakers) {
    const changed = project.version !== 2;
    project.version = 2;
    return changed;
  }
  project.version = 2;
  project.speakers = [];
  project.defaultSpeakerId = '';
  const identities = new Map<string, string>();
  function add(configuration: ModelSettings, voice: VoiceAssignment, name: string) {
    const identity = JSON.stringify([
      configuration.model,
      configuration.request,
      voice.kind === 'builtin' ? voice : [voice.data.id, voice.data.transcript]
    ]);
    const found = identities.get(identity);
    if (found) return found;
    const speaker: Speaker = {
      id: crypto.randomUUID(),
      name,
      configuration: structuredClone(configuration),
      voice: structuredClone(voice)
    };
    project.speakers!.push(speaker);
    identities.set(identity, speaker.id);
    return speaker.id;
  }
  if (project.speech)
    project.defaultSpeakerId = add(
      project.speech,
      project.reference
        ? { kind: 'reference', data: project.reference }
        : { kind: 'builtin', id: project.speech.request.voice || '' },
      'Main speaker'
    );
  for (const chunk of [...project.chunks, ...project.archivedChunks]) {
    if (!chunk.override && !chunk.voice) continue;
    const configuration = chunk.override || project.speech;
    if (!configuration) continue;
    const voice =
      chunk.voice ||
      (project.reference
        ? { kind: 'reference' as const, data: project.reference }
        : { kind: 'builtin' as const, id: configuration.request.voice || '' });
    chunk.speakerId = add(configuration, voice, `Speaker ${project.speakers.length + 1}`);
  }
  return true;
}
