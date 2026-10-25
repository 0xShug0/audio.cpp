import type { SavedModelPreset } from '$lib/presets/types';

export interface ReferenceVoice { id: string; name: string; audio: Blob; transcript: string }
export type VoiceAssignment = { kind: 'builtin'; id: string } | { kind: 'reference'; data: ReferenceVoice };
export interface SilenceInsertion { id: string; at: number; seconds: number }
export interface AudioCut { start: number; end: number }
export interface TakeAudioEdits { silences: SilenceInsertion[]; removed?: AudioCut[] }
export type AudioEditSnapshot = TakeAudioEdits | SilenceInsertion[];
export interface Take {
  id: string; text: string; configuration: SavedModelPreset; reference?: ReferenceVoice;
  signature: string; seed: number; audio: Blob; duration: number; createdAt: number;
  editing?: TakeAudioEdits & { undo: AudioEditSnapshot[]; redo: AudioEditSnapshot[] };
  rendered?: { audio: Blob; duration: number };
}
export interface Chunk {
  id: string; text: string; takes: Take[]; takeId: string;
  override: SavedModelPreset | null; voice?: VoiceAssignment; pauseAfter: number | null;
}
export interface TtsProject {
  version: 1; id: string; title: string; updatedAt: number;
  pauseIncrement?: number;
  source: string; chunks: Chunk[]; archivedChunks: Chunk[]; reference: ReferenceVoice | null;
  speech: SavedModelPreset | null; transcription: SavedModelPreset | null;
  enhancement: SavedModelPreset | null;
  split: { method: 'paragraph' | 'sentence' | 'size'; characters: number };
  assembly: { pause: number; sampleRate: number };
  finish: { normalize: boolean; peakDb: number; gainDb: number; format: 'wav' | 'webm' };
  assembled: { audio: Blob; signature: string } | null;
  finished: { audio: Blob; signature: string } | null;
}
export function newProject(): TtsProject {
  return { version: 1, id: crypto.randomUUID(), title: 'Untitled narration', updatedAt: Date.now(),
    pauseIncrement: .1, source: '', chunks: [], archivedChunks: [], reference: null, speech: null, transcription: null, enhancement: null,
    split: { method: 'paragraph', characters: 400 }, assembly: { pause: .25, sampleRate: 24000 },
    finish: { normalize: false, peakDb: -1, gainDb: 0, format: 'wav' }, assembled: null, finished: null };
}
export function newChunk(text: string, override: SavedModelPreset | null = null): Chunk {
  return { id: crypto.randomUUID(), text, takes: [], takeId: '', override, pauseAfter: null };
}
export function selectedTake(chunk: Chunk) { return chunk.takes.find(take => take.id === chunk.takeId); }
export function chunkConfiguration(project: TtsProject, chunk: Chunk): SavedModelPreset | null {
  const preset = chunk.override || project.speech;
  if (!preset || chunk.voice?.kind !== 'builtin') return preset;
  const request = { ...preset.request };
  if (chunk.voice.id) request.voice = chunk.voice.id;
  else delete request.voice;
  return { ...preset, request };
}
export function chunkReference(project: TtsProject, chunk: Chunk): ReferenceVoice | null {
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
  return JSON.stringify([project.chunks.map(c => {
    const entry: unknown[] = [c.id, c.takeId, c.pauseAfter];
    const edits = selectedTake(c)?.editing;
    if (edits && (edits.silences.length || edits.removed?.length)) { entry.push(edits.silences);if(edits.removed?.length)entry.push(edits.removed); }
    return entry;
  }), project.assembly]);
}
export function assemblyCurrent(project: TtsProject) {
  return project.chunks.length > 0 && project.chunks.every(c => chunkReady(project, c)) &&
    project.assembled?.signature === assemblySignature(project);
}
export function finishSignature(project: TtsProject) {
  const { format: _format, ...processing } = project.finish;
  return JSON.stringify([project.assembled?.signature, processing, project.enhancement?.model, project.enhancement?.request]);
}
