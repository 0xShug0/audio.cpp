import { health, models, loadModel, unloadModel, pathStatus, speech, transcription, runTask } from '$lib/api';
import type { SavedModelPreset } from './types';

export type PresetInputs = {
  text?: string;
  audio?: string;
  voice_ref?: string;
  reference_text?: string;
};
export type PresetResult =
  | { kind: 'audio'; blob: Blob }
  | { kind: 'json'; data: Record<string, unknown> };

// One runtime per app operation. Preparation applies the saved configuration
// once; all chunks in that operation reuse it. No Studio state or catalog lookup.
export class PresetRuntime {
  private readonly preset: SavedModelPreset;
  readonly resolvedSeed: number;
  private prepared = false;

  constructor(preset: SavedModelPreset, private readonly signal: AbortSignal) {
    if (preset.version !== 1) throw new Error('Unsupported model preset version.');
    this.preset = structuredClone(preset);
    const seed = preset.request.seed;
    if (!Number.isInteger(seed) || seed < -1 || seed > 0xffffffff) throw new Error('Invalid preset seed.');
    this.resolvedSeed = seed === -1 ? crypto.getRandomValues(new Uint32Array(1))[0] : seed;
  }

  async prepare(): Promise<void> {
    if (this.prepared) return;
    this.signal.throwIfAborted();
    const server = await health();
    const registered = await models();
    this.signal.throwIfAborted();
    const saved = this.preset.model;
    const resident = registered.find((model) => model.id === saved.id);
    if (!server.ui_management || resident?.configured) {
      if (!resident || resident.family !== saved.family || resident.task !== saved.task || resident.mode !== saved.mode) {
        throw new Error('This preset does not match a model registered by this configured server.');
      }
      if (resident.path !== saved.path || Object.entries(saved.session_options)
        .some(([key, value]) => resident.session_options?.[key] !== value)) {
        throw new Error('The preset differs from the server configuration. Update the server configuration before using it.');
      }
      if (Object.keys(saved.load_options).length) {
        throw new Error('Saved load options cannot be applied to a server-configured model. Use a model managed by the UI.');
      }
    } else {
      if (!(await pathStatus(saved.path)).exists) throw new Error('The preset model is not installed at its saved path. Install it in Models first.');
      this.signal.throwIfAborted();
      // Load options are not reported by /v1/models. Explicitly reload to apply
      // the complete saved configuration rather than assuming a resident match.
      if (resident?.loaded) await unloadModel(saved.id, this.signal);
      this.signal.throwIfAborted();
      await loadModel({ id: saved.id, family: saved.family, task: saved.task,
        mode: saved.mode, path: saved.path,
        load_options: saved.load_options, session_options: saved.session_options }, this.signal);
    }
    this.signal.throwIfAborted();
    this.prepared = true;
  }

  async run(inputs: PresetInputs, sequence = 0): Promise<PresetResult> {
    await this.prepare();
    this.signal.throwIfAborted();
    const { model, request } = this.preset;
    const { seed: _seed, ...settings } = request;
    const options = { ...request.options };
    if (['tts', 'clon', 'vdes'].includes(model.task)) {
      if (!inputs.text?.trim()) throw new Error('Enter text to synthesize.');
      const body: Record<string, unknown> = { ...settings, options, model: model.id,
        input: inputs.text, response_format: 'wav', seed: (this.resolvedSeed + sequence) % 0x100000000 };
      if (inputs.voice_ref) { delete body.voice; body.voice_ref = inputs.voice_ref; }
      if (inputs.reference_text) body.reference_text = inputs.reference_text;
      const result = await speech(body, this.signal);
      return { kind: 'audio', blob: result.blob };
    }
    if (model.task === 'asr') {
      if (!inputs.audio) throw new Error('Choose audio to transcribe.');
      const detail = ['owsm', 'crisperwhisper'].includes(model.family) && options.return_timestamps === true;
      const data = await transcription({ model: model.id, audio: inputs.audio,
        language: request.language, text: inputs.text || '', options }, this.signal, detail);
      return { kind: 'json', data };
    }
    const data = await runTask({ model: model.id, request: { ...settings, ...inputs, options, seed: (this.resolvedSeed + sequence) % 0x100000000 } }, this.signal);
    if (typeof data.audio !== 'string') throw new Error('The processing model did not return a single audio output.');
    const binary = atob(data.audio);
    const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
    return { kind: 'audio', blob: new Blob([bytes], { type: 'audio/wav' }) };
  }
}
