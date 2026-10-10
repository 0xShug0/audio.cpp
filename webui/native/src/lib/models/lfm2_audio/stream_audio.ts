// Live playback of an LFM2.5-Audio reply as /v1/tasks/stream sends it: each
// event carries a small WAV of the reply's next samples.

export function base64Bytes(data: string): Uint8Array<ArrayBuffer> {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

// Reads the first channel of a 16-bit PCM WAV. The server writes these, so
// other formats are an error.
export function readPcm16Wav(bytes: Uint8Array): { sampleRate: number; samples: Float32Array<ArrayBuffer> } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (offset: number) =>
    String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
  if (bytes.length < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('The reply audio is not a WAV file.');
  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  for (let offset = 12; offset + 8 <= bytes.length;) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === 'fmt ' && body + 16 <= bytes.length) {
      format = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
    } else if (id === 'data') {
      if (format !== 1 || bits !== 16 || channels < 1 || sampleRate < 1) {
        throw new Error('The reply audio is not 16-bit PCM.');
      }
      const frames = Math.floor(Math.min(size, bytes.length - body) / (2 * channels));
      const samples = new Float32Array(frames);
      for (let frame = 0; frame < frames; frame += 1) {
        samples[frame] = view.getInt16(body + frame * 2 * channels, true) / 32768;
      }
      return { sampleRate, samples };
    }
    offset = body + size + (size & 1);
  }
  throw new Error('The reply audio has no samples.');
}

// Plays the chunks of one reply back to back on the audio clock, so that
// they join without gaps however the events arrive. It starts a little
// ahead of the first chunk, and further ahead after a chunk came too late.
export class StreamPlayer {
  private context: AudioContext | null = null;
  private sources = new Set<AudioBufferSourceNode>();
  private enabled = false;
  private open = false;
  private startTime = 0;
  private frames = 0;
  private rate = 0;
  private lead = 0.15;
  unavailable = false;
  onchange: () => void = () => {};

  // Call from the Run click or key press, before anything is awaited: a
  // browser lets a page start audio only then. Stops the reply before.
  beginTurn(enabled: boolean) {
    this.open = true;
    this.stop();
    this.enabled = enabled;
    this.lead = 0.15;
    if (!enabled) return;
    if (!this.context) {
      try {
        // At the reply's rate each chunk plays as it is; the context
        // resamples once, to the device.
        this.context = new AudioContext({ sampleRate: 24000, latencyHint: 'interactive' });
      } catch {
        this.unavailable = true;
        this.enabled = false;
        return;
      }
    }
    // Also when it reads running: a suspend from the turn before may be
    // under way.
    this.context.resume().catch(() => undefined);
  }

  // The turn is over: no more chunks come.
  endTurn() {
    this.open = false;
    this.suspendIfIdle();
  }

  // True when the browser did not let the audio start.
  get blocked() {
    return this.enabled && this.context !== null && this.context.state !== 'running';
  }

  get playing() {
    return this.sources.size > 0;
  }

  // Queues a chunk and returns the seconds until it is heard, or null when
  // live playback is off.
  push(samples: Float32Array<ArrayBuffer>, sampleRate: number): number | null {
    const context = this.context;
    if (!this.enabled || !context || !samples.length) return null;
    const now = context.currentTime;
    if (!this.frames || sampleRate !== this.rate) {
      this.startTime = now + this.lead;
      this.frames = 0;
      this.rate = sampleRate;
    } else if (this.startTime + this.frames / this.rate < now + 0.02) {
      this.lead = Math.min(1.5, this.lead * 2);
      this.startTime = now + this.lead;
      this.frames = 0;
    }
    const buffer = context.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.onended = () => {
      source.disconnect();
      this.sources.delete(source);
      if (!this.sources.size) {
        this.onchange();
        this.suspendIfIdle();
      }
    };
    const when = this.startTime + this.frames / this.rate;
    source.start(when);
    this.frames += samples.length;
    this.sources.add(source);
    if (this.sources.size === 1) this.onchange();
    return Math.max(0, when - now);
  }

  // Stops what is queued, and the rest of this turn's chunks.
  stop() {
    this.enabled = false;
    this.frames = 0;
    const had = this.sources.size > 0;
    for (const source of this.sources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // Not started yet.
      }
      source.disconnect();
    }
    this.sources.clear();
    if (had) this.onchange();
    this.suspendIfIdle();
  }

  close() {
    this.stop();
    this.context?.close().catch(() => undefined);
    this.context = null;
  }

  private suspendIfIdle() {
    if (!this.open && !this.sources.size && this.context?.state === 'running') {
      this.context.suspend().catch(() => undefined);
    }
  }
}
