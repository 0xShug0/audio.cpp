// Studio playback uses the audio clock for both streamed WAVs and complete files.
// Keep the context alive between requests so a Generate click unlocks later audio.
export class StreamPlayback {
  private context: AudioContext | null = null;
  private sources = new Set<AudioBufferSourceNode>();
  private nextTime = 0;
  private generation = 0;
  private enabled = false;
  onchange = () => {};

  get playing() { return this.sources.size > 0; }

  begin(): Promise<void> {
    this.stop();
    this.enabled = true;
    this.context ??= new AudioContext({ latencyHint: 'interactive' });
    return this.context.resume();
  }

  async push(data: ArrayBuffer): Promise<void> {
    const context = this.context;
    const generation = this.generation;
    if (!context || !this.enabled) return;
    const buffer = await context.decodeAudioData(data);
    if (generation !== this.generation) return;
    if (context.state !== 'running') throw new Error('Playback was blocked. Use the output player to listen.');
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.onended = () => {
      source.disconnect();
      this.sources.delete(source);
      this.onchange();
    };
    this.nextTime = Math.max(this.nextTime, context.currentTime + 0.04);
    source.start(this.nextTime);
    this.nextTime += buffer.duration;
    this.sources.add(source);
    this.onchange();
  }

  stop() {
    this.enabled = false;
    ++this.generation;
    for (const source of this.sources) {
      source.onended = null;
      source.stop();
      source.disconnect();
    }
    this.sources.clear();
    this.nextTime = 0;
    this.onchange();
  }

  close() {
    this.stop();
    void this.context?.close();
    this.context = null;
  }
}
