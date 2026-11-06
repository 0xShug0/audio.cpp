import { decodeBase64Frame, rgb24ToRgba } from '$lib/media/rgb24_video';

export interface VideoChunk { payload: string; meta: Record<string, string> }
interface VideoFrameBlock { video: VideoChunk; frames: number; width: number; height: number; fps: number }

// Audio is the clock. Each video block consumes matching streamed PCM, not a complete reply.
export class AvatarPlayback {
  private queue: (VideoFrameBlock & { start: number; end: number })[] = [];
  private videos: VideoFrameBlock[] = [];
  private audio: { samples: Float32Array<ArrayBuffer>; offset: number }[] = [];
  private sources = new Set<AudioBufferSourceNode>();
  private rate = 0;
  private buffered = 0;
  private received = 0;
  private scheduled = 0;
  private nextStart = 0;
  private animation = 0;
  private lastFrame = '';
  private stopped = false;
  private finish: (() => void) | null = null;
  private complete = false;
  private drawing: CanvasRenderingContext2D;
  readonly done = new Promise<void>(resolve => { this.finish = resolve; });

  constructor(private context: AudioContext, private canvas: HTMLCanvasElement,
    private speaking: (value: boolean) => void) {
    const drawing = canvas.getContext('2d', { alpha: false });
    if (!drawing) throw new Error('Canvas rendering is unavailable.');
    this.drawing = drawing;
  }
  get duration() { return this.rate ? this.received / this.rate : 0; }

  addAudio(samples: Float32Array<ArrayBuffer>, rate: number) {
    if (this.stopped) return;
    if (this.rate && this.rate !== rate) throw new Error('Reply audio changed sample rate.');
    this.rate = rate;
    this.received += samples.length;
    this.buffered += samples.length;
    if (samples.length) this.audio.push({ samples, offset: 0 });
    this.schedule();
  }

  add(video: VideoChunk) {
    if (this.stopped) return;
    const { meta } = video;
    const width = Number(meta.width), height = Number(meta.height), frames = Number(meta.frames), fps = Number(meta.fps);
    if (meta.format !== 'rgb24' || ![width, height, frames].every(x => Number.isSafeInteger(x) && x > 0) || !(fps > 0))
      throw new Error('The avatar returned invalid video metadata.');
    if (video.payload.length !== Math.ceil(width * height * 3 * frames / 3) * 4)
      throw new Error('The avatar returned an incomplete video chunk.');
    this.videos.push({ video, width, height, frames, fps });
    this.schedule();
  }

  private schedule() {
    if (!this.rate) return;
    while (this.videos.length) {
      const video = this.videos[0];
      const wanted = Math.round(video.frames / video.fps * this.rate);
      if (!this.complete && this.buffered < wanted) break;
      const count = Math.min(wanted, this.buffered);
      this.videos.shift();
      if (!count) continue; // Final avatar padding beyond the speech is not played.
      const buffer = this.context.createBuffer(1, count, this.rate);
      const pcm = buffer.getChannelData(0);
      for (let offset = 0; offset < count;) {
        const chunk = this.audio[0];
        const take = Math.min(count - offset, chunk.samples.length - chunk.offset);
        pcm.set(chunk.samples.subarray(chunk.offset, chunk.offset + take), offset);
        offset += take; chunk.offset += take;
        if (chunk.offset === chunk.samples.length) this.audio.shift();
      }
      this.buffered -= count; this.scheduled += count;
      const start = Math.max(this.context.currentTime + this.context.baseLatency, this.nextStart);
      const end = start + count / this.rate;
      const source = this.context.createBufferSource();
      source.buffer = buffer; source.connect(this.context.destination);
      source.onended = () => { source.disconnect(); this.sources.delete(source); };
      source.start(start);
      this.sources.add(source);
      this.queue.push({ ...video, start, end });
      this.nextStart = end;
    }
    if (!this.animation && this.queue.length) this.draw();
  }

  end() {
    this.complete = true;
    this.schedule();
    if (!this.scheduled) throw new Error('The avatar returned no synchronized audio/video.');
    if (this.buffered > this.rate * .04) throw new Error('The avatar video ended before the reply audio.');
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    cancelAnimationFrame(this.animation); this.animation = 0;
    for (const source of this.sources) { source.onended = null; source.stop(); source.disconnect(); }
    this.sources.clear(); this.queue = []; this.videos = []; this.audio = [];
    this.speaking(false); this.finish?.();
  }

  private draw = () => {
    if (this.stopped) return;
    const now = this.context.currentTime;
    while (this.queue.length && this.queue[0].end <= now) this.queue.shift();
    const chunk = this.queue[0];
    this.speaking(Boolean(chunk && now >= chunk.start));
    if (chunk && now >= chunk.start) {
      const frame = Math.min(chunk.frames - 1, Math.floor((now - chunk.start) * chunk.fps));
      const key = `${chunk.start}:${frame}`;
      if (key !== this.lastFrame) {
        if (this.canvas.width !== chunk.width || this.canvas.height !== chunk.height) {
          this.canvas.width = chunk.width; this.canvas.height = chunk.height;
        }
        const size = chunk.width * chunk.height * 4;
        const rgb = decodeBase64Frame(chunk.video.payload, frame * size, size);
        this.drawing.putImageData(new ImageData(rgb24ToRgba(rgb), chunk.width, chunk.height), 0, 0);
        this.lastFrame = key;
      }
    }
    if (this.complete && !this.queue.length) { this.stop(); return; }
    this.animation = requestAnimationFrame(this.draw);
  };
}
