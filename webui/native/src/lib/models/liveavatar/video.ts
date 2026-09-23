import {
  AudioBufferSource,
  BufferTarget,
  CanvasSource,
  Mp4OutputFormat,
  Output,
  Quality,
  canEncodeAudio,
  canEncodeVideo
} from 'mediabunny';

interface LiveAvatarVideoMeta {
  width?: string;
  height?: string;
  frames?: string;
  fps?: string;
  format?: string;
}

export interface LiveAvatarOutputArtifact {
  id: string;
  url: string;
  extension: string;
  mime: string;
}

interface LiveAvatarTaskArtifact {
  id?: unknown;
  payload?: unknown;
  meta?: Record<string, string>;
}

export interface LiveAvatarPreparedOutput {
  artifacts: LiveAvatarOutputArtifact[];
  warning: string;
}

function positiveInteger(value: string | undefined, name: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`LiveAvatar video has invalid ${name} metadata`);
  }
  return parsed;
}

function decodeBase64Frame(payload: string, offset: number, length: number): Uint8Array {
  const binary = atob(payload.slice(offset, offset + length));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function rgb24ToRgba(rgb: Uint8Array): Uint8ClampedArray<ArrayBuffer> {
  const rgba = new Uint8ClampedArray(rgb.length / 3 * 4);
  for (let source = 0, target = 0; source < rgb.length; source += 3, target += 4) {
    rgba[target] = rgb[source];
    rgba[target + 1] = rgb[source + 1];
    rgba[target + 2] = rgb[source + 2];
    rgba[target + 3] = 255;
  }
  return rgba;
}

function trimAudio(audioContext: AudioContext, source: AudioBuffer, duration: number): AudioBuffer {
  const frameCount = Math.min(source.length, Math.ceil(duration * source.sampleRate));
  if (frameCount === source.length) return source;
  const trimmed = audioContext.createBuffer(source.numberOfChannels, frameCount, source.sampleRate);
  for (let channel = 0; channel < source.numberOfChannels; channel += 1) {
    trimmed.copyToChannel(source.getChannelData(channel).subarray(0, frameCount), channel);
  }
  return trimmed;
}

export async function encodeLiveAvatarMp4(
  payload: string,
  meta: LiveAvatarVideoMeta,
  drivingAudio: File
): Promise<{ blob: Blob; audioIncluded: boolean }> {
  if (meta.format !== 'rgb24') throw new Error('LiveAvatar video artifact is not RGB24');
  if (typeof VideoEncoder === 'undefined') {
    throw new Error('This browser does not support H.264 encoding through WebCodecs');
  }

  const width = positiveInteger(meta.width, 'width');
  const height = positiveInteger(meta.height, 'height');
  const frames = positiveInteger(meta.frames, 'frame count');
  const fps = positiveInteger(meta.fps, 'frame rate');
  const bytesPerFrame = width * height * 3;
  const base64CharsPerFrame = bytesPerFrame / 3 * 4;
  const expectedPayloadLength = Math.ceil(bytesPerFrame * frames / 3) * 4;
  if (payload.length !== expectedPayloadLength) {
    throw new Error('LiveAvatar RGB24 payload size does not match its video metadata');
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { alpha: false });
  if (!context) throw new Error('Unable to create the LiveAvatar video canvas');

  const audioContext = new AudioContext();
  try {
    const decodedAudio = await audioContext.decodeAudioData(await drivingAudio.arrayBuffer());
    const audio = trimAudio(audioContext, decodedAudio, frames / fps);
    const videoQuality = new Quality('high');
    const audioQuality = new Quality('high');
    if (!await canEncodeVideo('avc', { width, height, quality: videoQuality })) {
      throw new Error('This browser cannot encode H.264 video for LiveAvatar MP4 output');
    }
    const audioConfig = {
      numberOfChannels: audio.numberOfChannels,
      sampleRate: audio.sampleRate,
      quality: audioQuality
    };
    const audioIncluded = typeof AudioEncoder !== 'undefined' && await canEncodeAudio('aac', audioConfig);
    const target = new BufferTarget();
    const output = new Output({
      format: new Mp4OutputFormat({ fastStart: 'in-memory' }),
      target
    });
    const videoSource = new CanvasSource(canvas, {
      codec: 'avc',
      quality: videoQuality,
      keyFrameInterval: 2,
      onEncodedPacket: (_packet, metadata) => {
        const decoderConfig = metadata?.decoderConfig;
        const description = decoderConfig?.description;
        if (metadata && decoderConfig && description !== undefined) {
          const normalizedConfig: VideoDecoderConfig = { ...decoderConfig };
          if (description && typeof description === 'object' && 'byteLength' in description) {
            const source = ArrayBuffer.isView(description)
              ? new Uint8Array(description.buffer, description.byteOffset, description.byteLength)
              : new Uint8Array(description as ArrayBuffer);
            normalizedConfig.description = source.slice().buffer;
          } else {
            delete normalizedConfig.description;
          }
          metadata.decoderConfig = normalizedConfig;
        }
      }
    });
    const audioSource = audioIncluded
      ? new AudioBufferSource({ codec: 'aac', quality: audioQuality })
      : null;
    output.addVideoTrack(videoSource);
    if (audioSource) output.addAudioTrack(audioSource);
    await output.start();

    if (audioSource) await audioSource.add(audio);
    const frameDuration = 1 / fps;
    for (let frame = 0; frame < frames; frame += 1) {
      const rgb = decodeBase64Frame(payload, frame * base64CharsPerFrame, base64CharsPerFrame);
      if (rgb.length !== bytesPerFrame) {
        await output.cancel();
        throw new Error(`LiveAvatar RGB24 frame ${frame} has an invalid size`);
      }
      context.putImageData(new ImageData(rgb24ToRgba(rgb), width, height), 0, 0);
      await videoSource.add(frame * frameDuration, frameDuration);
    }

    await output.finalize();
    if (!target.buffer) throw new Error('LiveAvatar MP4 encoder returned no output');
    return {
      blob: new Blob([target.buffer], { type: 'video/mp4' }),
      audioIncluded
    };
  } finally {
    await audioContext.close();
  }
}

export async function prepareLiveAvatarOutput(
  result: Record<string, unknown>,
  drivingAudio: File | null
): Promise<LiveAvatarPreparedOutput> {
  const artifacts: LiveAvatarOutputArtifact[] = [{
    id: 'liveavatar_result',
    extension: 'json',
    mime: 'application/json',
    url: URL.createObjectURL(new Blob([JSON.stringify(result)], { type: 'application/json' }))
  }];
  const videoArtifact = Array.isArray(result.artifacts)
    ? (result.artifacts as LiveAvatarTaskArtifact[]).find((entry) =>
        typeof entry?.payload === 'string' && entry.meta?.format === 'rgb24')
    : undefined;
  if (!videoArtifact || typeof videoArtifact.payload !== 'string') {
    return { artifacts, warning: 'LiveAvatar returned no RGB24 video artifact.' };
  }
  if (!drivingAudio) {
    return { artifacts, warning: 'LiveAvatar MP4 encoding requires the driving audio.' };
  }

  try {
    const video = await encodeLiveAvatarMp4(
      videoArtifact.payload,
      videoArtifact.meta || {},
      drivingAudio
    );
    artifacts.push({
      id: 'liveavatar_video',
      extension: 'mp4',
      mime: 'video/mp4',
      url: URL.createObjectURL(video.blob)
    });
    return {
      artifacts,
      warning: video.audioIncluded ? '' : 'Browser AAC encoding is unavailable; the MP4 has no audio track.'
    };
  } catch (error) {
    return {
      artifacts,
      warning: `LiveAvatar MP4 encoding failed: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}
