function writeAscii(view: DataView, offset: number, text: string) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

export function encodePcm16Wav(buffer: AudioBuffer): Blob {
  const channels = buffer.numberOfChannels;
  const frames = buffer.length;
  const bytes = new ArrayBuffer(44 + frames * channels * 2);
  const view = new DataView(bytes);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + frames * channels * 2, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, frames * channels * 2, true);

  const samples = Array.from({ length: channels }, (_, channel) => buffer.getChannelData(channel));
  let offset = 44;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, samples[channel][frame]));
      view.setInt16(offset, sample < 0 ? sample * 32768 : sample * 32767, true);
      offset += 2;
    }
  }
  return new Blob([bytes], { type: 'audio/wav' });
}

export async function concatenateAudioBlobs(blobs: Blob[]): Promise<Blob> {
  if (!blobs.length) throw new Error('No audio chunks were generated.');
  if (blobs.length === 1) return blobs[0];

  const context = new AudioContext();
  try {
    const decoded = await Promise.all(blobs.map(async (blob) =>
      context.decodeAudioData(await blob.arrayBuffer())));
    const sampleRate = decoded[0].sampleRate;
    const channels = decoded[0].numberOfChannels;
    for (const chunk of decoded) {
      if (chunk.sampleRate !== sampleRate || chunk.numberOfChannels !== channels) {
        throw new Error('Generated chunks use different audio formats and cannot be joined.');
      }
    }
    const totalFrames = decoded.reduce((sum, chunk) => sum + chunk.length, 0);
    const merged = context.createBuffer(channels, totalFrames, sampleRate);
    let offset = 0;
    for (const chunk of decoded) {
      for (let channel = 0; channel < channels; channel += 1) {
        merged.copyToChannel(chunk.getChannelData(channel), channel, offset);
      }
      offset += chunk.length;
    }
    return encodePcm16Wav(merged);
  } finally {
    await context.close();
  }
}

export async function browserDecodeToWav(file: File, targetSampleRate?: number): Promise<Blob> {
  if (!targetSampleRate && (file.type === 'audio/wav' || file.name.toLowerCase().endsWith('.wav'))) {
    return file;
  }
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    if (!targetSampleRate || decoded.sampleRate === targetSampleRate) {
      return encodePcm16Wav(decoded);
    }
    const frames = Math.max(1, Math.ceil(decoded.duration * targetSampleRate));
    const offline = new OfflineAudioContext(decoded.numberOfChannels, frames, targetSampleRate);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    return encodePcm16Wav(await offline.startRendering());
  } finally {
    await context.close();
  }
}
