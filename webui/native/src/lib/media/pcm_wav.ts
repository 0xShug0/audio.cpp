// PCM WAV chunks emitted by the streaming task API.

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
