import { AudioBufferSource, BufferTarget, Output, WebMOutputFormat, canEncodeAudio } from 'mediabunny';
import { encodePcm16Wav } from '$lib/audio';
import type { SilenceInsertion, AudioCut } from './project';

// Reuse the existing WAV encoder and browser media library. Decode sequentially;
// there is no concurrent generation or worker pool behind audio assembly.
export async function assembleAudio(parts: Array<{ audio: Blob; pause: number }>, sampleRate: number, signal: AbortSignal): Promise<Blob> {
  if (!parts.length) throw new Error('Generate speech before assembling audio.');
  const context = new AudioContext({ sampleRate });
  try {
    const decoded: Array<{ buffer: AudioBuffer; pause: number }> = [];
    for (const part of parts) {
      signal.throwIfAborted();
      if (!Number.isFinite(part.pause) || part.pause < 0) throw new Error('Pauses must be zero or positive.');
      decoded.push({ buffer: await context.decodeAudioData(await part.audio.arrayBuffer()), pause: part.pause });
    }
    const channels = decoded[0].buffer.numberOfChannels;
    if (decoded.some(part => part.buffer.numberOfChannels !== channels)) throw new Error('Chosen takes have different channel counts. Use compatible takes before assembly.');
    const frames = decoded.reduce((sum, part, index) => sum + part.buffer.length +
      (index < decoded.length - 1 ? Math.round(part.pause * sampleRate) : 0), 0);
    const output = context.createBuffer(channels, frames, sampleRate);
    let offset = 0;
    for (const part of decoded) {
      signal.throwIfAborted();
      for (let channel = 0; channel < channels; channel++) output.copyToChannel(part.buffer.getChannelData(channel), channel, offset);
      offset += part.buffer.length + Math.round(part.pause * sampleRate);
    }
    return encodePcm16Wav(output);
  } finally { await context.close(); }
}
export async function finishAudio(audio: Blob, settings: { normalize: boolean; peakDb: number; gainDb: number }, sampleRate: number, signal: AbortSignal): Promise<Blob> {
  if (!Number.isFinite(settings.gainDb) || !Number.isFinite(settings.peakDb) || settings.peakDb > 0) throw new Error('Enter a valid gain and a peak target at or below 0 dBFS.');
  const context = new AudioContext({ sampleRate });
  try {
    const buffer = await context.decodeAudioData(await audio.arrayBuffer());
    signal.throwIfAborted();
    let peak = 0;
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      for (const sample of buffer.getChannelData(channel)) peak = Math.max(peak, Math.abs(sample));
    }
    const gain = settings.normalize && peak > 0 ? Math.pow(10, settings.peakDb / 20) / peak : Math.pow(10, settings.gainDb / 20);
    if (peak * gain > 1.000001) throw new Error('This gain would clip the audio. Lower the gain or enable peak normalization.');
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      const samples = buffer.getChannelData(channel);
      for (let i = 0; i < samples.length; i++) samples[i] *= gain;
    }
    return encodePcm16Wav(buffer);
  } finally { await context.close(); }
}
export async function exportAudio(audio: Blob, format: 'wav' | 'webm', signal: AbortSignal): Promise<Blob> {
  if (format === 'wav') return audio;
  if (!(await canEncodeAudio('opus'))) throw new Error('This browser cannot encode Opus. Choose WAV or use a browser with Opus encoding support.');
  const context = new AudioContext();
  const target = new BufferTarget();
  const output = new Output({ format: new WebMOutputFormat(), target });
  try {
    const buffer = await context.decodeAudioData(await audio.arrayBuffer());
    signal.throwIfAborted();
    const source = new AudioBufferSource({ codec: 'opus', bitrate: 128000 });
    output.addAudioTrack(source);
    await output.start();
    await source.add(buffer);
    source.close();
    signal.throwIfAborted();
    await output.finalize();
    if (!target.buffer) throw new Error('The encoder returned no output.');
    return new Blob([target.buffer], { type: 'audio/webm' });
  } catch (error) { await output.cancel(); throw error; }
  finally { await context.close(); }
}

export function waveformEnvelope(buffer: AudioBuffer, from: number, seconds: number, resolution = 360): Array<[number,number]> {
    const start=Math.round(from*buffer.sampleRate),length=Math.min(Math.round(seconds*buffer.sampleRate),buffer.length-start);
    const bins=Math.min(resolution,length),channels=Array.from({length:buffer.numberOfChannels},(_,c)=>buffer.getChannelData(c));
    return Array.from({length:bins},(_,i)=>{
      let low=0,high=0;
      for(let frame=start+Math.floor(i*length/bins);frame<start+Math.floor((i+1)*length/bins);frame++)for(const samples of channels){low=Math.min(low,samples[frame]);high=Math.max(high,samples[frame]);}
      return [low,high];
    });
  }

export interface AudioEditSegment { start: number; frames: number; sourceFrame: number | null; silenceId?: string }
// Both display and rendering use sample-aligned segments of the original audio.
export function takeAudioSegments(frames: number, rate: number, silences: SilenceInsertion[], removed: AudioCut[] = []): AudioEditSegment[] {
  const inserts=new Map<number,SilenceInsertion[]>();
  for(const edit of silences){
    if(!Number.isFinite(edit.at)||edit.at<0||edit.at>frames/rate||!Number.isFinite(edit.seconds)||edit.seconds<=0)throw new Error('Invalid silence insertion.');
    const at=Math.round(edit.at*rate);inserts.set(at,[...(inserts.get(at)||[]),edit]);
  }
  const cuts=removed.map(c=>{
    if(!Number.isFinite(c.start)||!Number.isFinite(c.end)||c.start<0||c.end>frames/rate||c.end<=c.start)throw new Error('Invalid audio removal.');
    return {start:Math.round(c.start*rate),end:Math.round(c.end*rate)};
  }).sort((a,b)=>a.start-b.start);
  const boundaries=[...new Set([0,frames,...inserts.keys(),...cuts.flatMap(c=>[c.start,c.end])])].sort((a,b)=>a-b);
  const result:AudioEditSegment[]=[];let position=0;
  for(let i=0;i<boundaries.length;i++){
    const at=boundaries[i];
    for(const edit of inserts.get(at)||[]){const count=Math.round(edit.seconds*rate);if(count){result.push({start:position,frames:count,sourceFrame:null,silenceId:edit.id});position+=count;}}
    const end=boundaries[i+1];
    if(end!==undefined && end>at && !cuts.some(c=>c.start<=at && c.end>=end)){result.push({start:position,frames:end-at,sourceFrame:at});position+=end-at;}
  }
  return result;
}
// Every render starts from unchanged original samples, including after undo.
export function renderTakeAudio(context: AudioContext, original: AudioBuffer, silences: SilenceInsertion[], removed: AudioCut[] = []): AudioBuffer {
  const segments=takeAudioSegments(original.length,original.sampleRate,silences,removed);
  const frames=segments.reduce((sum,s)=>sum+s.frames,0);
  if(!frames)throw new Error('The selection removes the entire recording. Leave some audio or inserted silence.');
  const output=context.createBuffer(original.numberOfChannels,frames,original.sampleRate);
  for(const segment of segments)if(segment.sourceFrame!==null)for(let channel=0;channel<original.numberOfChannels;channel++)output.copyToChannel(original.getChannelData(channel).subarray(segment.sourceFrame,segment.sourceFrame+segment.frames),channel,segment.start);
  return output;
}
