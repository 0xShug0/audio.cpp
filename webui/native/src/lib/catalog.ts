import rawCatalog from '../../../configs/models_catalog.json';
import rawParams from '../../../configs/model_params.json';
import type { CatalogEntry, ParamSpec } from './types';

export const catalog = (rawCatalog.models as CatalogEntry[]).map((entry) => ({
  ...entry,
  display_name: entry.display_name_en || entry.display_name
}));

export const parameterCatalog = rawParams as unknown as Record<string, ParamSpec[]>;

export const taskLabels: Record<string, string> = {
  tts: 'Text to speech',
  clon: 'Voice cloning',
  asr: 'Transcription',
  gen: 'Music & sound',
  vc: 'Voice conversion',
  svc: 'Singing conversion',
  s2s: 'Speech editing',
  sep: 'Source separation',
  vad: 'Voice activity',
  diar: 'Speaker diarization',
  align: 'Forced alignment',
  vdes: 'Voice design',
  spk: 'Speaker analysis'
};
