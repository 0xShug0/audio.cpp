import rawCatalog from '../../../configs/models_catalog.json';
import rawParams from '../../../configs/model_params.json';
import type { CatalogEntry, InstallPackageChoice, ParamSpec } from './types';

interface PackageEntry {
  family: string;
  id: string;
  display_name?: string;
  target_directory: string;
  format: string;
  precision: string;
  files?: string[];
  strip_prefix?: string;
  default?: boolean;
}

interface PackageSpec {
  family: string;
  packages?: Array<Omit<PackageEntry, 'family'>>;
  options?: {
    request?: Array<{ name: string; required?: boolean }>;
  };
  ui?: {
    builtin_voices?: string[];
    default_voice?: string;
  };
}

const specModules = import.meta.glob('../../../../model_specs/*.json', {
  eager: true,
  import: 'default'
}) as Record<string, PackageSpec>;

// Package ids and install locations are sourced from model_specs at frontend
// build time. This keeps the embedded catalog aligned when package ids gain a
// precision/format suffix, without needing model_specs files at UI runtime.
const packages: PackageEntry[] = Object.values(specModules).flatMap((spec) =>
  (spec.packages || []).map((entry) => ({ ...entry, family: spec.family }))
);

const specsByFamily = new Map(Object.values(specModules).map((spec) => [spec.family, spec]));

const hanCharacters = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/u;

function englishUiText(preferred?: string, fallback?: string): string {
  for (const value of [preferred, fallback]) {
    if (value && !hanCharacters.test(value)) return value;
  }
  return '';
}

function parameterLabel(name: string, preferred?: string, fallback?: string): string {
  return englishUiText(preferred, fallback) || name.replace(/_/g, ' ');
}

const cleanPath = (value: string) => value
  .replace(/\\/g, '/')
  .replace(/^\.\//, '')
  .replace(/^models\//i, '')
  .replace(/\/$/, '')
  .toLowerCase();

const cleanId = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, '');

function preferredPackage(entries: PackageEntry[]): PackageEntry | undefined {
  return entries.find((entry) => entry.default) ||
    entries.find((entry) => entry.precision === 'q8_0') ||
    entries[0];
}

function relatedPackages(entry: CatalogEntry): PackageEntry[] {
  const family = packages.filter((candidate) => candidate.family === entry.family);
  if (!family.length) return [];
  if (entry.family === 'ace_step') return family;
  if (!entry.download_id) return family;
  const exact = family.find((candidate) => candidate.id === entry.download_id);
  if (exact) {
    const stem = exact.id.replace(/_(?:q8_0|q8|f16|fp16|bf16|safetensors|orig)$/i, '');
    const matches = family.filter((candidate) =>
      candidate.id.replace(/_(?:q8_0|q8|f16|fp16|bf16|safetensors|orig)$/i, '') === stem);
    return matches.length ? matches : [exact];
  }

  // Resolve a legacy family/variant id before considering its old target
  // directory. A directory such as models/pocket-tts may identify the gated
  // upstream safetensors package, while the family default is the public GGUF
  // package intended by the catalog's generic `pocket_tts` download id.
  const legacyId = cleanId(entry.download_id);
  const legacyMatches = family.filter((candidate) => {
    const currentId = cleanId(candidate.id);
    return currentId.startsWith(legacyId) || legacyId.startsWith(currentId);
  });
  if (legacyMatches.length) return legacyMatches;

  const target = cleanPath(entry.path);
  const targetMatches = family.filter((candidate) => cleanPath(candidate.target_directory) === target);
  if (targetMatches.length) {
    const stems = new Set(targetMatches.map((candidate) =>
      candidate.id.replace(/_(?:q8_0|q8|f16|fp16|bf16|safetensors|orig)$/i, '')));
    const matches = family.filter((candidate) => stems.has(
      candidate.id.replace(/_(?:q8_0|q8|f16|fp16|bf16|safetensors|orig)$/i, '')));
    return matches.length ? matches : targetMatches;
  }
  return family;
}

function packageLabel(entry: PackageEntry): string {
  if (entry.family === 'ace_step') {
    const precision = entry.precision === 'bf16'
      ? 'BF16'
      : ['q8_0', 'q8'].includes(entry.precision)
        ? 'Q8'
        : entry.precision.toUpperCase();
    if (entry.id.includes('_xl_turbo_')) return `GGUF Turbo XL ${precision}`;
    if (entry.id.includes('_xl_sft_')) return `GGUF Turbo XL SFT ${precision}`;
    if (entry.id.includes('_turbo_')) return `GGUF Turbo ${precision}`;
    return `GGUF ${precision}`;
  }
  if (entry.format === 'safetensors') return 'Safetensors';
  if (entry.id.includes('int8_dit')) return 'GGUF Q4 ConvRot';
  if (entry.precision === 'q4_k' || entry.precision === 'q4_0') return 'GGUF Q4';
  if (entry.precision === 'q8_0' || entry.precision === 'q8') return 'GGUF Q8';
  if (entry.precision === 'bf16') return 'GGUF BF16';
  if (entry.precision === 'f16' || entry.precision === 'fp16') return 'GGUF FP16';
  return `GGUF ${entry.precision.toUpperCase()}`;
}

function packageModelPath(entry: PackageEntry): string {
  let modelFile: string | undefined;
  if (entry.format === 'gguf' && entry.family === 'minimax_h3') {
    const entryName = entry.id.includes('int8_dit') ? 'dit_int8.gguf' : 'dit.gguf';
    modelFile = entry.files?.find((file) => file.toLowerCase().endsWith(`/${entryName}`));
  } else if (entry.format === 'gguf' && entry.family === 'minimax_music3') {
    return `models/${entry.target_directory}`;
  } else if (entry.format === 'gguf') {
    modelFile = entry.files?.find((file) => file.toLowerCase().endsWith('.gguf'));
  }
  if (!modelFile) {
    return `models/${entry.target_directory}`;
  }
  let relative = modelFile.replace(/\\/g, '/');
  const prefix = (entry.strip_prefix || '').replace(/\\/g, '/').replace(/\/$/, '');
  if (prefix && relative.startsWith(`${prefix}/`)) relative = relative.slice(prefix.length + 1);
  return `models/${entry.target_directory}/${relative}`.replace(/\/+/g, '/');
}

function installChoices(entry: CatalogEntry): InstallPackageChoice[] {
  const related = relatedPackages(entry);
  if (entry.family === 'ace_step') {
    return related
      .filter((candidate) => candidate.format === 'gguf')
      .map((candidate) => ({
        id: candidate.id,
        label: packageLabel(candidate),
        path: packageModelPath(candidate),
        format: candidate.format,
        precision: candidate.precision
      }));
  }
  const q8 = preferredPackage(related.filter((candidate) =>
    candidate.format === 'gguf' && ['q8_0', 'q8'].includes(candidate.precision)));
  const fp16 = preferredPackage(related.filter((candidate) =>
    candidate.format === 'gguf' && ['f16', 'fp16'].includes(candidate.precision))) ||
    preferredPackage(related.filter((candidate) =>
      candidate.format === 'gguf' && candidate.precision === 'bf16'));
  const otherGguf = !q8 && !fp16
    ? related.filter((candidate) => candidate.format === 'gguf')
    : [];
  // The native model manager intentionally exposes complete GGUF packages
  // only. Safetensors packages frequently depend on source-tree sidecars and
  // are not yet reliable as one-click UI installs.
  return [q8, fp16, ...otherGguf]
    .filter((candidate): candidate is PackageEntry => candidate !== undefined)
    .map((candidate) => ({
      id: candidate.id,
      label: packageLabel(candidate),
      path: packageModelPath(candidate),
      format: candidate.format,
      precision: candidate.precision
    }));
}

export const catalog = (rawCatalog.models as CatalogEntry[]).flatMap((entry) => {
  const choices = installChoices(entry);
  // A managed catalog entry with no remaining GGUF choice is Safetensors-only
  // (or otherwise not installable by the native manager). Do not expose it as
  // an apparently available Studio model after Safetensors UI support is
  // disabled. Entries without a download id are bundled or locally managed
  // and must remain visible.
  if (entry.download_id && choices.length === 0) return [];
  const installPackage = choices[0];
  const spec = specsByFamily.get(entry.family);
  return [{
    ...entry,
    display_name: englishUiText(entry.display_name_en, entry.display_name) || entry.id,
    input_hint: englishUiText(entry.input_hint_en, entry.input_hint),
    download_id: installPackage?.id || entry.download_id,
    install_packages: choices,
    path: installPackage?.path || entry.path,
    request_options: spec?.options?.request?.map((option) => option.name),
    required_request_options: spec?.options?.request
      ?.filter((option) => option.required === true)
      .map((option) => option.name),
    builtin_voices: spec?.ui?.builtin_voices,
    default_voice: spec?.ui?.default_voice
  }];
});

export const parameterCatalog = Object.fromEntries(
  Object.entries(rawParams as unknown as Record<string, ParamSpec[] | string>)
    .filter((entry): entry is [string, ParamSpec[]] => Array.isArray(entry[1]))
    .map(([family, specs]) => [
      family,
      specs.map((spec) => ({
        ...spec,
        label: parameterLabel(spec.name, spec.label_en, spec.label),
        placeholder: englishUiText(spec.placeholder_en, spec.placeholder),
        info: englishUiText(spec.info_en, spec.info)
      }))
    ])
) as Record<string, ParamSpec[]>;

export const taskLabels: Record<string, string> = {
  tts: 'Text to speech',
  clon: 'Voice cloning',
  asr: 'Transcription',
  gen: 'Music & sound',
  midi: 'Audio to MIDI',
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
