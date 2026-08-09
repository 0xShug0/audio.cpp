import rawCatalog from '../../../configs/models_catalog.json';
import rawParams from '../../../configs/model_params.json';
import type { CatalogEntry, InstallPackageChoice, ParamSpec } from './types';

interface PackageEntry {
  family: string;
  id: string;
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
    request?: Array<{ name: string }>;
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
  if (entry.format === 'safetensors') return 'Safetensors';
  if (entry.precision === 'q8_0' || entry.precision === 'q8') return 'GGUF Q8';
  if (entry.precision === 'bf16') return 'GGUF BF16';
  if (entry.precision === 'f16' || entry.precision === 'fp16') return 'GGUF FP16';
  return `GGUF ${entry.precision.toUpperCase()}`;
}

function packageModelPath(entry: PackageEntry): string {
  const modelFile = entry.format === 'gguf'
    ? entry.files?.find((file) => file.toLowerCase().endsWith('.gguf'))
    : undefined;
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
  const q8 = preferredPackage(related.filter((candidate) =>
    candidate.format === 'gguf' && ['q8_0', 'q8'].includes(candidate.precision)));
  const fp16 = preferredPackage(related.filter((candidate) =>
    candidate.format === 'gguf' && ['f16', 'fp16'].includes(candidate.precision))) ||
    preferredPackage(related.filter((candidate) =>
      candidate.format === 'gguf' && candidate.precision === 'bf16'));
  const otherGguf = !q8 && !fp16
    ? preferredPackage(related.filter((candidate) => candidate.format === 'gguf'))
    : undefined;
  const safetensors = preferredPackage(related.filter((candidate) => candidate.format === 'safetensors'));
  return [q8, fp16, otherGguf, safetensors]
    .filter((candidate): candidate is PackageEntry => candidate !== undefined)
    .map((candidate) => ({
      id: candidate.id,
      label: packageLabel(candidate),
      path: packageModelPath(candidate),
      format: candidate.format,
      precision: candidate.precision
    }));
}

export const catalog = (rawCatalog.models as CatalogEntry[]).map((entry) => {
  const choices = installChoices(entry);
  const installPackage = choices[0];
  const spec = specsByFamily.get(entry.family);
  return {
    ...entry,
    display_name: entry.display_name_en || entry.display_name,
    download_id: installPackage?.id || entry.download_id,
    install_packages: choices,
    path: installPackage?.path || entry.path,
    request_options: spec?.options?.request?.map((option) => option.name),
    builtin_voices: spec?.ui?.builtin_voices,
    default_voice: spec?.ui?.default_voice
  };
});

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
