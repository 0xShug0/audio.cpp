import rawCatalog from '../../../configs/models_catalog.json';
import rawParams from '../../../configs/model_params.json';
import type {
  CatalogEntry,
  InstallPackageChoice,
  InstallPackageSlot,
  ParamSpec
} from './types';

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
    recommended_package?: string;
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

// Package ids carry a trailing format/precision tag. Stripping it yields the
// "stem" that identifies the model itself, so a catalog download id written
// before (or after) a requantisation can still be matched to its package. Keep
// longer tags ahead of their prefixes (`q8_0` before `q8`, `q4_0` before `q4`)
// so the alternation strips the whole tag. This list is the single source of
// truth for suffix stripping anywhere in this module.
const packageIdSuffix =
  /_(?:q2_k|q3_k|q4_0|q4_k|q4|q5_0|q5_k|q6_k|q8_0|q8|f16|fp16|bf16|f32|safetensors|orig)$/i;

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

// The id that identifies the model behind a package, independent of how that
// package was quantised.
const packageIdStem = (value: string) => cleanId(value.replace(packageIdSuffix, ''));

const isGguf = (entry: PackageEntry) => entry.format === 'gguf';

const familyPackages = (family: string) =>
  packages.filter((candidate) => candidate.family === family);

const recommendedPackageId = (family: string) =>
  specsByFamily.get(family)?.ui?.recommended_package;

// Picks one package out of a candidate set. `ui.recommended_package` is the
// spec's own answer to "which build should a new user get", so it outranks the
// structural fallbacks; `default` and q8_0 remain for candidate sets that do
// not contain the recommendation (a second model in the same family).
function preferredPackage(entries: PackageEntry[], recommended?: string): PackageEntry | undefined {
  return (recommended ? entries.find((entry) => entry.id === recommended) : undefined) ||
    entries.find((entry) => entry.default) ||
    entries.find((entry) => entry.precision === 'q8_0') ||
    entries[0];
}

// Resolves the single package a catalog entry points at. Precedence, most
// specific first:
//   1. an exact package id — the catalog names the package outright;
//   2. the same id stem — the catalog names the model but a stale (or absent)
//      precision tag, e.g. `omnivoice` -> `omnivoice_q8_0`. Stems are compared
//      for equality rather than by prefix on purpose: a prefix test makes
//      `index_tts2` swallow the separate `index_tts2_5_*` model;
//   3. a GGUF package installing into the catalog `path`. Restricted to GGUF
//      because a legacy path such as models/pocket-tts names the upstream
//      Safetensors drop, not the GGUF build the entry actually wants;
//   4. the family's `ui.recommended_package`, for entries whose download id is
//      a bare family name that never matched a package;
//   5. anything GGUF in the family, so a not-yet-migrated entry still resolves.
// Steps 3-5 exist only for catalog entries that have not been migrated to real
// package ids; steps 1-2 cover every migrated entry.
function resolvedPackage(entry: CatalogEntry): PackageEntry | undefined {
  const family = familyPackages(entry.family);
  if (!family.length) return undefined;
  const recommended = recommendedPackageId(entry.family);
  if (entry.download_id) {
    const exact = family.find((candidate) => candidate.id === entry.download_id);
    if (exact) return exact;

    const stem = packageIdStem(entry.download_id);
    const stemMatches = family.filter((candidate) => packageIdStem(candidate.id) === stem);
    if (stemMatches.length) {
      return preferredPackage(stemMatches.filter(isGguf), recommended) ||
        preferredPackage(stemMatches, recommended);
    }

    const target = cleanPath(entry.path);
    const targetMatches = family.filter((candidate) =>
      isGguf(candidate) && cleanPath(candidate.target_directory) === target);
    if (targetMatches.length) return preferredPackage(targetMatches, recommended);
  }
  const recommendedMatch = recommended
    ? family.find((candidate) => candidate.id === recommended)
    : undefined;
  if (recommendedMatch) return recommendedMatch;
  return preferredPackage(family.filter(isGguf), recommended) ||
    preferredPackage(family, recommended);
}

// Packages that are precision variants of one model share a target directory;
// packages that are different models install into different ones. Grouping on
// that field is what the old hand-maintained family allowlist was approximating.
function relatedPackages(entry: CatalogEntry): PackageEntry[] {
  const resolved = resolvedPackage(entry);
  if (!resolved) return [];
  return familyPackages(entry.family).filter((candidate) =>
    candidate.target_directory === resolved.target_directory);
}

// Every quantised precision, kept in step with the tags in `packageIdSuffix` so
// a newly published quantisation sorts with its peers instead of falling
// through to the unknown bucket behind `orig`.
const quantisedPrecisions = [
  'q2_k', 'q3_k', 'q4', 'q4_0', 'q4_k', 'q5_0', 'q5_k', 'q6_k', 'q8_0', 'q8'
];

function exposedPackageRank(entry: PackageEntry, selectedId?: string): number {
  if (selectedId && entry.id === selectedId) return 0;
  if (entry.default) return 1;
  if (quantisedPrecisions.includes(entry.precision)) return 2;
  if (['f16', 'fp16', 'bf16'].includes(entry.precision)) return 3;
  if (entry.precision === 'f32') return 4;
  if (entry.precision === 'orig') return 5;
  return 6;
}

// Derived label, used only when a package has no display_name. It describes the
// build, not the model, so it cannot tell two models in one target directory
// apart — see packageLabel.
function derivedPackageLabel(entry: PackageEntry): string {
  if (entry.format === 'safetensors') return 'Safetensors';
  if (entry.id.includes('int8_dit')) return 'GGUF Q4 ConvRot';
  if (entry.precision === 'q4_k' || entry.precision === 'q4_0') return 'GGUF Q4';
  if (entry.precision === 'q8_0' || entry.precision === 'q8') return 'GGUF Q8';
  if (entry.precision === 'bf16') return 'GGUF BF16';
  if (entry.precision === 'f16' || entry.precision === 'fp16') return 'GGUF FP16';
  return `GGUF ${entry.precision.toUpperCase()}`;
}

// display_name is unique per package and already distinguishes both the model
// and its precision ("ACE-Step 1.5 XL Turbo BF16 GGUF"), which the derived
// label cannot: a target directory may hold several distinct models, and those
// would otherwise all render as "GGUF Q8".
function packageLabel(entry: PackageEntry): string {
  return englishUiText(entry.display_name) || derivedPackageLabel(entry);
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

function packageSessionOptions(entry: PackageEntry): Record<string, string> | undefined {
  if (entry.family !== 'minimax_music3') return undefined;
  if (entry.id === 'minimax_music3_q8_0') {
    return {
      'minimax_music3.language_model_gguf': 'language_model_q8_0.gguf',
      'minimax_music3.rvq_depth_decoder_gguf': 'rvq_depth_decoder_q8_0.gguf',
      'minimax_music3.flow_transformer_gguf': 'transformer_q8_0.gguf'
    };
  }
  if (entry.id === 'minimax_music3_bf16') {
    return {
      'minimax_music3.language_model_gguf': 'language_model_bf16.gguf',
      'minimax_music3.rvq_depth_decoder_gguf': 'rvq_depth_decoder_bf16.gguf',
      'minimax_music3.flow_transformer_gguf': 'transformer_bf16.gguf'
    };
  }
  if (entry.id === 'minimax_music3_q4_0') {
    return {
      'minimax_music3.language_model_gguf': 'language_model_q4_0.gguf',
      'minimax_music3.rvq_depth_decoder_gguf': 'rvq_depth_decoder_q8_0.gguf',
      'minimax_music3.flow_transformer_gguf': 'transformer_q4_0.gguf'
    };
  }
  return undefined;
}

// The build alone, without the model name or the "GGUF" every exposed package
// shares. Install buttons are around 90px wide, so the full display_name
// overflowed them; the precision is what actually distinguishes the buttons.
function shortPackageLabel(entry: PackageEntry): string {
  if (entry.format !== 'gguf') return derivedPackageLabel(entry);
  const precision = entry.precision === 'orig' ? 'Original' : entry.precision.toUpperCase();
  return entry.id.includes('int8_dit') ? `${precision} ConvRot` : precision;
}

const installChoice = (candidate: PackageEntry, shortLabel: string): InstallPackageChoice => ({
  id: candidate.id,
  label: packageLabel(candidate),
  short_label: shortLabel,
  path: packageModelPath(candidate),
  format: candidate.format,
  precision: candidate.precision,
  session_options: packageSessionOptions(candidate)
});

function installChoices(entry: CatalogEntry): InstallPackageChoice[] {
  // The native model manager intentionally exposes complete GGUF packages
  // only. Safetensors packages frequently depend on source-tree sidecars and
  // are not yet reliable as one-click UI installs. Every other build of the
  // resolved model is offered: they share a target directory, so switching
  // between them is a requantisation rather than a different model.
  const exposed = relatedPackages(entry).filter(isGguf);
  if (!exposed.length) return [];
  // An entry that names a package explicitly gets that package, even when the
  // family recommends another build of it: the catalog id is the per-entry
  // decision (a deliberate low-VRAM pick, say) and the recommendation is only
  // the family-wide default for entries that express no preference.
  const explicit = exposed.find((candidate) => candidate.id === entry.download_id);
  const selected = explicit || preferredPackage(exposed, recommendedPackageId(entry.family));
  const ordered = exposed
    .slice()
    .sort((left, right) =>
      exposedPackageRank(left, selected?.id) - exposedPackageRank(right, selected?.id));
  // Two distinct models can share a target directory, and the short label
  // describes the build only, so it cannot tell those apart. Keep the full
  // display_name on every button of such a set rather than render one
  // ambiguous button.
  const labels = distinctLabels(ordered.map(shortPackageLabel)) ||
    distinctLabels(trimmedPackageLabels(ordered.map(packageLabel))) ||
    ordered.map(packageLabel);
  return ordered.map((candidate, index) => installChoice(candidate, labels[index]));
}

function distinctLabels(labels: string[]): string[] | undefined {
  const seen = new Set(labels.map((value) => value.toLowerCase()));
  return seen.size === labels.length && labels.every(Boolean) ? labels : undefined;
}

// Second try when precisions alone collide, which happens when one target
// directory holds several models ("ACE-Step 1.5 Turbo BF16 GGUF" beside
// "ACE-Step 1.5 XL SFT BF16 GGUF"). Drop the leading words every package of the
// set shares and the trailing format word, leaving what actually differs.
function trimmedPackageLabels(labels: string[]): string[] {
  if (labels.length < 2) return labels;
  const words = labels.map((label) => label.split(' ').filter(Boolean));
  let shared = 0;
  while (words.every((entry) => shared < entry.length - 1 && entry[shared] === words[0][shared])) {
    shared++;
  }
  return words.map((entry) => {
    const rest = entry.slice(shared);
    if (rest.length > 1 && rest[rest.length - 1].toLowerCase() === 'gguf') rest.pop();
    return rest.join(' ');
  });
}

// Why a managed entry ended up with no installable package. Returned as display
// text so the entry can stay listed instead of vanishing from the whole UI.
function unavailableReason(entry: CatalogEntry): string {
  const family = familyPackages(entry.family);
  if (!family.length) {
    return `No model_specs package definition for family "${entry.family}".`;
  }
  return 'Only Safetensors packages are published for this model. The native ' +
    'model manager installs GGUF packages only.';
}

export const catalog = (rawCatalog.models as CatalogEntry[]).map((entry) => {
  const choices = installChoices(entry);
  // A managed catalog entry with no remaining GGUF choice is Safetensors-only
  // (or has no spec yet). Keep it listed and flag it: dropping it here made a
  // packaging gap look like the model had never existed. Entries without a
  // download id are bundled or locally managed and are never flagged.
  const unavailable = Boolean(entry.download_id) && choices.length === 0;
  const installPackage = choices[0];
  const spec = specsByFamily.get(entry.family);
  return {
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
    default_voice: spec?.ui?.default_voice,
    unavailable: unavailable || undefined,
    unavailable_reason: unavailable ? unavailableReason(entry) : undefined
  };
});

// The install buttons an entry renders. Every exposed package gets its own
// slot: catalog.ts already decided which packages belong together, so the UI
// must not re-derive that from family names or collapse the list to fixed
// q8/fp16 slots.
export function installPackageSlots(entry: CatalogEntry): InstallPackageSlot[] {
  return (entry.install_packages || []).map((choice) => ({
    key: choice.id,
    label: choice.short_label,
    choice
  }));
}

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
