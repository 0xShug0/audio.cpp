import type { ModelPackageSize } from '$lib/api';
import type { InstallPackageChoice, ParamSpec } from '$lib/types';

// Fallback mapping from YuE2 model_specs package ids to their GGUF component
// session option and filename when a package row has no load_target yet
// (for example, uninstalled packages in inventory).
const YUE2_PACKAGE_COMPONENTS: Record<string, { option: string; file: string }> = {
  yue2_main_q8_0: { option: 'yue2.model_gguf', file: 'yue2-3b-q8_0.gguf' },
  yue2_main_q4_0: { option: 'yue2.model_gguf', file: 'yue2-3b-q4_0.gguf' },
  yue2_main_bf16: { option: 'yue2.model_gguf', file: 'yue2-3b-bf16.gguf' },
  yue2_vae_f16: { option: 'yue2.vae_gguf', file: 'yue2-vae-f16.gguf' },
  yue2_vae_f32: { option: 'yue2.vae_gguf', file: 'yue2-vae-f32.gguf' }
};

export interface Yue2ComponentChoiceState {
  choices: string[];
  filtered: boolean;
  missing: boolean;
}

function packageWeightForOption(
  choice: InstallPackageChoice,
  row: ModelPackageSize | undefined,
  sessionOption: string
): string | undefined {
  const fromTarget = row?.load_target?.session_options?.[sessionOption] ??
    choice.load_target?.session_options?.[sessionOption] ??
    choice.session_options?.[sessionOption];
  if (fromTarget) return fromTarget;
  const fallback = YUE2_PACKAGE_COMPONENTS[choice.id];
  return fallback?.option === sessionOption ? fallback.file : undefined;
}

export function resolveYue2ComponentChoices(
  spec: ParamSpec,
  installPackages: readonly InstallPackageChoice[],
  packageSizes: Record<string, ModelPackageSize>,
  options: { uiManaged: boolean; residentValue?: string }
): Yue2ComponentChoiceState {
  const staticChoices = (spec.choices || []).map(String);
  const resident = options.residentValue?.trim() || '';
  const withResident = (list: string[]) =>
    resident && !list.includes(resident) ? [...list, resident] : list;

  if (!options.uiManaged) {
    return { choices: withResident(staticChoices), filtered: false, missing: false };
  }

  const sessionOption = spec.session_option || spec.name;
  const componentPackages = installPackages
    .map((pkg) => ({ pkg, file: packageWeightForOption(pkg, packageSizes[pkg.id], sessionOption) }))
    .filter((item): item is { pkg: InstallPackageChoice; file: string } => Boolean(item.file));

  const inventoryKnown = componentPackages.length > 0 &&
    componentPackages.every(({ pkg }) => packageSizes[pkg.id] !== undefined);
  if (!inventoryKnown) {
    return { choices: withResident(staticChoices), filtered: false, missing: false };
  }

  const installedFiles = new Set<string>();
  for (const { pkg, file } of componentPackages) {
    if (packageSizes[pkg.id]?.installed) installedFiles.add(file);
  }

  const ordered: string[] = [];
  for (const choice of staticChoices) {
    if (installedFiles.has(choice)) ordered.push(choice);
  }
  for (const file of installedFiles) {
    if (!ordered.includes(file)) ordered.push(file);
  }

  const choices = withResident(ordered);
  return { choices, filtered: true, missing: choices.length === 0 };
}

export function pickYue2WeightValue(
  currentValue: string,
  spec: ParamSpec,
  availableChoices: readonly string[]
): string {
  if (!availableChoices.length) return '';
  if (currentValue && availableChoices.includes(currentValue)) return currentValue;
  const declaredDefault = spec.default !== undefined ? String(spec.default) : '';
  if (declaredDefault && availableChoices.includes(declaredDefault)) return declaredDefault;
  return availableChoices[0];
}
