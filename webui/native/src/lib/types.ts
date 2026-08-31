export type StringMap = Record<string, string>;

export interface InstallPackageChoice {
  id: string;
  label: string;
  // Button text. `label` names the model as well as the build ("IndexTTS2.5
  // Original-Dtype GGUF"), which is wider than an install button; the model
  // name is already printed beside the buttons, so the short form keeps the
  // build only. It falls back to `label` when that would not be unique.
  short_label: string;
  path: string;
  format: string;
  precision: string;
  session_options?: StringMap;
}

// One rendered install button. The catalog decides which packages an entry
// exposes, so the UI renders every slot it is given rather than re-deriving the
// set from family names.
export interface InstallPackageSlot {
  key: string;
  label: string;
  choice: InstallPackageChoice;
}

export interface CatalogEntry {
  id: string;
  display_name: string;
  display_name_en?: string;
  family: string;
  path: string;
  task: string;
  mode: string;
  download_id?: string;
  install_packages?: InstallPackageChoice[];
  min_vram_gb?: number;
  input_hint?: string;
  input_hint_en?: string;
  default_options?: Record<string, unknown>;
  load_options?: StringMap;
  session_options?: StringMap;
  request_options?: string[];
  required_request_options?: string[];
  builtin_voices?: string[];
  default_voice?: string;
  // Set when a managed entry resolves to no installable package. The entry is
  // still listed so the gap is visible instead of the model disappearing from
  // the UI; `unavailable_reason` explains why it cannot be installed.
  unavailable?: boolean;
  unavailable_reason?: string;
}

export interface ParamSpec {
  name: string;
  type: 'slider' | 'number' | 'bool' | 'text' | 'choice';
  label: string;
  label_en?: string;
  info?: string;
  info_en?: string;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  step?: number;
  choices?: Array<string | number>;
  placeholder?: string;
  placeholder_en?: string;
  lines?: number;
}

export interface LoadedModel {
  id: string;
  family: string;
  task: string;
  mode: string;
  path: string;
  session_options?: StringMap;
  loaded: boolean;
}

export interface ServerHealth {
  status: string;
  backend: string;
  models: number;
  ui: boolean;
  ui_management: boolean;
}

export interface AudioOutput {
  id: string;
  url: string;
}
