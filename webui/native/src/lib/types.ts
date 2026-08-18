export type StringMap = Record<string, string>;

export interface InstallPackageChoice {
  id: string;
  label: string;
  path: string;
  format: string;
  precision: string;
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
