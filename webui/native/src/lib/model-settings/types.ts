import type { StringMap } from '$lib/types';

// A reusable execution configuration. No input files, prompt text or editor state.
export interface ModelConfiguration {
  id: string;
  display_name: string;
  family: string;
  task: string;
  mode: string;
  path: string;
  package_id?: string;
  load_options: StringMap;
  session_options: StringMap;
}

export interface ModelSettings {
  version: 1;
  id: string;
  name: string;
  createdAt: number;
  model: ModelConfiguration;
  request: {
    seed: number;
    language?: string;
    instructions?: string;
    duration_seconds?: number;
    max_tokens?: number;
    voice?: string;
    options: Record<string, unknown>;
  };
  chunking: { enabled: boolean; characters: number };
}
