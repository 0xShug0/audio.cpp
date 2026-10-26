import type { ComponentType, SvelteComponent } from 'svelte';

export interface AppProps {
  busy?: boolean;
  active?: boolean;
}

export interface AppDefinition {
  id: string;
  name: string;
  description: string;
  tasks: readonly string[];
  modes: readonly string[];
  load: () => Promise<{ default: ComponentType<SvelteComponent<AppProps>> }>;
}
