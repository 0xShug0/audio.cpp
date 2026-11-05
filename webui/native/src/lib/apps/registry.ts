import type { AppDefinition } from './types';

// Each app owns its manifest and component. Adding an app needs no shell edits.
const manifests = import.meta.glob('./*/app.ts', { eager: true, import: 'default' }) as Record<string, AppDefinition>;
const appOrder: Record<string, number> = { tts: 0, 'voice-chat-avatar': 1, 'cross-language-dubbing': 2 };
export const apps = Object.values(manifests).sort((a, b) =>
  (appOrder[a.id] ?? Infinity) - (appOrder[b.id] ?? Infinity) || a.name.localeCompare(b.name));
if (new Set(apps.map((app) => app.id)).size !== apps.length) throw new Error('Duplicate app id.');
