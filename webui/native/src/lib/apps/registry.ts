import type { AppDefinition } from './types';

// Each app owns its manifest and component. Adding an app needs no shell edits.
const manifests = import.meta.glob('./*/app.ts', { eager: true, import: 'default' }) as Record<string, AppDefinition>;
export const apps = Object.values(manifests).sort((a, b) => a.name.localeCompare(b.name));
if (new Set(apps.map((app) => app.id)).size !== apps.length) throw new Error('Duplicate app id.');
