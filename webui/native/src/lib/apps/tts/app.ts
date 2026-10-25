import type { AppDefinition } from '../types';
export default {
  id: 'tts', name: 'Text to speech',
  description: 'Write a script, cast named speakers, refine recordings, and export your narration.',
  tasks: ['tts', 'clon', 'vdes'], modes: ['offline'],
  load: () => import('./TtsApp.svelte')
} satisfies AppDefinition;
