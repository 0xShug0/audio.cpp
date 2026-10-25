import type { AppDefinition } from '../types';
export default {
  id: 'tts', name: 'Text to speech',
  description: 'Create a narration project. Split your script, refine each take, assemble and finish the audio.',
  tasks: ['tts', 'clon', 'vdes'], modes: ['offline'],
  load: () => import('./TtsApp.svelte')
} satisfies AppDefinition;
