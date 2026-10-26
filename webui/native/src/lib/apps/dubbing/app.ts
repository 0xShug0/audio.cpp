import type { AppDefinition } from '../types';

export default {
  id: 'cross-language-dubbing',
  name: 'Cross-language dubbing',
  description: 'Translate a recording and generate speech in the target language.',
  tasks: ['asr', 'tts'],
  modes: ['offline'],
  load: () => import('./DubbingApp.svelte')
} satisfies AppDefinition;
