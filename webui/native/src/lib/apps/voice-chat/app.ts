import type { AppDefinition } from '../types';
export default {
  id: 'voice-chat-avatar',
  name: 'Voice chat',
  description: 'Talk with LFM3-Audio in a voice you choose, with an animated portrait.',
  tasks: ['asr', 's2s', 'tts', 'gen'],
  modes: ['offline', 'streaming'],
  load: () => import('./VoiceChat.svelte')
} satisfies AppDefinition;
