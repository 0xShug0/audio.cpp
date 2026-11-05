import AuKPanel from './auk/AuKPanel.svelte';
import Lfm2AudioPanel from './lfm2_audio/Lfm2AudioPanel.svelte';
import LiveAvatarPanel from './liveavatar/LiveAvatarPanel.svelte';
import Yue2Panel from './yue2/Yue2Panel.svelte';
import Drama3Panel from './fish_drama3/Drama3Panel.svelte';
import D1OmniPanel from './lfm_d1_omni/D1OmniPanel.svelte';
import SoulXFlashHeadPanel from './soulx_flashhead/SoulXFlashHeadPanel.svelte';
import LatentSyncPanel from './latentsync/LatentSyncPanel.svelte';

export interface GenericControlReplacements {
  packageButtons?: boolean;
  text?: boolean;
  genSource?: boolean;
  language?: boolean;
  seed?: boolean;
  duration?: boolean;
  params?: boolean;
  advancedJson?: boolean;
  voice?: boolean;
  chunking?: boolean;
  settingsImport?: boolean;
}

export const modelStudioPanels = {
  lfm_d1_omni: {
    component: D1OmniPanel,
    requestMode: 'panel',
    blocksRunWhileUploading: false,
    replacesGenericControls: { text: true, language: true, voice: true, seed: true, duration: true, chunking: true }
  },
  soulx_flashhead: {
    component: SoulXFlashHeadPanel,
    requestMode: 'panel',
    blocksRunWhileUploading: false,
    replacesGenericControls: { text: true, language: true, duration: true, voice: true, chunking: true }
  },
  latentsync: {
    component: LatentSyncPanel,
    requestMode: 'panel',
    blocksRunWhileUploading: false,
    replacesGenericControls: { text: true, language: true, duration: true, voice: true, chunking: true }
  },
  fish_drama3: {
    component: Drama3Panel,
    requestMode: 'panel',
    blocksRunWhileUploading: false,
    replacesGenericControls: {
      text: true,
      language: true,
      voice: true,
      chunking: true,
      settingsImport: true,
      advancedJson: true
    }
  },
  auk: {
    component: AuKPanel,
    requestMode: 'default',
    blocksRunWhileUploading: false,
    replacesGenericControls: {
      packageButtons: true,
      text: false,
      genSource: false,
      language: true,
      seed: false,
      duration: true,
      params: true,
      advancedJson: false
    }
  },
  lfm2_audio: {
    component: Lfm2AudioPanel,
    // Only speech-to-speech: ASR and TTS keep the generic controls.
    tasks: ['s2s'],
    requestMode: 'panel',
    blocksRunWhileUploading: false,
    replacesGenericControls: {
      packageButtons: false,
      text: false,
      genSource: false,
      language: false,
      seed: false,
      duration: false,
      params: false,
      advancedJson: false
    }
  },
  lfm3_audio: {
    component: Lfm2AudioPanel,
    // Only speech-to-speech, as for lfm2_audio. Its S2S takes no language
    // but auto, so the language field is hidden.
    tasks: ['s2s'],
    requestMode: 'panel',
    blocksRunWhileUploading: false,
    replacesGenericControls: {
      packageButtons: false,
      text: false,
      genSource: false,
      language: true,
      seed: false,
      duration: false,
      params: false,
      advancedJson: false
    }
  },
  liveavatar: {
    component: LiveAvatarPanel,
    requestMode: 'default',
    blocksRunWhileUploading: true,
    replacesGenericControls: {
      packageButtons: false,
      text: false,
      genSource: false,
      language: true,
      seed: false,
      duration: true,
      params: true,
      advancedJson: true
    }
  },
  yue2: {
    component: Yue2Panel,
    requestMode: 'yue2',
    blocksRunWhileUploading: true,
    replacesGenericControls: {
      packageButtons: true,
      text: true,
      genSource: true,
      language: true,
      seed: true,
      duration: true,
      params: true,
      advancedJson: true
    }
  }
};

// A panel entry with a tasks list covers only those tasks of its family;
// one without covers them all.
export function modelStudioPanelFor(family?: string, task?: string) {
  if (!family) return undefined;
  const panel = modelStudioPanels[family as keyof typeof modelStudioPanels];
  const tasks = (panel as { tasks?: readonly string[] } | undefined)?.tasks;
  if (tasks && !tasks.includes(task || '')) return undefined;
  return panel;
}
