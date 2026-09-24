import AuKPanel from './auk/AuKPanel.svelte';
import LiveAvatarPanel from './liveavatar/LiveAvatarPanel.svelte';
import Yue2Panel from './yue2/Yue2Panel.svelte';

export interface GenericControlReplacements {
  packageButtons?: boolean;
  text?: boolean;
  genSource?: boolean;
  language?: boolean;
  seed?: boolean;
  duration?: boolean;
  params?: boolean;
  advancedJson?: boolean;
}

export const modelStudioPanels = {
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

export function modelStudioPanelFor(family?: string) {
  if (!family) return undefined;
  return modelStudioPanels[family as keyof typeof modelStudioPanels];
}
