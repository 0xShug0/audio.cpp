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

// A panel entry with a tasks list covers only those tasks of its family;
// one without covers them all.
export function modelStudioPanelFor(family?: string, task?: string) {
  if (!family) return undefined;
  const panel = modelStudioPanels[family as keyof typeof modelStudioPanels];
  const tasks = (panel as { tasks?: readonly string[] } | undefined)?.tasks;
  if (tasks && !tasks.includes(task || '')) return undefined;
  return panel;
}
