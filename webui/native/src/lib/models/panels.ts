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
  yue2: {
    component: Yue2Panel,
    requestMode: 'yue2',
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
