export type UiTheme = 'system' | 'dark' | 'light' | 'mocha';
export type ResolvedUiTheme = 'dark' | 'light' | 'mocha';

export const UI_THEME_STORAGE_KEY = 'audiocpp.ui.theme';

export const uiThemes: Array<{ id: UiTheme; label: string }> = [
  { id: 'mocha', label: 'Catppuccin Mocha' },
  { id: 'system', label: 'System' },
  { id: 'dark', label: 'Dark' },
  { id: 'light', label: 'Light' }
];

export function resolveUiTheme(value: string | null | undefined): UiTheme {
  return value === 'dark' || value === 'light' || value === 'system' || value === 'mocha' ? value : 'mocha';
}

export function resolvedTheme(theme: UiTheme, systemPrefersDark: boolean): ResolvedUiTheme {
  if (theme !== 'system') return theme;
  return systemPrefersDark ? 'dark' : 'light';
}
