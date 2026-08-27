export type UiTheme = 'system' | 'dark' | 'light';
export type ResolvedUiTheme = 'dark' | 'light';

export const UI_THEME_STORAGE_KEY = 'audiocpp.ui.theme';

export const uiThemes: Array<{ id: UiTheme; label: string }> = [
  { id: 'system', label: 'System' },
  { id: 'dark', label: 'Dark' },
  { id: 'light', label: 'Light' }
];

export function resolveUiTheme(value: string | null | undefined): UiTheme {
  return value === 'dark' || value === 'light' || value === 'system' ? value : 'system';
}

export function resolvedTheme(theme: UiTheme, systemPrefersDark: boolean): ResolvedUiTheme {
  if (theme === 'dark' || theme === 'light') return theme;
  return systemPrefersDark ? 'dark' : 'light';
}
