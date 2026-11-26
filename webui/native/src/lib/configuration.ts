import { get, writable } from 'svelte/store';

export interface UiConfiguration {
  historyLimit: number;
  autoSave: boolean;
  autoSaveSettings: boolean;
}

const defaultConfiguration: UiConfiguration = {
  historyLimit: 10,
  autoSave: false,
  autoSaveSettings: false
};

const storageKey = 'audiocpp.ui.configuration';
export const uiConfiguration = writable<UiConfiguration>({ ...defaultConfiguration });
export const uiConfigurationError = writable('');

function normalize(raw: Partial<UiConfiguration>): UiConfiguration {
  return {
    historyLimit: raw.historyLimit ?? defaultConfiguration.historyLimit,
    autoSave: raw.autoSave ?? defaultConfiguration.autoSave,
    autoSaveSettings: raw.autoSaveSettings ?? defaultConfiguration.autoSaveSettings
  };
}

function validate(configuration: UiConfiguration) {
  if (!Number.isSafeInteger(configuration.historyLimit) || configuration.historyLimit < 1) {
    throw new Error('History length must be a positive integer.');
  }
  if (typeof configuration.autoSave !== 'boolean') {
    throw new Error('Auto-save setting must be a boolean.');
  }
  if (typeof configuration.autoSaveSettings !== 'boolean') {
    throw new Error('Auto-save settings JSON option must be a boolean.');
  }
}

export function loadUiConfiguration() {
  const saved = localStorage.getItem(storageKey);
  if (saved === null) return;
  const parsed = JSON.parse(saved) as Partial<UiConfiguration>;
  const configuration = normalize(parsed);
  validate(configuration);
  uiConfiguration.set(configuration);
}

export function saveUiConfiguration(configuration: UiConfiguration) {
  const normalized = normalize(configuration);
  validate(normalized);
  localStorage.setItem(storageKey, JSON.stringify(normalized));
  uiConfiguration.set(normalized);
  uiConfigurationError.set('');
}

export function updateUiConfiguration(patch: Partial<UiConfiguration>) {
  saveUiConfiguration({ ...get(uiConfiguration), ...patch });
}
