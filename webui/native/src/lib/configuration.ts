import { writable } from 'svelte/store';

export interface UiConfiguration {
  historyLimit: number;
}

const storageKey = 'audiocpp.ui.configuration';
export const uiConfiguration = writable<UiConfiguration>({ historyLimit: 10 });
export const uiConfigurationError = writable('');

function validate(configuration: UiConfiguration) {
  if (!Number.isSafeInteger(configuration.historyLimit) || configuration.historyLimit < 1) {
    throw new Error('History length must be a positive integer.');
  }
}

export function loadUiConfiguration() {
  const saved = localStorage.getItem(storageKey);
  if (saved === null) return;
  const configuration = JSON.parse(saved) as UiConfiguration;
  validate(configuration);
  uiConfiguration.set({ historyLimit: configuration.historyLimit });
}

export function saveUiConfiguration(configuration: UiConfiguration) {
  validate(configuration);
  localStorage.setItem(storageKey, JSON.stringify(configuration));
  uiConfiguration.set(configuration);
  uiConfigurationError.set('');
}
