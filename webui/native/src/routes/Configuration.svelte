<script lang="ts">
  import {
    autoSaveDirectory,
    clearAutoSaveDirectory,
    pickAutoSaveDirectory,
    requestAutoSaveDirectoryPermission
  } from '$lib/auto_save';
  import { saveUiConfiguration, uiConfiguration, uiConfigurationError } from '$lib/configuration';
  import type { Translator } from '$lib/i18n';

  export let tr: Translator;

  let historyLimit = $uiConfiguration.historyLimit;
  let autoSave = $uiConfiguration.autoSave;
  let autoSaveSettings = $uiConfiguration.autoSaveSettings;
  let status = '';
  let folderError = '';
  let folderBusy = false;

  $: historyLimit = $uiConfiguration.historyLimit;
  $: autoSave = $uiConfiguration.autoSave;
  $: autoSaveSettings = $uiConfiguration.autoSaveSettings;

  function save() {
    try {
      saveUiConfiguration({ historyLimit, autoSave, autoSaveSettings });
      status = 'configuration.saved';
    } catch (error) {
      uiConfigurationError.set(error instanceof Error ? error.message : String(error));
      status = '';
    }
  }

  async function chooseFolder() {
    folderBusy = true;
    folderError = '';
    try {
      await pickAutoSaveDirectory();
    } catch (error) {
      folderError = error instanceof Error ? error.message : String(error);
    } finally {
      folderBusy = false;
    }
  }

  async function grantAccess() {
    folderBusy = true;
    folderError = '';
    try {
      const granted = await requestAutoSaveDirectoryPermission();
      if (!granted) {
        folderError = 'Folder write permission was not granted.';
      }
    } catch (error) {
      folderError = error instanceof Error ? error.message : String(error);
    } finally {
      folderBusy = false;
    }
  }

  async function useDownloads() {
    folderBusy = true;
    folderError = '';
    try {
      await clearAutoSaveDirectory();
    } catch (error) {
      folderError = error instanceof Error ? error.message : String(error);
    } finally {
      folderBusy = false;
    }
  }
</script>

<section class="configuration-page">
  <h1>{tr('nav.configuration')}</h1>
  <form on:submit|preventDefault={save}>
    <h2>{tr('configuration.history')}</h2>
    <label for="history-limit">{tr('configuration.historyLimit')}</label>
    <input id="history-limit" type="number" min="1" step="1" required bind:value={historyLimit} />

    <h2>{tr('configuration.autoSave')}</h2>
    <label class="checkbox-row">
      <input id="auto-save-enabled" type="checkbox" bind:checked={autoSave} />
      <span>{tr('configuration.autoSaveEnabled')}</span>
    </label>
    <label class="checkbox-row">
      <input id="auto-save-settings" type="checkbox" bind:checked={autoSaveSettings} disabled={!autoSave} />
      <span>{tr('configuration.autoSaveSettings')}</span>
    </label>

    <div class="folder-box">
      <span class="folder-label">{tr('configuration.autoSaveFolder')}</span>
      {#if $autoSaveDirectory.supported}
        <p class="folder-status">
          {#if $autoSaveDirectory.folderName}
            {$autoSaveDirectory.permission === 'granted'
              ? tr('configuration.folderSelected', { name: $autoSaveDirectory.folderName })
              : tr('configuration.folderNeedsPermission', { name: $autoSaveDirectory.folderName })}
          {:else}
            {tr('configuration.browserDownloads')}
          {/if}
        </p>
        <div class="folder-actions">
          <button type="button" disabled={folderBusy} on:click={chooseFolder}>{tr('configuration.chooseFolder')}</button>
          {#if $autoSaveDirectory.folderName && $autoSaveDirectory.permission !== 'granted'}
            <button type="button" disabled={folderBusy} on:click={grantAccess}>{tr('configuration.grantAccess')}</button>
          {/if}
          {#if $autoSaveDirectory.folderName}
            <button type="button" disabled={folderBusy} on:click={useDownloads}>{tr('configuration.useDownloads')}</button>
          {/if}
        </div>
      {:else}
        <p class="folder-status">{tr('configuration.folderUnsupportedNote')}</p>
      {/if}
      {#if folderError}<div class="configuration-error" role="alert">{folderError}</div>{/if}
    </div>

    <button type="submit">{tr('configuration.save')}</button>
    {#if $uiConfigurationError}<div class="configuration-error" role="alert">{$uiConfigurationError}</div>{/if}
    {#if status}<div role="status">{tr(status)}</div>{/if}
  </form>
</section>

<style>
  .configuration-page { max-width: 640px; margin: 28px auto; }
  h1 { font-size: 24px; margin: 0 0 24px; }
  h2 { font-size: 16px; margin: 12px 0 4px; }
  form { display: grid; gap: 12px; justify-items: start; }
  label { margin: 0; }
  input[type='number'] { width: 180px; max-width: 100%; }
  .checkbox-row { display: flex; align-items: center; gap: 10px; cursor: pointer; }
  .checkbox-row input { width: auto; min-height: 0; margin: 0; }
  .folder-box { display: grid; gap: 8px; padding: 12px 14px; border: 1px solid var(--line); border-radius: 8px; width: 100%; }
  .folder-label { font-size: 12px; color: var(--muted); }
  .folder-status { margin: 0; font-size: 13px; overflow-wrap: anywhere; }
  .folder-actions { display: flex; gap: 8px; flex-wrap: wrap; }
  .configuration-error { color: var(--danger); overflow-wrap: anywhere; }
</style>
