<script lang="ts">
  import { saveUiConfiguration, uiConfiguration, uiConfigurationError } from '$lib/configuration';
  import type { Translator } from '$lib/i18n';

  export let tr: Translator;

  let historyLimit = $uiConfiguration.historyLimit;
  let status = '';
  $: historyLimit = $uiConfiguration.historyLimit;

  function save() {
    try {
      saveUiConfiguration({ historyLimit });
      status = 'configuration.saved';
    } catch (error) {
      uiConfigurationError.set(error instanceof Error ? error.message : String(error));
      status = '';
    }
  }
</script>

<section class="configuration-page">
  <h1>{tr('nav.configuration')}</h1>
  <form on:submit|preventDefault={save}>
    <h2>{tr('configuration.history')}</h2>
    <label for="history-limit">{tr('configuration.historyLimit')}</label>
    <input id="history-limit" type="number" min="1" step="1" required bind:value={historyLimit} />
    <button type="submit">{tr('configuration.save')}</button>
    {#if $uiConfigurationError}<div class="configuration-error" role="alert">{$uiConfigurationError}</div>{/if}
    {#if status}<div role="status">{tr(status)}</div>{/if}
  </form>
</section>

<style>
  .configuration-page { max-width: 640px; margin: 28px auto; }
  h1 { font-size: 24px; margin: 0 0 24px; }
  h2 { font-size: 16px; margin: 0 0 8px; }
  form { display: grid; gap: 12px; justify-items: start; }
  label { margin: 0; }
  input { width: 180px; max-width: 100%; }
  .configuration-error { color: var(--danger); overflow-wrap: anywhere; }
</style>
