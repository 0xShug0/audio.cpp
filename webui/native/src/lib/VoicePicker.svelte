<script lang="ts">
  import type { Translator } from '$lib/i18n';
  import type { SavedVoice } from '$lib/library';

  export let id: string;
  export let value = '';
  export let presets: string[] = [];
  export let savedVoices: SavedVoice[] = [];
  export let presetLabel: string;
  export let emptyLabel: string;
  export let allowEmpty = true;
  export let referenceOption = false;
  export let disabled = false;
  export let tr: Translator;
  export let onselect: (value: string) => void;
</script>

<label for={id}>{tr('arena.voice.label')}</label>
<select {id} {value} {disabled} on:change={(event) => onselect(event.currentTarget.value)}>
  {#if allowEmpty}<option value="">{emptyLabel}</option>{/if}
  {#if referenceOption}<option value="reference">{tr('arena.voice.reference')}</option>{/if}
  {#if presets.length}
    <optgroup label={presetLabel}>
      {#each presets as voice}<option value={`preset:${voice}`}>{voice}</option>{/each}
    </optgroup>
  {/if}
  {#if savedVoices.length}
    <optgroup label={tr('voice.saved')}>
      {#each savedVoices as voice}<option value={`saved:${voice.id}`}>{voice.name}</option>{/each}
    </optgroup>
  {/if}
</select>
