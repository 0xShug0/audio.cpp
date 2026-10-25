<script lang="ts">
  import LoraGroup from './LoraGroup.svelte';
  import type { ParamSpec } from './types';

  export let specs: ParamSpec[];
  export let values: Record<string, unknown>;
  export let editable = false;
  export let busy = false;
  export let onChange: (spec: ParamSpec, value: unknown) => void;
  export let onBrowse: (spec: ParamSpec) => void;

  $: adapter = specs.find((spec) => spec.name === 'lora');
  $: strength = specs.find((spec) => spec.name === 'lora_scale');
  $: selected = String(values.lora ?? '').trim();
</script>

{#if adapter && strength}
  <LoraGroup selected={selected ? 1 : 0}>
      <p>{editable ? 'Applied when the model loads. Reload or generate to apply changes.' : 'Managed by the server configuration. Set the adapter and strength in its session options.'}</p>
      <div class="lora-fields">
        <div>
          <label for="lora-adapter-path">Adapter path <span>Optional</span></label>
          <div class="lora-path">
            <input id="lora-adapter-path" type="text" value={String(values.lora ?? '')}
              placeholder="Server adapter file or directory" disabled={busy || !editable}
              on:input={(event) => onChange(adapter, event.currentTarget.value.trim())} />
            <button type="button" disabled={busy || !editable} on:click={() => onBrowse(adapter)}>Browse folder</button>
          </div>
        </div>
        <div>
          <label for="lora-adapter-strength">Strength</label>
          <input id="lora-adapter-strength" type="number" step={strength.step}
            min={strength.minimum} max={strength.maximum} value={String(values.lora_scale ?? '')}
            placeholder="Adapter default" disabled={busy || !editable || !selected}
            on:input={(event) => {
              if (event.currentTarget.value === '') onChange(strength, '');
              else if (Number.isFinite(event.currentTarget.valueAsNumber)) onChange(strength, event.currentTarget.valueAsNumber);
            }} />
        </div>
      </div>
      {#if adapter.info}<small>{adapter.info}</small>{/if}
      {#if strength.info}<small>{strength.info}</small>{/if}
      <button type="button" disabled={busy || !editable || !selected} on:click={() => {
        onChange(adapter, '');
        onChange(strength, strength.default ?? '');
      }}>Clear adapter</button>
  </LoraGroup>
{/if}
