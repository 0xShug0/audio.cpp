<script lang="ts">
  import type { Translator } from '$lib/i18n';
  import type { ParamSpec } from '$lib/types';

  export let task = 'tts';
  export let busy = false;
  export let paramSpecs: ParamSpec[] = [];
  export let advancedValues: Record<string, unknown> = {};
  export let tr: Translator = (key, _values, fallback = key) => fallback;
  export let localizedParameterText:
    (spec: ParamSpec, field: 'label' | 'info' | 'placeholder', translate?: Translator) => string =
      (spec, field) => field === 'label' ? spec.name : '';
  export let setParameterValue: (spec: ParamSpec, value: unknown) => void = () => {};

  const componentNames = ['model_gguf', 'qwen_gguf', 'vae_gguf'];
  const samplingNames = ['num_inference_steps', 'guidance_scale', 'sway_sampling_coef'];
  const runtimeNames = ['attention', 'mem_saver'];

  function specsByName(names: string[]) {
    return names
      .map((name) => paramSpecs.find((spec) => spec.name === name))
      .filter((spec): spec is ParamSpec => spec !== undefined);
  }

  $: componentSpecs = specsByName(componentNames);
  $: samplingSpecs = specsByName(samplingNames);
  $: runtimeSpecs = specsByName(runtimeNames);
  $: generator = String(advancedValues.model_gguf || 'auk-base-f32.gguf');
  $: variant = generator.startsWith('auk-flash-') ? 'AuK-Flash' : 'Base';
</script>

<div class="auk-form">
  <div class="auk-heading">
    <strong>Components</strong>
    <span>{variant}</span>
  </div>
  <div class="auk-grid auk-components">
    {#each componentSpecs as spec}
      <div class="auk-field">
        <label for={'auk-param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
        <select id={'auk-param-' + spec.name} disabled={busy}
          value={String(advancedValues[spec.name] ?? spec.default ?? '')}
          on:change={(event) => setParameterValue(spec, event.currentTarget.value)}>
          {#each spec.choices || [] as choice}<option value={choice}>{choice}</option>{/each}
        </select>
        {#if localizedParameterText(spec, 'info', tr)}<small>{localizedParameterText(spec, 'info', tr)}</small>{/if}
      </div>
    {/each}
  </div>

  <div class="auk-grid">
    {#if task === 'tts'}
      {#each specsByName(['instruct']) as spec}
        <div class="auk-field wide">
          <label for="auk-param-instruct">{localizedParameterText(spec, 'label', tr)} <span>{tr('request.optional')}</span></label>
          <textarea id="auk-param-instruct" rows="2" disabled={busy}
            value={String(advancedValues.instruct ?? '')}
            placeholder={localizedParameterText(spec, 'placeholder', tr)}
            on:input={(event) => setParameterValue(spec, event.currentTarget.value)}></textarea>
          {#if localizedParameterText(spec, 'info', tr)}<small>{localizedParameterText(spec, 'info', tr)}</small>{/if}
        </div>
      {/each}
    {/if}
    {#each specsByName(['duration_sec']) as spec}
      <div class="auk-field">
        <label for="auk-param-duration">{localizedParameterText(spec, 'label', tr)} <span>{tr('request.optional')}</span></label>
        <input id="auk-param-duration" type="number" disabled={busy}
          min={spec.minimum} max={spec.maximum} step={spec.step}
          value={String(advancedValues.duration_sec ?? '')}
          placeholder={localizedParameterText(spec, 'placeholder', tr)}
          on:input={(event) => setParameterValue(spec,
            event.currentTarget.value === '' ? '' : event.currentTarget.valueAsNumber)} />
        {#if localizedParameterText(spec, 'info', tr)}<small>{localizedParameterText(spec, 'info', tr)}</small>{/if}
      </div>
    {/each}
  </div>

  <details>
    <summary>Sampling and memory <span>{samplingSpecs.length + runtimeSpecs.length}</span></summary>
    <div class="auk-grid details-grid">
      {#each [...samplingSpecs, ...runtimeSpecs] as spec}
        <div class="auk-field">
          <label for={'auk-param-' + spec.name}>{localizedParameterText(spec, 'label', tr)}</label>
          {#if spec.type === 'bool'}
            <label class="toggle">
              <input id={'auk-param-' + spec.name} type="checkbox" disabled={busy}
                checked={Boolean(advancedValues[spec.name])}
                on:change={(event) => setParameterValue(spec, event.currentTarget.checked)} />
              <span></span>{advancedValues[spec.name] ? tr('common.enabled') : tr('common.disabled')}
            </label>
          {:else if spec.type === 'choice'}
            <select id={'auk-param-' + spec.name} disabled={busy}
              value={String(advancedValues[spec.name] ?? spec.default ?? '')}
              on:change={(event) => setParameterValue(spec, event.currentTarget.value)}>
              {#each spec.choices || [] as choice}<option value={choice}>{choice}</option>{/each}
            </select>
          {:else if spec.type === 'slider'}
            <div class="auk-range">
              <input id={'auk-param-' + spec.name} type="range" disabled={busy}
                min={spec.minimum} max={spec.maximum} step={spec.step}
                value={Number(advancedValues[spec.name] ?? spec.default)}
                on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
              <output>{String(advancedValues[spec.name])}</output>
            </div>
          {:else}
            <input id={'auk-param-' + spec.name} type="number" disabled={busy}
              min={spec.minimum} max={spec.maximum} step={spec.step}
              value={String(advancedValues[spec.name] ?? '')}
              on:input={(event) => setParameterValue(spec, event.currentTarget.valueAsNumber)} />
          {/if}
          {#if localizedParameterText(spec, 'info', tr)}<small>{localizedParameterText(spec, 'info', tr)}</small>{/if}
        </div>
      {/each}
    </div>
  </details>
</div>

<style>
  .auk-form {
    display: grid;
    gap: 16px;
  }

  .auk-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .auk-heading span,
  .auk-field label span,
  .auk-field small,
  summary span {
    color: var(--muted);
    font-size: 0.78rem;
  }

  .auk-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }

  .auk-components {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .auk-field {
    min-width: 0;
  }

  .auk-field.wide {
    grid-column: 1 / -1;
  }

  .auk-field > label:first-child {
    display: block;
    margin-bottom: 6px;
  }

  .auk-field :global(input),
  .auk-field :global(select),
  .auk-field :global(textarea) {
    box-sizing: border-box;
    width: 100%;
  }

  .auk-field small {
    display: block;
    margin-top: 5px;
    line-height: 1.35;
  }

  details > summary {
    cursor: pointer;
    font-weight: 650;
  }

  .details-grid {
    margin-top: 12px;
  }

  .auk-range {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 52px;
    align-items: center;
    gap: 8px;
  }

  .auk-range output {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  @media (max-width: 760px) {
    .auk-grid,
    .auk-components {
      grid-template-columns: 1fr;
    }
  }
</style>
