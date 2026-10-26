<script lang="ts">
  import type { ModelSettings } from '$lib/model-settings/types';
  import { catalog, parameterCatalog } from '$lib/catalog';
  export let label: string;
  export let emptyLabel = 'Choose an installed model';
  export let inherit: ModelSettings | null = null;
  export let choices: ModelSettings[];
  export let tasks: readonly string[];
  export let families: readonly string[] | undefined = undefined;
  export let value: ModelSettings | null;
  export let disabled = false;
  export let settingsLabel = 'More generation options';
  export let onchange: (kind?: 'model' | 'settings') => void;
  let json = '';
  let error = '';
  let expanded = false;
  $: compatible = choices.filter(
    (p) =>
      tasks.includes(p.model.task) &&
      p.model.mode === 'offline' &&
      (!families || families.includes(p.model.family))
  );
  $: models = compatible;
  $: effective = value || inherit;
  $: entry =
    effective &&
    (catalog.find((e) => e.id === effective.model.id) ||
      catalog.find((e) => e.family === effective.model.family && e.task === effective.model.task));
  $: parameters = effective
    ? (parameterCatalog[effective.model.id] || parameterCatalog[effective.model.family] || []).filter(
        (p) =>
          p.scope !== 'session' &&
          ['number', 'slider', 'bool', 'choice'].includes(p.type) &&
          (entry?.request_options === undefined || entry.request_options.includes(p.name))
      )
    : [];
  function choose(id: string) {
    value = id ? structuredClone(compatible.find((p) => p.id === id)!) : null;
    error = '';
    expanded = false;
    onchange('model');
  }
  function request(name: string, v: unknown) {
    if (!effective) return;
    if (!value) value = structuredClone(effective);
    value = { ...value, request: { ...value.request, [name]: v } };
    onchange();
  }
  function option(name: string, v: unknown) {
    if (!effective) return;
    if (!value) value = structuredClone(effective);
    value = { ...value, request: { ...value.request, options: { ...effective.request.options, [name]: v } } };
    onchange();
  }
</script>

<div class="model-choice">
  <label
    >{label}<select
      aria-label={label}
      value={value?.id || ''}
      {disabled}
      on:change={(e) => choose(e.currentTarget.value)}
    >
      <option value="">{emptyLabel}</option>
      {#if value && !compatible.some((p) => p.id === value!.id)}<option value={value.id}
          >{value.model.display_name} · saved settings</option
        >{/if}
      {#if models.length}<optgroup label="Installed models"
          >{#each models as model}<option value={model.id}>{model.name}</option>{/each}</optgroup
        >{/if}
    </select></label
  >
  {#if effective}
    <div class="choice-meta">
      <span>{value ? effective.model.display_name : 'Uses the project default'}</span><button
        type="button"
        {disabled}
        aria-expanded={expanded}
        on:click={() => {
          expanded = !expanded;
          json = JSON.stringify(effective!.request.options, null, 2);
        }}>{expanded ? `Hide ${settingsLabel.toLowerCase()}` : settingsLabel}</button
      >
    </div>
    {#if expanded}
      <div class="choice-settings">
        {#if ['tts', 'clon', 'vdes'].includes(effective.model.task)}
          <label
            >Variation seed (-1 = random)<input
              aria-label={`${label} seed`}
              type="number"
              min="-1"
              max="4294967295"
              value={effective.request.seed}
              {disabled}
              on:change={(e) => request('seed', e.currentTarget.valueAsNumber)}
            /></label
          >
        {/if}
        <label
          >Language<input
            aria-label={`${label} language`}
            value={effective.request.language || ''}
            {disabled}
            placeholder="Model default"
            on:change={(e) => request('language', e.currentTarget.value)}
          /></label
        >
        {#each parameters as p}
          <label
            >{p.label || p.name}
            {#if p.type === 'bool'}<input
                type="checkbox"
                checked={Boolean(effective.request.options[p.name] ?? p.default)}
                {disabled}
                on:change={(e) => option(p.name, e.currentTarget.checked)}
              />
            {:else if p.type === 'choice'}<select
                value={String(effective.request.options[p.name] ?? p.default)}
                {disabled}
                on:change={(e) => option(p.name, e.currentTarget.value)}
                >{#each p.choices || [] as v}<option value={v}>{v}</option>{/each}</select
              >
            {:else}<input
                type="number"
                min={p.minimum}
                max={p.maximum}
                step={p.step || 'any'}
                value={Number(effective.request.options[p.name] ?? p.default)}
                {disabled}
                on:change={(e) => option(p.name, e.currentTarget.valueAsNumber)}
              />{/if}
          </label>
        {/each}
      </div>
      <details
        on:toggle={(e) => {
          if (e.currentTarget.open) json = JSON.stringify(effective!.request.options, null, 2);
        }}
      >
        <summary>Request options JSON</summary><textarea
          aria-label={`${label} options JSON`}
          rows="4"
          bind:value={json}
          {disabled}></textarea><button
          {disabled}
          on:click={() => {
            try {
              const options = JSON.parse(json);
              if (!options || typeof options !== 'object' || Array.isArray(options))
                throw new Error('Options must be a JSON object.');
              value = { ...effective!, request: { ...effective!.request, options } };
              error = '';
              onchange();
            } catch (e) {
              error = String(e);
            }
          }}>Apply options</button
        >
      </details>
    {/if}
  {:else if !compatible.length}<p class="empty-choice">
      Install a compatible model from Models.
    </p>{/if}
  {#if error}<p role="alert">{error}</p>{/if}
</div>

<style>
  button {
    min-height: 34px;
  }
  .model-choice {
    min-width: 0;
  }
  label {
    display: grid;
    gap: 8px;
    font-size: 13px;
    color: var(--muted);
  }
  select,
  input,
  textarea {
    width: 100%;
    min-width: 0;
  }
  .choice-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    font-size: 11px;
    color: var(--muted);
  }
  .choice-meta button {
    font-size: 12px;
    padding: 6px 10px;
    border: 1px solid var(--line);
    color: var(--text);
  }
  .choice-settings {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    padding: 16px 0;
  }
  .choice-settings input[type='checkbox'] {
    width: 18px;
    height: 18px;
  }
  details {
    margin: 12px 0;
    font-size: 12px;
  }
  textarea {
    margin: 12px 0;
  }
  [role='alert'] {
    color: var(--danger);
  }
  .empty-choice {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.6;
  }
</style>
