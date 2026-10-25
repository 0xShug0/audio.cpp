<script lang="ts">
  import { onMount } from 'svelte';
  import { Search, SlidersHorizontal, Save, Trash2, Play, Users } from '@lucide/svelte';
  import { listModelPresets, saveModelPreset, deleteModelPreset, type SavedModelPreset, type SavedVoice } from '$lib/library';
  import VoiceLibrary from './VoiceLibrary.svelte';

  export let section: 'voices' | 'presets' = 'voices';
  export let onvoice: (voice: SavedVoice) => void | Promise<void>;
  export let onvoicechange: () => void | Promise<void>;
  export let onapps: (preset: SavedModelPreset) => void;
  export let onpreset: (preset: SavedModelPreset) => Promise<void>;
  export let taskLabel: (task: string) => string;
  let voiceLibrary: VoiceLibrary | null = null;
  let presets: SavedModelPreset[] = [];
  let selected: SavedModelPreset | null = null;
  let name = '';
  let query = '';
  let busy = false;
  let error = '';
  $: dirty = selected !== null && name !== selected.name;
  $: filtered = presets.filter((preset) =>
    `${preset.name} ${preset.model.display_name} ${preset.model.family} ${preset.model.task} ${taskLabel(preset.model.task)}`
      .toLowerCase().includes(query.toLowerCase()));

  export function canClose() {
    if (section === 'voices') return voiceLibrary?.canClose() ?? true;
    return !busy && (!dirty || confirm('Discard unsaved preset edits?'));
  }
  function switchSection(next: typeof section) {
    if (next === section || !canClose()) return;
    if (selected) name = selected.name;
    section = next;
  }
  function select(preset: SavedModelPreset) {
    if (!canClose()) return;
    selected = preset;
    name = preset.name;
    error = '';
  }
  async function rename() {
    if (!selected || busy) return;
    if (!name.trim()) { error = 'Give this preset a name.'; return; }
    busy = true; error = '';
    try {
      const updated = { ...selected, name: name.trim() };
      await saveModelPreset(updated);
      presets = await listModelPresets();
      selected = updated; name = updated.name;
    } catch (e) { error = String(e); }
    finally { busy = false; }
  }
  async function remove() {
    if (!selected || busy || !confirm(`Delete preset "${selected.name}"?`)) return;
    busy = true; error = '';
    try {
      await deleteModelPreset(selected.id);
      presets = await listModelPresets();
      selected = null; name = '';
    } catch (e) { error = String(e); }
    finally { busy = false; }
  }
  async function use() {
    if (!selected || busy || dirty) return;
    busy = true; error = '';
    try { await onpreset(selected); }
    catch (e) { error = e instanceof Error ? e.message : String(e); }
    finally { busy = false; }
  }
  onMount(() => { listModelPresets().then((items) => presets = items).catch((e) => error = String(e)); });
</script>

<svelte:window on:beforeunload={event => { if (section === 'presets' && (dirty || busy)) { event.preventDefault(); event.returnValue = ''; } }}/>

<div class="library-tabs" aria-label="Library collections">
  <button class:active={section === 'voices'} on:click={() => switchSection('voices')}><Users size={17}/>Voices</button>
  <button class:active={section === 'presets'} on:click={() => switchSection('presets')}><SlidersHorizontal size={17}/>Model presets</button>
</div>
{#if section === 'voices'}
  <VoiceLibrary bind:this={voiceLibrary} ontext={onvoice} onchange={onvoicechange}/>
{:else}
  <section class="preset-library">
    <header><h1>Model presets</h1><span>{presets.length} saved · This browser</span></header>
    {#if error}<p class="preset-error" role="alert">{error}</p>{/if}
    <div class="preset-layout">
      <div class="preset-browser">
        <label class="preset-search"><Search size={16}/><input aria-label="Search model presets" placeholder="Search presets, models or tasks" bind:value={query}/></label>
        <div class="preset-grid">
          {#each filtered as preset (preset.id)}
            <button class="preset-tile" class:active={selected?.id === preset.id} disabled={busy} on:click={() => select(preset)}>
              <strong>{preset.name}</strong>
              <span class="preset-task">{taskLabel(preset.model.task)}</span>
              <span class="preset-model">{preset.model.display_name}</span>
            </button>
          {/each}
        </div>
        {#if !filtered.length}
          <div class="preset-empty"><SlidersHorizontal size={32}/><p>{query ? 'No matching presets' : 'No saved model presets'}</p>
            {#if !query}<span>Find a model you like in Studio, adjust its settings, then choose Save preset.</span>{/if}
          </div>
        {/if}
      </div>
      <aside class="preset-preview">
        {#if selected}
          <label>Name<input aria-label="Preset name" bind:value={name} disabled={busy}/></label>
          <dl>
            <dt>Task</dt><dd>{taskLabel(selected.model.task)}</dd>
            <dt>Model</dt><dd>{selected.model.display_name}</dd>
            <dt>Seed</dt><dd>{selected.request.seed === -1 ? 'Random' : selected.request.seed}</dd>
          </dl>
          <div class="preset-tools">
            <button disabled={busy || !dirty} on:click={rename}><Save size={16}/>{dirty ? 'Save name' : 'Saved'}</button>
            <button class="icon" title="Delete preset" aria-label="Delete preset" disabled={busy} on:click={remove}><Trash2 size={16}/></button>
          </div>
          <button class="primary preset-use" disabled={busy || dirty} on:click={() => selected && onapps(selected)}><Play size={16}/>Use in Apps</button>
          <button class="preset-use" disabled={busy || dirty} on:click={use}><Play size={16}/>Use in Studio</button>
        {:else}<div class="preset-empty"><SlidersHorizontal size={32}/><p>Select a model preset</p></div>{/if}
      </aside>
    </div>
  </section>
{/if}

<style>
  .library-tabs { display:flex; gap:8px; margin-bottom:20px; border-bottom:1px solid var(--line); padding-bottom:12px; }
  button { display:inline-flex; align-items:center; justify-content:center; gap:8px; }
  .library-tabs .active { background:var(--active-bg); color:var(--cyan); border-color:var(--cyan); }
  header { display:flex; align-items:center; gap:12px; flex-wrap:wrap; padding:8px 0 20px; }
  h1 { margin:0; font-size:24px; }
  header span { color:var(--muted); font-size:12px; }
  .preset-layout { display:grid; grid-template-columns:minmax(0,1fr) minmax(280px,400px); border-top:1px solid var(--line); }
  .preset-browser { min-width:0; padding:20px 24px 20px 0; }
  .preset-search { display:flex; align-items:center; gap:10px; margin-bottom:20px; color:var(--muted); }
  .preset-search input { width:100%; }
  .preset-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:12px; max-height:60dvh; overflow:auto; }
  .preset-tile { display:flex; flex-direction:column; align-items:flex-start; text-align:left; padding:16px; background:var(--panel); border:1px solid var(--line); }
  .preset-tile.active { border-color:var(--cyan); }
  .preset-tile strong,.preset-model,dd { overflow-wrap:anywhere; }
  .preset-task { color:var(--cyan); font-size:12px; }
  .preset-model { color:var(--muted); font-size:13px; }
  .preset-preview { min-width:0; padding:20px 0 20px 24px; border-left:1px solid var(--line); }
  .preset-preview label { display:grid; gap:8px; margin:18px 0; font-size:12px; color:var(--muted); }
  .preset-preview input { width:100%; }
  dl { display:grid; grid-template-columns:auto minmax(0,1fr); gap:12px 18px; font-size:13px; }
  dt { color:var(--muted); } dd { margin:0; }
  .preset-tools { display:flex; gap:8px; margin-top:24px; }
  .icon { width:36px; height:36px; padding:0; }
  .preset-use { margin-top:24px; }
  .preset-empty { display:grid; justify-items:center; align-content:center; min-height:220px; color:var(--muted); text-align:center; }
  .preset-empty span { max-width:340px; font-size:13px; }
  .preset-error { color:var(--danger); }
  @media(max-width:700px) {
    .preset-layout { display:block; }.preset-browser { padding-right:0; }
    .preset-grid { max-height:32dvh; grid-template-columns:repeat(2,minmax(0,1fr)); }
    .preset-preview { border-left:0; border-top:1px solid var(--line); padding:20px 0; }
  }
</style>
