<script lang="ts">
  import { onMount } from 'svelte';
  import { Plus, Search, Pencil, Copy, Trash2, UserRound } from '@lucide/svelte';
  import { listVoices, type SavedVoice } from '$lib/library';
  import { loadModelChoices } from '$lib/model-settings/choices';
  import type { ModelSettings } from '$lib/model-settings/types';
  import { listSpeakers, saveSpeaker, deleteSpeaker } from './speakers';
  import { speakerVoiceLabel, type Speaker } from './project';
  import SpeakerEditor from './SpeakerEditor.svelte';
  let items: Speaker[] = [];
  let choices: ModelSettings[] = [];
  let voices: SavedVoice[] = [];
  let query = '';
  let selectedId = '';
  let editing: Speaker | null = null;
  let editorOpen = false;
  let busy = false;
  let loading = true;
  let error = '';
  $: filtered = items.filter((speaker) => `${speaker.name} ${speaker.configuration.model.display_name} ${speakerVoiceLabel(speaker)}`.toLowerCase().includes(query.toLowerCase()));
  $: selected = items.find((speaker) => speaker.id === selectedId);
  export function canClose() {
    // The modal owns draft fields. Keep navigation in place until Save or Cancel.
    return !busy && !editorOpen;
  }
  async function refresh() {
    items = await listSpeakers();
  }
  function open(speaker: Speaker | null) {
    editing = speaker ? structuredClone(speaker) : null;
    editorOpen = true;
    error = '';
  }
  async function save(speaker: Speaker) {
    await saveSpeaker(speaker);
    await refresh();
    selectedId = speaker.id;
    query = '';
  }
  async function duplicate() {
    if (!selected || busy) return;
    busy = true;
    error = '';
    try {
      const base = `${selected.name} copy`;
      let name = base;
      for (let n = 2; items.some((speaker) => speaker.name === name); n++) name = `${base} ${n}`;
      await save({ ...structuredClone(selected), id: crypto.randomUUID(), name });
    } catch (e) { error = `Could not duplicate speaker: ${String(e)}`; }
    finally { busy = false; }
  }
  async function remove() {
    if (!selected || busy || !confirm(`Delete speaker preset "${selected.name}"? Existing projects keep their own copy.`)) return;
    busy = true;
    error = '';
    try {
      await deleteSpeaker(selected.id);
      await refresh();
      selectedId = '';
    } catch (e) { error = `Could not delete speaker: ${String(e)}`; }
    finally { busy = false; }
  }
  onMount(async () => {
    try {
      const [speakers, models, recordings] = await Promise.all([listSpeakers(), loadModelChoices().catch((e) => { error = `Models are unavailable. Saved speakers can still be edited. ${String(e)}`; return []; }), listVoices()]);
      items = speakers; choices = models; voices = recordings;
    } catch (e) { error = `Could not open speaker presets: ${String(e)}`; }
    finally { loading = false; }
  });
</script>

<section class="speaker-library" aria-label="Speaker preset library">
  <header>
    <div><h1>Speakers</h1><p>{items.length} saved · Model, voice, and settings together</p></div>
    <button class="primary" disabled={busy || loading} on:click={() => open(null)}><Plus size={16} />New speaker preset</button>
  </header>
  <p class="purpose">Manage speaker presets here, then choose them in Text to speech. Existing projects keep their own speaker settings.</p>
  {#if error}<p role="alert">{error}</p>{/if}
  {#if loading}<p role="status">Loading speaker presets…</p>{:else}
    <div class="speaker-layout">
      <div class="browser">
        <label class="search"><Search size={16} /><input aria-label="Search speaker presets" placeholder="Search speakers, voices, or models" bind:value={query} /></label>
        <div class="speaker-grid">
          {#each filtered as speaker (speaker.id)}
            <button class="speaker-tile" class:active={selectedId === speaker.id} aria-pressed={selectedId === speaker.id} aria-label={`Select ${speaker.name}`} disabled={busy} on:click={() => (selectedId = speaker.id)}>
              <span class="avatar"><UserRound size={20} /></span><strong>{speaker.name}</strong>
              <span>{speakerVoiceLabel(speaker)}</span><small>{speaker.configuration.model.display_name}</small>
            </button>
          {/each}
        </div>
        {#if !filtered.length}<div class="empty"><UserRound size={32} /><h2>{query ? 'No matching speakers' : 'Create your first speaker preset'}</h2><p>{query ? 'Try a different name, voice, or model.' : 'Save a model, voice, and generation settings once, then reuse them in your TTS projects.'}</p></div>{/if}
      </div>
      <aside aria-label="Speaker preset details">
        {#if selected}
          <h2>{selected.name}</h2>
          <dl><dt>Model</dt><dd>{selected.configuration.model.display_name}</dd><dt>Voice</dt><dd>{speakerVoiceLabel(selected)}</dd><dt>Voice source</dt><dd>{selected.voice.kind === 'reference' ? 'Reference recording' : 'Built-in voice'}</dd><dt>Seed</dt><dd>{selected.configuration.request.seed === -1 ? 'Random' : selected.configuration.request.seed}</dd><dt>Language</dt><dd>{selected.configuration.request.language || 'Model default'}</dd></dl>
          <div class="tools"><button disabled={busy} on:click={() => open(selected!)}><Pencil size={15} />Edit speaker</button><button disabled={busy} on:click={duplicate}><Copy size={15} />Duplicate</button><button class="delete" disabled={busy} on:click={remove}><Trash2 size={15} />Delete preset</button></div>
        {:else}<div class="empty"><UserRound size={28} /><p>Select a speaker to see its settings.</p></div>{/if}
      </aside>
    </div>
  {/if}
</section>
{#if editorOpen}<SpeakerEditor initial={editing} {choices} {voices} number={items.length + 1} onsave={save} onclose={() => (editorOpen = false)} onbusy={(value) => (busy = value)} presetEditor />{/if}

<style>
  header { display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 8px 0 12px; }
  h1 { margin: 0; font-size: 24px; } h2 { font-size: 18px; margin: 0; overflow-wrap: anywhere; }
  p { font-size: 13px; color: var(--muted); line-height: 1.6; }
  header p { margin: 8px 0 0; } .purpose { margin: 0 0 24px; }
  button { display: inline-flex; align-items: center; justify-content: center; gap: 8px; min-height: 38px; }
  .speaker-layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(270px, 370px); border-top: 1px solid var(--line); }
  .browser { min-width: 0; padding: 22px 24px 22px 0; }
  .search { display: flex; align-items: center; gap: 10px; color: var(--muted); margin-bottom: 20px; }
  input { width: 100%; min-width: 0; }
  .speaker-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(185px, 1fr)); gap: 12px; max-height: 56dvh; overflow: auto; }
  .speaker-tile { flex-direction: column; align-items: flex-start; padding: 18px; gap: 8px; text-align: left; background: var(--panel); border: 1px solid var(--line); }
  .speaker-tile.active { border-color: var(--cyan); background: var(--active-bg); }
  .speaker-tile > span:not(.avatar), small { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
  .speaker-tile:hover > span:not(.avatar), .speaker-tile:hover small { color: var(--text); }
  strong { overflow-wrap: anywhere; }
  .avatar { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 50%; background: var(--active-bg); color: var(--cyan); margin-bottom: 4px; }
  aside { min-width: 0; border-left: 1px solid var(--line); padding: 24px; }
  dl { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 14px; margin: 24px 0; font-size: 13px; } dt { color: var(--muted); } dd { margin: 0; overflow-wrap: anywhere; }
  .tools { display: flex; gap: 8px; flex-wrap: wrap; } .delete { color: var(--danger); }
  .empty { display: grid; justify-items: center; align-content: center; text-align: center; min-height: 220px; gap: 14px; color: var(--muted); padding: 24px; } .empty p { margin: 0; max-width: 320px; }
  [role='alert'] { color: var(--danger); }
  @media (max-width: 700px) { header { flex-wrap: wrap; } .speaker-layout { display: block; } .browser { padding-right: 0; } .speaker-grid { max-height: 36dvh; grid-template-columns: repeat(2, minmax(0, 1fr)); } aside { border-left: 0; border-top: 1px solid var(--line); padding: 24px 0; } }
  @media (max-width: 380px) { .speaker-grid { grid-template-columns: 1fr; } }
</style>
