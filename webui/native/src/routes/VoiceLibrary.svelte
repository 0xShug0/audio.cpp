<script lang="ts">
  import { randomUUID } from '$lib/uuid.js';
  import { onMount } from 'svelte';
  import { Users, Upload, Search, Save, Trash2, Download, FileText, Mic } from '@lucide/svelte';
  import { listVoices, saveVoice, deleteVoice, exportVoiceLibrary, importVoiceLibrary, type SavedVoice } from '$lib/library';
  import { browserDecodeToWav } from '$lib/audio';
  import MediaPreview from '$lib/MediaPreview.svelte';

  export let ontext: (voice: SavedVoice) => void | Promise<void>;
  export let onchange: () => void | Promise<void>;
  let voices: SavedVoice[] = [];
  let draft: SavedVoice | undefined;
  let original: SavedVoice | undefined;
  let query = '';
  let sort = 'name';
  let notice = '';
  let backupInput: HTMLInputElement;
  let busy = false;
  let error = '';
  let input: HTMLInputElement;
  $: filtered = voices.filter(v => `${v.name} ${v.transcript}`.toLowerCase().includes(query.toLowerCase())).sort((a, b) => sort === 'name'
    ? a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }) || a.id.localeCompare(b.id)
    : b.createdAt - a.createdAt);
  $: dirty = !!draft && (!original || draft.name !== original.name || draft.transcript !== original.transcript || draft.audio !== original.audio);

  export function canClose() {
    return !busy && (!dirty || confirm('Discard unsaved voice edits?'));
  }
  async function refresh() {
    voices = await listVoices();
  }
  function select(voice: SavedVoice) {
    if (dirty && !confirm('Discard unsaved voice edits?')) return;
    original = voice; draft = { ...voice }; error = '';
  }
  async function importVoice(file?: File) {
    if (!file || busy) return;
    if (dirty && !confirm('Discard unsaved voice edits?')) { input.value = ''; return; }
    busy = true; error = '';
    try {
      const audio = await browserDecodeToWav(file);
      draft = { id: randomUUID(), name: file.name.replace(/\.[^.]+$/, ''), transcript: '', audio, createdAt: Date.now() };
      original = undefined;
    } catch (e) { error = `Cannot import ${file.name}: ${String(e)}`; }
    finally { busy = false; input.value = ''; }
  }
  async function save() {
    if (!draft) return;
    if (!draft.name.trim()) { error = 'Give this voice a name.'; return; }
    busy = true; error = '';
    try {
      const saved = { ...draft, name: draft.name.trim() };
      await saveVoice(saved); await refresh();
      original = saved; draft = { ...saved };
      await onchange();
    } catch (e) { error = String(e); }
    finally { busy = false; }
  }
  async function remove() {
    if (!draft || !confirm(`Delete voice "${draft.name}"?`)) return;
    busy = true; error = '';
    try {
      if (original) await deleteVoice(draft.id);
      draft = undefined; original = undefined; await refresh(); await onchange();
    } catch (e) { error = String(e); }
    finally { busy = false; }
  }
  async function use() {
    if (!draft || dirty || busy) return;
    error = '';
    try { await ontext(draft); } catch (e) { error = String(e); }
  }
  function download() {
    if (!draft) return;
    const url = URL.createObjectURL(draft.audio);
    const link = document.createElement('a');
    link.href = url; link.download = `${draft.name}.wav`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function exportLibrary() {
    busy = true; error = ''; notice = '';
    try {
      const url = URL.createObjectURL(await exportVoiceLibrary());
      const link = document.createElement('a');
      link.href = url; link.download = 'audiocpp-voices.json'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      notice = `Exported ${voices.length} saved voices. Unsaved edits are not included.`;
    } catch (e) { error = String(e); }
    finally { busy = false; }
  }
  async function importLibrary(file?: File) {
    if (!file || busy) return;
    if ((dirty && !confirm('Discard unsaved voice edits?')) ||
        !confirm('Import this backup? Existing voices with matching IDs will be updated. Other saved voices will be kept.')) {
      backupInput.value = ''; return;
    }
    busy = true; error = ''; notice = '';
    try {
      const count = await importVoiceLibrary(file);
      await refresh(); draft = undefined; original = undefined;
      await onchange(); notice = `Imported ${count} voices.`;
    } catch (e) { error = String(e); }
    finally { busy = false; backupInput.value = ''; }
  }
  onMount(() => { refresh().catch(e => error = String(e)); });
</script>

<svelte:window on:beforeunload={event => { if (dirty || busy) { event.preventDefault(); event.returnValue = ''; } }}/>

<section class="voice-library-page">
  <header class="library-head">
    <div class="library-title"><Users size={25}/><h1>Voices</h1><span>{voices.length} saved · This browser</span></div>
    <div class="library-actions">
      <button disabled={busy || !voices.length} on:click={exportLibrary}><Download size={16}/>Export library</button>
      <button disabled={busy} on:click={() => backupInput.click()}><Upload size={16}/>Import library</button>
      <button disabled={busy} on:click={() => input.click()}><Upload size={16}/>Add voice</button>
    </div>
    <input hidden bind:this={backupInput} type="file" accept="application/json,.json" on:change={e => importLibrary(e.currentTarget.files?.[0])}/>
    <input hidden bind:this={input} type="file" accept="audio/*" on:change={e => importVoice(e.currentTarget.files?.[0])}/>
  </header>
  <p class="library-help">Export all saved voices, including recordings and transcripts, to transfer them to another browser or device. Backups stay on your device.</p>
  {#if notice}<p role="status">{notice}</p>{/if}
  {#if error}<p class="library-error" role="alert">{error}</p>{/if}
  <div class="library-layout">
    <div class="voice-browser">
      <label class="voice-search"><Search size={16}/><input aria-label="Search voices" placeholder="Search voices" bind:value={query}/></label>
      <label class="voice-sort">Sort voices<select aria-label="Sort voices" bind:value={sort}><option value="name">Name (A–Z)</option><option value="newest">Newest first</option></select></label>
      <div class="voice-grid">
        {#each filtered as voice (voice.id)}
          <button disabled={busy} class="voice-tile" class:active={draft?.id === voice.id} on:click={() => select(voice)}>
            <span class="voice-avatar">{voice.name.slice(0, 1).toUpperCase()}</span><strong>{voice.name}</strong>
            <span class="voice-excerpt">{voice.transcript || 'No transcript'}</span>
          </button>
        {/each}
      </div>
      {#if !filtered.length}<div class="voice-empty"><Users size={36}/><p>{query ? 'No matching voices' : 'Your voice collection is empty'}</p>{#if !query}<button disabled={busy} on:click={() => input.click()}><Upload size={16}/>Add voice</button>{/if}</div>{/if}
    </div>
    <aside class="voice-editor">
      {#if draft}
        <label>Name<input aria-label="Voice name" bind:value={draft.name} disabled={busy}/></label>
        <MediaPreview file={draft.audio} name={draft.name} label="Voice preview"/>
        <label>Words spoken in the recording (optional)<textarea aria-label="Voice transcript" rows="5" bind:value={draft.transcript} disabled={busy}></textarea></label>
        <div class="tools">
          <button class="primary" disabled={busy || !dirty} on:click={save}><Save size={16}/>{dirty ? 'Save voice' : 'Saved'}</button>
          <button class="icon" title="Download voice" aria-label="Download voice" disabled={busy} on:click={download}><Download size={16}/></button>
          <button class="icon" title="Delete voice" aria-label="Delete voice" disabled={busy} on:click={remove}><Trash2 size={16}/></button>
        </div>
        <div class="voice-use"><button disabled={busy || dirty} on:click={use}><FileText size={16}/>Use in Studio</button></div>
      {:else}<div class="voice-empty"><Mic size={32}/><p>Select a voice</p></div>{/if}
    </aside>
  </div>
</section>

<style>
  .voice-library-page { min-width: 0; }
  .library-actions { display:flex; gap:8px; flex-wrap:wrap; }
  .library-help { color:var(--muted); font-size:12px; margin:0 0 20px; }
  .voice-sort { display:flex; align-items:center; gap:12px; margin-bottom:16px; color:var(--muted); font-size:12px; }
  .library-head, .library-title { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .library-head { justify-content: space-between; padding: 8px 0 20px; }
  .library-title h1 { margin: 0; font-size: 24px; }
  .library-title span { color: var(--muted); font-size: 12px; }
  .voice-library-page button { display: inline-flex; align-items: center; justify-content: center; gap: 8px; }
  .library-layout { display: grid; grid-template-columns: minmax(0,1fr) minmax(280px,400px); border-top: 1px solid var(--line); }
  .voice-browser { min-width: 0; padding: 20px 24px 20px 0; }
  .voice-search { display:flex; align-items:center; gap:10px; color:var(--muted); margin-bottom:20px; }
  .voice-search input { width:100%; }
  .voice-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(165px,1fr)); gap:12px; max-height:60dvh; overflow:auto; }
  .voice-tile { display:grid !important; grid-template-columns:32px minmax(0,1fr); text-align:left; justify-content:stretch !important; padding:14px; background:var(--panel); border:1px solid var(--line); }
  .voice-tile.active { border-color:var(--cyan); }
  .voice-tile strong { overflow-wrap:anywhere; }
  .voice-avatar { display:grid; place-items:center; width:32px; height:32px; background:var(--active-bg); color:var(--cyan); border-radius:50%; }
  .voice-excerpt { grid-column:1/-1; color:var(--muted); font-size:12px; display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:2; line-clamp:2; overflow:hidden; }
  .voice-editor { min-width:0; padding:20px 0 20px 24px; border-left:1px solid var(--line); }
  .voice-editor label { display:grid; gap:8px; margin:18px 0; font-size:12px; color:var(--muted); }
  .voice-editor input,.voice-editor textarea { width:100%; }
  .tools, .voice-use { display:flex; gap:8px; flex-wrap:wrap; margin-top:24px; }
  .icon { width:36px; height:36px; padding:0; }
  .voice-empty { display:grid; justify-items:center; align-content:center; min-height:220px; color:var(--muted); }
  .library-error { color: var(--danger); }
  @media(max-width:700px) {
    .library-layout { display:block; }.voice-browser { padding-right:0; }
    .voice-grid { max-height:32dvh; grid-template-columns:repeat(2,minmax(0,1fr)); }
    .voice-editor { border-left:0; border-top:1px solid var(--line); padding:20px 0; }
  }
</style>
