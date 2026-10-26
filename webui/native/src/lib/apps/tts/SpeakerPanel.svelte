<script lang="ts">
  import { onMount } from 'svelte';
  import { Plus, Pencil, Check, Users } from '@lucide/svelte';
  import type { SavedVoice } from '$lib/library';
  import type { ModelSettings } from '$lib/model-settings/types';
  import { assignedSpeaker, speakerVoiceLabel, type Speaker, type TtsProject, type Chunk } from './project';
  import { listSpeakers, saveSpeaker, speakerLibraryRevision } from './speakers';
  import SpeakerEditor from './SpeakerEditor.svelte';
  export let project: TtsProject;
  export let chunk: Chunk | undefined;
  export let choices: ModelSettings[];
  export let voices: SavedVoice[];
  export let disabled = false;
  export let onchange: () => void;
  export let onbusy: (value: boolean) => void;
  export let onerror: (message: string) => void;
  let saved: Speaker[] = [];
  let editorOpen = false;
  let editing: Speaker | null = null;
  $: selected = assignedSpeaker(project, chunk);
  $: selectedNumber = chunk ? project.chunks.indexOf(chunk) + 1 : 0;
  export function addSpeaker() {
    if (disabled) return;
    editing = null;
    editorOpen = true;
  }
  async function save(speaker: Speaker) {
    await saveSpeaker(speaker);
    const first = !project.speakers?.length;
    saved = [speaker, ...saved.filter((s) => s.id !== speaker.id)];
    project.speakers = project.speakers?.some((s) => s.id === speaker.id)
      ? project.speakers.map((s) => (s.id === speaker.id ? speaker : s))
      : [...(project.speakers || []), speaker];
    if (!project.defaultSpeakerId) project.defaultSpeakerId = speaker.id;
    if (!editing && chunk && !first) chunk.speakerId = speaker.id;
    onchange();
  }
  function assign(speaker: Speaker) {
    if (chunk) chunk.speakerId = speaker.id;
    else project.defaultSpeakerId = speaker.id;
    onchange();
  }
  function useEverywhere() {
    if (!selected) return;
    project.defaultSpeakerId = selected.id;
    for (const item of project.chunks) item.speakerId = null;
    onchange();
  }
  onMount(() => {
    let current = 0;
    const unsubscribe = speakerLibraryRevision.subscribe(async () => {
      const request = ++current;
      try {
        const speakers = await listSpeakers();
        if (request === current) saved = speakers;
      } catch (e) {
        if (request === current) onerror(`Could not open the speaker library: ${String(e)}`);
      }
    });
    return () => { current++; unsubscribe(); };
  });
</script>

<section class="speaker-panel" aria-label="Speakers">
  <header>
    <h3>Speakers</h3>
    <button class="icon" aria-label="Add a speaker" {disabled} on:click={addSpeaker}><Plus size={17} /></button>
  </header>
  <p class="context">
    {chunk ? `Assign a speaker to passage ${selectedNumber}.` : 'Choose the default speaker for your narration.'}
  </p>
  <div class="speaker-cast">
    {#each project.speakers || [] as speaker (speaker.id)}
      <article class:assigned={selected?.id === speaker.id}>
        <button
          class="speaker-card"
          aria-label={`Assign ${speaker.name}`}
          aria-pressed={selected?.id === speaker.id}
          {disabled}
          on:click={() => assign(speaker)}
          ><span class="avatar">{speaker.name.slice(0, 1).toUpperCase()}</span><span class="identity"
            ><strong>{speaker.name}</strong><small
              >{speakerVoiceLabel(speaker)} · {speaker.configuration.model.display_name.split(' (')[0]}</small
            >{#if project.defaultSpeakerId === speaker.id}<span class="default-note">Project default</span>{/if}</span
          >{#if selected?.id === speaker.id}<Check size={14} />{/if}</button
        >
        <button
          class="edit-speaker icon"
          aria-label={`Edit ${speaker.name}`}
          {disabled}
          on:click={() => {
            editing = speaker;
            editorOpen = true;
          }}><Pencil size={14} /></button
        >
      </article>
    {/each}
    {#if !project.speakers?.length}<div class="empty-cast">
        <Users size={26} />
        <h4>Add your first speaker</h4>
        <p>Pick a model and a voice once, then use that speaker throughout your script.</p>
      </div>{/if}
  </div>
  <button class="add-speaker" {disabled} on:click={addSpeaker}><Plus size={14} />Add speaker</button>
  <p class="preset-help">Manage presets in Lib → Speakers.</p>
  {#if selected && chunk}<div class="assignment-summary">
      <span class="label">APPLY THIS SPEAKER</span>{#if chunk.speakerId}<button
          class="text-button"
          {disabled}
          on:click={() => {
            chunk!.speakerId = null;
            onchange();
          }}>Follow project default</button
        >{/if}<button {disabled} on:click={useEverywhere}>Use {selected.name} for all passages</button>
      <p>Earlier recordings stay available for comparison.</p>
    </div>{/if}
</section>
{#if editorOpen}<SpeakerEditor
    initial={editing}
    {saved}
    {choices}
    {voices}
    number={(project.speakers?.length || 0) + 1}
    onsave={save}
    onclose={() => (editorOpen = false)}
    {onbusy}
  />{/if}

<style>
  .preset-help { color: var(--muted); font-size: 11px; margin: 10px 0 0; line-height: 1.6; }
  .speaker-panel {
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }
  h3 {
    margin: 0;
    font-size: 15px;
  }
  h4 {
    margin: 0;
    font-size: 13px;
  }
  p {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.65;
    margin: 0;
  }
  .context {
    padding: 12px 0 20px;
  }
  button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    font-size: 12px;
    min-height: 34px;
  }
  .icon {
    padding: 6px;
    min-height: 30px;
    background: transparent;
    border: 0;
  }
  .speaker-cast {
    min-height: 0;
    overflow: auto;
    flex: 0 1 auto;
  }
  article {
    display: flex;
    border: 1px solid var(--line);
    border-radius: 9px;
    margin-bottom: 10px;
    padding: 5px;
    align-items: center;
  }
  article.assigned {
    border-color: var(--cyan);
    background: var(--active-bg);
  }
  .speaker-card {
    flex: 1;
    min-width: 0;
    justify-content: flex-start;
    text-align: left;
    border: 0;
    background: transparent;
    padding: 8px 5px;
    gap: 10px;
  }
  .avatar {
    width: 32px;
    height: 32px;
    flex-shrink: 0;
    display: grid;
    place-items: center;
    border-radius: 50%;
    background: var(--button-bg);
    font-size: 13px;
    color: var(--text);
  }
  .identity {
    min-width: 0;
    flex: 1;
    display: grid;
    gap: 5px;
  }
  .identity strong {
    font-size: 13px;
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .identity small {
    color: var(--muted);
    font-size: 10px;
    line-height: 1.5;
    overflow-wrap: anywhere;
  }
  .default-note {
    font-size: 9px;
    color: var(--muted);
  }
  .speaker-card :global(svg) {
    flex-shrink: 0;
    color: var(--cyan);
  }
  .edit-speaker {
    align-self: flex-start;
  }
  .add-speaker {
    width: 100%;
    background: transparent;
    border-style: dashed;
    margin-top: 2px;
    flex-shrink: 0;
  }
  .empty-cast {
    display: grid;
    gap: 12px;
    justify-items: start;
    padding: 18px 0 22px;
  }
  .empty-cast :global(svg) {
    color: var(--muted);
  }
  .assignment-summary {
    margin-top: 24px;
    padding-top: 20px;
    border-top: 1px solid var(--line);
    display: grid;
    gap: 10px;
  }
  .assignment-summary .label {
    font-size: 9px;
    letter-spacing: 1.1px;
    color: var(--muted);
  }
  .assignment-summary p {
    font-size: 11px;
  }
  .assignment-summary button {
    width: 100%;
    font-size: 11px;
  }
  .text-button {
    background: transparent;
    border: 0;
    color: var(--muted);
  }
  button:focus-visible {
    outline: 2px solid var(--cyan);
    outline-offset: 2px;
  }
</style>
