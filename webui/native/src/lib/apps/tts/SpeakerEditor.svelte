<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { X, UserRound } from '@lucide/svelte';
  import type { SavedVoice } from '$lib/library';
  import type { ModelSettings } from '$lib/model-settings/types';
  import { ModelRuntime } from '$lib/model-settings/runtime';
  import { uploadWav } from '$lib/api';
  import type { Speaker, VoiceAssignment } from './project';
  import ModelChoice from './ModelChoice.svelte';
  import VoiceChoice from './VoiceChoice.svelte';
  export let initial: Speaker | null = null;
  export let saved: Speaker[] = [];
  export let choices: ModelSettings[];
  export let voices: SavedVoice[];
  export let number = 1;
  export let presetEditor = false;
  export let onsave: (speaker: Speaker) => Promise<void>;
  export let onclose: () => void;
  export let onbusy: (value: boolean) => void;
  let name = initial?.name || `Speaker ${number}`;
  let configuration: ModelSettings | null = initial ? structuredClone(initial.configuration) : null;
  let assignment: VoiceAssignment | undefined = initial ? structuredClone(initial.voice) : { kind: 'builtin', id: '' };
  let transcription: ModelSettings | null = null;
  let savedId = '';
  let working = false;
  let error = '';
  let dialog: HTMLDialogElement;
  let controller: AbortController | null = null;
  function chooseSaved(id: string) {
    savedId = id;
    error = '';
    const speaker = saved.find((s) => s.id === id);
    name = speaker?.name || `Speaker ${number}`;
    configuration = speaker ? structuredClone(speaker.configuration) : null;
    assignment = speaker ? structuredClone(speaker.voice) : { kind: 'builtin', id: '' };
  }
  function setWorking(value: boolean) {
    working = value;
    onbusy(value);
  }
  async function transcribe() {
    if (assignment?.kind !== 'reference' || !transcription || working) return;
    const reference = assignment.data;
    setWorking(true);
    error = '';
    controller = new AbortController();
    try {
      const runtime = new ModelRuntime(transcription, controller.signal);
      const audio = await uploadWav(reference.audio, controller.signal);
      const result = await runtime.run({ audio });
      if (result.kind !== 'json' || typeof result.data.text !== 'string')
        throw new Error('The model did not return a transcript.');
      assignment = { kind: 'reference', data: { ...reference, transcript: result.data.text } };
    } catch (e) {
      error = controller?.signal.aborted ? '' : String(e);
    } finally {
      controller = null;
      setWorking(false);
    }
  }
  async function save() {
    if (!name.trim() || !configuration || !assignment || working) return;
    setWorking(true);
    error = '';
    try {
      await onsave({
        id: initial?.id || savedId || crypto.randomUUID(),
        name: name.trim(),
        configuration: structuredClone(configuration),
        voice: structuredClone(assignment)
      });
      onclose();
    } catch (e) {
      error = `Could not save speaker: ${String(e)}`;
    } finally {
      setWorking(false);
    }
  }
  onMount(() => dialog.showModal());
  onDestroy(() => {
    controller?.abort();
    onbusy(false);
  });
</script>

<dialog
  bind:this={dialog}
  aria-labelledby="speaker-dialog-title"
  on:cancel={(event) => {
    // File inputs also emit a bubbling cancel when their picker is dismissed.
    // Only cancellation of this dialog should close the speaker draft.
    if (event.target !== event.currentTarget) return;
    if (working) { event.preventDefault(); controller?.abort(); }
    else onclose();
  }}
>
  <header>
    <div class="speaker-symbol"><UserRound size={20} /></div>
    <div>
      <h2 id="speaker-dialog-title">{presetEditor ? initial ? 'Edit speaker preset' : 'New speaker preset' : initial ? 'Edit speaker' : 'Add a speaker'}</h2>
      <p>A speaker preset saves the model, voice, and generation settings together.</p>
    </div>
    <button class="close" aria-label="Close speaker editor" disabled={working} on:click={onclose}
      ><X size={18} /></button
    >
  </header>
  <div class="dialog-body">
    {#if !initial && saved.length}<label
        >Speaker preset<select
          aria-label="Speaker preset"
          value={savedId}
          disabled={working}
          on:change={(e) => chooseSaved(e.currentTarget.value)}
          ><option value="">Create a new speaker preset</option>{#each saved as speaker}<option value={speaker.id}
              >{speaker.name}</option
            >{/each}</select
        ></label
      >{/if}
    <label
      >Speaker name<input
        aria-label="Speaker name"
        bind:value={name}
        disabled={working}
        placeholder="e.g. Alex, Host, or Storyteller"
      /></label
    >
    <ModelChoice
      label="Speech model"
      emptyLabel="Choose an installed speech model"
      settingsLabel="Model settings"
      {choices}
      tasks={['tts', 'clon', 'vdes']}
      bind:value={configuration}
      disabled={working}
      onchange={(kind) => {
        if (kind === 'model') assignment = { kind: 'builtin', id: configuration?.request.voice || '' };
      }}
    />
    <VoiceChoice
      label="Speaker voice"
      {configuration}
      {assignment}
      {voices}
      {choices}
      bind:transcription
      disabled={working}
      onchange={(voice) => (assignment = voice)}
      onbusy={setWorking}
      ontranscribe={transcribe}
      onconfiguration={() => {}}
    />
    {#if error}<p role="alert">{error}</p>{/if}
  </div>
  <footer>
    <span>Saved as a speaker preset for future projects.</span><button disabled={working} on:click={onclose}
      >Cancel</button
    ><button class="save-speaker" disabled={working || !name.trim() || !configuration || !assignment} on:click={save}
      >{working ? 'Working…' : presetEditor ? 'Save preset' : initial ? 'Save speaker' : 'Add speaker'}</button
    >
  </footer>
</dialog>

<style>
  dialog {
    width: min(620px, calc(100vw - 28px));
    max-height: calc(100dvh - 40px);
    padding: 0;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: var(--panel);
    color: var(--text);
    box-shadow: 0 30px 100px #0005;
  }
  dialog::backdrop {
    background: #05060db3;
    backdrop-filter: blur(3px);
  }
  header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 24px;
    border-bottom: 1px solid var(--line);
  }
  .speaker-symbol {
    display: grid;
    place-items: center;
    background: var(--active-bg);
    color: var(--cyan);
    height: 42px;
    width: 42px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  h2 {
    margin: 0 0 6px;
    font-size: 19px;
  }
  p {
    margin: 0;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.6;
  }
  button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    min-height: 36px;
    font-size: 12px;
  }
  .close {
    margin-left: auto;
    padding: 7px;
    background: transparent;
    border: 0;
  }
  .dialog-body :global(.choice-meta > span) {
    display: none;
  }
  .dialog-body :global(.choice-meta button) {
    border: 0;
    background: transparent;
    padding: 0;
    min-height: 36px;
    color: var(--cyan);
    font-size: 12px;
  }
  .dialog-body {
    padding: 24px;
    display: grid;
    gap: 22px;
    max-height: calc(100dvh - 240px);
    overflow: auto;
  }
  label {
    display: grid;
    gap: 8px;
    font-size: 12px;
    color: var(--muted);
  }
  input,
  select {
    width: 100%;
    min-width: 0;
  }
  footer {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 18px 24px;
    border-top: 1px solid var(--line);
  }
  footer span {
    margin-right: auto;
    font-size: 11px;
    color: var(--muted);
    max-width: 200px;
    line-height: 1.5;
  }
  .save-speaker {
    color: var(--text-invert);
    background: var(--cyan);
    border-color: var(--cyan);
    padding-inline: 16px;
  }
  [role='alert'] {
    color: var(--danger);
  }
  button:focus-visible {
    outline: 2px solid var(--cyan);
    outline-offset: 3px;
  }
  @media (max-width: 600px) {
    header,
    .dialog-body {
      padding: 20px;
    }
    footer {
      padding: 16px 20px;
      flex-wrap: wrap;
    }
    footer span {
      flex-basis: 100%;
      max-width: none;
    }
    footer button:first-of-type {
      margin-left: auto;
    }
    .speaker-symbol {
      display: none;
    }
  }
</style>
