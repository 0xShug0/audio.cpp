<script lang="ts">
  import { onMount } from 'svelte';
  import { ArrowDown, ArrowUp, Check, MessageSquare, Plus, Trash2, User, Users } from '@lucide/svelte';
  import { browserDecodeToWav } from '$lib/audio';
  import { speech, uploadWav } from '$lib/api';
  import type { PanelRunContext, PanelRunOutcome, RegisterPanelRunner } from '$lib/models/panel_runner';
  import type { SavedVoice } from '$lib/voices';
  import Drama3Reference from './Drama3Reference.svelte';

  export let busy = false;
  export let savedVoices: SavedVoice[] = [];
  export let setPanelRunner: RegisterPanelRunner = () => () => {};

  // Official Drama 3 defines speaker IDs 0..15, independently of VibeVoice.
  const maxSpeakers = 16;
  type Speaker = { id: number; name: string; file: File | null; transcript: string };
  type Line = { id: number; speaker: number; text: string };
  const newSpeaker = (id: number): Speaker => ({ id, name: `Speaker ${id + 1}`, file: null, transcript: '' });
  let mode: 'single' | 'multi' = 'single';
  let referenceFile: File | null = null;
  let referenceText = '';
  let singleText = '';
  let speakers: Speaker[] = [newSpeaker(0), newSpeaker(1)];
  let selectedSpeaker = 0;
  let nextLineId = 2;
  let lines: Line[] = [{ id: 0, speaker: 0, text: '' }, { id: 1, speaker: 1, text: '' }];

  $: activeIndex = speakers.findIndex((speaker) => speaker.id === selectedSpeaker);
  $: completeSpeakers = speakers.filter((speaker) => speaker.file && speaker.transcript.trim()).length;

  function addSpeaker() {
    const id = Array.from({ length: maxSpeakers }, (_, index) => index)
      .find((id) => !speakers.some((speaker) => speaker.id === id));
    if (id === undefined) return;
    speakers = [...speakers, newSpeaker(id)];
    selectedSpeaker = id;
  }

  function removeSpeaker(id: number) {
    speakers = speakers.filter((speaker) => speaker.id !== id);
    selectedSpeaker = speakers[0].id;
  }

  function moveLine(index: number, direction: number) {
    const copy = [...lines];
    [copy[index], copy[index + direction]] = [copy[index + direction], copy[index]];
    lines = copy;
  }

  function addLine() {
    lines = [...lines, { id: nextLineId++, speaker: selectedSpeaker, text: '' }];
  }

  function example() {
    if (mode === 'single') {
      singleText = 'The observatory opens at sunset. Tonight, we will follow the planets across the sky and listen for the first signs of a coming storm.';
    } else {
      lines = [
        { id: nextLineId++, speaker: speakers[0].id, text: 'The observatory opens at sunset. Did you bring the star charts?' },
        { id: nextLineId++, speaker: speakers[1].id, text: 'Yes, and the sky is finally clear. We should have a wonderful view tonight.' }
      ];
    }
  }

  async function run(context: PanelRunContext): Promise<PanelRunOutcome> {
    // Snapshot before loading or uploading so a submitted scene stays intact.
    const single = mode === 'single';
    const cast = speakers.map((speaker) => ({ ...speaker }));
    const dialogue = lines.map((line) => ({ ...line }));
    const file = referenceFile;
    const transcript = referenceText.trim();
    let input = singleText.trim();
    if (single) {
      if (!file) throw context.warning('Choose a reference recording for single-voice cloning.');
      if (!transcript) throw context.warning('Enter the words spoken in the reference recording.');
      if (!input) throw context.warning('Enter the text to generate.');
    } else {
      if (!dialogue.length || dialogue.some((line) => !line.text.trim())) {
        throw context.warning('Enter text for every dialogue turn, or remove the empty turns.');
      }
      const used = new Set(dialogue.map((line) => line.speaker));
      if (used.size < 2) throw context.warning('Assign dialogue to at least two different speakers.');
      for (const speaker of cast.filter((speaker) => used.has(speaker.id))) {
        if (!speaker.file || !speaker.transcript.trim()) {
          selectedSpeaker = speaker.id;
          throw context.warning(`${speaker.name}: add a reference recording and its exact transcript.`);
        }
      }
      input = dialogue.map((line) => `<|speaker:${line.speaker}|> ${line.text.trim()}`).join('\n');
    }
    await context.ensureLoaded('offline');
    const options = { ...context.options };
    const body: Record<string, unknown> = {
      model: context.modelId, input, seed: context.seed, response_format: 'wav', options
    };
    if (context.maxTokens !== undefined) body.max_tokens = context.maxTokens;
    context.setStatus('Preparing voice references...');
    if (single) {
      body.voice_ref = await uploadWav(await browserDecodeToWav(file!), context.signal);
      body.reference_text = transcript;
    } else {
      const used = new Set(dialogue.map((line) => line.speaker));
      const references = [];
      for (const speaker of cast.filter((speaker) => used.has(speaker.id))) {
        const audio = await browserDecodeToWav(speaker.file!);
        references.push({
          text: `<|speaker:${speaker.id}|> ${speaker.transcript.trim()}`,
          audio_path: await uploadWav(audio, context.signal)
        });
      }
      options.multi_reference_cond = JSON.stringify(references);
    }
    context.setStatus(single ? 'Cloning voice...' : 'Generating multi-speaker dialogue...');
    const result = await speech(body, context.signal);
    return {
      audio: result.blob,
      text: '',
      json: JSON.stringify({
        mode: single ? 'single_voice_clone' : 'multi_speaker',
        speakers: single ? 1 : new Set(dialogue.map((line) => line.speaker)).size,
        seed: context.seed, text_chunk_size: options.text_chunk_size,
        wall_ms: result.wallMs, rtf: result.rtf
      }, null, 2)
    };
  }

  onMount(() => setPanelRunner(run));
</script>

<div class="drama-workspace">
  <div class="mode-switch" role="group" aria-label="Drama 3 mode">
    <button class:chosen={mode === 'single'} aria-pressed={mode === 'single'} disabled={busy}
      on:click={() => mode = 'single'}><User size={17} />Single-voice clone</button>
    <button class:chosen={mode === 'multi'} aria-pressed={mode === 'multi'} disabled={busy}
      on:click={() => mode = 'multi'}><Users size={17} />Multi-speaker<span class="test-label">Testing</span></button>
  </div>

  {#if mode === 'single'}
    <section aria-label="Single-voice reference" class="reference-section">
      <h3><User size={17} />Reference voice</h3>
      <Drama3Reference id="drama-single" bind:file={referenceFile} bind:transcript={referenceText} {busy} {savedVoices} />
    </section>
    <section class="script-section">
      <div class="section-heading"><h3><MessageSquare size={17} />Text to generate</h3>
        <button type="button" class="example" disabled={busy} on:click={example}>Example</button></div>
      <label class="sr-only" for="drama-single-text">Text to generate</label>
      <textarea id="drama-single-text" rows="6" bind:value={singleText} disabled={busy}
        placeholder="What should this voice say?"></textarea>
      <div class="text-count">{singleText.length.toLocaleString()} characters</div>
    </section>
  {:else}
    <p class="testing-warning" role="note">Multi-speaker generation is still under testing. Voice assignment and consistency may vary.</p>
    <section class="cast-section" aria-label="Speaker cast">
      <div class="section-heading"><h3><Users size={17} />Cast <span>{speakers.length} / {maxSpeakers}</span></h3>
        <button type="button" disabled={busy || speakers.length >= maxSpeakers} on:click={addSpeaker}><Plus size={15} />Add speaker</button></div>
      <div class="cast-layout">
        <div class="cast-list" role="group" aria-label="Speakers">
          {#each speakers as speaker (speaker.id)}
            <button type="button" class:chosen={selectedSpeaker === speaker.id} disabled={busy}
              aria-pressed={selectedSpeaker === speaker.id} on:click={() => selectedSpeaker = speaker.id}>
              <span class="speaker-number">{speaker.id + 1}</span><span class="speaker-name">{speaker.name}</span>
              {#if speaker.file && speaker.transcript.trim()}<Check size={14} aria-label="Reference ready" />{/if}
            </button>
          {/each}
        </div>
        <div class="cast-editor">
          {#key selectedSpeaker}
            <div class="name-row">
              <div><label for="drama-speaker-name">Speaker name</label><input id="drama-speaker-name" bind:value={speakers[activeIndex].name} disabled={busy} /></div>
              <button class="icon-button" type="button" aria-label="Remove speaker"
                title={lines.some((line) => line.speaker === selectedSpeaker) ? 'Reassign or remove this speaker\'s dialogue first' : 'Remove speaker'}
                disabled={busy || speakers.length <= 2 || lines.some((line) => line.speaker === selectedSpeaker)}
                on:click={() => removeSpeaker(selectedSpeaker)}><Trash2 size={16} /></button>
            </div>
            <Drama3Reference id={`drama-speaker-${selectedSpeaker}`} bind:file={speakers[activeIndex].file}
              bind:transcript={speakers[activeIndex].transcript} {busy} {savedVoices} />
          {/key}
        </div>
      </div>
      <div class="text-count">{completeSpeakers} of {speakers.length} references ready</div>
    </section>
    <section class="script-section" aria-label="Dialogue editor">
      <div class="section-heading"><h3><MessageSquare size={17} />Dialogue <span>{lines.length} turns</span></h3>
        <button type="button" class="example" disabled={busy} on:click={example}>Example</button></div>
      <div class="dialogue-list">
        {#each lines as line, index (line.id)}
          <div class="dialogue-turn">
            <div class="turn-heading">
              <span class="turn-number">{String(index + 1).padStart(2, '0')}</span>
              <label class="sr-only" for={`drama-line-speaker-${line.id}`}>Turn {index + 1} speaker</label>
              <select id={`drama-line-speaker-${line.id}`} bind:value={line.speaker} disabled={busy}>
                {#each speakers as speaker}<option value={speaker.id}>{speaker.name}</option>{/each}
              </select>
              <div class="turn-actions">
                <button class="icon-button" type="button" title="Move turn up" aria-label="Move turn up" disabled={busy || index === 0} on:click={() => moveLine(index, -1)}><ArrowUp size={14} /></button>
                <button class="icon-button" type="button" title="Move turn down" aria-label="Move turn down" disabled={busy || index === lines.length - 1} on:click={() => moveLine(index, 1)}><ArrowDown size={14} /></button>
                <button class="icon-button" type="button" title="Remove turn" aria-label="Remove turn" disabled={busy} on:click={() => lines = lines.filter((entry) => entry.id !== line.id)}><Trash2 size={14} /></button>
              </div>
            </div>
            <label class="sr-only" for={`drama-line-${line.id}`}>Turn {index + 1} text</label>
            <textarea id={`drama-line-${line.id}`} rows="2" bind:value={line.text} disabled={busy} placeholder="Dialogue for this speaker..."></textarea>
          </div>
        {/each}
      </div>
      <button class="add-turn" type="button" disabled={busy} on:click={addLine}><Plus size={15} />Add turn</button>
    </section>
  {/if}
</div>

<style>
  .drama-workspace { display: grid; gap: 22px; min-width: 0; }
  .mode-switch { display: flex; gap: 4px; border-bottom: 1px solid var(--line); padding-bottom: 10px; flex-wrap: wrap; }
  button { display: inline-flex; align-items: center; justify-content: center; gap: 7px; }
  button.chosen { border-color: var(--active-border); background: var(--active-bg); color: var(--text-strong); }
  .test-label { color: var(--muted); font-size: 10px; text-transform: uppercase; }
  section { min-width: 0; }
  h3 { display: flex; align-items: center; gap: 8px; font-size: 14px; margin: 0 0 12px; }
  h3 span { font-size: 11px; color: var(--muted); font-weight: normal; }
  .section-heading { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
  .section-heading h3 { margin: 0; }
  .example { font-size: 11px; padding: 5px 9px; }
  .testing-warning { margin: 0; padding: 3px 0 3px 12px; border-left: 3px solid var(--blue); color: var(--muted); font-size: 12px; line-height: 1.6; }
  .cast-layout { display: grid; grid-template-columns: minmax(100px, 0.8fr) minmax(0, 2fr); gap: 16px; }
  .cast-list { display: flex; flex-direction: column; gap: 5px; max-height: 325px; overflow-y: auto; padding-right: 5px; align-self: start; }
  .cast-list button { min-height: 36px; flex-shrink: 0; width: 100%; justify-content: flex-start; padding: 6px 8px; font-size: 12px; }
  .speaker-number { width: 18px; flex-shrink: 0; font-size: 10px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .speaker-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; text-align: left; }
  .cast-editor { min-width: 0; border-left: 1px solid var(--line); padding-left: 16px; }
  .name-row { display: flex; align-items: end; gap: 8px; margin-bottom: 14px; }
  .name-row > div { min-width: 0; flex: 1; }
  .name-row label { font-size: 12px; margin-top: 0; }
  input, textarea, select { width: 100%; box-sizing: border-box; }
  .text-count { margin-top: 7px; color: var(--muted); font-size: 11px; text-align: right; }
  .dialogue-list { display: grid; gap: 12px; }
  .dialogue-turn { border-top: 1px solid var(--line); padding-top: 10px; min-width: 0; }
  .turn-heading { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
  .turn-heading select { width: auto; flex: 1; min-width: 0; padding: 5px 8px; font-size: 12px; }
  .turn-number { font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .turn-actions { display: flex; gap: 3px; }
  .icon-button { width: 28px; height: 28px; padding: 0; flex-shrink: 0; }
  .add-turn { margin-top: 12px; }
  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); border: 0; }
  @media (max-width: 600px) {
    .cast-layout { grid-template-columns: minmax(0, 1fr); }
    .cast-list { flex-direction: row; max-height: none; overflow-x: auto; padding-bottom: 6px; }
    .cast-list button { width: 125px; }
    .cast-editor { border-left: 0; padding-left: 0; }
    .mode-switch button { padding: 8px; font-size: 12px; }
  }
</style>
