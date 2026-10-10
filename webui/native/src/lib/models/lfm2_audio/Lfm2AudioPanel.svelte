<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import { browserDecodeToWav } from '$lib/audio';
  import { runTask, taskStreamEvents, TaskStreamClosedError, uploadWav } from '$lib/api';
  import type { PanelRunContext, PanelRunOutcome, RegisterPanelRunner } from '$lib/models/panel_runner';
  import type { CatalogEntry, LoadedModel, ParamSpec, ServerHealth } from '$lib/types';
  import { clearConversation, conversation, keptTurns, releaseTurn, type ReplyArtifact, type Turn } from './conversation';
  import { base64Bytes, readPcm16Wav, StreamPlayer } from './stream_audio';

  export let modelId = '';
  export let busy = false;
  export let server: ServerHealth | null = null;
  export let loadedModels: LoadedModel[] = [];
  export let catalogEntries: CatalogEntry[] = [];
  export let paramSpecs: ParamSpec[] = [];
  export let sourceFile: File | null = null;
  export let clearSource: () => void = () => {};
  export let log: (message: string) => void = () => {};
  export let setPanelRunner: RegisterPanelRunner = () => () => {};

  const player = new StreamPlayer();
  let turns: Turn[] = conversation.turns;
  let conversationModelId = conversation.modelId;
  let liveAudio = true;
  let playing = false;
  let blocked = false;
  let unavailable = false;
  let list: HTMLOListElement | null = null;
  let current: AbortController | null = null;
  let shownModelId = modelId;

  player.onchange = () => { playing = player.playing; };

  $: keptCount = turns.filter((turn) => turn.state === 'done' && turn.reply && !turn.leftOut).length;
  $: otherModel = turns.length > 0 && conversationModelId !== '' && conversationModelId !== modelId;
  // An entry whose id is not the catalog's gets the family's parameters,
  // which are the ASR ones.
  $: familySpecs = paramSpecs.some((spec) => spec.name === 'audio_chunk_mode');
  $: if (modelId !== shownModelId) {
    shownModelId = modelId;
    player.stop();
  }
  $: if (!liveAudio) player.stop();

  function displayName(id: string) {
    return catalogEntries.find((entry) => entry.id === id)?.display_name || id;
  }

  function refresh() {
    const follow = !list || list.scrollTop + list.clientHeight >= list.scrollHeight - 24;
    turns = conversation.turns;
    conversationModelId = conversation.modelId;
    if (follow) tick().then(() => { if (list) list.scrollTop = list.scrollHeight; });
  }

  function checkAborted(signal: AbortSignal) {
    if (signal.aborted) throw new DOMException('The turn was stopped.', 'AbortError');
  }

  function isReplyArtifact(entry: unknown): entry is ReplyArtifact {
    const candidate = entry as Partial<ReplyArtifact> | null;
    return typeof candidate === 'object' && candidate !== null && candidate.id === 'lfm2_audio.reply' &&
      typeof candidate.kind === 'string' && typeof candidate.payload === 'string';
  }

  function resultJson(result: Record<string, unknown>) {
    return JSON.stringify(result, (key, value) =>
      (key === 'audio' || key === 'payload') && typeof value === 'string'
        ? `<base64 data: ${value.length} chars>` : value, 2);
  }

  // The new question goes in as its upload path; each earlier turn goes with
  // it as its question's path and its reply artifact as returned.
  function requestBody(context: PanelRunContext, path: string, history: Turn[]) {
    const options: Record<string, unknown> = { ...context.options, return_codes: true };
    // An entry whose id is not the catalog's gets the family's ASR
    // parameters: S2S turns their chunking options away, and their
    // max_tokens would clash with the request's Max tokens.
    if (familySpecs) for (const spec of paramSpecs) delete options[spec.name];
    const request: Record<string, unknown> = { audio: path, seed: context.seed, options };
    if (context.maxTokens !== undefined) request.max_tokens = context.maxTokens;
    if (context.language) request.language = context.language;
    if (history.length) {
      request.artifacts = history.flatMap((turn) => [
        { id: 'lfm2_audio.question', kind: 'custom', path: turn.questionPath },
        turn.reply
      ]);
    }
    return { model: context.modelId, request };
  }

  function finishTurn(turn: Turn, result: Record<string, unknown>) {
    if (typeof result.text === 'string') turn.replyText = result.text;
    let audio: Blob | null = null;
    if (typeof result.audio === 'string') {
      audio = new Blob([base64Bytes(result.audio)], { type: 'audio/wav' });
      turn.replyUrl = URL.createObjectURL(audio);
    }
    const artifact = (Array.isArray(result.artifacts) ? result.artifacts : []).find(isReplyArtifact);
    if (artifact) {
      turn.reply = { id: artifact.id, kind: artifact.kind, payload: artifact.payload };
      if (artifact.meta) turn.reply.meta = artifact.meta;
      turn.cut = artifact.meta?.ended === 'false';
      turn.steps = artifact.meta?.steps || '';
    } else {
      turn.message = 'The server returned no reply artifact, so this turn does not go with the next question.';
    }
    return audio;
  }

  async function runTurn(context: PanelRunContext): Promise<PanelRunOutcome> {
    // Before anything is awaited, so that it is still inside the Run click
    // or key press: a browser starts audio only then.
    player.beginTurn(liveAudio);
    unavailable = player.unavailable;
    blocked = false;
    const started = performance.now();

    if (conversation.modelId && conversation.modelId !== context.modelId) {
      clearConversation();
      log(`New LFM2.5-Audio conversation with ${displayName(context.modelId)}.`);
    }
    conversation.modelId = context.modelId;
    // A failed or stopped turn is not part of the conversation; this Run
    // takes its place.
    for (const turn of conversation.turns) {
      if (turn.state === 'failed' || turn.state === 'stopped') releaseTurn(turn);
    }
    conversation.turns = conversation.turns.filter((turn) => turn.state !== 'failed' && turn.state !== 'stopped');
    const history = keptTurns();
    const turn: Turn = {
      key: conversation.nextKey++, state: server?.ui_management ? 'loading' : 'uploading', seed: context.seed,
      questionPath: '', questionUrl: '', replyText: '', replyUrl: '', reply: null, cut: false, steps: '',
      leftOut: false, live: true, firstAudioMs: null, message: ''
    };
    conversation.turns = [...conversation.turns, turn];
    refresh();

    const controller = new AbortController();
    const abort = () => controller.abort();
    if (context.signal.aborted) abort();
    else context.signal.addEventListener('abort', abort);
    current = controller;
    const signal = controller.signal;
    let stage: 'load' | 'read' | 'upload' | 'run' = 'load';
    try {
      if (server?.ui_management) {
        await context.ensureLoadedMode('streaming');
        checkAborted(signal);
        if (modelId !== context.modelId) throw new Error('The model changed while it loaded; press Run again.');
      } else {
        const registered = loadedModels.find((model) => model.id === context.modelId);
        if (!registered) throw new Error(`This server does not list ${context.modelId}.`);
        turn.live = registered.mode === 'streaming';
      }

      stage = 'read';
      turn.state = 'uploading';
      context.setStatus('Uploading the question…');
      refresh();
      const wav = await browserDecodeToWav(context.question);
      checkAborted(signal);
      stage = 'upload';
      const path = await uploadWav(wav, signal);
      checkAborted(signal);
      turn.questionPath = path;
      turn.questionUrl = URL.createObjectURL(wav);

      stage = 'run';
      const body = requestBody(context, path, history);
      turn.state = 'waiting';
      context.setStatus(turn.live ? 'Waiting for the reply…'
        : 'Waiting for the whole reply: this server runs the entry offline, so the reply does not stream.');
      refresh();
      let result: Record<string, unknown> | null = null;
      if (turn.live) {
        let streamed = 0;
        for await (const message of taskStreamEvents(body, signal)) {
          if (message.type === 'task.stream.done') {
            result = message.result;
            continue;
          }
          const event = message.event;
          const partial = event.partial_text as { text?: unknown } | undefined;
          if (typeof partial?.text === 'string') turn.replyText += partial.text;
          if (typeof event.audio === 'string') {
            const chunk = readPcm16Wav(base64Bytes(event.audio));
            streamed += chunk.samples.length / chunk.sampleRate;
            const delay = player.push(chunk.samples, chunk.sampleRate);
            if (turn.firstAudioMs === null) {
              turn.firstAudioMs = performance.now() - started + (delay ?? 0) * 1000;
              blocked = player.blocked;
              // Queued chunks would wait for the browser and play late.
              if (blocked) player.stop();
            }
          }
          turn.state = 'streaming';
          context.setStatus(`Streaming the reply: ${streamed.toFixed(1)} s`);
          refresh();
        }
      } else {
        result = await runTask(body, signal);
      }
      checkAborted(signal);
      if (!result) throw new TaskStreamClosedError();
      if (!conversation.turns.includes(turn)) throw new DOMException('The conversation was reset.', 'AbortError');
      const audio = finishTurn(turn, result);
      turn.state = 'done';
      // A question recorded while the reply played stays for the next turn.
      if (sourceFile === context.question) clearSource();
      return { audio, text: turn.replyText, json: resultJson(result) };
    } catch (error) {
      player.stop();
      if (signal.aborted || (error as Error)?.name === 'AbortError') {
        turn.state = 'stopped';
        turn.message = 'Stopped before the reply finished; this turn is not part of the conversation.';
        throw (error as Error)?.name === 'AbortError' ? error : new DOMException('The turn was stopped.', 'AbortError');
      }
      if (error instanceof TaskStreamClosedError) {
        turn.state = 'stopped';
        turn.message = 'The connection closed before the reply finished; this turn is not part of the conversation.';
        throw new Error(turn.message);
      }
      let message = error instanceof Error ? error.message : String(error);
      if (stage === 'read') message = `Could not read the question: ${message}`;
      else if (stage === 'upload') message = `Upload failed: ${message}`;
      else if (stage === 'run' && history.some((kept) => message.includes(kept.questionPath))) {
        message += ' The server no longer has the questions this conversation uploaded, as after a restart;' +
          ' start a new conversation.';
      }
      turn.state = 'failed';
      turn.message = message;
      throw new Error(message);
    } finally {
      context.signal.removeEventListener('abort', abort);
      if (current === controller) current = null;
      player.endTurn();
      refresh();
    }
  }

  function newConversation() {
    if (busy) return;
    player.stop();
    clearConversation();
    refresh();
  }

  function toggleLeftOut(turn: Turn) {
    if (busy) return;
    turn.leftOut = !turn.leftOut;
    refresh();
  }

  function stopAudio() {
    player.stop();
  }

  function badge(turn: Turn) {
    if (turn.state === 'loading') return 'loading the model';
    if (turn.state !== 'done') return turn.state;
    if (turn.leftOut) return 'left out';
    return turn.cut ? 'cut at max_tokens' : 'done';
  }

  function details(turn: Turn) {
    const parts = [`seed ${turn.seed}`];
    if (turn.firstAudioMs !== null) parts.push(`first audio ${(turn.firstAudioMs / 1000).toFixed(2)} s`);
    if (turn.steps) parts.push(`${turn.steps} steps`);
    if (!turn.live) parts.push('not streamed');
    return parts.join(' · ');
  }

  onMount(() => setPanelRunner(runTurn));

  onDestroy(() => {
    // The conversation stays in its module; a turn in flight stops.
    current?.abort();
    player.close();
  });
</script>

<section class="lfm2-conversation" aria-label="Conversation">
  <div class="conversation-head">
    <div>
      <strong>Conversation</strong>
      <small>{keptCount
        ? `${keptCount} earlier ${keptCount === 1 ? 'turn goes' : 'turns go'} with the next question`
        : 'The next question starts the conversation'}</small>
    </div>
    <div class="conversation-actions">
      {#if playing}<button type="button" on:click={stopAudio}>Stop audio</button>{/if}
      <button type="button" disabled={busy || !turns.length} on:click={newConversation}>New conversation</button>
    </div>
  </div>
  {#if otherModel}
    <small class="note">These turns are with {displayName(conversationModelId)}. Run starts a new conversation with {displayName(modelId)}.</small>
  {/if}
  {#if familySpecs}
    <small class="note">This entry's id is not the catalog's (lfm2-audio-s2s or lfm2-audio-jp-s2s), so Model parameters shows the family's ASR settings; turns leave them out and use Max tokens.</small>
  {/if}

  {#if turns.length}
    <ol class="turns" bind:this={list}>
      {#each turns as turn, index (turn.key)}
        <li data-state={turn.state} class:left-out={turn.leftOut}>
          <div class="turn-head">
            <strong>Turn {index + 1}</strong>
            <span class="badge" class:bad={turn.state === 'failed' || turn.state === 'stopped'} aria-live="polite">{badge(turn)}</span>
            <small>{details(turn)}</small>
            {#if turn.state === 'done' && turn.reply}
              <button type="button" disabled={busy} on:click={() => toggleLeftOut(turn)}
                aria-label={`${turn.leftOut ? 'Include' : 'Leave out'} turn ${index + 1}`}>
                {turn.leftOut ? 'Include' : 'Leave out'}
              </button>
            {/if}
          </div>
          <div class="turn-row">
            <span>You</span>
            {#if turn.questionUrl}
              <audio controls src={turn.questionUrl} on:play={stopAudio} aria-label={`Turn ${index + 1} question`}></audio>
            {:else}
              <small>…</small>
            {/if}
          </div>
          <div class="turn-row">
            <span>Reply</span>
            <div class="reply">
              <p>{turn.replyText || (turn.state === 'done' ? '(no text)' : '…')}</p>
              {#if turn.replyUrl}
                <div class="reply-audio">
                  <audio controls src={turn.replyUrl} on:play={stopAudio} aria-label={`Turn ${index + 1} reply`}></audio>
                  <a href={turn.replyUrl} download={`${conversationModelId}-turn-${index + 1}-reply.wav`}>Save WAV</a>
                </div>
              {/if}
            </div>
          </div>
          {#if turn.message}<small class="turn-message">{turn.message}</small>{/if}
        </li>
      {/each}
    </ol>
  {:else}
    <p class="empty">Record or choose a spoken question under Source audio, then press Run. Each Run adds a turn, and the earlier turns go with it.</p>
  {/if}

  <label class="toggle">
    <input type="checkbox" bind:checked={liveAudio} />
    <span></span>Play replies as they stream in
  </label>
  {#if unavailable}
    <small class="note">This browser cannot play the stream; each reply is in its player when it finishes.</small>
  {:else if blocked}
    <small class="note">The browser did not let live audio start; each reply is in its player when it finishes.</small>
  {/if}
</section>

<style>
  .lfm2-conversation { display: grid; gap: 8px; margin: 4px 0 6px; padding: 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--card-bg); }
  .conversation-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
  .conversation-head strong, .conversation-head small { display: block; }
  .conversation-head strong { color: var(--text-strong); font-size: 12px; }
  .conversation-head small, .note, .empty, .turn-head small, .turn-row small { color: var(--muted); font-size: 10px; line-height: 1.4; }
  .conversation-actions { display: flex; gap: 6px; }
  .conversation-actions button, .turn-head button { padding: 5px 9px; font-size: 11px; white-space: nowrap; }
  .empty { margin: 0; }
  .turns { display: grid; gap: 8px; max-height: 460px; margin: 0; padding: 0; overflow-y: auto; list-style: none; }
  .turns li { display: grid; gap: 6px; padding: 8px; border: 1px solid var(--line); border-radius: 7px; }
  .turns li.left-out { opacity: .55; }
  .turn-head { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
  .turn-head strong { font-size: 11px; }
  .turn-head button { margin-left: auto; }
  .badge { padding: 1px 7px; border: 1px solid var(--active-border); border-radius: 999px; color: var(--cyan); font-size: 10px; }
  .badge.bad { border-color: rgba(255,119,139,.45); color: var(--danger); }
  .turn-row { display: grid; grid-template-columns: 38px minmax(0, 1fr); align-items: start; gap: 6px; }
  .turn-row > span { padding-top: 3px; color: var(--text-label); font-size: 10px; font-weight: 650; }
  .reply p { margin: 2px 0 4px; color: var(--text); font-size: 12px; line-height: 1.45; white-space: pre-wrap; }
  .reply-audio { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 8px; }
  .reply-audio a { color: var(--cyan); font-size: 11px; text-decoration: none; }
  .turn-message { color: var(--danger); font-size: 10px; line-height: 1.4; }
  .lfm2-conversation .toggle { margin-top: 2px; }
</style>
