<script lang="ts">
  import { afterUpdate, createEventDispatcher, onDestroy, tick } from 'svelte';
  import { Mic, Square, Upload, Settings2, RotateCcw, AudioLines } from '@lucide/svelte';
  import { base64AudioBlob, uploadFile, uploadWav } from '$lib/api';
  import { browserDecodeToWav } from '$lib/audio';
  import { catalog } from '$lib/catalog';
  import { base64Bytes, readPcm16Wav } from '$lib/media/pcm_wav';
  import { loadModelChoices } from '$lib/model-settings/choices';
  import type { ModelSettings } from '$lib/model-settings/types';
  import { ChatWorkflow, appModel, type Artifact } from './workflow';
  import { AvatarPlayback, type VideoChunk } from './playback';

  export let busy = false;
  export let active = true;
  const dispatch = createEventDispatcher<{ models: { search: string; task: string } }>();
  let wasActive = false;
  let choices: ModelSettings[] = [];
  let asrId = '', brainId = '', avatarId = '';
  let voice = 'american_english_female';
  let personality = '';
  let speakingSpeed = 1;
  let settingsOpen = true;
  let portrait: File | null = null;
  let portraitUrl = '';
  let uploadedPortrait: { file: File; path: string } | null = null;
  let root: HTMLElement;
  let canvas: HTMLCanvasElement;
  let videoVisible = false;
  let scroll: HTMLDivElement;
  let status = 'Choose your models and a portrait to begin.';
  let error = '';
  let loading = true;
  let speaking = false;
  let recording = false;
  let requestingMicrophone = false;
  let captureGeneration = 0;
  let recorder: MediaRecorder | null = null;
  let microphone: MediaStream | null = null;
  let controller: AbortController | null = null;
  let audioContext: AudioContext | null = null;
  let playback: AvatarPlayback | null = null;
  let history: Artifact[] = [];
  let messages: { role: 'user' | 'assistant'; text: string; audio?: string; failed?: boolean }[] = [];
  let metrics: { wall: number; duration: number; firstVideo: number } | null = null;
  let chatIdentity = '';
  const workflow = new ChatWorkflow();
  $: asrChoices = choices.filter(c => c.model.task === 'asr');
  $: brainChoices = choices.filter(c => c.model.family === 'lfm3_audio' && c.model.task === 's2s' &&
    choices.some(t => t.model.family === 'lfm3_audio' && t.model.task === 'tts' && t.model.path === c.model.path));
  $: avatarChoices = choices.filter(c => c.model.family === 'soulx_flashhead');
  $: voices = catalog.find(c => c.family === 'lfm3_audio' && c.task === 'tts')?.builtin_voices || [];
  $: modelsReady = asrChoices.some(c => c.id === asrId) && brainChoices.some(c => c.id === brainId) && avatarChoices.some(c => c.id === avatarId);
  $: ready = !loading && modelsReady && Boolean(portrait);
  $: { if (active && !wasActive) void refresh(); wasActive = active; }
  $: if (ready && status === 'Choose your models and a portrait to begin.') status = 'Ready when you are.';
  $: if (!active && (busy || recording)) cancel();

  const label = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());

  async function refresh() {
    loading = true; error = '';
    try {
      choices = (await loadModelChoices()).filter(c => c.model.task !== 'asr' ||
        (c.model.family === 'owsm' && /base[ -_]102m/i.test(`${c.model.path} ${c.name}`)));
      if (!choices.some(c => c.id === asrId)) asrId = choices.find(c => c.model.task === 'asr')?.id || '';
      if (!choices.some(c => c.id === brainId)) brainId = choices.find(c => c.model.family === 'lfm3_audio' && c.model.task === 's2s')?.id || '';
      if (!choices.some(c => c.id === avatarId)) avatarId = choices.find(c => c.model.family === 'soulx_flashhead')?.id || '';
    } catch (e) { error = String(e); }
    finally { loading = false; }
  }

  function choosePortrait(file?: File) {
    if (!file) return;
    if (portraitUrl) URL.revokeObjectURL(portraitUrl);
    videoVisible = false; portrait = file; portraitUrl = URL.createObjectURL(file);
    uploadedPortrait = null;
  }

  function resetChat() {
    history = []; metrics = null; chatIdentity = '';
    for (const message of messages) if (message.audio) URL.revokeObjectURL(message.audio);
    messages = []; error = ''; status = 'Ready when you are.';
  }

  function cancel() {
    ++captureGeneration;
    controller?.abort();
    workflow.clear();
    playback?.stop();
    root?.querySelectorAll('audio').forEach(audio => audio.pause());
    playback = null;
    if (recorder && recorder.state !== 'inactive') { recorder.onstop = null; recorder.stop(); }
    microphone?.getTracks().forEach(track => track.stop());
    microphone = null; recorder = null; recording = false;
    status = 'Stopped';
  }

  async function record() {
    if (recording) { recorder?.stop(); return; }
    if (!ready || busy || requestingMicrophone) return;
    root?.querySelectorAll('audio').forEach(audio => audio.pause());
    requestingMicrophone = true;
    const generation = ++captureGeneration;
    error = '';
    try {
      audioContext ||= new AudioContext({ sampleRate: 24000, latencyHint: 'interactive' });
      await audioContext.resume();
      microphone = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      if (generation !== captureGeneration || !active) {
        microphone.getTracks().forEach(track => track.stop()); microphone = null; return;
      }
      const chunks: Blob[] = [];
      recorder = new MediaRecorder(microphone);
      recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      recorder.onstop = () => {
        const mime = recorder!.mimeType;
        microphone?.getTracks().forEach(track => track.stop());
        microphone = null; recorder = null; recording = false;
        void send(new File(chunks, 'question.webm', { type: mime }));
      };
      recorder.start(); recording = true; status = 'Listening — press again when you finish.';
    } catch (e) {
      microphone?.getTracks().forEach(track => track.stop()); microphone = null;
      error = `Microphone: ${String(e)}`;
    } finally { requestingMicrophone = false; }
  }

  async function send(file?: File) {
    if (!file || !ready || busy) return;
    root?.querySelectorAll('audio').forEach(audio => audio.pause());
    videoVisible = false;
    busy = true; error = ''; speaking = false; metrics = null;
    const aborter = new AbortController(); controller = aborter;
    const signal = aborter.signal;
    const start = performance.now();
    const speechChoice = choices.find(c => c.id === brainId)!;
    const speech = appModel(speechChoice, 'conversation');
    const recognition = appModel(choices.find(c => c.id === asrId)!, 'recognition');
    const avatar = appModel(choices.find(c => c.id === avatarId)!, 'avatar');
    const ttsChoice = choices.find(c => c.model.family === 'lfm3_audio' && c.model.task === 'tts' && c.model.path === speech.model.path);
    if (!ttsChoice) { error = 'Register an LFM3 TTS model using the same checkpoint as the conversation model.'; busy = false; return; }
    const synthesis = appModel(ttsChoice, 'voice');
    const identity = JSON.stringify([speech.model, personality.trim()]);
    if (chatIdentity && identity !== chatIdentity) resetChat();
    chatIdentity = identity;
    const user = messages.length;
    messages = [...messages, { role: 'user', text: 'Transcribing…' }, { role: 'assistant', text: '' }];
    await tick(); scroll?.scrollTo({ top: scroll.scrollHeight, behavior: 'smooth' });
    try {
      audioContext ||= new AudioContext({ sampleRate: 24000, latencyHint: 'interactive' }); await audioContext.resume();
      status = 'Listening to your message…';
      const selectedPortrait = portrait!;
      const [audio, image] = await Promise.all([
        browserDecodeToWav(file, 16000).then(wav => uploadWav(wav, signal)),
        uploadedPortrait?.file === selectedPortrait
          ? Promise.resolve(uploadedPortrait.path)
          : uploadFile(selectedPortrait, signal)
      ]);
      signal.throwIfAborted();
      uploadedPortrait = { file: selectedPortrait, path: image };
      let transcript = '';
      const recognitionOptions = { ...recognition.request.options };
      if (['owsm', 'owsm_ctc'].includes(recognition.model.family)) recognitionOptions.language = 'auto';
      const recognize = workflow.execute(recognition, { audio, options: recognitionOptions }, signal)
        .then(result => {
          transcript = String(result.text || '').trim();
          if (!transcript) throw new Error('The recognition model returned no transcript.');
          messages[user] = { ...messages[user], text: transcript }; messages = [...messages];
        });
      status = 'Thinking…';
      playback = new AvatarPlayback(audioContext, canvas,
        value => { if (speaking !== value) speaking = value; if (value) videoVisible = true; });
      let firstVideo = 0;
      const answer = workflow.speak(speech, synthesis, avatar, { audio, artifacts: history,
        text: personality.trim() ? `Respond with interleaved text and audio. ${personality.trim()}` : undefined,
        voice_id: voice, options: { speaking_speed: speakingSpeed, 'conversation.return_codes': true,
          'avatar.reference_image_file': image } }, signal, event => {
        const text = event.partial_text as { text?: string } | undefined;
        if (text?.text) {
          messages[user + 1].text += text.text; messages = [...messages];
          if (!speaking) status = 'Streaming your chosen voice…';
        }
        if (typeof event.audio === 'string') {
          const audio = readPcm16Wav(base64Bytes(event.audio));
          playback!.addAudio(audio.samples, audio.sampleRate);
        }
        for (const artifact of (event.artifacts as (Artifact & VideoChunk)[] | undefined) || []) {
          if (artifact.meta?.format === 'rgb24') {
            if (!firstVideo) firstVideo = performance.now() - start;
            playback!.add(artifact);
          }
        }
      });
      const [, video] = await Promise.all([recognize, answer]);
      const text = String(video.text || '').trim();
      const replyArtifact = (video.artifacts as Artifact[] | undefined)?.find(a => a.id === 'lfm3_audio.reply');
      if (!text || !replyArtifact) throw new Error('LFM3 returned no complete conversation reply.');
      messages[user + 1].text = text; messages = [...messages];
      signal.throwIfAborted();
      playback.end();
      if (typeof video.audio !== 'string') throw new Error('The streaming recipe returned no replay audio.');
      messages[user + 1].audio = URL.createObjectURL(base64AudioBlob(video.audio)); messages = [...messages];
      metrics = { wall: performance.now() - start, duration: playback.duration * 1000, firstVideo };
      history = [...history, { id: 'lfm3_audio.question', kind: 'custom', path: audio }, replyArtifact];
      status = 'Reply ready';
      await playback.done;
      if (!signal.aborted) status = 'Your turn';
    } catch (e) {
      aborter.abort(); workflow.clear(); playback?.stop();
      uploadedPortrait = null;
      messages[user + 1].failed = true; messages = [...messages];
      status = signal.aborted && (e as Error)?.name === 'AbortError' ? 'Stopped' : 'Could not finish this reply';
      if ((e as Error)?.name !== 'AbortError') error = e instanceof Error ? e.message : String(e);
    } finally { busy = false; controller = null; speaking = false; }
  }

  afterUpdate(() => {
    if (busy && scroll && scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 120) scroll.scrollTop = scroll.scrollHeight;
  });

  onDestroy(() => { cancel(); for (const m of messages) if (m.audio) URL.revokeObjectURL(m.audio); if (portraitUrl) URL.revokeObjectURL(portraitUrl); void audioContext?.close(); });
</script>

<section class="voice-chat" bind:this={root}>
  <header><div><span class="eyebrow">LFM3 · SOULX</span><h1>Voice chat</h1><p>Talk naturally, with a voice and avatar you choose.</p></div>
    <div class="tools"><button disabled={busy || recording} on:click={resetChat} aria-label="New conversation"><RotateCcw size={16}/> New chat</button><button on:click={() => settingsOpen = !settingsOpen} aria-expanded={settingsOpen}><Settings2 size={16}/> Setup</button></div>
  </header>
  <div class="setup">
  <section class="model-readiness" class:collapsed={!settingsOpen} aria-label="Model readiness">
    {#each [
      { name: 'ASR', available: asrChoices.some(c => c.id === asrId), selected: choices.find(c => c.id === asrId)?.name, missing: 'OWSM v4 Base 102M not installed', search: 'OWSM v4 Base 102M', task: 'asr' },
      { name: 'LFM3-Audio', available: brainChoices.some(c => c.id === brainId), selected: choices.find(c => c.id === brainId)?.name, missing: 'LFM3-Audio not installed', search: 'LFM3', task: '' },
      { name: 'SoulX', available: avatarChoices.some(c => c.id === avatarId), selected: choices.find(c => c.id === avatarId)?.name, missing: 'SoulX-FlashHead not installed', search: 'SoulX', task: '' }
    ] as requirement}
      <div class="readiness-row" class:ready={!loading && requirement.available} class:checking={loading}>
        {#if !loading && !requirement.available}
          <button class="install-link" aria-label={`Install ${requirement.name}`} title={requirement.missing} disabled={busy || recording} on:click={() => dispatch('models', { search: requirement.search, task: requirement.task })}><span class="status-dot" aria-hidden="true"></span>{requirement.name}</button>
        {:else}
          <span class="readiness-indicator" title={requirement.selected}><span class="status-dot" role="img" aria-label={loading ? 'Checking' : 'Ready'}></span>{requirement.name}</span>
        {/if}
      </div>
    {/each}
  </section>
  {#if settingsOpen}
      <label>Recognition<select aria-label="Recognition" bind:value={asrId} disabled={busy || recording}><option value="">Install OWSM v4 Base 102M</option>{#each asrChoices as c}<option value={c.id}>{c.name}</option>{/each}</select></label>
      <label>Voice<select aria-label="Voice" bind:value={voice} disabled={busy || recording}>{#each voices as v}<option value={v}>{label(v)}</option>{/each}</select></label>
      <label>Speaking speed<select aria-label="Speaking speed" bind:value={speakingSpeed} disabled={busy || recording}>
        <option value={1}>1× · Normal</option><option value={1.15}>1.15× · Faster</option><option value={1.3}>1.3× · Brisk</option><option value={1.5}>1.5× · Fast</option>
      </select></label>
      <div class="portrait-pick">
        <label>Portrait
          <span class="portrait-control">
            <span class="portrait-filename">{portrait?.name || 'Choose an image…'}</span>
            <Upload size={16}/>
            <input aria-label="Portrait" type="file" accept="image/*" disabled={busy || recording} on:change={e => choosePortrait(e.currentTarget.files?.[0])}/>
          </span>
        </label>
        {#if portraitUrl}
          <figure class="crop-preview">
            <img src={portraitUrl} alt="Avatar center-crop preview"/>
            <figcaption>512 × 512<br/>Center crop</figcaption>
          </figure>
        {/if}
      </div>
      <label class="personality">Personality (optional)
        <textarea aria-label="Personality" rows="2" placeholder="Describe how the avatar should respond, such as warm, concise and a little witty." bind:value={personality} disabled={busy || recording}></textarea>
      </label>
  {/if}
  </div>
  <div class="conversation">
    <div class="portrait-stage" class:speaking>
      {#if portraitUrl}<img src={portraitUrl} alt="Your conversation portrait"/>{:else}<div class="portrait-empty"><AudioLines size={48}/><span>Give your assistant a visual interface</span></div>{/if}
      <canvas bind:this={canvas} aria-label="Animated speaking portrait" class:visible={videoVisible}></canvas>
      <div class="stage-caption"><span class="live-dot" class:pulse={speaking || recording}></span>{speaking ? 'Speaking' : recording ? 'Listening' : 'LFM3-Audio'}<span class="private">ON YOUR SERVER</span></div>
    </div>
    <div class="chat-panel">
      <div class="chat-heading"><span>Conversation</span><span class="session-tag">{label(voice)}</span></div>
      <div class="messages" bind:this={scroll} aria-live="polite" aria-relevant="additions text">
        {#if !messages.length}<div class="welcome"><span class="greeting">Hello, you.</span><p>Ask a question, tell a story, or see where the conversation takes you.</p><div class="suggestion">“What’s something worth learning today?”</div></div>{/if}
        {#each messages as message}<article class:user={message.role === 'user'} class:failed={message.failed}><span class="speaker-name">{message.role === 'user' ? 'YOU' : 'LFM3'}</span><p>{message.text || 'Thinking…'}</p>{#if message.audio && !busy}<audio controls src={message.audio} preload="metadata"></audio>{/if}{#if message.failed}<small>Reply interrupted. You can try again.</small>{/if}</article>{/each}
      </div>
      {#if metrics}<div class="timing" data-session-wall-ms={metrics.wall} data-audio-duration-ms={metrics.duration} data-first-video-ms={metrics.firstVideo}>{(metrics.wall / 1000).toFixed(1)} s to generate · {(metrics.duration / 1000).toFixed(1)} s of speech · RTF {(metrics.wall / metrics.duration).toFixed(2)}</div>{/if}
      {#if error}<p class="error" role="alert">{error}</p>{/if}
      <div class="composer"><div class="status" role="status">{speaking ? 'Speaking…' : status}</div><div class="actions"><label class="upload" title="Upload an audio recording of your question"><Upload size={18}/><span>Upload recording</span><input aria-label="Send a recording" type="file" accept="audio/*" disabled={!ready || busy || recording} on:change={e => { void send(e.currentTarget.files?.[0]); e.currentTarget.value = ''; }}/></label>{#if busy}<button class="talk stop" on:click={cancel}><Square size={18}/> Stop</button>{:else}<button class="talk" class:recording on:click={record} disabled={!ready || loading || requestingMicrophone}>{#if recording}<Square size={18}/> Finish{:else}<Mic size={18}/> Talk{/if}</button>{/if}</div></div>
      <p class="footnote">Record with Talk, then press Finish to send. Or upload a recording of your question. New chat clears conversation context.</p>
    </div>
  </div>
</section>

<style>
  .model-readiness { grid-column: 1 / -1; display: flex; align-items: center; gap: clamp(.75rem, 3vw, 2rem); overflow-x: auto; padding-bottom: 1rem; border-bottom: .0625rem solid var(--line); }
  .model-readiness.collapsed { padding-bottom: 0; border-bottom: 0; }
  .readiness-row { flex-shrink: 0; }
  .readiness-indicator, .install-link { display: inline-flex; align-items: center; gap: .5rem; font-size: .8125rem; line-height: 1.5; white-space: nowrap; }
  .status-dot { width: .875rem; height: .875rem; flex-shrink: 0; border-radius: 50%; background: var(--danger); }
  .readiness-row.ready .status-dot { background: var(--green, #a6e3a1); }
  .readiness-row.checking .status-dot { background: var(--muted); }
  .install-link { padding: 0; border: 0; background: none; color: var(--text); }
  .install-link:hover { color: var(--cyan); }
  .voice-chat { max-width: 77.5rem; margin: 1.25rem auto 2.5rem; color: var(--text); }
  .voice-chat p { font-family: inherit; }
  header { display: flex; justify-content: space-between; gap: 1.25rem; align-items: center; margin-bottom: 1.625rem; }
  .eyebrow { font-size: .625rem; letter-spacing: .2em; color: var(--cyan); }
  h1 { font-size: 2.25rem; letter-spacing: -.04em; margin: .5rem 0; }
  header p { color: var(--muted); font-size: .8125rem; margin: 0; }
  .tools { display: flex; gap: .5rem; flex-shrink: 0; }
  .tools button { display: flex; gap: .4375rem; align-items: center; font-size: .75rem; }
  .setup {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 16rem), 1fr));
    gap: 1.25rem;
    align-items: start;
    padding: 1.25rem;
    margin-bottom: 1.375rem;
    background: var(--panel);
    border: .0625rem solid var(--line);
    border-radius: .875rem;
  }
  .setup label { display: flex; flex-direction: column; align-self: start; gap: .5rem; margin: 0; line-height: 1.4; font-size: .75rem; color: var(--muted); }
  select, input { width: 100%; min-width: 0; box-sizing: border-box; }
  .setup select, .portrait-control {
    box-sizing: border-box;
    min-block-size: 3em;
    margin: 0;
    padding: .625em .75em;
    font-size: .875rem;
    line-height: 1.5;
    background: var(--control-bg);
    color: var(--text);
    border: .0625rem solid var(--control-border);
    border-radius: .5rem;
  }
  .portrait-pick { min-width: 0; }
  .portrait-control { position: relative; display: flex; align-items: center; justify-content: space-between; gap: .75rem; cursor: pointer; }
  .portrait-control input { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; padding: 0; opacity: 0; cursor: pointer; }
  .portrait-control:focus-within { outline: .125rem solid var(--cyan); outline-offset: .125rem; }
  .portrait-filename { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .portrait-control :global(svg) { flex: none; }
  .personality { grid-column: 1 / -1; }
  .personality textarea { width: 100%; min-width: 0; resize: vertical; font: inherit; font-size: .875rem; line-height: 1.5; }
  .crop-preview { display: flex; align-items: center; gap: .75rem; margin: .75rem 0 0; }
  .crop-preview img { display: block; inline-size: 4rem; aspect-ratio: 1; object-fit: cover; object-position: center; border-radius: .625rem; border: .0625rem solid var(--line); }
  .crop-preview figcaption { margin: 0; line-height: 1.6; font-size: .6875rem; color: var(--muted); }
  .conversation {
    display: grid;
    grid-template-columns: minmax(0, .85fr) minmax(0, 1.15fr);
    block-size: clamp(30rem, 70dvh, 48rem);
    border: .0625rem solid var(--line);
    border-radius: 1.25rem;
    overflow: hidden;
    background: var(--panel);
  }
  .portrait-stage { position: relative; background: radial-gradient(ellipse at 50% 30%, #303945, #111922); min-height: 0; overflow: hidden; }
  .portrait-stage img, .portrait-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center; }
  .portrait-stage canvas { opacity: 0; background: #111922; }
  .portrait-stage canvas.visible { opacity: 1; }
  .portrait-stage:after { content: ''; position: absolute; inset: 50% 0 0; background: linear-gradient(transparent, #0c1119c9); pointer-events: none; }
  .portrait-empty { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1.25rem; color: #aab8c9; }
  .portrait-empty span { font-size: 1.125rem; }
  .stage-caption { position: absolute; bottom: 1.5rem; left: 1.5rem; right: 1.5rem; z-index: 1; display: flex; align-items: center; gap: .625rem; color: white; font-size: .75rem; }
  .live-dot { width: .4375rem; aspect-ratio: 1; flex: none; border-radius: 50%; background: #6b8195; }
  .live-dot.pulse { background: #93dfbd; box-shadow: 0 0 1rem #93dfbd; }
  .private { margin-left: auto; font-size: .5625rem; letter-spacing: .12em; color: #bbc5cf; }
  .chat-panel { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  .chat-heading { padding: 1.4375rem 1.5625rem; border-bottom: .0625rem solid var(--line); display: flex; justify-content: space-between; gap: 1rem; font-size: .8125rem; }
  .session-tag { font-size: .6875rem; color: var(--muted); text-align: right; }
  .messages { padding: 1.5rem; flex: 1; overflow: auto; min-height: 0; }
  .welcome { padding: 2.8125rem .75rem; color: var(--muted); }
  .greeting { font-size: 1.75rem; letter-spacing: -.03em; color: var(--text); }
  .welcome p { font-size: .875rem; line-height: 1.8; max-width: 24ch; }
  .suggestion { margin-top: 1.875rem; font-size: .75rem; border: .0625rem solid var(--line); border-radius: .625rem; padding: .875rem; display: inline-block; }
  article { max-width: 92%; margin: 0 0 1.375rem; padding: 1rem 1.125rem; border-radius: .875rem; background: var(--bg); }
  article.user { margin-left: auto; background: color-mix(in srgb, var(--cyan) 9%, var(--panel)); border: .0625rem solid color-mix(in srgb, var(--cyan) 18%, transparent); }
  .speaker-name { font-size: .5625rem; letter-spacing: .12em; color: var(--muted); }
  article p { font-size: .875rem; line-height: 1.7; margin: .4375rem 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
  article audio { color-scheme: dark; width: 100%; height: 2rem; margin-top: .75rem; }
  article.failed { border: .0625rem solid var(--danger); }
  article small, .error { color: var(--danger); }
  .error { margin: 0; padding: .75rem 1.5rem; font-size: .75rem; }
  .composer { border-top: .0625rem solid var(--line); padding: 1.125rem 1.5rem; display: flex; align-items: center; gap: 1rem; justify-content: space-between; }
  .status { font-size: .75rem; color: var(--muted); }
  .actions { display: flex; align-items: center; gap: .75rem; flex-shrink: 0; }
  .talk { display: flex; align-items: center; gap: .5625rem; background: var(--cyan); color: #07171c; border: 0; border-radius: 2em; padding: .8125em 1.375em; font-weight: 650; min-inline-size: 7em; justify-content: center; }
  .talk.stop, .talk.recording { background: #f4a99e; }
  .upload { position: relative; cursor: pointer; display: flex; align-items: center; gap: .5em; margin: 0; font-size: .75rem; color: var(--muted); }
  .upload input { position: absolute; inset: 0; opacity: 0; width: 100%; cursor: pointer; }
  .footnote { font-size: .625rem; color: var(--muted); padding: 0 1.5rem 1rem; margin: 0; }
  .timing { padding: .5rem 1.5rem; font-size: .625rem; color: var(--muted); }
  @media (max-width: 50rem) {
    header { align-items: flex-start; flex-wrap: wrap; }
    h1 { font-size: 1.75rem; }
    .conversation { grid-template-columns: 1fr; block-size: auto; }
    .portrait-stage { aspect-ratio: 1; max-block-size: 60dvh; }
    .chat-panel { block-size: min(65dvh, 38rem); min-block-size: 24rem; }
    .tools { flex-wrap: wrap; }
    .composer { flex-wrap: wrap; }
    .actions { flex-wrap: wrap; max-width: 100%; }
  }
</style>
