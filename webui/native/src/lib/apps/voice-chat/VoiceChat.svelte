<script lang="ts">
  import { afterUpdate, onDestroy, onMount, tick } from 'svelte';
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
  let choices: ModelSettings[] = [];
  let asrId = '', brainId = '', avatarId = '';
  let voice = 'american_english_female';
  let settingsOpen = true;
  let portrait: File | null = null;
  let portraitUrl = '';
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
  $: brainChoices = choices.filter(c => c.model.family === 'lfm3_audio' && c.model.task === 's2s');
  $: avatarChoices = choices.filter(c => c.model.family === 'soulx_flashhead');
  $: voices = catalog.find(c => c.family === 'lfm3_audio' && c.task === 'tts')?.builtin_voices || [];
  $: ready = Boolean(asrId && brainId && avatarId && portrait);
  $: if (ready && status === 'Choose your models and a portrait to begin.') status = 'Ready when you are.';
  $: if (!active && (busy || recording)) cancel();

  const label = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());

  async function refresh() {
    loading = true; error = '';
    try {
      choices = await loadModelChoices();
      asrId = choices.find(c => c.model.task === 'asr' && c.model.family === 'owsm' && /base[ -_]102m/i.test(`${c.model.path} ${c.name}`))?.id
        || choices.find(c => c.model.task === 'asr' && c.model.family === 'owsm_ctc')?.id
        || choices.find(c => c.model.task === 'asr' && c.model.family === 'lfm3_audio')?.id
        || choices.find(c => c.model.task === 'asr' && c.model.family === 'crisperwhisper')?.id || '';
      brainId = choices.find(c => c.model.family === 'lfm3_audio' && c.model.task === 's2s')?.id || '';
      avatarId = choices.find(c => c.model.family === 'soulx_flashhead')?.id || '';
    } catch (e) { error = String(e); }
    finally { loading = false; }
  }

  function choosePortrait(file?: File) {
    if (!file) return;
    if (portraitUrl) URL.revokeObjectURL(portraitUrl);
    videoVisible = false; portrait = file; portraitUrl = URL.createObjectURL(file);
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
    const identity = JSON.stringify(speech.model);
    if (chatIdentity && identity !== chatIdentity) resetChat();
    chatIdentity = identity;
    const user = messages.length;
    messages = [...messages, { role: 'user', text: 'Transcribing…' }, { role: 'assistant', text: '' }];
    await tick(); scroll?.scrollTo({ top: scroll.scrollHeight, behavior: 'smooth' });
    try {
      audioContext ||= new AudioContext({ sampleRate: 24000, latencyHint: 'interactive' }); await audioContext.resume();
      status = 'Listening to your message…';
      const wav = await browserDecodeToWav(file, 16000);
      const audio = await uploadWav(wav, signal);
      signal.throwIfAborted();
      const image = await uploadFile(portrait!, signal);
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
        voice_id: voice, options: { 'conversation.return_codes': true,
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
      messages[user + 1].failed = true; messages = [...messages];
      status = signal.aborted && (e as Error)?.name === 'AbortError' ? 'Stopped' : 'Could not finish this reply';
      if ((e as Error)?.name !== 'AbortError') error = e instanceof Error ? e.message : String(e);
    } finally { busy = false; controller = null; speaking = false; }
  }

  afterUpdate(() => {
    if (busy && scroll && scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 120) scroll.scrollTop = scroll.scrollHeight;
  });
  onMount(refresh);
  onDestroy(() => { cancel(); for (const m of messages) if (m.audio) URL.revokeObjectURL(m.audio); if (portraitUrl) URL.revokeObjectURL(portraitUrl); void audioContext?.close(); });
</script>

<section class="voice-chat" bind:this={root}>
  <header><div><span class="eyebrow">LFM3 · SOULX</span><h1>A face to talk to.</h1><p>Your words. A familiar voice. A conversation that moves.</p></div>
    <div class="tools"><button disabled={busy || recording} on:click={resetChat} aria-label="New conversation"><RotateCcw size={16}/> New chat</button><button on:click={() => settingsOpen = !settingsOpen} aria-expanded={settingsOpen}><Settings2 size={16}/> Setup</button></div>
  </header>
  {#if settingsOpen}
    <div class="setup">
      <label>Recognition<select aria-label="Recognition" bind:value={asrId} disabled={busy || recording}><option value="">Choose an ASR model</option>{#each asrChoices as c}<option value={c.id}>{c.name}</option>{/each}</select></label>
      <label>Conversation<select aria-label="Conversation" bind:value={brainId} disabled={busy || recording}><option value="">Choose LFM3-Audio</option>{#each brainChoices as c}<option value={c.id}>{c.name}</option>{/each}</select></label>
      <label>Voice<select aria-label="Voice" bind:value={voice} disabled={busy || recording}>{#each voices as v}<option value={v}>{label(v)}</option>{/each}</select></label>
      <label>Avatar model<select aria-label="Avatar model" bind:value={avatarId} disabled={busy || recording}><option value="">Choose SoulX-FlashHead</option>{#each avatarChoices as c}<option value={c.id}>{c.name}</option>{/each}</select></label>
      <div class="portrait-pick">
        <label>Portrait<input type="file" accept="image/*" disabled={busy || recording} on:change={e => choosePortrait(e.currentTarget.files?.[0])}/></label>
        {#if portraitUrl}
          <figure class="crop-preview">
            <img src={portraitUrl} alt="Avatar center-crop preview"/>
            <figcaption>512 × 512 · center crop</figcaption>
          </figure>
        {/if}
      </div>
      <button disabled={loading || busy || recording} on:click={refresh}>Refresh models</button>
      {#if !loading && (!asrChoices.length || !brainChoices.length || !avatarChoices.length)}<p class="setup-note">Install or register ASR, LFM3-Audio (S2S and TTS), and SoulX-FlashHead models first.</p>{/if}
    </div>
  {/if}
  <div class="conversation">
    <div class="portrait-stage" class:speaking>
      {#if portraitUrl}<img src={portraitUrl} alt="Your conversation portrait"/>{:else}<div class="portrait-empty"><AudioLines size={48}/><span>Give your conversation a face</span><small>Choose a portrait in Setup</small></div>{/if}
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
      <div class="composer"><div class="status" role="status">{speaking ? 'Speaking…' : status}</div><div class="actions"><label class="upload" title="Send an audio recording"><Upload size={18}/><span class="sr-only">Send a recording</span><input aria-label="Send a recording" type="file" accept="audio/*" disabled={!ready || busy || recording} on:change={e => { void send(e.currentTarget.files?.[0]); e.currentTarget.value = ''; }}/></label>{#if busy}<button class="talk stop" on:click={cancel}><Square size={18}/> Stop</button>{:else}<button class="talk" class:recording on:click={record} disabled={!ready || loading || requestingMicrophone}>{#if recording}<Square size={18}/> Finish{:else}<Mic size={18}/> Talk{/if}</button>{/if}</div></div>
      <p class="footnote">Press Talk to record. Press Finish to send. New chat clears conversation context.</p>
    </div>
  </div>
</section>

<style>
  .voice-chat{max-width:1240px;margin:20px auto 40px;color:var(--text)}header{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:26px}.eyebrow{font-size:10px;letter-spacing:.2em;color:var(--cyan)}h1{font-size:36px;letter-spacing:-1.4px;margin:8px 0}header p{color:var(--muted);font-size:13px;margin:0}.tools{display:flex;gap:8px}.tools button{display:flex;gap:7px;align-items:center;font-size:12px}.setup{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;padding:20px;margin-bottom:22px;background:var(--panel);border:1px solid var(--border);border-radius:14px}.setup label{display:flex;flex-direction:column;gap:7px;font-size:11px;color:var(--muted)}select,input{width:100%;min-width:0}.portrait-pick{grid-column:span 2}.setup button{align-self:end}.setup-note{grid-column:1/-1;color:var(--muted);font-size:12px;margin:0}.conversation{display:grid;grid-template-columns:minmax(280px,.85fr) minmax(360px,1.15fr);border:1px solid var(--border);border-radius:20px;overflow:hidden;background:var(--panel);min-height:600px}.portrait-stage{position:relative;background:radial-gradient(ellipse at 50% 30%,#303945,#111922);min-height:560px;overflow:hidden}.portrait-stage img,.portrait-stage canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center}.portrait-stage canvas{opacity:0;object-fit:cover;background:#111922}.portrait-stage canvas.visible{opacity:1}.portrait-stage:after{content:'';position:absolute;inset:50% 0 0;background:linear-gradient(transparent,#0c1119c9);pointer-events:none}.portrait-empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px;color:#aab8c9}.portrait-empty span{font-size:18px}.portrait-empty small{font-size:12px}.stage-caption{position:absolute;bottom:24px;left:24px;right:24px;z-index:1;display:flex;align-items:center;gap:10px;color:white;font-size:12px}.live-dot{width:7px;height:7px;border-radius:50%;background:#6b8195}.live-dot.pulse{background:#93dfbd;box-shadow:0 0 16px #93dfbd}.private{margin-left:auto;font-size:9px;letter-spacing:.12em;color:#bbc5cf}.chat-panel{display:flex;flex-direction:column;min-width:0}.chat-heading{padding:23px 25px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;font-size:13px}.session-tag{font-size:11px;color:var(--muted)}.messages{padding:24px;flex:1;overflow:auto;max-height:520px;min-height:330px}.welcome{padding:45px 12px;color:var(--muted)}.greeting{font-size:28px;letter-spacing:-.8px;color:var(--text)}.welcome p{font-size:14px;line-height:1.8;max-width:300px}.suggestion{margin-top:30px;font-size:12px;border:1px solid var(--border);border-radius:10px;padding:14px;display:inline-block}article{max-width:92%;margin:0 0 22px;padding:16px 18px;border-radius:14px;background:var(--bg)}article.user{margin-left:auto;background:color-mix(in srgb,var(--cyan) 9%,var(--panel));border:1px solid color-mix(in srgb,var(--cyan) 18%,transparent)}.speaker-name{font-size:9px;letter-spacing:.12em;color:var(--muted)}article p{font-size:14px;line-height:1.7;margin:7px 0 0;white-space:pre-wrap}article audio{color-scheme:dark;width:100%;height:32px;margin-top:12px}article.failed{border:1px solid var(--danger)}article small,.error{color:var(--danger)}.error{margin:0;padding:12px 24px;font-size:12px}.composer{border-top:1px solid var(--border);padding:18px 24px;display:flex;align-items:center;gap:16px;justify-content:space-between}.status{font-size:12px;color:var(--muted)}.actions{display:flex;align-items:center;gap:12px}.talk{display:flex;align-items:center;gap:9px;background:var(--cyan);color:#07171c;border:0;border-radius:30px;padding:13px 22px;font-weight:650;min-width:110px;justify-content:center}.talk.stop,.talk.recording{background:#f4a99e}.upload{position:relative;cursor:pointer;display:flex;color:var(--muted)}.upload input{position:absolute;inset:0;opacity:0;width:24px;cursor:pointer}.footnote{font-size:10px;color:var(--muted);padding:0 24px 16px;margin:0}.timing{padding:8px 24px;font-size:10px;color:var(--muted)}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}@media(max-width:800px){header{align-items:flex-start}h1{font-size:28px}.conversation{grid-template-columns:1fr}.portrait-stage{min-height:340px;max-height:420px}.setup{grid-template-columns:repeat(2,minmax(0,1fr))}.messages{max-height:430px}.tools button{font-size:0}.tools :global(svg){width:18px}}
  .crop-preview{margin:12px 0 0}.crop-preview img{display:block;width:160px;height:160px;object-fit:cover;object-position:center;border-radius:10px;border:1px solid var(--border)}.crop-preview figcaption{margin-top:7px;font-size:11px;color:var(--muted)}
</style>
