<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Play, Square, Plus, Minus, RotateCcw } from '@lucide/svelte';
  import { waveformEnvelope as envelope } from './audio';
  import { selectedTake, chunkReady, type Chunk, type TtsProject } from './project';
  export let project: TtsProject;
  export let chunk: Chunk;
  export let next: Chunk | undefined;
  export let index: number;
  export let increment: number;
  export let disabled = false;
  export let active = true;
  export let onchange: () => void;
  let mounted=false;
  let context: AudioContext;
  let loadedLeft: Blob | undefined;
  let loadedRight: Blob | undefined;
  let left: AudioBuffer | undefined;
  let right: AudioBuffer | undefined;
  let leftEnvelope: Array<[number,number]> = [];
  let rightEnvelope: Array<[number,number]> = [];
  let loading=false;
  let error='';
  let revision=0;
  let sources: AudioBufferSourceNode[]=[];
  let playing=false;
  let progress=0;
  let animation=0;
  let startedAt=0;
  let playbackGap=0;
  $: take=selectedTake(chunk);
  $: nextTake=next && selectedTake(next);
  $: gap=chunk.pauseAfter ?? project.assembly.pause;
  $: leftSeconds=left ? Math.min(2,left.duration) : 0;
  $: rightSeconds=right ? Math.min(2,right.duration) : 0;
  $: total=leftSeconds+(next ? gap+rightSeconds : 0);
  $: leftWidth=total ? leftSeconds/total*1000 : 0;
  $: gapWidth=total && next ? gap/total*1000 : 0;
  $: leftAudio=take?.rendered?.audio || take?.audio;
  $: rightAudio=nextTake?.rendered?.audio || nextTake?.audio;
  $: if(mounted && (leftAudio!==loadedLeft || rightAudio!==loadedRight))void load(leftAudio,rightAudio);
  $: if(playing && (!active || disabled || gap!==playbackGap))stopPlayback();
  async function load(a: Blob | undefined,b: Blob | undefined) {
    const current=++revision;loadedLeft=a;loadedRight=b;stopPlayback();left=undefined;right=undefined;leftEnvelope=[];rightEnvelope=[];loading=true;error='';
    try {
      const first=a ? await context.decodeAudioData(await a.arrayBuffer()) : undefined;
      const second=b ? await context.decodeAudioData(await b.arrayBuffer()) : undefined;
      if(current!==revision)return;
      left=first;right=second;
      if(first)leftEnvelope=envelope(first,Math.max(0,first.duration-2),Math.min(2,first.duration));
      if(second)rightEnvelope=envelope(second,0,Math.min(2,second.duration));
      if(first && second && first.numberOfChannels!==second.numberOfChannels)throw new Error('The selected recordings have different channel counts. Choose compatible takes before joining them.');
    }catch(e){if(current===revision)error=`Could not read this boundary: ${String(e)}`;}finally{if(current===revision)loading=false;}
  }
  function adjust(delta: number) {
    if(!Number.isFinite(increment) || increment<=0){error='Choose a positive pause increment.';return;}
    error='';chunk.pauseAfter=Math.max(0,Math.round((gap+delta)*1000000)/1000000);onchange();
  }
  function stopPlayback() {
    cancelAnimationFrame(animation);
    for(const source of sources){source.onended=null;source.stop();source.disconnect();}
    sources=[];playing=false;progress=0;
  }
  async function listen() {
    if(!left || (next && !right))return;
    if(!Number.isFinite(gap) || gap<0){error='Gap must be zero or positive.';return;}
    stopPlayback();error='';
    try {
      document.querySelectorAll<HTMLAudioElement>('.tts-app audio').forEach(a=>a.pause());
      await context.resume();
      if(!mounted || !active || disabled)return;
      const at=context.currentTime+.03;
      const inserted=Math.round(gap*project.assembly.sampleRate)/project.assembly.sampleRate;
      const first=context.createBufferSource();first.buffer=left;first.connect(context.destination);first.start(at,Math.max(0,left.duration-2),leftSeconds);sources=[first];
      if(next && right){const second=context.createBufferSource();second.buffer=right;second.connect(context.destination);second.start(at+leftSeconds+inserted,0,rightSeconds);sources.push(second);}
      playbackGap=gap;startedAt=at;playing=true;
      sources[sources.length-1].onended=stopPlayback;
      const advance=()=>{progress=Math.max(0,Math.min(total,context.currentTime-startedAt));if(playing)animation=requestAnimationFrame(advance);};advance();
    }catch(e){stopPlayback();error=`Could not play this boundary: ${String(e)}`;}
  }
  onMount(()=>{context=new AudioContext({sampleRate:project.assembly.sampleRate});mounted=true;document.addEventListener('play',stopPlayback,true);});
  onDestroy(()=>{revision++;if(mounted){mounted=false;stopPlayback();document.removeEventListener('play',stopPlayback,true);void context.close();}});
</script>
<section class="boundary-editor" aria-label="Selected audio boundary">
  <header><div><span class="eyebrow">{next ? 'LISTEN TO THE JOIN' : 'END OF NARRATION'}</span><h4>{next ? `Chunk ${index+1} → ${index+2}` : `Chunk ${index+1} · final chunk`}</h4></div><button class="listen" disabled={disabled || loading || !left || (!!next && !right) || (!!left && !!right && left.numberOfChannels!==right.numberOfChannels)} on:click={()=>playing?stopPlayback():listen()}>{#if playing}<Square size={15}/>Stop preview{:else}<Play size={15}/>{next?'Listen across boundary':'Listen to ending'}{/if}</button></header>
  <div class="passages"><div><strong>Chunk {index+1} · {take?.reference?.name || take?.configuration.request.voice || take?.configuration.model.display_name || 'No recording'}</strong><p>{chunk.text}</p>{#if !chunkReady(project,chunk)}<small>Choose up-to-date audio in Voices & audio.</small>{/if}</div>{#if next}<div><strong>Chunk {index+2} · {nextTake?.reference?.name || nextTake?.configuration.request.voice || nextTake?.configuration.model.display_name || 'No recording'}</strong><p>{next.text}</p>{#if !chunkReady(project,next)}<small>Choose up-to-date audio in Voices & audio.</small>{/if}</div>{/if}</div>
  {#if loading}<p role="status">Reading the selected recordings…</p>{:else if left && (!next || right)}
    <div class="waveform" data-gap={gap}>
      <svg viewBox="0 0 1000 160" preserveAspectRatio="none" role="img" aria-label={`Actual audio waveform: last ${leftSeconds.toFixed(2)} seconds of chunk ${index+1}${next?`, ${gap} seconds inserted silence, first ${rightSeconds.toFixed(2)} seconds of chunk ${index+2}`:''}`}>
        <rect x="0" y="0" width={leftWidth} height="160" class="audio-region"/>
        {#if next}<rect x={leftWidth} y="0" width={gapWidth} height="160" class="inserted-gap"/><rect x={leftWidth+gapWidth} y="0" width={1000-leftWidth-gapWidth} height="160" class="audio-region next"/>{/if}
        <line x1="0" x2="1000" y1="80" y2="80" class="baseline"/>
        {#each leftEnvelope as pair,i}<line class="wave" x1={(i+.5)/leftEnvelope.length*leftWidth} x2={(i+.5)/leftEnvelope.length*leftWidth} y1={80-pair[1]*68} y2={80-pair[0]*68}/>{/each}
        {#each rightEnvelope as pair,i}<line class="wave next" x1={leftWidth+gapWidth+(i+.5)/rightEnvelope.length*(1000-leftWidth-gapWidth)} x2={leftWidth+gapWidth+(i+.5)/rightEnvelope.length*(1000-leftWidth-gapWidth)} y1={80-pair[1]*68} y2={80-pair[0]*68}/>{/each}
        <line x1={leftWidth} x2={leftWidth} y1="0" y2="160" class="edge"/>
        {#if next}<line x1={leftWidth+gapWidth} x2={leftWidth+gapWidth} y1="0" y2="160" class="edge"/>{/if}
        {#if playing}<line x1={progress/total*1000} x2={progress/total*1000} y1="0" y2="160" class="playhead"/>{/if}
      </svg>
      {#if next}<button class="end-handle" style:left={`clamp(70px, ${leftWidth/10}%, calc(100% - 70px))`} disabled={disabled} aria-label={`Add ${increment} seconds at end of chunk ${index+1}`} on:click={()=>adjust(increment)}><Plus size={15}/>Add {increment.toFixed(2)} s</button>{/if}
    </div>
    <div class="wave-labels"><span>Ending · {leftSeconds.toFixed(2)} s</span>{#if next}<span class="gap-label">Inserted gap · {gap.toFixed(2)} s</span><span>Beginning · {rightSeconds.toFixed(2)} s</span>{/if}</div>
    <p class="wave-hint">Waveforms include silence already in the recordings. {next?'The shaded gap is extra silence you insert.':'The final chunk has no trailing gap.'}</p>
  {:else}<div class="missing-audio">Generate and choose audio for {next?'both chunks':'this chunk'} in Voices & audio to see the waveforms and hear the join.</div>{/if}
  {#if next}
    <div class="gap-controls"><div><span>{chunk.pauseAfter===null?'Default gap':'Custom gap'}</span><details><summary aria-label="Edit exact gap">{gap.toFixed(2)} s <small>Edit</small></summary><label>Exact gap (seconds)<input aria-label={`Pause after chunk ${index+1}`} type="number" min="0" step="0.01" value={gap} disabled={disabled} on:change={e=>{const value=e.currentTarget.valueAsNumber;if(!Number.isFinite(value)||value<0){error='Enter a zero or positive gap.';return;}error='';chunk.pauseAfter=value;onchange();}}/></label></details></div><button disabled={disabled || gap===0} on:click={()=>adjust(-increment)}><Minus size={14}/>{increment.toFixed(2)} s</button><button class="add-pause" disabled={disabled} on:click={()=>adjust(increment)}><Plus size={15}/>Add {increment.toFixed(2)} s</button><button disabled={disabled || gap===0} on:click={()=>{chunk.pauseAfter=0;onchange();}}>No gap</button><button disabled={disabled || chunk.pauseAfter===null} on:click={()=>{chunk.pauseAfter=null;onchange();}}><RotateCcw size={14}/>Reset to default</button></div>
    <p class="click-hint">Click the + handle at the end of the waveform to add {increment.toFixed(2)} s. Click again to add more.</p>
  {/if}
  {#if error}<p role="alert">{error}</p>{/if}
</section>
<style>
  .boundary-editor{min-width:0;border:1px solid var(--line);border-radius:10px;padding:20px;background:var(--panel);}header{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;}h4{font-size:20px;margin:6px 0 0;}.eyebrow{font-size:10px;letter-spacing:1.5px;color:var(--cyan);}button{display:inline-flex;align-items:center;justify-content:center;gap:6px;min-height:38px;font-size:13px;}.listen,.add-pause{background:var(--cyan);color:var(--bg);border-color:var(--cyan);}.passages{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 0;}.passages>div{min-width:0;padding:12px;background:var(--active-bg);border-radius:6px;}.passages strong{font-size:12px;}.passages p{font-size:14px;line-height:1.7;max-height:100px;overflow:auto;margin:8px 0 0;}.passages small{display:block;color:var(--muted);font-size:12px;margin-top:8px;}.waveform{position:relative;padding-bottom:36px;}svg{display:block;width:100%;height:130px;border:1px solid var(--line);border-radius:6px;}.audio-region{fill:var(--active-bg);}.inserted-gap{fill:var(--cyan);fill-opacity:.18;}.baseline{stroke:var(--muted);stroke-opacity:.3;}.wave{stroke:var(--cyan);stroke-width:1.4;}.wave.next{stroke:var(--text);}.edge{stroke:var(--muted);stroke-dasharray:4 4;}.playhead{stroke:var(--text);stroke-width:3;}.end-handle{position:absolute;bottom:0;transform:translateX(-50%);background:var(--cyan);color:var(--bg);border:1px solid var(--cyan);white-space:nowrap;padding:6px 10px;}.wave-labels{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:10px 0;}.gap-label{color:var(--cyan);}.wave-hint,.click-hint{font-size:12px;line-height:1.6;color:var(--muted);}.gap-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding-top:18px;border-top:1px solid var(--line);margin-top:18px;}.gap-controls>div{margin-right:auto;display:grid;gap:6px;}.gap-controls>div>span{font-size:12px;color:var(--muted);}summary{list-style:none;cursor:pointer;color:var(--cyan);font-size:20px;font-weight:600;}summary::-webkit-details-marker{display:none;}summary small{font-size:12px;font-weight:400;margin-left:8px;}details label{display:grid;gap:8px;font-size:12px;margin:12px 0;}input{max-width:150px;}[role=alert]{color:var(--danger);font-size:13px;}.missing-audio{padding:24px 16px;border:1px dashed var(--line);color:var(--muted);font-size:13px;line-height:1.7;}.click-hint{margin-bottom:0;}button:focus-visible,summary:focus-visible{outline:2px solid var(--cyan);outline-offset:3px;}@media(max-width:760px){.boundary-editor{padding:16px;}.passages{grid-template-columns:1fr;}.gap-controls>div{flex-basis:100%;}svg{height:110px;}}
</style>
