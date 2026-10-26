<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import { Play, Plus, ZoomIn, ZoomOut, Undo2, Redo2 } from '@lucide/svelte';
  import { encodePcm16Wav } from '$lib/audio';
  import { renderTakeAudio, waveformEnvelope, takeAudioSegments } from './audio';
  import type { Take, TakeAudioEdits, AudioEditSnapshot } from './project';
  export let take: Take;
  export let sampleRate: number;
  export let increment: number;
  export let disabled = false;
  export let active = true;
  export let onchange: (replacedAudio?: Blob) => void;
  export let onincrement: (seconds: number) => void;
  let context: AudioContext;
  let original: AudioBuffer;
  let buffer: AudioBuffer;
  let loading=true;
  let destroyed=false;
  let error='';
  let cursor=0;
  let selectedSilence='';
  let selectionStart=0;
  let selectionEnd=0;
  let pointerStart=0;
  let pointerX=0;
  let pointerRegion='';
  let dragging=false;
  let zoom=1;
  let viewStart=0;
  let width=700;
  let viewport: HTMLDivElement;
  let player: HTMLAudioElement;
  let audioUrl='';
  let lastAudio: Blob | undefined;
  let observer: ResizeObserver;
  let animation=0;
  $: duration=buffer?.duration || 0;
  $: windowSeconds=duration/zoom;
  $: peaks=buffer ? waveformEnvelope(buffer,viewStart,Math.min(windowSeconds,duration-viewStart),Math.min(1000,Math.round(width))) : [];
  $: silences=take.editing?.silences || [];
  $: removed=take.editing?.removed || [];
  $: segments=original ? takeAudioSegments(original.length,original.sampleRate,silences,removed) : [];
  $: regions=segments.filter(s=>s.silenceId).map(s=>({id:s.silenceId!,start:s.start/original.sampleRate,seconds:s.frames/original.sampleRate}));
  $: rangeStart=Math.min(selectionStart,selectionEnd);
  $: rangeEnd=Math.max(selectionStart,selectionEnd);
  $: currentAudio=take.rendered?.audio || take.audio;
  $: if(currentAudio!==lastAudio){if(audioUrl)URL.revokeObjectURL(audioUrl);lastAudio=currentAudio;audioUrl=URL.createObjectURL(currentAudio);}
  $: if(!active || disabled)player?.pause();
  function seek(seconds: number){if(!Number.isFinite(seconds)){error='Enter a valid playhead time.';return;}if(buffer)error='';cursor=Math.max(0,Math.min(duration,seconds));if(player)player.currentTime=cursor;}
  function scrollView(){viewStart=duration && viewport ? viewport.scrollLeft/(width*zoom)*duration : 0;}
  async function changeZoom(value: number){
    zoom=Math.max(1,Math.min(64,value));await tick();
    viewport.scrollLeft=Math.max(0,cursor/duration*width*zoom-width/2);scrollView();
  }
  function selectRegion(id: string){const region=regions.find(e=>e.id===id)!;selectedSilence=id;seek(region.start+region.seconds/2);void tick().then(()=>{viewport.scrollLeft=Math.max(0,cursor/duration*width*zoom-width/2);scrollView();});}
  function commit(next: TakeAudioEdits,undo: AudioEditSnapshot[],redo: AudioEditSnapshot[]){
    player.pause();
    try{
      const previous=take.rendered?.audio;
      const ordered=[...next.silences].sort((a,b)=>a.at-b.at),cuts=next.removed || [];
      const edited=ordered.length>0 || cuts.length>0;
      const rendered=edited ? renderTakeAudio(context,original,ordered,cuts) : original;
      const audio=edited ? encodePcm16Wav(rendered) : undefined;
      take.editing={silences:ordered,removed:cuts,undo,redo};take.rendered=audio ? {audio,duration:rendered.duration} : undefined;
      buffer=rendered;cursor=Math.min(cursor,buffer.duration);selectedSilence='';selectionStart=selectionEnd=0;error='';onchange(previous);
      void tick().then(()=>{if(!destroyed){scrollView();if(player)player.currentTime=cursor;}});
    }catch(e){error=`Could not edit this recording: ${String(e)}`;}
  }
  function insert(){
    if(!Number.isFinite(increment) || increment<=0){error='Choose a positive silence increment.';return;}
    const next=structuredClone(silences),frame=Math.round(cursor*original.sampleRate);
    const segment=segments.find(s=>frame>=s.start && frame<s.start+s.frames) || segments[segments.length-1];
    if(segment.sourceFrame===null){next.find(e=>e.id===segment.silenceId)!.seconds+=increment;}
    else {const at=(segment.sourceFrame+Math.min(segment.frames,Math.max(0,frame-segment.start)))/original.sampleRate;next.push({id:crypto.randomUUID(),at,seconds:increment});}
    commit({silences:next,removed},[...(take.editing?.undo || []),structuredClone({silences,removed})],[]);
  }
  function removeSelected(){cursor=regions.find(e=>e.id===selectedSilence)!.start;commit({silences:silences.filter(e=>e.id!==selectedSilence),removed},[...(take.editing?.undo || []),structuredClone({silences,removed})],[]);}
  function deleteRange(){
    if(!Number.isFinite(rangeStart)||!Number.isFinite(rangeEnd)||rangeStart<0||rangeEnd>duration||rangeEnd<=rangeStart){error='Select a valid audio range.';return;}
    const rate=original.sampleRate,start=Math.round(rangeStart*rate),end=Math.round(rangeEnd*rate);
    const next=structuredClone(silences),cuts=structuredClone(removed);
    for(const segment of segments){
      const from=Math.max(start,segment.start),to=Math.min(end,segment.start+segment.frames);if(to<=from)continue;
      if(segment.sourceFrame===null){const edit=next.find(e=>e.id===segment.silenceId)!;edit.seconds=Math.max(0,Math.round(edit.seconds*rate)-(to-from))/rate;}
      else cuts.push({start:(segment.sourceFrame+from-segment.start)/rate,end:(segment.sourceFrame+to-segment.start)/rate});
    }
    cursor=rangeStart;
    commit({silences:next.filter(e=>e.seconds>0),removed:cuts},[...(take.editing?.undo || []),structuredClone({silences,removed})],[]);
  }
  function undo(){const stack=take.editing!.undo;const snapshot=stack[stack.length-1];commit(structuredClone(Array.isArray(snapshot)?{silences:snapshot}:snapshot),stack.slice(0,-1),[...(take.editing?.redo || []),structuredClone({silences,removed})]);}
  function redo(){const stack=take.editing!.redo;const snapshot=stack[stack.length-1];commit(structuredClone(Array.isArray(snapshot)?{silences:snapshot}:snapshot),[...(take.editing?.undo || []),structuredClone({silences,removed})],stack.slice(0,-1));}
  function point(event: PointerEvent){return Math.max(0,Math.min(duration,viewStart+(event.clientX-(event.currentTarget as SVGElement).getBoundingClientRect().left)/width*windowSeconds));}
  function pointerDown(event: PointerEvent){if(disabled||loading||!buffer)return;player.pause();pointerStart=point(event);pointerX=event.clientX;pointerRegion=(event.target as SVGElement).getAttribute('data-silence') || '';dragging=true;(event.currentTarget as SVGElement).setPointerCapture(event.pointerId);}
  function pointerMove(event: PointerEvent){if(dragging && Math.abs(event.clientX-pointerX)>3){selectedSilence='';selectionStart=pointerStart;selectionEnd=point(event);}}
  function pointerUp(event: PointerEvent){if(!dragging)return;dragging=false;if(Math.abs(event.clientX-pointerX)>3){selectionStart=pointerStart;selectionEnd=point(event);seek(Math.min(selectionStart,selectionEnd));}else if(event.shiftKey){selectionStart=cursor;selectionEnd=point(event);}else{selectionStart=selectionEnd=0;if(pointerRegion)selectRegion(pointerRegion);else{selectedSilence='';seek(point(event));}}}
  function playback(){
    document.querySelectorAll<HTMLAudioElement>('.tts-app audio').forEach(a=>{if(a!==player)a.pause();});
    const update=()=>{cursor=player.currentTime;animation=requestAnimationFrame(update);};cancelAnimationFrame(animation);update();
  }
  onMount(()=>{
    context=new AudioContext({sampleRate});
    observer=new ResizeObserver(entries=>{width=entries[0].contentRect.width;scrollView();});observer.observe(viewport);
    void (async()=>{try{original=await context.decodeAudioData(await take.audio.arrayBuffer());if(destroyed)return;buffer=silences.length || removed.length?renderTakeAudio(context,original,silences,removed):original;}catch(e){if(!destroyed)error=`Could not read this recording: ${String(e)}`;}finally{if(!destroyed)loading=false;}})();
  });
  onDestroy(()=>{destroyed=true;cancelAnimationFrame(animation);player?.pause();observer?.disconnect();if(audioUrl)URL.revokeObjectURL(audioUrl);if(context)void context.close();});
</script>
<section class="wave-editor" aria-label="Chunk waveform editor">
  <header><div><h4>Edit this take’s audio</h4><p>Click to place the playhead and add silence. Drag across a gap to select and delete it. Your original recording stays unchanged.</p></div><span>{duration.toFixed(2)} s {silences.length || removed.length?'· Edited':'· Original'}</span></header>
  <div class="edit-toolbar">
    <button aria-label="Zoom out" disabled={loading || !buffer || zoom===1} on:click={()=>changeZoom(zoom/2)}><ZoomOut size={16}/></button><span class="zoom">{zoom}×</span><button aria-label="Zoom in" disabled={loading || !buffer || zoom===64} on:click={()=>changeZoom(zoom*2)}><ZoomIn size={16}/></button><button disabled={loading || !buffer} on:click={()=>changeZoom(1)}>Fit recording</button>
    <button disabled={disabled || loading || !take.editing?.undo.length} on:click={undo}><Undo2 size={15}/>Undo</button><button disabled={disabled || loading || !take.editing?.redo.length} on:click={redo}><Redo2 size={15}/>Redo</button>
  </div>
  <div class="viewport" bind:this={viewport} on:scroll={scrollView}>
    <div class="wave-content" style:width={`${zoom*100}%`}>
      <svg width={width} viewBox="0 0 1000 150" preserveAspectRatio="none" role="slider" tabindex="0" aria-label="Audio waveform playhead" aria-valuemin="0" aria-valuemax={duration} aria-valuenow={cursor} aria-valuetext={`${cursor.toFixed(3)} seconds`} on:pointerdown={pointerDown} on:pointermove={pointerMove} on:pointerup={pointerUp} on:pointercancel={()=>dragging=false} on:keydown={e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();if(e.shiftKey){if(rangeEnd===rangeStart)selectionStart=cursor;selectionEnd=Math.max(0,Math.min(duration,cursor+(e.key==='ArrowRight'?.1:-.1)));}else selectionStart=selectionEnd=0;seek(cursor+(e.key==='ArrowRight'?1:-1)*(e.shiftKey?.1:.01));}if((e.key==='Delete'||e.key==='Backspace') && rangeEnd>rangeStart && !disabled){e.preventDefault();deleteRange();}if(e.key==='Home'){e.preventDefault();seek(0);}if(e.key==='End'){e.preventDefault();seek(duration);}}}>
        {#if buffer}<rect width="1000" height="150" class="wave-bg"/>
        {#each regions as region}<rect class="silence-region" class:chosen={region.id===selectedSilence} x={(region.start-viewStart)/windowSeconds*1000} y="0" width={region.seconds/windowSeconds*1000} height="150" data-silence={region.id} role="button" tabindex="0" aria-label={`Select inserted silence at ${region.start.toFixed(2)} seconds`} on:click|stopPropagation={()=>selectRegion(region.id)} on:keydown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();selectRegion(region.id);}}}/>{/each}
        <rect class="range-selection" x={(rangeStart-viewStart)/windowSeconds*1000} width={(rangeEnd-rangeStart)/windowSeconds*1000} y="0" height="150"/>
        <line class="baseline" x1="0" x2="1000" y1="75" y2="75"/>
        {#each peaks as pair,i}<line class="wave" x1={(i+.5)/peaks.length*1000} x2={(i+.5)/peaks.length*1000} y1={75-pair[1]*65} y2={75-pair[0]*65}/>{/each}
        <line class="playhead" x1={(cursor-viewStart)/windowSeconds*1000} x2={(cursor-viewStart)/windowSeconds*1000} y1="0" y2="150"/>{/if}
      </svg>
    </div>
  </div>
  <div class="ruler"><span>{viewStart.toFixed(2)} s</span><span>Visible {windowSeconds.toFixed(2)} s · scroll horizontally to pan</span><span>{Math.min(duration,viewStart+windowSeconds).toFixed(2)} s</span></div>
  {#if loading}<p role="status">Reading this take…</p>{/if}
  <div class="insert-toolbar"><label>Playhead (seconds)<input aria-label="Audio playhead seconds" type="number" min="0" max={duration} step="0.01" value={cursor.toFixed(3)} disabled={loading} on:change={e=>seek(e.currentTarget.valueAsNumber)}/></label><label>Add per click<select aria-label="Insert silence increment" value={increment} disabled={disabled} on:change={e=>onincrement(Number(e.currentTarget.value))}>{#each [.05,.1,.25,.5,1] as seconds}<option value={seconds}>+{seconds.toFixed(2)} s</option>{/each}</select></label><button class="insert" disabled={disabled || loading || !buffer} on:click={insert}><Plus size={15}/>Insert {increment.toFixed(2)} s at playhead</button><button disabled={disabled || loading || !selectedSilence} on:click={removeSelected}>Remove selected silence</button></div>
  <div class="range-toolbar"><label>Selection start (s)<input aria-label="Selection start seconds" type="number" min="0" max={duration} step="0.01" bind:value={selectionStart} disabled={loading || disabled}/></label><label>Selection end (s)<input aria-label="Selection end seconds" type="number" min="0" max={duration} step="0.01" bind:value={selectionEnd} disabled={loading || disabled}/></label><button disabled={disabled || loading || !buffer || rangeEnd<=rangeStart} on:click={deleteRange}>Delete selected gap / range</button><button disabled={rangeEnd<=rangeStart} on:click={()=>selectionStart=selectionEnd=0}>Clear selection</button></div>
  <p class="selection-hint">Drag to select, or Shift-click to extend a selection. Delete removes all audio in that range, including any speech. Undo restores it.</p>
  <audio controls bind:this={player} src={audioUrl} aria-label="Edited take playback" on:play={playback} on:pause={()=>cancelAnimationFrame(animation)} on:ended={()=>cancelAnimationFrame(animation)}></audio>
  <button class="play-cursor" disabled={loading || disabled || !buffer} on:click={()=>{player.currentTime=cursor;void player.play().catch(e=>error=String(e));}}><Play size={14}/>Play from playhead</button>
  {#if regions.length}<div class="silence-list" aria-label="Inserted silences">{#each regions as region}<button class:selected={region.id===selectedSilence} on:click={()=>selectRegion(region.id)}>Silence {region.start.toFixed(2)}–{(region.start+region.seconds).toFixed(2)} s</button>{/each}</div>{/if}
  {#if error}<p role="alert">{error}</p>{/if}
</section>
<style>
  .wave-editor{padding:20px;border-top:1px solid var(--line);min-width:0;}header{display:flex;justify-content:space-between;gap:16px;align-items:start;}h4{margin:0;font-size:17px;}header p{font-size:13px;color:var(--muted);line-height:1.6;margin:8px 0 16px;}header>span{font-size:13px;white-space:nowrap;color:var(--cyan);}button{display:inline-flex;align-items:center;justify-content:center;gap:6px;min-height:36px;font-size:12px;}.edit-toolbar,.insert-toolbar,.range-toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:12px 0;}.zoom{font-size:12px;}.viewport{overflow-x:auto;border:1px solid var(--line);border-radius:6px;}.wave-content{position:relative;height:150px;}svg{position:sticky;left:0;display:block;height:150px;max-width:none;cursor:crosshair;overflow:hidden;}.range-selection{fill:var(--cyan);fill-opacity:.35;stroke:var(--cyan);pointer-events:none;}.selection-hint{font-size:12px;color:var(--muted);line-height:1.6;}svg{touch-action:none;}.wave-bg{fill:var(--active-bg);}.wave{stroke:var(--cyan);stroke-width:1.2;pointer-events:none;}.baseline{stroke:var(--muted);stroke-opacity:.3;pointer-events:none;}.playhead{stroke:var(--text);stroke-width:2;pointer-events:none;}.silence-region{fill:var(--cyan);fill-opacity:.18;cursor:pointer;}.silence-region.chosen{fill-opacity:.4;stroke:var(--cyan);stroke-width:2;}.ruler{display:flex;justify-content:space-between;gap:8px;font-size:11px;color:var(--muted);margin:8px 0;}.insert-toolbar,.range-toolbar{align-items:end;}label{display:grid;gap:7px;font-size:12px;color:var(--muted);}input{width:130px;}select{min-width:110px;}.insert{background:var(--cyan);color:var(--bg);border-color:var(--cyan);}audio{display:block;width:100%;height:36px;margin-top:16px;}.play-cursor{margin-top:10px;}.silence-list{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}.silence-list .selected{border-color:var(--cyan);}button:focus-visible,svg:focus-visible{outline:2px solid var(--cyan);outline-offset:2px;}[role=alert]{font-size:13px;color:var(--danger);}@media(max-width:760px){header{flex-wrap:wrap;}.wave-editor{padding:16px;}.ruler{flex-wrap:wrap;}.ruler>span:nth-child(2){display:none;}}
</style>
