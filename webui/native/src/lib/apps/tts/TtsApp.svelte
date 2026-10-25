<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import { FileText, AudioLines, Layers, SlidersHorizontal, Plus, Play, RotateCcw, Scissors, ChevronRight, ChevronDown, Download, Check, Square, ArrowUp, ArrowDown, Merge, Trash2 } from '@lucide/svelte';
  import { listAppProjects, getAppProject, saveAppProject, deleteAppProject, listVoices, type SavedVoice } from '$lib/library';
  import { loadModelChoices } from '$lib/presets/choices';
  import type { SavedModelPreset } from '$lib/presets/types';
  import { PresetRuntime } from '$lib/presets/runtime';
  import { uploadWav } from '$lib/api';
  import { splitTtsChunks } from '$lib/text';
  import { modelCapabilities } from '$lib/catalog';
  import { assembleAudio, finishAudio, exportAudio } from './audio';
  import { newProject, newChunk, selectedTake, chunkConfiguration, chunkReference, speechSignature, chunkReady, assemblySignature, assemblyCurrent, finishSignature, type TtsProject, type Chunk, type Take, type VoiceAssignment } from './project';
  import ModelChoice from './ModelChoice.svelte';
  import VoiceChoice from './VoiceChoice.svelte';
  import BoundaryEditor from './BoundaryEditor.svelte';
  import WaveformEditor from './WaveformEditor.svelte';
  let speechBoundaryOpen=false;
  export let initialPresetId = '';
  export let busy = false;
  export let active = true;
  export let openLibrary: () => void;
  let project: TtsProject | null = null;
  let projects: Array<{id:string;title:string;updatedAt:number}> = [];
  let choices: SavedModelPreset[] = [];
  let voices: SavedVoice[] = [];
  let stage: 'script'|'speech'|'assemble'|'finish' = 'script';
  let selectedId = '';
  let navigatorPage = 0;
  const navigatorPageSize = 40;
  let jumpNumber = 1;
  let assemblyPage = 0;
  let assemblySearch = '';
  let assemblyOnlyAttention = false;
  let archiveOpen = false;
  let archivePage = 0;
  let defaultsOpen = true;
  $: archivePages = Math.max(1,Math.ceil((project?.archivedChunks.length || 0)/navigatorPageSize));
  $: if(archivePage>=archivePages)archivePage=archivePages-1;
  let checked = new Set<string>();
  let filter = 'all';
  let search = '';
  let preview: string[] | null = null;
  let takingId = '';
  let workingId = '';
  let status = '';
  let error = '';
  let loading = true;
  let wasActive = active;
  $: if(active !== wasActive) { wasActive=active; if(active && !loading)refreshChoices(); }
  let dirty = false;
  let saving = false;
  let saveTimer: ReturnType<typeof setTimeout> | undefined;
  let savePromise: Promise<void> | null = null;
  let aborter: AbortController | null = null;
  let exportLink = '';
  let exportName = '';
  let exportedSignature = '';
  let editors: Record<string, HTMLTextAreaElement> = {};
  let urls = new Map<Blob,string>();
  let player: HTMLAudioElement | undefined;
  let lastInitialPreset = '';
  let suggested: SavedModelPreset | undefined;
  $: if (initialPresetId && initialPresetId !== lastInitialPreset && choices.length) {
    lastInitialPreset = initialPresetId;
    suggested = choices.find(c => c.id === initialPresetId && ['tts','clon','vdes'].includes(c.model.task));
  }
  $: if (!active) {
    player?.pause();
    if (typeof document !== 'undefined') document.querySelectorAll<HTMLAudioElement>('.tts-app audio').forEach(a => a.pause());
  }
  $: ready = project ? project.chunks.filter(c => chunkReady(project!, c)).length : 0;
  $: needsGeneration = project?.chunks.filter(c=>!c.takes.some(t=>t.signature===speechSignature(project!,c))) || [];
  $: needsSelection = project?.chunks.filter(c=>!chunkReady(project!,c) && c.takes.some(t=>t.signature===speechSignature(project!,c))) || [];
  $: visible = project?.chunks.filter(c => c.text.toLowerCase().includes(search.toLowerCase()) &&
    (filter === 'all' || filter === 'missing' && !selectedTake(c) || filter === 'changed' && !!selectedTake(c) && !chunkReady(project!, c))) || [];
  $: selectedChunk = project?.chunks.find(c=>c.id===selectedId);
  $: selectedIndex = project?.chunks.findIndex(c=>c.id===selectedId) ?? -1;
  $: if(selectedIndex>=0)jumpNumber=selectedIndex+1;
  $: navigatorPages = Math.max(1,Math.ceil(visible.length/navigatorPageSize));
  $: if(navigatorPage>=navigatorPages)navigatorPage=navigatorPages-1;
  $: navigatorRows = visible.slice(navigatorPage*navigatorPageSize,(navigatorPage+1)*navigatorPageSize);
  $: assemblyMatches = project?.chunks.filter(c=>c.text.toLowerCase().includes(assemblySearch.toLowerCase()) && (!assemblyOnlyAttention || !chunkReady(project!,c))) || [];
  $: assemblyPages = Math.max(1,Math.ceil(assemblyMatches.length/navigatorPageSize));
  $: if(assemblyPage>=assemblyPages)assemblyPage=assemblyPages-1;
  $: assemblyRows = assemblyMatches.slice(assemblyPage*navigatorPageSize,(assemblyPage+1)*navigatorPageSize);
  function selectChunk(id: string) {
    if(!project)return;
    document.querySelectorAll<HTMLAudioElement>('.tts-app audio').forEach(a=>a.pause());
    selectedId=id;takingId='';
    localStorage.setItem(`audiocpp.tts.chunk:${project.id}`,id);
    const at=(search || filter!=='all' ? visible : project.chunks).findIndex(c=>c.id===id);
    if(at>=0)navigatorPage=Math.floor(at/navigatorPageSize);
    void tick().then(()=>document.querySelector('.editor-navigation')?.scrollIntoView({block:'start'}));
  }
  function navigateAssembly(index: number) {
    if(!project)return;
    assemblySearch='';assemblyOnlyAttention=false;
    assemblyPage=Math.floor(index/navigatorPageSize);
    selectChunk(project.chunks[index].id);
    void tick().then(()=>{const row=document.getElementById(`assembly-${project!.chunks[index].id}`);const list=row?.parentElement;if(row && list)list.scrollTo({top:row.offsetTop-(list.clientHeight-row.offsetHeight)/2});});
  }
  function jumpToChunk() {
    if(!project)return;
    if(!Number.isInteger(jumpNumber) || jumpNumber<1 || jumpNumber>project.chunks.length) { error=`Enter a chunk number between 1 and ${project.chunks.length}.`;return; }
    if(stage==='assemble') { navigateAssembly(jumpNumber-1);error='';return; }
    search='';filter='all';navigatorPage=Math.floor((jumpNumber-1)/navigatorPageSize);
    selectChunk(project.chunks[jumpNumber-1].id);error='';
  }
  function nextAttention() {
    if(!project)return;
    const ordered=[...project.chunks.slice(selectedIndex+1),...project.chunks.slice(0,selectedIndex+1)];
    const next=ordered.find(c=>!chunkReady(project!,c));
    if(next && stage==='assemble') { navigateAssembly(project.chunks.indexOf(next));return; }
    if(next) { search='';filter='all';selectChunk(next.id);navigatorPage=Math.floor(project.chunks.indexOf(next)/navigatorPageSize); }
  }
  $: finalCurrent = !!project?.finished && assemblyCurrent(project) && project.finished.signature === finishSignature(project);
  $: assemblyIsCurrent = project ? assemblyCurrent(project) : false;
  $: enhancementFamilies = Object.keys(modelCapabilities).filter(family=>modelCapabilities[family].includes('audio_enhancement'));
  function urlFor(audio: Blob) {
    let url = urls.get(audio);
    if (!url) { url = URL.createObjectURL(audio); urls.set(audio,url); }
    return url;
  }
  function release(audio: Blob | undefined) {
    if (!audio) return;
    const url=urls.get(audio); if(url) { URL.revokeObjectURL(url);urls.delete(audio); }
  }
  function clearUrls() { for(const url of urls.values()) URL.revokeObjectURL(url);urls.clear(); }
  function touch() {
    if (!project) return;
    project = { ...project, updatedAt: Math.max(Date.now(), project.updatedAt + 1) }; dirty = true;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => { flush().catch(e => error = `Could not save project: ${String(e)}`); }, 450);
  }
  export async function saveBeforeClose() { await flush(); }
  async function flush() {
    clearTimeout(saveTimer);
    if (savePromise) await savePromise;
    if (!project || !dirty) return;
    const snapshot = structuredClone(project);
    saving = true;
    savePromise = saveAppProject({id:snapshot.id,appId:'tts',title:snapshot.title,updatedAt:snapshot.updatedAt,data:snapshot}).then(() => {
      if(project?.id === snapshot.id && project.updatedAt === snapshot.updatedAt) dirty=false;
      projects = [{id:snapshot.id,title:snapshot.title,updatedAt:snapshot.updatedAt},...projects.filter(p=>p.id!==snapshot.id)].sort((a,b)=>b.updatedAt-a.updatedAt);
    }).finally(() => { saving=false;savePromise=null; });
    await savePromise;
    if(dirty) await flush();
  }
  async function selectProject(id: string) {
    if(busy) return;
    try {
      await flush();
      const record = await getAppProject(id);
      if(!record || (record.data as TtsProject).version !== 1) throw new Error('Unsupported or missing TTS project.');
      clearUrls();project=record.data as TtsProject;localStorage.setItem('audiocpp.tts.project',project.id);stage=project.chunks.length?'speech':'script';selectedId=project.chunks.find(c=>c.id===localStorage.getItem(`audiocpp.tts.chunk:${project!.id}`))?.id || project.chunks[0]?.id || '';search='';filter='all';navigatorPage=Math.max(0,Math.floor(project.chunks.findIndex(c=>c.id===selectedId)/navigatorPageSize));assemblyPage=0;assemblySearch='';assemblyOnlyAttention=false;archivePage=0;defaultsOpen=!project.speech;
      preview=null;checked=new Set();error='';status='';exportLink='';
    }catch(e){error=String(e);}
  }
  async function createProject() {
    if(busy) return;
    try { await flush();clearUrls();project=newProject();localStorage.setItem('audiocpp.tts.project',project.id);stage='script';selectedId='';search='';filter='all';navigatorPage=0;assemblyPage=0;assemblySearch='';assemblyOnlyAttention=false;archivePage=0;defaultsOpen=!project.speech;checked=new Set();preview=null;error='';status='';exportLink='';touch(); }
    catch(e){error=String(e);}
  }
  async function removeProject() {
    if(!project || busy || !confirm(`Delete project "${project.title}" and its audio?`)) return;
    try { await flush();const id=project.id;await deleteAppProject(id);projects=projects.filter(p=>p.id!==id);clearUrls();project=null;dirty=false;
      if(projects.length) await selectProject(projects[0].id);else await createProject();
    }catch(e){error=String(e);}
  }
  async function refreshChoices() {
    try { [choices,voices]=await Promise.all([loadModelChoices(),listVoices()]); error=''; }
    catch(e){error=`Could not refresh models and presets: ${String(e)}`;}
  }
  function suggest() {
    if(!project || !suggested) return;
    project.speech=structuredClone(suggested);suggested=undefined;touch();stage='speech';
  }
  function previewSplit() {
    if(!project) return;
    error='';
    if(!project.source.trim()) { error='Paste text or import a TXT/Markdown file first.';return; }
    if(!Number.isInteger(project.split.characters) || project.split.characters<40) { error='Chunk size must be at least 40 characters.';return; }
    const text=project.source.trim();
    preview=project.split.method==='sentence'
      ? Array.from(new Intl.Segmenter(undefined,{granularity:'sentence'}).segment(text),s=>s.segment.trim()).filter(Boolean).flatMap(s=>splitTtsChunks(s,project!.split.characters))
      : project.split.method==='paragraph' ? text.split(/\n\s*\n/).filter(s=>s.trim()).flatMap(s=>splitTtsChunks(s,project!.split.characters))
      : splitTtsChunks(text,project.split.characters);
  }
  function applySplit() {
    if(!project || !preview?.length) return;
    if(project.chunks.length && !confirm('Replace the current chunks? Existing chunks and takes will remain in the project archive.')) return;
    project.archivedChunks.push(...project.chunks);project.chunks=preview.map(text=>newChunk(text));selectedId=project.chunks[0].id;
    preview=null;checked=new Set();search='';filter='all';navigatorPage=0;assemblyPage=0;assemblySearch='';assemblyOnlyAttention=false;archivePage=0;defaultsOpen=!project.speech;touch();stage='speech';
  }
  async function importText(file: File | undefined) {
    if(!file || !project) return;
    error='';
    if(!/\.(txt|md|markdown)$/i.test(file.name)) { error='Choose a TXT or Markdown file.';return; }
    try { project.source=await file.text();if(project.title==='Untitled narration')project.title=file.name.replace(/\.[^.]+$/,'');preview=null;touch(); }
    catch(e){error=String(e);}
  }
  function splitChunk(chunk: Chunk) {
    if(!project || busy) return;
    const at=editors[chunk.id]?.selectionStart;
    if(at===undefined || !chunk.text.slice(0,at).trim() || !chunk.text.slice(at).trim()) { error='Place the text cursor where you want to split this chunk.';return; }
    const index=project.chunks.indexOf(chunk);
    project.archivedChunks.push(structuredClone(chunk));
    const parts=[newChunk(chunk.text.slice(0,at).trim(),chunk.override),newChunk(chunk.text.slice(at).trim(),chunk.override)];
    for(const part of parts)part.voice=structuredClone(chunk.voice);
    project.chunks.splice(index,1,...parts);selectChunk(parts[0].id);error='';touch();
  }
  function mergeChunk(chunk: Chunk) {
    if(!project || busy) return;
    const index=project.chunks.indexOf(chunk);const next=project.chunks[index+1];if(!next)return;
    if(JSON.stringify([chunkConfiguration(project,chunk),chunkReference(project,chunk)?.id])!==JSON.stringify([chunkConfiguration(project,next),chunkReference(project,next)?.id]) && !confirm('These chunks use different narrators. Joining them uses the first chunk’s model and voice. Continue?'))return;
    project.archivedChunks.push(structuredClone(chunk),structuredClone(next));
    const merged=newChunk(`${chunk.text}\n${next.text}`,chunk.override);merged.voice=structuredClone(chunk.voice);project.chunks.splice(index,2,merged);selectChunk(merged.id);touch();
  }
  function moveChunk(chunk: Chunk, offset: number) {
    if(!project || busy)return;const i=project.chunks.indexOf(chunk);const j=i+offset;if(j<0 || j>=project.chunks.length)return;
    [project.chunks[i],project.chunks[j]]=[project.chunks[j],project.chunks[i]];touch();
  }
  function archiveChunk(chunk: Chunk) {
    if(!project || busy)return;project.archivedChunks.push(chunk);const index=project.chunks.indexOf(chunk);project.chunks=project.chunks.filter(c=>c.id!==chunk.id);checked.delete(chunk.id);if(selectedId===chunk.id)selectChunk(project.chunks[Math.min(index,project.chunks.length-1)]?.id || '');touch();
  }
  function useUpdatedTakes() {
    if(!project)return;
    for(const chunk of needsSelection)chunk.takeId=[...chunk.takes].reverse().find(t=>t.signature===speechSignature(project!,chunk))!.id;
    touch();
  }
  function useTake(chunk: Chunk,take: Take) { chunk.takeId=take.id;touch(); }
  function projectVoice(voice: VoiceAssignment | undefined) {
    if(!project || !voice)return;
    if(voice.kind==='reference') project.reference=voice.data;
    else {
      project.reference=null;
      if(project.speech) {
        const request={...project.speech.request};
        if(voice.id)request.voice=voice.id;else delete request.voice;
        project.speech={...project.speech,request};
      }
    }
    touch();
  }
  async function transcribeReference(chunk?: Chunk) {
    if(!project || busy)return;
    const reference=chunk ? chunkReference(project,chunk) : project.reference;
    if(!reference || !project.transcription)return;
    busy=true;error='';aborter=new AbortController();status='Recognizing the words in the voice recording…';
    try {
      const runtime=new PresetRuntime(project.transcription,aborter.signal);await runtime.prepare();
      const audio=await uploadWav(reference.audio,aborter.signal);const result=await runtime.run({audio});
      if(result.kind!=='json' || typeof result.data.text!=='string')throw new Error('The model did not return a transcript.');
      if(chunk)chunk.voice={kind:'reference',data:{...reference,transcript:result.data.text}};
      else project.reference={...reference,transcript:result.data.text};
      touch();status='Transcript filled in. Check that the words match the recording.';
    }catch(e){if(aborter.signal.aborted)status='Transcription cancelled.';else error=String(e);}finally{busy=false;aborter=null;}
  }
  async function generate(ids: string[]) {
    if(!project || busy || !ids.length)return;
    const jobs=project.chunks.filter(c=>ids.includes(c.id)).map(c=>({id:c.id,text:c.text,configuration:structuredClone(chunkConfiguration(project!,c)),reference:structuredClone(chunkReference(project!,c)),signature:speechSignature(project!,c),sequence:project!.chunks.indexOf(c)+c.takes.length}));
    if(jobs.some(job=>!job.configuration)) { error='Choose a default model above, or choose a model on each chunk you want to generate.';return; }
    if(jobs.some(job=>!job.text.trim())) { error='Remove empty chunks or enter text before generation.';return; }
    if(jobs.some(job=>job.reference && job.configuration!.model.task!=='clon' && !modelCapabilities[job.configuration!.model.family]?.includes('speaker_reference'))) { error='A selected speech model does not support a reference recording. Remove the reference or choose a voice-cloning model.';return; }
    busy=true;error='';aborter=new AbortController();const signal=aborter.signal;
    let runtime:PresetRuntime|undefined;let key='';const references=new Map<string,string>();
    try {
      for(let i=0;i<jobs.length;i++) {
        signal.throwIfAborted();const job=jobs[i];workingId=job.id;status=`Generating chunk ${i+1} of ${jobs.length}…`;
        const nextKey=JSON.stringify([job.configuration!.model,job.configuration!.request]);
        if(!runtime || key!==nextKey) { runtime=new PresetRuntime(job.configuration!,signal);key=nextKey; }
        const reference=job.reference;
        if(reference && !references.has(reference.id))references.set(reference.id,await uploadWav(reference.audio,signal));
        const voice_ref=reference ? references.get(reference.id) : undefined;
        const result=await runtime.run({text:job.text,voice_ref,reference_text:reference?.transcript},job.sequence);
        if(result.kind!=='audio')throw new Error('The speech model did not return audio.');
        signal.throwIfAborted();
        const context=new AudioContext();let duration:number;
        try { duration=(await context.decodeAudioData(await result.blob.arrayBuffer())).duration; }finally{await context.close();}
        const chunk=project.chunks.find(c=>c.id===job.id)!;
        const take:Take={id:crypto.randomUUID(),text:job.text,configuration:job.configuration!,reference:reference || undefined,signature:job.signature,
          seed:(runtime.resolvedSeed+job.sequence)%0x100000000,audio:result.blob,duration,createdAt:Date.now()};
        chunk.takes=[...chunk.takes,take];if(!chunk.takeId)chunk.takeId=take.id;
        if(chunk.id===selectedId && chunk.takes.length>1)takingId=chunk.id;touch();
      }
      status='Audio generated. The first take is selected automatically; use Compare takes to choose a replacement.';
    }catch(e){status=signal.aborted?'Cancelled. Completed takes are saved.':'Generation stopped. Completed takes are saved.';if(!signal.aborted)error=String(e);}
    finally{busy=false;workingId='';aborter=null;await flush().catch(e=>error=`Could not save takes: ${String(e)}`);}
  }
  async function assemble() {
    if(!project || busy)return;
    if(!project.chunks.length || project.chunks.some(c=>!chunkReady(project!,c))) { error='Choose an up-to-date take for every chunk before assembly.';return; }
    busy=true;error='';aborter=new AbortController();status='Assembling chosen takes…';
    try {
      const audio=await assembleAudio(project.chunks.map(c=>({audio:selectedTake(c)!.rendered?.audio || selectedTake(c)!.audio,pause:c.pauseAfter ?? project!.assembly.pause})),project.assembly.sampleRate,aborter.signal);
      aborter.signal.throwIfAborted();release(project.assembled?.audio);project.assembled={audio,signature:assemblySignature(project)};touch();status='Narration assembled. Listen across chunk boundaries before finishing.';
    }catch(e){error=aborter.signal.aborted?'Cancelled.':String(e);}finally{busy=false;aborter=null;}
  }
  async function renderFinish(): Promise<boolean> {
    if(!project || !assemblyCurrent(project) || !project.assembled || busy) { error='Combine the chosen chunk audio in step 3 first.';return false; }
    busy=true;error='';aborter=new AbortController();status='Processing final audio…';const signal=aborter.signal;
    try {
      let audio=project.assembled.audio;
      if(project.enhancement) {
        status=`Processing with ${project.enhancement.name}…`;
        const runtime=new PresetRuntime(project.enhancement,signal);await runtime.prepare();const path=await uploadWav(audio,signal);const result=await runtime.run({audio:path});
        if(result.kind!=='audio')throw new Error('The processing model returned no audio.');audio=result.blob;
      }
      audio=await finishAudio(audio,project.finish,project.assembly.sampleRate,signal);signal.throwIfAborted();
      release(project.finished?.audio);project.finished={audio,signature:finishSignature(project)};exportLink='';touch();status='Processing preview ready. Listen before and after, or download your narration.';return true;
    }catch(e){if(signal.aborted)status='Processing cancelled.';else error=String(e);return false;}finally{busy=false;aborter=null;}
  }
  async function prepareExport(): Promise<boolean> {
    if(!project?.finished || !finalCurrent || busy)return false;
    busy=true;error='';aborter=new AbortController();status='Preparing download…';
    try {
      const audio=await exportAudio(project.finished.audio,project.finish.format,aborter.signal);aborter.signal.throwIfAborted();
      exportLink=urlFor(audio);exportName=`${project.title.replace(/[\\/:*?"<>|]/g,'_') || 'narration'}.${project.finish.format}`;
      exportedSignature=finishSignature(project)+project.finish.format;status='Download ready.';return true;
    }catch(e){if(aborter.signal.aborted)status='Export cancelled.';else error=String(e);return false;}finally{busy=false;aborter=null;}
  }
  async function downloadNarration() {
    if(!project || busy)return;
    if(!finalCurrent && !await renderFinish())return;
    if(!await prepareExport())return;
    const link=document.createElement('a');link.href=exportLink;link.download=exportName;document.body.append(link);link.click();link.remove();
  }
  function playback(event: Event) {
    const target=event.currentTarget;
    document.querySelectorAll<HTMLAudioElement>('.tts-app audio').forEach(a=>{if(a!==target)a.pause();});
  }
  onMount(async()=>{
    try {
      const [records, available, savedVoices]=await Promise.all([listAppProjects('tts'),loadModelChoices(),listVoices()]);
      choices=available;voices=savedVoices;projects=records.map(({id,title,updatedAt})=>({id,title,updatedAt}));
      if(records.length){const chosen=records.find(r=>r.id===localStorage.getItem('audiocpp.tts.project')) || records[0];const data=chosen.data as TtsProject;if(data.version!==1)throw new Error('Unsupported TTS project version.');project=data;stage=project.chunks.length?'speech':'script';selectedId=project.chunks.find(c=>c.id===localStorage.getItem(`audiocpp.tts.chunk:${project!.id}`))?.id || project.chunks[0]?.id || '';search='';filter='all';navigatorPage=Math.max(0,Math.floor(project.chunks.findIndex(c=>c.id===selectedId)/navigatorPageSize));assemblyPage=0;assemblySearch='';assemblyOnlyAttention=false;archivePage=0;defaultsOpen=!project.speech;}
      else {project=newProject();localStorage.setItem('audiocpp.tts.project',project.id);touch();}
    }catch(e){error=String(e);}finally{loading=false;}
  });
  onDestroy(()=>{aborter?.abort();clearTimeout(saveTimer);clearUrls();});
</script>

<svelte:window on:beforeunload={event=>{if(dirty || saving || busy){event.preventDefault();event.returnValue='';}}}/>
<section class="tts-app">
  <header class="project-bar">
    <div class="project-brand"><div class="project-mark"><AudioLines size={22}/></div><div><span class="eyebrow">TEXT TO SPEECH</span><h2>Narration projects</h2></div></div>
    <div class="project-actions">
      <label class="project-switch">Switch project<select aria-label="Open project" value={project?.id || ''} disabled={busy || loading} on:change={e=>selectProject(e.currentTarget.value)}>
        {#if project && !projects.some(p=>p.id===project!.id)}<option value={project.id}>{project.title}</option>{/if}
        {#each projects as p}<option value={p.id}>{p.title}</option>{/each}
      </select></label>
      <button disabled={busy || loading} on:click={createProject}><Plus size={15}/>New project</button>
    </div>
  </header>
  {#if loading}<p class="notice" role="status">Opening your projects…</p>{/if}
  {#if error}<div class="error" role="alert"><span>{error}</span><button aria-label="Dismiss error" on:click={()=>error=''}>×</button></div>{/if}
  {#if project}
    <div class="document-bar"><label class="project-title-field">Project name<input class="project-title" aria-label="Project title" bind:value={project.title} on:input={touch} disabled={busy}/></label><span class="save-status">{saving?'Saving…':dirty?'Unsaved changes':'Saved in this browser'}</span><button class="text-button" disabled={busy} on:click={()=>flush().catch(e=>error=String(e))}>Save project</button><button class="delete-project" title="Delete project" aria-label="Delete project" disabled={busy} on:click={removeProject}><Trash2 size={15}/>Delete project</button></div>
    {#if suggested}<div class="suggestion"><span>Use <strong>{suggested.name}</strong> for this project's speech?</span><button disabled={busy} on:click={suggest}>Use preset</button><button class="text-button" on:click={()=>suggested=undefined}>Dismiss</button></div>{/if}
    <nav class="pipeline" aria-label="TTS project pipeline">
      <button class:active={stage==='script'} aria-current={stage==='script'?'step':undefined} disabled={busy} on:click={()=>stage='script'}><span class="stage-icon"><FileText size={18}/></span><span><strong>1. Text & splitting</strong><small>{project.chunks.length?`${project.chunks.length} chunks`: 'Import & split text'}</small></span><ChevronRight size={14}/></button>
      <button class:active={stage==='speech'} aria-current={stage==='speech'?'step':undefined} disabled={busy} on:click={()=>stage='speech'}><span class="stage-icon"><AudioLines size={18}/></span><span><strong>2. Voices & audio</strong><small>{project.speech?.name || 'Choose model / preset'}</small></span><ChevronRight size={14}/></button>
      <button class:active={stage==='assemble'} aria-current={stage==='assemble'?'step':undefined} disabled={busy} on:click={()=>{stage='assemble';if(selectedIndex>=0)navigateAssembly(selectedIndex);}}><span class="stage-icon"><Layers size={18}/></span><span><strong>3. Order & pauses</strong><small>{ready}/{project.chunks.length} chunks ready</small></span><ChevronRight size={14}/></button>
      <button class:active={stage==='finish'} aria-current={stage==='finish'?'step':undefined} disabled={busy} on:click={()=>stage='finish'}><span class="stage-icon"><SlidersHorizontal size={18}/></span><span><strong>4. Finish & download</strong><small>{project.finish.format.toUpperCase()} · {project.enhancement?'Model + audio processing':'Audio processing'}</small></span></button>
    </nav>
    {#if status}<div class="operation-status" role="status"><span class:working={busy}></span>{status}{#if busy && aborter}<button on:click={()=>aborter?.abort()}><Square size={13}/>Cancel</button>{/if}</div>{/if}

    {#if stage==='script'}
      <div class="editor-layout">
        <section class="document-panel">
          <div class="section-title"><div><span class="eyebrow">START WITH WORDS</span><h3>Your script</h3></div><label class="file-button"><FileText size={15}/>Upload text file<input type="file" accept=".txt,.md,.markdown,text/plain,text/markdown" disabled={busy} on:change={e=>importText(e.currentTarget.files?.[0])}/></label></div>
          <textarea class="source-editor" aria-label="Script text" bind:value={project.source} on:input={()=>{preview=null;touch();}} disabled={busy} placeholder="Paste your script, story, or manuscript here…" rows="15"></textarea>
          <div class="source-footer"><span>{project.source.length.toLocaleString()} characters</span><button class="primary" disabled={busy || !project.source.trim()} on:click={previewSplit}><Scissors size={15}/>Split & preview text</button></div>
          {#if preview}
            <section class="split-preview"><div class="section-title"><h3>{preview.length} chunks</h3><button class="primary" on:click={applySplit}>Continue with {preview.length} chunks<ChevronRight size={14}/></button></div><p>Review the boundaries before generating. You can split or merge individual chunks later.</p><ol>{#each preview as text}<li>{text}</li>{/each}</ol></section>
          {/if}
        </section>
        <aside class="settings-panel"><span class="eyebrow">TEXT PREPARATION</span><h3>How should the text split?</h3><p>Each chunk becomes one piece of speech that you can listen to and regenerate separately.</p>
          <label>Split by<select aria-label="Split method" bind:value={project.split.method} disabled={busy} on:change={()=>{preview=null;touch();}}><option value="paragraph">Paragraphs</option><option value="sentence">Sentences</option><option value="size">Text length</option></select></label>
          <label>Maximum characters per chunk<input type="number" aria-label="Chunk size" min="40" step="10" bind:value={project.split.characters} disabled={busy} on:input={()=>{preview=null;touch();}}/></label>
          <div class="hint">Splitting an existing script keeps its previous chunks and takes in the project archive.</div>
        </aside>
      </div>
    {:else if stage==='speech'}
      <div class="stage-intro"><div><h3>Choose who reads each chunk</h3><p>Find a passage in the manuscript, then edit its voice, text and audio here.</p></div><button class:primary={ready===project.chunks.length && ready>0} disabled={busy || !project.chunks.length} on:click={()=>{stage='assemble';if(selectedIndex>=0)navigateAssembly(selectedIndex);}}>Next: order & pauses<ChevronRight size={16}/></button></div>
      <details class="speech-settings" bind:open={defaultsOpen}><summary>Project default narrator <span>{project.speech?.name || 'Choose a model and voice'}</span></summary><section aria-label="Default narrator">
        <div class="default-heading"><h4>Default narrator</h4><span>Used by chunks that follow the project default</span></div>
        <div class="narrator-fields">
          <ModelChoice label="Default model or preset" {choices} tasks={['tts','clon','vdes']} bind:value={project.speech} disabled={busy} onchange={touch}/>
          <VoiceChoice label="Default voice" configuration={project.speech} assignment={project.reference ? {kind:'reference',data:project.reference} : {kind:'builtin',id:project.speech?.request.voice || ''}} {voices} {choices} bind:transcription={project.transcription} disabled={busy} onchange={projectVoice} onbusy={v=>busy=v} ontranscribe={()=>transcribeReference()} onconfiguration={touch}/>
        </div>
        <div class="default-footer"><p>These choices belong to this project. Each chunk can use a different narrator.</p><button disabled={busy} on:click={refreshChoices}><RotateCcw size={14}/>Refresh models & presets</button><button disabled={busy} on:click={openLibrary}>Browse Lib</button></div>
      </section></details>
      <div class="chunk-toolbar"><div><h3>Your chunks <span>{project.chunks.length}</span></h3><p>{ready} ready · {needsGeneration.length} need recording{needsSelection.length ? ` · ${needsSelection.length} need a take chosen` : ''}</p></div><div class="tools"><button disabled={busy || !checked.size} on:click={()=>generate([...checked])}>Generate selected ({checked.size})</button><button class="primary" disabled={busy || !needsGeneration.length} on:click={()=>generate(needsGeneration.map(c=>c.id))}><Play size={15}/>Generate {needsGeneration.length} chunks needing audio</button></div></div>
      {#if needsSelection.length}<div class="choice-notice"><span>New recordings are ready for {needsSelection.length} chunks. Compare the takes below or use the newest matching audio.</span><button disabled={busy} on:click={useUpdatedTakes}>Use updated audio for {needsSelection.length} chunks</button></div>{/if}
      {#if !project.chunks.length}<div class="empty-state"><FileText size={36}/><h3>Add the words you want to hear</h3><p>Upload or paste text, split it into chunks, then choose a narrator here.</p><button class="primary" on:click={()=>stage='script'}>Go to text & splitting</button></div>{/if}
      {#if project.chunks.length}
      <div class="manuscript-workspace">
        <aside class="manuscript-nav" aria-label="Manuscript navigation">
          <h4>Manuscript <span>{project.chunks.length} chunks</span></h4>
          <label>Find a passage<input aria-label="Search chunks" placeholder="Search the book's text…" bind:value={search} on:input={()=>navigatorPage=0}/></label>
          <label>Show<select aria-label="Filter chunks" bind:value={filter} on:change={()=>navigatorPage=0}><option value="all">All chunks</option><option value="missing">Without audio</option><option value="changed">Audio needs updating</option></select></label>
          <form class="jump-control" on:submit|preventDefault={jumpToChunk}><label>Jump to chunk<input aria-label="Jump to chunk" type="number" min="1" max={project.chunks.length} bind:value={jumpNumber}/></label><button type="submit" disabled={busy}>Go</button></form>
          <div class="nav-shortcuts"><button disabled={busy || selectedIndex===0} on:click={()=>{search='';filter='all';navigatorPage=0;selectChunk(project!.chunks[0].id);}}>First</button><button disabled={busy || selectedIndex===project.chunks.length-1} on:click={()=>{search='';filter='all';navigatorPage=Math.floor((project!.chunks.length-1)/navigatorPageSize);selectChunk(project!.chunks.at(-1)!.id);}}>Last</button><button disabled={busy || ready===project.chunks.length} on:click={nextAttention}>Next needing attention</button></div>
          <label class="select-all"><input type="checkbox" aria-label="Select all matching chunks" checked={visible.length>0 && visible.every(c=>checked.has(c.id))} disabled={busy} on:change={e=>{const next=new Set(checked);for(const c of visible)e.currentTarget.checked?next.add(c.id):next.delete(c.id);checked=next;}}/>Select all {visible.length} matching</label>
          <div class="manuscript-outline" aria-label="Chunk outline">
            {#each navigatorRows as item (item.id)}
              {@const number=project.chunks.indexOf(item)+1}{@const isReady=chunkReady(project,item)}
              <div class="outline-row" class:current={item.id===selectedId}>
                <input type="checkbox" aria-label={`Select chunk ${number}`} disabled={busy} checked={checked.has(item.id)} on:change={e=>{const next=new Set(checked);e.currentTarget.checked?next.add(item.id):next.delete(item.id);checked=next;}}/>
                <button disabled={busy} aria-current={item.id===selectedId?'true':undefined} aria-label={`Edit chunk ${number}`} on:click={()=>selectChunk(item.id)}><span><strong>{number}</strong><small class:ready={isReady}>{workingId===item.id?'Generating…':isReady?'Ready':selectedTake(item)?'Needs update':'No audio'}</small></span><span class="outline-copy">{item.text.trim().slice(0,110) || 'Empty chunk'}</span></button>
              </div>
            {/each}
            {#if !visible.length}<p>No passages match. Clear the search or change the filter.</p>{/if}
          </div>
          <div class="outline-pages"><button aria-label="Previous outline page" disabled={navigatorPage===0} on:click={()=>navigatorPage--}>Previous</button><span>{navigatorPage+1} / {navigatorPages}</span><button aria-label="Next outline page" disabled={navigatorPage===navigatorPages-1} on:click={()=>navigatorPage++}>Next</button></div>
          <button disabled={busy} on:click={()=>{const chunk=newChunk('');project!.chunks=[...project!.chunks,chunk];search='';filter='all';selectChunk(chunk.id);navigatorPage=Math.floor((project!.chunks.length-1)/navigatorPageSize);touch();}}><Plus size={14}/>Add chunk at end</button>
        </aside>
        <section class="chunk-editor" aria-label="Selected chunk editor">
          <div class="editor-navigation"><button disabled={busy || selectedIndex<=0} on:click={()=>selectChunk(project!.chunks[selectedIndex-1].id)}>Previous chunk</button><strong>Editing chunk {selectedIndex+1} of {project.chunks.length}</strong><button disabled={busy || selectedIndex<0 || selectedIndex===project.chunks.length-1} on:click={()=>selectChunk(project!.chunks[selectedIndex+1].id)}>Next chunk</button></div>
          {#each selectedChunk ? [selectedChunk] : [] as chunk (chunk.id)}
          {@const take=selectedTake(chunk)}{@const index=project.chunks.indexOf(chunk)}{@const current=chunkReady(project,chunk)}
          <article class="chunk" class:selected={selectedId===chunk.id} class:generating={workingId===chunk.id} aria-label={`Chunk ${index+1}`}>
            <header class="chunk-head"><strong class="chunk-select">Chunk {index+1}</strong><span class="chunk-badge" class:ready={current}>{workingId===chunk.id?'Generating audio…':!take?'No audio yet':current?'Ready to use':chunk.takes.some(t=>t.signature===speechSignature(project!,chunk))?'Choose a new take':'Audio needs updating'}</span><div class="chunk-tools"><button title="Move this chunk earlier in the narration" disabled={busy || index===0} on:click={()=>moveChunk(chunk,-1)}><ArrowUp size={14}/>Move up</button><button title="Move this chunk later in the narration" disabled={busy || index===project!.chunks.length-1} on:click={()=>moveChunk(chunk,1)}><ArrowDown size={14}/>Move down</button></div></header>
            <div class="chunk-narrator">
              <ModelChoice label={`Chunk ${index+1} model or preset`} emptyLabel={project.speech ? `Use project default · ${project.speech.name}` : 'Use project default (choose one above)'} inherit={project.speech} {choices} tasks={['tts','clon','vdes']} bind:value={chunk.override} disabled={busy} onchange={kind=>{if(kind==='model' && chunk.voice?.kind==='builtin')chunk.voice=undefined;touch();}}/>
              <VoiceChoice label={`Chunk ${index+1} voice`} configuration={chunk.override || project.speech} assignment={chunk.voice} inheritedReference={project.reference} allowInherit {voices} {choices} bind:transcription={project.transcription} disabled={busy} onchange={v=>{chunk.voice=v;touch();}} onbusy={v=>busy=v} ontranscribe={()=>transcribeReference(chunk)} onconfiguration={touch}/>
            </div>
            <label class="chunk-copy-label" for={`text-${chunk.id}`}>Text to read</label>
            <textarea id={`text-${chunk.id}`} class="chunk-text" aria-label={`Chunk ${index+1} text`} bind:this={editors[chunk.id]} bind:value={chunk.text} rows="3" disabled={busy} on:focus={()=>selectedId=chunk.id} on:input={touch}></textarea>
            <div class="chunk-edit-tools"><button disabled={busy} on:click={()=>splitChunk(chunk)}><Scissors size={14}/>Split text at cursor</button><button disabled={busy || index===project.chunks.length-1} on:click={()=>mergeChunk(chunk)}><Merge size={14}/>Join with next chunk</button><button class="remove-chunk" disabled={busy} on:click={()=>archiveChunk(chunk)}>Remove chunk</button><span>{chunk.text.length} characters</span></div>
            {#if take}<div class="chosen-audio"><span>{current?`Used in narration · Take ${chunk.takes.indexOf(take)+1}`:chunk.takes.some(t=>t.signature===speechSignature(project!,chunk))?'Older audio · choose an updated take below':'Older audio · generate an update'}<small>{(take.rendered?.duration ?? take.duration).toFixed(1)} seconds</small></span></div>{/if}
            {#if take}
              {#key take.id}<WaveformEditor {take} sampleRate={project.assembly.sampleRate} increment={project.pauseIncrement ?? .1} disabled={busy} {active} onchange={previous=>{release(previous);touch();}} onincrement={seconds=>{project!.pauseIncrement=seconds;touch();}}/>{/key}
              <details class="speech-boundary" bind:open={speechBoundaryOpen}><summary>Gap to the next chunk & boundary preview</summary>{#if speechBoundaryOpen}<BoundaryEditor {project} {chunk} next={project.chunks[index+1]} {index} increment={project.pauseIncrement ?? .1} disabled={busy} {active} onchange={touch}/>{/if}</details>
            {/if}
            <footer class="chunk-footer"><button class="primary generate-chunk" disabled={busy} on:click={()=>generate([chunk.id])}><Play size={15}/>{!take?'Generate audio':current?'Try another take':'Generate updated audio'}</button>{#if chunk.takes.length}<button class="compare-takes" aria-expanded={takingId===chunk.id} on:click={()=>takingId=takingId===chunk.id?'':chunk.id}>{takingId===chunk.id?'Hide take comparison':`Compare ${chunk.takes.length} ${chunk.takes.length===1?'take':'takes'}`}<ChevronDown size={14}/></button>{/if}<p>A take is one recording. Trying again keeps the audio you've already chosen.</p></footer>
            {#if takingId===chunk.id && chunk.takes.length}<div class="takes" aria-label={`Chunk ${index+1} takes`}><p>Play the alternatives, then choose which recording goes into the narration.</p>{#each chunk.takes as candidate,i}<div class="take" class:chosen={candidate.id===chunk.takeId}><div><strong>Take {i+1}</strong><span>{(candidate.rendered?.duration ?? candidate.duration).toFixed(1)} seconds · {candidate.configuration.model.display_name}</span>{#if candidate.signature!==speechSignature(project,chunk)}<small>Recorded with earlier text or settings</small>{/if}</div><audio controls src={urlFor(candidate.rendered?.audio || candidate.audio)} on:play={playback} aria-label={`Chunk ${index+1} take ${i+1}`}></audio><button disabled={busy || candidate.id===chunk.takeId} on:click={()=>useTake(chunk,candidate)}>{#if candidate.id===chunk.takeId}<Check size={14}/>Used in narration{:else}Use in narration{/if}</button></div>{/each}</div>{/if}
          </article>
        {/each}
        </section>
      </div>
      {/if}
      {#if project.archivedChunks.length}<details class="archive" bind:open={archiveOpen}><summary>Removed & replaced chunks ({project.archivedChunks.length})</summary><p>Splitting, joining or removing chunks keeps the earlier text and recordings here.</p>{#if archiveOpen}{#each project.archivedChunks.slice(archivePage*navigatorPageSize,(archivePage+1)*navigatorPageSize) as chunk,pageIndex}{@const index=archivePage*navigatorPageSize+pageIndex}<div><span>{chunk.text.slice(0,100)} · {chunk.takes.length} recordings</span><button disabled={busy} on:click={()=>{project!.chunks=[...project!.chunks,chunk];project!.archivedChunks.splice(index,1);touch();}}>Restore as last chunk</button></div>{/each}<div class="outline-pages"><button disabled={archivePage===0} on:click={()=>archivePage--}>Previous archived chunks</button><span>Page {archivePage+1}</span><button disabled={(archivePage+1)*navigatorPageSize>=project.archivedChunks.length} on:click={()=>archivePage++}>Next archived chunks</button></div>{/if}</details>{/if}
      {#if project.chunks.length}<div class="stage-next"><span>{ready===project.chunks.length?'All chunks have chosen audio. Ready to combine them.':`${project.chunks.length-ready} chunks still need audio or a matching take.`}</span><button class="primary" disabled={busy} on:click={()=>{stage='assemble';if(selectedIndex>=0)navigateAssembly(selectedIndex);}}>Next: order & pauses<ChevronRight size={15}/></button></div>{/if}
    {:else if stage==='assemble'}
      <div class="editor-layout">
        <section class="document-panel"><div class="section-title"><div><span class="eyebrow">BRING IT TOGETHER</span><h3>Combine the chosen recordings</h3></div><span>{ready}/{project.chunks.length} ready</span></div>
          {#if project.chunks.length}
          <section class="assembly-navigation" aria-label="Order and pauses navigation">
            <div class="assembly-find"><label>Find a passage<input aria-label="Search order and pauses" placeholder="Search the book's text…" bind:value={assemblySearch} on:input={()=>assemblyPage=0}/></label><label>Show<select aria-label="Filter order and pauses" bind:value={assemblyOnlyAttention} on:change={()=>assemblyPage=0}><option value={false}>All chunks</option><option value={true}>Needs attention</option></select></label></div>
            <div class="assembly-jump"><form class="jump-control" on:submit|preventDefault={jumpToChunk}><label>Jump to chunk<input aria-label="Jump to chunk" type="number" min="1" max={project.chunks.length} bind:value={jumpNumber}/></label><button type="submit" disabled={busy}>Go</button></form><button disabled={busy} on:click={()=>navigateAssembly(0)}>First chunk</button><button disabled={busy} on:click={()=>navigateAssembly(project!.chunks.length-1)}>Last chunk</button><button disabled={busy || ready===project.chunks.length} on:click={nextAttention}>Next needing attention</button></div>
            <div class="assembly-traverse"><button disabled={busy || selectedIndex<=0} on:click={()=>navigateAssembly(selectedIndex-1)}>Previous chunk</button><span>{selectedIndex>=0?`Selected chunk ${selectedIndex+1} of ${project.chunks.length}`:'Choose a chunk'}</span><button disabled={busy || selectedIndex<0 || selectedIndex===project.chunks.length-1} on:click={()=>navigateAssembly(selectedIndex+1)}>Next chunk</button></div>
            <div class="assembly-page-controls"><button disabled={assemblyPage===0} on:click={()=>assemblyPage--}>Previous page</button><span>{assemblyMatches.length ? `${assemblyPage*navigatorPageSize+1}–${Math.min((assemblyPage+1)*navigatorPageSize,assemblyMatches.length)} of ${assemblyMatches.length} matching chunks` : 'No passages match'}</span><button disabled={assemblyPage===assemblyPages-1} on:click={()=>assemblyPage++}>Next page</button><button disabled={assemblyPage===assemblyPages-1} on:click={()=>assemblyPage=assemblyPages-1}>Last page</button></div>
          </section>
          {/if}
          <div class="boundary-tools"><label>Add per click<select aria-label="Pause increment" value={project.pauseIncrement ?? .1} disabled={busy} on:change={e=>{project!.pauseIncrement=Number(e.currentTarget.value);touch();}}>{#each [.05,.1,.25,.5,1] as seconds}<option value={seconds}>+{seconds.toFixed(2)} seconds</option>{/each}</select></label><p>Choose a chunk to adjust the gap after it. Recorded silence stays visible in the waveform.</p></div>
          <div class="boundary-workspace">
            <div class="assembly-list" aria-label="Boundary navigator">
              {#each assemblyRows as chunk (chunk.id)}
                {@const index=project.chunks.indexOf(chunk)}
                <button class="assembly-row" id={`assembly-${chunk.id}`} class:assembly-selected={selectedId===chunk.id} aria-label={`Select chunk ${index+1}`} aria-current={selectedId===chunk.id?'true':undefined} disabled={busy} on:click={()=>selectChunk(chunk.id)}><strong>{index+1}</strong><span>{chunk.text.slice(0,100)}<small>{index<project.chunks.length-1?`${chunk.pauseAfter===null?'Default':'Custom'} gap · ${(chunk.pauseAfter ?? project.assembly.pause).toFixed(2)} s`:'Final chunk'}{!chunkReady(project,chunk)?' · Needs audio':''}</small></span></button>
              {/each}
            </div>
            {#if selectedChunk}
              <div class="selected-boundary">
                <BoundaryEditor {project} chunk={selectedChunk} next={project.chunks[selectedIndex+1]} index={selectedIndex} increment={project.pauseIncrement ?? .1} disabled={busy} active={active && !busy} onchange={touch}/>
                <div class="boundary-order"><button disabled={busy || selectedIndex===0} on:click={()=>{const destination=selectedIndex-1;moveChunk(selectedChunk!,-1);navigateAssembly(destination);}}>Move chunk up</button><button disabled={busy || selectedIndex===project.chunks.length-1} on:click={()=>{const destination=selectedIndex+1;moveChunk(selectedChunk!,1);navigateAssembly(destination);}}>Move chunk down</button><button disabled={busy} on:click={()=>{stage='speech';filter='all';search='';selectChunk(selectedId);}}>Edit selected chunk voice / text / audio</button></div>
              </div>
            {/if}
          </div>
          {#if project.assembled}<div class="combined-preview"><h4>{assemblyIsCurrent?'Narration preview':'Previous assembly · out of date'}</h4><audio controls src={urlFor(project.assembled.audio)} bind:this={player} on:play={playback}></audio></div>{/if}
        </section>
        <aside class="settings-panel"><span class="eyebrow">ASSEMBLY</span><h3>Spacing & audio</h3><label>Default gap between chunks (seconds)<input aria-label="Default pause" type="number" min="0" step="0.05" bind:value={project.assembly.pause} disabled={busy} on:input={touch}/></label><p>Applies to boundaries using the default. Custom gaps stay unchanged.</p><label>Output sample rate<select aria-label="Output sample rate" bind:value={project.assembly.sampleRate} disabled={busy} on:change={touch}>{#each [16000,22050,24000,44100,48000] as rate}<option value={rate}>{rate.toLocaleString()} Hz</option>{/each}</select></label><button class="primary" disabled={busy || !project.chunks.length || ready!==project.chunks.length} on:click={assemble}><Layers size={15}/>{project.assembled?'Update combined audio':'Combine chosen audio'}</button>{#if ready!==project.chunks.length}<p class="hint">Choose an up-to-date take for each chunk. Use Edit voice / text / audio to fix the ones needing attention.</p>{/if}<button disabled={busy || !assemblyIsCurrent} on:click={()=>stage='finish'}>Next: finish & download<ChevronRight size={14}/></button></aside>
      </div>
    {:else}
      <div class="editor-layout">
        <section class="document-panel"><div class="section-title"><div><span class="eyebrow">THE FINAL LISTEN</span><h3>Finish your narration</h3></div><span class="chunk-badge" class:ready={finalCurrent}>{finalCurrent?'Preview ready':'Preview optional'}</span></div>
          <p class="intro">Optional enhancement and volume adjustment apply to the combined narration. Download applies these settings automatically; the chosen chunk recordings stay unchanged.</p>
          {#if !assemblyIsCurrent}<div class="notice">The assembly is missing or out of date.<button disabled={busy} on:click={()=>{stage='assemble';if(selectedIndex>=0)navigateAssembly(selectedIndex);}}>Review assembly</button></div>{/if}
          <div class="ab-preview"><div><h4>Before processing</h4>{#if project.assembled}<audio controls src={urlFor(project.assembled.audio)} on:play={playback}></audio>{:else}<p>Build your assembly first.</p>{/if}</div><div><h4>After processing {#if project.finished && !finalCurrent}<small>Out of date</small>{/if}</h4>{#if project.finished}<audio controls src={urlFor(project.finished.audio)} on:play={playback}></audio>{:else}<p>Use Preview processing to hear the result before downloading.</p>{/if}</div></div>
          <div class="finish-process"><h4>1. Optional enhancement</h4><ModelChoice label="Enhancement model or preset" emptyLabel="No enhancement · keep the original sound" {choices} tasks={['s2s','gen']} families={enhancementFamilies} bind:value={project.enhancement} disabled={busy} onchange={touch}/><p class="hint">Leave unconfigured to keep the original audio. The model runs before level adjustment.</p><button class="text-button" disabled={busy} on:click={refreshChoices}>Refresh models & presets</button></div>
          <div class="finish-process"><h4>2. Adjust volume</h4><label class="toggle-label"><input type="checkbox" aria-label="Peak normalization" bind:checked={project.finish.normalize} disabled={busy} on:change={touch}/>Set a consistent peak volume</label><div class="level-controls">{#if project.finish.normalize}<label>Maximum peak level (dBFS)<input aria-label="Target peak" type="number" max="0" min="-60" step="0.5" bind:value={project.finish.peakDb} disabled={busy} on:input={touch}/></label>{:else}<label>Make louder / quieter (dB)<input aria-label="Audio gain" type="number" min="-60" max="24" step="0.5" bind:value={project.finish.gainDb} disabled={busy} on:input={touch}/></label>{/if}</div></div>
          <button class="primary" disabled={busy || !assemblyIsCurrent} on:click={renderFinish}><SlidersHorizontal size={15}/>{project.finished?'Update processing preview':'Preview processing'}</button>
        </section>
        <aside class="settings-panel"><span class="eyebrow">DOWNLOAD</span><h3>Save your narration</h3><label>Audio format<select aria-label="Export format" bind:value={project.finish.format} disabled={busy} on:change={()=>{exportLink='';touch();}}><option value="wav">WAV · PCM 16-bit</option><option value="webm">WebM · Opus</option></select></label><p>WAV preserves uncompressed audio. WebM creates a smaller file using the browser's Opus encoder.</p><button class="primary download-now" disabled={busy || !assemblyIsCurrent} on:click={downloadNarration}><Download size={17}/>Download {project.finish.format.toUpperCase()}</button><p>Applies your processing choices and downloads the file. Previewing first is optional.</p>{#if exportLink && finalCurrent && exportedSignature===finishSignature(project)+project.finish.format}<a class="download-again" href={exportLink} download={exportName}>Download again</a>{/if}<div class="hint">Projects and generated audio are saved in this browser. Download the final audio to keep a separate file.</div></aside>
      </div>
    {/if}
  {/if}
</section>

<style>
  .tts-app { --tts-accent:var(--cyan); max-width:1440px; margin:0 auto; padding-bottom:50px; }
  button { display:inline-flex; align-items:center; justify-content:center; gap:7px; } a { color:var(--tts-accent); }
  .project-bar,.project-brand,.project-actions,.document-bar,.section-title,.tools { display:flex; align-items:center; gap:12px; }
  .project-bar { justify-content:space-between; padding:18px 0 24px; }.project-brand { gap:14px; }.project-mark { display:grid; place-items:center; height:44px; width:44px; background:var(--active-bg); color:var(--tts-accent); border-radius:12px; }
  .eyebrow { font-size:10px; font-weight:600; letter-spacing:1.8px; color:var(--tts-accent); }.project-brand h2 { font-size:18px; margin:4px 0 0; }.project-actions select { width:220px; }
  .document-bar { padding:14px 0; border-top:1px solid var(--line); }.project-title { flex:1; width:120px; min-width:0; font-size:24px; background:transparent; border:0; padding-left:0; }.save-status { font-size:11px; color:var(--muted); white-space:nowrap; }
  .text-button { background:transparent; border:0; color:var(--muted); }
  .pipeline { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:10px 0 26px; }.pipeline button { text-align:left; justify-content:flex-start; padding:15px 12px; background:var(--panel); border:1px solid var(--line); gap:12px; min-width:0; }.pipeline button.active { border-color:var(--tts-accent); background:var(--active-bg); }.pipeline button>span:nth-child(2) { flex:1; min-width:0; }.pipeline strong { display:block; font-size:12px; }.pipeline small { display:block; color:var(--muted); font-size:10px; margin-top:7px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }.stage-icon { color:var(--tts-accent); }
  .editor-layout { display:grid; grid-template-columns:minmax(0,1fr) 310px; gap:28px; align-items:start; }.document-panel { padding:24px; background:var(--panel); border:1px solid var(--line); border-radius:10px; min-width:0; }.section-title { justify-content:space-between; margin-bottom:22px; flex-wrap:wrap; }.section-title h3 { margin:6px 0 0; font-size:20px; }.section-title>span { font-size:12px; color:var(--muted); }
  .settings-panel { padding:24px; border:1px solid var(--line); border-radius:10px; display:grid; gap:20px; background:var(--panel); }.settings-panel h3 { margin:-8px 0 0; font-size:18px; } .settings-panel p,.intro,.split-preview p { color:var(--muted); font-size:12px; line-height:1.7; margin:0; }
  label { display:grid; gap:8px; color:var(--muted); font-size:12px; } input,textarea,select { min-width:0; width:100%; } textarea { resize:vertical; line-height:1.8; }.source-editor { background:var(--control-bg); min-height:330px; border-color:var(--line); font-size:15px; }.file-button { display:inline-flex; align-items:center; gap:7px; cursor:pointer; font-size:12px; color:var(--tts-accent); }.file-button input { display:none; }
  .source-footer { display:flex; justify-content:space-between; align-items:center; margin-top:18px; gap:10px; }.source-footer span { font-size:11px; color:var(--muted); }.hint { padding:12px; background:var(--active-bg); font-size:11px; line-height:1.7; color:var(--muted); border-radius:6px; }
  .split-preview { margin-top:28px; border-top:1px solid var(--line); padding-top:22px; }.split-preview ol { padding-left:28px; max-height:400px; overflow:auto; }.split-preview li { padding:14px 10px; border-bottom:1px solid var(--line); font-size:14px; line-height:1.8; white-space:pre-wrap; }
  .speech-settings { border:1px solid var(--line); border-radius:10px; padding:22px; background:var(--panel); margin-bottom:28px; } audio { width:100%; height:34px; }
  .chunk-toolbar { display:flex; justify-content:space-between; gap:18px; align-items:center; margin-bottom:18px; }.chunk-toolbar h3 { margin:0; font-size:19px; }.chunk-toolbar h3 span { font-size:12px; color:var(--muted); margin-left:8px; }.chunk-toolbar p { font-size:11px; color:var(--muted); margin:6px 0 0; }
  .chunk { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }.chunk.selected { border-color:var(--tts-accent); }.chunk.generating { outline:1px solid var(--tts-accent); }.chunk-head { display:flex; align-items:center; gap:12px; padding:14px 20px 8px; }.chunk-select { display:flex; align-items:center; gap:10px; color:var(--muted); font-size:12px; }.chunk-badge { font-size:10px; padding:4px 8px; background:var(--active-bg); border-radius:20px; color:var(--muted); }.chunk-badge.ready { color:var(--tts-accent); }.chunk-tools { display:flex; gap:4px; margin-left:auto; }.chunk-text { display:block; width:calc(100% - 40px); margin:4px 20px; border:0; background:transparent; padding:6px 0; min-height:76px; font-size:16px; box-shadow:none; }.chunk-text:focus { outline:none; }.chunk-edit-tools { display:flex; gap:4px; padding:2px 14px 12px; flex-wrap:wrap; }.chunk-edit-tools button { background:transparent; border:0; color:var(--muted); font-size:11px; padding:5px 7px; }.chosen-audio { display:flex; align-items:center; gap:18px; padding:14px 20px; border-top:1px solid var(--line); }.chosen-audio>span { font-size:11px; color:var(--muted); min-width:140px; }.chunk-footer { display:flex; align-items:center; gap:10px; padding:12px 20px; border-top:1px solid var(--line); }.generate-chunk { font-size:12px; border-color:var(--tts-accent); }.takes { padding:12px 20px; border-top:1px solid var(--line); display:grid; gap:10px; }.take { display:flex; align-items:center; gap:14px; padding:10px; border:1px solid var(--line); border-radius:6px; }.take.chosen { border-color:var(--tts-accent); }.take>div { min-width:140px; display:grid; gap:5px; }.take strong { font-size:12px; }.take span,.take small { font-size:10px; color:var(--muted); }.take button { min-width:90px; font-size:11px; }
  .assembly-list { display:grid; gap:8px; }.combined-preview { margin-top:28px; }.combined-preview h4 { font-size:13px; }
  .ab-preview { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:24px 0; }.ab-preview>div { min-width:0; padding:16px; background:var(--active-bg); border-radius:8px; }.ab-preview h4 { margin:0 0 14px; font-size:12px; }.ab-preview p { font-size:12px; color:var(--muted); }.ab-preview small { font-size:10px; color:var(--muted); }.finish-process { padding:24px 0; border-top:1px solid var(--line); }.finish-process h4 { margin:0 0 18px; font-size:14px; }.finish-process>.hint { margin-top:14px; }.toggle-label { display:flex; align-items:center; gap:8px; }.toggle-label input { height:16px; width:16px; accent-color:var(--tts-accent); }.level-controls { max-width:240px; margin-top:16px; }
  .error,.notice,.suggestion { display:flex; align-items:center; gap:12px; padding:14px 18px; margin:12px 0; border:1px solid var(--line); border-radius:8px; font-size:12px; line-height:1.6; }.error { color:var(--danger); }.error>span { flex:1; }.error button { padding:2px 8px; }.notice { color:var(--muted); }.suggestion { background:var(--active-bg); }.suggestion>span { flex:1; }.operation-status { display:flex; align-items:center; gap:10px; font-size:12px; color:var(--muted); margin:0 0 20px; }.operation-status>span { height:7px; width:7px; border-radius:50%; background:var(--muted); }.operation-status>span.working { background:var(--tts-accent); }.operation-status>button { margin-left:auto; }.empty-state { display:grid; justify-items:center; gap:14px; padding:60px 20px; color:var(--muted); border:1px dashed var(--line); border-radius:10px; }.empty-state h3,.empty-state p { margin:0; }.empty-state p { font-size:12px; }.archive { margin-top:24px; font-size:12px; color:var(--muted); }.archive>div { display:flex; gap:12px; align-items:center; justify-content:space-between; padding:10px 0; border-bottom:1px solid var(--line); }
  @media(max-width:1000px) { .editor-layout { grid-template-columns:minmax(0,1fr) 260px; gap:18px; }.pipeline button { padding:12px 8px; gap:8px; }.stage-icon { display:none; } }
  @media(max-width:760px) { .editor-layout { grid-template-columns:1fr; }.pipeline { grid-template-columns:1fr 1fr; }.project-bar { align-items:flex-start; gap:20px; flex-direction:column; }.project-actions { width:100%; }.project-actions select { flex:1; width:100px; }.document-bar { flex-wrap:wrap; }.project-title { flex-basis:60%; font-size:20px; }.save-status { font-size:10px; }.chunk-toolbar { flex-direction:column; align-items:flex-start; }.chunk-toolbar .tools { flex-wrap:wrap; }.chosen-audio { flex-direction:column; align-items:flex-start; gap:10px; }.take { flex-wrap:wrap; }.take>div { flex:1; }.take audio { order:3; }.chunk-footer { flex-wrap:wrap; }.ab-preview { grid-template-columns:1fr; }.document-panel,.settings-panel { padding:18px; }.suggestion { flex-wrap:wrap; } }

  .tts-app button { min-height:38px; font-size:13px; }
  .tts-app button:focus-visible,.tts-app a:focus-visible { outline:2px solid var(--tts-accent); outline-offset:3px; }
  .project-switch { font-size:11px; gap:6px; }
  .project-title-field { flex:1; min-width:160px; font-size:11px; gap:4px; }
  .project-title { width:100%; padding:4px 0; font-size:25px; }
  .document-bar { align-items:center; }.document-bar>.text-button { border:1px solid var(--line); color:var(--text); }
  .delete-project { font-size:12px; }.save-status { font-size:12px; }
  .stage-intro { display:flex; align-items:center; justify-content:space-between; gap:24px; margin:4px 0 22px; }
  .stage-intro h3 { font-size:22px; margin:0 0 8px; }.stage-intro p { font-size:14px; color:var(--muted); line-height:1.6; margin:0; }
  .stage-intro>button { flex-shrink:0; }
  .default-heading { display:flex; align-items:baseline; flex-wrap:wrap; gap:12px; margin-bottom:20px; }
  .default-heading h4 { margin:0; font-size:17px; }.default-heading span { font-size:12px; color:var(--muted); }
  .narrator-fields,.chunk-narrator { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
  .default-footer { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:16px; border-top:1px solid var(--line); padding-top:16px; }
  .default-footer p { margin:0; flex:1; min-width:200px; font-size:12px; color:var(--muted); line-height:1.5; }
  .default-footer button { font-size:12px; }
  .pipeline strong { font-size:13px; }.pipeline small { font-size:12px; }
  .file-button { border:1px solid var(--tts-accent); border-radius:6px; padding:10px 14px; font-size:13px; background:var(--active-bg); }
  .source-footer span,.section-title>span,.settings-panel p,.intro,.split-preview p { font-size:13px; }
  .hint { font-size:12px; }.source-editor { font-size:16px; }
  .select-all { display:flex; align-items:center; gap:8px; white-space:nowrap; font-size:13px; }.select-all input { width:16px; height:16px; }
  .chunk-toolbar p { font-size:13px; line-height:1.6; }
  .chunk { border-radius:10px; }.chunk-head { padding:18px 20px; background:var(--active-bg); }.chunk-select { font-size:15px; color:var(--text); }
  .chunk-badge { font-size:12px; padding:5px 10px; }.chunk-tools button { font-size:12px; }
  .chunk-narrator { padding:20px; border-bottom:1px solid var(--line); }
  .chunk-copy-label { margin:18px 20px 0; font-size:12px; color:var(--muted); }
  .chunk-text { border:1px solid var(--line); border-radius:6px; background:var(--control-bg); padding:12px; margin:8px 20px; min-height:90px; font-size:16px; }
  .chunk-text:focus { outline:2px solid var(--tts-accent); outline-offset:1px; }
  .chunk-edit-tools { padding:4px 20px 18px; gap:8px; align-items:center; }
  .chunk-edit-tools button { background:var(--control-bg); border:1px solid var(--line); color:var(--text); font-size:12px; padding:7px 10px; }
  .chunk-edit-tools>span { margin-left:auto; font-size:12px; color:var(--muted); }.chunk-edit-tools .remove-chunk { margin-left:auto; }
  .chosen-audio>span { font-size:13px; min-width:210px; color:var(--text); }.chosen-audio small { display:block; color:var(--muted); margin-top:6px; font-size:12px; }
  .chunk-footer { flex-wrap:wrap; }.chunk-footer p { flex-basis:100%; margin:4px 0 0; font-size:12px; line-height:1.5; color:var(--muted); }
  .takes>p { font-size:13px; line-height:1.6; margin:0 0 6px; color:var(--muted); }
  .take>div { min-width:180px; }.take strong { font-size:14px; }.take span,.take small { font-size:12px; }.take button { min-width:140px; font-size:13px; }.settings-panel label { font-size:13px; }
  .ab-preview h4,.finish-process h4 { font-size:15px; }.ab-preview p { font-size:13px; }
  .download-now { min-height:48px!important; font-size:16px!important; }.download-again { font-size:13px; text-align:center; }
  .stage-next,.choice-notice { display:flex; align-items:center; gap:18px; justify-content:space-between; padding:18px; border:1px solid var(--line); background:var(--active-bg); border-radius:8px; margin:20px 0; }
  .stage-next>span,.choice-notice>span { font-size:13px; line-height:1.6; color:var(--muted); }.stage-next button,.choice-notice button { flex-shrink:0; }
  @media(max-width:760px) {
    .narrator-fields,.chunk-narrator { grid-template-columns:1fr; gap:18px; }
    .stage-intro,.stage-next,.choice-notice { flex-direction:column; align-items:stretch; gap:14px; }
    .project-actions { align-items:end; }.project-switch { flex:1; min-width:0; }.project-actions select { width:100%; }
    .document-bar { align-items:center; }.project-title-field { flex-basis:100%; }.save-status { flex:1; }
    .chunk-head { flex-wrap:wrap; gap:10px; }.chunk-tools { margin-left:0; flex-basis:100%; }
    .chunk-edit-tools>span { flex-basis:100%; margin-left:0; }.chunk-edit-tools .remove-chunk { margin-left:0; }
    .chunk-narrator { padding:18px; }.take>div { min-width:0; }.take button { min-width:130px; }
  }


  .speech-settings>summary { cursor:pointer; font-size:15px; font-weight:600; padding:2px 0; }
  .speech-settings>summary>span { margin-left:14px; font-weight:400; font-size:13px; color:var(--muted); }
  .speech-settings[open]>section { margin-top:22px; }
  .manuscript-workspace { display:grid; grid-template-columns:300px minmax(0,1fr); gap:22px; align-items:start; }
  .manuscript-nav { position:sticky; top:82px; border:1px solid var(--line); border-radius:10px; background:var(--panel); padding:16px; display:grid; gap:12px; min-width:0; }
  .manuscript-nav h4 { display:flex; justify-content:space-between; margin:0; font-size:16px; }
  .manuscript-nav h4 span { font-size:12px; font-weight:400; color:var(--muted); }
  .manuscript-nav label { font-size:12px; }
  .jump-control { display:flex; align-items:end; gap:8px; }.jump-control label { flex:1; }.jump-control input { width:100%; }
  .nav-shortcuts { display:flex; flex-wrap:wrap; gap:6px; }.nav-shortcuts button { flex:1; font-size:12px; }.nav-shortcuts button:last-child { flex-basis:100%; }
  .manuscript-outline { max-height:min(390px,35dvh); overflow-y:auto; overscroll-behavior:contain; border-block:1px solid var(--line); }
  .outline-row { display:flex; align-items:center; gap:8px; border-bottom:1px solid var(--line); padding:4px 8px; border-left:3px solid transparent; }
  .outline-row.current { background:var(--active-bg); border-left-color:var(--tts-accent); }
  .outline-row>input { width:16px; height:16px; flex-shrink:0; }
  .outline-row>button { flex:1; min-width:0; text-align:left; display:grid; gap:6px; border:0; background:transparent; padding:8px 0; }
  .outline-row>button>span:first-child { display:flex; align-items:center; justify-content:space-between; width:100%; }
  .outline-row strong { font-size:13px; }.outline-row small { font-size:11px; color:var(--muted); }.outline-row small.ready { color:var(--tts-accent); }
  .outline-copy { display:-webkit-box; line-clamp:2;-webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; font-size:12px; line-height:1.6; font-weight:400; }
  .outline-pages { display:flex; align-items:center; justify-content:space-between; gap:6px; font-size:12px; color:var(--muted); }.outline-pages button { font-size:12px; }
  .chunk-editor { min-width:0; }.editor-navigation { scroll-margin-top:90px; display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:14px; }.editor-navigation strong { font-size:14px; }
  .assembly-page-controls { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-bottom:18px; font-size:13px; }
  @media(max-width:1000px) { .manuscript-workspace { grid-template-columns:260px minmax(0,1fr); }.chunk-narrator { grid-template-columns:1fr; }.editor-navigation { flex-wrap:wrap; }.editor-navigation strong { order:-1; flex-basis:100%; } }
  @media(max-width:760px) { .manuscript-workspace { grid-template-columns:1fr; }.manuscript-nav { position:static; }.manuscript-outline { max-height:190px; }.speech-settings>summary>span { display:block; margin:8px 0 0; }.editor-navigation { scroll-margin-top:150px; } }


  .assembly-navigation { display:grid; gap:14px; padding:16px; border:1px solid var(--line); border-radius:8px; margin-bottom:18px; background:var(--active-bg); }
  .assembly-find { display:grid; grid-template-columns:minmax(0,1fr) 170px; gap:12px; }
  .assembly-jump { display:flex; flex-wrap:wrap; align-items:end; gap:8px; }.assembly-jump form { flex:1; min-width:160px; }
  .assembly-traverse { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; font-size:13px; }
  .assembly-navigation .assembly-page-controls { margin:0; padding-top:14px; border-top:1px solid var(--line); }
  .assembly-list { position:relative; max-height:60dvh; overflow-y:auto; overscroll-behavior:contain; padding:4px; }
  @media(max-width:760px) { .assembly-find { grid-template-columns:1fr; }.assembly-traverse>span { flex-basis:100%; order:-1; }.assembly-jump form { flex-basis:100%; } }

  .boundary-tools { display:flex; align-items:center; gap:18px; margin:18px 0; }.boundary-tools label { min-width:170px; }.boundary-tools p { color:var(--muted); font-size:13px; line-height:1.6; margin:0; }
  .boundary-workspace { display:grid; grid-template-columns:210px minmax(0,1fr); gap:18px; align-items:start; }
  .boundary-workspace .assembly-list { max-height:65dvh; padding:0; }
  .assembly-row { display:flex; align-items:start; justify-content:flex-start; gap:10px; width:100%; text-align:left; padding:12px; border:1px solid var(--line); border-radius:6px; background:var(--panel); }
  .assembly-row.assembly-selected { border-color:var(--tts-accent); background:var(--active-bg); }.assembly-row>strong { font-size:13px; color:var(--tts-accent); }.assembly-row>span { min-width:0; font-size:12px; line-height:1.6; font-weight:400; }.assembly-row small { display:block; font-size:11px; color:var(--muted); margin-top:8px; }
  .selected-boundary { min-width:0; }.boundary-order { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }.boundary-order button { font-size:12px; }
  @media(max-width:1150px) { .boundary-workspace { grid-template-columns:1fr; }.boundary-workspace .assembly-list { max-height:190px; }.boundary-tools { flex-wrap:wrap; } }
.speech-boundary{margin:0 20px 18px;}.speech-boundary>summary{cursor:pointer;padding:12px 0;font-size:14px;color:var(--tts-accent);}
</style>
