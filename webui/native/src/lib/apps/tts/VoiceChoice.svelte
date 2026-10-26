<script lang="ts">
  import { onDestroy } from 'svelte';
  import { catalog, modelCapabilities } from '$lib/catalog';
  import { browserDecodeToWav } from '$lib/audio';
  import type { SavedVoice } from '$lib/library';
  import type { ModelSettings } from '$lib/model-settings/types';
  import type { VoiceAssignment, ReferenceVoice } from './project';
  import ModelChoice from './ModelChoice.svelte';
  export let label: string;
  export let configuration: ModelSettings | null;
  export let assignment: VoiceAssignment | undefined;
  export let inheritedReference: ReferenceVoice | null = null;
  export let allowInherit = false;
  export let voices: SavedVoice[];
  export let choices: ModelSettings[];
  export let transcription: ModelSettings | null;
  export let disabled = false;
  export let onchange: (voice: VoiceAssignment | undefined) => void;
  export let onbusy: (busy: boolean) => void;
  export let ontranscribe: () => void;
  export let onconfiguration: () => void;
  let error = '';
  let audioUrl = '';
  let lastAudio: Blob | undefined;
  $: if(reference?.audio !== lastAudio) { if(audioUrl)URL.revokeObjectURL(audioUrl);lastAudio=reference?.audio;audioUrl=lastAudio?URL.createObjectURL(lastAudio):''; }
  onDestroy(()=>{if(audioUrl)URL.revokeObjectURL(audioUrl);});
  $: entry = configuration && (catalog.find(e=>e.id===configuration.model.id) || catalog.find(e=>e.family===configuration.model.family && e.task===configuration.model.task));
  $: builtin = entry?.builtin_voices || [];
  $: supportsReference = !!configuration && (configuration.model.task==='clon' || modelCapabilities[configuration.model.family]?.includes('speaker_reference'));
  $: reference = assignment?.kind==='reference' ? assignment.data : assignment?.kind==='builtin' ? null : inheritedReference;
  $: selected = assignment?.kind==='reference' ? 'reference' : assignment?.kind==='builtin' ? `builtin:${assignment.id}` : allowInherit ? 'inherit' : `builtin:${configuration?.request.voice || ''}`;
  function choose(id: string) {
    error='';
    if(id==='inherit') { onchange(undefined);return; }
    if(id.startsWith('builtin:')) { onchange({kind:'builtin',id:id.slice(8)});return; }
    const voice=voices.find(v=>v.id===id);
    if(voice) onchange({kind:'reference',data:{id:crypto.randomUUID(),name:voice.name,audio:voice.audio,transcript:voice.transcript}});
  }
  async function upload(file: File | undefined) {
    if(!file)return;
    onbusy(true);error='';
    try { const audio=await browserDecodeToWav(file);onchange({kind:'reference',data:{id:crypto.randomUUID(),name:file.name,audio,transcript:''}}); }
    catch(e){error=String(e);}finally{onbusy(false);}
  }
</script>
<div class="voice-choice">
  <label>{label}
    <select aria-label={label} value={selected} disabled={disabled || !configuration} on:change={e=>choose(e.currentTarget.value)}>
      {#if allowInherit}<option value="inherit">Use default · {inheritedReference?.name || configuration?.request.voice || 'model default'}</option>{/if}
      {#if reference}<option value="reference">Recording: {reference.name}</option>{/if}
      <option value="builtin:">Use model's default voice</option>
      {#if configuration?.request.voice && !builtin.includes(configuration.request.voice)}<option value={`builtin:${configuration.request.voice}`}>{configuration.request.voice} · saved setting</option>{/if}
      {#if builtin.length}<optgroup label="Built-in voices">{#each builtin as id}<option value={`builtin:${id}`}>{id.includes('_') ? id.slice(id.indexOf('_')+1).replace(/_/g,' ') : id} · {id}</option>{/each}</optgroup>{/if}
      {#if supportsReference && voices.length}<optgroup label="Recorded voices from Lib">{#each voices as voice}<option value={voice.id}>{voice.name}</option>{/each}</optgroup>{/if}
      {#if assignment?.kind==='builtin' && assignment.id && !builtin.includes(assignment.id) && configuration?.request.voice!==assignment.id}<option value={`builtin:${assignment.id}`}>{assignment.id} · saved in project</option>{/if}
    </select>
  </label>
  {#if !configuration}<p>Choose a speech model first.</p>{:else if !reference}<p>{assignment?.kind==='builtin' ? `Voice: ${assignment.id || 'model default'}` : `Current voice: ${configuration.request.voice || 'model default'}`}</p>{/if}
  {#if supportsReference}<label class="upload">Upload a different voice recording<input type="file" accept="audio/*" disabled={disabled} on:change={e=>upload(e.currentTarget.files?.[0])}/></label>{/if}
  {#if reference}
    {#if !supportsReference}<p role="alert">This model does not support recorded voices. Choose a cloning model or a built-in voice.</p>{/if}
    <audio controls src={audioUrl} on:play={e=>document.querySelectorAll<HTMLAudioElement>('.tts-app audio').forEach(a=>{if(a!==e.currentTarget)a.pause();})}></audio>
    <label>Words spoken in this recording<textarea aria-label={`${label} transcript`} rows="2" value={reference.transcript} disabled={disabled} on:input={e=>onchange({kind:'reference',data:{...reference!,transcript:e.currentTarget.value}})}></textarea></label>
    <details><summary>Fill transcript using a recognition model</summary><ModelChoice label={`${label} recognition model`} {choices} tasks={['asr']} bind:value={transcription} {disabled} onchange={onconfiguration}/><button type="button" disabled={disabled || !transcription} on:click={ontranscribe}>Transcribe this recording</button></details>
  {/if}
  {#if error}<p role="alert">{error}</p>{/if}
</div>
<style>
  .voice-choice{min-width:0;}label{display:grid;gap:8px;font-size:13px;color:var(--muted);}select,textarea{width:100%;min-width:0;}p{font-size:12px;color:var(--muted);margin:8px 0 0;line-height:1.5;}.upload{margin-top:10px;font-size:12px;}.upload input{width:100%;font-size:12px;margin-top:4px;}audio{width:100%;height:34px;margin:12px 0;}details{margin-top:12px;font-size:12px;}summary{cursor:pointer;padding:8px 0;}details>button{margin-top:12px;min-height:36px;}[role=alert]{color:var(--danger);}
</style>
