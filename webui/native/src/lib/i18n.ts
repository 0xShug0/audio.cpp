export type TranslationValues = Record<string, string | number>;
export type Translator = (key: string, values?: TranslationValues, fallback?: string) => string;

type LanguageFile = {
  code: string;
  name: string;
  translations: Record<string, string>;
};

export type UiLanguage = {
  code: string;
  name: string;
};

const english: Record<string, string> = {
  'app.nativeStudio': 'Native Studio',
  'nav.studio': 'Studio',
  'nav.models': 'Models',
  'nav.runtime': 'Runtime',
  'nav.primary': 'Primary navigation',
  'nav.workflows': 'Audio workflows',
  'language.label': 'Interface language',
  'workflow.tts': 'Text to speech',
  'workflow.asr': 'ASR / Transcription',
  'workflow.music': 'Music generation',
  'workflow.vc': 'Voice conversion',
  'workflow.sep': 'Source separation',
  'workflow.analysis': 'Audio analysis',
  'workflow.design': 'Voice design',
  'task.tts': 'Text to speech',
  'task.clon': 'Voice cloning',
  'task.asr': 'Transcription',
  'task.gen': 'Music generation',
  'task.midi': 'Audio to MIDI',
  'task.vc': 'Voice conversion',
  'task.svc': 'Singing voice conversion',
  'task.s2s': 'Speech editing',
  'task.sep': 'Source separation',
  'task.vad': 'Voice activity',
  'task.diar': 'Speaker diarization',
  'task.align': 'Forced alignment',
  'task.vdes': 'Voice design',
  'studio.eyebrow': 'LOCAL AUDIO INTELLIGENCE',
  'studio.title': 'Audio studio',
  'studio.subtitle.tts': 'Generate natural speech from text, with voice presets and cloning when supported.',
  'studio.subtitle.asr': 'Transcribe spoken audio into text, with language and timestamp controls when supported.',
  'studio.subtitle.music': 'Create music and sound from a prompt, lyrics, or reference audio when supported.',
  'studio.subtitle.vc': 'Transform a recording into another voice while preserving the spoken or sung performance.',
  'studio.subtitle.sep': 'Split a recording into vocals, instruments, or other available audio stems.',
  'studio.subtitle.analysis': 'Analyze audio for speech activity, speakers, timing, and alignment.',
  'studio.subtitle.design': 'Create or refine a voice from a written description and supported reference controls.',
  'studio.model': 'Model',
  'studio.noModel': 'No model selected',
  'studio.resident': 'Resident',
  'studio.notInstalled': 'Not installed',
  'studio.available': 'Available',
  'studio.chooseInstalled': 'Choose an installed model',
  'studio.notDownloaded': 'not downloaded',
  'studio.pathFound': 'Path found',
  'studio.pathMissing': 'Path missing',
  'studio.pathUnknown': 'Path not inspected',
  'studio.estimatedVram': '{value} GB estimated VRAM',
  'studio.vram': 'VRAM —',
  'studio.load': 'Load model',
  'studio.unload': 'Unload model',
  'studio.working': 'Working…',
  'studio.bundledLoaded': 'Bundled · loaded',
  'request.label': 'REQUEST',
  'request.title': 'Input & controls',
  'request.prompt': 'Prompt',
  'request.alignmentText': 'Alignment text',
  'request.text': 'Text',
  'request.soundPlaceholder': 'Describe the sound or music…',
  'request.textPlaceholder': 'Enter the text…',
  'request.splitLongText': 'Split and merge long text',
  'request.charactersPerChunk': 'Characters per chunk',
  'request.lyrics': 'Lyrics',
  'request.optional': 'optional',
  'request.context': 'Context prompt',
  'request.contextHint': 'optional terminology or names',
  'request.voiceDescription': 'Voice description',
  'request.voiceDescriptionPlaceholder': 'A warm, calm voice with measured pacing…',
  'request.language': 'Language',
  'request.autoLanguage': 'blank = auto',
  'request.seed': 'Seed',
  'request.randomSeed': '-1 = random',
  'request.maxTokens': 'Maximum tokens',
  'request.duration': 'Duration seconds',
  'request.minimaxFrames': '{frames} aligned output frames',
  'request.sourceAudio': 'Source audio',
  'request.stopRecording': 'Stop recording',
  'request.recordingMicrophone': 'Recording microphone',
  'request.recordMicrophone': 'Record microphone',
  'request.liveTitle': 'Live microphone transcription',
  'request.liveDescription': "Processes consecutive four-second requests using the model's streaming mode.",
  'request.stopLive': 'Stop live',
  'request.startLive': 'Start live',
  'voice.quickStart': 'Quick-start voice presets (demo voices)',
  'voice.configured': 'Configured voices',
  'voice.useReference': 'Use a reference audio file below',
  'voice.bundledNote': 'The bundled reference audio and its matching transcript are supplied automatically.',
  'voice.reference': 'Reference voice',
  'voice.required': 'required',
  'voice.optional': 'optional',
  'voice.referenceText': 'Reference text',
  'voice.recording': 'Recording voice reference',
  'voice.transcript': 'Reference transcript',
  'voice.requiredClone': 'required for this voice clone',
  'voice.recommendedClone': 'recommended for cloning',
  'voice.transcriptPlaceholder': 'Type the exact words spoken in the reference audio, or load a matching .txt file above.',
  'voice.saved': 'Saved voices',
  'voice.browserOnly': 'stored only in this browser',
  'voice.chooseSaved': 'Choose a saved voice...',
  'voice.libraryName': 'Library name',
  'voice.namePlaceholder': 'My reference voice',
  'voice.save': 'Save voice',
  'common.delete': 'Delete',
  'common.enabled': 'Enabled',
  'common.disabled': 'Disabled',
  'common.cancel': 'Cancel',
  'common.close': 'Close',
  'common.refresh': 'Refresh',
  'common.browse': 'Browse',
  'common.apply': 'Apply',
  'common.applying': 'Applying…',
  'file.choose': 'Choose file',
  'file.none': 'No file chosen',
  'options.modelParameters': 'Model parameters',
  'options.additional': 'Additional options',
  'param.minimax_h3.num_inference_steps.label': 'Denoising steps',
  'param.minimax_h3.num_inference_steps.info': 'Twelve denoising steps provide a practical quality and performance balance.',
  'param.minimax_h3.num_frames.label': 'Output frames',
  'param.minimax_h3.num_frames.info': 'Calculated from duration at approximately 24 frames per output second. Editing frames also updates duration.',
  'param.minimax_h3.guidance_scale.label': 'Guidance scale',
  'param.minimax_h3.sampler.label': 'Sampler',
  'param.minimax_h3.dit_acceleration.label': 'DiT acceleration',
  'param.minimax_h3.dit_acceleration.info': 'None uses the quality-first full-DiT path. Acceleration modes are experimental and may distort some outputs.',
  'param.minimax_h3.return_video.label': 'Decode video',
  'param.minimax_h3.return_video.info': 'Disabled by default to reduce memory use and return audio only.',
  'model.minimax_h3.hint': 'MiniMax-H3 uses a joint audio/video DiT. GGUF Q4 is the quality-first choice; CUDA-only GGUF Q4 ConvRot uses slightly more VRAM for higher speed. The default full-DiT path prioritizes audio quality and video decoding is disabled.',
  'run.run': 'Run',
  'run.cancel': 'Cancel',
  'run.working': 'Working…',
  'result.label': 'RESULT',
  'result.title': 'Output',
  'result.track': 'track',
  'result.tracks': 'tracks',
  'result.saveWav': 'Save WAV',
  'result.empty': 'Generated audio and structured results appear here.',
  'models.eyebrow': 'MODEL LIBRARY',
  'models.title': 'Local packages',
  'models.subtitle': 'Download and manage model packages without leaving the native interface.',
  'models.folder': 'Models folder',
  'models.folderHint': 'downloads, local detection, and model loading',
  'models.folderPlaceholder': 'models folder beside audiocpp_server',
  'models.default': 'Default',
  'models.useDefault': 'Use default',
  'models.showTypes': 'Show model types',
  'models.model': 'model',
  'models.variants': 'variants',
  'models.update': 'Update',
  'models.reinstall': 'Reinstall',
  'models.upToDate': 'Up to date',
  'models.updateAvailable': 'Update available',
  'models.versionUnknown': 'Version unknown',
  'models.selected': 'Selected',
  'models.downloaded': 'Downloaded',
  'models.queued': 'queued',
  'models.stopping': 'stopping',
  'models.checkingSize': 'checking size…',
  'models.hfAccess': 'HF access required',
  'models.sizeUnavailable': 'size unavailable',
  'models.stopDownload': 'Stop download',
  'models.cleanPartial': 'Clean partial download',
  'models.sharedPackage': 'Uses the shared {name} package shown above.',
  'runtime.eyebrow': 'RUNTIME',
  'runtime.title': 'Session log',
  'runtime.subtitle': 'Browser-side lifecycle and request events.',
  'runtime.status': 'Status',
  'runtime.backend': 'Backend',
  'runtime.registered': 'Registered',
  'runtime.resident': 'Resident',
  'runtime.noEvents': 'No events yet.',
  'status.ready': 'Ready',
  'status.modelReady': '{model} is resident and ready.',
  'status.runningTask': 'Running {task}...',
  'status.completeIn': 'Complete in {seconds}s.',
  'status.packageAvailable': '{model} will use {format}. It is available in Studio.',
  'folder.eyebrow': 'MODELS FOLDER',
  'folder.title': 'Choose a folder',
  'folder.closeLabel': 'Close folder browser',
  'folder.loading': 'Loading...',
  'folder.up': 'Up one level',
  'folder.loadingFolders': 'Loading folders...',
  'folder.empty': 'This folder has no subfolders.',
  'folder.select': 'Select this folder',
  'footer.embedded': 'SvelteKit · embedded in audiocpp_server'
};

const modules = import.meta.glob('../../lang/lang_*.json', {
  eager: true,
  import: 'default'
}) as Record<string, LanguageFile>;

const dictionaries = new Map<string, Record<string, string>>([['en', english]]);
const discovered: UiLanguage[] = [{ code: 'en', name: 'English' }];

for (const [path, file] of Object.entries(modules).sort(([left], [right]) => left.localeCompare(right))) {
  const filenameCode = path.match(/lang_([^/\\.]+)\.json$/)?.[1]?.toLowerCase();
  const code = (file.code || filenameCode || '').trim().toLowerCase();
  if (!code || code === 'en' || !file.name || !file.translations) continue;
  dictionaries.set(code, file.translations);
  discovered.push({ code, name: file.name });
}

export const uiLanguages = discovered;

export function createTranslator(code: string): Translator {
  const selected = dictionaries.get(code) || english;
  return (key, values = {}, fallback = key) => {
    let text = selected[key] || english[key] || fallback;
    for (const [name, value] of Object.entries(values)) {
      text = text.replaceAll(`{${name}}`, String(value));
    }
    return text;
  };
}

export function resolveUiLanguage(preferred: readonly string[]): string {
  for (const requested of preferred) {
    const normalized = requested.toLowerCase();
    if (dictionaries.has(normalized)) return normalized;
    const base = normalized.split('-')[0];
    if (dictionaries.has(base)) return base;
  }
  return 'en';
}
