import { jsonRequest, taskStreamEvents } from '$lib/api';
import { ModelRuntime } from '$lib/model-settings/runtime';
import type { ModelSettings } from '$lib/model-settings/types';
export interface Artifact { id: string; kind: string; payload?: string; path?: string; meta?: Record<string, string> }

// A conversation reuses prepared sessions. A configuration change replaces only its role.
export class ChatWorkflow {
  private runtimes = new Map<string, { key: string; runtime: ModelRuntime }>();
  clear() { this.runtimes.clear(); }
  async prepare(settings: ModelSettings, signal: AbortSignal) {
    const key = JSON.stringify(settings);
    let saved = this.runtimes.get(settings.model.id);
    if (saved?.key !== key) {
      saved = { key, runtime: new ModelRuntime(settings, signal) };
      this.runtimes.set(settings.model.id, saved);
    }
    await saved.runtime.prepare();
    signal.throwIfAborted();
  }
  async execute(settings: ModelSettings, request: Record<string, unknown>, signal: AbortSignal,
    event: (value: Record<string, unknown>) => void | Promise<void> = () => {}) {
    try {
      await this.prepare(settings, signal);
      const body = { model: settings.model.id, request, include_artifacts: true };
      if (settings.model.mode === 'offline') return await jsonRequest<Record<string, unknown>>('/v1/tasks/run', {
        method: 'POST', body: JSON.stringify(body)
      }, signal);
      let result: Record<string, unknown> | null = null;
      for await (const message of taskStreamEvents(body, signal)) {
        signal.throwIfAborted();
        if (message.type === 'task.stream.event') await event(message.event);
        else result = message.result;
      }
      if (!result) throw new Error('The model stream ended without a result.');
      return result;
    } catch (error) { this.runtimes.delete(settings.model.id); throw error; }
  }
  async speak(conversation: ModelSettings, synthesis: ModelSettings, avatar: ModelSettings, request: Record<string, unknown>,
    signal: AbortSignal, event: (value: Record<string, unknown>) => void | Promise<void>) {
    if ([conversation, synthesis, avatar].some(settings => settings.model.mode !== 'streaming'))
      throw new Error('Real-time avatar playback requires streaming conversation, TTS and avatar model entries.');
    await this.prepare(conversation, signal);
    await this.prepare(synthesis, signal);
    await this.prepare(avatar, signal);
    let result: Record<string, unknown> | null = null;
    for await (const message of taskStreamEvents({ recipe: 'voice-chat-avatar', stream: true,
      include_artifacts: true, models: { conversation: conversation.model.id, tts: synthesis.model.id, avatar: avatar.model.id }, request }, signal, '/v1/recipes/run')) {
      signal.throwIfAborted();
      if (message.type === 'task.stream.event') await event(message.event);
      else result = message.result;
    }
    if (!result) throw new Error('The speaking-avatar stream ended without a result.');
    return result;
  }

}

export function appModel(choice: ModelSettings, role: string): ModelSettings {
  const settings = structuredClone(choice);
  // Server-configured entries retain their declared mode and identity.
  if (!choice.id.endsWith(':configured')) {
    settings.model.id = `voice-chat-${role}`;
    if (['lfm3_audio', 'soulx_flashhead'].includes(settings.model.family)) settings.model.mode = 'streaming';
  }
  return settings;
}
