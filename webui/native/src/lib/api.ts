import type { LoadedModel, ServerHealth } from './types';

async function errorFrom(response: Response): Promise<Error> {
  let message = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    message = body?.error?.message || body?.message || message;
  } catch {
    const text = await response.text();
    if (text) message = text;
  }
  return new Error(message);
}

export async function jsonRequest<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal
): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(path, { ...init, headers, signal });
  if (!response.ok) throw await errorFrom(response);
  return response.json() as Promise<T>;
}

export async function health(): Promise<ServerHealth> {
  return jsonRequest<ServerHealth>('/health');
}

export async function models(): Promise<LoadedModel[]> {
  const response = await jsonRequest<{ data: LoadedModel[] }>('/v1/models');
  return response.data;
}

export async function loadModel(body: Record<string, unknown>): Promise<void> {
  await jsonRequest('/v1/models/load', { method: 'POST', body: JSON.stringify(body) });
}

export async function unloadModel(id: string): Promise<void> {
  await jsonRequest('/v1/models/unload', {
    method: 'POST',
    body: JSON.stringify({ id })
  });
}

export async function pathStatus(path: string): Promise<{ exists: boolean; directory: boolean; file: boolean }> {
  return jsonRequest('/v1/ui/path-status', {
    method: 'POST',
    body: JSON.stringify({ path })
  });
}

export interface ModelInstallJob {
  id: string;
  state: 'idle' | 'queued' | 'running' | 'cancelling' | 'cancelled' | 'cleaned' | 'complete' | 'failed';
  message: string;
  exit_code: number;
  downloaded_bytes: number;
  total_bytes: number;
  progress_percent: number;
  started_at_ms: number;
  finished_at_ms: number;
}

export async function installModelPackage(body: { id: string; overwrite?: boolean }): Promise<ModelInstallJob> {
  return jsonRequest('/v1/ui/models/install', {
    method: 'POST',
    body: JSON.stringify(body)
  });
}

export async function stopModelInstall(id: string): Promise<ModelInstallJob> {
  return jsonRequest('/v1/ui/models/install/stop', {
    method: 'POST',
    body: JSON.stringify({ id })
  });
}

export async function cleanPartialModelInstall(id: string): Promise<{ id: string; cleaned: boolean; message: string }> {
  return jsonRequest('/v1/ui/models/clean-partial', {
    method: 'POST',
    body: JSON.stringify({ id })
  });
}

export async function deleteModelPackage(id: string): Promise<{ id: string; removed: boolean; message: string }> {
  return jsonRequest('/v1/ui/models/delete', {
    method: 'POST',
    body: JSON.stringify({ id })
  });
}

export async function modelInstallJobs(): Promise<ModelInstallJob[]> {
  const response = await jsonRequest<{ data: ModelInstallJob[] }>('/v1/ui/models/install-status');
  return response.data;
}

export interface ModelPackageSize {
  id: string;
  size_bytes: number | null;
  state: 'pending' | 'ok' | 'gated' | 'unknown' | 'error';
  message: string;
  installed: boolean;
  version_state: 'not_installed' | 'unknown' | 'up_to_date' | 'update_available';
  local_revision: string;
  remote_revision: string;
}

export interface ModelPackageSizesResponse {
  state: 'idle' | 'running' | 'complete' | 'failed';
  message: string;
  data: ModelPackageSize[];
}

export async function modelPackageSizes(): Promise<ModelPackageSizesResponse> {
  return jsonRequest<ModelPackageSizesResponse>('/v1/ui/models/package-sizes');
}

export interface ModelsRootResponse {
  models_root: string;
  default_models_root: string;
  is_default: boolean;
}

export async function getModelsRoot(): Promise<ModelsRootResponse> {
  return jsonRequest<ModelsRootResponse>('/v1/ui/models-root');
}

export async function setModelsRoot(path = ''): Promise<ModelsRootResponse> {
  return jsonRequest<ModelsRootResponse>('/v1/ui/models-root', {
    method: 'POST',
    body: JSON.stringify({ path })
  });
}

export interface DirectoryBrowserResponse {
  current: string;
  parent: string;
  roots: string[];
  directories: Array<{ name: string; path: string }>;
}

export async function browseDirectories(path = ''): Promise<DirectoryBrowserResponse> {
  return jsonRequest<DirectoryBrowserResponse>('/v1/ui/browse-directories', {
    method: 'POST',
    body: JSON.stringify({ path })
  });
}

export async function availableVoices(model = ''): Promise<string[]> {
  const query = model ? `?model=${encodeURIComponent(model)}` : '';
  const response = await jsonRequest<{ voices: string[] }>(`/v1/audio/voices${query}`);
  return response.voices;
}

export async function uploadWav(blob: Blob, signal?: AbortSignal): Promise<string> {
  const response = await jsonRequest<{ path: string }>('/v1/ui/upload', {
    method: 'POST',
    headers: {
      'Content-Type': 'audio/wav'
    },
    body: blob
  }, signal);
  return response.path;
}

export async function speech(body: Record<string, unknown>, signal?: AbortSignal) {
  const response = await fetch('/v1/audio/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal
  });
  if (!response.ok) throw await errorFrom(response);
  return {
    blob: await response.blob(),
    wallMs: response.headers.get('X-AudioCPP-Wall-Ms'),
    rtf: response.headers.get('X-AudioCPP-RTF')
  };
}

export async function transcription(body: Record<string, unknown>, signal?: AbortSignal) {
  return jsonRequest<Record<string, unknown>>('/v1/audio/transcriptions', {
    method: 'POST',
    body: JSON.stringify(body)
  }, signal);
}

export async function runTask(body: Record<string, unknown>, signal?: AbortSignal) {
  return jsonRequest<Record<string, unknown>>('/v1/tasks/run', {
    method: 'POST',
    body: JSON.stringify(body)
  }, signal);
}

export function base64AudioUrl(data: string): string {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
}
