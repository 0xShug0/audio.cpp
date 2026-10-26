import type { ModelSettings } from './model-settings/types';
export type { ModelConfiguration, ModelSettings } from './model-settings/types';

export interface SavedVoice {
  id: string;
  name: string;
  transcript: string;
  audio: Blob;
  createdAt: number;
}

const voiceDatabaseName = 'audiocpp-native-studio';
const appDatabaseName = 'audiocpp-native-apps';
const voiceStoreName = 'voices';
const legacyPresetStoreName = 'model-presets';

function openDatabase(storeName: string): Promise<IDBDatabase> {
  const isVoiceStore = storeName === voiceStoreName;
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(isVoiceStore ? voiceDatabaseName : appDatabaseName, 1);
    request.onupgradeneeded = () => {
      for (const name of isVoiceStore ? [voiceStoreName] : [legacyPresetStoreName, 'app-projects']) {
        request.result.createObjectStore(name, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Could not open the library.'));
  });
}

function transaction<T>(
  storeName: string,
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>
): Promise<T> {
  return openDatabase(storeName).then((database) => new Promise<T>((resolve, reject) => {
    const tx = database.transaction(storeName, mode);
    const request = operation(tx.objectStore(storeName));
    request.onerror = () => reject(request.error || new Error('Library operation failed.'));
    tx.oncomplete = () => { database.close(); resolve(request.result); };
    tx.onabort = () => { database.close(); reject(tx.error || new Error('Library transaction was aborted.')); };
    tx.onerror = () => {
      database.close();
      reject(tx.error || new Error('Library transaction failed.'));
    };
  }));
}

export async function listVoices(): Promise<SavedVoice[]> {
  const voices = await transaction<SavedVoice[]>(voiceStoreName, 'readonly', (store) => store.getAll());
  return voices.sort((left, right) => right.createdAt - left.createdAt);
}

export function saveVoice(voice: SavedVoice): Promise<IDBValidKey> {
  return transaction<IDBValidKey>(voiceStoreName, 'readwrite', (store) => store.put(voice));
}

export function deleteVoice(id: string): Promise<undefined> {
  return transaction<undefined>(voiceStoreName, 'readwrite', (store) => store.delete(id));
}

// Read-only compatibility for importing old speech settings into app-owned speakers.
// The existing version 1 store stays in place; new model presets are never written.
export async function readLegacySpeechSettings(): Promise<ModelSettings[]> {
  const items = await transaction<ModelSettings[]>(legacyPresetStoreName, 'readonly', (store) => store.getAll());
  return items.filter((item) => ['tts', 'clon', 'vdes'].includes(item.model.task));
}

// App data is opaque to the library. Each app owns its document schema.
export interface AppProject { id: string; appId: string; title: string; updatedAt: number; data: unknown }
export async function listAppProjects(appId: string): Promise<AppProject[]> {
  const records = await transaction<AppProject[]>('app-projects', 'readonly', store => store.getAll());
  return records.filter(record => record.appId === appId).sort((a, b) => b.updatedAt - a.updatedAt);
}
export function saveAppProject(project: AppProject): Promise<IDBValidKey> {
  return transaction('app-projects', 'readwrite', store => store.put(project));
}
export function deleteAppProject(id: string): Promise<undefined> {
  return transaction('app-projects', 'readwrite', store => store.delete(id));
}

export function getAppProject(id: string): Promise<AppProject | undefined> {
  return transaction('app-projects', 'readonly', store => store.get(id));
}
