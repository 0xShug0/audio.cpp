import type { Speaker } from './project';
import { writable } from 'svelte/store';

export const speakerLibraryRevision = writable(0);
type StoredSpeaker = Speaker | { id: string; deleted: true };
import { readLegacySpeechSettings } from '$lib/library';

// TTS owns its speaker library. Audio references stay in IndexedDB with the model snapshot.
function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('audiocpp-tts-speakers', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('speakers', { keyPath: 'id' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function transaction<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>
): Promise<T> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('speakers', mode);
    const request = operation(tx.objectStore('speakers'));
    tx.oncomplete = () => {
      db.close();
      resolve(request.result);
    };
    tx.onabort = tx.onerror = () => {
      db.close();
      reject(tx.error || new Error('Could not save the speaker library.'));
    };
  });
}
export async function listSpeakers(): Promise<Speaker[]> {
  // Import each speech preset once. Later speaker edits keep precedence over
  // the legacy copy. Database and store versions remain unchanged.
  const legacy = await readLegacySpeechSettings();
  const records = await transaction('readwrite', (store) => {
    const request = store.getAll() as IDBRequest<StoredSpeaker[]>;
    request.onsuccess = () => {
      const speakers = request.result;
      const ids = new Set(speakers.map((speaker) => speaker.id));
      for (const preset of legacy) {
        const id = `model-preset:${preset.id}`;
        if (!['tts', 'clon', 'vdes'].includes(preset.model.task) || ids.has(id)) continue;
        const speaker: Speaker = {
          id,
          name: preset.name,
          configuration: structuredClone(preset),
          voice: { kind: 'builtin', id: preset.request.voice || '' }
        };
        store.add(speaker);
        speakers.push(speaker);
        ids.add(id);
      }
    };
    return request;
  });
  return records.filter((record): record is Speaker => !('deleted' in record));
}
export async function saveSpeaker(speaker: Speaker): Promise<IDBValidKey> {
  const key = await transaction('readwrite', (store) => store.put(speaker));
  speakerLibraryRevision.update((value) => value + 1);
  return key;
}
export async function deleteSpeaker(id: string): Promise<void> {
  // A tombstone in the existing store prevents deleted legacy imports from
  // returning on the next read, without changing the version or other projects.
  if (id.startsWith('model-preset:')) await transaction('readwrite', (store) => store.put({ id, deleted: true }));
  else await transaction('readwrite', (store) => store.delete(id));
  speakerLibraryRevision.update((value) => value + 1);
}
