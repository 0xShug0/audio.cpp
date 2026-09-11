export interface SavedVoice {
  id: string;
  name: string;
  transcript: string;
  audio: Blob;
  createdAt: number;
}

const databaseName = 'audiocpp-native-studio';
const storeName = 'voices';

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(databaseName, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(storeName)) {
        request.result.createObjectStore(storeName, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Could not open the voice library.'));
  });
}

function transaction<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>
): Promise<T> {
  return openDatabase().then((database) => new Promise<T>((resolve, reject) => {
    const tx = database.transaction(storeName, mode);
    const request = operation(tx.objectStore(storeName));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Voice library operation failed.'));
    tx.oncomplete = () => database.close();
    tx.onerror = () => {
      database.close();
      reject(tx.error || new Error('Voice library transaction failed.'));
    };
  }));
}

export async function listVoices(): Promise<SavedVoice[]> {
  const voices = await transaction<SavedVoice[]>('readonly', (store) => store.getAll());
  return voices.sort((left, right) => right.createdAt - left.createdAt);
}

export function saveVoice(voice: SavedVoice): Promise<IDBValidKey> {
  return transaction<IDBValidKey>('readwrite', (store) => store.put(voice));
}

export function deleteVoice(id: string): Promise<undefined> {
  return transaction<undefined>('readwrite', (store) => store.delete(id));
}
