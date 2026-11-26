import { writable } from 'svelte/store';

export interface AutoSaveFile {
  filename: string;
  blob: Blob;
}

export interface AutoSaveResult {
  method: 'directory' | 'download';
  folderName?: string;
  savedFiles: string[];
  warning?: string;
}

export interface AutoSaveDirectoryState {
  supported: boolean;
  folderName: string;
  permission: 'none' | 'granted' | 'prompt' | 'denied';
}

type PermissionStateLike = 'granted' | 'denied' | 'prompt';

interface DirectoryHandleLike {
  name: string;
  queryPermission?: (descriptor: { mode: 'readwrite' }) => Promise<PermissionStateLike>;
  requestPermission?: (descriptor: { mode: 'readwrite' }) => Promise<PermissionStateLike>;
  getFileHandle: (
    name: string,
    options?: { create?: boolean }
  ) => Promise<{
    createWritable: () => Promise<{
      write: (data: Blob) => Promise<void>;
      close: () => Promise<void>;
      abort?: () => Promise<void>;
    }>;
  }>;
}

type PickerWindow = Window & {
  showDirectoryPicker?: (options?: { id?: string; mode?: 'readwrite' }) => Promise<DirectoryHandleLike>;
};

const DB_NAME = 'audiocpp-ui-autosave';
const STORE_NAME = 'handles';
const HANDLE_KEY = 'output-directory';

let activeDirectoryHandle: DirectoryHandleLike | null = null;

export function supportsDirectoryPicker(): boolean {
  return typeof window !== 'undefined' && typeof (window as PickerWindow).showDirectoryPicker === 'function';
}

export const autoSaveDirectory = writable<AutoSaveDirectoryState>({
  supported: false,
  folderName: '',
  permission: 'none'
});

function setDirectoryState(handle: DirectoryHandleLike | null, permission: AutoSaveDirectoryState['permission']) {
  activeDirectoryHandle = handle;
  autoSaveDirectory.set({
    supported: supportsDirectoryPicker(),
    folderName: handle?.name || '',
    permission: handle ? permission : 'none'
  });
}

function openHandleDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);
  return new Promise((resolve) => {
    try {
      const request = indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function readStoredHandle(): Promise<DirectoryHandleLike | null> {
  const db = await openHandleDb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const request = tx.objectStore(STORE_NAME).get(HANDLE_KEY);
      request.onsuccess = () => {
        const handle = request.result as DirectoryHandleLike | undefined;
        resolve(handle && typeof handle.getFileHandle === 'function' ? handle : null);
      };
      request.onerror = () => resolve(null);
    } catch {
      resolve(null);
    } finally {
      db.close();
    }
  });
}

async function writeStoredHandle(handle: DirectoryHandleLike | null): Promise<void> {
  const db = await openHandleDb();
  if (!db) return;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      if (handle) store.put(handle, HANDLE_KEY);
      else store.delete(HANDLE_KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
      tx.onabort = () => resolve();
    } catch {
      resolve();
    } finally {
      db.close();
    }
  });
}

async function queryHandlePermission(handle: DirectoryHandleLike): Promise<PermissionStateLike> {
  if (typeof handle.queryPermission !== 'function') return 'granted';
  try {
    return await handle.queryPermission({ mode: 'readwrite' });
  } catch {
    return 'prompt';
  }
}

export async function loadAutoSaveDirectory(): Promise<void> {
  if (!supportsDirectoryPicker()) {
    setDirectoryState(null, 'none');
    return;
  }
  const stored = await readStoredHandle();
  if (!stored) {
    setDirectoryState(null, 'none');
    return;
  }
  const permission = await queryHandlePermission(stored);
  setDirectoryState(stored, permission);
}

export async function pickAutoSaveDirectory(): Promise<AutoSaveDirectoryState | null> {
  if (!supportsDirectoryPicker()) {
    throw new Error('This browser does not support choosing a local save folder.');
  }
  try {
    const handle = await (window as PickerWindow).showDirectoryPicker!({
      id: 'audiocpp-outputs',
      mode: 'readwrite'
    });
    const permission = await queryHandlePermission(handle);
    await writeStoredHandle(handle);
    setDirectoryState(handle, permission);
    return {
      supported: true,
      folderName: handle.name || '',
      permission
    };
  } catch (error) {
    if ((error as Error)?.name === 'AbortError') return null;
    throw error;
  }
}

export async function clearAutoSaveDirectory(): Promise<void> {
  await writeStoredHandle(null);
  setDirectoryState(null, 'none');
}

export async function requestAutoSaveDirectoryPermission(): Promise<boolean> {
  if (!activeDirectoryHandle) return false;
  const current = await queryHandlePermission(activeDirectoryHandle);
  if (current === 'granted') {
    setDirectoryState(activeDirectoryHandle, 'granted');
    return true;
  }
  if (typeof activeDirectoryHandle.requestPermission !== 'function') {
    setDirectoryState(activeDirectoryHandle, 'granted');
    return true;
  }
  try {
    const next = await activeDirectoryHandle.requestPermission({ mode: 'readwrite' });
    setDirectoryState(activeDirectoryHandle, next);
    return next === 'granted';
  } catch {
    return false;
  }
}

// Call inside a user gesture (e.g. Run click) so a persisted folder handle from
// an earlier browser session can re-acquire write permission before a long run.
export async function ensureAutoSavePermissionOnUserGesture(): Promise<void> {
  if (!activeDirectoryHandle) return;
  const current = await queryHandlePermission(activeDirectoryHandle);
  if (current === 'prompt') {
    await requestAutoSaveDirectoryPermission();
  } else {
    setDirectoryState(activeDirectoryHandle, current);
  }
}

export function dataUrlToBlob(url: string, fallbackMime = 'application/octet-stream'): Blob | null {
  if (!url.startsWith('data:')) return null;
  const comma = url.indexOf(',');
  if (comma < 0) return null;
  const header = url.slice(5, comma);
  const body = url.slice(comma + 1);
  const mime = header.split(';')[0] || fallbackMime;
  if (header.includes(';base64')) {
    const binary = atob(body);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return new Blob([bytes], { type: mime });
  }
  return new Blob([decodeURIComponent(body)], { type: mime });
}

export function sanitizeFilenameSegment(value: string, fallback = 'output'): string {
  const cleaned = value
    .replace(/[\x00-\x1f<>:"/\\|?*]+/g, '-')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^[.\-]+|[.\-]+$/g, '');
  return cleaned || fallback;
}

export function formatAutoSaveTimestamp(date = new Date()): string {
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

export function sanitizeFilename(filename: string): string {
  const lastDot = filename.lastIndexOf('.');
  if (lastDot > 0 && lastDot < filename.length - 1) {
    const base = sanitizeFilenameSegment(filename.slice(0, lastDot), 'output');
    const ext = sanitizeFilenameSegment(filename.slice(lastDot + 1), 'bin');
    return `${base}.${ext}`;
  }
  return sanitizeFilenameSegment(filename, 'output');
}

function splitExtension(filename: string): [string, string] {
  if (filename.endsWith('.settings.json')) {
    return [filename.slice(0, -'.settings.json'.length), '.settings.json'];
  }
  const dot = filename.lastIndexOf('.');
  if (dot > 0 && dot < filename.length - 1) {
    return [filename.slice(0, dot), filename.slice(dot)];
  }
  return [filename, ''];
}

async function resolveUniqueDirectoryFilename(
  handle: DirectoryHandleLike,
  preferred: string
): Promise<string> {
  const safe = sanitizeFilename(preferred);
  const [stem, ext] = splitExtension(safe);
  for (let attempt = 1; attempt <= 999; attempt += 1) {
    const candidate = attempt === 1 ? safe : `${stem}-${attempt}${ext}`;
    try {
      await handle.getFileHandle(candidate, { create: false });
      // Exists already: keep searching for a unique name.
    } catch (error) {
      if ((error as Error)?.name === 'NotFoundError') {
        return candidate;
      }
      // If the implementation doesn't distinguish NotFoundError, use candidate.
      return candidate;
    }
  }
  return `${stem}-${Date.now()}${ext}`;
}

async function writeFilesToDirectory(
  handle: DirectoryHandleLike,
  files: AutoSaveFile[],
  written: string[]
): Promise<void> {
  for (const file of files) {
    const filename = await resolveUniqueDirectoryFilename(handle, file.filename);
    const fileHandle = await handle.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    try {
      await writable.write(file.blob);
      await writable.close();
    } catch (error) {
      await writable.abort?.().catch(() => {});
      throw error;
    }
    written.push(filename);
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function triggerBrowserDownloads(files: AutoSaveFile[]): Promise<string[]> {
  const saved: string[] = [];
  for (let i = 0; i < files.length; i += 1) {
    if (i > 0) await delay(120);
    const filename = sanitizeFilename(files[i].filename);
    const url = URL.createObjectURL(files[i].blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.rel = 'noopener';
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    link.remove();
    // Delay revocation so Safari/Firefox finish initiating the download.
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    saved.push(filename);
  }
  return saved;
}

export async function autoSaveFiles(files: AutoSaveFile[]): Promise<AutoSaveResult> {
  if (!files.length) {
    return { method: activeDirectoryHandle ? 'directory' : 'download', savedFiles: [] };
  }

  let warning: string | undefined;
  let remainingFiles = files;
  const writtenToDirectory: string[] = [];
  if (activeDirectoryHandle) {
    const permission = await queryHandlePermission(activeDirectoryHandle);
    setDirectoryState(activeDirectoryHandle, permission);
    if (permission === 'granted') {
      try {
        await writeFilesToDirectory(activeDirectoryHandle, files, writtenToDirectory);
        return {
          method: 'directory',
          folderName: activeDirectoryHandle.name || undefined,
          savedFiles: writtenToDirectory
        };
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        remainingFiles = files.slice(writtenToDirectory.length);
        warning = `Could not write to folder "${activeDirectoryHandle.name}" (${reason}); fell back to browser downloads.`;
      }
    } else {
      warning = `Folder "${activeDirectoryHandle.name}" needs write permission; fell back to browser downloads.`;
    }
  }

  const downloadedFiles = await triggerBrowserDownloads(remainingFiles);
  return {
    method: 'download',
    savedFiles: [...writtenToDirectory, ...downloadedFiles],
    warning
  };
}
