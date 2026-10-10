// A studio panel registered with requestMode 'panel' runs the page's request
// itself: the page's Run, Ctrl+Enter and Cancel call the function the panel
// registered. These types live apart from panels.ts, which imports the panel
// components, so that a panel can import them without an import cycle.

export interface PanelRunContext {
  modelId: string;
  // The page's source audio at Run time.
  question: File;
  // Already resolved: -1 has become a random seed.
  seed: number;
  // The page's Max tokens, when the entry takes it.
  maxTokens?: number;
  // The page's Language field, trimmed; empty means not sent.
  language: string;
  // The page's request options: catalog defaults, model parameters and the
  // Additional JSON box.
  options: Record<string, unknown>;
  // Aborted by Cancel.
  signal: AbortSignal;
  // Sets the run bar's status line.
  setStatus: (text: string) => void;
  // With UI management, loads the selected entry in this mode. It reloads
  // an entry that is resident in another mode, with another package or with
  // imported settings that differ.
  ensureLoadedMode: (mode: string) => Promise<void>;
}

export interface PanelRunOutcome {
  // Shown in the Result column.
  audio: Blob | null;
  text: string;
  json: string;
}

export type PanelRunner = (context: PanelRunContext) => Promise<PanelRunOutcome>;

// Registers a panel's runner and returns the function that removes it.
export type RegisterPanelRunner = (runner: PanelRunner) => () => void;
