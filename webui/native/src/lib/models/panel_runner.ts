// A studio panel registered with requestMode 'panel' runs the page's request
// itself: the page's Run, Ctrl+Enter and Cancel call the function the panel
// registered. The panel checks the inputs it needs, such as the source audio
// the page passes it as a prop. These types live apart from panels.ts, which
// imports the panel components, so that a panel can import them without an
// import cycle.

export interface PanelRunContext {
  modelId: string;
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
  // Makes an error for something the user has to do before the run, such as
  // choosing a file. The runner throws it, and the page shows it as a
  // warning.
  warning: (text: string) => Error;
  // Makes the entry ready for the request. With UI management it loads the
  // entry in this mode: it reloads an entry that is resident in another
  // mode, with another package or with imported settings that differ, and
  // fails when the user selects another entry meanwhile. On a server with a
  // config file the entry keeps its mode, and this fails when the server
  // does not list the entry or imported settings ask for a reload.
  ensureLoaded: (mode: string) => Promise<void>;
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
