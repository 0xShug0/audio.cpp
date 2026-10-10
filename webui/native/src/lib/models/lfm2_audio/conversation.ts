// The LFM2.5-Audio conversation the S2S panel shows. It lives in this
// module, not in the panel, so that it survives the panel being unmounted
// when the user opens another tab or entry and comes back. It is a store:
// a turn that a panel started goes on for a moment after that panel is
// unmounted, and the panel mounted by then has to see how it ends.

import { get, writable, type Readable } from 'svelte/store';

export interface ReplyArtifact {
  id: string;
  kind: string;
  payload: string;
  meta?: Record<string, string>;
}

export type TurnState = 'loading' | 'uploading' | 'waiting' | 'streaming' | 'done' | 'stopped' | 'failed';

export interface Turn {
  key: number;
  state: TurnState;
  seed: number;
  // The question as uploaded, and its server path.
  questionPath: string;
  questionUrl: string;
  replyText: string;
  replyUrl: string;
  // As the result returned it; sent back with each later turn.
  reply: ReplyArtifact | null;
  cut: boolean;
  steps: string;
  leftOut: boolean;
  live: boolean;
  firstAudioMs: number | null;
  message: string;
}

export interface Conversation {
  // The entry the turns are with.
  modelId: string;
  turns: Turn[];
}

const store = writable<Conversation>({ modelId: '', turns: [] });
let nextKey = 1;

export const conversation: Readable<Conversation> = { subscribe: store.subscribe };

function releaseTurn(turn: Turn) {
  if (turn.questionUrl) URL.revokeObjectURL(turn.questionUrl);
  if (turn.replyUrl) URL.revokeObjectURL(turn.replyUrl);
  turn.questionUrl = '';
  turn.replyUrl = '';
}

export function clearConversation() {
  for (const turn of get(store).turns) releaseTurn(turn);
  store.set({ modelId: '', turns: [] });
}

// The turns the next question goes with: those that finished with a reply
// artifact and that the user has not left out, in order.
export function keptTurns(turns: Turn[]) {
  return turns.filter((turn) => turn.state === 'done' && turn.reply && !turn.leftOut);
}

// Adds the turn of a Run with this entry, and returns it with the turns its
// question goes with. A failed or stopped turn is not part of the
// conversation: the new turn takes its place. Turns with another entry do
// not go with the question, and they stay until the new turn is done, so
// that a Run that fails does not lose them.
export function addTurn(modelId: string, state: TurnState, seed: number) {
  const current = get(store);
  const turns = current.turns.filter((turn) => turn.state !== 'failed' && turn.state !== 'stopped');
  for (const turn of current.turns) if (!turns.includes(turn)) releaseTurn(turn);
  const other = turns.length > 0 && current.modelId !== modelId;
  const turn: Turn = {
    key: nextKey++, state, seed, questionPath: '', questionUrl: '', replyText: '', replyUrl: '', reply: null,
    cut: false, steps: '', leftOut: false, live: true, firstAudioMs: null, message: ''
  };
  store.set({ modelId: other ? current.modelId : modelId, turns: [...turns, turn] });
  return { turn, history: other ? [] : keptTurns(turns) };
}

// Changes a turn and tells the panel.
export function updateTurn(turn: Turn, changes: Partial<Turn>) {
  Object.assign(turn, changes);
  store.update((current) => current);
}

// The turn of a Run with this entry is done. When the turns before it are
// with another entry, it starts a new conversation and they go; returns
// whether it did.
export function completeTurn(modelId: string, turn: Turn, changes: Partial<Turn>) {
  Object.assign(turn, changes, { state: 'done' });
  const current = get(store);
  const fresh = current.modelId !== modelId;
  if (fresh) for (const other of current.turns) if (other !== turn) releaseTurn(other);
  store.set(fresh ? { modelId, turns: [turn] } : current);
  return fresh;
}
