// The LFM2.5-Audio conversation the S2S panel shows. It lives in this
// module, not in the panel, so that it survives the panel being unmounted
// when the user opens another tab or entry and comes back.

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

export const conversation = {
  modelId: '',
  turns: [] as Turn[],
  nextKey: 1
};

export function releaseTurn(turn: Turn) {
  if (turn.questionUrl) URL.revokeObjectURL(turn.questionUrl);
  if (turn.replyUrl) URL.revokeObjectURL(turn.replyUrl);
  turn.questionUrl = '';
  turn.replyUrl = '';
}

export function clearConversation() {
  for (const turn of conversation.turns) releaseTurn(turn);
  conversation.turns = [];
  conversation.modelId = '';
}

// The turns the next question goes with: those that finished with a reply
// artifact and that the user has not left out, in order.
export function keptTurns() {
  return conversation.turns.filter((turn) => turn.state === 'done' && turn.reply && !turn.leftOut);
}
