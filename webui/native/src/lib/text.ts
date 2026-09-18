const speakerLine = /^\s*(Speaker\s+\d+\s*:)\s*(.*)$/i;

// Sentence terminators. '.' is handled separately by endsSentence, because it is
// the only one that is ambiguous: it also marks decimals, abbreviations,
// initials and file extensions.
const terminators = new Set(['。', '！', '？', '!', '?', '；', ';', '…', '.']);

// Abbreviations that end in a full stop without ending a sentence. Not
// exhaustive -- it cannot be -- but it covers what prose actually contains, and
// a miss costs a split in a slightly wrong place, not a failure.
const abbreviations = new Set([
  'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'mt', 'rev', 'hon',
  'vs', 'etc', 'eg', 'ie', 'approx', 'dept', 'est', 'fig', 'no', 'vol',
  'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct',
  'nov', 'dec', 'inc', 'ltd', 'co', 'corp',
]);

function isDigit(character: string): boolean {
  return character >= '0' && character <= '9';
}

function isWordCharacter(character: string): boolean {
  return /[\p{L}\p{N}']/u.test(character);
}

/** Whether a full stop ends a sentence rather than a number or an abbreviation. */
function endsSentence(text: string, index: number): boolean {
  // Must be followed by whitespace or end of text, which keeps file.txt and
  // example.com intact.
  const next = text[index + 1];
  if (next !== undefined && !/\s/.test(next)) return false;

  // Not a decimal point.
  if (index > 0 && isDigit(text[index - 1]) && next !== undefined && isDigit(next)) return false;

  let start = index;
  while (start > 0 && isWordCharacter(text[start - 1])) start -= 1;
  const word = text.slice(start, index);

  // A single letter is an initial: "J. R. R. Tolkien".
  if (word.length === 1 && /\p{L}/u.test(word)) return false;

  return !abbreviations.has(word.toLowerCase());
}

/**
 * Break text after each sentence terminator, keeping the terminator and any
 * following whitespace attached to the sentence it ends.
 */
function splitSentences(text: string): string[] {
  const sentences: string[] = [];
  let start = 0;

  for (let index = 0; index < text.length; index += 1) {
    if (!terminators.has(text[index])) continue;
    if (text[index] === '.' && !endsSentence(text, index)) continue;

    let end = index + 1;
    while (end < text.length && terminators.has(text[end])) end += 1;
    while (end < text.length && /\s/.test(text[end])) end += 1;

    sentences.push(text.slice(start, end));
    start = end;
    index = end - 1;
  }

  if (start < text.length) sentences.push(text.slice(start));
  return sentences;
}

/**
 * Cut text into budget-sized pieces at whitespace, falling back to a hard cut
 * only for a single token that is itself longer than the budget.
 */
function breakOnWords(text: string, budget: number): string[] {
  const pieces: string[] = [];
  let rest = text.trim();

  while (rest.length > budget) {
    let cut = rest.lastIndexOf(' ', budget);
    if (cut <= 0) cut = budget;

    const piece = rest.slice(0, cut).trim();
    if (piece) pieces.push(piece);
    rest = rest.slice(cut).trim();
  }

  if (rest) pieces.push(rest);
  return pieces;
}

function splitLongLine(line: string, budget: number): string[] {
  const match = speakerLine.exec(line);
  const prefix = match ? `${match[1]} ` : '';
  const body = match ? match[2] : line.trim();

  // The prefix is repeated onto every chunk, so it has to come out of the
  // budget or chunks carrying one exceed the limit the caller asked for.
  const room = Math.max(1, budget - prefix.length);

  const sentences = splitSentences(body);
  if (!sentences.length) sentences.push(body);

  const chunks: string[] = [];
  let current = '';
  // Whatever whitespace actually followed the last sentence added to `current`.
  // splitSentences keeps it, and it has to be carried rather than replaced with
  // a space: sentences in Chinese, Japanese and Korean are adjacent, separated
  // by a full-width terminator and nothing else. Inserting a space there both
  // changes the text the model is asked to speak and spends a character of the
  // budget, so three 20-character sentences stop fitting in two 40-character
  // chunks and become three requests instead of two.
  let separator = '';

  for (const raw of sentences) {
    const sentence = raw.trimEnd();
    if (!sentence) continue;
    const trailing = raw.slice(sentence.length);

    if (current && current.length + separator.length + sentence.length > room) {
      chunks.push(prefix + current.trim());
      current = '';
      separator = '';
    }
    if (sentence.length <= room) {
      current = current ? `${current}${separator}${sentence}` : sentence;
      separator = trailing;
      continue;
    }
    if (current) {
      chunks.push(prefix + current.trim());
      current = '';
      separator = '';
    }
    // No sentence boundary fits, so fall back to word boundaries. Only a token
    // longer than the whole budget is ever cut mid-word.
    for (const piece of breakOnWords(sentence, room)) chunks.push(prefix + piece);
  }
  if (current) chunks.push(prefix + current.trim());

  return chunks.length ? chunks : [line];
}

export function splitTtsChunks(text: string, budget: number): string[] {
  const units: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    units.push(...(line.length > budget ? splitLongLine(line, budget) : [line]));
  }

  const chunks: string[] = [];
  let current: string[] = [];
  let currentLength = 0;
  for (const unit of units) {
    const separator = current.length ? 1 : 0;
    if (current.length && currentLength + separator + unit.length > budget) {
      chunks.push(current.join('\n'));
      current = [];
      currentLength = 0;
    }
    current.push(unit);
    currentLength += (current.length > 1 ? 1 : 0) + unit.length;
  }
  if (current.length) chunks.push(current.join('\n'));
  return chunks.length ? chunks : text.trim() ? [text] : [];
}

export function defaultChunkBudget(family: string): number {
  if (family === 'vibevoice') return 600;
  if (family === 'voxcpm2') return 60;
  return 1000;
}
