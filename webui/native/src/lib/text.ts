const speakerLine = /^\s*(Speaker\s+\d+\s*:)\s*(.*)$/i;
const sentenceParts = /[^。！？!?；;…]*[。！？!?；;…]+|[^。！？!?；;…]+$/g;

function splitLongLine(line: string, budget: number): string[] {
  const match = speakerLine.exec(line);
  const prefix = match ? `${match[1]} ` : '';
  const body = match ? match[2] : line.trim();
  const sentences = body.match(sentenceParts) || [body];
  const chunks: string[] = [];
  let current = '';

  for (const sentence of sentences) {
    if (current && current.length + sentence.length > budget) {
      chunks.push(prefix + current);
      current = '';
    }
    if (sentence.length <= budget) {
      current += sentence;
      continue;
    }
    if (current) {
      chunks.push(prefix + current);
      current = '';
    }
    for (let offset = 0; offset < sentence.length; offset += budget) {
      chunks.push(prefix + sentence.slice(offset, offset + budget));
    }
  }
  if (current) chunks.push(prefix + current);
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
