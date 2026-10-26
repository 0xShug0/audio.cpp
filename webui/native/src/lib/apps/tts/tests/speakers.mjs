import assert from 'node:assert/strict';
import {
  newProject,
  newChunk,
  ensureSpeakers,
  speechSignature,
  chunkConfiguration,
  chunkReference,
  chunkReady,
  assemblySignature,
  assemblyCurrent,
  assignedSpeaker
} from '../project.ts';
function configuration(id, voice = 'af_heart') {
  return {
    version: 1,
    id: `model:${id}`,
    name: id,
    createdAt: 0,
    model: {
      id,
      display_name: id,
      family: id,
      task: 'tts',
      mode: 'offline',
      path: `/${id}.gguf`,
      load_options: {},
      session_options: {}
    },
    request: { voice, seed: 1, options: {} },
    chunking: { enabled: true, characters: 400 }
  };
}
function legacy() {
  const p = newProject();
  p.version = 1;
  delete p.speakers;
  delete p.defaultSpeakerId;
  p.speech = configuration('kokoro');
  p.chunks = [newChunk('First.'), newChunk('Second.'), newChunk('Third.')];
  return p;
}
function take(p, c) {
  const t = {
    id: crypto.randomUUID(),
    text: c.text,
    configuration: structuredClone(chunkConfiguration(p, c)),
    signature: speechSignature(p, c),
    seed: 1,
    audio: new Blob(['audio']),
    duration: 1,
    createdAt: 0
  };
  c.takes.push(t);
  c.takeId = t.id;
}
{
  const p = legacy();
  p.chunks[1].voice = { kind: 'builtin', id: 'am_adam' };
  p.chunks[2].override = configuration('another', 'different');
  p.archivedChunks = [newChunk('Archived', configuration('archived'))];
  p.chunks.forEach((c) => take(p, c));
  p.assembled = { audio: new Blob(['assembled']), signature: assemblySignature(p) };
  const signatures = [...p.chunks, ...p.archivedChunks].map((c) => speechSignature(p, c));
  assert.equal(ensureSpeakers(p), true);
  assert.equal(p.version, 2);
  assert.deepEqual(
    [...p.chunks, ...p.archivedChunks].map((c) => speechSignature(p, c)),
    signatures
  );
  assert(p.chunks.every((c) => chunkReady(p, c)));
  assert(assemblyCurrent(p));
  assert.equal(p.speakers.length, 4);
  const ids = p.speakers.map((s) => s.id);
  assert.equal(ensureSpeakers(p), false);
  assert.deepEqual(
    p.speakers.map((s) => s.id),
    ids
  );
  console.log('PASS legacy model, voice, override, archive, take and assembly migration');
}
{
  const p = legacy();
  p.reference = { id: 'ref', name: 'A voice', audio: new Blob(['ref']), transcript: 'Example' };
  p.chunks[1].voice = { kind: 'builtin', id: '' };
  p.chunks[2].voice = { kind: 'reference', data: { ...p.reference, id: 'custom' } };
  const signatures = p.chunks.map((c) => speechSignature(p, c));
  ensureSpeakers(p);
  assert.deepEqual(
    p.chunks.map((c) => speechSignature(p, c)),
    signatures
  );
  assert.equal(chunkReference(p, p.chunks[0]).id, 'ref');
  assert.equal(chunkReference(p, p.chunks[1]), null);
  assert.equal(chunkReference(p, p.chunks[2]).id, 'custom');
  console.log('PASS recorded speakers and explicit model-default voices preserve signatures');
}
{
  const p = legacy();
  ensureSpeakers(p);
  const first = p.speakers[0];
  const second = {
    ...structuredClone(first),
    id: 'second',
    name: 'Second speaker',
    voice: { kind: 'builtin', id: 'am_adam' }
  };
  p.speakers.push(second);
  p.chunks[1].speakerId = second.id;
  p.chunks.forEach((c) => take(p, c));
  assert.equal(chunkConfiguration(p, p.chunks[0]).request.voice, 'af_heart');
  assert.equal(chunkConfiguration(p, p.chunks[1]).request.voice, 'am_adam');
  first.name = 'Renamed';
  assert(p.chunks.every((c) => chunkReady(p, c)));
  first.voice = { kind: 'builtin', id: 'af_sarah' };
  assert.equal(chunkReady(p, p.chunks[0]), false);
  assert.equal(chunkReady(p, p.chunks[1]), true);
  assert.equal(chunkReady(p, p.chunks[2]), false);
  p.defaultSpeakerId = second.id;
  assert.equal(assignedSpeaker(p, p.chunks[2]).id, second.id);
  assert.equal(chunkReady(p, p.chunks[1]), true);
  console.log('PASS speaker assignment, naming, defaults, and selective audio invalidation');
}
{
  const p = newProject();
  assert.equal(ensureSpeakers(p), false);
  assert.deepEqual(p.speakers, []);
  const c = newChunk('Fresh');
  assert.equal(chunkConfiguration(p, c), null);
  assert.equal(chunkReference(p, c), null);
  console.log('PASS new projects start with an empty cast');
}
