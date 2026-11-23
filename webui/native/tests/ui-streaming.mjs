// Real browser interaction against deterministic, incrementally delivered HTTP
// responses. No model weights or prerecorded output fixtures are required.
import { createRequire } from 'node:module';
import { readFile, readdir } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import assert from 'node:assert/strict';

const require = createRequire(process.env.UI_BROWSER_TOOLS
  ? path.resolve(process.env.UI_BROWSER_TOOLS, 'package.json') : import.meta.url);
const { chromium } = require('playwright');
const root = path.resolve(import.meta.dirname, '../../..');
const html = await readFile(path.join(root, 'webui/native/dist/index.html'));
const baselineHtml = process.env.UI_BASELINE_HTML ? await readFile(process.env.UI_BASELINE_HTML) : null;
const specs = await Promise.all((await readdir(path.join(root, 'model_specs')))
  .filter(name => name.endsWith('.json')).map(async name => JSON.parse(await readFile(path.join(root, 'model_specs', name)))));
const packages = specs.flatMap(spec => (spec.packages || []).map(pkg => ({
  id: pkg.id, installed: true, state: 'ok', size_bytes: 100,
  load_target: { family: spec.family,
    model_path: `/fixture/${pkg.target_directory}/${pkg.files?.[0] || 'weights.gguf'}`,
    load_options: {}, session_options: {} }
})));
// Stereo signal detects accidentally dropping the second channel during playback.
const wav = Buffer.alloc(44 + 6400 * 4);
wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(2, 22);
wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(64000, 28);
wav.writeUInt16LE(4, 32); wav.writeUInt16LE(16, 34);
wav.write('data', 36); wav.writeUInt32LE(wav.length - 44, 40);
for (let i = 0; i < 6400; ++i) {
  wav.writeInt16LE(Math.round(1000 * Math.sin(i * .08)), 44 + i * 4);
  wav.writeInt16LE(Math.round(1000 * Math.cos(i * .09)), 46 + i * 4);
}
const audio = wav.toString('base64');
const registered = new Map();
const requests = [];
const loads = [];
let finishStream;
let emitStream;
let releaseSpeech;
let holdSpeech = false;
let streamClosed = 0;
let streamBehavior = 'audio';
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://fixture');
    let json = { data: [], voices: [] };
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const raw = Buffer.concat(chunks);
    const data = req.headers['content-type']?.includes('application/json') ? JSON.parse(raw || '{}') : {};
    if (url.pathname === '/health') json = { status: 'ok', backend: 'fixture', models: registered.size, ui: true, ui_management: true };
    else if (url.pathname === '/v1/models') json = { data: [...registered.values()] };
    else if (url.pathname === '/v1/models/load') { loads.push(data); registered.set(data.id, { ...data, loaded: true, configured: false }); }
    else if (url.pathname === '/v1/models/unload') registered.delete(data.id);
    else if (url.pathname === '/v1/ui/models-root') json = { models_root: '/fixture', default_models_root: '/fixture', is_default: true };
    else if (url.pathname === '/v1/ui/models/package-sizes') json = { state: 'complete', message: '', data: packages };
    else if (url.pathname === '/v1/ui/path-status') json = { exists: true, file: true, directory: false };
    else if (url.pathname === '/v1/ui/upload') json = { path: '/fixture/input.wav' };
    else if (url.pathname === '/v1/audio/speech') {
      requests.push({ path: url.pathname, body: data });
      res.writeHead(200, { 'Content-Type': 'audio/wav' });
      if (holdSpeech) await new Promise(resolve => { releaseSpeech = resolve; });
      res.end(wav); return;
    } else if (url.pathname === '/v1/tasks/stream') {
      requests.push({ path: url.pathname, body: data });
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
      res.on('close', () => { ++streamClosed; });
      const send = value => res.write(`data: ${JSON.stringify(value)}\n\n`);
      emitStream = () => send({ type: 'task.stream.event', event: streamBehavior === 'text'
        ? { partial_text_snapshot: { text: 'Hello again' } }
        : { named_audio_outputs: [{ id: 'chunk_1', audio }] } });
      send({ type: 'task.stream.event', event: streamBehavior === 'text'
        ? { partial_text: { text: 'Hello' } }
        : { partial_text: { text: 'Speaking now' }, named_audio_outputs: [{ id: 'chunk_0', audio }] } });
      finishStream = () => {
        send({ type: 'task.stream.done', result: streamBehavior === 'text'
          ? { text: 'Hello again', timing: { ttft_ms: 10 } }
          : streamBehavior === 'named'
            ? { named_audio_outputs: [{ id: 'output', audio }], timing: { ttft_ms: 10 } }
            : { audio, timing: { ttft_ms: 10 } } });
        res.end('data: [DONE]\n\n');
      };
      return;
    } else if (!url.pathname.startsWith('/v1/')) {
      res.writeHead(200, { 'Content-Type': 'text/html',
        'Content-Security-Policy': "default-src 'self' 'unsafe-inline' blob: data:; connect-src 'self'; media-src 'self' blob: data:" }); res.end(url.searchParams.has('baseline') && baselineHtml ? baselineHtml : html); return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(json));
  } catch (error) { res.writeHead(500); res.end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({ channel: process.env.UI_BROWSER_CHANNEL || 'chrome', headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  await context.addInitScript(() => {
    window.playbackProbe = { contexts: 0, starts: [], stops: 0 };
    const NativeAudioContext = window.AudioContext;
    window.AudioContext = class extends NativeAudioContext {
      constructor(options) { super(options); ++window.playbackProbe.contexts; }
      createBufferSource() {
        const source = super.createBufferSource();
        const start = source.start.bind(source), stop = source.stop.bind(source);
        source.start = (...args) => { window.playbackProbe.starts.push({ channels: source.buffer.numberOfChannels, time: args[0] }); return start(...args); };
        source.stop = (...args) => { ++window.playbackProbe.stops; return stop(...args); };
        return source;
      }
    };
  });
  let baselineRequest;
  if (baselineHtml) {
    const baseline = await context.newPage();
    await baseline.goto(`http://127.0.0.1:${server.address().port}?baseline`);
    await baseline.getByRole('button', { name: /^Text to speech/ }).click();
    await baseline.locator('#model').selectOption('pocket-tts');
    await baseline.waitForFunction(() => !document.querySelector('button.run').disabled);
    await baseline.locator('#text').fill('Hello from Pocket TTS.');
    await baseline.locator('button.run').click();
    await baseline.waitForFunction(() => document.querySelector('.runbar .status')?.textContent.includes('Complete in'));
    baselineRequest = requests.at(-1);
    assert.equal(await baseline.evaluate(() => window.playbackProbe.starts.length), 0);
    await baseline.close();
    registered.clear(); requests.length = 0; loads.length = 0;
  }
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.getByRole('button', { name: /^Text to speech/ }).click();
  const choose = async id => {
    await page.locator('#model').selectOption(id);
    await page.waitForFunction(() => !document.querySelector('button.run').disabled);
  };
  const complete = () => page.waitForFunction(() => document.querySelector('.runbar .status')?.textContent.includes('Complete in'));
  const starts = () => page.evaluate(() => window.playbackProbe.starts.length);
  await choose('pocket-tts');
  await page.locator('#text').fill('Hello from Pocket TTS.');
  assert.equal(await page.getByRole('checkbox', { name: 'Streaming mode', exact: true }).isChecked(), false);
  assert.equal(await page.getByRole('checkbox', { name: 'Auto play', exact: true }).isChecked(), false);
  await page.locator('button.run').click(); await complete();
  const original = requests.at(-1).body;
  assert.equal(requests.at(-1).path, '/v1/audio/speech');
  if (baselineRequest) assert.deepEqual(requests.at(-1), baselineRequest, 'default request must match main');
  assert.equal(loads.at(-1).mode, 'offline');
  assert.equal(await starts(), 0);
  console.log('PASS default path: original speech endpoint, offline mode, no autoplay');

  holdSpeech = true;
  await page.getByText('Auto play', { exact: true }).click();
  await page.locator('button.run').click();
  await page.waitForFunction(() => window.playbackProbe.contexts > 0);
  // Complete while another tab is foregrounded: no second gesture may be needed.
  const other = await context.newPage(); await other.goto('about:blank'); await other.bringToFront();
  while (!releaseSpeech) await new Promise(resolve => setTimeout(resolve, 10));
  releaseSpeech(); holdSpeech = false;
  await complete(); assert.equal(await starts(), 1);
  assert.equal(await page.evaluate(() => window.playbackProbe.starts[0].channels), 2);
  assert.deepEqual(requests.at(-1).body, original);
  await other.close(); await page.bringToFront();
  console.log('PASS autoplay: same request, stereo playback after completion in another tab');

  await page.getByText('Auto play', { exact: true }).click();
  streamBehavior = 'named';
  await page.getByText('Streaming mode', { exact: true }).click();
  await page.locator('button.run').click();
  await page.getByLabel('Streaming text', { exact: true }).waitFor();
  assert.equal(await starts(), 1, 'streaming alone must not autoplay');
  await page.getByRole('button', { name: 'Play live', exact: true }).click();
  await page.waitForFunction(() => window.playbackProbe.starts.length === 2);
  assert.equal(loads.at(-1).mode, 'streaming');
  assert.equal(requests.at(-1).path, '/v1/tasks/stream');
  assert.equal(requests.at(-1).body.request.text, original.input);
  assert.equal(requests.at(-1).body.request.voice_id, original.voice);
  assert.deepEqual(requests.at(-1).body.request.options, original.options);
  assert.equal(requests.at(-1).body.speech_request, undefined);
  assert.equal(await page.locator('button.run').isDisabled(), true);
  finishStream(); await complete(); assert.equal(await starts(), 2);
  console.log('PASS streaming: text and stereo audio before completion, correct voice/options, named final audio, no replay');
  if (process.env.UI_SCREENSHOT) await page.screenshot({ path: process.env.UI_SCREENSHOT, fullPage: true });

  await page.locator('button.run').click();
  await page.getByRole('button', { name: 'Play live', exact: true }).click();
  await page.waitForFunction(() => window.playbackProbe.starts.length === 3);
  await page.getByRole('button', { name: 'Stop playback', exact: true }).click();
  emitStream(); finishStream(); await complete();
  assert.equal(await starts(), 3);
  console.log('PASS stop playback: remaining stream chunks stay muted');

  await page.locator('button.run').click();
  await page.getByRole('button', { name: 'Play live', exact: true }).click();
  await page.waitForFunction(() => window.playbackProbe.starts.length === 4);
  const closed = streamClosed;
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.runbar .status')?.textContent.includes('Cancelled'));
  const closeDeadline = Date.now() + 5000;
  while (streamClosed === closed && Date.now() < closeDeadline) await new Promise(resolve => setTimeout(resolve, 10));
  assert(streamClosed > closed, 'cancel must close the HTTP stream');
  assert((await page.evaluate(() => window.playbackProbe.stops)) > 0);
  await page.locator('button.run').click();
  await page.getByRole('button', { name: 'Play live', exact: true }).click();
  await page.waitForFunction(() => window.playbackProbe.starts.length === 5);
  finishStream(); await complete();
  console.log('PASS cancellation closes the stream and clears audio; the next request plays normally');

  await page.getByText('Streaming mode', { exact: true }).click();
  await page.locator('button.run').click(); await complete();
  assert.equal(loads.at(-1).mode, 'offline');
  assert.equal(requests.at(-1).path, '/v1/audio/speech');
  assert.deepEqual(requests.at(-1).body, original);
  assert.equal(await starts(), 5);
  console.log('PASS disabling streaming restores original request, mode, and manual playback');

  await choose('voxmel-tts');
  assert.equal(await page.getByRole('checkbox', { name: 'Streaming mode', exact: true }).count(), 0);
  assert.equal(await page.getByRole('checkbox', { name: 'Auto play', exact: true }).count(), 1);
  console.log('PASS non-streaming model offers autoplay without streaming');

  await page.getByRole('button', { name: /^ASR \/ Transcription/ }).click();
  await choose('moonshine-asr-tiny');
  await page.locator('input[type=file][accept*="audio"]').first().setInputFiles({ name: 'input.wav', mimeType: 'audio/wav', buffer: wav });
  await page.getByText('Streaming mode', { exact: true }).click();
  streamBehavior = 'text';
  await page.locator('button.run').click();
  await page.getByLabel('Streaming text', { exact: true }).waitFor();
  assert.equal(await page.getByLabel('Streaming text', { exact: true }).textContent(), 'Hello');
  emitStream();
  await page.waitForFunction(() => document.querySelector('[aria-label="Streaming text"]')?.textContent === 'Hello again');
  finishStream(); await complete();
  assert.equal(await starts(), 5);
  console.log('PASS text-only streaming handles deltas and snapshots without audio playback');
  assert.deepEqual(errors, []);
  console.log('PASS no browser runtime errors');
} finally {
  await browser.close();
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
