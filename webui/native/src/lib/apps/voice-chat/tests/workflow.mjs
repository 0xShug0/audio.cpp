import { createRequire } from 'node:module';
import path from 'node:path';
import assert from 'node:assert/strict';
const require = createRequire(process.env.UI_BROWSER_TOOLS ? path.resolve(process.env.UI_BROWSER_TOOLS, 'package.json') : import.meta.url);
const { chromium } = require('playwright');
const base = process.env.UI_BASE_URL || 'http://127.0.0.1:8080';
const wav = Buffer.alloc(44 + 15360 * 2);
wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
wav.write('data', 36); wav.writeUInt32LE(wav.length - 44, 40);
const audio = wav.toString('base64');
const rows = [
  { id: 'asr', family: 'owsm_ctc', task: 'asr', mode: 'offline', path: '/fixture/asr.gguf' },
  { id: 'chat', family: 'lfm3_audio', task: 's2s', mode: 'streaming', path: '/fixture/lfm3.gguf' },
  { id: 'voice', family: 'lfm3_audio', task: 'tts', mode: 'streaming', path: '/fixture/lfm3.gguf' },
  { id: 'avatar', family: 'soulx_flashhead', task: 'gen', mode: 'streaming', path: '/fixture/soulx.gguf' }
].map(m => ({ ...m, loaded: true, configured: true, session_options: {} }));
const calls = []; let failAvatar = false;
const managed = process.env.UI_MANAGED_CASE === '1';
const registered = new Map(managed ? [] : rows.map(row => [row.id, row]));
const loads = [];
const chatId = managed ? 'voice-chat-conversation' : 'chat';
const voiceId = managed ? 'voice-chat-voice' : 'voice';
const avatarId = managed ? 'voice-chat-avatar' : 'avatar';
const browser = await chromium.launch({ channel: process.env.UI_BROWSER_CHANNEL || 'chrome', headless: true,
  args: ['--autoplay-policy=no-user-gesture-required', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.route('**/health', r => r.fulfill({ json: { status: 'ok', backend: 'cuda', ui: true, ui_management: managed, models: registered.size } }));
  await page.route('**/v1/**', async route => {
    const u = new URL(route.request().url());
    if (u.pathname === '/v1/models') return route.fulfill({ json: { data: [...registered.values()] } });
    if (u.pathname === '/v1/models/load') { const data = route.request().postDataJSON(); loads.push(data); registered.set(data.id, { ...data, loaded: true, configured: false }); return route.fulfill({ json: {} }); }
    if (u.pathname === '/v1/models/unload') { registered.delete(route.request().postDataJSON().id); return route.fulfill({ json: {} }); }
    if (u.pathname === '/v1/ui/path-status') return route.fulfill({ json: { exists: true, file: true, directory: false } });
    if (u.pathname === '/v1/ui/models/package-sizes') return route.fulfill({ json: { state: 'complete', data: [
      ['owsm_ctc_v4_1b_q8_0', 'owsm_ctc', '/fixture/asr.gguf'],
      ['lfm3_audio_2b_q8_0', 'lfm3_audio', '/fixture/lfm3.gguf'],
      ['soulx_flashhead_lite_q8_0', 'soulx_flashhead', '/fixture/soulx.gguf']
    ].map(([id, family, model_path]) => ({ id, installed: managed, size_bytes: 100, state: 'ok', load_target: { family, model_path, load_options: {}, session_options: {} } })) } });
    if (u.pathname === '/v1/ui/upload') return route.fulfill({ json: { path: '/fixture/upload.wav' } });
    if (u.pathname === '/v1/tasks/run') { calls.push(route.request().postDataJSON()); return route.fulfill({ json: { text: 'What should I learn today?' } }); }
    if (u.pathname === '/v1/tasks/stream') {
      const body = route.request().postDataJSON(); calls.push(body);
      if (body.model === avatarId && failAvatar) { failAvatar = false; return route.fulfill({ status: 500, json: { error: { message: 'Avatar test failure' } } }); }
      const event = body.model === chatId ? { partial_text: { text: 'Try learning a little about the stars.' } } : body.model === voiceId ? { audio } : {
        artifacts: [{ id: 'video_0', kind: 'custom', payload: Buffer.alloc(32 * 32 * 3 * 24, 140).toString('base64'), meta: { format: 'rgb24', width: '32', height: '32', frames: '24', fps: '25' } }]
      };
      const result = body.model === chatId ? { text: 'Try learning a little about the stars.', audio,
        artifacts: [{ id: 'lfm3_audio.reply', kind: 'acoustic_tokens', payload: 'AAAAAA==', meta: { format: 'lfm3_audio.reply/1' } }] } : body.model === voiceId ? { audio } : {};
      return route.fulfill({ contentType: 'text/event-stream', body: `data: ${JSON.stringify({ type: 'task.stream.event', event })}\n\ndata: ${JSON.stringify({ type: 'task.stream.done', result })}\n\ndata: [DONE]\n\n` });
    }
    return route.fulfill({ json: { data: [], voices: [], exists: true, models_root: '/fixture', default_models_root: '/fixture', is_default: true } });
  });
  await page.exposeFunction('fixtureRecipe', body => {
    calls.push(body);
    if (failAvatar) { failAvatar = false; return { error: 'Avatar test failure' }; }
    const video = { id: 'video_0', kind: 'custom', payload: Buffer.alloc(32 * 32 * 3 * 24, 140).toString('base64'), meta: { format: 'rgb24', width: '32', height: '32', frames: '24', fps: '25' } };
    const line = value => `data: ${JSON.stringify(value)}\n\n`;
    return {
      prefix: line({ type: 'task.stream.event', event: { partial_text: { text: 'Try learning ' } } }) + line({ type: 'task.stream.event', event: { audio } }) + line({ type: 'task.stream.event', event: { artifacts: [video] } }),
      tail: line({ type: 'task.stream.event', event: { partial_text: { text: 'a little about the stars.' } } }) + line({ type: 'task.stream.done', result: { audio, text: 'Try learning a little about the stars.', artifacts: [{ id: 'lfm3_audio.reply', kind: 'acoustic_tokens', payload: 'AAAAAA==', meta: { format: 'lfm3_audio.reply/1' } }] } }) + 'data: [DONE]\n\n'
    };
  });
  await page.addInitScript(() => {
    const nativeFetch = window.fetch;
    window.fetch = async (input, init) => {
      const url = new URL(typeof input === 'string' ? input : input.url, location.href);
      if (url.pathname !== '/v1/recipes/run') return nativeFetch(input, init);
      const data = await window.fixtureRecipe(JSON.parse(init.body));
      if (data.error) return new Response(JSON.stringify({ error: { message: data.error } }), { status: 500 });
      window.__ttsComplete = false;
      const encoder = new TextEncoder(); let timer, ended = false, abort;
      const stream = new ReadableStream({
        start(controller) {
          abort = () => { if (!ended) { ended = true; clearTimeout(timer); controller.error(new DOMException('Aborted', 'AbortError')); } };
          if (init.signal?.aborted) { abort(); return; }
          init.signal?.addEventListener('abort', abort, { once: true });
          controller.enqueue(encoder.encode(data.prefix));
          timer = setTimeout(() => {
            ended = true; window.__ttsComplete = true; init.signal?.removeEventListener('abort', abort);
            controller.enqueue(encoder.encode(data.tail)); controller.close();
          }, 1600);
        },
        cancel() { ended = true; clearTimeout(timer); init.signal?.removeEventListener('abort', abort); }
      });
      return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
    };
  });
  await page.goto(base);
  await page.getByRole('button', { name: 'Apps', exact: true }).click();
  await page.getByRole('button', { name: /Voice chat Talk with LFM3/ }).click();
  assert.equal(await page.getByLabel('Conversation', { exact: true }).count(), 0);
  assert.equal(await page.getByLabel('Avatar model', { exact: true }).count(), 0);
  await page.getByLabel('Portrait', { exact: true }).setInputFiles({ name: 'portrait.svg', mimeType: 'image/svg+xml', buffer: Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><rect width="512" height="512" fill="#678"/></svg>') });
  await page.getByLabel('Voice', { exact: true }).selectOption('british_english_male');
  await page.getByLabel('Speaking speed', { exact: true }).selectOption('1.3');
  await page.getByLabel('Personality', { exact: true }).fill('A warm, concise assistant.');
  assert.equal(await page.getByRole('button', { name: 'Refresh models', exact: true }).count(), 0);
  const controls = await Promise.all([
    page.getByLabel('Recognition', { exact: true }).boundingBox(),
    page.getByLabel('Voice', { exact: true }).boundingBox(),
    page.locator('.portrait-control').boundingBox()
  ]);
  assert.ok(controls.every(box => Math.abs(box.y - controls[0].y) < 1 && Math.abs(box.height - controls[0].height) < 1), 'setup controls must align');
  const send = () => page.getByLabel('Send a recording', { exact: true }).setInputFiles({ name: 'question.wav', mimeType: 'audio/wav', buffer: wav });
  await send();
  await page.locator('.portrait-stage.speaking').waitFor();
  assert.equal(await page.evaluate(() => window.__ttsComplete), false, 'playback must start before conversation finishes');
  assert.equal(await page.locator('.messages article').last().locator('p').textContent(), 'Try learning ', 'reply text is still incomplete during playback');
  await page.getByRole('status').filter({ hasText: 'Your turn' }).waitFor({ timeout: 10000 }).catch(async error => { console.error(await page.locator('.voice-chat').innerText()); console.error(errors); throw error; });
  assert.equal(calls.find(c => c.recipe === 'voice-chat-avatar').request.voice_id, 'british_english_male');
  assert.equal(calls.find(c => c.recipe === 'voice-chat-avatar').request.options.speaking_speed, 1.3);
  assert.equal(calls.find(c => c.recipe === 'voice-chat-avatar').request.text, 'Respond with interleaved text and audio. A warm, concise assistant.');
  assert.deepEqual(calls.find(c => c.recipe === 'voice-chat-avatar').models, { conversation: chatId, tts: voiceId, avatar: avatarId });
  assert.equal(calls.filter(c => c.model === voiceId).length, 0, 'no complete TTS audio request before avatar');
  assert.equal(calls.find(c => c.recipe === 'voice-chat-avatar').request.artifacts.length, 0);
  assert.equal(await page.locator('.messages article').count(), 2);
  await send();
  await page.getByRole('status').filter({ hasText: 'Your turn' }).waitFor();
  assert.equal(calls.filter(c => c.recipe === 'voice-chat-avatar')[1].request.artifacts.length, 2);
  if (managed) {
    assert.equal(loads.length, 4, 'prepared sessions must be reused across turns');
    assert.ok(loads.every(model => model.id.startsWith('voice-chat-')));
    assert.ok(loads.filter(model => model.family !== 'owsm_ctc').every(model => model.mode === 'streaming'));
  }
  await page.getByRole('button', { name: 'New conversation', exact: true }).click();
  assert.equal(await page.locator('.messages article').count(), 0);
  failAvatar = true; await send();
  await page.getByRole('alert').filter({ hasText: 'Avatar test failure' }).waitFor();
  await send(); await page.getByRole('status').filter({ hasText: 'Your turn' }).waitFor();
  assert.equal(calls.filter(c => c.recipe === 'voice-chat-avatar').at(-1).request.artifacts.length, 0);
  await send(); await page.locator('.portrait-stage.speaking').waitFor();
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await page.getByRole('status').filter({ hasText: 'Stopped' }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Talk', exact: true }).isEnabled(), true);
  await page.getByRole('button', { name: 'Talk', exact: true }).click();
  await page.getByRole('button', { name: 'Finish', exact: true }).waitFor();
  await page.waitForTimeout(400);
  await page.getByRole('button', { name: 'Finish', exact: true }).click();
  await page.getByRole('status').filter({ hasText: 'Your turn' }).waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'mobile layout must not overflow');
  assert.deepEqual(errors, []);
  console.log('PASS: voice selection, ASR routing, multi-turn artifacts, new chat, avatar failure recovery and synchronized playback');
} finally { await browser.close(); }
