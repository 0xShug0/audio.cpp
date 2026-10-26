// Browser regressions with deterministic HTTP fixtures. Real-model validation
// is separate; these checks exercise UI state and recipe request construction.
import { createRequire } from 'node:module';
import path from 'node:path';
import assert from 'node:assert/strict';

const require = createRequire(process.env.UI_BROWSER_TOOLS
  ? path.resolve(process.env.UI_BROWSER_TOOLS, 'package.json') : import.meta.url);
const { chromium } = require('playwright');
const wav = Buffer.alloc(44 + 1600 * 2);
wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28);
wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
wav.write('data', 36); wav.writeUInt32LE(wav.length - 44, 40);
const file = { name: 'speech.wav', mimeType: 'audio/wav', buffer: wav };
const packageIds = ['ming_3_0_tts_0_5b_q8_0', 'kitten_tts2_20261021_q4_0', 'canary_180m_flash_q8_0', 'moonshine_streaming_tiny_q8_0'];
const registered = new Map();
const speech = [];
const recipes = [];
let recipeBehavior = 'success';
let holdSpeech = false;
let releaseSpeech;
const browser = await chromium.launch({ channel: process.env.UI_BROWSER_CHANNEL || 'chrome', headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/health', route => route.fulfill({ json: {
    status: 'ok', backend: 'fixture', models: registered.size, ui: true, ui_management: true
  } }));
  await page.route('**/v1/**', async route => {
    const req = route.request();
    const url = new URL(req.url());
    const data = req.method() === 'POST' && req.headers()['content-type']?.includes('application/json')
      ? req.postDataJSON() : {};
    let json = { data: [], voices: [] };
    if (url.pathname === '/v1/models') json = { data: [...registered.values()] };
    if (url.pathname === '/v1/models/load') registered.set(data.id, { ...data, loaded: true, configured: false });
    if (url.pathname === '/v1/models/unload') registered.delete(data.id);
    if (url.pathname === '/v1/ui/models-root') json = { models_root: '/fixture', default_models_root: '/fixture', is_default: true };
    if (url.pathname === '/v1/ui/models/package-sizes') json = { state: 'complete', message: '',
      data: packageIds.map(id => ({ id, installed: true, size_bytes: 100, state: 'ok' })) };
    if (url.pathname === '/v1/ui/path-status') json = { exists: true, file: true, directory: false };
    if (url.pathname === '/v1/ui/upload') json = { path: '/fixture/upload.wav' };
    if (url.pathname === '/v1/audio/speech') {
      speech.push(data);
      if (holdSpeech) await new Promise(resolve => { releaseSpeech = resolve; });
      return route.fulfill({ status: 200, contentType: 'audio/wav', body: wav });
    }
    if (url.pathname === '/v1/recipes/run') {
      recipes.push(data);
      if (recipeBehavior === 'error') return route.fulfill({ status: 400, json: { error: { message: 'Translation failed' } } });
      if (recipeBehavior === 'wait') await new Promise(resolve => setTimeout(resolve, 500));
      json = { text: 'The train leaves in ten minutes.', audio: wav.toString('base64'),
        timing: { wall_ms: 100, audio_duration_ms: 100, rtf: 1 } };
    }
    return route.fulfill({ json });
  });
  await page.goto(process.env.UI_BASE_URL || 'http://127.0.0.1:8080');
  await page.getByRole('button', { name: /^Text to speech/ }).click();
  await page.locator('#model').selectOption('ming3-tts');
  await page.waitForFunction(() => document.querySelector('.runbar .status')?.textContent.trim() === 'Ready');
  console.log('PASS selecting an installed model clears stale status');
  const text = 'The library opens at nine in the morning. Please return the book before Friday. We will meet beside the old clock after lunch.';
  await page.locator('#text').fill(text);
  await page.getByRole('checkbox', { name: 'Split and merge long text' }).check();
  await page.locator('#chunk-budget').fill('40');
  await page.locator('button.run').click();
  await page.waitForFunction(() => document.querySelector('.runbar .status')?.textContent.includes('Complete in'));
  assert.equal(speech.length, 1);
  assert.equal(speech[0].input, text);
  assert.equal(speech[0].options.text_chunk_size, 40);
  console.log('PASS Ming sends one complete request with native chunk settings');
  speech.length = 0;
  await page.locator('#model').selectOption('kitten-tts2-20261021');
  await page.locator('#chunk-budget').fill('40');
  await page.locator('#text').fill(text);
  await page.locator('button.run').click();
  await page.waitForFunction(() => document.querySelector('.runbar .status')?.textContent.includes('Complete in'));
  assert(speech.length > 1);
  assert(speech.every(request => request.model === 'kitten-tts2-20261021'));
  console.log('PASS other models retain client-side splitting');

  await page.getByRole('button', { name: 'Arena', exact: true }).click();
  await page.locator('#arena-model').selectOption('kitten-tts2-20261021');
  await page.locator('#arena-package').selectOption('kitten_tts2_20261021_q4_0');
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await page.locator('#arena-text').fill('A short comparison.');
  holdSpeech = true;
  await page.getByRole('button', { name: 'Run arena', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.arena-mode-switch button')?.disabled);
  assert(await page.getByRole('button', { name: 'ASR', exact: true }).isDisabled());
  for (let attempt = 0; attempt < 100 && !releaseSpeech; attempt++) {
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  assert(releaseSpeech, 'Arena must dispatch the held speech request');
  holdSpeech = false;
  releaseSpeech();
  await page.getByRole('button', { name: 'ASR', exact: true }).waitFor();
  await page.waitForFunction(() => !document.querySelector('.arena-mode-switch button')?.disabled);
  assert.equal(await page.locator('.arena-grid audio').count(), 1);
  await page.getByRole('button', { name: 'ASR', exact: true }).click();
  assert.equal(await page.locator('.arena-grid audio').count(), 0);
  assert(await page.getByRole('button', { name: 'Run arena', exact: true }).isDisabled());
  console.log('PASS Arena locks task changes during a run and clears queue/results on task change');

  await page.getByRole('button', { name: 'Apps', exact: true }).click();
  await page.getByRole('button', { name: /Cross-language dubbing/ }).click();
  const translator = page.getByLabel('Translation model', { exact: true });
  await translator.locator('option[value="model:canary-asr:canary_180m_flash_q8_0"]').waitFor({ state: 'attached' });
  assert.equal(await translator.locator('option').count(), 2, 'Only translation-capable ASR should be listed');
  await translator.selectOption('model:canary-asr:canary_180m_flash_q8_0');
  await page.getByLabel('Speech model', { exact: true }).selectOption('model:kitten-tts2-20261021:kitten_tts2_20261021_q4_0');
  await page.getByLabel('Audio file', { exact: true }).setInputFiles(file);
  await page.getByLabel('Source language', { exact: true }).fill('de');
  await page.getByLabel('Target language', { exact: true }).fill('en');
  await page.getByLabel('Dub voice', { exact: true }).selectOption('Bruno');
  await page.getByRole('button', { name: 'Generate dub', exact: true }).click();
  await page.getByText('Dub ready', { exact: true }).waitFor();
  const recipe = recipes.at(-1);
  assert.equal(recipe.recipe, 'cross-language-dubbing');
  assert.deepEqual(recipe.models, { translator: 'canary-asr', tts: 'kitten-tts2-20261021' });
  assert.equal(recipe.request.audio, '/fixture/upload.wav');
  assert.equal(recipe.request.voice_id, 'Bruno');
  assert.equal(recipe.request.options['translator.language'], 'de');
  assert.equal(recipe.request.options['translator.target_language'], 'en');
  assert(Number.isInteger(recipe.request.options['tts.seed']));
  assert(!('language' in recipe.request.options));
  const result = page.getByRole('region', { name: 'Dub result' });
  const previousUrl = await result.locator('audio').getAttribute('src');
  assert.equal(await result.getByRole('link', { name: 'WAV', exact: true }).getAttribute('download'), 'dub.wav');
  assert.equal(await result.getByRole('link', { name: 'Translation', exact: true }).getAttribute('download'), 'translation.txt');
  recipeBehavior = 'error';
  await page.getByRole('button', { name: 'Generate dub', exact: true }).click();
  await page.getByRole('alert').filter({ hasText: 'Translation failed' }).waitFor();
  assert.equal(await result.locator('audio').getAttribute('src'), previousUrl);
  recipeBehavior = 'wait';
  await page.getByRole('button', { name: 'Generate dub', exact: true }).click();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.getByText('Cancelled', { exact: true }).waitFor();
  assert.equal(await result.locator('audio').getAttribute('src'), previousUrl);
  console.log('PASS recipe routing, role options, voice, downloads, failure recovery and cancellation');
  recipeBehavior = 'success';
  await page.getByLabel('Speech model', { exact: true }).selectOption('model:ming3-tts:ming_3_0_tts_0_5b_q8_0');
  await page.locator('.dubbing summary').filter({ hasText: 'Reference voice' }).click();
  await page.getByLabel('Reference audio', { exact: true }).setInputFiles(file);
  await page.getByLabel('Reference transcript', { exact: true }).fill('A reference sentence.');
  await page.getByLabel('Source language', { exact: true }).fill('');
  await page.locator('.dubbing .model-choice').first().getByRole('button', { name: 'More generation options', exact: true }).click();
  await page.getByLabel('Translation model language', { exact: true }).fill('de');
  await page.getByRole('button', { name: 'Generate dub', exact: true }).click();
  await page.getByText('Dub ready', { exact: true }).waitFor();
  assert.equal(recipes.at(-1).request.voice_ref, '/fixture/upload.wav');
  assert.equal(recipes.at(-1).request.options['tts.reference_text'], 'A reference sentence.');
  assert.equal(recipes.at(-1).request.options['translator.language'], 'de');
  assert(!('voice_id' in recipes.at(-1).request));
  assert(!('translator.reference_text' in recipes.at(-1).request.options));
  console.log('PASS reference conditioning remains TTS-only and translator language settings are honored');
  await page.setViewportSize({ width: 390, height: 844 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  assert.deepEqual(errors, []);
  console.log('PASS mobile width and no browser exceptions');
} finally {
  await browser.close();
}
