// Focused regression for file-picker cancellation bubbling into the speaker dialog.
// This uses model metadata fixtures; the separate two-speaker walkthrough uses real inference.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
const require = createRequire(process.env.TTS_BROWSER_TOOLS
  ? path.resolve(process.env.TTS_BROWSER_TOOLS, 'package.json') : import.meta.url);
const { chromium } = require('playwright');
const browser = await chromium.launch({ channel: process.env.TTS_BROWSER_CHANNEL || 'chrome', headless: true });
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('**/health', (route) => route.fulfill({ json: {
    status: 'ok', backend: 'fixture', models: 1, ui: true, ui_management: false
  } }));
  await page.route('**/v1/**', (route) => route.fulfill({ json:
    new URL(route.request().url()).pathname === '/v1/models'
      ? { data: [{ id: 'breeze-tts-clone', family: 'breeze_tts', task: 'clon', mode: 'offline',
        path: '/fixture/breeze.gguf', session_options: {}, loaded: true, configured: true }] }
      : { data: [], voices: [], exists: true }
  }));
  await page.goto(process.env.TTS_BASE_URL || 'http://127.0.0.1:8080');
  await page.getByRole('button', { name: 'Lib', exact: true }).click();
  await page.getByRole('navigation', { name: 'Library collections' })
    .getByRole('button', { name: 'Speakers', exact: true }).click();
  await page.getByRole('button', { name: 'New speaker preset', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'New speaker preset', exact: true });
  await dialog.getByLabel('Speaker name', { exact: true }).fill('Keep this draft');
  await dialog.getByLabel('Speech model', { exact: true }).selectOption('model:breeze-tts-clone:configured');
  const input = dialog.locator('input[type=file]');
  await input.setInputFiles(fileURLToPath(new URL('../../../../../demo_voices/demo_1_man.wav', import.meta.url)));
  await dialog.getByRole('option', { name: 'Recording: demo_1_man.wav', exact: true }).waitFor({ state: 'attached' });
  const transcript = "okay,I'm Cemo and what you just heard wasn't a human voice.";
  await dialog.getByLabel('Speaker voice transcript').fill(transcript);
  await input.evaluate((element) => {
    window.__uploadCancelEvents = [];
    element.addEventListener('cancel', (event) => window.__uploadCancelEvents.push({
      target: event.target.tagName, bubbles: event.bubbles, trusted: event.isTrusted
    }));
  });
  // With no chooser interception, headless Chrome dismisses the native picker and
  // delivers its real, trusted input cancel event. Repeat to catch stale state.
  for (let count = 1; count <= 2; count++) {
    await input.click();
    await page.waitForFunction((expected) => window.__uploadCancelEvents.length >= expected, count);
    assert.equal(await dialog.count(), 1, 'Cancelling upload must retain the speaker dialog');
    assert.equal(await dialog.isVisible(), true);
    assert.equal(await dialog.getByLabel('Speaker name', { exact: true }).inputValue(), 'Keep this draft');
    assert.equal(await dialog.getByLabel('Speech model', { exact: true }).inputValue(), 'model:breeze-tts-clone:configured');
    assert.equal(await dialog.getByLabel('Speaker voice', { exact: true }).inputValue(), 'reference');
    assert.equal(await dialog.getByLabel('Speaker voice transcript').inputValue(), transcript);
  }
  const events = await page.evaluate(() => window.__uploadCancelEvents);
  assert(events.every((event) => event.target === 'INPUT' && event.bubbles && event.trusted));
  console.log('PASS native upload cancellation preserves the speaker name, model, recording, and transcript');
  await page.keyboard.press('Escape');
  await dialog.waitFor({ state: 'hidden' });
  assert.deepEqual(errors, []);
  console.log('PASS Escape still closes the speaker dialog itself; no browser runtime errors');
} finally {
  await browser.close();
}
