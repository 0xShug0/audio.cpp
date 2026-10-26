// Run against the real UI with deterministic model API responses.
// See ../DESIGN.md for setup and the boundary between fixtures and real audio work.
import { createRequire } from 'node:module';
import path from 'node:path';
import os from 'node:os';
const require = createRequire(
  process.env.TTS_BROWSER_TOOLS ? path.resolve(process.env.TTS_BROWSER_TOOLS, 'package.json') : import.meta.url
);
const { chromium } = require('playwright');
const { default: AxeBuilder } = require('@axe-core/playwright');
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const out = process.env.TTS_TEST_OUTPUT || path.join(os.tmpdir(), 'audiocpp-tts-ui-results');
await fs.mkdir(out, { recursive: true });
const rate = 24000,
  seconds = 4,
  frames = rate * seconds;
const wav = Buffer.alloc(44 + frames * 2);
wav.write('RIFF');
wav.writeUInt32LE(wav.length - 8, 4);
wav.write('WAVE', 8);
wav.write('fmt ', 12);
wav.writeUInt32LE(16, 16);
wav.writeUInt16LE(1, 20);
wav.writeUInt16LE(1, 22);
wav.writeUInt32LE(rate, 24);
wav.writeUInt32LE(rate * 2, 28);
wav.writeUInt16LE(2, 32);
wav.writeUInt16LE(16, 34);
wav.write('data', 36);
wav.writeUInt32LE(frames * 2, 40);
for (let i = 0; i < frames; i++)
  wav.writeInt16LE(
    Math.round(Math.sin((i * 2 * Math.PI * 180) / rate) * 2500 * Math.min(1, i / 2000, (frames - i) / 2000)),
    44 + i * 2
  );
let mode = 'ok';
const requests = [],
  errors = [],
  checks = [];
const browser = await chromium.launch({ channel: process.env.TTS_BROWSER_CHANNEL || 'chrome', headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
const page = await context.newPage();
page.on('pageerror', (e) => errors.push(e.message));
await page.route('**/health', (r) =>
  r.fulfill({ json: { status: 'ok', backend: 'fixture', models: 1, ui: true, ui_management: false } })
);
await page.route('**/v1/**', async (r) => {
  const path = new URL(r.request().url()).pathname;
  if (path === '/v1/models')
    return r.fulfill({
      json: {
        data: [
          {
            id: 'kokoro-tts',
            family: 'kokoro_tts',
            task: 'tts',
            mode: 'offline',
            path: '/fixture/kokoro.gguf',
            session_options: {},
            loaded: true,
            configured: true
          }
        ]
      }
    });
  if (path === '/v1/audio/speech') {
    requests.push(r.request().postDataJSON());
    if (mode === 'fail')
      return r.fulfill({ status: 500, json: { error: { message: 'Test generation failure. Please try again.' } } });
    if (mode === 'slow') await new Promise((resolve) => setTimeout(resolve, 1500));
    return r.fulfill({ body: wav, contentType: 'audio/wav' }).catch(() => {});
  }
  return r.fulfill({ json: { data: [], voices: [], exists: true } });
});
const app = () => page.locator('.tts-app');
const primary = () => app().locator('.workflow-footer .primary');
const step = (name) =>
  app()
    .getByRole('navigation', { name: 'Narration steps' })
    .getByRole('button', { name: new RegExp(name) });
const summary = (text) =>
  text === 'Projects'
    ? app().locator('summary[aria-label="Projects"]')
    : app().locator('summary').filter({ hasText: text });
const check = (text) => {
  checks.push(text);
  console.log('PASS', text);
};
async function openApp() {
  await page.getByRole('button', { name: 'Apps', exact: true }).click();
  await page.getByRole('button', { name: /Text to speech.*Open app/ }).click();
  await app().waitFor();
}
async function screenshot(name) {
  await page.screenshot({ path: `${out}/${name}.png`, fullPage: false });
}
async function primaryIs(text) {
  await primary().filter({ hasText: text }).waitFor();
  assert.equal(await primary().isEnabled(), true);
}
async function speakerDialog(name, voice = 'af_heart') {
  const dialog = page.getByRole('dialog', { name: 'Add a speaker', exact: true });
  await dialog.waitFor();
  await dialog.getByLabel('Speaker name', { exact: true }).fill(name);
  if (name === 'Alex') {
    const presets = dialog.getByLabel('Speaker preset', { exact: true });
    assert.equal(await presets.locator('option').filter({ hasText: 'Recognition settings' }).count(), 0);
    await presets.selectOption('model-preset:legacy-speech');
    assert.equal(await dialog.getByLabel('Speaker name', { exact: true }).inputValue(), 'Legacy narrator');
    assert.equal(await dialog.getByLabel('Speaker voice', { exact: true }).inputValue(), 'builtin:am_adam');
    await dialog.getByRole('button', { name: 'Model settings', exact: true }).click();
    assert.equal(await dialog.getByLabel('Speech model seed').inputValue(), '41');
    assert.equal(await dialog.getByLabel('Speech model language').inputValue(), 'en');
    await dialog.locator('summary').filter({ hasText: 'Request options JSON' }).click();
    assert.equal(JSON.parse(await dialog.getByLabel('Speech model options JSON').inputValue()).speed, 1.15);
    await dialog.getByLabel('Speech model seed').fill('42');
    await dialog.getByLabel('Speech model seed').blur();
    await dialog.getByLabel('Speaker name', { exact: true }).fill(name);
    check('Legacy speech settings migrate into a complete editable speaker preset');
  } else {
    await dialog.getByLabel('Speech model', { exact: true }).selectOption('model:kokoro-tts:configured');
  }
  await dialog.getByLabel('Speaker voice', { exact: true }).selectOption(`builtin:${voice}`);
  await dialog.getByRole('button', { name: 'Add speaker', exact: true }).click();
  await dialog.waitFor({ state: 'hidden' });
}
async function acceptTake() {
  const d = page.getByRole('dialog', { name: 'Choose a take', exact: true });
  await d.waitFor();
  await d.getByRole('button', { name: 'Use this take', exact: true }).click();
  await d.waitFor({ state: 'hidden' });
}
async function assertLayout(label) {
  const geometry = await app().evaluate((e) => ({
    app: e.getBoundingClientRect().toJSON(),
    width: e.clientWidth,
    scroll: e.scrollWidth,
    page: document.documentElement.scrollHeight,
    viewport: innerHeight,
    scrollY: window.scrollY,
    primary: document.querySelector('.workflow-footer .primary')?.getBoundingClientRect().toJSON()
  }));
  assert(geometry.scroll <= geometry.width + 1, `${label}: horizontal overflow`);
  assert(geometry.app.bottom <= geometry.viewport + 1, `${label}: app below viewport ${JSON.stringify(geometry)}`);
  assert(geometry.page <= geometry.viewport + 1, `${label}: page scrolling ${JSON.stringify(geometry)}`);
  assert.equal(geometry.scrollY, 0, `${label}: window scrolled`);
  if (geometry.primary)
    assert(
      geometry.primary.bottom <= geometry.viewport && geometry.primary.top >= 0,
      `${label}: main action off-screen`
    );
}
await page.goto(process.env.TTS_BASE_URL || 'http://127.0.0.1:5173');
await page.evaluate(async () => {
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open('audiocpp-native-apps', 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore('model-presets', { keyPath: 'id' });
      request.result.createObjectStore('app-projects', { keyPath: 'id' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  await new Promise((resolve, reject) => {
    const tx = db.transaction('model-presets', 'readwrite');
    const legacy = {
      version: 1, id: 'legacy-speech', name: 'Legacy narrator', createdAt: 1,
      model: { id: 'kokoro-tts', display_name: 'Kokoro', family: 'kokoro_tts', task: 'tts',
        mode: 'offline', path: '/fixture/kokoro.gguf', load_options: {}, session_options: {} },
      request: { seed: 41, language: 'en', voice: 'am_adam', options: { speed: 1.15 } },
      chunking: { enabled: true, characters: 400 }
    };
    tx.objectStore('model-presets').put(legacy);
    tx.objectStore('model-presets').put({ ...legacy, id: 'legacy-asr', name: 'Recognition settings',
      model: { ...legacy.model, task: 'asr' } });
    tx.oncomplete = resolve;
    tx.onabort = () => reject(tx.error);
  });
  db.close();
});
assert.equal(await page.getByRole('button', { name: 'Save preset', exact: true }).count(), 0);
await page.getByRole('button', { name: 'Lib', exact: true }).click();
await page.getByRole('heading', { name: 'Voices', exact: true }).waitFor();
assert.equal(await page.getByRole('button', { name: 'Model presets', exact: true }).count(), 0);
assert.equal(await page.getByRole('combobox', { name: 'Speaker preset', exact: true }).count(), 0);
check('Model preset controls are removed; Lib provides a Speakers collection');
assert.equal(await page.getByRole('navigation', { name: 'Library collections' }).getByRole('button', { name: 'Speakers', exact: true }).count(), 1);
const voiceLibrary = page.locator('.voice-library-page');
await voiceLibrary.locator('input[type="file"]').setInputFiles({
  name: 'Reference.wav', mimeType: 'audio/wav', buffer: wav
});
await voiceLibrary.getByLabel('Voice name').fill('Reference recording');
await voiceLibrary.getByLabel('Voice transcript').fill('Words spoken in the recording.');
await voiceLibrary.getByRole('button', { name: 'Save voice', exact: true }).click();
await voiceLibrary.getByRole('button', { name: 'Saved', exact: true }).waitFor();
await voiceLibrary.getByLabel('Voice name').fill('Unsaved name');
page.once('dialog', (dialog) => dialog.dismiss());
await page.getByRole('button', { name: 'Apps', exact: true }).click();
assert.equal(await voiceLibrary.isVisible(), true);
assert.equal(await voiceLibrary.getByLabel('Voice name').inputValue(), 'Unsaved name');
await voiceLibrary.getByLabel('Voice name').fill('Reference recording');
await voiceLibrary.getByRole('button', { name: 'Use in Studio', exact: true }).click();
await voiceLibrary.waitFor({ state: 'detached' });
assert.equal(await page.getByRole('button', { name: 'Studio', exact: true }).getAttribute('class'), 'active');
check('Recorded voice import, saving, unsaved-edit protection, and Studio reuse still work');
await openApp();
assert.equal(await step('Speakers').isDisabled(), true);
assert.equal(await step('Arrange').isDisabled(), true);
assert.equal(await primary().isDisabled(), true);
const script =
  'Welcome to your new narration. Start with a story, and give every word a voice.\n\nA good interface helps you focus on one thing at a time.';
await app()
  .getByLabel('Import text file')
  .setInputFiles({ name: 'Speaker studio.md', mimeType: 'text/markdown', buffer: Buffer.from(script) });
await page.waitForFunction(() =>
  document.querySelector('textarea[aria-label="Script text"]')?.value.includes('Welcome')
);
await assertLayout('Script');
await screenshot('01-script');
await primary().click();
await app().getByRole('button', { name: 'Review passage 2', exact: true }).click();
assert.equal(await app().getByLabel('Preview passage 2').inputValue(), script.split('\n\n')[1]);
await primaryIs('Continue to speakers');
await primary().click();
await primaryIs('Add speaker');
assert.equal(await app().locator('.passage-rail:visible').count(), 1);
assert.equal(await app().locator('.settings-rail:visible').count(), 1);
assert.equal(await app().locator('.passage-voice').count(), 0);
await assertLayout('Empty speakers');
check('Script preview and three-pane workspace fit without page scrolling');
await primary().click();
await speakerDialog('Alex');
await app().getByRole('button', { name: 'Assign Alex', exact: true }).waitFor();
await primaryIs('Generate narration');
await assertLayout('Speaker saved');
await screenshot('02-speakers');
assert.equal(await app().getByLabel('Default model or preset', { exact: true }).count(), 0);
check('A named speaker combines model and voice in a single editor');
await primary().click();
await primaryIs('Continue to arrange');
assert.equal(requests.length, 2);
assert(requests.every((r) => r.voice === 'af_heart'));
assert.deepEqual(requests.map((r) => r.seed), [42, 43]);
assert(requests.every((r) => r.language === 'en' && r.options.speed === 1.15));
check('One project speaker generates both passages with the chosen voice');
const selectedAudio = app().getByLabel('Selected take playback');
await selectedAudio.click({ position: { x: 20, y: 18 } });
await page.waitForFunction(() => !document.querySelector('audio[aria-label="Selected take playback"]').paused);
await selectedAudio.click({ position: { x: 20, y: 18 } });
check('Native passage playback works');
await app().getByRole('button', { name: 'Edit passage 2', exact: true }).click();
await app().getByRole('button', { name: 'Add speaker', exact: true }).click();
await speakerDialog('Maya', 'am_adam');
await primaryIs('Generate narration');
assert.equal(await app().locator('.chunk-badge').textContent(), 'Audio needs updating');
await primary().click();
await acceptTake();
await primaryIs('Continue to arrange');
assert.equal(requests.at(-1).voice, 'am_adam');
assert.equal(requests.at(-1).input, script.split('\n\n')[1]);
check('A second speaker changes only its assigned passage and retains the old take');
await app().getByRole('button', { name: 'Edit Maya', exact: true }).click();
let edit = page.getByRole('dialog', { name: 'Edit speaker', exact: true });
await edit.getByLabel('Speaker name', { exact: true }).fill('Maya Rivera');
await edit.getByRole('button', { name: 'Save speaker', exact: true }).click();
await edit.waitFor({ state: 'hidden' });
await primaryIs('Continue to arrange');
check('Renaming a speaker keeps generated audio current');
await app().getByRole('button', { name: 'Use Maya Rivera for all passages', exact: true }).click();
await primaryIs('Generate narration');
await primary().click();
await primaryIs('Review updated takes');
await primary().click();
await acceptTake();
await primaryIs('Continue to arrange');
assert.equal(requests.at(-1).voice, 'am_adam');
check('Applying a speaker across the narration updates inherited assignments');
await app()
  .getByLabel('Chunk 1 text', { exact: true })
  .fill('Welcome to a quieter studio. Every speaker has a clear place.');
assert.equal(await step('Arrange').isDisabled(), true);
await primary().click();
await acceptTake();
await primaryIs('Continue to arrange');
check('Editing passage text invalidates and regenerates only that passage');
await app().getByRole('button', { name: 'Try another take', exact: true }).click();
let compare = page.getByRole('dialog', { name: 'Choose a take', exact: true });
await compare.waitFor();
assert.equal(await compare.locator('.take-list button').filter({ hasText: 'Used in narration' }).count(), 1);
await compare.getByRole('button', { name: 'Close', exact: true }).click();
await primaryIs('Continue to arrange');
check('Take comparison is a dialog and trying again retains the selected recording');
mode = 'slow';
await app().getByRole('button', { name: 'Try another take', exact: true }).click();
await app().getByRole('button', { name: 'Cancel', exact: true }).click();
await primaryIs('Continue to arrange');
mode = 'ok';
mode = 'fail';
await app().getByRole('button', { name: 'Try another take', exact: true }).click();
await app().getByRole('alert').filter({ hasText: 'Test generation failure' }).waitFor();
await app().getByRole('button', { name: 'Dismiss error' }).click();
await primaryIs('Continue to arrange');
mode = 'ok';
check('Cancellation and generation failure preserve selected audio');
await app().getByRole('button', { name: 'Edit audio', exact: true }).click();
const audioEditor = page.getByRole('dialog', { name: 'Edit passage audio', exact: true });
await audioEditor.waitFor();
await page.waitForFunction(
  () =>
    !Array.from(document.querySelectorAll('button')).find((e) => e.textContent.includes('Insert 0.10 s at playhead'))
      ?.disabled
);
await audioEditor.getByLabel('Insert silence increment').selectOption('0.25');
await audioEditor.getByRole('button', { name: 'Insert 0.25 s at playhead', exact: true }).click();
await page.waitForFunction(() => document.querySelector('.wave-editor header>span')?.textContent.includes('4.25'));
await audioEditor.getByRole('button', { name: 'Undo', exact: true }).click();
await audioEditor.getByRole('button', { name: 'Redo', exact: true }).click();
await audioEditor.getByRole('button', { name: 'Done', exact: true }).click();
check('Waveform editing runs in a dialog with silence insertion and undo/redo');
await app().getByRole('textbox', { name: 'Search passages' }).fill('one thing');
assert.equal(await app().getByRole('button', { name: 'Edit passage 2', exact: true }).count(), 1);
assert.equal(await app().getByRole('button', { name: 'Edit passage 1', exact: true }).count(), 0);
await app().getByRole('textbox', { name: 'Search passages' }).fill('');
check('The left passage rail searches independently of the editor');
await primary().click();
await app().getByLabel('Default pause', { exact: true }).fill('0.5');
await app().getByRole('button', { name: 'Lengthen pause by 0.25 seconds', exact: true }).click();
assert.equal(await app().getByLabel('Pause after chunk 1').inputValue(), '0.75');
await app().getByRole('button', { name: 'Use default', exact: true }).click();
await app().getByRole('button', { name: 'Listen across boundary', exact: true }).click();
await app().getByRole('button', { name: 'Stop preview', exact: true }).click();
await app().getByRole('button', { name: 'Move later', exact: true }).click();
await app().getByRole('button', { name: 'Move earlier', exact: true }).click();
await assertLayout('Arrange');
await screenshot('03-arrange');
await primaryIs('Combine narration');
await primary().click();
await primaryIs('Continue to export');
await page.waitForFunction(() =>
  Number.isFinite(document.querySelector('audio[aria-label="Combined narration playback"]')?.duration)
);
assert(
  Math.abs(
    (await app()
      .getByLabel('Combined narration playback')
      .evaluate((a) => a.duration)) - 8.75
  ) < 0.02
);
check('Real browser assembly includes waveform edits and pauses');
await app().getByLabel('Default pause', { exact: true }).fill('0.75');
assert.equal(await step('Export').isDisabled(), true);
await primary().click();
await primaryIs('Continue to export');
await primary().click();
await summary('Polish audio').click();
await app().getByLabel('Audio gain', { exact: true }).fill('-3');
await app().getByRole('button', { name: 'Preview processing', exact: true }).click();
await app().getByLabel('Processed narration playback').waitFor();
await summary('Polish audio').click();
let event = page.waitForEvent('download');
await primary().click();
let download = await event;
await download.saveAs(`${out}/narration.wav`);
const exported = await fs.readFile(`${out}/narration.wav`);
assert.equal(exported.toString('ascii', 0, 4), 'RIFF');
assert.equal((exported.length - 44) / 48000, 9);
await app().getByLabel('Export format').selectOption('webm');
event = page.waitForEvent('download');
await primary().click();
download = await event;
await download.saveAs(`${out}/narration.webm`);
assert.equal((await fs.readFile(`${out}/narration.webm`)).subarray(0, 4).toString('hex'), '1a45dfa3');
await screenshot('04-export');
check('Audio processing, WAV, and WebM export work with speaker projects');
await page.waitForTimeout(650);
await page.reload();
await openApp();
await app().getByRole('heading', { name: 'Export your narration', exact: true }).waitFor();
assert.equal(await app().getByLabel('Export format').inputValue(), 'webm');
check('Reload preserves speakers, audio, format, and workflow step');
const storage = await page.evaluate(async () => {
  const info = await indexedDB.databases();
  const request = indexedDB.open('audiocpp-tts-speakers', 1);
  const db = await new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  const tx = db.transaction('speakers', 'readonly');
  const records = await new Promise((resolve, reject) => {
    const read = tx.objectStore('speakers').getAll();
    read.onsuccess = () => resolve(read.result);
    read.onerror = () => reject(read.error);
  });
  db.close();
  return { info, imported: records.filter((s) => s.id === 'model-preset:legacy-speech') };
});
assert.equal(storage.imported.length, 1);
assert.equal(storage.imported[0].name, 'Alex');
assert.equal(storage.imported[0].configuration.request.seed, 42);
for (const name of ['audiocpp-native-apps', 'audiocpp-native-studio', 'audiocpp-tts-speakers'])
  assert.equal(storage.info.find((db) => db.name === name)?.version, 1, name);
check('Migration is idempotent, preserves preset edits, and keeps all databases at version 1');
await summary('Projects').click();
const originalId = await app().getByLabel('Open project').inputValue();
await app().getByRole('button', { name: 'New project', exact: true }).click();
await app().getByLabel('Script text').fill('A different project, with the same speaker.');
await primary().click();
await primary().click();
await primary().click();
let savedDialog = page.getByRole('dialog', { name: 'Add a speaker', exact: true });
const savedMaya = await savedDialog
  .getByLabel('Speaker preset', { exact: true })
  .locator('option')
  .evaluateAll((es) => es.find((e) => e.textContent === 'Maya Rivera')?.value);
assert(savedMaya);
await savedDialog.getByLabel('Speaker preset', { exact: true }).selectOption(savedMaya);
await savedDialog.getByRole('button', { name: 'Add speaker', exact: true }).click();
await savedDialog.waitFor({ state: 'hidden' });
await app().getByRole('button', { name: 'Assign Maya Rivera', exact: true }).waitFor();
check('Speaker presets can be reused across projects');
await app().getByRole('button', { name: 'Edit Maya Rivera', exact: true }).click();
edit = page.getByRole('dialog', { name: 'Edit speaker', exact: true });
await edit.getByLabel('Speaker name').fill('Other project name');
await edit.getByRole('button', { name: 'Save speaker', exact: true }).click();
await edit.waitFor({ state: 'hidden' });
await summary('Projects').click();
await app().getByLabel('Open project').selectOption(originalId);
await step('Speakers').click();
await app().getByRole('button', { name: 'Assign Maya Rivera', exact: true }).waitFor();
assert.equal(await app().getByRole('button', { name: 'Assign Other project name', exact: true }).count(), 0);
check('Editing a reused speaker keeps the other project’s speaker snapshot intact');
await page.getByRole('button', { name: 'Lib', exact: true }).click();
await page.getByRole('navigation', { name: 'Library collections' }).getByRole('button', { name: 'Speakers', exact: true }).click();
const library = page.getByRole('region', { name: 'Speaker preset library', exact: true });
await library.getByRole('button', { name: 'Select Alex', exact: true }).waitFor();
await library.getByLabel('Search speaker presets').fill('kokoro');
await library.getByRole('button', { name: 'Select Alex', exact: true }).click();
await library.getByRole('button', { name: 'Edit speaker', exact: true }).click();
let preset = page.getByRole('dialog', { name: 'Edit speaker preset', exact: true });
await preset.getByLabel('Speaker name', { exact: true }).fill('Alex library edition');
await preset.getByRole('button', { name: 'Model settings', exact: true }).click();
await preset.getByLabel('Speech model seed').fill('99');
await preset.getByRole('button', { name: 'Save preset', exact: true }).click();
await preset.waitFor({ state: 'hidden' });
await library.getByRole('heading', { name: 'Alex library edition', exact: true }).waitFor();
await screenshot('07-speaker-library');
check('Lib searches and edits speaker presets including model generation settings');
await library.getByRole('button', { name: 'Duplicate', exact: true }).click();
await library.getByRole('heading', { name: 'Alex library edition copy', exact: true }).waitFor();
page.once('dialog', (dialog) => dialog.accept());
await library.getByRole('button', { name: 'Delete preset', exact: true }).click();
await library.getByRole('button', { name: 'Select Alex library edition copy', exact: true }).waitFor({ state: 'detached' });
await library.getByRole('button', { name: 'Select Alex library edition', exact: true }).click();
page.once('dialog', (dialog) => dialog.accept());
await library.getByRole('button', { name: 'Delete preset', exact: true }).click();
await library.getByRole('button', { name: 'Select Alex library edition', exact: true }).waitFor({ state: 'detached' });
// Remount the collection to ensure deleted legacy imports stay deleted.
const collections = page.getByRole('navigation', { name: 'Library collections' });
await collections.getByRole('button', { name: 'Voices', exact: true }).click();
await collections.getByRole('button', { name: 'Speakers', exact: true }).click();
await library.getByRole('button', { name: 'Select Other project name', exact: true }).waitFor();
assert.equal(await library.getByRole('button', { name: /Select Alex/ }).count(), 0);
check('Duplicating and deleting presets works, including permanent removal of legacy imports');
await library.getByRole('button', { name: 'New speaker preset', exact: true }).click();
preset = page.getByRole('dialog', { name: 'New speaker preset', exact: true });
await preset.getByLabel('Speaker name', { exact: true }).fill('Avery');
await preset.getByLabel('Speech model', { exact: true }).selectOption('model:kokoro-tts:configured');
await preset.getByLabel('Speaker voice', { exact: true }).selectOption('builtin:af_heart');
await preset.getByRole('button', { name: 'Model settings', exact: true }).click();
await preset.getByLabel('Speech model seed').fill('73');
await preset.getByRole('button', { name: 'Save preset', exact: true }).click();
await preset.waitFor({ state: 'hidden' });
await library.getByRole('heading', { name: 'Avery', exact: true }).waitFor();
await page.getByRole('button', { name: 'Apps', exact: true }).click();
await app().getByRole('button', { name: 'Add speaker', exact: true }).click();
savedDialog = page.getByRole('dialog', { name: 'Add a speaker', exact: true });
await savedDialog.getByLabel('Speaker preset', { exact: true }).locator('option').filter({ hasText: /^Avery$/ }).waitFor({ state: 'attached' });
assert.equal(await savedDialog.getByLabel('Speaker preset', { exact: true }).locator('option').filter({ hasText: /^Alex library edition$/ }).count(), 0);
await savedDialog.getByRole('button', { name: 'Cancel', exact: true }).click();
check('The mounted TTS app refreshes its preset picker after library changes');
await page.reload();
await page.getByRole('button', { name: 'Lib', exact: true }).click();
await page.getByRole('navigation', { name: 'Library collections' }).getByRole('button', { name: 'Speakers', exact: true }).click();
await library.getByRole('button', { name: 'Select Avery', exact: true }).click();
assert.equal(await library.locator('dd').filter({ hasText: /^73$/ }).count(), 1);
assert.equal(await library.getByRole('button', { name: /Select Alex/ }).count(), 0);
assert.equal(await library.getByRole('button', { name: 'Use in Studio', exact: true }).count(), 0);
check('New speaker presets and deletions persist across reload and retain generation settings');
for (const [width, height] of [[1440,900], [390,844], [320,844]]) {
  await page.setViewportSize({ width, height });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  const result = await new AxeBuilder({ page }).include('.speaker-library').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
  assert.deepEqual(result.violations.map((v) => v.id), [], `Speaker library ${width}`);
  if (width === 390) await screenshot('08-speaker-library-mobile');
}
check('Speaker management fits desktop and mobile widths and passes accessibility checks');
await page.setViewportSize({ width: 1440, height: 900 });
await openApp();
await step('Speakers').click();
await app().getByRole('button', { name: 'Assign Alex', exact: true }).waitFor();
await primaryIs('Continue to arrange');
await app().getByRole('button', { name: 'Add speaker', exact: true }).click();
savedDialog = page.getByRole('dialog', { name: 'Add a speaker', exact: true });
await savedDialog.getByLabel('Speaker preset', { exact: true }).locator('option').filter({ hasText: /^Avery$/ }).waitFor({ state: 'attached' });
const averyId = await savedDialog.getByLabel('Speaker preset', { exact: true }).locator('option').evaluateAll((es) => es.find((e) => e.textContent === 'Avery')?.value);
await savedDialog.getByLabel('Speaker preset', { exact: true }).selectOption(averyId);
await savedDialog.getByRole('button', { name: 'Model settings', exact: true }).click();
assert.equal(await savedDialog.getByLabel('Speech model seed').inputValue(), '73');
await savedDialog.getByRole('button', { name: 'Cancel', exact: true }).click();
check('Library changes appear in TTS while existing project speakers and selected audio stay intact');
for (const name of ['Script', 'Speakers', 'Arrange', 'Export']) {
  await step(name).click();
  for (const [width, height] of [
    [1440, 900],
    [1280, 720],
    [1024, 768],
    [900, 900],
    [700, 900],
    [390, 844],
    [320, 844]
  ]) {
    await page.setViewportSize({ width, height });
    await page.waitForTimeout(50);
    await assertLayout(`${name} ${width}x${height}`);
    if (width <= 1000) {
      for (const pane of ['Passages', 'Editor', name === 'Speakers' ? 'Speakers' : 'Settings']) {
        await app()
          .getByRole('navigation', { name: 'Workspace panes' })
          .getByRole('button', { name: pane, exact: true })
          .click();
        await assertLayout(`${name} ${width} ${pane}`);
      }
      await app()
        .getByRole('navigation', { name: 'Workspace panes' })
        .getByRole('button', { name: 'Editor', exact: true })
        .click();
    }
    if (width === 390 && name === 'Speakers') await screenshot('05-mobile-editor');
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  const result = await new AxeBuilder({ page })
    .include('.tts-app')
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
    .analyze();
  await fs.writeFile(`${out}/accessibility-${name}.json`, JSON.stringify(result.violations, null, 2));
  assert.deepEqual(
    result.violations.map((v) => v.id),
    [],
    `Accessibility in ${name}`
  );
}
check('All four stages and mobile panes fit 320–1440px without page scrolling and pass WCAG A/AA scans');
await step('Speakers').click();
await screenshot('06-speakers-ready');
assert.deepEqual(errors, []);
check('No browser runtime errors');
await fs.writeFile(
  `${out}/report.json`,
  JSON.stringify(
    {
      checks,
      requests: requests.length,
      errors,
      generatedAt: new Date().toISOString(),
      fixture:
        'Kokoro API fixture; browser audio editing, assembly, processing, speaker storage, and export use the real application.'
    },
    null,
    2
  )
);
console.log('COMPLETE', checks.length, 'checks');
await browser.close();
