// Browser regressions verifying browser-side auto-save of generations:
// off by default with backward-compatible config loading, Studio toggle
// persistence, automatic browser download fallback, optional companion
// .settings.json saving, File System Access API folder writes with
// collision avoidance, graceful fallback on folder write error, and no
// unwanted downloads on text-only ASR or failed runs.
import { createRequire } from 'node:module';
import path from 'node:path';
import assert from 'node:assert/strict';

const require = createRequire(process.env.UI_BROWSER_TOOLS
  ? path.resolve(process.env.UI_BROWSER_TOOLS, 'package.json') : import.meta.url);
const { chromium } = require('playwright');

const baseUrl = process.env.UI_BASE_URL || 'http://127.0.0.1:8080';
const browser = await chromium.launch({ channel: process.env.UI_BROWSER_CHANNEL || 'chrome', headless: true });

function makeWavBuffer() {
  const sampleRate = 16000;
  const numSamples = 1600;
  const dataSize = numSamples * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);
  return buffer;
}

const wavBytes = makeWavBuffer();

async function createConfiguredPage(options = {}) {
  const {
    model = {
      id: 'pocket-tts',
      family: 'pocket_tts',
      task: 'tts',
      mode: 'offline',
      path: '/fixture/pocket-tts.gguf',
      loaded: true,
      configured: true,
      session_options: {}
    },
    failSpeech = false,
    initScript = null
  } = options;

  const errors = [];
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    acceptDownloads: true
  });
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));

  if (initScript) {
    await page.addInitScript(initScript);
  }

  // Record every programmatic <a download> click reliably without depending on OS download timings.
  await page.addInitScript(() => {
    window.__recordedDownloads = [];
    const origClick = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function () {
      if (this.hasAttribute('download')) {
        window.__recordedDownloads.push({
          filename: this.getAttribute('download') || '',
          href: this.getAttribute('href') || ''
        });
      }
      return origClick.call(this);
    };
  });

  await page.route('**/health', (route) => route.fulfill({
    json: { status: 'ok', backend: 'fixture', models: 1, ui: true, ui_management: false }
  }));
  await page.route('**/v1/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === '/v1/models') {
      return route.fulfill({ json: { data: [model] } });
    }
    if (url.pathname === '/v1/audio/voices') {
      return route.fulfill({ json: { voices: ['alba'] } });
    }
    if (url.pathname === '/v1/audio/speech') {
      if (failSpeech) {
        return route.fulfill({ status: 500, json: { error: { message: 'synthetic failure' } } });
      }
      return route.fulfill({
        status: 200,
        contentType: 'audio/wav',
        body: wavBytes
      });
    }
    if (url.pathname === '/v1/audio/transcriptions') {
      return route.fulfill({ json: { text: 'hello world' } });
    }
    if (url.pathname === '/v1/ui/upload') {
      return route.fulfill({ json: { path: '/fixture/uploaded.wav' } });
    }
    if (url.pathname === '/v1/tasks/run') {
      return route.fulfill({
        json: {
          named_audio_outputs: [
            { id: 'vocals', audio: wavBytes.toString('base64') },
            { id: 'drums', audio: wavBytes.toString('base64') }
          ],
          artifacts: [
            {
              id: 'score',
              payload: Buffer.from('MThd').toString('base64'),
              meta: { format: 'midi', mime: 'audio/midi', extension: 'mid' }
            }
          ]
        }
      });
    }
    if (url.pathname === '/v1/ui/path-status') {
      return route.fulfill({ json: { exists: true, file: true, directory: false } });
    }
    return route.fulfill({ json: { data: [], voices: [] } });
  });

  await page.goto(baseUrl);
  await page.waitForFunction(() => !document.querySelector('button.run')?.disabled);
  const runGeneration = async (promptText = 'Hello auto save') => {
    await page.waitForFunction(() => !document.querySelector('button.run')?.disabled);
    const textInput = page.locator('#text');
    await textInput.waitFor({ state: 'visible' });
    await textInput.fill(promptText);
    await page.locator('button.run').click();
  };
  return { context, page, errors, runGeneration };
}

try {
  // 1. Default OFF + legacy localStorage compatibility (only historyLimit stored).
  {
    const { context, page, errors, runGeneration } = await createConfiguredPage({
      initScript: () => {
        localStorage.setItem('audiocpp.ui.configuration', JSON.stringify({ historyLimit: 5 }));
      }
    });
    const autoSaveToggle = page.getByLabel('Auto save', { exact: true });
    assert.equal(await autoSaveToggle.isChecked(), false, 'auto-save must default to off');
    await runGeneration();
    await page.locator('.audio-list audio').waitFor({ state: 'visible' });
    const downloads = await page.evaluate(() => window.__recordedDownloads);
    assert.equal(downloads.length, 0, 'no auto-download when auto-save is off');
    assert.deepEqual(errors, []);
    console.log('PASS auto-save defaults off and loads legacy localStorage cleanly');
    await context.close();
  }

  // 2. Studio quick toggle persists across reload and auto-downloads WAV only when autoSaveSettings is off;
  //    enabling companion settings JSON in Configuration downloads both WAV and .settings.json with matching stem.
  {
    const { context, page, errors, runGeneration } = await createConfiguredPage({
      initScript: () => {
        delete window.showDirectoryPicker;
      }
    });
    await page.locator('label.toggle', { hasText: 'Auto save' }).click();
    assert.equal(await page.getByLabel('Auto save', { exact: true }).isChecked(), true);
    await page.reload();
    await page.locator('button.run').waitFor({ state: 'visible' });
    assert.equal(await page.getByLabel('Auto save', { exact: true }).isChecked(), true, 'auto-save persists across reload');

    await runGeneration();
    await page.locator('.auto-save-notice').waitFor({ state: 'visible' });
    let downloads = await page.evaluate(() => window.__recordedDownloads);
    assert.equal(downloads.length, 1, 'auto-downloads single WAV when settings JSON option is off');
    assert.match(downloads[0].filename, /^pocket-tts-\d{8}-\d{6}-output\.wav$/);

    // Enable companion .settings.json in Configuration and verify persistence + paired download.
    await page.getByRole('button', { name: 'Configuration', exact: true }).click();
    assert.equal(await page.locator('#auto-save-enabled').isChecked(), true);
    await page.locator('#auto-save-settings').check();
    await page.getByRole('button', { name: 'Save configuration', exact: true }).click();
    await page.getByText('Configuration saved.').waitFor({ state: 'visible' });

    await page.reload();
    await page.locator('button.run').waitFor({ state: 'visible' });
    await page.evaluate(() => { window.__recordedDownloads = []; });
    await runGeneration();
    await page.locator('.auto-save-notice').waitFor({ state: 'visible' });
    await page.waitForFunction(() => window.__recordedDownloads.length === 2);
    downloads = await page.evaluate(() => window.__recordedDownloads);
    assert.equal(downloads.length, 2, 'auto-downloads WAV and companion settings JSON');
    const wavName = downloads[0].filename;
    const jsonName = downloads[1].filename;
    assert.ok(wavName.endsWith('-output.wav'), `expected WAV filename, got ${wavName}`);
    assert.ok(jsonName.endsWith('.settings.json'), `expected settings JSON filename, got ${jsonName}`);
    const wavStem = wavName.slice(0, -'-output.wav'.length);
    const jsonStem = jsonName.slice(0, -'.settings.json'.length);
    assert.equal(wavStem, jsonStem, 'WAV and settings JSON share the same timestamped stem');

    assert.deepEqual(errors, []);
    console.log('PASS Studio toggle and Configuration persist and auto-download WAV + optional settings JSON');
    await context.close();
  }

  // 3. File System Access API folder mode: writes directly to chosen folder without <a download>,
  //    avoids overwriting collisions with -2 suffix, and falls back gracefully on write failure.
  {
    const { context, page, errors, runGeneration } = await createConfiguredPage({
      initScript: () => {
        const files = new Map();
        window.__mockFolderFiles = files;
        window.__mockFailWrites = false;
        window.__mockPermission = 'granted';
        window.showDirectoryPicker = async () => ({
          name: 'MyOutputs',
          queryPermission: async () => window.__mockPermission,
          requestPermission: async () => window.__mockPermission,
          getFileHandle: async (name, opts = {}) => {
            if (!opts.create) {
              if (!files.has(name)) {
                const err = new Error('Not found');
                err.name = 'NotFoundError';
                throw err;
              }
            }
            return {
              createWritable: async () => {
                if (window.__mockFailWrites) {
                  throw new Error('disk full');
                }
                let written = null;
                return {
                  write: async (blob) => { written = blob; },
                  close: async () => { files.set(name, written); }
                };
              }
            };
          }
        });
      }
    });

    await page.getByRole('button', { name: 'Configuration', exact: true }).click();
    await page.locator('#auto-save-enabled').check();
    await page.locator('#auto-save-settings').check();
    await page.getByRole('button', { name: 'Choose folder', exact: true }).click();
    await page.getByText('Folder: MyOutputs').waitFor({ state: 'visible' });
    await page.getByRole('button', { name: 'Save configuration', exact: true }).click();

    await page.getByRole('button', { name: 'Studio', exact: true }).click();
    await runGeneration();
    await page.locator('.auto-save-notice').waitFor({ state: 'visible' });
    assert.ok((await page.locator('.auto-save-notice').textContent())?.includes('Auto-saved 2 files to MyOutputs'));
    assert.equal(await page.evaluate(() => window.__recordedDownloads.length), 0,
      'directory mode must write directly without triggering <a download>');
    const firstBatch = await page.evaluate(() => [...window.__mockFolderFiles.keys()]);
    assert.equal(firstBatch.length, 2, 'wrote WAV and settings JSON into chosen folder');

    // Second run within the same second (or same timestamp) must not clobber existing files.
    await page.evaluate(() => {
      const fixed = new Date('2026-11-26T14:50:00Z');
      const OrigDate = Date;
      // @ts-ignore
      window.Date = class extends OrigDate {
        constructor(...args) { super(...(args.length ? args : [fixed.getTime()])); }
        static now() { return fixed.getTime(); }
      };
    });
    await runGeneration();
    await page.locator('.auto-save-notice').waitFor({ state: 'visible' });
    await runGeneration();
    await page.locator('.auto-save-notice').waitFor({ state: 'visible' });
    const allNames = await page.evaluate(() => [...window.__mockFolderFiles.keys()]);
    assert.ok(allNames.some((name) => /-output-2\.wav$/.test(name)),
      `expected collision-safe -2 WAV filename, got: ${allNames.join(', ')}`);
    assert.ok(allNames.some((name) => /-2\.settings\.json$/.test(name)),
      `expected collision-safe -2 settings JSON filename, got: ${allNames.join(', ')}`);

    // If folder write fails, auto-save falls back gracefully to browser downloads with a warning.
    await page.evaluate(() => {
      window.__mockFailWrites = true;
      window.__recordedDownloads = [];
    });
    await runGeneration();
    await page.locator('.auto-save-warning').waitFor({ state: 'visible' });
    assert.ok((await page.locator('.auto-save-warning').textContent())?.includes('disk full'));
    await page.waitForFunction(() => window.__recordedDownloads.length === 2);

    assert.deepEqual(errors, []);
    console.log('PASS folder mode writes directly, avoids collisions, and falls back gracefully on write error');
    await context.close();
  }

  // 4. Failed run must NOT trigger any auto-save download.
  {
    const { context, page, errors, runGeneration } = await createConfiguredPage({
      failSpeech: true,
      initScript: () => {
        delete window.showDirectoryPicker;
        localStorage.setItem('audiocpp.ui.configuration', JSON.stringify({
          historyLimit: 10,
          autoSave: true,
          autoSaveSettings: true
        }));
      }
    });
    await runGeneration();
    await page.locator('.status.error').waitFor({ state: 'visible' });
    assert.equal(await page.evaluate(() => window.__recordedDownloads.length), 0,
      'failed generation must not auto-save');
    assert.deepEqual(errors, []);
    console.log('PASS failed run does not trigger auto-save');
    await context.close();
  }

  // 5. Multi-track named_audio_outputs + base64 artifact auto-save cleanly under strict CSP (no fetch).
  {
    const { context, page, errors } = await createConfiguredPage({
      model: {
        id: 'org/custom:gen',
        family: 'custom_gen',
        task: 'gen',
        mode: 'offline',
        path: '/fixture/gen.gguf',
        loaded: true,
        configured: true,
        session_options: {}
      },
      initScript: () => {
        delete window.showDirectoryPicker;
        localStorage.setItem('audiocpp.ui.configuration', JSON.stringify({
          historyLimit: 10,
          autoSave: true,
          autoSaveSettings: true
        }));
      }
    });
    await page.locator('button.run').click();
    await page.locator('.auto-save-notice').waitFor({ state: 'visible' });
    await page.waitForFunction(() => window.__recordedDownloads.length === 4);
    const names = await page.evaluate(() => window.__recordedDownloads.map((item) => item.filename));
    assert.match(names[0], /^org-custom-gen-\d{8}-\d{6}-vocals\.wav$/);
    assert.match(names[1], /^org-custom-gen-\d{8}-\d{6}-drums\.wav$/);
    assert.match(names[2], /^org-custom-gen-\d{8}-\d{6}-score\.mid$/);
    assert.match(names[3], /^org-custom-gen-\d{8}-\d{6}\.settings\.json$/);
    assert.deepEqual(errors, []);
    console.log('PASS multi-track audio and base64 artifacts auto-save with sanitized filenames');
    await context.close();
  }

  // 6. Text-only ASR run must NOT trigger auto-save even when enabled.
  {
    const { context, page, errors } = await createConfiguredPage({
      model: {
        id: 'whisper-tiny',
        family: 'whisper',
        task: 'asr',
        mode: 'offline',
        path: '/fixture/whisper.gguf',
        loaded: true,
        configured: true,
        session_options: {}
      },
      initScript: () => {
        delete window.showDirectoryPicker;
        localStorage.setItem('audiocpp.ui.configuration', JSON.stringify({
          historyLimit: 10,
          autoSave: true,
          autoSaveSettings: true
        }));
      }
    });
    await page.locator('#source').setInputFiles({
      name: 'clip.wav',
      mimeType: 'audio/wav',
      buffer: wavBytes
    });
    await page.locator('button.run').click();
    const transcript = page.locator('textarea.transcript');
    await transcript.waitFor({ state: 'visible' });
    assert.equal(await transcript.inputValue(), 'hello world');
    assert.equal(await page.evaluate(() => window.__recordedDownloads.length), 0,
      'text-only ASR run must not trigger auto-save');
    assert.deepEqual(errors, []);
    console.log('PASS text-only ASR run does not trigger auto-save');
    await context.close();
  }
} finally {
  await browser.close();
}
