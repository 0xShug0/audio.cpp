// Browser regressions with deterministic HTTP fixtures verifying that Studio
// prompts to reload when model or weight session settings change while a model
// is loaded, reloads on confirmation, clears when reverted, and guides users on
// server-configured models.
import { createRequire } from 'node:module';
import path from 'node:path';
import assert from 'node:assert/strict';

const require = createRequire(process.env.UI_BROWSER_TOOLS
  ? path.resolve(process.env.UI_BROWSER_TOOLS, 'package.json') : import.meta.url);
const { chromium } = require('playwright');

const baseUrl = process.env.UI_BASE_URL || 'http://127.0.0.1:8080';
const browser = await chromium.launch({ channel: process.env.UI_BROWSER_CHANNEL || 'chrome', headless: true });

async function runManagedScenario(context) {
  const registered = new Map();
  const loads = [];
  const unloads = [];
  const errors = [];
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));

  const packages = [
    {
      id: 'm5_tts_50m_customvoice_f16',
      installed: true,
      size_bytes: 100,
      state: 'ok',
      load_target: {
        family: 'm5_tts',
        model_path: 'models/M5-TTS-GGUF/m5-tts-50m-customvoice-f16.gguf',
        load_options: {},
        session_options: {}
      }
    },
    {
      id: 'zonos2_q8_0',
      installed: true,
      size_bytes: 100,
      state: 'ok',
      load_target: {
        family: 'zonos2',
        model_path: 'models/ZONOS2-GGUF',
        load_options: {},
        session_options: {}
      }
    },
    {
      id: 'yue2_main_q8_0',
      installed: true,
      size_bytes: 100,
      state: 'ok',
      load_target: {
        family: 'yue2',
        model_path: 'models/Yue2-3B-GGUF',
        load_options: {},
        session_options: {
          'yue2.model_gguf': 'yue2-3b-q8_0.gguf',
          'yue2.vae_gguf': 'yue2-vae-f16.gguf'
        }
      }
    },
    {
      id: 'yue2_main_q4_0',
      installed: true,
      size_bytes: 100,
      state: 'ok',
      load_target: {
        family: 'yue2',
        model_path: 'models/Yue2-3B-GGUF',
        load_options: {},
        session_options: {
          'yue2.model_gguf': 'yue2-3b-q4_0.gguf',
          'yue2.vae_gguf': 'yue2-vae-f16.gguf'
        }
      }
    }
  ];

  await page.route('**/health', (route) => route.fulfill({
    json: { status: 'ok', backend: 'fixture', models: registered.size, ui: true, ui_management: true }
  }));
  await page.route('**/v1/**', async (route) => {
    const req = route.request();
    const url = new URL(req.url());
    const data = req.method() === 'POST' && req.headers()['content-type']?.includes('application/json')
      ? req.postDataJSON() : {};
    let json = { data: [], voices: [] };
    if (url.pathname === '/v1/models') json = { data: [...registered.values()] };
    if (url.pathname === '/v1/models/load') {
      loads.push(data);
      registered.set(data.id, { ...data, loaded: true, configured: false });
      json = { loaded: true };
    }
    if (url.pathname === '/v1/models/unload') {
      unloads.push(data.id);
      registered.delete(data.id);
      json = { unloaded: true };
    }
    if (url.pathname === '/v1/ui/models-root') {
      json = { models_root: 'models', default_models_root: 'models', is_default: true };
    }
    if (url.pathname === '/v1/ui/models/package-sizes') {
      json = { state: 'complete', message: '', data: packages };
    }
    if (url.pathname === '/v1/ui/path-status') {
      json = { exists: true, file: true, directory: false };
    }
    return route.fulfill({ json });
  });

  await page.goto(baseUrl);
  await page.getByRole('button', { name: /^Text to speech/ }).click();
  await page.locator('#model').selectOption('m5-tts');
  await page.locator('.studio-package-buttons button:not(:disabled), button.single-model-toggle:not(:disabled)').first().click();
  await page.waitForFunction(() => document.querySelector('.studio-model-status')?.textContent?.trim() === 'Resident');
  assert.equal(await page.locator('.reload-prompt').count(), 0, 'Loaded model with unchanged defaults must not show reload prompt');

  // Change a session-scoped parameter on a generic model (m5_tts)
  await page.locator('.studio-options > summary').filter({ hasText: 'Model parameters' }).click();
  await page.locator('#param-attention').selectOption('flash');
  await page.locator('.reload-prompt').waitFor({ state: 'visible' });
  assert(await page.locator('.reload-prompt').textContent().then((text) => text?.includes('Reload required to apply settings')));
  assert.equal(await page.locator('.reload-prompt button', { hasText: 'Reload model' }).count(), 1);
  console.log('PASS changing a session parameter while loaded shows reload prompt');

  // Reverting the setting clears the prompt without reloading
  await page.locator('#param-attention').selectOption('auto');
  await page.locator('.reload-prompt').waitFor({ state: 'hidden' });
  assert.equal(loads.length, 1, 'Reverting before reloading must not trigger an extra load');
  console.log('PASS reverting session parameter clears reload prompt');

  // Changing again and clicking Reload model reloads with updated session_options
  await page.locator('#param-attention').selectOption('flash');
  await page.locator('.reload-prompt button', { hasText: 'Reload model' }).click();
  await page.locator('.reload-prompt').waitFor({ state: 'hidden' });
  await page.waitForFunction(() => document.querySelector('.studio-model-status')?.textContent?.trim() === 'Resident');
  assert.equal(unloads.at(-1), 'm5-tts');
  assert.equal(loads.at(-1).session_options['m5_tts.attention'], 'flash');
  console.log('PASS Reload model button unloads and reloads with updated session options');

  // Switching to another model and back preserves the loaded session setting without a false prompt
  await page.locator('#model').selectOption('zonos2');
  await page.locator('#model').selectOption('m5-tts');
  await page.waitForFunction(() => document.querySelector('.studio-model-status')?.textContent?.trim() === 'Resident');
  assert.equal(await page.locator('.reload-prompt').count(), 0, 'Returning to a resident model must not show a false reload prompt');
  assert.equal(await page.locator('#param-attention').inputValue(), 'flash');
  console.log('PASS returning to resident model keeps its loaded session setting');

  // Changing a weight selector on a model panel (YuE2) prompts and reloads cleanly
  await page.getByRole('button', { name: /^Music \/ video generation/ }).click();
  await page.locator('#model').selectOption('yue2');
  await page.locator('.studio-package-buttons button:not(:disabled), button.single-model-toggle:not(:disabled)').first().click();
  await page.waitForFunction(() => document.querySelector('.studio-model-status')?.textContent?.trim() === 'Resident');
  const vaeOptions = await page.locator('#param-vae_gguf option').evaluateAll((opts) => opts.map((o) => o.value));
  assert(vaeOptions.length >= 2, 'YuE2 should expose multiple VAE weight choices');
  const currentVae = await page.locator('#param-vae_gguf').inputValue();
  const nextVae = vaeOptions.find((value) => value !== currentVae);
  await page.locator('#param-vae_gguf').selectOption(nextVae);
  await page.locator('.reload-prompt').waitFor({ state: 'visible' });
  await page.locator('.reload-prompt button', { hasText: 'Reload model' }).click();
  await page.locator('.reload-prompt').waitFor({ state: 'hidden' });
  assert.equal(loads.at(-1).id, 'yue2');
  assert.equal(loads.at(-1).session_options['yue2.vae_gguf'], nextVae);
  console.log('PASS changing weight setting in model panel prompts and reloads with new weight');

  await page.locator('#param-main_gguf').selectOption('yue2-3b-bf16.gguf');
  await page.locator('.reload-prompt button', { hasText: 'Reload model' }).click();
  await page.locator('.reload-prompt').waitFor({ state: 'hidden' });
  const loadsBeforeNavigation = loads.length;
  const unloadsBeforeNavigation = unloads.length;
  await page.getByRole('button', { name: /^Text to speech/ }).click();
  await page.getByRole('button', { name: /^Music \/ video generation/ }).click();
  assert.equal(await page.locator('#param-main_gguf').inputValue(), 'yue2-3b-bf16.gguf');
  assert.equal(await page.locator('.reload-prompt').count(), 0, 'Returning to resident YuE2 must retain its settings');
  assert.equal(loads.length, loadsBeforeNavigation);
  assert.equal(unloads.length, unloadsBeforeNavigation);
  await page.locator('#param-vae_gguf').selectOption(currentVae);
  await page.locator('.reload-prompt button', { hasText: 'Reload model' }).click();
  await page.locator('.reload-prompt').waitFor({ state: 'hidden' });
  assert.equal(loads.at(-1).session_options['yue2.model_gguf'], 'yue2-3b-bf16.gguf');
  console.log('PASS workflow return restores resident YuE2 settings in controls and load requests');

  await page.locator('#param-main_gguf').selectOption('yue2-3b-q8_0.gguf');
  await page.locator('.reload-prompt button', { hasText: 'Reload model' }).click();
  await page.locator('.reload-prompt').waitFor({ state: 'hidden' });
  const loadsBeforePackageChoice = loads.length;
  const unloadsBeforePackageChoice = unloads.length;
  await page.getByRole('button', { name: 'Models', exact: true }).click();
  await page.getByRole('searchbox', { name: 'Search models' }).fill('yue2');
  await page.locator('#package-yue2').selectOption('yue2_main_q4_0');
  await page.locator('.model-row').filter({ has: page.locator('#package-yue2') })
    .getByRole('button', { name: 'Open in Studio' }).click();
  assert.equal(await page.locator('#param-main_gguf').inputValue(), 'yue2-3b-q4_0.gguf');
  assert.equal(registered.get('yue2').session_options['yue2.model_gguf'], 'yue2-3b-q8_0.gguf');
  assert.equal(loads.length, loadsBeforePackageChoice, 'Opening a package must not load it');
  assert.equal(unloads.length, unloadsBeforePackageChoice, 'Opening a package must not unload its resident model');
  await page.locator('.reload-prompt').waitFor({ state: 'visible' });
  await page.locator('.reload-prompt button', { hasText: 'Reload model' }).click();
  await page.locator('.reload-prompt').waitFor({ state: 'hidden' });
  assert.equal(loads.at(-1).session_options['yue2.model_gguf'], 'yue2-3b-q4_0.gguf');
  console.log('PASS explicit Q4 package selection overrides resident Q8 in controls and load requests');

  // Mobile layout check while prompt is visible
  await page.locator('#param-vae_gguf').selectOption(nextVae);
  await page.locator('.reload-prompt').waitFor({ state: 'visible' });
  await page.setViewportSize({ width: 390, height: 844 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'Mobile layout must not overflow horizontally');
  assert.deepEqual(errors, [], 'Unexpected page errors in managed mode');
  console.log('PASS mobile layout fits with reload prompt visible and zero errors');
  await page.close();
}

async function runConfiguredScenario(context) {
  const errors = [];
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));

  const configuredModel = {
    id: 'm5-tts',
    family: 'm5_tts',
    task: 'tts',
    mode: 'offline',
    path: 'models/M5-TTS-GGUF/m5-tts-50m-customvoice-f16.gguf',
    loaded: true,
    configured: true,
    session_options: {}
  };

  await page.route('**/health', (route) => route.fulfill({
    json: { status: 'ok', backend: 'fixture', models: 1, ui: true, ui_management: false }
  }));
  await page.route('**/v1/**', async (route) => {
    const url = new URL(route.request().url());
    let json = { data: [], voices: [] };
    if (url.pathname === '/v1/models') json = { data: [configuredModel] };
    if (url.pathname === '/v1/ui/path-status') json = { exists: true, file: true, directory: false };
    return route.fulfill({ json });
  });

  await page.goto(baseUrl);
  await page.waitForFunction(() => document.querySelector('#model')?.value === 'm5-tts');
  assert.equal(await page.locator('.reload-prompt').count(), 0, 'Configured model with omitted defaults must not show a false reload prompt');

  await page.locator('.studio-options > summary').filter({ hasText: 'Model parameters' }).click();
  await page.locator('#param-attention').selectOption('flash');
  await page.locator('.reload-prompt').waitFor({ state: 'visible' });
  const text = await page.locator('.reload-prompt').textContent();
  assert(text?.includes('Update the server configuration to apply them.'), `Expected configured guidance, got: ${text}`);
  assert.equal(await page.locator('.reload-prompt button').count(), 0, 'Configured models must not offer an in-UI reload button');
  assert.deepEqual(errors, [], 'Unexpected page errors in configured mode');
  console.log('PASS configured server model shows server-config reload guidance without a reload button');
  await page.close();
}

try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await runManagedScenario(context);
  await runConfiguredScenario(context);
} finally {
  await browser.close();
}
