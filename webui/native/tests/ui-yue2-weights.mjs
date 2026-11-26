// Browser regressions with deterministic HTTP fixtures verifying that YuE2
// only lists downloaded main/VAE weights in UI-managed mode, selects valid
// downloaded defaults, handles missing components cleanly, preserves resident
// weights, and retains all static choices in server-configured mode.
import { createRequire } from 'node:module';
import path from 'node:path';
import assert from 'node:assert/strict';

const require = createRequire(process.env.UI_BROWSER_TOOLS
  ? path.resolve(process.env.UI_BROWSER_TOOLS, 'package.json') : import.meta.url);
const { chromium } = require('playwright');

const baseUrl = process.env.UI_BASE_URL || 'http://127.0.0.1:8080';
const browser = await chromium.launch({ channel: process.env.UI_BROWSER_CHANNEL || 'chrome', headless: true });

function yue2Packages(installedIds) {
  const installed = new Set(installedIds);
  const defs = [
    ['yue2_main_q8_0', { 'yue2.model_gguf': 'yue2-3b-q8_0.gguf' }],
    ['yue2_main_q4_0', { 'yue2.model_gguf': 'yue2-3b-q4_0.gguf' }],
    ['yue2_main_bf16', { 'yue2.model_gguf': 'yue2-3b-bf16.gguf' }],
    ['yue2_vae_f16', { 'yue2.vae_gguf': 'yue2-vae-f16.gguf' }],
    ['yue2_vae_f32', { 'yue2.vae_gguf': 'yue2-vae-f32.gguf' }]
  ];
  return defs.map(([id, session_options]) => ({
    id,
    installed: installed.has(id),
    size_bytes: 100,
    state: 'ok',
    load_target: installed.has(id)
      ? { family: 'yue2', model_path: 'models/Yue2-3B-GGUF', load_options: {}, session_options }
      : null
  }));
}

async function createManagedPage(context, installedIds, initialModels = []) {
  const registered = new Map(initialModels.map((model) => [model.id, model]));
  const loads = [];
  const errors = [];
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));

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
      registered.delete(data.id);
      json = { unloaded: true };
    }
    if (url.pathname === '/v1/ui/models-root') {
      json = { models_root: 'models', default_models_root: 'models', is_default: true };
    }
    if (url.pathname === '/v1/ui/models/package-sizes') {
      json = { state: 'complete', message: '', data: yue2Packages(installedIds) };
    }
    if (url.pathname === '/v1/ui/path-status') {
      json = { exists: true, file: false, directory: true };
    }
    return route.fulfill({ json });
  });

  await page.goto(baseUrl);
  await page.getByRole('button', { name: /^Music \/ video generation/ }).click();
  await page.locator('#model').selectOption('yue2');
  await page.locator('#param-main_gguf').waitFor({ state: 'visible' });
  return { page, loads, errors };
}

async function optionValues(page, selector) {
  return page.locator(`${selector} option`).evaluateAll((opts) => opts.map((opt) => opt.value));
}

try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });

  // 1. Partial downloads: only Q4_0 + VAE F32 downloaded (neither static default installed)
  {
    const { page, loads, errors } = await createManagedPage(context, ['yue2_main_q4_0', 'yue2_vae_f32']);
    assert.deepEqual(await optionValues(page, '#param-main_gguf'), ['yue2-3b-q4_0.gguf']);
    assert.deepEqual(await optionValues(page, '#param-vae_gguf'), ['yue2-vae-f32.gguf']);
    assert.equal(await page.locator('#param-main_gguf').inputValue(), 'yue2-3b-q4_0.gguf');
    assert.equal(await page.locator('#param-vae_gguf').inputValue(), 'yue2-vae-f32.gguf');
    await page.locator('button.single-model-toggle:not(:disabled)').click();
    await page.waitForFunction(() => document.querySelector('.studio-model-status')?.textContent?.trim() === 'Resident');
    assert.equal(loads.at(-1).session_options['yue2.model_gguf'], 'yue2-3b-q4_0.gguf');
    assert.equal(loads.at(-1).session_options['yue2.vae_gguf'], 'yue2-vae-f32.gguf');
    assert.equal(await page.locator('.reload-prompt').count(), 0);
    assert.deepEqual(errors, []);
    console.log('PASS partial YuE2 downloads show only downloaded weights and load valid defaults');
    await page.close();
  }

  // 2. Multiple downloaded variants: prefers declared default when installed, preserves Open in Studio choice
  {
    const { page, errors } = await createManagedPage(context, ['yue2_main_q4_0', 'yue2_main_bf16', 'yue2_vae_f16', 'yue2_vae_f32']);
    assert.deepEqual(await optionValues(page, '#param-main_gguf'), ['yue2-3b-q4_0.gguf', 'yue2-3b-bf16.gguf']);
    assert.deepEqual(await optionValues(page, '#param-vae_gguf'), ['yue2-vae-f16.gguf', 'yue2-vae-f32.gguf']);
    assert.equal(await page.locator('#param-main_gguf').inputValue(), 'yue2-3b-q4_0.gguf');
    assert.equal(await page.locator('#param-vae_gguf').inputValue(), 'yue2-vae-f16.gguf');

    await page.getByRole('button', { name: 'Models', exact: true }).click();
    const yue2Row = page.locator('.model-row').filter({ hasText: 'Yue2 3B' });
    await yue2Row.locator('select').selectOption('yue2_vae_f32');
    await yue2Row.getByRole('button', { name: 'Open in Studio', exact: true }).click();
    await page.locator('#param-vae_gguf').waitFor({ state: 'visible' });
    assert.equal(await page.locator('#param-vae_gguf').inputValue(), 'yue2-vae-f32.gguf');
    assert.equal(await page.locator('#param-main_gguf').inputValue(), 'yue2-3b-q4_0.gguf');
    assert.deepEqual(errors, []);
    console.log('PASS multiple YuE2 downloads prefer valid defaults and keep Open in Studio VAE choice');
    await page.close();
  }

  // 3. Missing component: only Main Q4_0 downloaded, no VAE downloaded
  {
    const { page, errors } = await createManagedPage(context, ['yue2_main_q4_0']);
    assert.deepEqual(await optionValues(page, '#param-main_gguf'), ['yue2-3b-q4_0.gguf']);
    assert.equal(await page.locator('#param-main_gguf').isDisabled(), false);
    assert.equal(await page.locator('#param-vae_gguf').isDisabled(), true);
    assert.deepEqual(await optionValues(page, '#param-vae_gguf'), ['']);
    assert.equal(
      await page.locator('#param-vae_gguf option').first().textContent(),
      'No downloaded weights'
    );
    assert.deepEqual(errors, []);
    console.log('PASS missing YuE2 component shows disabled placeholder instead of uninstalled weight');
    await page.close();
  }

  // 4. Resident weight loaded outside installed package list is preserved without false reload prompt
  {
    const resident = {
      id: 'yue2',
      family: 'yue2',
      task: 'gen',
      mode: 'offline',
      path: 'models/Yue2-3B-GGUF',
      loaded: true,
      configured: false,
      session_options: {
        'yue2.model_gguf': 'yue2-3b-bf16.gguf',
        'yue2.vae_gguf': 'yue2-vae-f16.gguf'
      }
    };
    const { page, errors } = await createManagedPage(context, ['yue2_main_q4_0', 'yue2_vae_f32'], [resident]);
    assert.deepEqual(await optionValues(page, '#param-main_gguf'), ['yue2-3b-q4_0.gguf', 'yue2-3b-bf16.gguf']);
    assert.deepEqual(await optionValues(page, '#param-vae_gguf'), ['yue2-vae-f32.gguf', 'yue2-vae-f16.gguf']);
    assert.equal(await page.locator('#param-main_gguf').inputValue(), 'yue2-3b-bf16.gguf');
    assert.equal(await page.locator('#param-vae_gguf').inputValue(), 'yue2-vae-f16.gguf');
    assert.equal(await page.locator('.reload-prompt').count(), 0);
    assert.deepEqual(errors, []);
    console.log('PASS resident YuE2 weights remain visible and selected without a false reload prompt');
    await page.close();
  }

  // 5. Server-configured mode keeps all static choices
  {
    const errors = [];
    const page = await context.newPage();
    page.on('pageerror', (error) => errors.push(error.message));
    await page.route('**/health', (route) => route.fulfill({
      json: { status: 'ok', backend: 'fixture', models: 1, ui: true, ui_management: false }
    }));
    await page.route('**/v1/**', async (route) => {
      const url = new URL(route.request().url());
      let json = { data: [], voices: [] };
      if (url.pathname === '/v1/models') {
        json = {
          data: [{
            id: 'yue2',
            family: 'yue2',
            task: 'gen',
            mode: 'offline',
            path: 'models/Yue2-3B-GGUF',
            loaded: true,
            configured: true,
            session_options: {}
          }]
        };
      }
      if (url.pathname === '/v1/ui/path-status') json = { exists: true, file: false, directory: true };
      return route.fulfill({ json });
    });
    await page.goto(baseUrl);
    await page.locator('#param-main_gguf').waitFor({ state: 'visible' });
    assert.deepEqual(
      await optionValues(page, '#param-main_gguf'),
      ['yue2-3b-q8_0.gguf', 'yue2-3b-q4_0.gguf', 'yue2-3b-bf16.gguf']
    );
    assert.deepEqual(
      await optionValues(page, '#param-vae_gguf'),
      ['yue2-vae-f16.gguf', 'yue2-vae-f32.gguf']
    );
    assert.deepEqual(errors, []);
    console.log('PASS configured server mode retains all static YuE2 weight choices');
    await page.close();
  }
} finally {
  await browser.close();
}
