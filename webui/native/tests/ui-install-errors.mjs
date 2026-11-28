// Browser regressions verifying that download/install failures display the
// actual failure reason clearly (never masked by "0% · 0 B / ..." or a
// spinning progress bar), remain visible in the top Downloads summary even
// when another package variant is selected, can be dismissed and retried,
// surface gated/error inventory messages, and log background failures.
import { createRequire } from 'node:module';
import path from 'node:path';
import assert from 'node:assert/strict';

const require = createRequire(process.env.UI_BROWSER_TOOLS
  ? path.resolve(process.env.UI_BROWSER_TOOLS, 'package.json') : import.meta.url);
const { chromium } = require('playwright');

const baseUrl = process.env.UI_BASE_URL || 'http://127.0.0.1:8080';
const browser = await chromium.launch({ channel: process.env.UI_BROWSER_CHANNEL || 'chrome', headless: true });

async function openModelsPage(options = {}) {
  const {
    packageSizes = [],
    initialJobs = [],
    onInstall = null,
    onInstallStatus = null
  } = options;
  const errors = [];
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('dialog', (dialog) => dialog.accept());

  let statusPollCount = 0;
  await page.route('**/health', (route) => route.fulfill({
    json: { status: 'ok', backend: 'fixture', models: 0, ui: true, ui_management: true }
  }));
  await page.route('**/v1/**', async (route) => {
    const req = route.request();
    const url = new URL(req.url());
    const data = req.method() === 'POST' && req.headers()['content-type']?.includes('application/json')
      ? req.postDataJSON() : {};
    let json = { data: [], voices: [] };
    if (url.pathname === '/v1/ui/models-root') {
      json = { models_root: 'models', default_models_root: 'models', is_default: true };
    } else if (url.pathname === '/v1/ui/models/package-sizes') {
      json = { state: 'complete', message: '', data: packageSizes };
    } else if (url.pathname === '/v1/ui/models/install-status') {
      statusPollCount += 1;
      json = { data: onInstallStatus ? onInstallStatus(statusPollCount) : initialJobs };
    } else if (url.pathname === '/v1/ui/models/install') {
      if (onInstall) {
        const result = onInstall(data);
        if (result.status && result.status >= 400) {
          return route.fulfill({ status: result.status, json: result.json });
        }
        json = result;
      }
    } else if (url.pathname === '/v1/ui/path-status') {
      json = { exists: false, file: false, directory: false };
    }
    return route.fulfill({ json });
  });

  await page.goto(baseUrl);
  await page.getByRole('button', { name: 'Models', exact: true }).click();
  await page.locator('.library').waitFor({ state: 'visible' });
  return { context, page, errors };
}

try {
  // 1. Background download fails with total_bytes > 0 (e.g. gated 403):
  //    must show "Download failed" (not "0% · ..."), no <progress> bar,
  //    show exact failure reason in both model row and Downloads summary,
  //    stay visible if package dropdown changes, and respect Dismiss across polls.
  {
    const hfReason = 'org/yue2/yue2-3b-q8_0.gguf requires accepted Hugging Face access and a valid HF token (set HF_TOKEN; HTTP 403)';
    let phase = 'idle';
    let finishTs = 1700000001;
    const { context, page, errors } = await openModelsPage({
      packageSizes: [
        {
          id: 'yue2_main_q8_0', installed: false, size_bytes: 1932735283, state: 'ok', message: '',
          load_target: { family: 'yue2', model_path: 'models/Yue2-3B-GGUF', load_options: {}, session_options: { 'yue2.model_gguf': 'yue2-3b-q8_0.gguf' } }
        },
        {
          id: 'yue2_main_q4_0', installed: false, size_bytes: 1073741824, state: 'ok', message: '',
          load_target: { family: 'yue2', model_path: 'models/Yue2-3B-GGUF', load_options: {}, session_options: { 'yue2.model_gguf': 'yue2-3b-q4_0.gguf' } }
        }
      ],
      onInstall: (body) => {
        phase = 'failed';
        return {
          id: body.id,
          state: 'queued',
          message: 'Waiting for model preparation worker',
          exit_code: -1,
          downloaded_bytes: 0,
          total_bytes: 1932735283,
          progress_percent: 0,
          started_at_ms: 1700000000,
          finished_at_ms: 0
        };
      },
      onInstallStatus: () => phase === 'failed' ? [{
        id: 'yue2_main_q8_0',
        state: 'failed',
        message: hfReason,
        exit_code: -1,
        downloaded_bytes: 0,
        total_bytes: 1932735283,
        progress_percent: 0,
        started_at_ms: 1700000000,
        finished_at_ms: finishTs
      }] : []
    });

    const yue2Row = page.locator('.model-row').filter({ hasText: 'Yue2 3B' });
    await yue2Row.getByRole('button', { name: 'Download', exact: true }).click();

    const rowAlert = yue2Row.locator('.job.failed[role="alert"]');
    await rowAlert.waitFor({ state: 'visible' });
    const statusText = await rowAlert.locator('.job-status').textContent();
    assert.equal(statusText?.trim(), 'Download failed');
    assert.equal(await rowAlert.locator('progress').count(), 0, 'failed job must not show a progress bar');
    assert.equal((await rowAlert.locator('.job-message').textContent())?.trim(), hfReason);

    // Top Downloads summary shows the failure too and remains visible even after switching package selection.
    const downloadsPanel = page.locator('details.downloads');
    await downloadsPanel.waitFor({ state: 'visible' });
    assert.ok((await downloadsPanel.textContent())?.includes(hfReason));
    assert.ok((await page.locator('.download-count.failed').textContent())?.includes('1 failed download'));

    await yue2Row.locator('select').selectOption('yue2_main_q4_0');
    assert.equal(await yue2Row.locator('.job.failed').count(), 0);
    assert.ok((await downloadsPanel.textContent())?.includes(hfReason),
      'top Downloads panel must retain failed reason when another package is selected');

    // Dismissing hides the failure even if install-status still returns the same finished_at_ms.
    await downloadsPanel.getByRole('button', { name: 'Dismiss', exact: true }).click();
    assert.equal(await page.locator('details.downloads').count(), 0);

    // Retrying clears the dismissal and surfaces a new failure with a new finished_at_ms.
    await yue2Row.locator('select').selectOption('yue2_main_q8_0');
    finishTs = 1700000002;
    await yue2Row.getByRole('button', { name: 'Download', exact: true }).click();
    await rowAlert.waitFor({ state: 'visible' });
    assert.equal((await rowAlert.locator('.job-message').textContent())?.trim(), hfReason);

    assert.deepEqual(errors, []);
    console.log('PASS failed download shows actual reason, hides progress bar, survives dropdown switch, and dismisses/retries cleanly');
    await context.close();
  }

  // 2. Partial download fails midway (e.g. disk full): shows downloaded/total alongside "Download failed"
  //    and displays the exact OS error reason.
  {
    const diskFullReason = 'could not write models/.Yue2-3B-GGUF.123/yue2-3b-q8_0.gguf: No space left on device';
    const { context, page, errors } = await openModelsPage({
      packageSizes: [
        {
          id: 'yue2_main_q8_0', installed: false, size_bytes: 2147483648, state: 'ok', message: '',
          load_target: { family: 'yue2', model_path: 'models/Yue2-3B-GGUF', load_options: {}, session_options: { 'yue2.model_gguf': 'yue2-3b-q8_0.gguf' } }
        }
      ],
      initialJobs: [{
        id: 'yue2_main_q8_0',
        state: 'failed',
        message: diskFullReason,
        exit_code: -1,
        downloaded_bytes: 536870912,
        total_bytes: 2147483648,
        progress_percent: 25,
        started_at_ms: 1700000000,
        finished_at_ms: 1700000010
      }]
    });

    const yue2Row = page.locator('.model-row').filter({ hasText: 'Yue2 3B' });
    const rowAlert = yue2Row.locator('.job.failed[role="alert"]');
    await rowAlert.waitFor({ state: 'visible' });
    const statusText = (await rowAlert.locator('.job-status').textContent())?.trim() || '';
    assert.ok(statusText.startsWith('Download failed · 512.0 MB / 2.0 GB'),
      `expected partial failure label, got: ${statusText}`);
    assert.equal((await rowAlert.locator('.job-message').textContent())?.trim(), diskFullReason);
    assert.equal(await rowAlert.locator('progress').count(), 0);
    assert.deepEqual(errors, []);
    console.log('PASS partial download failure shows byte progress with Download failed and OS reason');
    await context.close();
  }

  // 3. Synchronous POST /v1/ui/models/install failure and gated/error size notes display actual reasons.
  {
    const gatedNote = 'Hugging Face access and a valid token are required';
    const sizeErr = 'could not connect to huggingface.co (connection failed)';
    const syncErr = 'some package files already exist in: models/Yue2-3B-GGUF (use overwrite)';
    const { context, page, errors } = await openModelsPage({
      packageSizes: [
        {
          id: 'yue2_main_q8_0', installed: false, size_bytes: null, state: 'gated', message: gatedNote,
          load_target: { family: 'yue2', model_path: 'models/Yue2-3B-GGUF', load_options: {}, session_options: { 'yue2.model_gguf': 'yue2-3b-q8_0.gguf' } }
        },
        {
          id: 'yue2_main_q4_0', installed: false, size_bytes: null, state: 'error', message: sizeErr,
          load_target: { family: 'yue2', model_path: 'models/Yue2-3B-GGUF', load_options: {}, session_options: { 'yue2.model_gguf': 'yue2-3b-q4_0.gguf' } }
        }
      ],
      onInstall: () => ({ status: 400, json: { error: { message: syncErr } } })
    });

    const yue2Row = page.locator('.model-row').filter({ hasText: 'Yue2 3B' });
    assert.ok((await yue2Row.locator('.package .size-warning').textContent())?.includes(gatedNote),
      'gated package shows its access requirement');

    await yue2Row.locator('select').selectOption('yue2_main_q4_0');
    assert.ok((await yue2Row.locator('.package .size-warning').textContent())?.includes(sizeErr),
      'size-check error shows actual reason');

    await yue2Row.getByRole('button', { name: 'Download', exact: true }).click();
    const rowAlert = yue2Row.locator('.job.failed[role="alert"]');
    await rowAlert.waitFor({ state: 'visible' });
    assert.equal((await rowAlert.locator('.job-message').textContent())?.trim(), syncErr);
    assert.deepEqual(errors, []);
    console.log('PASS gated/error size notes and synchronous install start errors display actual reasons');
    await context.close();
  }
} finally {
  await browser.close();
}
