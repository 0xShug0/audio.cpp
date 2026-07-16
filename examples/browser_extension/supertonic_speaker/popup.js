"use strict";

const fields = {
  serverUrl: document.getElementById("serverUrl"),
  modelId: document.getElementById("modelId"),
  voice: document.getElementById("voice"),
  sampleRate: document.getElementById("sampleRate"),
  chunkChars: document.getElementById("chunkChars"),
  steps: document.getElementById("steps"),
  textInput: document.getElementById("textInput"),
};

const buttons = {
  selection: document.getElementById("speakSelection"),
  page: document.getElementById("speakPage"),
  stop: document.getElementById("stop"),
  offline: document.getElementById("openOffline"),
};

const statusEl = document.getElementById("status");
const backendStatusEl = document.getElementById("backendStatus");
const metrics = {
  chunk: document.getElementById("metricChunk"),
  firstDelta: document.getElementById("metricFirstDelta"),
  serverTtft: document.getElementById("metricServerTtft"),
  deltas: document.getElementById("metricDeltas"),
};
let abortController = null;
let audioContext = null;
let nextAudioTime = 0;
let preloadedText = "";
let playbackTimer = null;

function setStatus(text) {
  statusEl.textContent = text;
}

async function refreshBackendStatus() {
  const url = fields.serverUrl.value.replace(/\/+$/, "");
  backendStatusEl.textContent = "checking";
  try {
    const response = await fetch(`${url}/health`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const health = await response.json();
    backendStatusEl.textContent = health.backend || "unknown";
  } catch (error) {
    backendStatusEl.textContent = "offline";
  }
}

function loadSettings() {
  for (const [key, input] of Object.entries(fields)) {
    if (key === "textInput") {
      continue;
    }
    const value = localStorage.getItem(key);
    if (value !== null) {
      input.value = value;
    }
    input.addEventListener("change", () => localStorage.setItem(key, input.value));
  }
  fields.serverUrl.addEventListener("change", () => {
    refreshBackendStatus();
  });
}

function resetMetrics(totalChunks = 0) {
  if (playbackTimer !== null) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }
  metrics.chunk.textContent = totalChunks > 0 ? `0/${totalChunks}` : "-";
  metrics.firstDelta.textContent = "-";
  metrics.serverTtft.textContent = "-";
  metrics.deltas.textContent = "-";
}

function formatMs(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${Math.round(value)} ms`;
}

function formatSeconds(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${Math.max(0, value).toFixed(1)}s`;
}

function settings() {
  return {
    serverUrl: fields.serverUrl.value.replace(/\/+$/, ""),
    modelId: fields.modelId.value.trim(),
    voice: fields.voice.value.trim(),
    sampleRate: Number(fields.sampleRate.value),
    chunkChars: Number(fields.chunkChars.value),
    steps: Number(fields.steps.value),
  };
}

async function activeTabText(selectionOnly) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || tab.id === undefined) {
    throw new Error("No active tab.");
  }
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    args: [selectionOnly],
    func: (onlySelection) => {
      const selected = window.getSelection ? window.getSelection().toString().trim() : "";
      if (onlySelection) {
        return selected;
      }
      return selected || document.body.innerText.trim();
    },
  });
  return (result || "").replace(/\s+/g, " ").trim();
}

function directText() {
  return (fields.textInput?.value || preloadedText || "").replace(/\s+/g, " ").trim();
}

function splitText(text, maxChars) {
  if (!Number.isFinite(maxChars) || maxChars <= 0) {
    throw new Error("Chunk chars must be positive.");
  }
  const sentences = text.match(/[^.!?。！？]+[.!?。！？]?/g) || [text];
  const chunks = [];
  let current = "";
  for (const raw of sentences) {
    const sentence = raw.trim();
    if (!sentence) {
      continue;
    }
    if (!current) {
      current = sentence;
    } else if ((current + " " + sentence).length <= maxChars) {
      current += " " + sentence;
    } else {
      chunks.push(current);
      current = sentence;
    }
    while (current.length > maxChars) {
      let splitAt = current.lastIndexOf(" ", maxChars);
      if (splitAt <= 0) {
        splitAt = maxChars;
      }
      chunks.push(current.slice(0, splitAt).trim());
      current = current.slice(splitAt).trim();
    }
  }
  if (current) {
    chunks.push(current);
  }
  return chunks;
}

function decodeBase64Pcm16(base64) {
  const binary = atob(base64);
  const samples = new Float32Array(binary.length / 2);
  for (let i = 0; i < samples.length; ++i) {
    const lo = binary.charCodeAt(i * 2);
    const hi = binary.charCodeAt(i * 2 + 1);
    let value = lo | (hi << 8);
    if (value & 0x8000) {
      value -= 0x10000;
    }
    samples[i] = Math.max(-1, Math.min(1, value / 32768));
  }
  return samples;
}

async function ensureAudioContext(sampleRate) {
  if (!audioContext || audioContext.sampleRate !== sampleRate) {
    if (audioContext) {
      await audioContext.close();
    }
    audioContext = new AudioContext({ sampleRate });
    nextAudioTime = audioContext.currentTime;
  }
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
}

function playPcm(samples, sampleRate) {
  const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  const startAt = Math.max(audioContext.currentTime + 0.02, nextAudioTime);
  source.start(startAt);
  nextAudioTime = startAt + buffer.duration;
  return buffer.duration;
}

function playbackRemainingSeconds() {
  if (!audioContext) {
    return 0;
  }
  return Math.max(0, nextAudioTime - audioContext.currentTime);
}

function startPlaybackCountdown() {
  if (playbackTimer !== null) {
    clearInterval(playbackTimer);
  }
  playbackTimer = setInterval(() => {
    const remaining = playbackRemainingSeconds();
    metrics.deltas.textContent = formatSeconds(remaining);
    if (remaining <= 0.05) {
      clearInterval(playbackTimer);
      playbackTimer = null;
      metrics.deltas.textContent = "done";
      setStatus("Playback done.");
    } else {
      setStatus(`Playing, ${formatSeconds(remaining)} left.`);
    }
  }, 250);
}

function parseSseLines(buffer, onEvent) {
  let offset = 0;
  while (true) {
    const newline = buffer.indexOf("\n", offset);
    if (newline < 0) {
      return buffer.slice(offset);
    }
    const line = buffer.slice(offset, newline).trim();
    offset = newline + 1;
    if (!line.startsWith("data:")) {
      continue;
    }
    const data = line.slice(5).trim();
    if (data && data !== "[DONE]") {
      onEvent(JSON.parse(data));
    }
  }
}

async function speakChunk(text, index, total, opts) {
  const startMs = performance.now();
  const response = await fetch(`${opts.serverUrl}/v1/audio/speech`, {
    method: "POST",
    signal: abortController.signal,
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
    },
    body: JSON.stringify({
      model: opts.modelId,
      input: text,
      voice: opts.voice || undefined,
      response_format: "pcm",
      stream_format: "sse",
      num_inference_steps: opts.steps,
    }),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }
  if (!response.body) {
    throw new Error("Streaming response has no body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let deltas = 0;
  let firstDeltaMs = null;
  let serverTtftMs = null;
  let queuedSeconds = 0;
  metrics.chunk.textContent = `${index + 1}/${total}`;
  metrics.firstDelta.textContent = "waiting";
  metrics.serverTtft.textContent = "waiting";
  metrics.deltas.textContent = "0";
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSseLines(buffer, (event) => {
      if (event.type === "speech.audio.delta") {
        deltas += 1;
        if (deltas === 1) {
          firstDeltaMs = performance.now() - startMs;
          metrics.firstDelta.textContent = formatMs(firstDeltaMs);
        }
        metrics.deltas.textContent = String(deltas);
        queuedSeconds += playPcm(decodeBase64Pcm16(event.audio), opts.sampleRate);
        metrics.deltas.textContent = `${formatSeconds(playbackRemainingSeconds())} queued`;
        setStatus(`Speaking chunk ${index + 1}/${total}, delta ${deltas}.`);
      } else if (event.type === "speech.audio.done") {
        serverTtftMs = event.timing?.ttft_ms ?? null;
        metrics.serverTtft.textContent = formatMs(serverTtftMs);
      } else if (event.type === "error") {
        throw new Error(event.error?.message || "Server streaming error.");
      }
    });
  }
  return {
    deltas,
    firstDeltaMs,
    serverTtftMs,
    queuedSeconds,
  };
}

async function speakText(text) {
  if (abortController) {
    abortController.abort();
  }
  abortController = new AbortController();
  const opts = settings();
  if (!opts.modelId) {
    throw new Error("Model id is required.");
  }
  await ensureAudioContext(opts.sampleRate);
  if (!text) {
    throw new Error("No text to speak.");
  }
  const chunks = splitText(text, opts.chunkChars);
  resetMetrics(chunks.length);
  setStatus(`Sending ${chunks.length} text chunks.`);
  for (let i = 0; i < chunks.length; ++i) {
    if (abortController.signal.aborted) {
      break;
    }
    const stat = await speakChunk(chunks[i], i, chunks.length, opts);
    metrics.chunk.textContent = `${i + 1}/${chunks.length} queued`;
    metrics.firstDelta.textContent = formatMs(stat.firstDeltaMs);
    metrics.serverTtft.textContent = formatMs(stat.serverTtftMs);
    metrics.deltas.textContent = `${formatSeconds(playbackRemainingSeconds())} queued`;
  }
  setStatus(`Requests done. Playing, ${formatSeconds(playbackRemainingSeconds())} left.`);
  startPlaybackCountdown();
}

async function speak(selectionOnly) {
  const typed = directText();
  const text = typed || await activeTabText(selectionOnly);
  if (fields.textInput) {
    fields.textInput.value = text;
  }
  await speakText(text);
}

async function run(action) {
  buttons.selection.disabled = true;
  buttons.page.disabled = true;
  try {
    await action();
  } catch (error) {
    if (error.name === "AbortError") {
      setStatus("Stopped.");
    } else {
      setStatus(error.message || String(error));
    }
  } finally {
    abortController = null;
    buttons.selection.disabled = false;
    buttons.page.disabled = false;
  }
}

buttons.selection.addEventListener("click", () => run(() => speak(true)));
buttons.page.addEventListener("click", () => run(() => speak(false)));
buttons.stop.addEventListener("click", async () => {
  if (abortController) {
    abortController.abort();
  }
  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }
  nextAudioTime = 0;
  setStatus("Stopped.");
});
buttons.offline?.addEventListener("click", () => {
  chrome.tabs.create({ url: chrome.runtime.getURL("offline.html") });
});

async function loadContextPayload() {
  const params = new URLSearchParams(location.search);
  const id = params.get("id");
  if (!id) {
    return;
  }
  const payload = (await chrome.storage.session.get(id))[id];
  await chrome.storage.session.remove(id);
  if (!payload?.text) {
    return;
  }
  preloadedText = payload.text;
  if (fields.textInput) {
    fields.textInput.value = payload.text;
  }
  setStatus(`Loaded ${payload.source || "context"} text.`);
  if (payload.autoStart) {
    await run(() => speakText(payload.text));
  }
}

loadSettings();
resetMetrics();
refreshBackendStatus();
loadContextPayload().catch((error) => setStatus(error.message || String(error)));
