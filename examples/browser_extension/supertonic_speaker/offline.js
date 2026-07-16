"use strict";

const fields = {
  serverUrl: document.getElementById("serverUrl"),
  modelId: document.getElementById("modelId"),
  voice: document.getElementById("voice"),
  chunkChars: document.getElementById("chunkChars"),
  steps: document.getElementById("steps"),
  textFile: document.getElementById("textFile"),
  textInput: document.getElementById("textInput"),
  normalizeLatin: document.getElementById("normalizeLatin"),
};

const buttons = {
  start: document.getElementById("start"),
  stop: document.getElementById("stop"),
  download: document.getElementById("download"),
};

const metrics = {
  progress: document.getElementById("metricProgress"),
  elapsed: document.getElementById("metricElapsed"),
  eta: document.getElementById("metricEta"),
  rtf: document.getElementById("metricRtf"),
  audio: document.getElementById("metricAudio"),
  wall: document.getElementById("metricWall"),
};

const backendStatusEl = document.getElementById("backendStatus");
const statusEl = document.getElementById("status");
const textStatsEl = document.getElementById("textStats");
const progressBar = document.getElementById("progressBar");
const downloadsEl = document.getElementById("downloads");

let abortController = null;
let partUrls = [];
let timer = null;

const kMaxPartSeconds = 30 * 60;

function setStatus(text) {
  statusEl.textContent = text;
}

function settings() {
  return {
    serverUrl: fields.serverUrl.value.replace(/\/+$/, ""),
    modelId: fields.modelId.value.trim(),
    voice: fields.voice.value,
    chunkChars: Number(fields.chunkChars.value),
    steps: Number(fields.steps.value),
  };
}

function saveSettings() {
  for (const [key, input] of Object.entries(fields)) {
    if (key === "textFile" || key === "textInput") {
      continue;
    }
    localStorage.setItem(`offline.${key}`, input.type === "checkbox" ? String(input.checked) : input.value);
  }
}

function loadSettings() {
  for (const [key, input] of Object.entries(fields)) {
    if (key === "textFile" || key === "textInput") {
      continue;
    }
    const value = localStorage.getItem(`offline.${key}`);
    if (value !== null) {
      if (input.type === "checkbox") {
        input.checked = value === "true";
      } else {
        input.value = value;
      }
    }
    input.addEventListener("change", () => {
      saveSettings();
      if (key === "serverUrl") {
        refreshBackendStatus();
      }
    });
  }
}

async function refreshBackendStatus() {
  backendStatusEl.textContent = "checking";
  try {
    const response = await fetch(`${settings().serverUrl}/health`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const health = await response.json();
    backendStatusEl.textContent = health.backend || "unknown";
  } catch (error) {
    backendStatusEl.textContent = "offline";
  }
}

function cleanText(text) {
  return text.replace(/\s+/g, " ").trim();
}

function normalizeLatinText(text) {
  const replacements = new Map([
    ["\u2018", "'"],
    ["\u2019", "'"],
    ["\u201A", "'"],
    ["\u201B", "'"],
    ["\u201C", "\""],
    ["\u201D", "\""],
    ["\u201E", "\""],
    ["\u201F", "\""],
    ["\u2013", "-"],
    ["\u2014", "-"],
    ["\u2015", "-"],
    ["\u2212", "-"],
    ["\u2026", "..."],
    ["\u00A0", " "],
    ["\u00AD", ""],
    ["\u00C6", "AE"],
    ["\u00E6", "ae"],
    ["\u0152", "OE"],
    ["\u0153", "oe"],
    ["\u00DF", "ss"],
  ]);
  let changed = 0;
  let out = "";
  for (const ch of text) {
    const mapped = replacements.has(ch)
      ? replacements.get(ch)
      : ch.normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
    if (mapped !== ch) {
      changed += 1;
    }
    out += mapped;
  }
  return { text: out, changed };
}

function updateTextStats() {
  const text = cleanText(fields.textInput.value);
  const words = text ? text.match(/\S+/g).length : 0;
  textStatsEl.textContent = `${words.toLocaleString()} words · ${text.length.toLocaleString()} chars`;
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

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) {
    return "-";
  }
  const value = Math.max(0, seconds);
  if (value < 60) {
    return `${value.toFixed(1)}s`;
  }
  const minutes = Math.floor(value / 60);
  const secs = Math.round(value % 60);
  if (minutes < 60) {
    return `${minutes}m ${secs}s`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function resetMetrics() {
  metrics.progress.textContent = "-";
  metrics.elapsed.textContent = "-";
  metrics.eta.textContent = "-";
  metrics.rtf.textContent = "-";
  metrics.audio.textContent = "-";
  metrics.wall.textContent = "-";
  progressBar.style.width = "0%";
}

function clearResultUrls() {
  for (const item of partUrls) {
    URL.revokeObjectURL(item.url);
  }
  partUrls = [];
  downloadsEl.replaceChildren();
}

function parseWav(buffer) {
  const view = new DataView(buffer);
  const tag = (offset, length) => String.fromCharCode(...new Uint8Array(buffer, offset, length));
  if (tag(0, 4) !== "RIFF" || tag(8, 4) !== "WAVE") {
    throw new Error("Server response is not WAV.");
  }
  let offset = 12;
  let sampleRate = 0;
  let channels = 0;
  let bitsPerSample = 0;
  let pcm = null;
  while (offset + 8 <= buffer.byteLength) {
    const id = tag(offset, 4);
    const size = view.getUint32(offset + 4, true);
    const dataOffset = offset + 8;
    if (id === "fmt ") {
      const format = view.getUint16(dataOffset, true);
      channels = view.getUint16(dataOffset + 2, true);
      sampleRate = view.getUint32(dataOffset + 4, true);
      bitsPerSample = view.getUint16(dataOffset + 14, true);
      if (format !== 1 || bitsPerSample !== 16) {
        throw new Error("Only PCM16 WAV output is supported by this demo.");
      }
    } else if (id === "data") {
      pcm = new Int16Array(buffer.slice(dataOffset, dataOffset + size));
    }
    offset = dataOffset + size + (size % 2);
  }
  if (!pcm || sampleRate <= 0 || channels <= 0) {
    throw new Error("WAV missing fmt or data chunk.");
  }
  return { sampleRate, channels, pcm };
}

function wavBlob(chunks, sampleRate, channels) {
  const sampleCount = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const dataBytes = sampleCount * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; ++i) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataBytes, true);
  const out = new Int16Array(buffer, 44);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function convertChunk(text, opts) {
  const response = await fetch(`${opts.serverUrl}/v1/audio/speech`, {
    method: "POST",
    signal: abortController.signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: opts.modelId,
      input: text,
      voice: opts.voice,
      response_format: "wav",
      num_inference_steps: opts.steps,
    }),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }
  const wallMs = Number(response.headers.get("X-AudioCPP-Wall-Ms"));
  const durationMs = Number(response.headers.get("X-AudioCPP-Audio-Duration-Ms"));
  const rtf = Number(response.headers.get("X-AudioCPP-RTF"));
  return {
    wav: parseWav(await response.arrayBuffer()),
    wallMs,
    durationMs,
    rtf,
  };
}

function updateProgress(done, total, startedMs, wallMs, audioMs) {
  const elapsed = (performance.now() - startedMs) / 1000;
  const ratio = total > 0 ? done / total : 0;
  const eta = done > 0 ? elapsed * (total - done) / done : NaN;
  metrics.progress.textContent = `${done}/${total}`;
  metrics.elapsed.textContent = formatDuration(elapsed);
  metrics.eta.textContent = done === total ? "done" : formatDuration(eta);
  metrics.wall.textContent = formatDuration(wallMs / 1000);
  metrics.audio.textContent = formatDuration(audioMs / 1000);
  metrics.rtf.textContent = audioMs > 0 ? (wallMs / audioMs).toFixed(3) : "-";
  progressBar.style.width = `${Math.round(ratio * 100)}%`;
}

async function startConvert() {
  if (abortController) {
    abortController.abort();
  }
  clearResultUrls();
  abortController = new AbortController();
  buttons.start.disabled = true;
  buttons.download.disabled = true;
  resetMetrics();

  const opts = settings();
  let text = cleanText(fields.textInput.value);
  if (fields.normalizeLatin.checked) {
    const normalized = normalizeLatinText(text);
    text = cleanText(normalized.text);
    if (normalized.changed > 0) {
      setStatus(`Normalized ${normalized.changed.toLocaleString()} characters before conversion.`);
    }
  }
  const chunks = splitText(text, opts.chunkChars);
  if (chunks.length === 0) {
    throw new Error("No text to convert.");
  }
  let partChunks = [];
  let partSamples = 0;
  let sampleRate = 0;
  let channels = 0;
  let wallMs = 0;
  let audioMs = 0;
  const startedMs = performance.now();
  setStatus(`Converting ${chunks.length} chunks.`);
  let completedChunks = 0;
  timer = setInterval(() => updateProgress(completedChunks, chunks.length, startedMs, wallMs, audioMs), 250);
  const flushPart = (isFinal) => {
    if (partChunks.length === 0) {
      return;
    }
    const blob = wavBlob(partChunks, sampleRate, channels);
    const url = URL.createObjectURL(blob);
    const index = partUrls.length + 1;
    const name = `audiocpp-offline-part-${String(index).padStart(3, "0")}.wav`;
    partUrls.push({ url, name });
    const item = document.createElement("div");
    item.className = "download-part";
    const title = document.createElement("strong");
    title.textContent = `Part ${index} · ${formatDuration(partSamples / sampleRate / channels)}`;
    const player = document.createElement("audio");
    player.controls = true;
    player.preload = "metadata";
    player.src = url;
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    link.textContent = name;
    item.append(title, player, link);
    downloadsEl.append(item);
    partChunks = [];
    partSamples = 0;
    buttons.download.disabled = false;
    if (!isFinal) {
      setStatus(`Saved browser part ${index}; continuing conversion.`);
    }
  };
  try {
    for (let i = 0; i < chunks.length; ++i) {
      setStatus(`Converting chunk ${i + 1}/${chunks.length}.`);
      const result = await convertChunk(chunks[i], opts);
      if (sampleRate === 0) {
        sampleRate = result.wav.sampleRate;
        channels = result.wav.channels;
      } else if (sampleRate !== result.wav.sampleRate || channels !== result.wav.channels) {
        throw new Error("Chunk WAV formats do not match.");
      }
      partChunks.push(result.wav.pcm);
      partSamples += result.wav.pcm.length;
      wallMs += Number.isFinite(result.wallMs) ? result.wallMs : 0;
      audioMs += Number.isFinite(result.durationMs) ? result.durationMs : 0;
      completedChunks = i + 1;
      updateProgress(i + 1, chunks.length, startedMs, wallMs, audioMs);
      if (partSamples >= sampleRate * channels * kMaxPartSeconds) {
        flushPart(false);
      }
    }
    flushPart(true);
    buttons.download.disabled = false;
    setStatus(`Done. Generated ${formatDuration(audioMs / 1000)} audio in ${partUrls.length} playable/downloadable part(s).`);
  } finally {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
    abortController = null;
    buttons.start.disabled = false;
  }
}

fields.textFile.addEventListener("change", async () => {
  const file = fields.textFile.files?.[0];
  if (!file) {
    return;
  }
  fields.textInput.value = await file.text();
  updateTextStats();
  setStatus(`Loaded ${file.name}.`);
});

fields.textInput.addEventListener("input", updateTextStats);

buttons.start.addEventListener("click", async () => {
  try {
    await startConvert();
  } catch (error) {
    if (error.name === "AbortError") {
      setStatus("Stopped.");
    } else {
      setStatus(error.message || String(error));
    }
  }
});

buttons.stop.addEventListener("click", () => {
  if (abortController) {
    abortController.abort();
  }
});

buttons.download.addEventListener("click", () => {
  if (partUrls.length === 0) {
    return;
  }
  const first = downloadsEl.querySelector("a");
  first?.click();
});

loadSettings();
resetMetrics();
refreshBackendStatus();
updateTextStats();
