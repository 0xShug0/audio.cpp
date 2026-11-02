// Reuse the production CLI and its model/session code inside a browser worker.
importScripts("audiocpp_cli.js" + self.location.search);

self.onmessage = async ({ data }) => {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("The browser did not provide a WebGPU adapter.");
    const info = adapter.info;
    self.postMessage({ type: "adapter", info: {
      vendor: info.vendor, architecture: info.architecture,
      device: info.device, description: info.description,
      features: [...adapter.features],
      limits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      },
    }});
    self.postMessage({ type: "status", message: "Initializing WebAssembly…" });
    const module = await createAudioCpp({
      mainScriptUrlOrBlob: new URL("audiocpp_cli.js" + self.location.search, self.location.href).href,
      locateFile: path => path + self.location.search,
      preRun: [module => Object.assign(module.ENV, data.environment)],
      print: line => self.postMessage({ type: "log", line }),
      printErr: line => self.postMessage({ type: "log", line }),
    });
    const largeFiles = data.files.filter(({ file }) => file.size > 0x7fffffff);
    if (largeFiles.length) {
      if (!module.WORKERFS) throw new Error("Files larger than 2 GiB require the WASM64 browser build.");
      module.FS.mkdir("/large-inputs");
      module.FS.mount(module.WORKERFS, {
        blobs: largeFiles.map(({ path, file }) => ({ name: path.slice(1), data: file })),
      }, "/large-inputs");
    }
    for (const { path, file } of data.files) {
      self.postMessage({ type: "status", message: `Loading ${file.name}…` });
      module.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")) || "/");
      if (file.size > 0x7fffffff) {
        module.FS.symlink("/large-inputs" + path, path);
        continue;
      }
      const stream = module.FS.open(path, "w");
      try {
        module.FS.truncate(path, file.size);
        for (let offset = 0; offset < file.size; offset += 64 * 1024 * 1024) {
          const bytes = new Uint8Array(await file.slice(offset, offset + 64 * 1024 * 1024).arrayBuffer());
          module.FS.write(stream, bytes, 0, bytes.length, offset);
        }
      } finally {
        module.FS.close(stream);
      }
    }
    const args = [...data.args];
    if (!args.includes("--log")) args.push("--log");
    const started = performance.now();
    self.postMessage({ type: "status", message: "Running model on browser WebGPU…" });
    const status = await module.callMain(args);
    if (status !== 0) throw new Error(`audio.cpp exited with status ${status}. See the log.`);
    const outputs = [];
    for (const flag of ["--out", "--text-out", "--words-out"]) {
      const index = data.args.indexOf(flag);
      if (index >= 0) {
        const path = data.args[index + 1];
        const bytes = module.FS.readFile(path);
        outputs.push({ path, bytes });
      }
    }
    const directoryIndex = args.indexOf("--out-dir");
    if (directoryIndex >= 0) {
      const directory = args[directoryIndex + 1];
      for (const name of module.FS.readdir(directory)) {
        const path = directory + "/" + name;
        if (module.FS.isFile(module.FS.stat(path).mode)) outputs.push({ path, bytes: module.FS.readFile(path) });
      }
    }
    self.postMessage({ type: "done", elapsedMs: performance.now() - started, outputs },
      outputs.map(output => output.bytes.buffer));
  } catch (error) {
    self.postMessage({ type: "error", message: error.stack || error.message || JSON.stringify(error) || String(error) });
  }
};
