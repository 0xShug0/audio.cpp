#pragma once

#include <cstdint>
#include <string>

namespace engine::core {

// Model-load progress accounting for hosts that stream the trace log
// (audiocpp_server --log-file, audiocpp_cli --log). Mirrors the way
// llama.cpp reports model loading: a bytes-done / bytes-total curve fed
// once per tensor while weights upload, so load UIs get a continuous,
// honest signal instead of a single "loaded" milestone at the end.
//
// One curve per ModelRegistry::load call; the registry brackets it with
// runtime.load.phase lines and BackendWeightStore::upload feeds it:
//
//   [TRACE ...] runtime.load.phase inspect
//   [TRACE ...] runtime.load.phase load
//   [TRACE ...] <store>.weights.upload_progress 0.002
//   ...
//   [TRACE ...] <store>.weights.upload_progress 1
//   [TRACE ...] runtime.load.phase loaded
//
// Thread-safe: a lazy first-use load can overlap an eager load.

/// Begin tracking a new model load. total_bytes seeds the denominator from
/// the model's on-disk weight files; stores raise it if they end up
/// uploading more (dequantized expansion, derived tensors). No-op unless
/// the trace log is enabled.
void begin_model_load(uint64_t total_bytes);

/// A weight store is about to upload `bytes` (BackendWeightStore::upload).
void register_weight_bytes(uint64_t bytes);

/// One tensor finished uploading.
void add_uploaded_weight_bytes(uint64_t bytes);

/// Trace `<store>.weights.upload_progress <0..1>` when the fraction has
/// moved enough since the last emitted line.
void emit_weight_upload_progress(const std::string & store_name);

/// The model load finished: emit the final curve point at 1.0.
void end_model_load();

}  // namespace engine::core
