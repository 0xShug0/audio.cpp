#pragma once

// The turbo GGUF (cstr/chatterbox-turbo-GGUF) names every tensor with a flat dot-separated path
// (e.g. "t3.blk.0.attn_norm.weight", "conds.gen.embedding", "s3.flow.encoder_proj") with no "/"
// namespace separator anywhere. The framework's own prefix-stripping helpers
// (engine::assets::open_tensor_source(path, prefix) / make_prefixed_tensor_source) expect a
// "/"-delimited packed-GGUF namespace instead (matching how this project's own GGUF converter
// packs multiple checkpoints into one file, e.g. "t3_english/tfmr.layers.0...") and throw
// "packed GGUF namespace does not exist" against a dot-separated name. This is therefore a
// separate, dot-based sibling used only for chatterbox_turbo's third-party GGUF.

#include "engine/framework/assets/tensor_source.h"

#include <memory>
#include <string>

namespace engine::models::chatterbox_turbo {

// Strips a literal "<prefix>." from every queried name before delegating to `source` (e.g.
// prefix="t3" turns a query for "blk.0.attn_norm.weight" into "t3.blk.0.attn_norm.weight" on the
// underlying source).
std::shared_ptr<const engine::assets::TensorSource> make_dot_prefixed_tensor_source(
    std::shared_ptr<const engine::assets::TensorSource> source,
    std::string prefix);

}  // namespace engine::models::chatterbox_turbo
