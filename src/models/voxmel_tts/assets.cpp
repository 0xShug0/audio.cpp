#include "engine/models/voxmel_tts/assets.h"

#include "engine/framework/io/json.h"

#include <cmath>
#include <stdexcept>

namespace engine::models::voxmel_tts {

std::shared_ptr<const ModelAssets> load_assets(const std::filesystem::path & path) {
    auto a = std::make_shared<ModelAssets>();
    a->resources = model_spec::load_resource_bundle_for_family(path, "voxmel_tts");
    const auto config = a->resources.parse_json("config");
    const auto & b = config.require("backbone");
    a->hidden = b.require("hidden_size").as_i64();
    a->intermediate = b.require("intermediate_size").as_i64();
    a->layers = b.require("num_hidden_layers").as_i64();
    a->heads = b.require("num_attention_heads").as_i64();
    a->kv_heads = b.require("num_key_value_heads").as_i64();
    a->vocabulary = b.require("vocab_size").as_i64();
    a->max_positions = b.require("max_position_embeddings").as_i64();
    a->norm_eps = b.require("rms_norm_eps").as_f32();
    a->rope_theta = b.require("rope_theta").as_f32();
    const auto & m = config.require("model");
    a->mels = m.require("n_mels").as_i64();
    a->sample_rate = m.require("sample_rate").as_i64();
    a->hop = m.require("hop_length").as_i64();
    a->tier_hidden = m.require("dim").as_i64();
    a->tier_heads = m.require("num_heads").as_i64();
    a->flow_hidden = m.require("flow_hidden").as_i64();
    a->flow_layers = m.require("flow_depth").as_i64();
    a->detail_norm = m.require("detail_norm").as_bool();
    for (const auto & v : m.require("split_axes").as_array()) {
        const auto axis = v.as_string();
        if (axis != "t" && axis != "f") throw std::runtime_error("VoxMel split axis must be t or f");
        a->split_axes.push_back(axis.front());
        a->tier_windows.push_back(m.require("tier_attn_window").require(axis).as_i64());
    }
    const auto & counts = m.require("n_layers").as_array();
    for (size_t i = 1; i < counts.size(); ++i) a->tier_layers.push_back(counts[i].as_i64());
    if (a->split_axes != std::vector<char>{'t', 'f'} || a->tier_windows != std::vector<int64_t>{-1, 0} ||
        a->tier_layers.size() != 2 || a->sample_rate != 24000 || a->hop != 256 ||
        a->mels != 100 || a->hidden <= 0 || a->heads <= 0 || a->hidden % a->heads ||
        a->tier_heads <= 0 || a->tier_hidden <= 0 || a->tier_hidden % a->tier_heads ||
        a->flow_hidden <= 0 || a->flow_layers <= 0 || a->layers <= 0)
        throw std::runtime_error("Unsupported VoxMel architecture");
    a->model = a->resources.open_tensor_source("model_weights");
    a->vocos = a->resources.open_tensor_source("vocos_weights");
    a->mel_mean = a->model->require_f32("mel_mean", {a->mels});
    a->mel_std = a->model->require_f32("mel_std", {a->mels});
    for (float value : a->mel_std) if (!(value > 0) || !std::isfinite(value))
        throw std::runtime_error("VoxMel Mel standard deviations must be positive and finite");
    a->special = a->model->require_f32("special_embedding.weight", {50, a->hidden});
    a->null_text = a->model->require_f32("null_text", {1, a->hidden});
    a->mel_filterbank.shape = {a->mels, 513};
    a->mel_filterbank.values = a->model->require_f32("frontend.mel_filterbank", {a->mels, 513});
    a->mel_window = a->model->require_f32("frontend.window", {1024});
    return a;
}

} // namespace engine::models::voxmel_tts
