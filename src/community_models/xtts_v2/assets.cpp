#include "engine/community_models/xtts_v2/assets.h"
#include "engine/community_models/xtts_v2/tokenizer_text.h"
#include "engine/framework/io/filesystem.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "gguf.h"

namespace engine::models::xtts_v2 {
namespace {

uint32_t gguf_kv_u32(gguf_context* gctx, const char* key, uint32_t def) {
    int ki = gguf_find_key(gctx, key);
    if (ki < 0) return def;
    if (gguf_get_kv_type(gctx, ki) != GGUF_TYPE_UINT32) return def;
    const auto* data = static_cast<const uint32_t*>(gguf_get_arr_data(gctx, ki));
    return data[0];
}

bool is_gguf_file(const std::string& name) {
    return name.size() >= 5 && name.compare(name.size() - 5, 5, ".gguf") == 0;
}

} // namespace

bool discover_xtts_v2_files(const std::filesystem::path& root,
                             std::filesystem::path& out_gpt,
                             std::filesystem::path& out_vocoder) {
    std::vector<std::filesystem::path> ggufs;
    try {
        for (auto& entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_regular_file() && is_gguf_file(entry.path().filename().string())) {
                ggufs.push_back(entry.path());
            }
        }
    } catch (...) {
        return false;
    }

    // Prefer non-quantized (larger) files; skip symlinks to avoid duplicates
    for (auto& p : ggufs) {
        std::string fn = p.filename().string();
        bool is_gpt = fn.find("gpt") != std::string::npos || fn.find("GPT") != std::string::npos;
        bool is_voc = fn.find("vocoder") != std::string::npos || fn.find("voc") != std::string::npos;
        if (is_gpt) {
            if (out_gpt.empty() ||
                (std::filesystem::is_regular_file(p) && !std::filesystem::is_symlink(p) &&
                 std::filesystem::file_size(p) > std::filesystem::file_size(out_gpt)))
                out_gpt = p;
        } else if (is_voc) {
            if (out_vocoder.empty() ||
                (std::filesystem::is_regular_file(p) && !std::filesystem::is_symlink(p) &&
                 std::filesystem::file_size(p) > std::filesystem::file_size(out_vocoder)))
                out_vocoder = p;
        }
    }

    // If no specific naming, try by file count
    if (ggufs.size() == 2 && (out_gpt.empty() || out_vocoder.empty())) {
        for (auto& p : ggufs) {
            if (out_gpt.empty()) { out_gpt = p; continue; }
            if (out_vocoder.empty()) { out_vocoder = p; }
        }
    }

    if (out_gpt.empty() || out_vocoder.empty()) {
        fprintf(stderr, "xtts_v2: could not discover GGUF files (gpt=%s, vocoder=%s)\n",
                out_gpt.string().c_str(), out_vocoder.string().c_str());
        return false;
    }
    return true;
}

std::optional<XttSv2Assets> load_xtts_v2_assets(const std::filesystem::path& root) {
    XttSv2Assets assets;
    assets.model_root = root;

    if (!discover_xtts_v2_files(root, assets.gpt_gguf_path, assets.vocoder_gguf_path)) {
        return std::nullopt;
    }

    // Read hyperparameters from GPT GGUF
    gguf_context* g = gguf_init_from_file(assets.gpt_gguf_path.string().c_str(), {true, 0});
    if (!g) {
        fprintf(stderr, "xtts_v2: failed to open GPT GGUF for metadata: %s\n",
                assets.gpt_gguf_path.string().c_str());
        return std::nullopt;
    }

    auto& hp = assets.hyper_params;
    hp.n_layers = gguf_kv_u32(g, "xtts.n_layers", hp.n_layers);
    hp.d_model = gguf_kv_u32(g, "xtts.d_model", hp.d_model);
    hp.n_heads = gguf_kv_u32(g, "xtts.n_heads", hp.n_heads);
    hp.head_dim = gguf_kv_u32(g, "xtts.head_dim", hp.head_dim);
    hp.ff_dim = gguf_kv_u32(g, "xtts.ff_dim", hp.ff_dim);
    hp.text_vocab_size = gguf_kv_u32(g, "xtts.text_vocab_size", hp.text_vocab_size);
    hp.mel_vocab_size = gguf_kv_u32(g, "xtts.mel_vocab_size", hp.mel_vocab_size);
    hp.text_pos_size = gguf_kv_u32(g, "xtts.text_pos_size", hp.text_pos_size);
    hp.mel_pos_size = gguf_kv_u32(g, "xtts.mel_pos_size", hp.mel_pos_size);
    hp.start_text_token = gguf_kv_u32(g, "xtts.start_text_token", hp.start_text_token);
    hp.stop_text_token = gguf_kv_u32(g, "xtts.stop_text_token", hp.stop_text_token);
    hp.start_mel_token = gguf_kv_u32(g, "xtts.start_mel_token", hp.start_mel_token);
    hp.stop_mel_token = gguf_kv_u32(g, "xtts.stop_mel_token", hp.stop_mel_token);
    hp.cond_d_model = gguf_kv_u32(g, "xtts.cond_d_model", hp.cond_d_model);
    hp.cond_n_layers = gguf_kv_u32(g, "xtts.cond_n_layers", hp.cond_n_layers);
    hp.perceiver_n_layers = gguf_kv_u32(g, "xtts.perceiver_n_layers", hp.perceiver_n_layers);
    hp.perceiver_n_latents = gguf_kv_u32(g, "xtts.perceiver_n_latents", hp.perceiver_n_latents);
    hp.perceiver_head_dim = gguf_kv_u32(g, "xtts.perceiver_head_dim", hp.perceiver_head_dim);
    gguf_free(g);

    // Load tokenizer from GPT GGUF
    if (!load_xtts_v2_tokenizer(&assets.vocab, assets.gpt_gguf_path)) {
        fprintf(stderr, "xtts_v2: failed to load tokenizer from GPT GGUF\n");
        return std::nullopt;
    }

    // Load mel_stats from vocoder GGUF
    {
        gguf_context* vg = gguf_init_from_file(assets.vocoder_gguf_path.string().c_str(), {true, 0});
        if (vg) {
            int ki = gguf_find_key(vg, "xtts.mel_stats");
            if (ki >= 0 && gguf_get_kv_type(vg, ki) == GGUF_TYPE_FLOAT32) {
                int n = gguf_get_arr_n(vg, ki);
                const float* data = static_cast<const float*>(gguf_get_arr_data(vg, ki));
                assets.mel_stats.assign(data, data + n);
            }
            gguf_free(vg);
        }
    }

    return assets;
}

} // namespace engine::models::xtts_v2

