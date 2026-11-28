#include "engine/models/moss_soundeffect/text_encoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/lookup_modules.h"
#include <ggml-alloc.h>
#include "engine/framework/tokenizers/qwen_bpe_bundle.h"

#include <utf8proc.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <map>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::moss_soundeffect {
namespace {
#include "html5_entities.inc"

constexpr int32_t kC1[] = {
    0x20ac,0x81,0x201a,0x192,0x201e,0x2026,0x2020,0x2021,
    0x2c6,0x2030,0x160,0x2039,0x152,0x8d,0x17d,0x8f,
    0x90,0x2018,0x2019,0x201c,0x201d,0x2022,0x2013,0x2014,
    0x2dc,0x2122,0x161,0x203a,0x153,0x9d,0x17e,0x178,
};
const std::unordered_map<int32_t, std::string_view> kLigatures = {
    {0x132,"IJ"},{0x133,"ij"},{0x149,"\xca\xbcn"},
    {0x1c4,"D\xc5\xbd"},{0x1c5,"D\xc5\xbe"},{0x1c6,"d\xc5\xbe"},
    {0x1c7,"LJ"},{0x1c8,"Lj"},{0x1c9,"lj"},{0x1ca,"NJ"},{0x1cb,"Nj"},{0x1cc,"nj"},
    {0x1f1,"DZ"},{0x1f2,"Dz"},{0x1f3,"dz"},
    {0xfb00,"ff"},{0xfb01,"fi"},{0xfb02,"fl"},{0xfb03,"ffi"},{0xfb04,"ffl"},
    {0xfb05,"\xc5\xbft"},{0xfb06,"st"},
};

// Python html.unescape semantics, or ftfy's stricter semicolon/alphanumeric stage.
std::string unescape_html(const std::string & text, bool strict) {
    std::string out;
    for (size_t i = 0; i < text.size();) {
        const size_t start = i++;
        if (text[start] != '&') { out += text[start]; continue; }
        size_t end = i;
        if (end < text.size() && text[end] == '#') {
            size_t digits = ++end;
            int base = 10;
            if (end < text.size() && (text[end] == 'x' || text[end] == 'X')) { base = 16; digits = ++end; }
            uint32_t cp = 0;
            while (end < text.size()) {
                const char ch = text[end];
                const int digit = ch >= '0' && ch <= '9' ? ch - '0' :
                    (ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 : (ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : 99));
                if (digit >= base) break;
                cp = std::min(uint32_t(0x110000), cp * base + digit);
                ++end;
            }
            const bool semicolon = end < text.size() && text[end] == ';';
            if (end == digits || (strict && (!semicolon || end - (start + 2) > 24))) { out += '&'; continue; }
            i = end + semicolon;
            if (cp >= 0x80 && cp <= 0x9f) cp = kC1[cp - 0x80];
            else if (!cp || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
            else if ((cp >= 1 && cp <= 8) || cp == 0xb || (cp >= 0xe && cp <= 0x1f) || cp == 0x7f ||
                     (cp >= 0xfdd0 && cp <= 0xfdef) || (cp & 0xffff) >= 0xfffe) continue;
            utf8proc_uint8_t bytes[4];
            out.append(reinterpret_cast<char *>(bytes), utf8proc_encode_char(cp, bytes));
            continue;
        }
        while (end < text.size() && end - i < (strict ? 24u : 32u)) {
            const unsigned char ch = text[end];
            if (strict ? !((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) :
                (ch == '\t' || ch == '\n' || ch == '\f' || ch == ' ' || ch == '<' || ch == '&' || ch == '#' || ch == ';')) break;
            ++end;
        }
        if (end < text.size() && text[end] == ';') ++end;
        if (strict && (end == i || text[end - 1] != ';')) { out += '&'; continue; }
        bool matched = false;
        for (size_t length = end - i; length >= 2; --length) {
            const std::string_view name(text.data() + i, length);
            if (name.front() == '+') continue;  // '+' is reserved for ftfy-only aliases in the data.
            std::string alias;
            for (int attempt = 0; attempt < (strict ? 2 : 1); ++attempt) {
                if (attempt) alias = "+" + std::string(name);
                const std::string_view key = attempt ? std::string_view(alias) : name;
                const auto found = std::lower_bound(std::begin(kEntities), std::end(kEntities), key,
                    [](const auto & entity, std::string_view value) { return entity.first < value; });
                if (found != std::end(kEntities) && found->first == key) {
                    out += found->second; i += length; matched = true; break;
                }
            }
            if (matched || strict) break;
        }
        if (!matched) out += '&';
    }
    return out;
}

// NFKC is restricted to ftfy's width block, never applied to ordinary text.
// The final whitespace pass is separate because upstream unescapes HTML twice *after* ftfy.
std::string clean_unicode(const std::string & text, bool whitespace_only = false) {
    std::string out;
    bool pending_space = false;
    for (size_t i = 0; i < text.size();) {
        int32_t cp;
        const auto length = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t *>(text.data() + i), text.size() - i, &cp);
        if (length < 0) throw std::runtime_error("MOSS text input must be valid UTF-8");
        i += static_cast<size_t>(length);
        if (whitespace_only) {
            const auto category = utf8proc_category(cp);
            if ((cp >= 9 && cp <= 13) || cp == 0x85 || category == UTF8PROC_CATEGORY_ZS ||
                category == UTF8PROC_CATEGORY_ZL || category == UTF8PROC_CATEGORY_ZP) {
                pending_space = !out.empty(); continue;
            }
            if (pending_space) { out += ' '; pending_space = false; }
        } else {
            if (cp >= 0x80 && cp <= 0x9f) cp = kC1[cp - 0x80];
            const auto ligature = kLigatures.find(cp);
            if (ligature != kLigatures.end()) { out += ligature->second; continue; }
            if (cp == 0x2bc || (cp >= 0x2018 && cp <= 0x201b)) cp = '\'';
            if (cp >= 0x201c && cp <= 0x201f) cp = '"';
            if (cp == '\r') { if (i < text.size() && text[i] == '\n') ++i; cp = '\n'; }
            if (cp == 0x2028 || cp == 0x2029) cp = '\n';
            if (cp <= 8 || cp == 0xb || (cp >= 0xe && cp <= 0x1f) || cp == 0x7f ||
                (cp >= 0x206a && cp <= 0x206f) || cp == 0xfeff || (cp >= 0xfff9 && cp <= 0xfffc)) continue;
            if (cp == 0x3000) cp = ' ';
            if (cp >= 0xff01 && cp < 0xfff0) {
                int32_t width[32];
                const auto count = utf8proc_decompose_char(cp, width, 32,
                    static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPAT | UTF8PROC_DECOMPOSE), nullptr);
                if (count < 0 || count > 32) throw std::runtime_error("MOSS width normalization failed");
                for (int j = 0; j < count; ++j) {
                    utf8proc_uint8_t bytes[4];
                    out.append(reinterpret_cast<char *>(bytes), utf8proc_encode_char(width[j], bytes));
                }
                continue;
            }
        }
        utf8proc_uint8_t bytes[4];
        out.append(reinterpret_cast<char *>(bytes), utf8proc_encode_char(cp, bytes));
    }
    if (whitespace_only) return out;
    utf8proc_uint8_t * normalized = nullptr;
    const auto length = utf8proc_map(reinterpret_cast<const utf8proc_uint8_t *>(out.data()), out.size(), &normalized,
                                   static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> owned(normalized, std::free);
    if (length < 0) throw std::runtime_error(std::string("MOSS NFC normalization failed: ") + utf8proc_errmsg(length));
    return std::string(reinterpret_cast<char *>(owned.get()), static_cast<size_t>(length));
}

std::string clean_prompt(const std::string & prompt) {
    std::string fixed;
    bool html_auto = true;
    static const std::regex ansi("\x1b\\[[0-9;]*[A-Za-z]");
    // ftfy processes newline-delimited segments, disabling auto HTML decoding at '<'.
    for (size_t start = 0; start < prompt.size();) {
        const auto newline = prompt.find('\n', start);
        const auto end = newline == std::string::npos ? prompt.size() : newline + 1;
        auto segment = prompt.substr(start, end - start);
        if (segment.find('<') != std::string::npos) html_auto = false;
        for (;;) {
            const auto previous = segment;
            if (html_auto) segment = unescape_html(segment, true);
            segment = clean_unicode(std::regex_replace(segment, ansi, ""));
            if (segment == previous) break;
        }
        fixed += segment;
        start = end;
    }
    return clean_unicode(unescape_html(unescape_html(fixed, false), false), true);
}
}  // namespace

struct TextEncoderRuntime::Impl {
    struct EncodingGraph {
        core::ExecutionContext & execution;
        core::ConstantTensorCache constants;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
        core::HostGraphPlan plan;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * input = nullptr;
        core::TensorValue output;

        explicit EncodingGraph(core::ExecutionContext & execution)
            : execution(execution), constants(execution.backend(), execution.config().threads,
                "moss_soundeffect.text.constants", 1024 * 1024) {}
        ~EncodingGraph() {
            plan.reset();
            if (graph) core::release_backend_graph_resources(execution.backend(), graph, true);
        }
    };
    core::ExecutionContext & execution;
    std::shared_ptr<const Assets> assets;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer;
    int32_t pad_token_id;
    std::unique_ptr<core::BackendWeightStore> weights;
    modules::DecoderHiddenConfig config;
    modules::CausalDecoderRuntimeWeights binding;
    std::map<int64_t, std::unique_ptr<EncodingGraph>> graphs;

    Impl(core::ExecutionContext & execution, std::shared_ptr<const Assets> model, assets::TensorStorageType storage)
        : execution(execution), assets(std::move(model)) {
        if (!assets) throw std::runtime_error("MOSS text encoder requires assets");
        tokenizer = tokenizers::load_qwen_bpe_tokenizer(assets->resources);
        const auto tokenizer_config = assets->resources.parse_json("tokenizer_config");
        const auto & pad_token = tokenizer_config.require("pad_token");
        const auto pad_id = tokenizer->find_token_id(
            pad_token.is_string() ? pad_token.as_string() : pad_token.require("content").as_string());
        if (!pad_id) throw std::runtime_error("MOSS tokenizer pad_token is absent from the vocabulary");
        pad_token_id = *pad_id;
        weights = std::make_unique<core::BackendWeightStore>(execution.backend(), execution.backend_type(),
            "moss_soundeffect.text.weights", 4 * 1024 * 1024);
        binding = load_text_weights(*assets, *weights, storage);
        weights->upload();
        config.stack = assets->text;
        auto & cast = config.stack.activation_cast;
        cast.enabled = true;
        cast.fused_round = true;
        cast.norm_intermediate_type = GGML_TYPE_BF16;
        cast.rope_intermediate_type = GGML_TYPE_BF16;
        cast.after_input_norm = true;
        cast.after_qkv_projection = true;
        cast.after_qk_norm = true;
        cast.after_rope = true;
        cast.after_attention = true;
        cast.after_attention_output = true;
        cast.after_residual = true;
        cast.after_ffn_norm = true;
        cast.after_mlp_projection = true;
        cast.after_mlp_silu = true;
        cast.after_mlp_mul = true;
        cast.after_output = true;
        config.hidden_mode = modules::CausalDecoderLogitsMode::AllSteps;
    }

    std::unique_ptr<EncodingGraph> build_graph(int64_t steps) {
        const auto started = std::chrono::steady_clock::now();
        auto g = std::make_unique<EncodingGraph>(execution);
        g->context.reset(ggml_init({64 * 1024 * 1024, nullptr, true}));
        if (!g->context) throw std::runtime_error("failed to initialize MOSS text graph");
        core::ModuleBuildContext ctx{g->context.get(), "moss_soundeffect.text", execution.backend_type()};
        g->input = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, steps);
        ggml_set_input(g->input);
        auto ids = core::wrap_tensor(g->input, core::TensorShape::from_dims({steps}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({assets->vocab_size, assets->text.hidden_size}).build(ctx, ids, binding.token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, steps, assets->text.hidden_size}));
        const auto position_values = modules::decoder_position_ids(steps);
        const auto mask_values = modules::causal_prefill_mask_values(1, steps);
        const auto positions = g->constants.make_tensor(core::TensorShape::from_dims({steps}), GGML_TYPE_I32,
            position_values.data(), position_values.size() * sizeof(int32_t));
        const auto mask = g->constants.make_tensor(core::TensorShape::from_dims({1, 1, steps, steps}), GGML_TYPE_F16,
            mask_values.data(), mask_values.size() * sizeof(ggml_fp16_t));
        g->constants.ensure_uploaded();
        g->output = modules::DecoderHiddenModule(config).build(ctx, x,
            positions,
            {binding.stack, binding.final_norm}, std::nullopt,
            mask).hidden;
        ggml_set_output(g->output.tensor);
        g->graph = ggml_new_graph_custom(ctx.ggml, 65536, false);
        ggml_build_forward_expand(g->graph, g->output.tensor);
        core::validate_backend_graph_supported(execution.backend(), g->graph, "MOSS text encoder");
        g->allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!g->allocator || !ggml_gallocr_alloc_graph(g->allocator.get(), g->graph))
            throw std::runtime_error("failed to allocate MOSS text graph");

        core::prepare_host_graph_plan(execution, g->graph, g->plan);
        debug::timing_log_scalar("moss_soundeffect.text.buffer_mb", ggml_gallocr_get_buffer_size(g->allocator.get(), 0) / (1024.0 * 1024.0));
        debug::timing_log_scalar("moss_soundeffect.text.build_ms", debug::elapsed_ms(started));
        return g;
    }
};

TextEncoderRuntime::TextEncoderRuntime(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets,
                                     assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(execution, std::move(assets), storage)) {}
TextEncoderRuntime::~TextEncoderRuntime() = default;

std::vector<int32_t> TextEncoderRuntime::tokenize(const std::string & prompt) const {
    auto ids = impl_->tokenizer->encode(clean_prompt(prompt));
    ids.resize(std::min(ids.size(), static_cast<size_t>(impl_->assets->text_length)));
    return ids;
}

std::vector<float> TextEncoderRuntime::encode(const std::string & prompt) {
    const auto started = std::chrono::steady_clock::now();
    auto ids = tokenize(prompt);
    const auto valid_steps = ids.size();
    const auto & a = *impl_->assets;
    std::vector<float> hidden(static_cast<size_t>(a.text_length * a.text.hidden_size), 0.0F);
    if (!ids.empty()) {
        // Future causal padding does not affect valid rows. Cache each bounded
        // graph shape once, rather than computing the full 512-row canvas.
        int64_t steps = std::min<int64_t>(32, a.text_length);
        while (steps < static_cast<int64_t>(valid_steps)) steps = std::min(steps * 2, a.text_length);
        ids.resize(static_cast<size_t>(steps), impl_->pad_token_id);
        auto found = impl_->graphs.find(steps);
        if (found == impl_->graphs.end()) found = impl_->graphs.emplace(steps, impl_->build_graph(steps)).first;
        auto & g = *found->second;
        ggml_backend_tensor_set(g.input, ids.data(), 0, ids.size() * sizeof(int32_t));
        if (core::compute_graph(impl_->execution, g.graph, g.plan, "MOSS text encoder") != GGML_STATUS_SUCCESS)
            throw std::runtime_error("MOSS text graph execution failed");
        ggml_backend_tensor_get(g.output.tensor, hidden.data(), 0,
            valid_steps * static_cast<size_t>(a.text.hidden_size) * sizeof(float));
        core::round_f32_to_bf16_in_place(hidden);
    }
    debug::timing_log_scalar("moss_soundeffect.text.encode_ms", debug::elapsed_ms(started));
    return hidden;
}
}  // namespace engine::models::moss_soundeffect
