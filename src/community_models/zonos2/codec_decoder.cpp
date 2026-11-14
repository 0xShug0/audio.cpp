#include "engine/community_models/zonos2/codec_decoder.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::zonos2 {
namespace {

constexpr int kCodebooks = 9;
constexpr int kCodebookSize = 1024;
constexpr int kAudioPad = 1025;
constexpr int kGraphNodes = 4096;
// Each cached graph keeps its own scratch (about 1.8 MB per frame on the CPU, dominated by the im2col patches of the
// last block), so only the two most recent window sizes stay prepared.
constexpr size_t kGraphCache = 2;
constexpr std::array<int, 3> kDilations = {1, 3, 9};

using core::TensorValue;

struct ResidualUnit {
    modules::Snake1dWeights snake1, snake2;
    modules::Conv1dWeights conv1, conv2;
    int dilation = 1;
};

struct Block {
    int stride = 0;
    int64_t in_channels = 0;
    modules::Snake1dWeights snake;
    modules::ConvTranspose1dWeights convt;
    std::array<ResidualUnit, 3> residuals;
};

struct DecodeGraph {
    int frames = 0;
    uint64_t last_use = 0;
    std::unique_ptr<core::PreparedGraph> prepared;
    std::array<ggml_tensor *, kCodebooks> ids{};
    ggml_tensor * audio = nullptr;
};

}  // namespace

class Zonos2CodecDecoder::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
        : source_(std::move(source)), execution_(execution) {
        load_weights();
    }

    ~Impl() { graphs_.clear(); }

    int samples_per_frame() const { return samples_per_frame_; }

    std::vector<float> decode(const int32_t * frames, int n, int left, int right) {
        if (n <= 0 || left < 0 || right < 0 || left + right >= n) {
            throw std::runtime_error("zonos2: decoder window has no output frames");
        }
        auto & graph = graph_for(n);
        std::vector<int32_t> column(static_cast<size_t>(n));
        for (int c = 0; c < kCodebooks; ++c) {
            for (int t = 0; t < n; ++t) {
                const int32_t code = frames[static_cast<size_t>(t) * kCodebooks + static_cast<size_t>(c)];
                if (code < 0 || code >= kCodebookSize) throw std::runtime_error("zonos2: decoder code out of range");
                column[static_cast<size_t>(t)] = code;
            }
            ggml_backend_tensor_set(graph.ids[static_cast<size_t>(c)], column.data(), 0, column.size() * sizeof(int32_t));
        }
        graph.prepared->compute();
        const size_t first = static_cast<size_t>(left) * static_cast<size_t>(samples_per_frame_);
        const size_t count = static_cast<size_t>(n - left - right) * static_cast<size_t>(samples_per_frame_);
        std::vector<float> out(count);
        ggml_backend_tensor_get(graph.audio, out.data(), first * sizeof(float), count * sizeof(float));
        return out;
    }

private:
    void load_weights() {
        const auto & s = *source_;
        const assets::GgufMetadata meta(s.source_path());
        for (const auto rate : meta.require_int_array("dac.decoder_rates")) {
            if (rate <= 0 || rate % 2 != 0 || rate > 64) throw std::runtime_error("zonos2: dac.decoder_rates must be even 2..64");
            rates_.push_back(static_cast<int>(rate));
        }
        samples_per_frame_ = 1;
        for (const int rate : rates_) samples_per_frame_ *= rate;
        latent_ = static_cast<int64_t>(meta.require_int("dac.latent_dim", 1, 65536));
        const int64_t width = static_cast<int64_t>(meta.require_int("dac.decoder_dim", 1, 65536));
        store_ = std::make_unique<core::BackendWeightStore>(execution_.backend(), execution_.backend_type(),
                                                            "zonos2.decoder.weights", 4 * 1024 * 1024);
        // Biases and snake alphas are stored ne [1, C]; GGUF drops trailing unit dimensions, so C = 1 reads as [1].
        auto vector = [&](const std::string & name, int64_t channels) {
            consumed_.insert(name);
            const std::vector<int64_t> shape = channels == 1 ? std::vector<int64_t>{1} : std::vector<int64_t>{channels, 1};
            return store_->make_f32(core::TensorShape::from_dims({channels}), s.require_f32(name, shape));
        };
        auto snake = [&](const std::string & prefix, int64_t channels) {
            modules::Snake1dWeights w;
            w.alpha = vector(prefix + ".alpha", channels);
            w.inverse_alpha = vector(prefix + ".inv_alpha", channels);
            return w;
        };
        auto conv = [&](const std::string & prefix, int64_t out, int64_t in, int64_t kernel) {
            consumed_.insert(prefix + ".weight");
            modules::Conv1dWeights w;
            const auto shape = core::TensorShape::from_dims({out, in, kernel});
            w.weight = out == 1 ? store_->load_tensor_as_shape(s, prefix + ".weight", assets::TensorStorageType::F32, {in, kernel}, shape)
                                : store_->load_tensor(s, prefix + ".weight", assets::TensorStorageType::F32, {out, in, kernel});
            w.bias = vector(prefix + ".bias", out);
            return w;
        };
        for (int c = 0; c < kCodebooks; ++c) {
            const std::string name = "quant." + std::to_string(c) + ".table";
            consumed_.insert(name);
            tables_.push_back(store_->load_tensor(s, name, assets::TensorStorageType::F32, {kCodebookSize, latent_}));
        }
        consumed_.insert("quant.bias");
        quant_bias_ = store_->load_f32_tensor(s, "quant.bias", {latent_});
        conv_in_ = conv("dec.conv_in", width, latent_, 7);
        int64_t channels = width;
        for (size_t b = 0; b < rates_.size(); ++b) {
            const std::string p = "dec.b" + std::to_string(b);
            const int stride = rates_[b];
            const int64_t out = channels / 2;
            Block block;
            block.stride = stride;
            block.in_channels = channels;
            block.snake = snake(p + ".snake", channels);
            consumed_.insert(p + ".convt.weight");
            block.convt.weight = store_->load_tensor(s, p + ".convt.weight", assets::TensorStorageType::F32,
                                                     {channels, out, 2 * stride});
            block.convt.bias = vector(p + ".convt.bias", out);
            for (size_t r = 0; r < kDilations.size(); ++r) {
                const std::string q = p + ".res" + std::to_string(r);
                auto & unit = block.residuals[r];
                unit.dilation = kDilations[r];
                unit.snake1 = snake(q + ".snake1", out);
                unit.conv1 = conv(q + ".conv1", out, out, 7);
                unit.snake2 = snake(q + ".snake2", out);
                unit.conv2 = conv(q + ".conv2", out, out, 1);
            }
            blocks_.push_back(std::move(block));
            channels = out;
        }
        out_channels_ = channels;
        snake_out_ = snake("dec.snake_out", channels);
        conv_out_ = conv("dec.conv_out", 1, channels, 7);
        for (const auto & tensor : s.tensors()) {
            if (!consumed_.count(tensor.name)) throw std::runtime_error("zonos2: decoder GGUF: unmapped tensor '" + tensor.name + "'");
        }
        store_->upload();
        source_->release_storage();
    }

    DecodeGraph & graph_for(int frames) {
        ++clock_;
        for (auto & graph : graphs_) {
            if (graph->frames == frames) {
                graph->last_use = clock_;
                return *graph;
            }
        }
        if (graphs_.size() >= kGraphCache) {
            graphs_.erase(std::min_element(graphs_.begin(), graphs_.end(),
                                           [](const auto & a, const auto & b) { return a->last_use < b->last_use; }));
        }
        graphs_.push_back(build_graph(frames));
        graphs_.back()->last_use = clock_;
        return *graphs_.back();
    }

    static TensorValue channels_first(ggml_tensor * x) {
        return core::wrap_tensor(x, core::TensorShape::from_dims({1, x->ne[1], x->ne[0]}));
    }

    std::unique_ptr<DecodeGraph> build_graph(int frames) {
        auto out = std::make_unique<DecodeGraph>();
        out->frames = frames;
        const size_t metadata = ggml_graph_overhead_custom(kGraphNodes, false) + kGraphNodes * ggml_tensor_overhead();
        out->prepared = std::make_unique<core::PreparedGraph>(execution_, core::PreparedGraphConfig{
            "zonos2.decoder", metadata, kGraphNodes, 16 * ggml_tensor_overhead()});
        auto * persistent = out->prepared->persistent_build_context().ggml;
        for (auto *& ids : out->ids) {
            ids = ggml_new_tensor_1d(persistent, GGML_TYPE_I32, frames);
            ggml_set_input(ids);
        }
        out->prepared->allocate_persistent();
        out->prepared->clear_persistent();
        auto & ctx = out->prepared->build_context();
        auto * g = ctx.ggml;
        // quantizer.from_codes: the sum of the folded per-codebook tables plus the summed output-projection bias.
        ggml_tensor * z = nullptr;
        for (int c = 0; c < kCodebooks; ++c) {
            auto * e = ggml_get_rows(g, tables_[static_cast<size_t>(c)].tensor, out->ids[static_cast<size_t>(c)]);
            z = z ? ggml_add(g, z, e) : e;
        }
        z = ggml_add(g, z, quant_bias_.tensor);  // [latent, T]
        auto x = channels_first(ggml_cont(g, ggml_transpose(g, z)));  // {1, latent, T}
        x = modules::Conv1dModule({latent_, conv_in_.weight.shape.dims[0], 7, 1, 3, 1, true}).build(ctx, x, conv_in_);
        for (const auto & block : blocks_) {
            const int64_t out_channels = block.in_channels / 2;
            const int crop = block.stride / 2;
            x = modules::Snake1dModule({block.in_channels}).build(ctx, x, block.snake);
            modules::ConvTranspose1dConfig config{block.in_channels, out_channels, 2 * block.stride, block.stride, crop, 1, true};
            // ConvTranspose1d padding = stride / 2 on the col2im fast path; elsewhere unpadded, then cropped.
            const bool crop_after = !modules::is_conv_transpose1d_col2im_fast_path_eligible(ctx, config);
            if (crop_after) config.padding = 0;
            x = modules::ConvTranspose1dModule(config).build(ctx, x, block.convt);
            if (crop_after) x = modules::SliceModule({2, crop, x.shape.dims[2] - 2 * crop}).build(ctx, x);
            for (const auto & unit : block.residuals) {
                auto y = modules::Snake1dModule({out_channels}).build(ctx, x, unit.snake1);
                y = modules::Conv1dModule({out_channels, out_channels, 7, 1, 3 * unit.dilation, unit.dilation, true})
                        .build(ctx, y, unit.conv1);
                y = modules::Snake1dModule({out_channels}).build(ctx, y, unit.snake2);
                y = modules::Conv1dModule({out_channels, out_channels, 1, 1, 0, 1, true}).build(ctx, y, unit.conv2);
                x = core::wrap_tensor(ggml_add(g, x.tensor, y.tensor), x.shape, GGML_TYPE_F32);
            }
        }
        x = modules::Snake1dModule({out_channels_}).build(ctx, x, snake_out_);
        x = modules::Conv1dModule({out_channels_, 1, 7, 1, 3, 1, true}).build(ctx, x, conv_out_);
        x = modules::TanhModule().build(ctx, x);
        if (x.shape.dims[2] != static_cast<int64_t>(frames) * samples_per_frame_) {
            throw std::runtime_error("zonos2: decoder graph produced an unexpected sample count");
        }
        out->audio = ggml_cont(g, ggml_reshape_1d(g, x.tensor, ggml_nelements(x.tensor)));
        ggml_set_output(out->audio);
        ggml_build_forward_expand(out->prepared->graph(), out->audio);
        out->prepared->allocate();
        return out;
    }

    std::shared_ptr<const assets::TensorSource> source_;
    core::ExecutionContext & execution_;
    std::unique_ptr<core::BackendWeightStore> store_;
    std::set<std::string> consumed_;
    std::vector<int> rates_;
    int samples_per_frame_ = 0;
    int64_t latent_ = 0;
    int64_t out_channels_ = 0;
    std::vector<TensorValue> tables_;
    TensorValue quant_bias_;
    modules::Conv1dWeights conv_in_, conv_out_;
    std::vector<Block> blocks_;
    modules::Snake1dWeights snake_out_;
    std::vector<std::unique_ptr<DecodeGraph>> graphs_;
    uint64_t clock_ = 0;
};

Zonos2CodecDecoder::Zonos2CodecDecoder(std::shared_ptr<const assets::TensorSource> dac, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(dac), execution)) {}

Zonos2CodecDecoder::~Zonos2CodecDecoder() = default;

int Zonos2CodecDecoder::sample_rate() const { return 44100; }

int Zonos2CodecDecoder::samples_per_frame() const { return impl_->samples_per_frame(); }

std::vector<float> Zonos2CodecDecoder::decode(const int32_t * frames, int n, int left, int right) {
    return impl_->decode(frames, n, left, right);
}

std::vector<float> Zonos2CodecDecoder::decode_tiled(const int32_t * frames, int n, int block, int context) {
    if (n <= 0 || block <= 0 || context < 0) throw std::runtime_error("zonos2: invalid tiled decode");
    std::vector<float> out;
    out.reserve(static_cast<size_t>(n) * static_cast<size_t>(impl_->samples_per_frame()));
    for (int lo = 0; lo < n; lo += block) {
        const int hi = std::min(n, lo + block);
        const int a = std::max(0, lo - context);
        const int b = std::min(n, hi + context);
        const auto part = impl_->decode(frames + static_cast<size_t>(a) * kCodebooks, b - a, lo - a, b - hi);
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

std::vector<int32_t> zonos2_undelay(const int32_t * rows, int row_count, int frame_begin, int frame_end) {
    if (frame_begin < 0 || frame_end < frame_begin) throw std::runtime_error("zonos2: invalid undelay range");
    std::vector<int32_t> out(static_cast<size_t>(frame_end - frame_begin) * kCodebooks);
    for (int t = frame_begin; t < frame_end; ++t) {
        for (int j = 0; j < kCodebooks; ++j) {
            const int row = t + j;
            const int code = row < row_count ? rows[static_cast<size_t>(row) * kCodebooks + static_cast<size_t>(j)] : kAudioPad;
            out[static_cast<size_t>(t - frame_begin) * kCodebooks + static_cast<size_t>(j)] = std::clamp(code, 0, kCodebookSize - 1);
        }
    }
    return out;
}

}  // namespace engine::models::zonos2
