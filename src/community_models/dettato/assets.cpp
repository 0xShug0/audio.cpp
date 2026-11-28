#include "engine/community_models/dettato/assets.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/modules/weight_binding.h"
#include <algorithm>
#include <cstring>
#include <numeric>

namespace engine::community_models::dettato {
std::shared_ptr<const DettatoAssets> load_dettato_assets(const std::filesystem::path &path) {
    auto out = std::make_shared<DettatoAssets>();
    out->resources = model_spec::load_resource_bundle_for_family(path, "dettato");
    out->config = io::json::parse_file(out->resources.require_file("config"));
    if (io::json::optional_string(out->config, "weight_layout", "") != "audiocpp_v1")
        throw std::runtime_error("Dettato requires an audio.cpp native GGUF; reconvert the checkpoint");
    out->transcriber = out->resources.open_tensor_source("transcriber");
    out->note_repair = out->resources.open_tensor_source("evidence");
    out->grouper = out->resources.open_tensor_source("grouper");
    out->drums = out->resources.open_tensor_source("drums");
    return out;
}
} // namespace engine::community_models::dettato

namespace engine::community_models::dettato::detail {
namespace {
using namespace core;
using namespace modules;

struct TensorBindings {
    const assets::TensorSource &source;
    std::unique_ptr<BackendWeightStore> storage;
    TensorBindings(const assets::TensorSource &s, const ExecutionContext &e, std::string label, size_t metadata)
        : source(s),
          storage(std::make_unique<BackendWeightStore>(e.backend(), e.backend_type(), std::move(label), metadata)) {}
    LinearWeights linear(const std::string &prefix) {
        const auto shape = source.require_metadata(prefix + ".weight").shape;
        if (shape.size() != 2)
            throw std::runtime_error("Dettato native linear weight must have two dimensions: " + prefix);
        return binding::linear_from_source(*storage, source, prefix, assets::TensorStorageType::Native, shape[0],
                                           shape[1], source.has_tensor(prefix + ".bias"));
    }
    Conv2dWeights conv(const std::string &prefix) {
        const auto shape = source.require_metadata(prefix + ".weight").shape;
        if (shape.size() != 4)
            throw std::runtime_error("Dettato native convolution weight must have four dimensions: " + prefix);
        return binding::conv2d_from_source(*storage, source, prefix, assets::TensorStorageType::Native, shape[0],
                                           shape[1], shape[2], shape[3], true);
    }
    NormWeights norm(const std::string &prefix) {
        const auto shape = source.require_metadata(prefix + ".weight").shape;
        if (shape.size() != 1)
            throw std::runtime_error("Dettato native normalization weight must be a vector: " + prefix);
        return source.has_tensor(prefix + ".bias")
                   ? binding::norm_from_source(*storage, source, prefix, shape[0])
                   : binding::norm_weight_from_source(*storage, source, prefix, shape[0]);
    }
    ChannelAffineWeights batch_norm(const std::string &prefix) {
        return {binding::tensor_from_named_source(*storage, source, prefix + ".eval_scale",
                                                  assets::TensorStorageType::Native),
                binding::tensor_from_named_source(*storage, source, prefix + ".eval_bias",
                                                  assets::TensorStorageType::Native)};
    }
    BidirectionalWeights gru(const std::string &prefix, int64_t layer = 0) {
        BidirectionalWeights result;
        for (size_t d = 0; d < result.size(); ++d) {
            const auto suffix = "_l" + std::to_string(layer) + (d ? "_reverse" : "");
            result[d] = {binding::tensor_from_named_source(*storage, source, prefix + ".weight_ih" + suffix,
                                                           assets::TensorStorageType::Native),
                         binding::tensor_from_named_source(*storage, source, prefix + ".weight_hh" + suffix,
                                                           assets::TensorStorageType::Native),
                         binding::tensor_from_named_source(*storage, source, prefix + ".bias_ih" + suffix,
                                                           assets::TensorStorageType::Native),
                         binding::tensor_from_named_source(*storage, source, prefix + ".bias_hh" + suffix,
                                                           assets::TensorStorageType::Native)};
        }
        return result;
    }
    // Only backend-dependent layouts remain here. Conversion has already sorted
    // the sparse taps; dense backends materialize their native lane matrix once.
    PitchMixWeights pitch(const std::string &prefix, int64_t lanes, BackendType backend) {
        PitchMixWeights result;
        assets::require_tensor_shape(source, prefix + ".offsets", {14, 2});
        const auto offset_data = source.require_tensor_data(prefix + ".offsets");
        if (offset_data.metadata.dtype != "i32" || offset_data.bytes.size() != 28 * sizeof(int32_t))
            throw std::runtime_error("Dettato pitch offsets must contain 14 I32 coordinate pairs");
        if (backend == BackendType::Cuda) {
            result.coefficients = binding::tensor_from_named_source(*storage, source, prefix + ".coefficients",
                                                                    assets::TensorStorageType::Native);
            result.offsets = storage->make_tensor(TensorShape::from_dims({14, 2}), GGML_TYPE_I32,
                                                  offset_data.bytes.data(), offset_data.bytes.size());
        } else {
            const auto taps = source.require_f32_tensor(prefix + ".coefficients");
            std::vector<int32_t> offsets(28);
            std::memcpy(offsets.data(), offset_data.bytes.data(), offset_data.bytes.size());
            const auto channels = taps.shape.dims[0], count = taps.shape.dims[1];
            std::vector<float> matrix(static_cast<size_t>(channels * lanes * lanes));
            for (int64_t c = 0; c < channels; ++c)
                for (int64_t p = 0; p < lanes; ++p)
                    for (int64_t k = 0; k < count; ++k) {
                        const auto q = p + offsets[static_cast<size_t>(2 * k)];
                        if (q >= 0 && q < lanes)
                            matrix[static_cast<size_t>((c * lanes + p) * lanes + q)] =
                                taps.values[static_cast<size_t>(c * count + k)];
                    }
            result.matrix = storage->make_f32(TensorShape::from_dims({channels, lanes, lanes}), std::move(matrix));
        }
        return result;
    }
    std::unique_ptr<BackendWeightStore> finish() {
        storage->upload();
        return std::move(storage);
    }
};

audio::SparseMelFilterbank load_filter(const assets::TensorSource &source, const std::string &name) {
    const auto data = source.require_f32_tensor(name);
    auto bank = audio::MelFilterbank{}.prepare_sparse({data.values, {data.shape.dims[0], data.shape.dims[1]}});
    for (size_t row = 0; row < bank.starts.size(); ++row)
        if (bank.starts[row] == data.shape.dims[1] && bank.ends[row] == 0)
            bank.starts[row] = 0;
    return bank;
}
} // namespace

HarmonicFrontendWeights load_harmonic_frontend(const assets::TensorSource &source, const std::string &prefix) {
    return {load_filter(source, prefix + ".fb"), load_filter(source, prefix + ".fb_x0"),
            source.require_f32(prefix + ".win"), source.require_f32(prefix + ".win_x0")};
}
DrumFrontendWeights load_drum_frontend(const assets::TensorSource &source) {
    DrumFrontendWeights result;
    for (const auto *name : {"lo", "hi", "cq"}) {
        const auto prefix = std::string("front.") + name;
        result.filters.push_back(load_filter(source, prefix + ".filterbank"));
        result.windows.push_back(source.require_f32(prefix + ".window"));
    }
    return result;
}

TranscriberWeights bind_transcriber_weights(const assets::TensorSource &source, const ExecutionContext &execution) {
    TensorBindings b(source, execution, "dettato.transcriber.weights", 8 * 1024 * 1024);
    TranscriberWeights w;
    w.frontend = load_harmonic_frontend(source, "front");
    const auto norm = [&](const std::string &name) {
        const bool channel = source.has_tensor(name + ".eval_scale");
        return TranscriberNormWeights{
            channel ? b.batch_norm(name)
                    : ChannelAffineWeights{binding::tensor_from_named_source(*b.storage, source, name + ".weight",
                                                                             assets::TensorStorageType::Native),
                                           binding::tensor_from_named_source(*b.storage, source, name + ".bias",
                                                                             assets::TensorStorageType::Native)},
            channel};
    };
    const auto lane = [&](const std::string &name) {
        LaneBlockWeights v;
        v.depthwise = b.conv(name + ".dw_t");
        v.pitch = b.pitch(name + ".pmix", 88, execution.backend_type());
        v.norm = b.batch_norm(name + ".norm");
        v.ffn_in = b.linear(name + ".pw1");
        v.ffn_out = b.linear(name + ".pw2");
        v.gamma =
            binding::tensor_from_named_source(*b.storage, source, name + ".gamma", assets::TensorStorageType::Native);
        return v;
    };
    const auto attention = [&](const std::string &name, bool time) {
        LaneAttentionWeights v;
        v.norm = norm(name + ".ln");
        // QKV padding depends on backend head alignment. It is binding, never graph math.
        if (execution.backend_type() == BackendType::Cuda || execution.backend_type() == BackendType::Vulkan) {
            const int64_t alignment = uses_ggml_cuda_or_hip_backend(execution.backend_type()) ? 64 : 8;
            for (int i = 0; i < 2; ++i) {
                const auto tensor_name = name + (i ? ".qkv.bias" : ".qkv.weight");
                const auto data = source.require_tensor(tensor_name, assets::TensorStorageType::Native,
                                                        source.require_metadata(tensor_name).shape);
                const auto rows = data.shape.dims[0];
                if (rows % 24)
                    throw std::runtime_error("Dettato QKV rows must contain three complete head banks");
                const auto width = rows / 24, padded_width = (width + alignment - 1) / alignment * alignment;
                auto shape = data.shape;
                shape.dims[0] = 24 * padded_width;
                const auto row_bytes = data.bytes.size() / static_cast<size_t>(rows);
                std::vector<std::byte> padded(static_cast<size_t>(shape.dims[0]) * row_bytes);
                for (int64_t h = 0; h < 24; ++h)
                    std::copy_n(data.bytes.data() + h * width * row_bytes, width * row_bytes,
                                padded.data() + h * padded_width * row_bytes);
                auto value = b.storage->make_tensor(shape, data.type, padded.data(), padded.size());
                if (i)
                    v.qkv.bias = value;
                else
                    v.qkv.weight = value;
            }
        } else
            v.qkv = b.linear(name + ".qkv");
        v.output = b.linear(name + ".out");
        v.gamma =
            binding::tensor_from_named_source(*b.storage, source, name + ".gamma", assets::TensorStorageType::Native);
        if (!time)
            v.relative_bias = source.require_f32(name + ".rel_bias", {8, 88, 88});
        return v;
    };
    w.input_norm = norm("stem.inorm");
    w.stem_norm1 = norm("stem.n1");
    w.stem_norm2 = norm("stem.n2");
    w.stem_norm3 = norm("stem.n3");
    w.stem1 = b.conv("stem.c1");
    w.stem2 = b.conv("stem.c2");
    w.stem_down = b.conv("stem.down");
    w.down = b.conv("down");
    int index = 0;
    for (auto &stage : w.stages) {
        for (auto &block : stage.lanes)
            block = lane("layers." + std::to_string(index++));
        stage.pitch_attention = attention("layers." + std::to_string(index++), false);
        stage.time_attention = attention("layers." + std::to_string(index++), true);
        const auto name = "layers." + std::to_string(index++);
        stage.recurrent = {norm(name + ".ln"), b.gru(name + ".gru"), b.linear(name + ".out")};
    }
    w.up_norm = norm("up_norm");
    for (size_t i = 0; i < w.refine.size(); ++i)
        w.refine[i] = lane("refine." + std::to_string(i));
    w.head = b.conv("head");
    w.storage = b.finish();
    return w;
}

NoteRepairWeights bind_note_repair_weights(const assets::TensorSource &source, const ExecutionContext &execution) {
    TensorBindings b(source, execution, "dettato.evidence.weights", 1024 * 1024);
    NoteRepairWeights w;
    w.instrument_embedding =
        binding::tensor_from_named_source(*b.storage, source, "emb.inst.weight", assets::TensorStorageType::Native);
    w.pitch_embedding =
        binding::tensor_from_named_source(*b.storage, source, "emb.pitch.weight", assets::TensorStorageType::Native);
    w.velocity_embedding =
        binding::tensor_from_named_source(*b.storage, source, "emb.vel.weight", assets::TensorStorageType::Native);
    w.continuous = {b.linear("emb.cont.0"), b.linear("emb.cont.2")};
    w.evidence = {b.linear("emb.evm.0"), b.linear("emb.evm.2")};
    w.input_norm = {
        binding::tensor_from_named_source(*b.storage, source, "emb.norm.w", assets::TensorStorageType::Native),
        std::nullopt};
    w.final_norm = {
        binding::tensor_from_named_source(*b.storage, source, "bb.norm.w", assets::TensorStorageType::Native),
        std::nullopt};
    for (size_t i = 0; i < w.blocks.size(); ++i) {
        const auto p = "bb.blocks." + std::to_string(i);
        auto &v = w.blocks[i];
        v.attention_norm = {
            binding::tensor_from_named_source(*b.storage, source, p + ".n1.w", assets::TensorStorageType::Native),
            std::nullopt};
        v.ffn_norm = {
            binding::tensor_from_named_source(*b.storage, source, p + ".n2.w", assets::TensorStorageType::Native),
            std::nullopt};
        v.query = b.linear(p + ".att.q");
        v.key_value = b.linear(p + ".att.kv");
        v.output = b.linear(p + ".att.o");
        v.ffn.up_gate_proj = b.linear(p + ".ff.w1");
        v.ffn.down_proj = b.linear(p + ".ff.w2");
    }
    const std::array<const char *, 5> heads{"keep", "don", "dur", "ins.0", "ins.1"};
    for (size_t i = 0; i < heads.size(); ++i)
        w.heads[i] = b.linear(heads[i]);
    w.storage = b.finish();
    return w;
}

GrouperWeights bind_grouper_weights(const assets::TensorSource &source, const ExecutionContext &execution) {
    TensorBindings b(source, execution, "dettato.grouper.weights", 1024 * 1024);
    GrouperWeights w;
    w.decoder = load_grouping_decoder_weights(source);
    w.frontend = load_harmonic_frontend(source, "front.h");
    w.harmonic_prior = source.require_f32("hd.prior");
    w.pitch_embedding =
        binding::tensor_from_named_source(*b.storage, source, "pitch_emb.weight", assets::TensorStorageType::Native);
    w.lane_stem = b.conv("lanes.stem");
    w.occupancy = b.conv("lanes.occ");
    for (size_t i = 0; i < w.lanes.size(); ++i) {
        const auto p = "lanes.blocks." + std::to_string(i);
        auto &v = w.lanes[i];
        v.depthwise = b.conv(p + ".dw_t");
        v.pitch = b.pitch(p + ".pm", 96, execution.backend_type());
        v.norm = {
            binding::tensor_from_named_source(*b.storage, source, p + ".norm.w", assets::TensorStorageType::Native),
            binding::tensor_from_named_source(*b.storage, source, p + ".norm.b", assets::TensorStorageType::Native)};
        v.ffn_in = b.conv(p + ".pw1");
        v.ffn_out = b.conv(p + ".pw2");
        v.gamma =
            binding::tensor_from_named_source(*b.storage, source, p + ".gamma", assets::TensorStorageType::Native);
    }
    for (size_t i = 0; i < w.lane_attention.size(); ++i) {
        const auto p = "lanes.attn." + std::to_string(2 + 3 * i);
        auto &v = w.lane_attention[i];
        v.norm = b.norm(p + ".ln");
        v.qkv = b.linear(p + ".qkv");
        v.output = b.linear(p + ".out");
        v.gamma =
            binding::tensor_from_named_source(*b.storage, source, p + ".gamma", assets::TensorStorageType::Native);
        v.relative_bias = source.require_f32(p + ".rel_bias", {4, 96, 96});
    }
    w.features = {b.linear("feat.0"), b.linear("feat.2")};
    w.harmonics = {b.linear("hd.enc.0"), b.linear("hd.enc.2")};
    w.profile = {b.linear("prof.net.0"), b.linear("prof.net.2")};
    w.taps = {b.linear("s15_net.0"), b.linear("s15_net.2")};
    w.lane_read = b.linear("lanes.read");
    w.tap_norm = b.norm("s15_ln");
    w.encoder.input_norm = b.norm("ln_in");
    w.encoder.final_norm = b.norm("ln_out");
    for (int i = 0; i < 7; ++i) {
        const auto p = "blocks." + std::to_string(i);
        RopeTransformerEncoderLayerWeights v;
        v.norm1 = b.norm(p + ".ln1");
        v.norm2 = b.norm(p + ".ln2");
        v.qkv = b.linear(p + ".qkv");
        v.attention_output = b.linear(p + ".o");
        v.ffn_in = b.linear(p + ".ff.0");
        v.ffn_out = b.linear(p + ".ff.2");
        w.encoder.layers.push_back(std::move(v));
    }
    w.heads = {b.linear("heads.inst"), b.linear("heads.emb"), b.linear("heads.keep")};
    w.storage = b.finish();
    return w;
}

GroupingDecoderWeights load_grouping_decoder_weights(const assets::TensorSource &source) {
    GroupingDecoderWeights w;
    w.credit = source.require_f32("instrument_credit", {73, 73});
    for (int i = 0; i < 3; ++i) {
        const auto p = "heads.th." + std::to_string(2 * i);
        TrackClassifierLayer v;
        v.input = i ? 176 : 182;
        v.output = i == 2 ? 73 : 176;
        const auto weight = source.require_f32(p + ".weight", {v.output, v.input}),
                   bias = source.require_f32(p + ".bias", {v.output});
        v.weight.assign(weight.begin(), weight.end());
        v.bias.assign(bias.begin(), bias.end());
        w.head.push_back(std::move(v));
    }
    return w;
}

DrumEmbedderWeights bind_drum_embedder_weights(const assets::TensorSource &source, const std::string &prefix,
                                               const io::json::Value &config, const ExecutionContext &execution) {
    TensorBindings b(source, execution, "dettato.drums." + prefix + "weights", 1024 * 1024);
    DrumEmbedderWeights w;
    for (const auto &spec : config.require("stem_layers").as_array()) {
        DrumStemLayer v;
        const auto kind = spec.require("kind").as_string();
        const auto p = prefix + "stem.net." + std::to_string(spec.require("index").as_i64());
        if (kind == "conv") {
            v.kind = DrumStemLayer::Kind::Convolution;
            v.conv = b.conv(p);
            v.norm = {binding::tensor_from_named_source(
                          *b.storage, source,
                          prefix + "stem.net." + std::to_string(spec.require("index").as_i64() + 1) + ".eval_scale",
                          assets::TensorStorageType::Native),
                      binding::tensor_from_named_source(
                          *b.storage, source,
                          prefix + "stem.net." + std::to_string(spec.require("index").as_i64() + 1) + ".eval_bias",
                          assets::TensorStorageType::Native)};
        } else if (kind == "residual") {
            v.kind = DrumStemLayer::Kind::Residual;
            v.conv = b.conv(p + ".dw");
            v.pointwise = b.conv(p + ".pw1");
            v.norm = {binding::tensor_from_named_source(*b.storage, source, p + ".bn1" + ".eval_scale",
                                                        assets::TensorStorageType::Native),
                      binding::tensor_from_named_source(*b.storage, source, p + ".bn1" + ".eval_bias",
                                                        assets::TensorStorageType::Native)};
            v.residual_norm = {binding::tensor_from_named_source(*b.storage, source, p + ".bn2" + ".eval_scale",
                                                                 assets::TensorStorageType::Native),
                               binding::tensor_from_named_source(*b.storage, source, p + ".bn2" + ".eval_bias",
                                                                 assets::TensorStorageType::Native)};
        } else if (kind == "pool") {
            v.kind = DrumStemLayer::Kind::Pool;
            v.time_pool = spec.require("time").as_i64();
            v.frequency_pool = spec.require("frequency").as_i64();
        } else
            throw std::runtime_error("Unknown Dettato converted drum stem layer: " + kind);
        w.stem.push_back(std::move(v));
    }
    w.recurrent = b.gru(prefix + "collapse.rnn");
    w.projection = b.linear(prefix + "collapse.proj");
    w.storage = b.finish();
    return w;
}

DrumTrunkWeights bind_drum_trunk_weights(const assets::TensorSource &source, const std::string &prefix,
                                         const io::json::Value &config, const ExecutionContext &execution) {
    TensorBindings b(source, execution, "dettato.drums." + prefix + "trunk.weights", 1024 * 1024);
    DrumTrunkWeights w;
    for (const auto &spec : config.require("temporal").as_array()) {
        for (int64_t repeat = 0; repeat < io::json::optional_i64(spec, "n", 1); ++repeat) {
            const auto p = prefix + "temporal." + std::to_string(w.blocks.size());
            DrumTemporalBlock v;
            const auto kind = spec.require("t").as_string();
            if (kind == "tb") {
                v.kind = DrumTemporalBlock::Kind::Convolution;
                v.kernel = io::json::optional_i64(spec, "k", 15);
                v.dilation = io::json::optional_i64(spec, "d", 1);
                v.depthwise = {binding::tensor_from_named_source(*b.storage, source, p + ".dw.weight",
                                                                 assets::TensorStorageType::Native),
                               binding::tensor_from_named_source(*b.storage, source, p + ".dw.bias",
                                                                 assets::TensorStorageType::Native)};
                v.pointwise = {binding::tensor_from_named_source(*b.storage, source, p + ".pw.weight",
                                                                 assets::TensorStorageType::Native),
                               binding::tensor_from_named_source(*b.storage, source, p + ".pw.bias",
                                                                 assets::TensorStorageType::Native)};
                v.affine = {binding::tensor_from_named_source(*b.storage, source, p + ".bn" + ".eval_scale",
                                                              assets::TensorStorageType::Native),
                            binding::tensor_from_named_source(*b.storage, source, p + ".bn" + ".eval_bias",
                                                              assets::TensorStorageType::Native)};
            } else if (kind == "gru") {
                v.kind = DrumTemporalBlock::Kind::Recurrent;
                v.hidden = io::json::optional_i64(spec, "h", 48);
                v.norm = b.norm(p + ".norm");
                v.recurrent = b.gru(p + ".rnn");
                v.output = b.linear(p + ".out");
            } else if (kind == "attn") {
                v.kind = DrumTemporalBlock::Kind::Attention;
                v.channels = io::json::optional_i64(spec, "da", 32);
                v.heads = io::json::optional_i64(spec, "heads", 2);
                v.norm = b.norm(p + ".n");
                v.qkv = b.linear(p + ".qkv");
                v.output = b.linear(p + ".o");
            } else
                throw std::runtime_error("Unknown Dettato drum temporal block: " + kind);
            w.blocks.push_back(std::move(v));
        }
    }
    for (size_t i = 0; i < config.require("towers").as_array().size(); ++i) {
        w.towers.push_back(b.gru(prefix + "on.towers." + std::to_string(i)));
        w.tower_heads.push_back(b.linear(prefix + "on.heads." + std::to_string(i)));
    }
    w.onset = b.linear(prefix + "on.rest");
    w.velocity_in = b.linear(prefix + "vel.0");
    w.velocity_out = b.linear(prefix + "vel.1");
    w.storage = b.finish();
    return w;
}
DrumAdapterWeights bind_drum_adapter_weights(const assets::TensorSource &source, size_t index,
                                             const io::json::Value &config, const ExecutionContext &execution) {
    const auto p = "orchestral." + std::to_string(index) + ".";
    TensorBindings b(source, execution, "dettato.drums.expert." + std::to_string(index) + ".weights", 1024 * 1024);
    DrumAdapterWeights w;
    w.embedder = bind_drum_embedder_weights(source, p + "base.", config.require("drum_kit"), execution);
    w.trunk = bind_drum_trunk_weights(source, p + "base.", config.require("drum_kit"), execution);
    for (size_t i = 0; i < w.convolutions.size(); ++i) {
        w.convolutions[i] = b.conv(p + "cq.c" + std::to_string(i + 1));
        w.norms[i] = {
            binding::tensor_from_named_source(*b.storage, source, p + "cq.n" + std::to_string(i + 1) + ".weight",
                                              assets::TensorStorageType::Native),
            binding::tensor_from_named_source(*b.storage, source, p + "cq.n" + std::to_string(i + 1) + ".bias",
                                              assets::TensorStorageType::Native)};
    }
    w.cq_projection = b.linear(p + "cq.proj");
    w.input = b.linear(p + "inp");
    for (int64_t i = 0; i < config.require("orchestral").require("adapter").require("layers").as_i64(); ++i)
        w.recurrent.push_back(b.gru(p + "rnn", i));
    w.qkv = b.linear(p + "qkv");
    w.attention_output = b.linear(p + "ao");
    w.norm = b.norm(p + "ln");
    w.delta = b.linear(p + "delta");
    w.timpani = b.linear(p + "timp");
    w.gate_in = b.linear(p + "gate.0");
    w.gate_out = b.linear(p + "gate.2");
    w.storage = b.finish();
    return w;
}
DrumWeights bind_drum_weights(const assets::TensorSource &source, const io::json::Value &config,
                              const ExecutionContext &execution) {
    DrumWeights w;
    w.kit_embedder = bind_drum_embedder_weights(source, "drum_kit.", config.require("drum_kit"), execution);
    w.kit_trunk = bind_drum_trunk_weights(source, "drum_kit.", config.require("drum_kit"), execution);
    for (int64_t i = 0; i < config.require("orchestral").require("n_models").as_i64(); ++i)
        w.experts.push_back(bind_drum_adapter_weights(source, static_cast<size_t>(i), config, execution));
    return w;
}
} // namespace engine::community_models::dettato::detail
