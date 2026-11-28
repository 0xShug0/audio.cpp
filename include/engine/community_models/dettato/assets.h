#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/speech_encoders/rope_transformer_encoder_runtime.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/community_models/dettato/gru.h"
#include <array>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::community_models::dettato {
struct DettatoAssets {
    assets::ResourceBundle resources;
    io::json::Value config;
    std::shared_ptr<const assets::TensorSource> transcriber, note_repair, grouper, drums;
};
std::shared_ptr<const DettatoAssets> load_dettato_assets(const std::filesystem::path &path);

namespace detail {
// Binding owns checkpoint interpretation and device preparation. Neural builders
// consume these semantic fields; they never look up tensor names.
struct HarmonicFrontendWeights {
    audio::SparseMelFilterbank long_filter, short_filter;
    std::vector<float> long_window, short_window;
};
struct DrumFrontendWeights {
    std::vector<audio::SparseMelFilterbank> filters;
    std::vector<std::vector<float>> windows;
};
using BidirectionalWeights = std::array<GRUWeights, 2>;
struct TranscriberNormWeights {
    modules::ChannelAffineWeights affine;
    bool channel_affine = false;
};
struct PitchMixWeights {
    core::TensorValue coefficients, offsets, matrix;
};
struct LaneBlockWeights {
    modules::Conv2dWeights depthwise;
    PitchMixWeights pitch;
    modules::ChannelAffineWeights norm;
    modules::LinearWeights ffn_in, ffn_out;
    core::TensorValue gamma;
};
struct LaneAttentionWeights {
    TranscriberNormWeights norm;
    modules::LinearWeights qkv, output;
    core::TensorValue gamma;
    std::vector<float> relative_bias;
};
struct LaneRecurrentWeights {
    TranscriberNormWeights norm;
    BidirectionalWeights gru;
    modules::LinearWeights output;
};
struct TranscriberStageWeights {
    std::array<LaneBlockWeights, 5> lanes;
    LaneAttentionWeights pitch_attention, time_attention;
    LaneRecurrentWeights recurrent;
};
struct TranscriberWeights {
    std::unique_ptr<core::BackendWeightStore> storage;
    HarmonicFrontendWeights frontend;
    TranscriberNormWeights input_norm, stem_norm1, stem_norm2, stem_norm3, up_norm;
    modules::Conv2dWeights stem1, stem2, stem_down, down, head;
    std::array<TranscriberStageWeights, 3> stages;
    std::array<LaneBlockWeights, 2> refine;
};
struct RepairBlockWeights {
    modules::NormWeights attention_norm, ffn_norm;
    modules::LinearWeights query, key_value, output;
    modules::GatedFeedForwardWeights ffn;
};
struct NoteRepairWeights {
    std::unique_ptr<core::BackendWeightStore> storage;
    core::TensorValue instrument_embedding, pitch_embedding, velocity_embedding;
    std::array<modules::LinearWeights, 2> continuous, evidence;
    modules::NormWeights input_norm, final_norm;
    std::array<RepairBlockWeights, 9> blocks;
    std::array<modules::LinearWeights, 5> heads;
};
struct GrouperLaneWeights {
    modules::Conv2dWeights depthwise, ffn_in, ffn_out;
    PitchMixWeights pitch;
    modules::NormWeights norm;
    core::TensorValue gamma;
};
struct GrouperAttentionWeights {
    modules::NormWeights norm;
    modules::LinearWeights qkv, output;
    core::TensorValue gamma;
    std::vector<float> relative_bias;
};
struct TrackClassifierLayer {
    int input = 0, output = 0;
    std::vector<double> weight, bias;
};
struct GroupingDecoderWeights {
    std::vector<TrackClassifierLayer> head;
    std::vector<float> credit;
};

struct GrouperWeights {
    GroupingDecoderWeights decoder;
    std::unique_ptr<core::BackendWeightStore> storage;
    HarmonicFrontendWeights frontend;
    std::vector<float> harmonic_prior;
    core::TensorValue pitch_embedding;
    modules::Conv2dWeights lane_stem, occupancy;
    std::array<GrouperLaneWeights, 6> lanes;
    std::array<GrouperAttentionWeights, 2> lane_attention;
    std::array<modules::LinearWeights, 2> features, harmonics, profile, taps;
    modules::LinearWeights lane_read;
    modules::NormWeights tap_norm;
    modules::RopeTransformerEncoderWeights encoder;
    std::array<modules::LinearWeights, 3> heads;
};
struct DrumStemLayer {
    enum class Kind { Convolution, Residual, Pool } kind;
    modules::Conv2dWeights conv, pointwise;
    modules::BatchNorm2dEvalWeights norm, residual_norm;
    int time_pool = 1, frequency_pool = 1;
};
struct DrumEmbedderWeights {
    std::unique_ptr<core::BackendWeightStore> storage;
    std::vector<DrumStemLayer> stem;
    BidirectionalWeights recurrent;
    modules::LinearWeights projection;
};
struct DrumTemporalBlock {
    enum class Kind { Convolution, Recurrent, Attention } kind;
    int64_t kernel = 15, dilation = 1, hidden = 48, channels = 32, heads = 2;
    modules::DepthwiseConv1dWeights depthwise;
    modules::Conv1dWeights pointwise;
    modules::BatchNorm1dEvalWeights affine;
    modules::NormWeights norm;
    BidirectionalWeights recurrent;
    modules::LinearWeights qkv, output;
};
struct DrumTrunkWeights {
    std::unique_ptr<core::BackendWeightStore> storage;
    std::vector<DrumTemporalBlock> blocks;
    std::vector<BidirectionalWeights> towers;
    std::vector<modules::LinearWeights> tower_heads;
    modules::LinearWeights onset, velocity_in, velocity_out;
};
struct DrumAdapterWeights {
    std::unique_ptr<core::BackendWeightStore> storage;
    DrumEmbedderWeights embedder;
    DrumTrunkWeights trunk;
    std::array<modules::Conv2dWeights, 3> convolutions;
    std::array<modules::ChannelAffineWeights, 3> norms;
    modules::LinearWeights cq_projection, input, qkv, attention_output, delta, timpani, gate_in, gate_out;
    std::vector<BidirectionalWeights> recurrent;
    modules::NormWeights norm;
};
struct DrumWeights {
    DrumEmbedderWeights kit_embedder;
    DrumTrunkWeights kit_trunk;
    std::vector<DrumAdapterWeights> experts;
};

HarmonicFrontendWeights load_harmonic_frontend(const assets::TensorSource &, const std::string &prefix);
DrumFrontendWeights load_drum_frontend(const assets::TensorSource &);
TranscriberWeights bind_transcriber_weights(const assets::TensorSource &, const core::ExecutionContext &);
NoteRepairWeights bind_note_repair_weights(const assets::TensorSource &, const core::ExecutionContext &);
GrouperWeights bind_grouper_weights(const assets::TensorSource &, const core::ExecutionContext &);
GroupingDecoderWeights load_grouping_decoder_weights(const assets::TensorSource &);
DrumEmbedderWeights bind_drum_embedder_weights(const assets::TensorSource &, const std::string &prefix,
                                               const io::json::Value &config, const core::ExecutionContext &);
DrumTrunkWeights bind_drum_trunk_weights(const assets::TensorSource &, const std::string &prefix,
                                         const io::json::Value &config, const core::ExecutionContext &);
DrumAdapterWeights bind_drum_adapter_weights(const assets::TensorSource &, size_t index, const io::json::Value &config,
                                             const core::ExecutionContext &);
DrumWeights bind_drum_weights(const assets::TensorSource &, const io::json::Value &config,
                              const core::ExecutionContext &);
} // namespace detail
} // namespace engine::community_models::dettato
