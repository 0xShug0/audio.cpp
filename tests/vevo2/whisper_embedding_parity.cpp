#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/speech_encoders/whisper_embedding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr size_t kWeightContextBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kGraphContextBytes = 512ull * 1024ull * 1024ull;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (value == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("unsupported Whisper embedding backend: " + value);
}

std::vector<float> read_f32_file(const std::filesystem::path & path, size_t expected_count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamoff bytes = input.tellg();
    input.seekg(0, std::ios::beg);
    if (bytes < 0 || static_cast<size_t>(bytes) != expected_count * sizeof(float)) {
        throw std::runtime_error("input file byte count does not match expected Whisper mel shape");
    }
    std::vector<float> values(expected_count);
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return values;
}

void write_f32_file(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }
    output.write(
        reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

engine::modules::WhisperEmbeddingConfig load_config(const std::filesystem::path & model_dir) {
    const auto root = engine::io::json::parse_file(model_dir / "config.json");
    const auto & audio = root.require("audio_encoder");
    engine::modules::WhisperEmbeddingConfig config;
    config.n_mels = audio.require("n_mels").as_i64();
    config.n_audio_ctx = audio.require("n_audio_ctx").as_i64();
    config.n_audio_state = audio.require("n_audio_state").as_i64();
    config.n_audio_head = audio.require("n_audio_head").as_i64();
    config.n_audio_layer = audio.require("n_audio_layer").as_i64();
    return config;
}

engine::modules::LinearWeights load_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    engine::modules::LinearWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", engine::assets::TensorStorageType::F32, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

engine::modules::NormWeights load_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", {hidden_size}),
        store.load_f32_tensor(source, prefix + ".bias", {hidden_size}),
    };
}

engine::modules::WhisperEmbeddingWeights load_weights(
    ggml_backend_t backend,
    const std::filesystem::path & model_dir,
    const engine::modules::WhisperEmbeddingConfig & config,
    std::shared_ptr<engine::core::BackendWeightStore> & store_out) {
    const auto source = engine::assets::open_tensor_source(model_dir / "model.safetensors");
    auto store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        "whisper_embedding.weights",
        kWeightContextBytes);
    engine::modules::WhisperEmbeddingWeights weights;
    weights.conv1 = {
        store->load_tensor(*source, "encoder.conv1.weight", engine::assets::TensorStorageType::F32, {config.n_audio_state, config.n_mels, 3}),
        store->load_f32_tensor(*source, "encoder.conv1.bias", {config.n_audio_state}),
    };
    weights.conv2 = {
        store->load_tensor(*source, "encoder.conv2.weight", engine::assets::TensorStorageType::F32, {config.n_audio_state, config.n_audio_state, 3}),
        store->load_f32_tensor(*source, "encoder.conv2.bias", {config.n_audio_state}),
    };
    weights.positional_embedding = store->load_f32_tensor(
        *source,
        "encoder.positional_embedding",
        {config.n_audio_ctx, config.n_audio_state});
    weights.layers.reserve(static_cast<size_t>(config.n_audio_layer));
    for (int64_t layer = 0; layer < config.n_audio_layer; ++layer) {
        const std::string prefix = "encoder.blocks." + std::to_string(layer);
        engine::modules::WhisperEncoderLayerWeights layer_weights;
        layer_weights.attention_norm = load_norm(*store, *source, prefix + ".attn_ln", config.n_audio_state);
        layer_weights.attention.query = load_linear(*store, *source, prefix + ".attn.query", config.n_audio_state, config.n_audio_state, true);
        layer_weights.attention.key = load_linear(*store, *source, prefix + ".attn.key", config.n_audio_state, config.n_audio_state, false);
        layer_weights.attention.value = load_linear(*store, *source, prefix + ".attn.value", config.n_audio_state, config.n_audio_state, true);
        layer_weights.attention.out = load_linear(*store, *source, prefix + ".attn.out", config.n_audio_state, config.n_audio_state, true);
        layer_weights.mlp_norm = load_norm(*store, *source, prefix + ".mlp_ln", config.n_audio_state);
        layer_weights.mlp.fc1_weight = store->load_tensor(
            *source,
            prefix + ".mlp.0.weight",
            engine::assets::TensorStorageType::F32,
            {config.n_audio_state * 4, config.n_audio_state});
        layer_weights.mlp.fc1_bias = store->load_f32_tensor(*source, prefix + ".mlp.0.bias", {config.n_audio_state * 4});
        layer_weights.mlp.fc2_weight = store->load_tensor(
            *source,
            prefix + ".mlp.2.weight",
            engine::assets::TensorStorageType::F32,
            {config.n_audio_state, config.n_audio_state * 4});
        layer_weights.mlp.fc2_bias = store->load_f32_tensor(*source, prefix + ".mlp.2.bias", {config.n_audio_state});
        weights.layers.push_back(std::move(layer_weights));
    }
    weights.final_norm = load_norm(*store, *source, "encoder.ln_post", config.n_audio_state);
    store->upload();
    store_out = std::move(store);
    return weights;
}

class WhisperEmbeddingGraph {
public:
    WhisperEmbeddingGraph(
        ggml_backend_t backend,
        engine::core::BackendType backend_type,
        const engine::modules::WhisperEmbeddingConfig & config,
        const engine::modules::WhisperEmbeddingWeights & weights)
        : backend_(backend),
          config_(config) {
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Whisper embedding graph context");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "whisper.embedding", backend_type};
        auto input = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, config_.n_mels, config_.n_audio_ctx * 2}));
        input_ = input.tensor;
        auto output = engine::modules::WhisperEmbeddingModule(config_).build(ctx, input, weights);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Whisper embedding graph");
        }
    }

    ~WhisperEmbeddingGraph() {
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    std::vector<float> run(const std::vector<float> & mel) {
        if (mel.size() != static_cast<size_t>(config_.n_mels * config_.n_audio_ctx * 2)) {
            throw std::runtime_error("Whisper embedding input shape mismatch");
        }
        ggml_backend_tensor_set(input_, mel.data(), 0, mel.size() * sizeof(float));
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Whisper embedding graph compute failed");
        }
        std::vector<float> output(static_cast<size_t>(config_.n_audio_ctx * config_.n_audio_state));
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        return output;
    }

private:
    ggml_backend_t backend_ = nullptr;
    engine::modules::WhisperEmbeddingConfig config_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_dir = arg_value(argc, argv, "--model", "models/whisper-medium");
        const std::filesystem::path input_path = arg_value(argc, argv, "--input", "build/whisper_embedding_input.f32");
        const std::filesystem::path output_path = arg_value(argc, argv, "--output", "build/whisper_embedding_cpp.f32");
        const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 8);

        engine::core::BackendConfig backend_config;
        backend_config.type = parse_backend(backend_name);
        backend_config.device = device;
        backend_config.threads = threads;
        ggml_backend_t backend = engine::core::init_backend(backend_config);
        if (backend_config.type == engine::core::BackendType::Cpu) {
            engine::core::set_backend_threads(backend, threads);
        }

        auto config = load_config(model_dir);
        const auto mel = read_f32_file(input_path, static_cast<size_t>(config.n_mels * config.n_audio_ctx * 2));
        {
            std::shared_ptr<engine::core::BackendWeightStore> weight_store;
            const auto weights = load_weights(backend, model_dir, config, weight_store);
            WhisperEmbeddingGraph graph(backend, backend_config.type, config, weights);
            const auto output = graph.run(mel);
            write_f32_file(output_path, output);
        }
        ggml_backend_free(backend);
        std::cout << "output=" << output_path << "\n";
        std::cout << "shape=1," << config.n_audio_ctx << "," << config.n_audio_state << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "whisper_embedding_parity failed: " << error.what() << "\n";
        return 1;
    }
}
