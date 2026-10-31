#include "engine/community_models/nemotron_transcribe/decoder.h"

#include <ggml-backend-impl.h>

#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

namespace {

using namespace engine;
using namespace engine::models::nemotron_transcribe;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

class TestWeights final : public assets::TensorSource {
public:
    struct Entry {
        assets::TensorMetadata metadata;
        std::vector<float> values;
    };

    explicit TestWeights(const DecoderConfig & c) {
        const auto inner = c.mamba_heads * c.mamba_head_dim;
        const auto conv = inner + 2 * c.mamba_groups * c.mamba_state_size;
        add("token_embd.weight", {c.vocab_size, c.hidden_size}, 0.1F);
        add("output_norm.weight", {c.hidden_size}, 1.0F);
        add("output.weight", {c.vocab_size, c.hidden_size});
        entries_.at("output.weight").values[0] = 1.0F; // unique highest logit for top_p=0
        add("blk.0.attn_norm.weight", {c.hidden_size}, 1.0F);
        add("blk.0.ssm_in.weight", {inner + conv + c.mamba_heads, c.hidden_size});
        add("blk.0.ssm_conv1d.weight", {conv, c.conv_kernel});
        add("blk.0.ssm_conv1d.bias", {conv});
        add("blk.0.ssm_dt.bias", {c.mamba_heads});
        add("blk.0.ssm_a", {c.mamba_heads, 1}, -1.0F);
        add("blk.0.ssm_d", {c.mamba_heads, 1});
        add("blk.0.ssm_norm.weight", {c.mamba_groups, inner / c.mamba_groups}, 1.0F);
        add("blk.0.ssm_out.weight", {c.hidden_size, inner});
    }

    const std::filesystem::path & source_path() const noexcept override { return path_; }
    bool has_tensor(std::string_view name) const noexcept override { return entries_.count(std::string(name)) != 0; }
    assets::TensorMetadata require_metadata(std::string_view name) const override {
        ++accesses;
        return entries_.at(std::string(name)).metadata;
    }
    std::vector<assets::TensorMetadata> tensors() const override {
        std::vector<assets::TensorMetadata> result;
        for (const auto & item : entries_) result.push_back(item.second.metadata);
        return result;
    }
    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        ++accesses;
        const auto & entry = entries_.at(std::string(name));
        std::vector<std::byte> bytes(entry.values.size() * sizeof(float));
        std::memcpy(bytes.data(), entry.values.data(), bytes.size());
        return {entry.metadata, std::move(bytes)};
    }
    std::vector<float> require_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & shape = std::nullopt) const override {
        ++accesses;
        const auto & entry = entries_.at(std::string(name));
        require(!shape || *shape == entry.metadata.shape, "test weight shape mismatch");
        return entry.values;
    }
    std::optional<std::vector<float>> optional_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & shape = std::nullopt) const override {
        return has_tensor(name) ? std::optional<std::vector<float>>(require_f32(name, shape)) : std::nullopt;
    }
    int64_t require_i64_scalar(std::string_view) const override { throw std::runtime_error("no test scalar"); }

    mutable size_t accesses = 0;

private:
    void add(const std::string & name, std::vector<int64_t> shape, float value = 0.0F) {
        size_t count = 1;
        for (auto dimension : shape) count *= static_cast<size_t>(dimension);
        entries_.emplace(name, Entry{{name, "F32", std::move(shape)}, std::vector<float>(count, value)});
    }

    std::filesystem::path path_ = "self-generated-decoder-weights";
    std::map<std::string, Entry> entries_;
};

DecoderConfig small_config() {
    DecoderConfig c;
    c.hidden_size = 8;
    c.vocab_size = 8;
    c.block_pattern = "M";
    c.attention_heads = 2;
    c.attention_kv_heads = 1;
    c.attention_head_dim = 4;
    c.mamba_heads = 2;
    c.mamba_head_dim = 4;
    c.mamba_groups = 2;
    c.experts = 2;
    c.experts_per_token = 1;
    c.expert_width = 8;
    c.shared_expert_width = 8;
    c.max_context = 8;
    c.prefill_batch_size = 2;
    return c;
}

// The test is single-threaded and restores this CPU buffer callback before any
// retry. Allow the cache allocation, then fail the decode graph allocation.
class FailDecodeAllocation {
public:
    explicit FailDecodeAllocation(core::ExecutionContext & execution)
        : type_(ggml_backend_get_default_buffer_type(execution.backend())) {
        original_ = type_->iface.alloc_buffer;
        allowed_ = 1;
        type_->iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) -> ggml_backend_buffer_t {
            return allowed_-- > 0 ? original_(type, size) : nullptr;
        };
    }
    ~FailDecodeAllocation() { type_->iface.alloc_buffer = original_; }

private:
    using Allocator = ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t, size_t);
    inline static Allocator original_ = nullptr;
    inline static int allowed_ = 0;
    ggml_backend_buffer_type_t type_;
};

void allocation_failure_is_retryable(core::ExecutionContext & execution, const DecoderConfig & c) {
    auto weights = std::make_shared<TestWeights>(c);
    NemotronHDecoderRuntime runtime(weights, execution, c);
    GenerationOptions options;
    options.max_new_tokens = 2;
    options.stop_token_ids.clear();
    bool failed = false;
    {
        FailDecodeAllocation failure(execution);
        try { runtime.generate({1, 2}, {}, 0, {}, options); }
        catch (const std::runtime_error &) { failed = true; }
    }
    require(failed, "decode graph allocation failure must propagate");
    const auto result = runtime.generate({1, 2}, {}, 0, {}, options);
    require(result.termination == TerminationReason::TokenLimit && result.token_ids.size() == 2,
        "a request after allocation failure must rebuild its decode graph");
    options.do_sample = true;
    options.top_p = 0.0F;
    options.seed = 42;
    const auto sampled = runtime.generate({1, 2}, {}, 0, {}, options);
    require(sampled.token_ids == result.token_ids && sampled.token_ids == std::vector<int32_t>({0, 0}),
        "top_p=0 sampling must retain the highest-probability token");
    for (const auto & audio : {std::vector<float>{}, std::vector<float>(static_cast<size_t>(c.hidden_size - 1))}) {
        bool rejected = false;
        try { runtime.generate({1, 2}, audio, 1, {}, options); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "audio rows must match the declared frame count and hidden width");
    }
}

void scalar_options_can_be_validated_without_execution(const DecoderConfig & c) {
    GenerationOptions options;
    options.max_new_tokens = 2;
    options.top_p = 0.0F;
    validate_generation_options(options, c.max_context);
    for (int scenario = 0; scenario < 9; ++scenario) {
        auto invalid = options;
        if (scenario == 0) invalid.max_new_tokens = 0;
        if (scenario == 1) invalid.max_new_tokens = c.max_context + 1;
        if (scenario == 2) invalid.top_k = -1;
        if (scenario == 3) invalid.top_p = -0.01F;
        if (scenario == 4) invalid.top_p = 1.01F;
        if (scenario == 5) invalid.top_p = std::numeric_limits<float>::quiet_NaN();
        if (scenario == 6) { invalid.do_sample = true; invalid.temperature = 0.0F; }
        if (scenario == 7) invalid.temperature = std::numeric_limits<float>::infinity();
        if (scenario == 8) invalid.repetition_penalty = 0.0F;
        bool rejected = false;
        try { validate_generation_options(invalid, c.max_context); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "invalid scalar generation options must be rejected without execution");
    }
}

void invalid_config_is_rejected_before_tensor_access(core::ExecutionContext & execution, const DecoderConfig & c) {
    for (int scenario = 0; scenario < 4; ++scenario) {
        auto invalid = c;
        if (scenario == 0) invalid.mamba_heads = std::numeric_limits<int64_t>::max();
        if (scenario == 1) invalid.max_context = std::numeric_limits<int64_t>::max();
        if (scenario == 2) {
            invalid.hidden_size = invalid.experts = invalid.expert_width = std::numeric_limits<int32_t>::max();
        }
        if (scenario == 3) invalid.block_pattern.assign(256, 'M');
        auto weights = std::make_shared<TestWeights>(c);
        bool rejected = false;
        try { NemotronHDecoderRuntime runtime(weights, execution, invalid); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected && weights->accesses == 0, "unsafe dimensions must be rejected before tensor access");
    }
}

}  // namespace

int main() {
    try {
        core::ExecutionContext execution({core::BackendType::Cpu, 0, 1});
        const auto config = small_config();
        scalar_options_can_be_validated_without_execution(config);
        allocation_failure_is_retryable(execution, config);
        invalid_config_is_rejected_before_tensor_access(execution, config);
    } catch (const std::exception & error) {
        std::cerr << "Nemotron decoder safety test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Nemotron decoder safety test passed\n";
    return 0;
}
