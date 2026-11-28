// This file is included by generator.cpp inside the OmniVoice anonymous namespace.

class GraphBufferSet {
public:
    ~GraphBufferSet() {
        clear();
    }

    void clear() noexcept {
        for (auto & buffer : buffers_) {
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
                buffer = nullptr;
            }
        }
    }

    void allocate(ggml_context * ctx, ggml_backend_t backend, const char * error_prefix) {
        if (ctx == nullptr || backend == nullptr) {
            throw std::runtime_error(std::string(error_prefix) + " requires initialized graph context and backend");
        }
        auto * const buft = ggml_backend_get_default_buffer_type(backend);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        const size_t max_size = ggml_backend_buft_get_max_size(buft);
        if (alignment == 0 || max_size == 0) {
            throw std::runtime_error(std::string(error_prefix) + " allocator returned invalid backend limits");
        }

        struct TensorRangePlan {
            ggml_tensor * first = nullptr;
            ggml_tensor * last = nullptr;
            size_t bytes = 0;
        };

        std::vector<TensorRangePlan> plans;
        plans.reserve(8);

        size_t current_bytes = 0;
        ggml_tensor * range_first = ggml_get_first_tensor(ctx);
        for (ggml_tensor * tensor = range_first; tensor != nullptr; tensor = ggml_get_next_tensor(ctx, tensor)) {
            size_t tensor_bytes = 0;
            if (tensor->data == nullptr && tensor->view_src == nullptr) {
                tensor_bytes = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
            }

            if (current_bytes > 0 && (current_bytes + tensor_bytes) > max_size) {
                plans.push_back({range_first, tensor, current_bytes});
                range_first = tensor;
                current_bytes = tensor_bytes;
            } else {
                current_bytes += tensor_bytes;
            }
        }
        if (current_bytes > 0) {
            plans.push_back({range_first, nullptr, current_bytes});
        }
        if (plans.empty()) {
            throw std::runtime_error(std::string(error_prefix) + " requires non-zero compute buffer");
        }

        if (buffers_.size() < plans.size()) {
            buffers_.resize(plans.size(), nullptr);
        }

        for (size_t i = 0; i < plans.size(); ++i) {
            auto & buffer = buffers_[i];
            if (buffer == nullptr || ggml_backend_buffer_get_size(buffer) < plans[i].bytes) {
                if (buffer != nullptr) {
                    ggml_backend_buffer_free(buffer);
                    buffer = nullptr;
                }
                buffer = ggml_backend_buft_alloc_buffer(buft, plans[i].bytes);
                if (buffer == nullptr) {
                    throw std::runtime_error(std::string("failed to allocate ") + error_prefix + " graph buffer");
                }
                ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            } else {
                ggml_backend_buffer_reset(buffer);
            }

            auto tallocr = ggml_tallocr_new(buffer);
            for (ggml_tensor * tensor = plans[i].first; tensor != plans[i].last; tensor = ggml_get_next_tensor(ctx, tensor)) {
                enum ggml_status status = GGML_STATUS_SUCCESS;
                if (tensor->data == nullptr) {
                    if (tensor->view_src == nullptr) {
                        status = ggml_tallocr_alloc(&tallocr, tensor);
                    } else if (tensor->buffer == nullptr) {
                        status = ggml_backend_view_init(tensor);
                    }
                } else if (tensor->view_src != nullptr && tensor->buffer == nullptr) {
                    status = ggml_backend_view_init(tensor);
                }
                if (status != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error(std::string("failed to allocate ") + error_prefix + " graph tensor");
                }
            }
        }
    }

private:
    std::vector<ggml_backend_buffer_t> buffers_;
};

class LayerwiseEmbeddingStage {
public:
    ~LayerwiseEmbeddingStage() {
        clear_graph();
    }

    void rebuild(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        int64_t rows,
        int64_t total_token_capacity) {
        clear_graph();
        runtime_ = std::move(runtime);
        graph_arena_bytes_ = graph_arena_bytes;
        rows_ = rows;
        total_tokens_capacity_ = total_token_capacity;
        if (runtime_ == nullptr || rows_ <= 0 || total_tokens_capacity_ <= 0) {
            throw std::runtime_error("OmniVoice layerwise embedding stage requires valid runtime and token capacity");
        }

        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice layerwise embedding graph context");
        }

        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "omnivoice.generator.layerwise.embedding", runtime_->backend_type()};
        std::array<core::TensorValue, 8> audio_id_values = {};
        auto text_ids =
            core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({rows_, total_tokens_capacity_}));
        text_ids_ = text_ids.tensor;
        ggml_set_input(text_ids_);
        for (int64_t codebook = 0; codebook < config.num_audio_codebook; ++codebook) {
            auto audio_ids =
                core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({rows_, total_tokens_capacity_}));
            audio_ids_[static_cast<size_t>(codebook)] = audio_ids.tensor;
            ggml_set_input(audio_ids_[static_cast<size_t>(codebook)]);
            audio_id_values[static_cast<size_t>(codebook)] = audio_ids;
        }
        auto audio_mask =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows_, total_tokens_capacity_, 1}));
        auto text_mask =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows_, total_tokens_capacity_, 1}));
        audio_mask_ = audio_mask.tensor;
        text_mask_ = text_mask.tensor;
        ggml_set_input(audio_mask_);
        ggml_set_input(text_mask_);
        output_ = ensure_contiguous(ctx, build_embeddings(ctx, config, weights, text_ids, audio_id_values, audio_mask, text_mask)).tensor;
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_set_output(output_);
        ggml_build_forward_expand(graph_, output_);
        buffers_.allocate(ctx_.get(), runtime_->backend(), "OmniVoice layerwise embedding");
    }

    void run(
        const std::vector<int32_t> & text_ids,
        const std::array<std::vector<int32_t>, 8> & audio_ids,
        const std::vector<float> & audio_mask,
        const std::vector<float> & text_mask,
        std::vector<float> & output,
        double & compute_ms,
        double & readback_ms) {
        if (graph_ == nullptr || output_ == nullptr || runtime_ == nullptr) {
            throw std::runtime_error("OmniVoice layerwise embedding stage is not built");
        }
        ggml_backend_tensor_set(text_ids_, text_ids.data(), 0, text_ids.size() * sizeof(int32_t));
        for (int64_t codebook = 0; codebook < runtime_->assets().config.num_audio_codebook; ++codebook) {
            const auto & ids = audio_ids[static_cast<size_t>(codebook)];
            ggml_backend_tensor_set(
                audio_ids_[static_cast<size_t>(codebook)],
                ids.data(),
                0,
                ids.size() * sizeof(int32_t));
        }
        ggml_backend_tensor_set(audio_mask_, audio_mask.data(), 0, audio_mask.size() * sizeof(float));
        ggml_backend_tensor_set(text_mask_, text_mask.data(), 0, text_mask.size() * sizeof(float));
        compute_graph(compute_ms);
        const size_t hidden_count = static_cast<size_t>(
            rows_ * total_tokens_capacity_ * runtime_->assets().config.llm.hidden_size);
        if (output.size() != hidden_count) {
            output.assign(hidden_count, 0.0F);
        }
        const auto readback_start = Clock::now();
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        const auto readback_end = Clock::now();
        readback_ms += engine::debug::elapsed_ms(readback_start, readback_end);
    }

    void clear_graph() {
        if (runtime_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        }
        graph_ = nullptr;
        output_ = nullptr;
        audio_mask_ = nullptr;
        text_mask_ = nullptr;
        for (auto & tensor : audio_ids_) {
            tensor = nullptr;
        }
        text_ids_ = nullptr;
        ctx_.reset();
    }

private:
    void compute_graph(double & compute_ms) {
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OmniVoice layerwise embedding graph compute failed");
        }
        compute_ms += engine::debug::elapsed_ms(compute_start, compute_end);
    }

    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime_;
    size_t graph_arena_bytes_ = 0;
    int64_t rows_ = 0;
    int64_t total_tokens_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * text_ids_ = nullptr;
    std::array<ggml_tensor *, 8> audio_ids_{};
    ggml_tensor * audio_mask_ = nullptr;
    ggml_tensor * text_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    GraphBufferSet buffers_;
};

class LayerwiseDecoderLayerStage {
public:
    ~LayerwiseDecoderLayerStage() {
        clear_graph();
    }

    void rebuild(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        int64_t rows,
        int64_t total_token_capacity,
        int64_t layer_index) {
        clear_graph();
        runtime_ = std::move(runtime);
        graph_arena_bytes_ = graph_arena_bytes;
        rows_ = rows;
        total_tokens_capacity_ = total_token_capacity;
        layer_index_ = layer_index;
        if (runtime_ == nullptr || rows_ <= 0 || total_tokens_capacity_ <= 0) {
            throw std::runtime_error("OmniVoice layerwise decoder stage requires valid runtime and token capacity");
        }
        const auto & config = runtime_->assets().config;
        if (layer_index_ < 0 || layer_index_ >= static_cast<int64_t>(runtime_->weights().layers.size())) {
            throw std::runtime_error("OmniVoice layerwise decoder layer index is out of range");
        }

        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice layerwise decoder graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "omnivoice.generator.layerwise.decoder", runtime_->backend_type()};
        auto input =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows_, total_tokens_capacity_, config.llm.hidden_size}));
        input_ = input.tensor;
        ggml_set_input(input_);
        positions_ =
            core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({total_tokens_capacity_})).tensor;
        ggml_set_input(positions_);
        auto positions =
            core::wrap_tensor(positions_, core::TensorShape::from_dims({total_tokens_capacity_}), GGML_TYPE_I32);
        auto attention_mask =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows_, 1, total_tokens_capacity_, total_tokens_capacity_}));
        attention_mask_ = attention_mask.tensor;
        ggml_set_input(attention_mask_);
        auto output = decoder_layer(
            ctx,
            input,
            positions,
            runtime_->weights().layers[static_cast<size_t>(layer_index_)],
            config.llm,
            attention_mask);
        output_ = ensure_contiguous(ctx, output).tensor;
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_set_output(output_);
        ggml_build_forward_expand(graph_, output_);
        buffers_.allocate(ctx_.get(), runtime_->backend(), "OmniVoice layerwise decoder");
    }

    void run(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        int64_t rows,
        int64_t total_token_capacity,
        int64_t layer_index,
        const std::vector<int32_t> & positions,
        const std::vector<float> & attention_mask,
        const std::vector<float> & input,
        std::vector<float> & output,
        double & compute_ms,
        double & readback_ms) {
        rebuild(std::move(runtime), graph_arena_bytes, rows, total_token_capacity, layer_index);
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(attention_mask_, attention_mask.data(), 0, attention_mask.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OmniVoice layerwise decoder graph compute failed");
        }
        compute_ms += engine::debug::elapsed_ms(compute_start, compute_end);
        if (output.size() != input.size()) {
            output.assign(input.size(), 0.0F);
        }
        const auto readback_start = Clock::now();
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        const auto readback_end = Clock::now();
        readback_ms += engine::debug::elapsed_ms(readback_start, readback_end);
    }

    void clear_graph() {
        if (runtime_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        }
        graph_ = nullptr;
        output_ = nullptr;
        attention_mask_ = nullptr;
        positions_ = nullptr;
        input_ = nullptr;
        ctx_.reset();
    }

private:
    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime_;
    size_t graph_arena_bytes_ = 0;
    int64_t rows_ = 0;
    int64_t total_tokens_capacity_ = 0;
    int64_t layer_index_ = -1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    GraphBufferSet buffers_;
};

class LayerwiseHeadStage {
public:
    ~LayerwiseHeadStage() {
        clear_graph();
    }

    void rebuild(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        int64_t batch,
        int64_t total_token_capacity,
        int64_t target_frame_capacity) {
        clear_graph();
        runtime_ = std::move(runtime);
        graph_arena_bytes_ = graph_arena_bytes;
        batch_ = batch;
        total_tokens_capacity_ = total_token_capacity;
        target_frame_capacity_ = target_frame_capacity;
        if (runtime_ == nullptr || batch_ <= 0 || total_tokens_capacity_ <= 0 || target_frame_capacity_ <= 0) {
            throw std::runtime_error("OmniVoice layerwise head stage requires valid runtime and capacities");
        }

        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice layerwise head graph context");
        }

        const auto & config = runtime_->assets().config;
        const int64_t rows = 2 * batch_;
        core::ModuleBuildContext ctx{ctx_.get(), "omnivoice.generator.layerwise.head", runtime_->backend_type()};
        auto input = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({rows, total_tokens_capacity_, config.llm.hidden_size}));
        input_ = input.tensor;
        ggml_set_input(input_);
        auto target_indices =
            core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({rows * target_frame_capacity_}));
        target_indices_ = target_indices.tensor;
        ggml_set_input(target_indices_);
        auto guidance_scales = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_}));
        guidance_scales_ = guidance_scales.tensor;
        ggml_set_input(guidance_scales_);

        logits_ = build_guided_logits(
            ctx,
            config,
            runtime_->weights(),
            input,
            target_indices,
            guidance_scales,
            batch_,
            rows * total_tokens_capacity_,
            target_frame_capacity_).tensor;
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffers_.allocate(ctx_.get(), runtime_->backend(), "OmniVoice layerwise head");
    }

    void run(
        const std::vector<float> & hidden,
        const std::vector<int32_t> & target_indices,
        const std::vector<float> & guidance_scales,
        std::vector<float> & logits,
        double & compute_ms,
        double & readback_ms) {
        if (graph_ == nullptr || logits_ == nullptr || runtime_ == nullptr) {
            throw std::runtime_error("OmniVoice layerwise head stage is not built");
        }
        ggml_backend_tensor_set(input_, hidden.data(), 0, hidden.size() * sizeof(float));
        ggml_backend_tensor_set(target_indices_, target_indices.data(), 0, target_indices.size() * sizeof(int32_t));
        ggml_backend_tensor_set(guidance_scales_, guidance_scales.data(), 0, guidance_scales.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OmniVoice layerwise head graph compute failed");
        }
        compute_ms += engine::debug::elapsed_ms(compute_start, compute_end);
        const size_t logits_count = static_cast<size_t>(
            batch_ * target_frame_capacity_ * runtime_->assets().config.num_audio_codebook *
            runtime_->assets().config.audio_vocab_size);
        if (logits.size() != logits_count) {
            logits.assign(logits_count, 0.0F);
        }
        const auto readback_start = Clock::now();
        ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
        const auto readback_end = Clock::now();
        readback_ms += engine::debug::elapsed_ms(readback_start, readback_end);
    }

    void clear_graph() {
        if (runtime_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        }
        graph_ = nullptr;
        logits_ = nullptr;
        guidance_scales_ = nullptr;
        target_indices_ = nullptr;
        input_ = nullptr;
        ctx_.reset();
    }

private:
    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime_;
    size_t graph_arena_bytes_ = 0;
    int64_t batch_ = 0;
    int64_t total_tokens_capacity_ = 0;
    int64_t target_frame_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * target_indices_ = nullptr;
    ggml_tensor * guidance_scales_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    GraphBufferSet buffers_;
};

class Qwen3DiffusionLayerwiseGraph final : public Qwen3DiffusionGraph {
public:
    Qwen3DiffusionLayerwiseGraph(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        const std::vector<GraphItemShape> & items)
        : runtime_(std::move(runtime)),
          graph_arena_bytes_(graph_arena_bytes) {
        rebuild(items);
    }

    bool matches(const Qwen3DiffusionWeightsRuntime & runtime, const std::vector<GraphItemShape> & items) const override {
        const GraphCapacity capacity(items);
        return runtime_.get() == &runtime &&
            batch_capacity_ == capacity.batch &&
            total_tokens_capacity_ == capacity.total_tokens &&
            target_frame_capacity_ == capacity.target_frames;
    }

    void rebuild(const std::vector<GraphItemShape> & items) override {
        const GraphCapacity capacity(items);
        const int64_t batch = capacity.batch;
        const int64_t total_token_capacity = capacity.total_tokens;
        const int64_t target_frame_capacity = capacity.target_frames;
        const auto clear_start = Clock::now();
        embedding_.clear_graph();
        decoder_.clear_graph();
        head_.clear_graph();
        const auto clear_end = Clock::now();
        rebuild_clear_ms_ = engine::debug::elapsed_ms(clear_start, clear_end);

        batch_capacity_ = batch;
        total_tokens_capacity_ = total_token_capacity;
        target_frame_capacity_ = target_frame_capacity;

        const auto build_start = Clock::now();
        embedding_.rebuild(runtime_, graph_arena_bytes_, 2 * batch_capacity_, total_tokens_capacity_);
        head_.rebuild(runtime_, graph_arena_bytes_, batch_capacity_, total_tokens_capacity_, target_frame_capacity_);
        const auto build_end = Clock::now();
        rebuild_build_ms_ = engine::debug::elapsed_ms(build_start, build_end);
        rebuild_alloc_ms_ = 0.0;

        const auto init_start = Clock::now();
        // The layerwise stages use the unpacked [2 * batch, tokens] layout.
        host_.reset(
            runtime_->assets(),
            std::vector<int64_t>(static_cast<size_t>(batch_capacity_), total_tokens_capacity_),
            std::vector<int64_t>(static_cast<size_t>(batch_capacity_), total_tokens_capacity_),
            target_frame_capacity_,
            false);
        hidden_a_.clear();
        hidden_b_.clear();
        logits_host_.clear();
        rebuild_init_ms_ = engine::debug::elapsed_ms(init_start, Clock::now());
    }

    void prepare_request(const std::vector<PackedInputs> & inputs, const std::vector<float> & guidance_scales) override {
        // Every stage reads the host copies on each compute, so nothing is
        // uploaded here.
        host_.prepare(inputs, guidance_scales);
        (void) host_.take_layout_dirty();
        (void) host_.take_guidance_dirty();
    }

    void update_generated_target_tokens(const std::vector<PackedInputs> & inputs) override {
        host_.update_targets(inputs);
    }

    bool computes_best_tokens() const noexcept override {
        return false;
    }

    GraphStepOutputs compute(bool, bool read_best_tokens, double & compute_ms, double & readback_ms) override {
        if (read_best_tokens) {
            throw std::runtime_error("OmniVoice layerwise generator does not score on the device");
        }
        if (!host_.prepared()) {
            throw std::runtime_error("OmniVoice layerwise generator compute requires a prepared request");
        }
        embedding_.run(
            host_.text_ids,
            host_.audio_ids,
            host_.audio_mask,
            host_.text_mask,
            hidden_a_,
            compute_ms,
            readback_ms);
        auto * input = &hidden_a_;
        auto * output = &hidden_b_;
        const int64_t layers = static_cast<int64_t>(runtime_->weights().layers.size());
        for (int64_t layer = 0; layer < layers; ++layer) {
            decoder_.run(
                runtime_,
                graph_arena_bytes_,
                2 * batch_capacity_,
                total_tokens_capacity_,
                layer,
                host_.positions,
                host_.attention_f32,
                *input,
                *output,
                compute_ms,
                readback_ms);
            std::swap(input, output);
        }
        head_.run(
            *input,
            host_.target_indices,
            host_.guidance_scales,
            logits_host_,
            compute_ms,
            readback_ms);
        GraphStepOutputs out;
        out.logits = &logits_host_;
        return out;
    }

    int64_t batch_capacity() const noexcept override { return batch_capacity_; }
    int64_t total_token_capacity() const noexcept override { return total_tokens_capacity_; }
    int64_t target_frame_capacity() const noexcept override { return target_frame_capacity_; }
    double rebuild_clear_ms() const noexcept override { return rebuild_clear_ms_; }
    double rebuild_build_ms() const noexcept override { return rebuild_build_ms_; }
    double rebuild_alloc_ms() const noexcept override { return rebuild_alloc_ms_; }
    double rebuild_init_ms() const noexcept override { return rebuild_init_ms_; }

private:
    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime_;
    size_t graph_arena_bytes_ = 0;
    int64_t batch_capacity_ = 0;
    int64_t total_tokens_capacity_ = 0;
    int64_t target_frame_capacity_ = 0;
    BatchedGraphInputs host_;
    std::vector<float> hidden_a_;
    std::vector<float> hidden_b_;
    std::vector<float> logits_host_;
    double rebuild_clear_ms_ = 0.0;
    double rebuild_build_ms_ = 0.0;
    double rebuild_alloc_ms_ = 0.0;
    double rebuild_init_ms_ = 0.0;
    LayerwiseEmbeddingStage embedding_;
    LayerwiseDecoderLayerStage decoder_;
    LayerwiseHeadStage head_;
};
