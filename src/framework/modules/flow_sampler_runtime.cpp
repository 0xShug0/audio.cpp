#include "engine/framework/modules/flow_sampler_runtime.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace engine::modules {
namespace {

int64_t element_count(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        return 0;
    }
    int64_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        count *= dim;
    }
    return count;
}

bool has_duplicate_cache_name(const std::vector<FlowSamplerCacheSpec> & values) {
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = i + 1; j < values.size(); ++j) {
            if (values[i].name == values[j].name) {
                return true;
            }
        }
    }
    return false;
}

bool has_duplicate_branch_name(const std::vector<FlowSamplerBranchSpec> & values) {
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = i + 1; j < values.size(); ++j) {
            if (values[i].name == values[j].name) {
                return true;
            }
        }
    }
    return false;
}

bool has_duplicate_prediction(const std::vector<FlowSamplerBranchPrediction> & values) {
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = i + 1; j < values.size(); ++j) {
            if (values[i].branch == values[j].branch) {
                return true;
            }
        }
    }
    return false;
}

bool has_duplicate_cache_update(const std::vector<FlowSamplerCacheUpdate> & values) {
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = i + 1; j < values.size(); ++j) {
            if (values[i].name == values[j].name) {
                return true;
            }
        }
    }
    return false;
}

bool cache_key_matches_config(
    const std::vector<FlowSamplerCacheKey> & key,
    const std::vector<FlowSamplerCacheSpec> & config) {
    if (key.size() != config.size()) {
        return false;
    }
    for (const auto & spec : config) {
        const auto it = std::find_if(key.begin(), key.end(), [&](const FlowSamplerCacheKey & state) {
            return state.name == spec.name && state.mode == spec.mode;
        });
        if (it == key.end()) {
            return false;
        }
    }
    return true;
}

const FlowSamplerBranchPrediction & require_prediction(
    const std::vector<FlowSamplerBranchPrediction> & predictions,
    const std::string & branch,
    const std::string & label) {
    const auto it = std::find_if(predictions.begin(), predictions.end(), [&](const FlowSamplerBranchPrediction & pred) {
        return pred.branch == branch;
    });
    if (it == predictions.end()) {
        throw std::runtime_error(label + " missing prediction for branch '" + branch + "'");
    }
    return *it;
}

}  // namespace

bool FlowSamplerCacheKey::operator==(const FlowSamplerCacheKey & other) const {
    return name == other.name && mode == other.mode;
}

bool FlowSamplerCacheState::operator==(const FlowSamplerCacheState & other) const {
    return name == other.name &&
           mode == other.mode &&
           revision == other.revision &&
           initialized == other.initialized;
}

bool FlowSamplerGraphKey::operator==(const FlowSamplerGraphKey & other) const {
    return latent_shape == other.latent_shape &&
           branch_count == other.branch_count &&
           schedule_steps == other.schedule_steps &&
           sampler_mode == other.sampler_mode &&
           caches == other.caches &&
           modulation_revision == other.modulation_revision;
}

FlowSamplerDenoiserRuntime::~FlowSamplerDenoiserRuntime() = default;

FlowSamplerUpdateRuntime::~FlowSamplerUpdateRuntime() = default;

class FlowSamplerEulerUpdate final : public FlowSamplerUpdateRuntime {
public:
    void update_latent(const FlowSamplerUpdateInput & input) override {
        const float dt = input.state.schedule.t_next - input.state.schedule.t;
        if (input.prediction.size() != input.latent.size()) {
            throw std::runtime_error("FlowSampler Euler update prediction shape mismatch");
        }
        for (size_t i = 0; i < input.latent.size(); ++i) {
            input.latent[i] += dt * input.prediction[i];
        }
    }
};

std::unique_ptr<FlowSamplerUpdateRuntime> make_flow_sampler_euler_update() {
    return std::make_unique<FlowSamplerEulerUpdate>();
}

FlowSamplerSingleBranchDenoiserRuntime::FlowSamplerSingleBranchDenoiserRuntime(
    FlowSamplerSingleBranchDenoiserRuntimeConfig config)
    : config_(std::move(config)) {
    if (config_.label.empty()) {
        throw std::runtime_error("FlowSampler single-branch denoiser requires label");
    }
    if (config_.branch_name.empty()) {
        throw std::runtime_error(config_.label + " requires branch name");
    }
}

FlowSamplerSingleBranchDenoiserRuntime::~FlowSamplerSingleBranchDenoiserRuntime() = default;

void FlowSamplerSingleBranchDenoiserRuntime::reset_sampler_caches(
    const std::vector<FlowSamplerCacheState> & caches) {
    reset_sequence_caches(caches);
}

std::vector<FlowSamplerCacheUpdate>
FlowSamplerSingleBranchDenoiserRuntime::begin_sampler_sequence(
    const FlowSamplerSequenceState & state) {
    sequence_latent_shape_ = state.latent_shape;
    sequence_schedule_steps_ = static_cast<int64_t>(state.schedule.size());
    return begin_sequence(state);
}

FlowSamplerGraphKey FlowSamplerSingleBranchDenoiserRuntime::sampler_graph_key(
    const FlowSamplerStepState & state) {
    FlowSamplerGraphKey key;
    key.latent_shape = sequence_latent_shape_;
    key.branch_count = static_cast<int64_t>(state.branches.size());
    key.schedule_steps = sequence_schedule_steps_;
    key.sampler_mode = sampler_mode(state);
    for (const auto & cache : state.caches) {
        key.caches.push_back({cache.name, cache.mode});
    }
    key.modulation_revision = modulation_revision(state);
    return key;
}

void FlowSamplerSingleBranchDenoiserRuntime::rebuild_sampler_graph(
    const FlowSamplerGraphKey & key,
    const FlowSamplerStepState & state) {
    rebuild_graph(key, state);
}

FlowSamplerDenoiserOutput FlowSamplerSingleBranchDenoiserRuntime::run_sampler_denoiser(
    const FlowSamplerDenoiserInput & input) {
    FlowSamplerDenoiserOutput output;
    output.predictions.push_back({config_.branch_name, predict_branch(input)});
    return output;
}

void FlowSamplerSingleBranchDenoiserRuntime::release_sampler_graphs() {
    release_runtime_graphs();
}

void FlowSamplerSingleBranchDenoiserRuntime::reset_sequence_caches(
    const std::vector<FlowSamplerCacheState> & caches) {
    if (!caches.empty()) {
        throw std::runtime_error(config_.label + " does not define cache reset handling");
    }
}

std::vector<FlowSamplerCacheUpdate>
FlowSamplerSingleBranchDenoiserRuntime::begin_sequence(
    const FlowSamplerSequenceState & state) {
    if (!state.caches.empty()) {
        throw std::runtime_error(config_.label + " does not define cache sequence handling");
    }
    return {};
}

int64_t FlowSamplerSingleBranchDenoiserRuntime::modulation_revision(
    const FlowSamplerStepState &) const {
    return 0;
}

void FlowSamplerSingleBranchDenoiserRuntime::release_runtime_graphs() {}

const FlowSamplerSingleBranchDenoiserRuntimeConfig &
FlowSamplerSingleBranchDenoiserRuntime::config() const noexcept {
    return config_;
}

class SingleBranchFlowSamplerDenoiser final : public FlowSamplerSingleBranchDenoiserRuntime {
public:
    explicit SingleBranchFlowSamplerDenoiser(FlowSamplerSingleBranchDenoiserConfig config)
        : FlowSamplerSingleBranchDenoiserRuntime({config.label, config.branch_name}),
          config_(std::move(config)) {
        if (!config_.sampler_mode) {
            throw std::runtime_error(config_.label + " requires sampler mode callback");
        }
        if (!config_.predict) {
            throw std::runtime_error(config_.label + " requires prediction callback");
        }
    }

private:
    void reset_sequence_caches(const std::vector<FlowSamplerCacheState> & caches) override {
        if (config_.reset_caches) {
            config_.reset_caches(caches);
            return;
        }
        if (!caches.empty()) {
            throw std::runtime_error(config_.label + " does not define cache reset handling");
        }
    }

    std::vector<FlowSamplerCacheUpdate> begin_sequence(
        const FlowSamplerSequenceState & state) override {
        if (config_.begin_sequence) {
            return config_.begin_sequence(state);
        }
        if (!state.caches.empty()) {
            throw std::runtime_error(config_.label + " does not define cache sequence handling");
        }
        return {};
    }

    std::string sampler_mode(const FlowSamplerStepState & state) const override {
        return config_.sampler_mode(state);
    }

    int64_t modulation_revision(const FlowSamplerStepState & state) const override {
        if (config_.modulation_revision) {
            return config_.modulation_revision(state);
        }
        return 0;
    }

    void rebuild_graph(
        const FlowSamplerGraphKey & key,
        const FlowSamplerStepState & state) override {
        if (config_.rebuild_graph) {
            config_.rebuild_graph(key, state);
        }
    }

    std::vector<float> predict_branch(const FlowSamplerDenoiserInput & input) override {
        return config_.predict(input);
    }

    void release_runtime_graphs() override {
        if (config_.release_graphs) {
            config_.release_graphs();
        }
    }

    FlowSamplerSingleBranchDenoiserConfig config_;
};

std::unique_ptr<FlowSamplerDenoiserRuntime>
make_flow_sampler_single_branch_denoiser(FlowSamplerSingleBranchDenoiserConfig config) {
    return std::make_unique<SingleBranchFlowSamplerDenoiser>(std::move(config));
}

class FlowSamplerRuntime::Impl {
public:
    Impl(
        FlowSamplerRuntimeConfig config,
        std::unique_ptr<FlowSamplerDenoiserRuntime> denoiser,
        std::unique_ptr<FlowSamplerUpdateRuntime> updater,
        bool default_euler_updater)
        : config_(std::move(config)),
          denoiser_(std::move(denoiser)),
          updater_(std::move(updater)) {
        validate_config();
        if (denoiser_ == nullptr) {
            throw std::runtime_error(config_.label + " requires denoiser runtime");
        }
        if (updater_ == nullptr) {
            throw std::runtime_error(config_.label + " requires latent update runtime");
        }
        if (default_euler_updater && config_.update_rule == FlowSamplerUpdateRule::Custom) {
            throw std::runtime_error(config_.label + " custom update rule requires explicit latent update runtime");
        }
        latent_ = config_.initial_latent;
        preserved_latent_ = config_.initial_latent;
        cache_states_.reserve(config_.caches.size());
        for (const auto & cache : config_.caches) {
            cache_states_.push_back({cache.name, cache.mode, 0, false});
        }
    }

    ~Impl() {
        release_runtime_graphs();
    }

    void run_sequence() {
        if (config_.initial_latent.empty()) {
            throw std::runtime_error(config_.label + " requires initial latent input");
        }
        run_sequence(config_.initial_latent);
    }

    void run_sequence(const std::vector<float> & initial_latent) {
        validate_initial_latent(initial_latent);
        reset_sequence_caches();
        const auto begin_updates = denoiser_->begin_sampler_sequence(make_sequence_state());
        apply_cache_updates(begin_updates);
        latent_ = initial_latent;
        preserved_latent_ = initial_latent;
        for (int64_t i = 0; i < static_cast<int64_t>(config_.schedule.size()); ++i) {
            auto state = make_step_state(i);
            ensure_graph(state);
            auto output = denoiser_->run_sampler_denoiser({state, latent_});
            validate_denoiser_output(state, output);
            const auto combined = combine_predictions(output.predictions);
            apply_update(state, combined);
            preserve_prefix();
            apply_cache_updates(output.cache_updates);
        }
        if (config_.release_graph_after_sequence) {
            release_runtime_graphs();
        }
    }

    void release_runtime_graphs() {
        if (denoiser_ != nullptr) {
            denoiser_->release_sampler_graphs();
        }
        active_graph_key_.reset();
    }

    void reset_graph_reuse_state() {
        active_graph_key_.reset();
    }

    void reset_runtime_caches() {
        for (auto & state : cache_states_) {
            ++state.revision;
            state.initialized = false;
        }
        denoiser_->reset_sampler_caches(cache_states_);
        reset_graph_reuse_state();
    }

    const std::vector<float> & latent() const noexcept {
        return latent_;
    }

    const FlowSamplerRuntimeConfig & config() const noexcept {
        return config_;
    }

    const std::optional<FlowSamplerGraphKey> & active_graph_key() const noexcept {
        return active_graph_key_;
    }

    const std::vector<FlowSamplerCacheState> & cache_states() const noexcept {
        return cache_states_;
    }

private:
    void validate_config() const {
        const int64_t latent_values = element_count(config_.latent_shape);
        if (config_.label.empty()) {
            throw std::runtime_error("FlowSamplerRuntime requires label");
        }
        if (latent_values <= 0) {
            throw std::runtime_error(config_.label + " requires positive latent shape");
        }
        if (!config_.initial_latent.empty() &&
            config_.initial_latent.size() != static_cast<size_t>(latent_values)) {
            throw std::runtime_error(config_.label + " initial latent shape mismatch");
        }
        if (!config_.prefix.preserve_mask.empty() &&
            config_.prefix.preserve_mask.size() != static_cast<size_t>(latent_values)) {
            throw std::runtime_error(config_.label + " prefix preserve mask shape mismatch");
        }
        if (config_.schedule.empty()) {
            throw std::runtime_error(config_.label + " requires non-empty schedule");
        }
        if (config_.branches.empty()) {
            throw std::runtime_error(config_.label + " requires at least one branch");
        }
        if (has_duplicate_branch_name(config_.branches)) {
            throw std::runtime_error(config_.label + " branch names must be unique");
        }
        for (const auto & branch : config_.branches) {
            if (branch.name.empty()) {
                throw std::runtime_error(config_.label + " branch name must not be empty");
            }
        }
        if (config_.guidance.mode == FlowSamplerGuidanceMode::ClassifierFree) {
            if (config_.guidance.cond_branch.empty() || config_.guidance.uncond_branch.empty()) {
                throw std::runtime_error(config_.label + " CFG requires cond and uncond branches");
            }
        }
        if (has_duplicate_cache_name(config_.caches)) {
            throw std::runtime_error(config_.label + " cache names must be unique");
        }
        for (const auto & cache : config_.caches) {
            if (cache.name.empty()) {
                throw std::runtime_error(config_.label + " cache name must not be empty");
            }
            if (cache.mode.empty()) {
                throw std::runtime_error(config_.label + " cache mode must not be empty");
            }
        }
    }

    void validate_initial_latent(const std::vector<float> & initial_latent) const {
        const int64_t latent_values = element_count(config_.latent_shape);
        if (initial_latent.size() != static_cast<size_t>(latent_values)) {
            throw std::runtime_error(config_.label + " initial latent shape mismatch");
        }
    }

    void validate_graph_key(const FlowSamplerGraphKey & key) const {
        if (!config_.require_complete_graph_key) {
            return;
        }
        if (key.latent_shape != config_.latent_shape) {
            throw std::runtime_error(config_.label + " graph key latent shape mismatch");
        }
        if (key.branch_count != static_cast<int64_t>(config_.branches.size())) {
            throw std::runtime_error(config_.label + " graph key branch count mismatch");
        }
        if (key.schedule_steps != static_cast<int64_t>(config_.schedule.size())) {
            throw std::runtime_error(config_.label + " graph key schedule length mismatch");
        }
        if (key.sampler_mode.empty()) {
            throw std::runtime_error(config_.label + " graph key requires sampler mode");
        }
        if (!cache_key_matches_config(key.caches, config_.caches)) {
            throw std::runtime_error(config_.label + " graph key cache state mismatch");
        }
    }

    FlowSamplerStepState make_step_state(int64_t sequence_index) const {
        FlowSamplerStepState state;
        state.sequence_index = sequence_index;
        state.schedule = config_.schedule[static_cast<size_t>(sequence_index)];
        state.branches = config_.branches;
        state.caches = cache_states_;
        return state;
    }

    FlowSamplerSequenceState make_sequence_state() const {
        FlowSamplerSequenceState state;
        state.latent_shape = config_.latent_shape;
        state.schedule = config_.schedule;
        state.branches = config_.branches;
        state.caches = cache_states_;
        return state;
    }

    void ensure_graph(FlowSamplerStepState & state) {
        state.graph_key = denoiser_->sampler_graph_key(state);
        validate_graph_key(state.graph_key);
        if (!active_graph_key_.has_value() || *active_graph_key_ != state.graph_key) {
            denoiser_->rebuild_sampler_graph(state.graph_key, state);
            active_graph_key_ = state.graph_key;
        }
    }

    void validate_denoiser_output(
        const FlowSamplerStepState & state,
        const FlowSamplerDenoiserOutput & output) const {
        if (output.predictions.size() != state.branches.size()) {
            throw std::runtime_error(config_.label + " denoiser branch count mismatch");
        }
        if (has_duplicate_prediction(output.predictions)) {
            throw std::runtime_error(config_.label + " denoiser output contains duplicate branches");
        }
        for (const auto & prediction : output.predictions) {
            const auto branch = std::find_if(state.branches.begin(), state.branches.end(), [&](const FlowSamplerBranchSpec & spec) {
                return spec.name == prediction.branch;
            });
            if (branch == state.branches.end()) {
                throw std::runtime_error(config_.label + " denoiser output references unknown branch '" + prediction.branch + "'");
            }
            if (prediction.values.size() != latent_.size()) {
                throw std::runtime_error(config_.label + " denoiser prediction shape mismatch");
            }
        }
        if (has_duplicate_cache_update(output.cache_updates)) {
            throw std::runtime_error(config_.label + " denoiser output contains duplicate cache updates");
        }
        for (const auto & update : output.cache_updates) {
            const auto it = std::find_if(config_.caches.begin(), config_.caches.end(), [&](const FlowSamplerCacheSpec & spec) {
                return spec.name == update.name;
            });
            if (it == config_.caches.end()) {
                throw std::runtime_error(config_.label + " denoiser output references unknown cache '" + update.name + "'");
            }
        }
    }

    std::vector<float> combine_predictions(const std::vector<FlowSamplerBranchPrediction> & predictions) const {
        if (config_.guidance.mode == FlowSamplerGuidanceMode::None) {
            if (predictions.size() != 1) {
                throw std::runtime_error(config_.label + " unguided sampler expects exactly one prediction");
            }
            return predictions.front().values;
        }
        if (config_.guidance.mode == FlowSamplerGuidanceMode::ClassifierFree) {
            const auto & cond = require_prediction(predictions, config_.guidance.cond_branch, config_.label);
            const auto & uncond = require_prediction(predictions, config_.guidance.uncond_branch, config_.label);
            std::vector<float> out(cond.values.size(), 0.0F);
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = uncond.values[i] + config_.guidance.scale * (cond.values[i] - uncond.values[i]);
            }
            return out;
        }
        throw std::runtime_error(config_.label + " unsupported guidance mode");
    }

    void apply_update(
        const FlowSamplerStepState & state,
        const std::vector<float> & prediction) {
        if (config_.prediction_type != FlowSamplerPredictionType::Velocity) {
            throw std::runtime_error(config_.label + " unsupported flow update");
        }
        if (config_.update_rule != FlowSamplerUpdateRule::Euler &&
            config_.update_rule != FlowSamplerUpdateRule::Custom) {
            throw std::runtime_error(config_.label + " unsupported flow update rule");
        }
        updater_->update_latent({state, prediction, latent_});
    }

    void preserve_prefix() {
        if (config_.prefix.preserve_mask.empty()) {
            return;
        }
        for (size_t i = 0; i < latent_.size(); ++i) {
            if (config_.prefix.preserve_mask[i] != 0.0F) {
                latent_[i] = preserved_latent_[i];
            }
        }
    }

    void reset_sequence_caches() {
        for (size_t i = 0; i < config_.caches.size(); ++i) {
            if (config_.caches[i].reset_on_sequence_begin) {
                ++cache_states_[i].revision;
                cache_states_[i].initialized = false;
            }
        }
    }

    void apply_cache_updates(const std::vector<FlowSamplerCacheUpdate> & updates) {
        for (const auto & update : updates) {
            auto it = std::find_if(cache_states_.begin(), cache_states_.end(), [&](const FlowSamplerCacheState & state) {
                return state.name == update.name;
            });
            if (it == cache_states_.end()) {
                throw std::runtime_error(config_.label + " received update for unknown cache '" + update.name + "'");
            }
            switch (update.kind) {
                case FlowSamplerCacheUpdateKind::Unchanged:
                    break;
                case FlowSamplerCacheUpdateKind::Updated:
                    ++it->revision;
                    it->initialized = true;
                    break;
                case FlowSamplerCacheUpdateKind::Reset:
                    ++it->revision;
                    it->initialized = false;
                    break;
            }
        }
    }

    FlowSamplerRuntimeConfig config_;
    std::unique_ptr<FlowSamplerDenoiserRuntime> denoiser_;
    std::unique_ptr<FlowSamplerUpdateRuntime> updater_;
    std::optional<FlowSamplerGraphKey> active_graph_key_;
    std::vector<FlowSamplerCacheState> cache_states_;
    std::vector<float> latent_;
    std::vector<float> preserved_latent_;
};

FlowSamplerRuntime::FlowSamplerRuntime(
    FlowSamplerRuntimeConfig config,
    std::unique_ptr<FlowSamplerDenoiserRuntime> denoiser)
    : impl_(std::make_unique<Impl>(
          std::move(config),
          std::move(denoiser),
          make_flow_sampler_euler_update(),
          true)) {}

FlowSamplerRuntime::FlowSamplerRuntime(
    FlowSamplerRuntimeConfig config,
    std::unique_ptr<FlowSamplerDenoiserRuntime> denoiser,
    std::unique_ptr<FlowSamplerUpdateRuntime> updater)
    : impl_(std::make_unique<Impl>(
          std::move(config),
          std::move(denoiser),
          std::move(updater),
          false)) {}

FlowSamplerRuntime::~FlowSamplerRuntime() = default;

void FlowSamplerRuntime::run_sequence() {
    impl_->run_sequence();
}

void FlowSamplerRuntime::run_sequence(const std::vector<float> & initial_latent) {
    impl_->run_sequence(initial_latent);
}

void FlowSamplerRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

void FlowSamplerRuntime::reset_graph_reuse_state() {
    impl_->reset_graph_reuse_state();
}

void FlowSamplerRuntime::reset_runtime_caches() {
    impl_->reset_runtime_caches();
}

const std::vector<float> & FlowSamplerRuntime::latent() const noexcept {
    return impl_->latent();
}

const FlowSamplerRuntimeConfig & FlowSamplerRuntime::config() const noexcept {
    return impl_->config();
}

const std::optional<FlowSamplerGraphKey> & FlowSamplerRuntime::active_graph_key() const noexcept {
    return impl_->active_graph_key();
}

const std::vector<FlowSamplerCacheState> & FlowSamplerRuntime::cache_states() const noexcept {
    return impl_->cache_states();
}

}  // namespace engine::modules
