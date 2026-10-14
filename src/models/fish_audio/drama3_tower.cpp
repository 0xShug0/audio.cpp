#include "engine/models/fish_audio/drama3_tower.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace engine::models::fish_audio {
namespace {
namespace binding = engine::modules::binding;

struct ContextDeleter { void operator()(ggml_context * p) const { if (p) ggml_free(p); } };
using Context = std::unique_ptr<ggml_context, ContextDeleter>;

std::vector<float> row(const assets::TensorDataF32 & data, int64_t index, int64_t width) {
    if (index < 0 || (index + 1) * width > static_cast<int64_t>(data.values.size())) {
        throw std::runtime_error("Drama 3 embedding index is out of range");
    }
    return {data.values.begin() + index * width, data.values.begin() + (index + 1) * width};
}

struct TowerWeights {
    assets::TensorDataF32 embedding;
    modules::NormWeights enorm, hnorm, anorm, fnorm, final_norm, qnorm, knorm;
    modules::LinearWeights eh, qkv, out, gate, up, down;
    std::optional<modules::LinearWeights> head;
};

class TowerLayer {
public:
    TowerLayer(core::ExecutionContext & execution, const FishAudioConfig & config,
        std::shared_ptr<core::BackendWeightStore> store, TowerWeights weights)
        : execution_(execution), config_(config), store_(std::move(store)), weights_(std::move(weights)) {}
    ~TowerLayer() { release(); }

    int64_t capacity() const { return capacity_; }
    void release_runtime() { release(); }
    void reset(int64_t capacity) {
        valid_ = 0;
        if (capacity <= capacity_) {
            ggml_backend_tensor_memset(key_,0,0,ggml_nbytes(key_));
            ggml_backend_tensor_memset(value_,0,0,ggml_nbytes(value_));
            return;
        }
        release();
        capacity_ = ((capacity + 255) / 256) * 256;
        ggml_init_params params{1024 * 1024, nullptr, true};
        cache_ctx_.reset(ggml_init(params));
        if (!cache_ctx_) throw std::runtime_error("Drama 3 tower cache context allocation failed");
        const int64_t kv = config_.fast.n_local_heads * config_.fast.head_dim;
        key_ = ggml_new_tensor_2d(cache_ctx_.get(), GGML_TYPE_F32, kv, capacity_);
        value_ = ggml_new_tensor_2d(cache_ctx_.get(), GGML_TYPE_F32, kv, capacity_);
        cache_buffer_ = ggml_backend_alloc_ctx_tensors(cache_ctx_.get(), execution_.backend());
        if (!cache_buffer_) throw std::runtime_error("Drama 3 tower cache allocation failed");
        ggml_backend_tensor_memset(key_, 0, 0, ggml_nbytes(key_));
        ggml_backend_tensor_memset(value_, 0, 0, ggml_nbytes(value_));
    }

    ggml_tensor * step_device(ggml_tensor * hidden, int32_t code, bool semantic,
        int64_t position, const std::vector<float> * extra) {
        const int64_t width = config_.fast.dim;
        if (position != valid_ || position >= capacity_ || !hidden || ggml_nelements(hidden) != width || hidden->type!=GGML_TYPE_F32) {
            throw std::runtime_error("Drama 3 tower cache transition or hidden shape is invalid");
        }
        ensure_graph();
        std::vector<float> embedding(static_cast<size_t>(width), 0.0F);
        if (semantic) embedding = row(weights_.embedding, code, width);
        if (extra) {
            if (extra->size() != embedding.size()) throw std::runtime_error("Drama 3 shared input size mismatch");
            for (size_t i = 0; i < embedding.size(); ++i) embedding[i] += (*extra)[i];
        }
        ggml_backend_tensor_copy_async(execution_.backend(),execution_.backend(),hidden,hidden_);
        ggml_backend_tensor_set(embedding_, embedding.data(), 0, embedding.size() * sizeof(float));
        const int32_t pos32 = static_cast<int32_t>(position);
        ggml_backend_tensor_set(position_, &pos32, 0, sizeof(pos32));
        ggml_backend_tensor_set(slot_, &position, 0, sizeof(position));
        std::vector<float> mask(static_cast<size_t>(capacity_), -std::numeric_limits<float>::infinity());
        std::fill(mask.begin(), mask.begin() + position + 1, 0.0F);
        ggml_backend_tensor_set(mask_, mask.data(), 0, mask.size() * sizeof(float));
        if(execution_.backend_type()==core::BackendType::Vulkan && std::getenv("GGML_VK_PERF_LOGGER"))
            ggml_backend_synchronize(execution_.backend());
        if (core::compute_backend_graph(execution_.backend(), graph_, nullptr, "fish_audio.drama3.tower") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Drama 3 tower computation failed");
        }
        ++valid_;
        return output_;
    }

    std::vector<float> logits(const std::vector<float> & hidden) {
        if (!weights_.head || !head_output_) throw std::runtime_error("Drama 3 residual layer has no codebook head");
        // The head belongs to the same graph; its output corresponds to the preceding step.
        (void) hidden;
        return core::read_tensor_f32(head_output_);
    }

    void release_graph() {
        if (execution_.backend()) ggml_backend_synchronize(execution_.backend());
        if(graph_)core::release_backend_graph_resources(execution_.backend(),graph_,true);
        if (allocator_) { ggml_gallocr_free(allocator_); allocator_ = nullptr; }
        if (input_buffer_) { ggml_backend_buffer_free(input_buffer_); input_buffer_ = nullptr; }
        graph_ctx_.reset(); input_ctx_.reset(); graph_ = nullptr; head_output_ = nullptr;
    }

private:
    friend class TowerSequenceGraph;
    core::TensorValue norm(core::ModuleBuildContext & ctx, ggml_tensor * input, const modules::NormWeights & weights,
        int64_t width, const core::TensorShape & shape) {
        return modules::GemmaRMSNormModule({width, config_.fast.norm_eps, true, false}).build(
            ctx, core::wrap_tensor(input, shape, GGML_TYPE_F32), weights);
    }
    ggml_tensor * linear(core::ModuleBuildContext & ctx, ggml_tensor * input,
        const modules::LinearWeights & weights, int64_t in, int64_t out) {
        return modules::LinearModule({in, out, false, GGML_PREC_F32}).build(ctx,
            core::wrap_tensor(input, core::TensorShape::from_dims({1, 1, in}), GGML_TYPE_F32), weights).tensor;
    }
    std::pair<ggml_tensor *, ggml_tensor *> build_layer(core::ModuleBuildContext & build,
        ggml_tensor * hidden, ggml_tensor * embedding, ggml_tensor * position,
        ggml_tensor * slot, ggml_tensor * mask, bool head) {
        auto * ctx = build.ggml;
        const auto & c = config_.fast;
        const int64_t d = c.head_dim, qwidth = c.n_head * d, kvwidth = c.n_local_heads * d;
        const auto shape = core::TensorShape::from_dims({1, 1, c.dim});
        auto e = norm(build, embedding, weights_.enorm, c.dim, shape);
        auto h = norm(build, hidden, weights_.hnorm, c.dim, shape);
        auto mixed = linear(build, ggml_concat(ctx, e.tensor, h.tensor, 0), weights_.eh, 2 * c.dim, c.dim);
        auto attention_input = norm(build, mixed, weights_.anorm, c.dim, shape);
        auto qkv = linear(build, attention_input.tensor, weights_.qkv, c.dim, 2 * qwidth + 2 * kvwidth);
        auto q = ggml_view_3d(ctx, qkv, d, c.n_head, 1, d * sizeof(float), qwidth * sizeof(float), 0);
        auto gate = ggml_view_3d(ctx, qkv, qwidth, 1, 1, qwidth * sizeof(float), qwidth * sizeof(float), qwidth * sizeof(float));
        auto k = ggml_view_3d(ctx, qkv, d, c.n_local_heads, 1, d * sizeof(float), kvwidth * sizeof(float), 2 * qwidth * sizeof(float));
        auto v = ggml_view_3d(ctx, qkv, d, c.n_local_heads, 1, d * sizeof(float), kvwidth * sizeof(float), (2 * qwidth + kvwidth) * sizeof(float));
        q = norm(build, q, weights_.qnorm, d, core::TensorShape::from_dims({1, c.n_head, d})).tensor;
        k = norm(build, k, weights_.knorm, d, core::TensorShape::from_dims({1, c.n_local_heads, d})).tensor;
        const int rotary = static_cast<int>(std::llround(d * config_.drama3->tower_rotary_percent));
        q = ggml_rope_ext(ctx, q, position, nullptr, rotary, GGML_ROPE_TYPE_NORMAL, 0, c.rope_base, 1.0F, 0.0F, 1.0F, 32.0F, 1.0F);
        k = ggml_rope_ext(ctx, k, position, nullptr, rotary, GGML_ROPE_TYPE_NORMAL, 0, c.rope_base, 1.0F, 0.0F, 1.0F, 32.0F, 1.0F);
        k = ggml_reshape_2d(ctx, ggml_cont(ctx, k), kvwidth, 1);
        v = ggml_reshape_2d(ctx, ggml_cont(ctx, v), kvwidth, 1);
        auto written_k = ggml_set_rows(ctx, key_, k, slot);
        auto written_v = ggml_set_rows(ctx, value_, v, slot);
        auto query = ggml_permute(ctx, q, 0, 2, 1, 3);
        auto keys = ggml_permute(ctx, ggml_reshape_3d(ctx, written_k, d, c.n_local_heads, capacity_), 0, 2, 1, 3);
        auto values = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, written_v, d, c.n_local_heads, capacity_), 1, 2, 0, 3));
        auto scores = ggml_mul_mat(ctx, keys, query);
        ggml_mul_mat_set_prec(scores,GGML_PREC_F32);
        scores = ggml_soft_max_ext(ctx, scores, mask, 1.0F / std::sqrt(static_cast<float>(d)), 0.0F);
        auto attended = ggml_mul_mat(ctx, values, scores);
        ggml_mul_mat_set_prec(attended,GGML_PREC_F32);
        attended = ggml_reshape_3d(ctx, ggml_cont(ctx, attended), qwidth, 1, 1);
        attended = ggml_mul(ctx, attended, ggml_sigmoid(ctx, gate));
        auto residual = ggml_add(ctx, mixed, linear(build, attended, weights_.out, qwidth, c.dim));
        auto ffn_input = norm(build, residual, weights_.fnorm, c.dim, shape);
        auto fg = linear(build, ffn_input.tensor, weights_.gate, c.dim, c.intermediate_size);
        auto fu = linear(build, ffn_input.tensor, weights_.up, c.dim, c.intermediate_size);
        auto down = linear(build, ggml_mul(ctx, ggml_silu(ctx, fg), fu), weights_.down, c.intermediate_size, c.dim);
        auto * output = norm(build, ggml_add(ctx, residual, down), weights_.final_norm, c.dim, shape).tensor;
        auto * logits = head && weights_.head ? linear(build, output, *weights_.head, c.dim, c.vocab_size) : nullptr;
        return {output, logits};
    }
    void ensure_graph() {
        if (graph_) return;
        ggml_init_params params{8 * 1024 * 1024, nullptr, true};
        graph_ctx_.reset(ggml_init(params));
        ggml_init_params inputs{1024 * 1024, nullptr, true};
        input_ctx_.reset(ggml_init(inputs));
        if (!graph_ctx_ || !input_ctx_) throw std::runtime_error("Drama 3 tower graph context allocation failed");
        auto * ctx = graph_ctx_.get();
        auto * ictx = input_ctx_.get();
        const auto & c = config_.fast;
        core::ModuleBuildContext build{ctx, "fish_audio.drama3.tower", execution_.backend_type()};
        hidden_ = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, c.dim, 1, 1);
        embedding_ = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, c.dim, 1, 1);
        position_ = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, 1);
        slot_ = ggml_new_tensor_1d(ictx, GGML_TYPE_I64, 1);
        mask_ = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, capacity_, 1);
        for (auto * tensor : {hidden_, embedding_, position_, slot_, mask_}) ggml_set_input(tensor);
        const auto result = build_layer(build, hidden_, embedding_, position_, slot_, mask_, true);
        output_ = result.first; head_output_ = result.second;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(graph_, output_);
        if (head_output_) { ggml_set_output(head_output_); ggml_build_forward_expand(graph_, head_output_); }
        input_buffer_ = ggml_backend_alloc_ctx_tensors(ictx, execution_.backend());
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!input_buffer_ || !allocator_ || !ggml_gallocr_alloc_graph(allocator_, graph_)) {
            release_graph();
            throw std::runtime_error("Drama 3 tower graph allocation failed");
        }
    }
    void release() {
        release_graph();
        if (cache_buffer_) { ggml_backend_buffer_free(cache_buffer_); cache_buffer_ = nullptr; }
        cache_ctx_.reset(); capacity_ = 0;
    }
    core::ExecutionContext & execution_;
    FishAudioConfig config_;
    std::shared_ptr<core::BackendWeightStore> store_;
    TowerWeights weights_;
    Context graph_ctx_, input_ctx_, cache_ctx_;
    ggml_backend_buffer_t input_buffer_ = nullptr, cache_buffer_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * key_ = nullptr, * value_ = nullptr, * hidden_ = nullptr, * embedding_ = nullptr;
    ggml_tensor * position_ = nullptr, * slot_ = nullptr, * mask_ = nullptr, * output_ = nullptr, * head_output_ = nullptr;
    int64_t capacity_ = 0, valid_ = 0;
};


// Layers with known input codes share one graph. Intermediate hidden states
// stay on the backend, and the teacher-forced path does not evaluate heads.
class TowerSequenceGraph {
public:
    TowerSequenceGraph(core::ExecutionContext & execution, std::vector<TowerLayer *> layers)
        : execution_(execution), layers_(std::move(layers)) {}
    ~TowerSequenceGraph() { release(); }
    void release() {
        if (graph_) core::release_backend_graph_resources(execution_.backend(), graph_, true);
        if (allocator_) { ggml_gallocr_free(allocator_); allocator_ = nullptr; }
        if (buffer_) { ggml_backend_buffer_free(buffer_); buffer_ = nullptr; }
        graph_ctx_.reset(); input_ctx_.reset(); graph_ = nullptr; embeddings_.clear();
    }
    std::vector<float> run(const std::vector<float> & hidden, const std::vector<int32_t> & codes,
        bool semantic, int64_t position, const std::vector<float> & extra, bool return_hidden=true) {
        if (layers_.empty() || codes.size() != layers_.size()) throw std::runtime_error("Drama 3 tower sequence shape mismatch");
        for (auto * layer : layers_) {
            if (position != layer->valid_ || position >= layer->capacity_ ||
                hidden.size() != static_cast<size_t>(layer->config_.fast.dim))
                throw std::runtime_error("Drama 3 tower sequence cache transition mismatch");
        }
        ensure();
        ggml_backend_tensor_set(hidden_, hidden.data(), 0, hidden.size()*sizeof(float));
        for (size_t i=0; i<layers_.size(); ++i) {
            auto * layer=layers_[i];
            std::vector<float> e(hidden.size(),0);
            if (semantic) e=row(layer->weights_.embedding,codes[i],layer->config_.fast.dim);
            if (i==0) {
                if (extra.size()!=e.size()) throw std::runtime_error("Drama 3 tower extra shape mismatch");
                for (size_t j=0;j<e.size();++j)e[j]+=extra[j];
            }
            ggml_backend_tensor_set(embeddings_[i],e.data(),0,e.size()*sizeof(float));
        }
        const int32_t p=static_cast<int32_t>(position);
        ggml_backend_tensor_set(position_,&p,0,sizeof(p));
        ggml_backend_tensor_set(slot_,&position,0,sizeof(position));
        std::vector<float> mask(static_cast<size_t>(layers_[0]->capacity_),-std::numeric_limits<float>::infinity());
        std::fill(mask.begin(),mask.begin()+position+1,0.0F);
        ggml_backend_tensor_set(mask_,mask.data(),0,mask.size()*sizeof(float));
        if (core::compute_backend_graph(execution_.backend(),graph_,nullptr,"fish_audio.drama3.tower_sequence")!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("Drama 3 tower sequence computation failed");
        for(auto * layer:layers_)++layer->valid_;
        if(!return_hidden)return {};
        auto result=core::read_tensor_f32(output_);
        for(float v:result)if(!std::isfinite(v))throw std::runtime_error("Drama 3 tower sequence non-finite output");
        return result;
    }
    ggml_tensor * output() const { return output_; }
private:
    void ensure() {
        if(graph_)return;
        ggml_init_params gp{32*1024*1024,nullptr,true},ip{1024*1024,nullptr,true};
        graph_ctx_.reset(ggml_init(gp)); input_ctx_.reset(ggml_init(ip));
        if(!graph_ctx_||!input_ctx_)throw std::runtime_error("Drama 3 tower sequence context allocation failed");
        auto * ic=input_ctx_.get();
        core::ModuleBuildContext build{graph_ctx_.get(),"fish_audio.drama3.tower_sequence",execution_.backend_type()};
        const auto width=layers_[0]->config_.fast.dim;
        hidden_=ggml_new_tensor_3d(ic,GGML_TYPE_F32,width,1,1);
        position_=ggml_new_tensor_1d(ic,GGML_TYPE_I32,1);
        slot_=ggml_new_tensor_1d(ic,GGML_TYPE_I64,1);
        mask_=ggml_new_tensor_2d(ic,GGML_TYPE_F32,layers_[0]->capacity_,1);
        for(auto * t:{hidden_,position_,slot_,mask_})ggml_set_input(t);
        auto * h=hidden_;
        for(auto * layer:layers_) {
            auto * embedding=ggml_new_tensor_3d(ic,GGML_TYPE_F32,width,1,1);
            ggml_set_input(embedding);embeddings_.push_back(embedding);
            h=layer->build_layer(build,h,embedding,position_,slot_,mask_,false).first;
        }
        output_=h;ggml_set_output(output_);
        graph_=ggml_new_graph_custom(graph_ctx_.get(),16384,false);ggml_build_forward_expand(graph_,output_);
        buffer_=ggml_backend_alloc_ctx_tensors(ic,execution_.backend());
        allocator_=ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if(!buffer_||!allocator_||!ggml_gallocr_alloc_graph(allocator_,graph_)) {
            release();throw std::runtime_error("Drama 3 tower sequence graph allocation failed");
        }
    }
    core::ExecutionContext & execution_;
    std::vector<TowerLayer *> layers_;
    Context graph_ctx_,input_ctx_;
    ggml_backend_buffer_t buffer_=nullptr;
    ggml_gallocr_t allocator_=nullptr;
    ggml_cgraph * graph_=nullptr;
    ggml_tensor * hidden_=nullptr,*position_=nullptr,*slot_=nullptr,*mask_=nullptr,*output_=nullptr;
    std::vector<ggml_tensor *> embeddings_;
};

}  // namespace

class Drama3Tower::Impl {
public:
    Impl(std::shared_ptr<const FishAudioAssets> assets, core::ExecutionContext & execution,
        size_t context_bytes, assets::TensorStorageType storage)
        : assets_(std::move(assets)), embeddings_(assets_->model_weights->require_f32_tensor("audio_decoder.codebook_embeddings.weight",
            {assets_->config.fast.num_codebooks * assets_->config.fast.vocab_size, assets_->config.fast.dim})) {
        const auto & c = assets_->config.fast;
        store_ = std::make_shared<core::BackendWeightStore>(execution.backend(), execution.backend_type(), "drama3.tower.weights", context_bytes);
        const auto & source = *assets_->model_weights;
        auto load = [&](const std::string & prefix, bool shared, int index) {
            TowerWeights w;
            w.embedding = source.require_f32_tensor(prefix + (shared ? ".semantic_embedding.weight" : ".codebook_embedding.weight"), {c.vocab_size, c.dim});
            auto n = [&](const std::string & name, int64_t width) { return binding::norm_weight_from_source(*store_, source, prefix + name, width); };
            auto l = [&](const std::string & name, int64_t in, int64_t out) { return binding::linear_from_source(*store_, source, prefix + name, storage, out, in, false); };
            w.enorm = n(".enorm", c.dim); w.hnorm = n(".hnorm", c.dim);
            w.anorm = n(".transformer_layer.attention_norm", c.dim);
            w.fnorm = n(".transformer_layer.ffn_norm", c.dim);
            w.final_norm = n(".final_layernorm", c.dim);
            w.qnorm = n(".transformer_layer.attention.q_norm", c.head_dim);
            w.knorm = n(".transformer_layer.attention.k_norm", c.head_dim);
            w.eh = l(".eh_proj", 2*c.dim, c.dim);
            w.qkv = l(".transformer_layer.attention.wqkv", c.dim, 2*c.head_dim*(c.n_head+c.n_local_heads));
            w.out = l(".transformer_layer.attention.wo", c.n_head*c.head_dim, c.dim);
            w.gate = l(".transformer_layer.feed_forward.w1", c.dim, c.intermediate_size);
            w.up = l(".transformer_layer.feed_forward.w3", c.dim, c.intermediate_size);
            w.down = l(".transformer_layer.feed_forward.w2", c.intermediate_size, c.dim);
            if (!shared) w.head = binding::linear_from_source(*store_, source,
                "audio_decoder_mtp.heads."+std::to_string(index), storage, c.vocab_size, c.dim, false);
            return std::make_unique<TowerLayer>(execution, assets_->config, store_, std::move(w));
        };
        for (int64_t i=0; i<assets_->config.drama3->shared_layers; ++i) shared_.push_back(load("audio_decoder_mtp_shared.layers."+std::to_string(i),true,static_cast<int>(i)));
        for (int64_t i=0; i<c.num_codebooks-1; ++i) residual_.push_back(load("audio_decoder_mtp.layers."+std::to_string(i),false,static_cast<int>(i)));
        store_->upload();
        std::vector<TowerLayer *> shared,all;
        for(auto & layer:shared_){shared.push_back(layer.get());all.push_back(layer.get());}
        for(auto & layer:residual_)all.push_back(layer.get());
        shared_graph_=std::make_unique<TowerSequenceGraph>(execution,std::move(shared));
        teacher_graph_=std::make_unique<TowerSequenceGraph>(execution,std::move(all));
    }
    bool semantic(int32_t token) const { return token >= assets_->config.semantic_start_token_id && token <= assets_->config.semantic_end_token_id; }
    std::vector<float> sum(const std::vector<int32_t> & frame, int64_t first, int64_t end) const {
        const auto & c=assets_->config.fast;
        if (frame.size()!=static_cast<size_t>(1+c.num_codebooks)) throw std::runtime_error("Drama 3 frame width mismatch");
        std::vector<float> result(static_cast<size_t>(c.dim),0.0F);
        for (int64_t i=first; i<end; ++i) {
            const auto e=row(embeddings_,i*c.vocab_size+frame[static_cast<size_t>(i+1)],c.dim);
            for (size_t j=0;j<e.size();++j) result[j]+=e[j];
        }
        return result;
    }
    std::shared_ptr<const FishAudioAssets> assets_;
    assets::TensorDataF32 embeddings_;
    std::shared_ptr<core::BackendWeightStore> store_;
    std::vector<std::unique_ptr<TowerLayer>> shared_, residual_;
    std::unique_ptr<TowerSequenceGraph> shared_graph_,teacher_graph_;
};

Drama3Tower::Drama3Tower(std::shared_ptr<const FishAudioAssets> assets, core::ExecutionContext & execution,
    size_t context, assets::TensorStorageType storage) : impl_(std::make_unique<Impl>(std::move(assets), execution, context, storage)) {}
Drama3Tower::~Drama3Tower() = default;
void Drama3Tower::reset(int64_t capacity) {
    if(capacity>impl_->shared_[0]->capacity()){impl_->shared_graph_->release();impl_->teacher_graph_->release();}
    for (auto & layer : impl_->shared_) layer->reset(capacity);
    for (auto & layer : impl_->residual_) layer->reset(capacity);
}
void Drama3Tower::release_graphs() {
    impl_->shared_graph_->release();impl_->teacher_graph_->release();
    for (auto & layer : impl_->shared_) layer->release_runtime();
    for (auto & layer : impl_->residual_) layer->release_runtime();
}
std::vector<float> Drama3Tower::embed_backbone(const std::vector<int32_t> & frame, const std::vector<float> & embedding) const {
    if (!impl_->semantic(frame.at(0))) return embedding;
    const auto count = impl_->assets_->config.drama3->backbone_codebooks;
    auto extra = impl_->sum(frame, 1, 1+count);
    if (extra.size()!=embedding.size()) throw std::runtime_error("Drama 3 backbone embedding size mismatch");
    const float divisor=std::sqrt(static_cast<float>(1+count));
    for (size_t i=0;i<extra.size();++i) extra[i]=(extra[i]+embedding[i])/divisor;
    return extra;
}
std::vector<float> Drama3Tower::run(const std::vector<float> & hidden, const std::vector<int32_t> & current,
    const std::vector<int32_t> & next, int64_t position, bool teacher_force,
    const std::function<int32_t(const std::vector<float> &)> & sample, std::vector<int32_t> & codes, bool return_hidden) {
    const auto & c=impl_->assets_->config;
    if (next.size()!=static_cast<size_t>(1+c.fast.num_codebooks)) throw std::runtime_error("Drama 3 next-frame width mismatch");
    const bool sem=impl_->semantic(next[0]);
    auto extra=impl_->sum(current,1+c.drama3->backbone_codebooks,c.fast.num_codebooks);
    if (!impl_->semantic(current[0])) std::fill(extra.begin(),extra.end(),0.0F);
    auto h=hidden;
    const auto first=static_cast<int32_t>(std::clamp<int64_t>(next[0]-c.semantic_start_token_id,0,c.fast.vocab_size-1));
    codes={first};
    std::vector<int32_t> input_codes(impl_->shared_.size(),first);
    if(teacher_force) {
        for(size_t i=0;i<impl_->residual_.size();++i) input_codes.push_back(next[i+1]);
        codes.assign(next.begin()+1,next.end());codes[0]=first;
        return impl_->teacher_graph_->run(h,input_codes,sem,position,extra,return_hidden);
    }
    impl_->shared_graph_->run(h,input_codes,sem,position,extra,false);
    auto * device_hidden=impl_->shared_graph_->output();
    for (size_t i=0;i<impl_->residual_.size();++i) {
        device_hidden=impl_->residual_[i]->step_device(device_hidden,codes.back(),sem,position,nullptr);
        const int32_t code=sample(impl_->residual_[i]->logits({}));
        if (code<0||code>=c.fast.vocab_size) throw std::runtime_error("Drama 3 codebook token out of range");
        codes.push_back(code);
    }
    if(!return_hidden)return {};
    h=core::read_tensor_f32(device_hidden);
    for(float v:h)if(!std::isfinite(v))throw std::runtime_error("Drama 3 tower non-finite hidden output");
    return h;
}

}  // namespace engine::models::fish_audio
