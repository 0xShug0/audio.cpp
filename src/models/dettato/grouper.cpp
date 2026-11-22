#include "grouper.h"
#include "frontend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include <algorithm>
#include <cmath>

namespace engine::models::dettato::detail {
using namespace core;
using namespace modules;

GrouperNetwork::GrouperNetwork(const assets::TensorSource &source, const ExecutionContext &execution)
    : execution_(execution),
      store_(execution.backend(), execution.backend_type(), "dettato.grouper.weights", 1024 * 1024),
      graphs_(32) {
    const std::array<std::string, 12> prefixes{"blocks.", "ln_in.",   "ln_out.",   "pitch_emb.",
                                               "feat.",   "hd.enc.",  "prof.net.", "lanes.",
                                               "s15_ln.", "s15_net.", "heads.",    "hd.prior"};
    const std::array<int,14> offsets{-2,-1,1,2,-12,12,-19,19,-24,24,-28,28,-31,31};
    std::array<size_t,14> order{};
    for (size_t k=0;k<order.size();++k) order[k]=k;
    std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return offsets[a]<offsets[b];});
    if (execution_.backend_type()==BackendType::Cuda) {
        std::array<int32_t,28> positions{};
        for (size_t k=0;k<order.size();++k) positions[2*k]=offsets[order[k]];
        weights_.emplace("pitch_offsets",store_.make_tensor(TensorShape::from_dims({14,2}),
            GGML_TYPE_I32,positions.data(),positions.size()*sizeof(int32_t)));
    }
    for (const auto &metadata : source.tensors()) {
        const bool pitch_taps=metadata.name.rfind("lanes.blocks.",0)==0 && metadata.name.find(".pm.taps")!=std::string::npos;
        if (!(execution_.backend_type()==BackendType::Cuda && pitch_taps) && std::any_of(prefixes.begin(), prefixes.end(),
                        [&](const auto &prefix) { return metadata.name.rfind(prefix, 0) == 0; }))
            weights_.emplace(metadata.name, store_.load_tensor(source, metadata.name, assets::TensorStorageType::Native, metadata.shape));
        const auto & name=metadata.name;
        if(name.rfind("lanes.attn.",0)==0 && name.find(".rel_bias")!=std::string::npos)
            relative_bias_.emplace(name.substr(0,name.size()-9),source.require_f32(name));
        if(pitch_taps) {
            constexpr int64_t P=96;
            const auto taps=source.require_f32(name);
            const int64_t C=metadata.shape[0];
            if (execution_.backend_type()==BackendType::Cuda) {
                std::vector<float> coefficients(taps.size());
                for (int64_t c=0;c<C;++c) for (size_t k=0;k<order.size();++k)
                    coefficients[static_cast<size_t>(c*order.size()+k)]=taps[static_cast<size_t>(c*order.size()+order[k])];
                weights_.emplace(name+".compact",store_.make_f32(TensorShape::from_dims({C,14}),std::move(coefficients)));
                continue;
            }
            std::vector<float> matrix(static_cast<size_t>(C*P*P));
            for(int64_t c=0;c<C;++c) for(int64_t p=0;p<P;++p) for(size_t k=0;k<offsets.size();++k) {
                const auto q=p+offsets[k];
                if(q>=0 && q<P) matrix[static_cast<size_t>((c*P+p)*P+q)]=taps[static_cast<size_t>(c*14+k)];
            }
            weights_.emplace(name+".matrix",store_.make_f32(TensorShape::from_dims({C,P,P}),matrix));
        }
    }
    const auto norm = [&](const std::string &prefix) {
        return NormWeights{weights_.at(prefix + ".weight"), weights_.at(prefix + ".bias")};
    };
    const auto affine = [&](const std::string &prefix) {
        return LinearWeights{weights_.at(prefix + ".weight"), weights_.at(prefix + ".bias")};
    };
    encoder_weights_.input_norm = norm("ln_in");
    encoder_weights_.final_norm = norm("ln_out");
    for (int layer = 0; layer < 7; ++layer) {
        const auto prefix = "blocks." + std::to_string(layer);
        RopeTransformerEncoderLayerWeights weights;
        weights.norm1 = norm(prefix + ".ln1");
        weights.norm2 = norm(prefix + ".ln2");
        weights.qkv = affine(prefix + ".qkv");
        weights.attention_output = affine(prefix + ".o");
        weights.ffn_in = affine(prefix + ".ff.0");
        weights.ffn_out = affine(prefix + ".ff.2");
        encoder_weights_.layers.push_back(std::move(weights));
    }
    store_.upload();
}

TensorValue GrouperNetwork::linear(ModuleBuildContext &ctx, const TensorValue &x, const std::string &name) {
    const auto weight = weights_.at(name + ".weight");
    return LinearModule({weight.shape.dims[1], weight.shape.dims[0], true, GGML_PREC_F32})
        .build(ctx, x, {weight, weights_.at(name + ".bias")});
}

void GrouperNetwork::build_lanes() {
    if(frames_<8) throw std::runtime_error("Dettato grouping frontend requires at least eight frames");
    auto & ctx=lane_graph_->build_context();
    const int64_t T=frames_/8,P=96,C=128;
    lane_input_=make_tensor(ctx,GGML_TYPE_F32,TensorShape::from_dims({1,9,frames_,288}));
    raster_input_=make_tensor(ctx,GGML_TYPE_F32,TensorShape::from_dims({1,4,T,P}));
    ggml_set_input(lane_input_.tensor);ggml_set_input(raster_input_.tensor);
    const auto convolution=[&](const TensorValue & value,const std::string & name,int dilation=1,bool depthwise=false) {
        const auto weight=weights_.at(name+".weight");
        const auto kh=weight.shape.dims[2],kw=weight.shape.dims[3];
        const Conv2dWeights parameters{weight,weights_.at(name+".bias")};
        if(depthwise) return DepthwiseConv2dModule({C,kh,kw,1,1,static_cast<int>(3*dilation),0,dilation,1,true}).build(ctx,value,parameters);
        Conv2dConfig cfg{weight.shape.dims[1],weight.shape.dims[0],kh,kw};
        if(name=="lanes.stem") {cfg.stride_height=8;cfg.stride_width=3;cfg.padding_width=1;}
        return Conv2dModule(cfg).build(ctx,value,parameters);
    };
    auto value=AddModule{}.build(ctx,convolution(lane_input_,"lanes.stem"),convolution(raster_input_,"lanes.occ"));
    for(int layer=0;layer<6;++layer) {
        const auto prefix="lanes.blocks."+std::to_string(layer);
        auto y=convolution(value,prefix+".dw_t",1<<(layer%4),true);
        TensorValue mixed;
        if (ctx.backend_type==BackendType::Cuda) {
            mixed=wrap_tensor(ggml_conv_2d_dw_sparse(ctx.ggml,weights_.at(prefix+".pm.taps.compact").tensor,
                y.tensor,weights_.at("pitch_offsets").tensor),y.shape);
        } else {
            mixed=TransposeModule({{1,0,2,3},4}).build(ctx,y);
            mixed=reshape_tensor(ctx,ensure_backend_addressable_layout(ctx,mixed),TensorShape::from_dims({C,T,P}));
            const auto matrix=TransposeModule({{0,2,1,3},3}).build(ctx,weights_.at(prefix+".pm.taps.matrix"));
            mixed=MatMulModule{}.build(ctx,mixed,matrix);
            mixed=reshape_tensor(ctx,mixed,TensorShape::from_dims({C,1,T,P}));
            mixed=TransposeModule({{1,0,2,3},4}).build(ctx,mixed);
        }
        y=AddModule{}.build(ctx,y,mixed);
        y=TransposeModule({{0,2,3,1},4}).build(ctx,y);
        y=LayerNormModule({C,1e-6f,true,true}).build(ctx,y,{weights_.at(prefix+".norm.w"),weights_.at(prefix+".norm.b")});
        y=TransposeModule({{0,3,1,2},4}).build(ctx,y);
        y=convolution(GeluModule{}.build(ctx,convolution(y,prefix+".pw1")),prefix+".pw2");
        const auto gamma=reshape_tensor(ctx,weights_.at(prefix+".gamma"),TensorShape::from_dims({1,C,1,1}));
        value=AddModule{}.build(ctx,value,MulModule{true}.build(ctx,y,gamma));
        if(layer==2 || layer==5) {
            const auto attention="lanes.attn."+std::to_string(layer);
            auto sequence=TransposeModule({{0,2,3,1},4}).build(ctx,value);
            sequence=reshape_tensor(ctx,ensure_backend_addressable_layout(ctx,sequence),TensorShape::from_dims({T,P,C}));
            auto normalized=LayerNormModule({C,1e-5f,true,true}).build(ctx,sequence,{weights_.at(attention+".ln.weight"),weights_.at(attention+".ln.bias")});
            const auto packed=linear(ctx,normalized,attention+".qkv");
            std::array<TensorValue,3> heads;
            for(int i=0;i<3;++i) {
                auto part=SliceModule({2,i*C,C}).build(ctx,packed);
                part=reshape_tensor(ctx,ensure_backend_addressable_layout(ctx,part),TensorShape::from_dims({T,P,4,32}));
                heads[i]=TransposeModule({{0,2,1,3},4}).build(ctx,part);
            }
            const auto bias=make_tensor(lane_graph_->persistent_build_context(),GGML_TYPE_F32,TensorShape::from_dims({1,4,P,P}));
            const auto & relative=relative_bias_.at(attention);
            std::vector<float> table(static_cast<size_t>(4*P*P));
            for(int64_t h=0;h<4;++h) for(int64_t p=0;p<P;++p) for(int64_t q=0;q<P;++q)
                table[static_cast<size_t>((h*P+p)*P+q)]=relative[static_cast<size_t>(h*(2*P-1)+q-p+P-1)];
            constants_.push_back({bias,std::move(table)});
            auto attended=ScaledDotProductAttentionModule({32,ScaledDotProductAttentionLowering::Explicit,GGML_PREC_F32})
                .build(ctx,heads[0],heads[1],heads[2],bias);
            attended=reshape_tensor(ctx,ensure_backend_addressable_layout(ctx,attended),TensorShape::from_dims({T,P,C}));
            attended=linear(ctx,attended,attention+".out");
            attended=MulModule{true}.build(ctx,attended,weights_.at(attention+".gamma"));
            attended=reshape_tensor(ctx,attended,TensorShape::from_dims({1,T,P,C}));
            value=AddModule{}.build(ctx,value,TransposeModule({{0,3,1,2},4}).build(ctx,attended));
        }
    }
    lane_output_=value;ggml_set_output(value.tensor);ggml_build_forward_expand(lane_graph_->graph(),value.tensor);
}

std::vector<float> GrouperNetwork::lane_features(const HarmonicFeatures & features,const std::vector<float> & notes) {
    if(features.batch!=1 || notes.size()%12!=0) throw std::runtime_error("Dettato lane reader expects one aligned note window");
    const int64_t count=static_cast<int64_t>(notes.size()/12),T=features.frames/8;
    prepare_lanes(features.frames);
    std::vector<int64_t> onset(count),end(count),lane(count);
    std::vector<float> raster(static_cast<size_t>(4*(T+1)*96),0);
    for(int64_t i=0;i<count;++i) {
        const auto * note=notes.data()+i*12;
        onset[i]=std::clamp<int64_t>(static_cast<int64_t>((note[0]+.5f)*12.5f),0,T-1);
        end[i]=std::max<int64_t>(std::min<int64_t>(static_cast<int64_t>((note[1]+.5f)*12.5f),T),onset[i]+1);
        lane[i]=std::clamp<int64_t>(static_cast<int64_t>(std::nearbyint(note[2]))-21,0,95);
        const auto a=static_cast<size_t>(onset[i]*96+lane[i]),b=static_cast<size_t>(end[i]*96+lane[i]);
        const size_t plane=static_cast<size_t>((T+1)*96);
        raster[a]+=1;raster[plane+a]+=1;raster[plane+b]-=1;
        raster[2*plane+a]+=note[note[3]>.5f ? 5 : 4];
        raster[3*plane+a]+=note[3]>.5f ? 1 : 0;
    }
    const auto plane=static_cast<size_t>((T+1)*96);
    for(int64_t p=0;p<96;++p) {
        float active=0;
        for(int64_t t=0;t<T;++t) {active+=raster[plane+static_cast<size_t>(t*96+p)];raster[plane+static_cast<size_t>(t*96+p)]=active;}
    }
    std::vector<float> compact(static_cast<size_t>(4*T*96));
    for(size_t ch=0;ch<4;++ch) std::copy_n(raster.begin()+ch*plane,static_cast<size_t>(T*96),compact.begin()+ch*T*96);
    write_tensor_f32(lane_input_,features.values);write_tensor_f32(raster_input_,compact);
    const auto start=std::chrono::steady_clock::now();lane_graph_->compute();
    const auto output=read_tensor_f32(lane_output_.tensor);
    std::vector<float> reads(static_cast<size_t>(count*256));
    std::vector<float> prefix(static_cast<size_t>((T+1)*128));
    for(int64_t p=0;p<96;++p) {
        if(std::find(lane.begin(),lane.end(),p)==lane.end()) continue;
        std::fill_n(prefix.begin(),128,0);
        for(int64_t c=0;c<128;++c) {
            double sum=0;
            for(int64_t t=0;t<T;++t) {sum+=output[static_cast<size_t>((c*T+t)*96+p)];prefix[static_cast<size_t>((t+1)*128+c)]=static_cast<float>(sum);}
        }
        for(int64_t i=0;i<count;++i) if(lane[i]==p) for(int64_t c=0;c<128;++c) {
            reads[static_cast<size_t>(i*256+c)]=output[static_cast<size_t>((c*T+onset[i])*96+p)];
            reads[static_cast<size_t>(i*256+128+c)]=(prefix[static_cast<size_t>(end[i]*128+c)]-prefix[static_cast<size_t>(onset[i]*128+c)])/
                static_cast<float>(end[i]-onset[i]);
        }
    }
    debug::timing_log_scalar("dettato.grouper.lanes_ms",debug::elapsed_ms(start));
    return reads;
}

void GrouperNetwork::prepare_lanes(int64_t frames) {
    if (lane_graph_ && frames_ == frames) return;
    lane_graph_.reset(); constants_.clear(); frames_ = frames;
    lane_graph_ = std::make_unique<PreparedGraph>(execution_,
        PreparedGraphConfig{"dettato.grouper.lanes", 4 * 1024 * 1024, 4096, 1024 * 1024});
    build_lanes();
    lane_graph_->allocate_persistent();
    lane_graph_->allocate();
    for (const auto & constant : constants_) write_tensor_f32(constant.first, constant.second);
}

GrouperNetwork::GraphState &GrouperNetwork::prepare(int64_t notes) {
    if (notes <= 0 || notes > 768)
        throw std::runtime_error("Dettato grouper graph expects 1..768 notes");
    if (auto *cached = graphs_.find(notes)) return **cached;
    auto state = std::make_unique<GraphState>();
    auto &graph_ = state->graph;
    auto &inputs_ = state->inputs;
    auto &outputs_ = state->outputs;
    graph_ = std::make_unique<PreparedGraph>(execution_, PreparedGraphConfig{"dettato.grouper",
        size_t{2} * 1024 * 1024, 8192, 1024 * 1024});
    auto &ctx = graph_->build_context();
    inputs_[0] = make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, notes}));
    const std::array<int64_t, 4> widths{11, 150, 36, 256};
    for (size_t i = 0; i < widths.size(); ++i)
        inputs_[i + 1] = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, notes, widths[i]}));
    inputs_[5] = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, notes, 3, 368}));
    for (size_t i = 6; i < 8; ++i)
        inputs_[i] = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 4, notes, 22}));
    for (const auto &input : inputs_)
        ggml_set_input(input.tensor);
    auto x = EmbeddingModule({128, 176}).build(ctx, inputs_[0], weights_.at("pitch_emb.weight"));
    x = AddModule{}.build(ctx, x, linear(ctx, GeluModule{}.build(ctx, linear(ctx, inputs_[1], "feat.0")), "feat.2"));
    auto audio = linear(ctx, GeluModule{}.build(ctx, linear(ctx, inputs_[2], "hd.enc.0")), "hd.enc.2");
    audio = AddModule{}.build(
        ctx, audio, linear(ctx, GeluModule{}.build(ctx, linear(ctx, inputs_[3], "prof.net.0")), "prof.net.2"));
    audio = AddModule{}.build(ctx, audio, linear(ctx, inputs_[4], "lanes.read"));
    x = AddModule{}.build(ctx, x, audio);
    const auto mean = ReduceMeanModule({1}).build(ctx, inputs_[5]);
    auto f = wrap_tensor(ggml_sub(ctx.ggml, inputs_[5].tensor, mean.tensor), inputs_[5].shape);
    const auto variance = ReduceMeanModule({1}).build(ctx, MulModule{}.build(ctx, f, f));
    const auto sd = SqrtModule{}.build(ctx, variance);
    auto *denominator = ggml_clamp(ctx.ggml, sd.tensor, 1e-3f, INFINITY);
    f = wrap_tensor(ggml_div(ctx.ggml, f.tensor, denominator), f.shape);
    f = LayerNormModule({368, 1e-5f, true, true})
            .build(ctx, f, {weights_.at("s15_ln.weight"), weights_.at("s15_ln.bias")});
    f = reshape_tensor(ctx, f, TensorShape::from_dims({1, notes, 1104}));
    f = linear(ctx, GeluModule{}.build(ctx, linear(ctx, f, "s15_net.0")), "s15_net.2");
    x = AddModule{}.build(ctx, x, f);
    RopeTransformerEncoderConfig encoder;
    encoder.hidden_size = 176;
    encoder.intermediate_size = 352;
    encoder.heads = 4;
    encoder.rotary_dimensions = 44;
    encoder.rotary = EncoderRoPE::SplitTables;
    encoder.projection_precision = GGML_PREC_F32;
    encoder.flash_attention = false;
    RopeTransformerEncoderInputs position;
    position.cosine = inputs_[6];
    position.sine = inputs_[7];
    x = RopeTransformerEncoderModule(encoder).build(ctx, x, position, encoder_weights_);
    outputs_[0] = linear(ctx, x, "heads.inst");
    outputs_[1] = linear(ctx, x, "heads.emb");
    outputs_[1] = wrap_tensor(ggml_l2_norm(ctx.ggml, outputs_[1].tensor, 1e-12f), outputs_[1].shape);
    outputs_[2] = linear(ctx, x, "heads.keep");
    outputs_[3] = x;
    outputs_[4] = SoftmaxModule{}.build(ctx, outputs_[0]);
    for (const auto &output : outputs_) {
        ggml_set_output(output.tensor);
        if (output.tensor->view_src) ggml_set_output(output.tensor->view_src);
        ggml_build_forward_expand(graph_->graph(), output.tensor);
    }
    debug::timing_log_context_reservation("dettato.grouper.graph", ctx.ggml);
    graph_->allocate();
    size_t retained = graph_->workspace_bytes() + graph_->persistent_bytes();
    graphs_.for_each_value([&](const auto &entry) {
        retained += entry->graph->workspace_bytes() + entry->graph->persistent_bytes();
    });
    // Exact note counts preserve attention math. Bound activation retention;
    // a single oversized shape remains usable without retaining other shapes.
    if (retained > 32 * 1024 * 1024) graphs_.clear();
    auto *result = state.get();
    graphs_.put(notes, std::move(state));
    return *result;
}

std::array<std::vector<float>, 5> GrouperNetwork::run(const GrouperInputs &inputs) {
    auto &state = prepare(static_cast<int64_t>(inputs.pitch.size()));
    write_tensor_i32(state.inputs[0], inputs.pitch);
    const std::array<const std::vector<float> *, 7> values{
        &inputs.scalars, &inputs.harmonics, &inputs.profile, &inputs.lanes, &inputs.taps, &inputs.cosine, &inputs.sine};
    for (size_t i = 0; i < values.size(); ++i)
        write_tensor_f32(state.inputs[i + 1], *values[i]);
    const auto start = std::chrono::steady_clock::now();
    state.graph->compute();
    std::array<std::vector<float>, 5> result;
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = read_tensor_f32(state.outputs[i].tensor);
    debug::timing_log_scalar("dettato.grouper.forward_ms", debug::elapsed_ms(start));
    return result;
}
} // namespace engine::models::dettato::detail
