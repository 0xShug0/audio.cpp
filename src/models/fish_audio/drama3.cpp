#include "engine/models/fish_audio/drama3.h"
#include "engine/models/fish_audio/drama3_tower.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace engine::models::fish_audio {
namespace {
namespace json = engine::io::json;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("Drama 3: " + message);
}
void request_require(bool condition, const std::string & message) {
    if(!condition)throw runtime::InvalidRequestError("Drama 3: "+message);
}
size_t white_prefix(const std::string & text) {
    size_t position=0;
    while (position<text.size()) {
        const auto byte=static_cast<unsigned char>(text[position]);
        uint32_t cp=byte;size_t length=1;
        if ((byte&0xE0)==0xC0) {cp=byte&0x1F;length=2;}
        else if ((byte&0xF0)==0xE0) {cp=byte&0x0F;length=3;}
        else if ((byte&0xF8)==0xF0) {cp=byte&7;length=4;}
        if (position+length>text.size()) break;
        bool valid=true;
        for(size_t j=1;j<length;++j) {
            const auto b=static_cast<unsigned char>(text[position+j]);
            if((b&0xC0)!=0x80) {valid=false;break;}
            cp=(cp<<6)|(b&0x3F);
        }
        if(!valid)break;
        const bool white=(cp>=9&&cp<=13)||cp==0x20||cp==0x85||cp==0xA0||cp==0x1680||
            (cp>=0x2000&&cp<=0x200A)||cp==0x2028||cp==0x2029||cp==0x202F||cp==0x205F||cp==0x3000;
        if(!white)break;
        position+=length;
    }
    return position;
}
void validate_text(const std::string & text) {
    request_require(white_prefix(text)!=text.size(),"text must not be empty");
    for(const auto * marker:{"<|im_start|>","<|im_end|>","<|voice|>","<|text|>","<|interleave|>","<|semantic:"})
        request_require(text.find(marker)==std::string::npos,"structural prompt tokens cannot appear in caller text");
    size_t offset=0;
    while((offset=text.find("<|speaker",offset))!=std::string::npos) {
        request_require(text.compare(offset,10,"<|speaker:")==0,"invalid speaker marker");
        const auto end=text.find("|>",offset+10);
        request_require(end!=std::string::npos,"unterminated speaker marker");
        const auto digits=text.substr(offset+10,end-offset-10);
        request_require(!digits.empty()&&digits.size()<=2&&(digits.size()==1||digits[0]!='0'),"speaker id must be canonical 0..15");
        int id=0;
        for(char ch:digits) {request_require(ch>='0'&&ch<='9',"speaker id must be 0..15");id=id*10+ch-'0';}
        request_require(id<=15,"speaker id must be 0..15");
        offset=end+2;
    }
}
std::string tagged(const std::string & text) {
    validate_text(text);
    if(text.compare(white_prefix(text),10,"<|speaker:")==0)return text;
    return "<|speaker:0|> "+text;
}

int32_t sample_logits(const std::vector<float> & logits, float temperature, int top_k, float top_p,
    std::mt19937 & random) {
    require(!logits.empty()&&top_k>0&&top_p>0&&top_p<=1&&std::isfinite(temperature)&&temperature>0,"invalid sampler settings");
    for(float x:logits)require(std::isfinite(x),"non-finite logits");
    std::vector<size_t> order(logits.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return logits[a]>logits[b];});
    const double inverse=1.0/std::max<double>(temperature,1e-5);
    std::vector<double> probabilities(logits.size());double total=0;
    for(size_t i=0;i<order.size();++i){probabilities[i]=std::exp((logits[order[i]]-logits[order[0]])*inverse);total+=probabilities[i];}
    double cumulative=0,best=-1;size_t chosen=0;
    for(size_t i=0;i<order.size();++i) {
        const double probability=probabilities[i]/total;cumulative+=probability;
        const bool keep=i==0||(i<static_cast<size_t>(top_k)&&cumulative<=top_p);
        if (!keep) {
            // All following probabilities are outside the retained prefix.
            // Preserve the RNG position exactly, without evaluating log/noise
            // for candidates that cannot win the exponential race.
            for (size_t j=i;j<order.size();++j)
                (void)std::generate_canonical<double,53>(random);
            break;
        }
        const double uniform=std::max(std::generate_canonical<double,53>(random),std::numeric_limits<double>::min());
        const double noise=-std::log(uniform);
        const double score=keep?probability/noise:0.0;
        if(score>best){best=score;chosen=i;}
    }
    return static_cast<int32_t>(order[chosen]);
}

}  // namespace

FishAudioConfig parse_drama3_config(const json::Value & root) {
    require(json::require_string(root,"model_type")=="fish_omni","model_type must be fish_omni");
    require(json::require_string(root,"speech_template_version")=="reference-bracket-v1","unsupported speech template");
    const auto & t=root.require("text_config");const auto & a=root.require("audio_decoder_config");
    const auto type=json::require_string(t,"model_type");
    require(type=="fish_omni_text"||type=="qwen3_5_moe_text","unsupported text backbone");
    for(const auto * flag:{"use_qk_norm","attn_output_gate","layernorm_zero_centered_gamma","shared_expert_gate"})
        require(json::optional_bool(t,flag,true),std::string("text flag must be enabled: ")+flag);
    require(!json::optional_bool(t,"rotary_interleaved",false),"backbone RoPE must be neox");
    require(!json::optional_bool(t,"tie_word_embeddings",false)&&
        !json::optional_bool(t,"attention_bias",false)&&json::optional_f32(t,"attention_dropout",0)==0,
        "unsupported backbone embedding/attention policy");
    if(const auto * types=t.find("mlp_layer_types")) {
        for(const auto & mlp:types->as_array())require(mlp.as_string()=="moe","all backbone MLPs must be MoE");
    }
    require(json::require_string(a,"decoder_type")=="codebook_mtp_shared","unsupported audio decoder type");
    require(json::optional_bool(a,"rotary_interleaved",true)&&json::optional_bool(a,"attention_output_gate",true)&&
        json::optional_bool(a,"layernorm_zero_centered_gamma",true),"unsupported tower attention/norm policy");
    require(!json::optional_bool(a,"use_moe",false)&&!json::optional_bool(a,"mtp_use_swa",false)&&
        !json::optional_bool(a,"attention_qkv_bias",false)&&!json::optional_bool(a,"attention_o_bias",false),"unsupported tower flags");
    require(json::optional_f32(a,"dropout",0)==0,"tower dropout must be zero");
    auto d=std::make_shared<Drama3Config>();auto & b=d->backbone;
    b.vocab_size=json::require_i64(t,"vocab_size");b.hidden_size=json::require_i64(t,"hidden_size");
    b.layers=json::require_i64(t,"num_hidden_layers");b.heads=json::require_i64(t,"num_attention_heads");
    b.kv_heads=json::require_i64(t,"num_key_value_heads");b.head_dim=json::require_i64(t,"head_dim");
    b.experts=json::require_i64(t,"num_experts");b.experts_per_token=json::require_i64(t,"num_experts_per_tok");
    b.expert_intermediate_size=json::require_i64(t,"moe_intermediate_size");
    b.shared_intermediate_size=json::require_i64(t,"shared_expert_intermediate_size");
    b.intermediate_size=b.shared_intermediate_size;
    b.normalize_topk=json::optional_bool(t,"norm_topk_prob",true);
    b.linear_conv_kernel_dim=json::require_i64(t,"linear_conv_kernel_dim");
    b.linear_key_head_dim=json::require_i64(t,"linear_key_head_dim");
    b.linear_num_key_heads=json::require_i64(t,"linear_num_key_heads");
    b.linear_num_value_heads=json::require_i64(t,"linear_num_value_heads");
    b.linear_value_head_dim=json::require_i64(t,"linear_value_head_dim");
    b.rms_norm_eps=json::require_f32(t,"rms_norm_eps");
    const auto & rope=t.require("rope_parameters");
    b.rope_theta=json::require_f32(rope,"rope_theta");b.partial_rotary_factor=json::require_f32(rope,"partial_rotary_factor");
    require(json::optional_string(rope,"rope_type","default")=="default","unsupported backbone RoPE scaling");
    b.weight_prefix="model.language_model";b.lm_head_prefix="drama3.candidate_lm_head";
    b.tie_word_embeddings=false;b.round_bf16_activations=false;
    b.decode_cache_steps=json::optional_i64(root,"audiocpp_context_size",8192);
    const auto & layers=t.require("layer_types").as_array();
    require(static_cast<int64_t>(layers.size())==b.layers,"layer_types length mismatch");
    for(size_t i=0;i<layers.size();++i)require(layers[i].as_string()==((i+1)%4==0?"full_attention":"linear_attention"),"unsupported hybrid layer pattern");
    require(b.layers>0&&b.hidden_size>0&&b.heads>0&&b.kv_heads>0&&b.heads%b.kv_heads==0&&b.experts>0&&
        b.experts_per_token>0&&b.experts_per_token<=b.experts&&b.decode_cache_steps>0&&
        b.expert_intermediate_size>0&&b.shared_intermediate_size>0&&b.rms_norm_eps>0,"invalid backbone dimensions");
    require(b.linear_num_key_heads>0&&b.linear_num_value_heads>0&&
        b.linear_num_value_heads%b.linear_num_key_heads==0&&b.linear_key_head_dim==b.linear_value_head_dim&&
        b.linear_key_head_dim==128&&b.linear_conv_kernel_dim>1,"unsupported Gated DeltaNet geometry");
    const auto rotary=static_cast<int64_t>(b.head_dim*b.partial_rotary_factor);
    require(b.rope_theta>0&&rotary>0&&rotary<=b.head_dim&&rotary%2==0,"invalid backbone rotary geometry");
    FishAudioConfig result;result.model_type="fish_omni";result.torch_dtype="bfloat16";
    result.semantic_start_token_id=json::require_i64(root,"semantic_start_token_id");
    result.semantic_end_token_id=json::require_i64(root,"semantic_end_token_id");
    result.im_end_token_id=json::require_i64(root,"eos_token_id");
    auto & f=result.fast;
    f.vocab_size=json::require_i64(a,"vocab_size");f.num_codebooks=json::require_i64(a,"num_codebooks");
    f.dim=json::require_i64(a,"dim");f.n_layer=json::require_i64(a,"n_layer");
    f.intermediate_size=json::require_i64(a,"intermediate_size");f.n_head=json::require_i64(a,"n_head");
    f.n_local_heads=json::require_i64(a,"n_local_heads");f.head_dim=json::require_i64(a,"head_dim");
    f.rope_base=json::require_f32(a,"rope_base");f.norm_eps=json::require_f32(a,"norm_eps");
    f.attention_qk_norm=true;f.max_seq_len=b.decode_cache_steps;
    d->shared_layers=json::require_i64(a,"shared_enrichment_layers");
    d->backbone_codebooks=json::require_i64(a,"vq_backbone_codebooks");
    d->tower_rotary_percent=json::require_f32(a,"rotary_percent");
    require(f.dim==b.hidden_size&&f.n_layer==d->shared_layers&&f.num_codebooks>1&&f.vocab_size>0&&
        f.intermediate_size>0&&f.n_head>0&&f.n_local_heads>0&&f.n_head%f.n_local_heads==0&&f.head_dim>0&&
        d->shared_layers>0&&d->backbone_codebooks>=0&&d->backbone_codebooks<f.num_codebooks,"invalid tower dimensions");
    const auto tower_rotary=static_cast<int64_t>(f.head_dim*d->tower_rotary_percent);
    require(f.norm_eps>0&&f.rope_base>0&&tower_rotary>0&&tower_rotary<=f.head_dim&&tower_rotary%2==0,"invalid tower rotary/norm geometry");
    require(result.semantic_start_token_id>=0&&result.semantic_end_token_id<b.vocab_size&&
        result.semantic_end_token_id-result.semantic_start_token_id+1==f.vocab_size&&
        result.im_end_token_id>=0&&result.im_end_token_id<result.semantic_start_token_id,"invalid semantic vocabulary");
    b.output_vocab_size=f.vocab_size+1;
    b.prefill_cache_steps=14;
    result.text.vocab_size=b.vocab_size;result.text.dim=b.hidden_size;result.text.n_layer=b.layers;
    result.text.max_seq_len=b.decode_cache_steps;result.codec.total_codebooks=f.num_codebooks;
    result.drama3=std::move(d);
    return result;
}

FishAudioPrompt build_drama3_prompt(const FishAudioAssets & assets, const FishAudioTextTokenizer & tokenizer,
    const FishAudioRequest & request, const std::vector<codecs::FishDacCodes> & references,
    const std::optional<FishAudioConversationTurn> & previous) {
    require(request.references.size()==references.size(),"reference count mismatch");
    require(tokenizer.token_id("<|im_end|>")==assets.config.im_end_token_id&&
        tokenizer.token_id("<|semantic:0|>")==assets.config.semantic_start_token_id&&
        tokenizer.token_id("<|semantic:"+std::to_string(assets.config.fast.vocab_size-1)+"|>")==assets.config.semantic_end_token_id,
        "tokenizer/config vocabulary mismatch");
    const int64_t books=assets.config.fast.num_codebooks;
    std::vector<std::vector<int32_t>> rows;
    auto text=[&](const std::string & s){for(int32_t id:tokenizer.encode(s)){std::vector<int32_t> r(static_cast<size_t>(books+1),0);r[0]=id;rows.push_back(std::move(r));}};
    auto audio=[&](const codecs::FishDacCodes & codes) {
        require(codes.codebooks==books&&codes.frames>0&&codes.codes.size()==static_cast<size_t>(books*codes.frames),"reference code shape mismatch");
        request_require(rows.size()+static_cast<size_t>(codes.frames)<=8192,"prompt exceeds 8192 rows");
        for(int64_t p=0;p<codes.frames;++p) {
            std::vector<int32_t> r(static_cast<size_t>(books+1));
            for(int64_t b=0;b<books;++b){const auto value=codes.codes[static_cast<size_t>(b*codes.frames+p)];require(value>=0&&value<assets.config.fast.vocab_size,"reference code out of range");r[static_cast<size_t>(b+1)]=value;}
            r[0]=static_cast<int32_t>(assets.config.semantic_start_token_id)+r[1];rows.push_back(std::move(r));
        }
    };
    text("<|im_start|>system\n");
    text(references.empty()?"convert the provided text to speech":"convert the provided text to speech reference to the following:");
    for(size_t i=0;i<references.size();++i) {
        auto s=tagged(request.references[i].text);const size_t start=white_prefix(s);const size_t end=s.find("|>",start)+2;
        s=s.substr(start,end-start)+s.substr(0,start)+s.substr(end);
        text("\n\n[Reference "+std::to_string(i+1)+"]\nText: "+s+"\nSpeech: ");audio(references[i]);
    }
    text("<|im_end|>\n");
    if(previous){text("<|im_start|>user\n");text(tagged(previous->text));text("<|im_end|>\n");text("<|im_start|>assistant\n<|voice|>");audio(previous->codes);text("<|im_end|>\n");}
    text("<|im_start|>user\n");text(tagged(request.text));text("<|im_end|>\n");text("<|im_start|>assistant\n<|voice|>");
    request_require(!rows.empty()&&rows.size()<=8192,"prompt exceeds 8192 rows");
    FishAudioPrompt result;result.codebook_rows=books+1;result.steps=static_cast<int64_t>(rows.size());result.text=request.text;
    result.matrix.resize(static_cast<size_t>(result.codebook_rows*result.steps));
    for(size_t p=0;p<rows.size();++p)for(size_t b=0;b<rows[p].size();++b)result.matrix[b*rows.size()+p]=rows[p][b];
    return result;
}

class Drama3ARRuntime::Impl {
public:
    Impl(std::shared_ptr<const FishAudioAssets> assets, core::BackendConfig backend, int threads,
        size_t arena, size_t context, assets::TensorStorageType storage)
        : assets_(std::move(assets)), execution_(configured(backend,threads)),
          backbone_(assets_->model_weights, assets_->config.drama3->backbone, execution_, arena, context, storage),
          tower_(assets_,execution_,context,storage) {}
    codecs::FishDacCodes generate(const FishAudioPrompt & prompt, const FishAudioGenerationOptions & options,
        const std::function<void(const std::vector<int32_t> &)> & on_frame) {
        using Clock = std::chrono::steady_clock;
        const auto started = Clock::now();
        const auto & c=assets_->config;const auto & b=c.drama3->backbone;
        require(prompt.steps>0&&prompt.codebook_rows==c.fast.num_codebooks+1&&
            prompt.matrix.size()==static_cast<size_t>(prompt.steps*prompt.codebook_rows),"prompt shape mismatch");
        request_require(options.max_new_tokens>0&&options.max_new_tokens<=b.decode_cache_steps-prompt.steps,"prompt and output exceed context capacity");
        std::vector<std::vector<int32_t>> frames;
        for(int64_t p=0;p<prompt.steps;++p){std::vector<int32_t> r(static_cast<size_t>(prompt.codebook_rows));for(int64_t i=0;i<prompt.codebook_rows;++i)r[static_cast<size_t>(i)]=prompt.matrix[static_cast<size_t>(i*prompt.steps+p)];frames.push_back(std::move(r));}
        std::vector<int32_t> ids;for(const auto & frame:frames)ids.push_back(frame[0]);
        auto embedded=backbone_.token_embedding(ids);
        std::vector<float> inputs;inputs.reserve(embedded.size());
        for(size_t i=0;i<frames.size();++i){std::vector<float> e(embedded.begin()+i*b.hidden_size,embedded.begin()+(i+1)*b.hidden_size);e=tower_.embed_backbone(frames[i],e);inputs.insert(inputs.end(),e.begin(),e.end());}
        auto decode=backbone_.create_decode_session(prompt.steps+options.max_new_tokens);
        tower_.reset(prompt.steps+options.max_new_tokens);
        engine::debug::timing_log_scalar("fish_audio.drama3.setup_ms", engine::debug::elapsed_ms(started));
        const auto prefill_start = Clock::now();
        const auto prefill=decode->prefill_embeddings(inputs,prompt.steps);
        engine::debug::timing_log_scalar("fish_audio.drama3.backbone_prefill_ms", engine::debug::elapsed_ms(prefill_start));
        const auto tower_start = Clock::now();
        std::vector<int32_t> scratch;
        for(int64_t p=0;p<prompt.steps-1;++p){std::vector<float> h(prefill.hidden.begin()+p*b.hidden_size,prefill.hidden.begin()+(p+1)*b.hidden_size);tower_.run(h,frames[static_cast<size_t>(p)],frames[static_cast<size_t>(p+1)],p,true,{},scratch,false);}
        engine::debug::timing_log_scalar("fish_audio.drama3.tower_prefill_ms", engine::debug::elapsed_ms(tower_start));
        double head_ms=0, tower_ms=0, decode_ms=0;
        std::vector<float> hidden(prefill.hidden.end()-b.hidden_size,prefill.hidden.end());
        std::mt19937 random(options.seed);
        auto sampler=[&](const std::vector<float> & logits){return sample_logits(logits,options.temperature,options.top_k,options.top_p,random);};
        std::vector<std::vector<int32_t>> produced;
        for(int64_t n=0;n<options.max_new_tokens;++n) {
            const auto head_start = Clock::now();
            const auto logits=backbone_.lm_head(hidden);
            const int32_t first=sample_logits(logits,options.temperature,options.top_k,options.top_p,random);
            const int32_t retry=sample_logits(logits,1.0F,30,0.95F,random);
            auto id=[&](int32_t choice){return choice==0?static_cast<int32_t>(c.im_end_token_id):static_cast<int32_t>(c.semantic_start_token_id)+choice-1;};
            int32_t main=id(first);
            if(first!=0)for(size_t p=frames.size()>16?frames.size()-16:0;p<frames.size();++p)if(frames[p][0]==main){main=id(retry);break;}
            std::vector<int32_t> next(static_cast<size_t>(prompt.codebook_rows),0);next[0]=main;
            head_ms += engine::debug::elapsed_ms(head_start);
            const auto frame_start = Clock::now();
            std::vector<int32_t> codes;tower_.run(hidden,frames.back(),next,prompt.steps-1+n,false,sampler,codes,false);
            tower_ms += engine::debug::elapsed_ms(frame_start);
            if(main==c.im_end_token_id)break;
            for(size_t i=0;i<codes.size();++i)next[i+1]=codes[i];
            produced.push_back(codes);frames.push_back(next);
            if(on_frame)on_frame(codes);
            if(n+1<options.max_new_tokens){const auto step_start=Clock::now();auto e=backbone_.token_embedding({main});e=tower_.embed_backbone(next,e);hidden=decode->run_embedding_step(e);decode_ms+=engine::debug::elapsed_ms(step_start);}
        }
        engine::debug::timing_log_scalar("fish_audio.drama3.head_sample_ms", head_ms);
        engine::debug::timing_log_scalar("fish_audio.drama3.tower_generate_ms", tower_ms);
        engine::debug::timing_log_scalar("fish_audio.drama3.backbone_decode_ms", decode_ms);
        codecs::FishDacCodes result;result.frames=static_cast<int64_t>(produced.size());result.codebooks=c.fast.num_codebooks;
        result.codes.resize(static_cast<size_t>(result.frames*result.codebooks));
        for(size_t p=0;p<produced.size();++p)for(size_t i=0;i<produced[p].size();++i)result.codes[i*produced.size()+p]=produced[p][i];
        return result;
    }
    void release() {backbone_.release_graphs();tower_.release_graphs();}
private:
    static core::BackendConfig configured(core::BackendConfig b,int threads){b.threads=threads;return b;}
    std::shared_ptr<const FishAudioAssets> assets_;
    core::ExecutionContext execution_;
    modules::Qwen35MoeDecoderRuntime backbone_;
    Drama3Tower tower_;
};

Drama3ARRuntime::Drama3ARRuntime(std::shared_ptr<const FishAudioAssets> assets,core::BackendConfig backend,
    int threads,size_t arena,size_t context,assets::TensorStorageType storage)
    :impl_(std::make_unique<Impl>(std::move(assets),backend,threads,arena,context,storage)){}
Drama3ARRuntime::~Drama3ARRuntime()=default;
codecs::FishDacCodes Drama3ARRuntime::generate(const FishAudioPrompt & prompt,const FishAudioGenerationOptions & options,
    const std::function<void(const std::vector<int32_t> &)> & on_frame){return impl_->generate(prompt,options,on_frame);}
void Drama3ARRuntime::release_runtime_graphs(){impl_->release();}

}  // namespace engine::models::fish_audio
