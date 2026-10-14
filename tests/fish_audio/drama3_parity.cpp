#include "engine/models/fish_audio/drama3.h"
#include "engine/models/fish_audio/drama3_tower.h"
#include "engine/framework/core/backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fish = engine::models::fish_audio;
namespace assets = engine::assets;
namespace core = engine::core;
namespace json = engine::io::json;
double tolerance=0.0001;

void compare(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    if (actual.size() != expected.size()) throw std::runtime_error(std::string(label)+" size mismatch");
    double maximum = 0, square = 0;
    for (size_t i=0; i<actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error(std::string(label)+" non-finite output");
        const double error = std::abs(actual[i]-expected[i]); maximum = std::max(maximum,error); square += error*error;
    }
    std::cout<<label<<" max_abs="<<maximum<<" rms="<<std::sqrt(square/actual.size())<<"\n";
    if (maximum > tolerance) throw std::runtime_error(std::string(label)+" exceeds reference tolerance");
}

void prompt_parity(const std::filesystem::path & directory) {
    auto model=std::make_shared<fish::FishAudioAssets>();
    model->config=fish::parse_drama3_config(json::parse_file(directory/"config.json"));
    model->resources.add_file("tokenizer_config",directory/"tokenizer_config.json");
    model->resources.add_file("tokenizer_json",directory/"tokenizer.json");
    fish::FishAudioTextTokenizer tokenizer(model);
    auto make_codes=[&](const json::Value & value){
        engine::codecs::FishDacCodes codes;codes.codebooks=model->config.fast.num_codebooks;
        for(const auto & v:value.as_array())codes.codes.push_back(static_cast<int32_t>(v.as_i64()));
        codes.frames=static_cast<int64_t>(codes.codes.size())/codes.codebooks;return codes;
    };
    const auto cases=json::parse_file(directory/"prompts.json");size_t count=0;
    for(const auto & test:cases.as_array()) {
        fish::FishAudioRequest request;request.text=json::require_string(test,"text");
        std::vector<engine::codecs::FishDacCodes> references;
        for(const auto & r:test.require("references").as_array()){
            fish::FishAudioReference ref;ref.text=json::require_string(r,"text");request.references.push_back(ref);
            references.push_back(make_codes(r.require("codes")));
        }
        std::optional<fish::FishAudioConversationTurn> history;
        if(const auto * h=test.find("history")) history=fish::FishAudioConversationTurn{json::require_string(*h,"text"),make_codes(h->require("codes"))};
        const bool error=json::optional_bool(test,"error",false);
        bool failed=false;fish::FishAudioPrompt result;
        try {result=fish::build_drama3_prompt(*model,tokenizer,request,references,history);}
        catch(const std::runtime_error &) {failed=true;}
        if(failed!=error)throw std::runtime_error("prompt rejection parity mismatch at "+std::to_string(count));
        if(!error){
            const auto & expected=test.require("matrix").as_array();
            if(expected.size()!=result.matrix.size())throw std::runtime_error("prompt size mismatch at "+std::to_string(count));
            for(size_t i=0;i<expected.size();++i)if(expected[i].as_i64()!=result.matrix[i])throw std::runtime_error("prompt token mismatch at "+std::to_string(count)+" element "+std::to_string(i));
        }
        ++count;
    }
    std::cout<<"PASS "<<count<<" original prompt/tokenizer cases\n";
}

int main(int argc, char ** argv) {
    try {
        if (argc!=3 && argc!=5) throw std::runtime_error("usage: drama3_parity FIXTURE_DIR cpu|cuda|vulkan|prompt [MODEL_GGUF OUTPUT_DIR]");
        const std::filesystem::path directory(argv[1]); const std::string kind(argv[2]);
        if(kind=="prompt"){prompt_parity(directory);return 0;}
        core::BackendConfig backend; backend.threads=4;
        if (kind=="cuda") backend.type=core::BackendType::Cuda;
        else if (kind=="vulkan") backend.type=core::BackendType::Vulkan;
        else if (kind!="cpu") throw std::runtime_error("invalid backend");
        // GPU attention uses the framework's F16 flash-attention/cache policy.
        if(kind!="cpu")tolerance=0.001;
        core::ExecutionContext execution(backend);
        auto source=argc==5?assets::open_tensor_source(argv[3],"model_weights"):assets::open_tensor_source(directory/"weights.safetensors");
        auto reference=assets::open_tensor_source(directory/"reference.safetensors");
        const auto cfg=fish::parse_drama3_config(json::parse_file(directory/"config.json"));
        const auto & b=cfg.drama3->backbone;
        engine::modules::Qwen35MoeDecoderRuntime runtime(source,b,execution,256*1024*1024,4*1024*1024,assets::TensorStorageType::Native);
        const auto input=reference->require_f32("input");
        const int64_t steps=static_cast<int64_t>(input.size())/b.hidden_size;
        if(argc==5) {
            const auto start=std::chrono::steady_clock::now();
            const auto hidden=runtime.forward_embeddings(input,steps).hidden;
            const auto logits=runtime.lm_head({hidden.end()-b.hidden_size,hidden.end()});
            const std::filesystem::path output(argv[4]);std::filesystem::create_directories(output);
            auto write=[&](const char * name,const std::vector<float> & values){std::ofstream file(output/name,std::ios::binary);file.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float));if(!file)throw std::runtime_error("probe output write failed");};
            write("hidden.f32",hidden);write("logits.f32",logits);
            std::cout<<"Full checkpoint prefill "<<steps<<" rows took "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"s\n";
            return 0;
        }
        const auto initial_hidden=runtime.forward_embeddings(input,steps).hidden;
        compare(initial_hidden,reference->require_f32("hidden"),"backbone full");
        if(kind=="vulkan") {
            if(runtime.forward_embeddings(input,steps).hidden!=initial_hidden)
                throw std::runtime_error("prefix hit changed hidden values");
            auto changed=input;
            changed.back()+=0.1F;
            const auto reused=runtime.forward_embeddings(changed,steps).hidden;
            runtime.release_graphs();
            if(runtime.forward_embeddings(changed,steps).hidden!=reused)
                throw std::runtime_error("changed suffix differs with prefix reuse");
            changed[0]+=0.2F;
            (void)runtime.forward_embeddings(changed,steps);
            if(runtime.forward_embeddings(input,steps).hidden!=initial_hidden)
                throw std::runtime_error("prefix invalidation retained stale state");
            std::cout<<"PASS exact prefix hit, changed suffix, and invalidation checks\n";
        }
        auto decode=runtime.create_decode_session(b.decode_cache_steps);
        auto prefix=std::vector<float>(input.begin(),input.begin()+3*b.hidden_size);
        auto cached=decode->prefill_embeddings(prefix,3).hidden;
        for(int64_t p=3;p<steps;++p){auto h=decode->run_embedding_step({input.begin()+p*b.hidden_size,input.begin()+(p+1)*b.hidden_size});cached.insert(cached.end(),h.begin(),h.end());}
        compare(cached,reference->require_f32("hidden"),"backbone cached");
        compare(runtime.lm_head({cached.end()-b.hidden_size,cached.end()}),reference->require_f32("logits"),"speech head");
        decode->reset();
        compare(decode->prefill_embeddings(input,steps).hidden,reference->require_f32("hidden"),"backbone reset");
        decode.reset();
        runtime.release_graphs();
        decode=runtime.create_decode_session(b.decode_cache_steps);
        compare(decode->prefill_embeddings(input,steps).hidden,reference->require_f32("hidden"),"backbone release/recreate");
        auto model=std::make_shared<fish::FishAudioAssets>(); model->config=cfg; model->model_weights=source;
        fish::Drama3Tower tower(model,execution,4*1024*1024,assets::TensorStorageType::Native);
        auto tower_inputs=reference->require_f32("tower_input");
        auto frames=reference->require_f32("frames");
        const size_t width=static_cast<size_t>(cfg.fast.num_codebooks+1);
        for(int repeat=0;repeat<3;++repeat){
            tower.reset(8);std::vector<float> outputs;
            for(int p=0;p<4;++p){
                std::vector<int32_t> frame(width),next(width);
                for(size_t i=0;i<width;++i){frame[i]=static_cast<int32_t>(frames[p*width+i]);next[i]=static_cast<int32_t>(frames[(p+1)*width+i]);}
                const std::vector<float> h(tower_inputs.begin()+p*b.hidden_size,tower_inputs.begin()+(p+1)*b.hidden_size);
                std::vector<int32_t> codes;
                size_t book=2;
                auto forced=[&](const std::vector<float> & logits){
                    if(logits.size()!=static_cast<size_t>(cfg.fast.vocab_size))throw std::runtime_error("codebook head shape mismatch");
                    for(float v:logits)if(!std::isfinite(v))throw std::runtime_error("non-finite codebook head");
                    return next.at(book++);
                };
                auto output=tower.run(h,frame,next,p,repeat!=2,forced,codes);outputs.insert(outputs.end(),output.begin(),output.end());
                if(repeat==2 && (book!=next.size() || !std::equal(codes.begin(),codes.end(),next.begin()+1)))
                    throw std::runtime_error("forced generation codebook mismatch");
                compare(tower.embed_backbone(frame,h),reference->require_f32("embed."+std::to_string(p)),"frame embedding");
            }
            compare(outputs,reference->require_f32("tower_hidden"),repeat==2?"tower device handoff":repeat?"tower reset":"tower causal");
            if(repeat==1)tower.release_graphs();
        }
        std::cout<<"PASS Drama 3 reference and cache parity on "<<kind<<"\n";return 0;
    } catch(const std::exception & e){std::cerr<<e.what()<<"\n";return 1;}
}
