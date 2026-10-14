#include "engine/models/fish_audio/assets.h"
#include "engine/framework/codecs/fish_dac_codec_runtime.h"
#include "engine/framework/core/execution_context.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace codecs=engine::codecs;
codecs::FishDacCodes slice(const codecs::FishDacCodes & all,int64_t first,int64_t count) {
    codecs::FishDacCodes out;out.codebooks=all.codebooks;out.frames=count;
    for(int64_t b=0;b<out.codebooks;++b)
        out.codes.insert(out.codes.end(),all.codes.begin()+b*all.frames+first,all.codes.begin()+b*all.frames+first+count);
    return out;
}
int main(int argc,char ** argv) {
    try {
        if(argc!=3)throw std::runtime_error("usage: drama3_codec_queue MODEL cpu|cuda|vulkan");
        engine::core::BackendConfig backend;backend.threads=8;
        const std::string kind(argv[2]);
        if(kind=="vulkan")backend.type=engine::core::BackendType::Vulkan;
        else if(kind=="cuda")backend.type=engine::core::BackendType::Cuda;
        else if(kind!="cpu")throw std::runtime_error("unknown backend");
        engine::core::ExecutionContext execution(backend);
        auto assets=engine::models::fish_audio::load_fish_audio_assets(argv[1]);
        codecs::FishDacCodecRuntimeOptions options;
        auto component=codecs::FishDacCodecComponent::load_from_tensor_source(assets->codec_weights,assets->config.codec,{},execution.backend(),execution.backend_type(),options);
        codecs::FishDacCodecRuntime codec(component,backend,8,options.graph_arena_bytes);
        assets->codec_weights->release_storage();
        codecs::FishDacCodes all;all.codebooks=assets->config.codec.total_codebooks;all.frames=137;
        for(int64_t b=0;b<all.codebooks;++b)for(int64_t p=0;p<all.frames;++p)
            all.codes.push_back(static_cast<int32_t>((p*97+b*173)%(b==0?4096:1024)));
        engine::runtime::AudioBuffer serial;
        codec.reset_decode_stream();
        for(int64_t p=0;p<all.frames;++p)engine::runtime::append_audio_buffer(serial,codec.decode_stream(slice(all,p,1)));
        codec.reset_decode_stream();
        const auto queued=codec.decode_stream_frames(all);
        if(queued.samples!=serial.samples || queued.sample_rate!=serial.sample_rate)throw std::runtime_error("queued codec differs from serial frame math");
        codec.reset_decode_stream();
        engine::runtime::AudioBuffer mixed;
        int64_t first=0;
        for(int64_t count:{1,7,8,9,112}) {
            auto part=slice(all,first,count);
            auto audio=count==1?codec.decode_stream(part):codec.decode_stream_frames(part);
            engine::runtime::append_audio_buffer(mixed,audio);first+=count;
        }
        if(first!=all.frames || mixed.samples!=serial.samples)throw std::runtime_error("mixed frame/batch continuation differs");
        codec.release_runtime_graphs();
        codec.reset_decode_stream();
        if(codec.decode_stream_frames(all).samples!=serial.samples)throw std::runtime_error("codec release/recreation differs");
        bool rejected=false;
        try {codec.decode_stream_frames({});}catch(const std::runtime_error &){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid queued codec shape not rejected");
        codec.reset_decode_stream();
        if(codec.decode_stream_frames(slice(all,0,9)).samples!=std::vector<float>(serial.samples.begin(),serial.samples.begin()+9*assets->config.codec.frame_length))
            throw std::runtime_error("codec did not recover after invalid input");
        std::cout<<"PASS "<<kind<<": exact serial/queued PCM, 137 frames, 128-frame attention window, mixed 1/7/8/9/112-frame calls, release/recreate, rejection/recovery\n";
        return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<"\n";return 1;}
}
