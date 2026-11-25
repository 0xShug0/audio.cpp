#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/safetensors.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace assets=engine::assets;
struct FixtureDirectory {
    std::filesystem::path path=std::filesystem::temp_directory_path()/(
        "audiocpp_expert_override_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    FixtureDirectory() { std::filesystem::create_directories(path); }
    ~FixtureDirectory() { std::error_code error; std::filesystem::remove_all(path,error); }
};
int main() {
    try {
        FixtureDirectory directory;
        const auto & folder=directory.path;
        const auto input=folder/"source.safetensors";
        const std::string bank="model.language_model.layers.0.mlp.experts.gate_proj.weight";
        std::vector<float> values(3*2*256);
        for(size_t i=0;i<values.size();++i) values[i]=std::sin(float(i)*.013F)*.9F;
        std::vector<unsigned char> bytes(values.size()*sizeof(float));
        std::memcpy(bytes.data(),values.data(),bytes.size());
        engine::io::write_safetensors_file(input,{{bank,"F32",{3,2,256},bytes},
            {"conv.weight","F32",{2,3,256},bytes},{"dense.weight","F32",{6,256},bytes}});
        auto convert=[&](const std::filesystem::path & output, assets::GgufConversionOptions options={}) {
            assets::convert_tensor_sources_to_gguf({{input,""}},output,assets::TensorStorageType::Q4_K,false,false,{}, {},std::nullopt,options);
            return assets::open_tensor_source(output);
        };
        auto defaults=convert(folder/"default.gguf");
        if(defaults->require_metadata(bank).dtype!="f32" || defaults->require_metadata("conv.weight").dtype!="f32" ||
            defaults->require_metadata("dense.weight").dtype!="q4_k")
            throw std::runtime_error("default conversion changed existing rank-3 tensor policy");
        for(const auto & profile:std::vector<std::pair<assets::TensorStorageType,std::string>>{
            {assets::TensorStorageType::Q2_K,"q2_k"},{assets::TensorStorageType::Q3_K,"q3_k"},
            {assets::TensorStorageType::Q4_K,"q4_k"},{assets::TensorStorageType::Q5_K,"q5_k"},
            {assets::TensorStorageType::Q6_K,"q6_k"},{assets::TensorStorageType::Q8_0,"q8_0"}}) {
            assets::GgufConversionOptions options;
            options.type_overrides.push_back({"model.language_model.layers.0.mlp.experts.*",profile.first});
            auto source=convert(folder/(profile.second+".gguf"),options);
            const auto metadata=source->require_metadata(bank);
            std::cout << "dtype=" << metadata.dtype << " shape=";
            for(const auto dimension:metadata.shape) std::cout << dimension << ',';
            std::cout << '\n';
            if(source->require_metadata(bank).dtype!=profile.second || source->require_metadata(bank).shape!=std::vector<int64_t>{3,2,256})
                throw std::runtime_error("expert type or logical rank lost");
            if(source->require_metadata("conv.weight").dtype!="f32") throw std::runtime_error("expert override quantized an ordinary convolution");
            auto restored=source->require_f32(bank,{3,2,256});
            float maximum=0;
            for(size_t i=0;i<values.size();++i) maximum=std::max(maximum,std::abs(restored[i]-values[i]));
            if(maximum>.35F) throw std::runtime_error("expert bank layout or values changed");
            std::cout << profile.second << " max_dequantization_error=" << maximum << '\n';
        }
        assets::GgufConversionOptions invalid;
        invalid.type_overrides.push_back({"conv.weight",assets::TensorStorageType::Q4_K});
        bool rejected=false;
        try { convert(folder/"invalid-convolution.gguf",invalid); } catch(const std::exception &){rejected=true;}
        if(!rejected) throw std::runtime_error("ordinary rank-3 convolution quantization was accepted");
        std::cout << "rank3_default_preserved=true rank2_default_preserved=true profiles=6 convolution_override_rejected=true\n";
    } catch(const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
