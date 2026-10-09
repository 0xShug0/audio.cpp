#pragma once

#include "engine/models/sherpa_kws/assets.h"
#include "engine/models/sherpa_kws/transducer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::sherpa_kws {

struct KeywordDefinition {
    std::vector<int32_t> tokens;
    std::string phrase;
    float score = 1.0F;
    float threshold = 0.25F;
};

struct KeywordDetection {
    std::string phrase;
    int64_t start_frame = 0;
    int64_t end_frame = 0;
    float confidence = 0.0F;
};

class KeywordDecoder {
public:
    explicit KeywordDecoder(std::shared_ptr<const SherpaKwsAssets> assets);
    ~KeywordDecoder();

    void configure(std::vector<KeywordDefinition> keywords, int max_active_paths,
                   int num_trailing_blanks);
    void reset();
    std::vector<KeywordDetection> append(const std::vector<float> & encoder_output,
                                         int64_t frames, int64_t channels);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::sherpa_kws
