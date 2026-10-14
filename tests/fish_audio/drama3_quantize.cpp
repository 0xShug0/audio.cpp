#include <ggml.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <deque>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: drama3_quantize TYPE COLS ROWS INPUT_F32 OUTPUT");
        const std::string name=argv[1];
        const ggml_type type=name=="q4_k"?GGML_TYPE_Q4_K:name=="q8_0"?GGML_TYPE_Q8_0:
            name=="bf16"?GGML_TYPE_BF16:name=="f32"?GGML_TYPE_F32:GGML_TYPE_COUNT;
        const int64_t cols=std::stoll(argv[2]),rows=std::stoll(argv[3]);
        if(type==GGML_TYPE_COUNT||cols<=0||rows<=0||cols%ggml_blck_size(type)!=0) throw std::runtime_error("invalid quantization shape/type");
        std::ifstream input(argv[4],std::ios::binary);std::ofstream output(argv[5],std::ios::binary|std::ios::trunc);
        if(!input||!output)throw std::runtime_error("cannot open quantization input/output");
        const int64_t chunk_rows=128;const size_t row_bytes=ggml_row_size(type,cols);
        ggml_quantize_init(type);
        std::deque<std::future<std::vector<uint8_t>>> pending;
        auto flush=[&](){auto bytes=pending.front().get();pending.pop_front();output.write(reinterpret_cast<const char *>(bytes.data()),static_cast<std::streamsize>(bytes.size()));if(!output)throw std::runtime_error("quantized tensor write failed");};
        for(int64_t offset=0;offset<rows;offset+=chunk_rows) {
            const int64_t count=std::min(chunk_rows,rows-offset);
            std::vector<float> values(static_cast<size_t>(count*cols));
            input.read(reinterpret_cast<char *>(values.data()),count*cols*sizeof(float));
            if(input.gcount()!=count*cols*static_cast<int64_t>(sizeof(float)))throw std::runtime_error("truncated F32 tensor");
            for(int64_t i=0;i<count*cols;++i)if(!std::isfinite(values[static_cast<size_t>(i)]))throw std::runtime_error("non-finite source value");
            pending.push_back(std::async(std::launch::async,[type,cols,count,row_bytes,values=std::move(values)](){
                std::vector<uint8_t> bytes(static_cast<size_t>(count)*row_bytes);
                const size_t written=ggml_quantize_chunk(type,values.data(),bytes.data(),0,count,cols,nullptr);
                if(written!=bytes.size())throw std::runtime_error("quantizer returned wrong byte count");
                if(type==GGML_TYPE_Q4_K||type==GGML_TYPE_Q8_0) {
                    const size_t block=ggml_type_size(type),scales=type==GGML_TYPE_Q4_K?2:1;
                    for(size_t offset=0;offset<bytes.size();offset+=block)for(size_t i=0;i<scales;++i) {
                        ggml_fp16_t value;std::memcpy(&value,bytes.data()+offset+i*sizeof(value),sizeof(value));
                        if(!std::isfinite(ggml_fp16_to_fp32(value)))throw std::runtime_error("quantization scale overflow; use BF16 precision");
                    }
                }
                return bytes;
            }));
            if(pending.size()>=8)flush();
        }
        while(!pending.empty())flush();
        if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing F32 tensor bytes");
        return 0;
    } catch(const std::exception & e) {std::cerr<<e.what()<<"\n";return 1;}
}
