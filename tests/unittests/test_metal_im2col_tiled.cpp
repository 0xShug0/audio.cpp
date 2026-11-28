#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
struct Storage {
    ggml_context * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    ~Storage() {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};
}

int main() {
    auto backend = ggml_backend_metal_init();
    if (!backend) return 77;
    size_t checked = 0;
    try {
        // Cover partial tiles, offset views, batches, dilation, and production channel widths.
        for (int i = 0; i < 56; ++i) for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            int w=i<48 ? 7+i%5 : 483, h=i<48 ? 5+i%4 : 9;
            int c=i<48 ? 1+i%3 : 64*(1+(i-48)%4), n=1+i%2;
            int kw=i<48 ? 1+2*(i%3) : 3, kh=1+2*((i/3)%2);
            int sx=1+i%2, sy=1+(i/2)%2, dx=1+(i/5)%2, dy=1+(i/7)%2;
            int px=kw/2, py=kh/2;
            Storage storage;
            auto ctx=storage.ctx;
            const size_t count=size_t(w)*h*c*n;
            auto base=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,count+2);
            auto x=ggml_view_4d(ctx,base,w,h,c,n,w*4,w*h*4,w*h*c*4,4);
            auto weight=ggml_new_tensor_4d(ctx,type,kw,kh,c,1);
            auto old=ggml_im2col(ctx,weight,x,sx,sy,px,py,dx,dy,true,type);
            auto tiled=ggml_im2col(ctx,weight,x,sx,sy,px,py,dx,dy,true,type);
            ggml_im2col_2d_set_lowering(tiled,GGML_IM2COL_2D_LOWERING_METAL_TILED);
            auto graph=ggml_new_graph(ctx);
            ggml_set_output(old); ggml_set_output(tiled);
            ggml_build_forward_expand(graph,old); ggml_build_forward_expand(graph,tiled);
            storage.buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
            if(!storage.buffer) throw std::runtime_error("allocation failed");
            std::vector<float> input(count+2);
            for (int pattern = 0; pattern < 2; ++pattern) {
                for (size_t j = 0; j < input.size(); ++j) {
                    input[j] = float(int((j * 37 + i + pattern * 71) % 401) - 200) / 8;
                }
                ggml_backend_tensor_set(base, input.data(), 0, input.size() * sizeof(float));
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("compute failed");
                }
                std::vector<unsigned char> actual(ggml_nbytes(tiled)), legacy(actual.size());
                ggml_backend_tensor_get(tiled, actual.data(), 0, actual.size());
                ggml_backend_tensor_get(old, legacy.data(), 0, legacy.size());
                if (actual != legacy) throw std::runtime_error("legacy GPU mismatch");
                size_t at = 0;
                // Independent coordinate-based oracle, not output from either GPU kernel.
                for (int bn = 0; bn < n; ++bn) for (int oy = 0; oy < tiled->ne[2]; ++oy)
                for (int ox = 0; ox < tiled->ne[1]; ++ox) for (int ch = 0; ch < c; ++ch)
                for (int ky = 0; ky < kh; ++ky) for (int kx = 0; kx < kw; ++kx) {
                    const int iy = oy * sy + ky * dy - py, ix = ox * sx + kx * dx - px;
                    const float expected = ix >= 0 && ix < w && iy >= 0 && iy < h
                        ? input[1 + ((bn * c + ch) * h + iy) * w + ix] : 0;
                    if (type == GGML_TYPE_F32) {
                        float value;
                        std::memcpy(&value, actual.data() + at * sizeof(value), sizeof(value));
                        if (value != expected) throw std::runtime_error("scalar F32 mismatch");
                    } else {
                        ggml_fp16_t value;
                        std::memcpy(&value, actual.data() + at * sizeof(value), sizeof(value));
                        if (value != ggml_fp32_to_fp16(expected)) throw std::runtime_error("scalar F16 mismatch");
                    }
                    ++at;
                }
                checked += at;
            }
        }
        std::cout<<"PASS 112 shapes/types, two input patterns, "<<checked<<" exact values; legacy Metal and independent scalar oracle\n";
    } catch(const std::exception & e) {
        std::cerr<<e.what()<<'\n';ggml_backend_free(backend);return 1;
    }
    ggml_backend_free(backend);
}
