#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char **) {
    ggml_backend_load_all();
    auto device = ggml_backend_dev_by_type(argc > 1 ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!device) return 77;
    auto backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return 1;
    int cases=0;
    for(int width:{31,33,67})for(int kernel:{1,3,7})for(int stride:{1,2})for(int dilation:{1,3}) {
        constexpr int ic=3,oc=5,padding=2;
        const int frames=(width+2*padding-dilation*(kernel-1)-1)/stride+1;
        if(frames<=0)continue;
        auto ctx=ggml_init({4*1024*1024,nullptr,true});
        auto a=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,kernel,ic,oc);
        auto b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,ic);
        auto old=ggml_conv_1d(ctx,a,b,stride,padding,dilation);
        auto tiled=ggml_conv_1d_tiled(ctx,a,b,stride,padding,dilation);
        auto graph=ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,old);
        int count=0;
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto op=ggml_graph_node(graph,i)->op;
            if(op==GGML_OP_MUL_MAT_TILED || op==GGML_OP_IM2COL_TILED)return 2;
        }
        ggml_build_forward_expand(graph,tiled);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto op=ggml_graph_node(graph,i)->op;
            count+=op==GGML_OP_MUL_MAT_TILED || op==GGML_OP_IM2COL_TILED;
        }
        if(count!=2)return 3;
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        std::vector<float> weights(kernel*ic*oc),input(width*ic),x(frames*oc),y(x.size());
        for(size_t i=0;i<weights.size();++i)weights[i]=std::sin(float(i)*0.37f)/7;
        for(size_t i=0;i<input.size();++i)input[i]=std::cos(float(i)*0.23f);
        ggml_backend_tensor_set(a,weights.data(),0,weights.size()*4);
        ggml_backend_tensor_set(b,input.data(),0,input.size()*4);
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 4;
        ggml_backend_tensor_get(old,x.data(),0,x.size()*4);
        ggml_backend_tensor_get(tiled,y.data(),0,y.size()*4);
        for(int out=0;out<oc;++out)for(int t=0;t<frames;++t) {
            double expected=0;
            for(int ch=0;ch<ic;++ch)for(int tap=0;tap<kernel;++tap) {
                int source=t*stride+tap*dilation-padding;
                if(source>=0 && source<width)expected+=double(input[ch*width+source])*weights[(out*ic+ch)*kernel+tap];
            }
            int index=out*frames+t;
            if(!std::isfinite(y[index]) || std::abs(y[index]-expected)>1e-5 || std::abs(x[index]-expected)>1e-5)return 5;
        }
        ggml_backend_buffer_free(buffer);ggml_free(ctx);++cases;
    }
    std::printf("PASS %d explicit tiled convolution graphs: scalar reference and untouched ordinary ops\n",cases);
    ggml_backend_free(backend);
}
