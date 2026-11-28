#include "rmvpe_cuda_runtime.h"
#include "engine/framework/core/backend.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace engine::modules::detail {
namespace {
void check(cudaError_t result, const char * operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

// The three recurrent dot products retain GGML's float-pair FMA and
// 128-thread XOR reduction. Each block writes a distinct next-state element;
// the following frame reads that completed output instead of overwriting its input.
__global__ void recurrent_step(const float * weight, const float * bias,
        const float * input, const float * old_state, const float * keep,
        float * next_state, int hidden) {
    const int c = blockIdx.x, tid = threadIdx.x;
    float sums[3] = {0,0,0};
    for (int k = 2*tid; k < hidden; k += 2*blockDim.x) {
        for (int g=0;g<3;++g) {
            const auto row=(g*hidden+c)*hidden;
            sums[g]=fmaf(weight[row+k],old_state[k],sums[g]);
            sums[g]=fmaf(weight[row+k+1],old_state[k+1],sums[g]);
        }
    }
    for(int d=16;d>0;d>>=1) for(int g=0;g<3;++g)
        sums[g]+=__shfl_xor_sync(0xffffffff,sums[g],d);
    __shared__ float partial[3][32];
    if(tid<32) for(int g=0;g<3;++g) partial[g][tid]=0;
    __syncthreads();
    if(tid%32==0) for(int g=0;g<3;++g) partial[g][tid/32]=sums[g];
    __syncthreads();
    if(tid<32) {
        for(int g=0;g<3;++g) sums[g]=partial[g][tid];
        for(int d=16;d>0;d>>=1) for(int g=0;g<3;++g)
            sums[g]+=__shfl_xor_sync(0xffffffff,sums[g],d);
        if(tid==0) {
            for(int g=0;g<3;++g) sums[g]=__fadd_rn(sums[g],bias[g*hidden+c]);
            const float old=old_state[c];
            const float r=1.0f/(1.0f+expf(-__fadd_rn(input[c],sums[0])));
            const float z=1.0f/(1.0f+expf(-__fadd_rn(input[hidden+c],sums[1])));
            const float n=tanhf(__fadd_rn(input[2*hidden+c],__fmul_rn(r,sums[2])));
            const float updated=__fadd_rn(n,__fmul_rn(z,__fsub_rn(old,n)));
            next_state[c]=__fadd_rn(old,__fmul_rn(*keep,__fsub_rn(updated,old)));
        }
    }
}

__global__ void transpose_conv(const float * input, const float * weight, float * output,
        int iw, int ih, int ow, int oh, int cin, int cout, int batch) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=ow*oh*cout*batch) return;
    const int x=index%ow,y=(index/ow)%oh,c=(index/(ow*oh))%cout,n=index/(ow*oh*cout);
    int input_offsets[4], weight_offsets[4], count=0;
    // The valid spatial taps do not depend on input channel. Compute their
    // bounds and stride alignment once, preserving kh/kw accumulation order.
    for(int ky=0;ky<3;++ky) {
        int sy=y-ky;
        if(sy<0 || sy%2 || sy/2>=ih) continue;
        for(int kx=0;kx<3;++kx) {
            int sx=x-kx;
            if(sx<0 || sx%2 || sx/2>=iw) continue;
            input_offsets[count]=(sy/2)*iw+sx/2;
            weight_offsets[count++]=ky*3+kx;
        }
    }
    float sum=0;
    for(int ic=0;ic<cin;++ic) {
        const auto * values=input+(n*cin+ic)*iw*ih;
        const auto * taps=weight+(ic*cout+c)*9;
#pragma unroll
        for(int j=0;j<4;++j) if(j<count)
            sum=fmaf(values[input_offsets[j]],taps[weight_offsets[j]],sum);
    }
    output[index]=sum;
}

}

void rmvpe_cuda_conv_transpose(ggml_backend_t backend, ggml_tensor * output) {
    const auto * w=output->src[0];
    const auto * x=output->src[1];
    if(output->op!=GGML_OP_CONV_TRANSPOSE_2D || output->op_params[0]!=2 ||
            w->ne[0]!=3 || w->ne[1]!=3 || w->type!=GGML_TYPE_F32 ||
            x->type!=GGML_TYPE_F32 || output->type!=GGML_TYPE_F32 ||
            !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(output))
        throw std::runtime_error("Native RMVPE transpose convolution requires its contiguous F32 3x3/stride-2 operator");
    const auto stream=static_cast<cudaStream_t>(core::backend_cuda_stream(backend));
    const auto elements=ggml_nelements(output);
    transpose_conv<<<(elements+255)/256,256,0,stream>>>(
        static_cast<const float *>(x->data),static_cast<const float *>(w->data),
        static_cast<float *>(output->data),x->ne[0],x->ne[1],output->ne[0],output->ne[1],
        x->ne[2],w->ne[2],x->ne[3]);
    check(cudaGetLastError(),"RMVPE native transpose convolution");
}

struct RmvpeCudaGru::Impl {
    int device = 0;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaGraph_t pair_graph = nullptr;
    cudaGraphExec_t pair_executable = nullptr;
    ~Impl() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (pair_executable) cudaGraphExecDestroy(pair_executable);
        if (pair_graph) cudaGraphDestroy(pair_graph);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};

RmvpeCudaGru::RmvpeCudaGru(ggml_backend_t backend, int device, int frames, int hidden,
        bool reverse, const ggml_tensor * projected, const ggml_tensor * weight,
        const ggml_tensor * bias, ggml_tensor * state, const ggml_tensor * keep,
        ggml_tensor * sequence) : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    if (hidden != 256 || frames <= 0) throw std::runtime_error("Native RMVPE recurrence expects its 256-unit GRU");
    const ggml_tensor * tensors[] = {projected, weight, bias, state, keep, sequence};
    for (const auto * tensor : tensors) {
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor))
            throw std::runtime_error("Native RMVPE recurrence requires contiguous F32 tensors");
    }
    check(cudaSetDevice(device), "RMVPE select device");
    impl_->stream = static_cast<cudaStream_t>(core::backend_cuda_stream(backend));
    check(cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeThreadLocal), "RMVPE begin capture");
    const float * previous = static_cast<const float *>(state->data);
    for (int i = 0; i < frames; ++i) {
        const int t = reverse ? frames-1-i : i;
        float * next = static_cast<float *>(sequence->data) + t*hidden;
        recurrent_step<<<hidden, 128, 0, impl_->stream>>>(
            static_cast<const float *>(weight->data), static_cast<const float *>(bias->data),
            static_cast<const float *>(projected->data) + t*3*hidden, previous,
            static_cast<const float *>(keep->data) + t, next, hidden);
        previous = next;
    }
    check(cudaMemcpyAsync(state->data, previous, hidden*sizeof(float), cudaMemcpyDeviceToDevice,
        impl_->stream), "RMVPE retain recurrent state");
    check(cudaStreamEndCapture(impl_->stream, &impl_->graph), "RMVPE end capture");
    check(cudaGraphInstantiate(&impl_->executable, impl_->graph, nullptr, nullptr, 0), "RMVPE instantiate recurrence");
}

RmvpeCudaGru::~RmvpeCudaGru() = default;

void RmvpeCudaGru::prepare_pair(const RmvpeCudaGru & reverse) {
    if (impl_->pair_executable) return;
    if (impl_->device != reverse.impl_->device || impl_->stream != reverse.impl_->stream)
        throw std::runtime_error("RMVPE paired recurrence requires the same CUDA backend");
    check(cudaSetDevice(impl_->device), "RMVPE select device");
    check(cudaGraphCreate(&impl_->pair_graph, 0), "RMVPE create paired recurrence");
    try {
        cudaGraphNode_t node;
        check(cudaGraphAddChildGraphNode(&node, impl_->pair_graph, nullptr, 0, impl_->graph),
            "RMVPE add forward recurrence");
        check(cudaGraphAddChildGraphNode(&node, impl_->pair_graph, nullptr, 0, reverse.impl_->graph),
            "RMVPE add reverse recurrence");
        check(cudaGraphInstantiate(&impl_->pair_executable, impl_->pair_graph, nullptr, nullptr, 0),
            "RMVPE instantiate paired recurrence");
    } catch (...) {
        if (impl_->pair_executable) cudaGraphExecDestroy(impl_->pair_executable);
        cudaGraphDestroy(impl_->pair_graph);
        impl_->pair_executable = nullptr;
        impl_->pair_graph = nullptr;
        throw;
    }
}

void RmvpeCudaGru::compute_pair() {
    check(cudaSetDevice(impl_->device), "RMVPE select device");
    check(cudaGraphLaunch(impl_->pair_executable, impl_->stream), "RMVPE launch paired recurrence");
    check(cudaStreamSynchronize(impl_->stream), "RMVPE finish paired recurrence");
}

void RmvpeCudaGru::compute() {
    check(cudaSetDevice(impl_->device), "RMVPE select device");
    check(cudaGraphLaunch(impl_->executable, impl_->stream), "RMVPE launch recurrence");
    // Match graph_compute: tensor_get uses a separate per-thread copy stream.
    check(cudaStreamSynchronize(impl_->stream), "RMVPE finish recurrence");
}
} // namespace engine::modules::detail
