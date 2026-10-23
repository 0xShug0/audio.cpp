#include "mmvq.cuh"
#include "../../ggml-cuda/vecdotq.cuh"
#include "../../ggml-cuda/unary.cuh"

namespace {

template<bool gate>
__global__ __launch_bounds__(128) void halo_mmvq_q8_0(
    const void * __restrict__ weights, const block_q8_1 * __restrict__ input,
    float * __restrict__ output, int blocks, int rows, int row_stride,
    ggml_cuda_mm_fusion_args_device fusion) {
    // Each wave owns one row: no cross-wave scratch, barrier or idle waves
    // after reduction. Four rows share a CTA and reuse the same input vector.
    const int row = int(blockIdx.x) * 4 + threadIdx.y;
    if(row>=rows)return;
    const int lane=threadIdx.x;
    float value=0, gated=0;
    for(int block=lane/4;block<blocks;block+=8) {
        const int quant=2*(lane%4);
        value+=vec_dot_q8_0_q8_1(weights,input+block,row*row_stride+block,quant);
        if constexpr(gate)gated+=vec_dot_q8_0_q8_1(fusion.gate,input+block,row*row_stride+block,quant);
    }
    value=warp_reduce_sum<32>(value);
    if constexpr(gate)gated=warp_reduce_sum<32>(gated);
    if(lane==0) {
        if(fusion.x_bias)value+=static_cast<const float *>(fusion.x_bias)[row];
        if constexpr(gate) {
            if(fusion.gate_bias)gated+=static_cast<const float *>(fusion.gate_bias)[row];
            switch(fusion.glu_op) {
                case GGML_GLU_OP_SWIGLU:value*=ggml_cuda_op_silu_single(gated);break;
                case GGML_GLU_OP_GEGLU:value*=ggml_cuda_op_gelu_single(gated);break;
                case GGML_GLU_OP_SWIGLU_OAI:value=ggml_cuda_op_swiglu_oai_single(gated,value);break;
                default:value*=gated;break;
            }
        }
        output[row]=value;
    }
}

} // namespace

void ggml_hip_halo_mmvq_q8_0(const void * weights,const void * input,
    float * output,int columns,int rows,int row_stride,
    ggml_cuda_mm_fusion_args_device fusion,hipStream_t stream) {
    if(fusion.gate)halo_mmvq_q8_0<true><<<dim3((rows+3)/4),dim3(32,4),0,stream>>>(
        weights,static_cast<const block_q8_1 *>(input),output,columns/32,rows,row_stride,fusion);
    else halo_mmvq_q8_0<false><<<dim3((rows+3)/4),dim3(32,4),0,stream>>>(
        weights,static_cast<const block_q8_1 *>(input),output,columns/32,rows,row_stride,fusion);
}
