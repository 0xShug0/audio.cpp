// Experimental alternatives stay in the extended probe until measured.
template<bool full, bool packed>
static __device__ float candidate_dot(const void * weights, const __half * scales,
    const block_q8_1 * input, int index, int block, int quant) {
    constexpr int count=full?8:2;
    int v[count],u[count];
    if constexpr(packed) {
        const int2 * values=static_cast<const int2 *>(weights)+index*4+quant/2;
#pragma unroll
        for(int i=0;i<count/2;++i) { const auto pair=values[i];v[i*2]=pair.x;v[i*2+1]=pair.y; }
    } else {
        const auto * row=static_cast<const block_q8_0 *>(weights)+index;
#pragma unroll
        for(int i=0;i<count;++i)v[i]=get_int_b2(row->qs,quant+i);
    }
#pragma unroll
    for(int i=0;i<count;++i)u[i]=get_int_b4(input[block].qs,quant+i);
    const float d=packed?__half2float(scales[index]):__half2float(static_cast<const block_q8_0 *>(weights)[index].d);
    return vec_dot_q8_0_q8_1_impl<float,count>(v,u,d,__half2float(__low2half(input[block].ds)));
}

template<bool gate,bool full,bool packed>
static __global__ __launch_bounds__(128) void candidate_mmvq(
    const void * weights,const __half * scales,const __half * gate_scales,const block_q8_1 * input,
    float * output,int blocks,int rows,int row_stride,ggml_cuda_mm_fusion_args_device fusion) {
    const int row=int(blockIdx.x)*4+threadIdx.y;
    if(row>=rows)return;
    const int lane=threadIdx.x;
    float value=0,gated=0;
    for(int block=full?lane:lane/4;block<blocks;block+=full?32:8) {
        const int quant=full?0:2*(lane%4);
        value+=candidate_dot<full,packed>(weights,scales,input,row*row_stride+block,block,quant);
        if constexpr(gate)gated+=candidate_dot<full,packed>(fusion.gate,gate_scales,input,row*row_stride+block,block,quant);
    }
    value=warp_reduce_sum<32>(value);
    if constexpr(gate)gated=warp_reduce_sum<32>(gated);
    if(lane==0) {
        if(fusion.x_bias)value+=static_cast<const float *>(fusion.x_bias)[row];
        if constexpr(gate) {
            if(fusion.gate_bias)gated+=static_cast<const float *>(fusion.gate_bias)[row];
            value*=ggml_cuda_op_silu_single(gated);
        }
        output[row]=value;
    }
}

template<bool full,bool packed>
static void launch_candidate(const void * weights,const __half * scales,const __half * gate_scales,
    const block_q8_1 * input,float * output,int k,int m,int rs,ggml_cuda_mm_fusion_args_device fusion) {
    if(fusion.gate)candidate_mmvq<true,full,packed><<<dim3((m+3)/4),dim3(32,4)>>>(
        weights,scales,gate_scales,input,output,k/32,m,rs,fusion);
    else candidate_mmvq<false,full,packed><<<dim3((m+3)/4),dim3(32,4)>>>(
        weights,scales,gate_scales,input,output,k/32,m,rs,fusion);
}
