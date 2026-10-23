// Frozen from c8365f391307a97bfe133e0872e632fe572ecace:
// external/ggml/src/ggml-cuda/im2col.cu, im2col_kernel. The name is changed,
// grid constants are expanded and GGML_UNUSED is spelled as a void cast.
template<typename T>
static __global__ void original_im2col_kernel(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KH_KW) return;
    const int64_t iic = i / KH_KW;
    const int64_t rem = i - iic * KH_KW;
    const int64_t ikh = rem / KW;
    const int64_t ikw = rem - ikh * KW;
    for (int64_t iow = blockIdx.y; iow < OW; iow += 65535) {
        for (int64_t iz = blockIdx.z; iz < N_OH; iz += 65535) {
            const int64_t in = iz / OH;
            const int64_t ioh = iz - in * OH;
            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;
            const int64_t offset_dst =
                ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;
            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) dst[offset_dst] = 0.0f;
            else {
                const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
                dst[offset_dst] = x[offset_src + iih * IW + iiw];
            }
        }
    }
    (void)IC; (void)KH;
}
