enable f16;
#include "common_decls.tmpl"

#ifdef SRC_F32
#define SRC_TYPE f32
#elif defined(SRC_F16)
#define SRC_TYPE f16
#elif defined(SRC_BF16)
#define SRC_TYPE u32
#elif defined(SRC_I32)
#define SRC_TYPE i32
#endif

#ifdef DST_F32
#define DST_TYPE f32
#elif defined(DST_F16)
#define DST_TYPE f16
#elif defined(DST_I32)
#define DST_TYPE i32
#endif

@group(0) @binding(0)
var<storage, read_write> src: array<SRC_TYPE>;

@group(0) @binding(1)
#ifdef DST_BF16
var<storage, read_write> dst: array<atomic<u32>>;
#else
var<storage, read_write> dst: array<DST_TYPE>;
#endif

struct Params{
    ne: u32,
    offset_src: u32,
    offset_dst: u32,

    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,


    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    src_ne0: u32,
    src_ne1: u32,
    src_ne2: u32,

    dst_ne0: u32,
    dst_ne1: u32,
    dst_ne2: u32
};

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if (gid.x >= params.ne) {
        return;
    }

    var i = gid.x;
    let i3 = i / (params.src_ne2 * params.src_ne1 * params.src_ne0);
    i = i % (params.src_ne2 * params.src_ne1 * params.src_ne0);
    let i2 = i / (params.src_ne1 * params.src_ne0);
    i = i % (params.src_ne1 * params.src_ne0);
    let i1 = i / params.src_ne0;
    let i0 = i % params.src_ne0;

    var j = gid.x;
    let j3 = j / (params.dst_ne2 * params.dst_ne1 * params.dst_ne0);
    j = j % (params.dst_ne2 * params.dst_ne1 * params.dst_ne0);
    let j2 = j / (params.dst_ne1 * params.dst_ne0);
    j = j % (params.dst_ne1 * params.dst_ne0);
    let j1 = j / params.dst_ne0;
    let j0 = j % params.dst_ne0;

    let src_idx = i0 * params.stride_src0 + i1 * params.stride_src1 +
                  i2 * params.stride_src2 + i3 * params.stride_src3;

    let dst_idx = j0 * params.stride_dst0 + j1 * params.stride_dst1 +
                  j2 * params.stride_dst2 + j3 * params.stride_dst3;

#ifdef SRC_BF16
    let source_index = params.offset_src + src_idx;
    let source_word = src[source_index / 2u];
    let value = bf16_word_to_f32(source_word, source_index & 1u);
#else
    let value = f32(src[params.offset_src + src_idx]);
#endif
#ifdef DST_BF16
#ifdef SRC_BF16
    let half_bits = (source_word >> ((source_index & 1u) * 16u)) & 0xffffu;
#else
    let bits = bitcast<u32>(value);
    let rounded = (bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u;
    let half_bits = select(rounded, (bits >> 16u) | 0x40u, (bits & 0x7fffffffu) > 0x7f800000u) & 0xffffu;
#endif
    let target_index = params.offset_dst + dst_idx;
    let word_index = target_index / 2u;
    let shift = (target_index & 1u) * 16u;
    let mask = 0xffffu << shift;
    // Two independent BF16 elements can occupy the same storage word.
    var old = atomicLoad(&dst[word_index]);
    loop {
        let desired = (old & ~mask) | (half_bits << shift);
        let result = atomicCompareExchangeWeak(&dst[word_index], old, desired);
        if (result.exchanged) { break; }
        old = result.old_value;
    }
#else
#ifdef SRC_BF16
    dst[params.offset_dst + dst_idx] = DST_TYPE(value);
#else
    dst[params.offset_dst + dst_idx] = DST_TYPE(src[params.offset_src + src_idx]);
#endif
#endif
}
