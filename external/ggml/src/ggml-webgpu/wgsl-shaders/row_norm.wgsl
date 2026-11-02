#if defined(SRC_F16) || defined(DST_F16)
enable f16;
#endif

#ifdef SRC_F16
#define SRC_TYPE f16
#else
#define SRC_TYPE f32
#endif

#ifdef DST_F16
#define DST_TYPE f16
#else
#define DST_TYPE f32
#endif

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of src/dst
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    eps: f32,
#ifdef GROUP_NORM
    groups: u32,
#endif
};

@group(0) @binding(0)
var<storage, read_write> src: array<SRC_TYPE>;

#ifdef INPLACE
@group(0) @binding(1)
var<uniform> params: Params;
#else
@group(0) @binding(1)
var<storage, read_write> dst: array<DST_TYPE>;

@group(0) @binding(2)
var<uniform> params: Params;
#endif

var<workgroup> scratch: array<f32, WG_SIZE * 2u>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    var i = wid.x + wid.y * num_wg.x;
#ifdef GROUP_NORM
    // One workgroup per channel group and batch, with GGML's ceil grouping.
    if (i >= params.groups * params.ne3) {
        return;
    }
    let i3 = i / params.groups;
    let channels = (params.ne2 + params.groups - 1u) / params.groups;
    let first_channel = (i % params.groups) * channels;
    if (first_channel >= params.ne2) {
        return;
    }
    let count = params.ne0 * params.ne1 * min(channels, params.ne2 - first_channel);
    let i_src_row = params.offset_src + i3 * params.stride_src3 + first_channel * params.stride_src2;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + first_channel * params.stride_dst2;
#else
    // one workgroup per row
    if (i >= params.ne1 * params.ne2 * params.ne3) {
        return;
    }
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_src_row = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2 + i1 * params.stride_src1;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;
    let count = params.ne0;
#endif

    let elems = (count + WG_SIZE - 1) / WG_SIZE;

    var sum = 0.0f;
    var col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= count) {
            break;
        }
#ifdef GROUP_NORM
        let idx = i_src_row + col % params.ne0 + (col / params.ne0) % params.ne1 * params.stride_src1 +
                  col / (params.ne0 * params.ne1) * params.stride_src2;
        let v = f32(src[idx]);
#else
        let v = f32(src[i_src_row + col]);
#endif
#ifdef NORM
        sum += v;
#else
        sum += v * v;
#endif
        col += WG_SIZE;
    }

    scratch[lid.x] = sum;
    workgroupBarrier();

    var offset: u32 = WG_SIZE / 2u;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] += scratch[lid.x + offset];
        }
        offset /= 2u;
        workgroupBarrier();
    }
    sum = scratch[0];

#ifdef NORM
    let mean = sum / f32(count);
    var sq_sum = 0.0f;
    col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= count) {
            break;
        }
#ifdef GROUP_NORM
        let idx = i_src_row + col % params.ne0 + (col / params.ne0) % params.ne1 * params.stride_src1 +
                  col / (params.ne0 * params.ne1) * params.stride_src2;
        let v = f32(src[idx]);
#else
        let v = f32(src[i_src_row + col]);
#endif
        let d = v - mean;
        sq_sum += d * d;
        col += WG_SIZE;
    }

    workgroupBarrier();
    scratch[lid.x] = sq_sum;
    workgroupBarrier();
    offset = WG_SIZE / 2u;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] += scratch[lid.x + offset];
        }
        offset /= 2u;
        workgroupBarrier();
    }

    let variance = scratch[0] / f32(count);
    let scale = 1.0 / sqrt(variance + params.eps);
#elif defined(RMS_NORM)
    let scale = 1.0/sqrt(sum/f32(params.ne0) + params.eps);
#elif defined(L2_NORM)
    let scale = 1.0/max(sqrt(sum), params.eps);
#endif

#ifdef NORM
    let mean_val = mean;
#else
    let mean_val = 0.0f;
#endif

    col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= count) {
            break;
        }
#ifdef GROUP_NORM
        let i0 = col % params.ne0;
        let i1 = (col / params.ne0) % params.ne1;
        let i2 = col / (params.ne0 * params.ne1);
        let i_src = i_src_row + i0 + i1 * params.stride_src1 + i2 * params.stride_src2;
        let i_dst = i_dst_row + i0 + i1 * params.stride_dst1 + i2 * params.stride_dst2;
#else
        let i_src = i_src_row + col;
        let i_dst = i_dst_row + col;
#endif
        let v = src[i_src];
#ifdef INPLACE
        src[i_dst] = scale * (v - mean_val);
#else
        dst[i_dst] = scale * (v - mean_val);
#endif
        col += WG_SIZE;
    }
}
