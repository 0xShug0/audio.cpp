#ifdef SRC_F16
enable f16;
#define SRC_TYPE f16
#else
#define SRC_TYPE f32
#endif

@group(0) @binding(0) var<storage, read_write> src: array<SRC_TYPE>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,
    offset_src: u32,
    offset_dst: u32,
    stride0: u32,
    stride1: u32,
    stride2: u32,
    stride3: u32,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    kernel_x: u32,
    kernel_y: u32,
    step_x: u32,
    step_y: u32,
    pad_x: i32,
    pad_y: i32,
};
@group(0) @binding(2) var<uniform> params: Params;

// Each lane owns one output: neighboring lanes read neighboring windows.
// Accumulate locally, without atomics, intermediate buffers or dispatches.
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
    let index = gid.x + groups.x * u32(WG_SIZE) * gid.y;
    if (index >= params.ne) { return; }
    let x = index % params.dst_width;
    let y = (index / params.dst_width) % params.dst_height;
    let plane = index / (params.dst_width * params.dst_height);
    let base = params.offset_src + (plane % params.channels) * params.stride2 +
               (plane / params.channels) * params.stride3;
    let left = i32(x * params.step_x) - params.pad_x;
    let top = i32(y * params.step_y) - params.pad_y;
#ifdef POOL_MAX
    var value = bitcast<f32>(0xff7fffffu);
#else
    var value = 0.0;
#endif
    for (var ky = 0u; ky < params.kernel_y; ky++) {
        let iy = top + i32(ky);
        if (iy < 0 || iy >= i32(params.src_height)) { continue; }
        for (var kx = 0u; kx < params.kernel_x; kx++) {
            let ix = left + i32(kx);
            if (ix < 0 || ix >= i32(params.src_width)) { continue; }
            let sample = f32(src[base + u32(iy) * params.stride1 + u32(ix) * params.stride0]);
#ifdef POOL_MAX
            value = max(sample, value);
#else
            value += sample;
#endif
        }
    }
#ifndef POOL_MAX
    // GGML includes padding in the averaging divisor.
    value /= f32(params.kernel_x * params.kernel_y);
#endif
    dst[params.offset_dst + index] = value;
}
