#ifdef WEIGHT_F16
enable f16;
#define WEIGHT_TYPE f16
#else
#define WEIGHT_TYPE f32
#endif

@group(0) @binding(0) var<storage, read_write> weights: array<WEIGHT_TYPE>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
struct Params {
    offset_w: u32, offset_i: u32, offset_o: u32,
    sw0: u32, sw1: u32, sw2: u32, sw3: u32,
    si0: u32, si1: u32, si2: u32, si3: u32,
    kw: u32, kh: u32, ic: u32,
    iw: u32, ih: u32,
    ow: u32, oh: u32, oc: u32, batches: u32,
    stride: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_w: array<f32, 256>;
var<workgroup> tile_i: array<f32, 256>;

fn store_output(oc: u32, column: u32, batch: u32, phase: vec2<u32>, value: f32) {
    let width = (params.ow + params.stride - 1u) / params.stride;
    let x = (column % width) * params.stride + phase.x;
    let y = (column / width) * params.stride + phase.y;
    if (oc < params.oc && x < params.ow && y < params.oh) {
        output[params.offset_o + x + params.ow * (y + params.oh * (oc + params.oc * batch))] = value;
    }
}

// A 16x16 output tile, with four outputs per lane. Each stride phase is a
// tiled matrix product over input channels; overlapping kernel contributions
// accumulate locally. No zero insertion, atomics or temporary activation bank.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let width = (params.ow + params.stride - 1u) / params.stride;
    let height = (params.oh + params.stride - 1u) / params.stride;
    let rows = (params.oc + 15u) / 16u;
    let columns = (width * height + 15u) / 16u;
    let linear = group.x + groups.x * group.y;
    let phase_batch = linear / (rows * columns);
    let batch = phase_batch / (params.stride * params.stride);
    if (batch >= params.batches) { return; }
    let phase = vec2<u32>(phase_batch % params.stride,
                         (phase_batch / params.stride) % params.stride);
    let row_base = (linear % rows) * 16u;
    let column_base = ((linear / rows) % columns) * 16u;
    let row = (lane % 8u) * 2u;
    let column = (lane / 8u) * 2u;
    var total = vec4<f32>(0.0);
    let nx = (params.kw + params.stride - 1u - phase.x) / params.stride;
    let ny = (params.kh + params.stride - 1u - phase.y) / params.stride;
    // Descending kernel offsets visit source pixels in increasing order,
    // matching the CPU reference's accumulation over overlapping windows.
    for (var y = 0u; y < ny; y++) {
        let ky = phase.y + (ny - 1u - y) * params.stride;
        for (var x = 0u; x < nx; x++) {
            let kx = phase.x + (nx - 1u - x) * params.stride;
            var acc = vec4<f32>(0.0);
            for (var channel = 0u; channel < params.ic; channel += 16u) {
                for (var t = lane; t < 256u; t += 64u) {
                    let oc = row_base + t / 16u;
                    let ic = channel + t % 16u;
                    var w = 0.0;
                    if (oc < params.oc && ic < params.ic) {
                        w = f32(weights[params.offset_w + kx * params.sw0 + ky * params.sw1 +
                                        oc * params.sw2 + ic * params.sw3]);
                    }
                    tile_w[t] = w;
                    let col = column_base + t % 16u;
                    let in_channel = channel + t / 16u;
                    let ix = i32(col % width) - i32(kx / params.stride);
                    let iy = i32(col / width) - i32(ky / params.stride);
                    var value = 0.0;
                    if (in_channel < params.ic && ix >= 0 && iy >= 0 &&
                        ix < i32(params.iw) && iy < i32(params.ih)) {
                        // The F16-kernel GGML operation also rounds input values to F16.
                        value = f32(WEIGHT_TYPE(input[params.offset_i + u32(ix) * params.si0 +
                            u32(iy) * params.si1 + in_channel * params.si2 + batch * params.si3]));
                    }
                    tile_i[t] = value;
                }
                workgroupBarrier();
                for (var k = 0u; k < 16u; k++) {
                    let w0 = tile_w[row * 16u + k];
                    let w1 = tile_w[(row + 1u) * 16u + k];
                    let i0 = tile_i[k * 16u + column];
                    let i1 = tile_i[k * 16u + column + 1u];
                    acc += vec4<f32>(w0 * i0, w1 * i0, w0 * i1, w1 * i1);
                }
                workgroupBarrier();
            }
            total += acc;
        }
    }
    store_output(row_base + row, column_base + column, batch, phase, total.x);
    store_output(row_base + row + 1u, column_base + column, batch, phase, total.y);
    store_output(row_base + row, column_base + column + 1u, batch, phase, total.z);
    store_output(row_base + row + 1u, column_base + column + 1u, batch, phase, total.w);
}
