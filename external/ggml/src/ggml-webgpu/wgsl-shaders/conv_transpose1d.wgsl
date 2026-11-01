@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    weight_offset: u32,
    input_offset: u32,
    output_offset: u32,
    weight_stride0: u32,
    weight_stride1: u32,
    weight_stride2: u32,
    input_stride0: u32,
    input_stride1: u32,
    input_length: u32,
    input_channels: u32,
    output_length: u32,
    output_channels: u32,
    kernel: u32,
    stride: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
    let index = gid.x + groups.x * WG_SIZE * gid.y;
    if (index >= params.output_length * params.output_channels) { return; }
    let position = index % params.output_length;
    let channel = index / params.output_length;
    var value = 0.0;
    // Increasing input positions match GGML's accumulation of overlapping taps.
    for (var tap = i32(params.kernel) - 1; tap >= 0; tap--) {
        let shifted = i32(position) - tap;
        if (shifted < 0 || u32(shifted) % params.stride != 0u) { continue; }
        let source_position = u32(shifted) / params.stride;
        if (source_position >= params.input_length) { continue; }
        var dot = 0.0;
        for (var c = 0u; c < params.input_channels; c++) {
            let wi = params.weight_offset + u32(tap) * params.weight_stride0 +
                     channel * params.weight_stride1 + c * params.weight_stride2;
            let xi = params.input_offset + source_position * params.input_stride0 + c * params.input_stride1;
            dot += weights[wi] * input[xi];
        }
        value += dot;
    }
    output[params.output_offset + index] = value;
}
