enable f16;

@group(0) @binding(0)
var<storage, read_write> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> output: array<OUTPUT_TYPE>;

struct Params {
    offset_i: u32, offset_o: u32,
    si0: u32, si1: u32, si2: u32, si3: u32,
    IW: u32, IH: u32, ID: u32, IC: u32,
    KW: u32, KH: u32, KD: u32,
    OW: u32, OH: u32, OD: u32,
    K: u32, total: u32,
    s0: u32, s1: u32, s2: u32,
    p0: i32, p1: i32, p2: i32,
    d0: u32, d1: u32, d2: u32,
}

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    let index = gid.x + num_wg.x * u32(WG_SIZE) * gid.y;
    if (index >= params.total) {
        return;
    }

    // The contiguous output is [IC*KD*KH*KW, OW, OH, N*OD].
    var k = index % params.K;
    let kw = k % params.KW;
    k /= params.KW;
    let kh = k % params.KH;
    k /= params.KH;
    let kd = k % params.KD;
    let ic = k / params.KD;

    var position = index / params.K;
    let ow = position % params.OW;
    position /= params.OW;
    let oh = position % params.OH;
    position /= params.OH;
    let od = position % params.OD;
    let n = position / params.OD;

    let iw = i32(ow * params.s0 + kw * params.d0) - params.p0;
    let ih = i32(oh * params.s1 + kh * params.d1) - params.p1;
    let id = i32(od * params.s2 + kd * params.d2) - params.p2;
    var value = 0.0;
    if (iw >= 0 && iw < i32(params.IW) &&
        ih >= 0 && ih < i32(params.IH) &&
        id >= 0 && id < i32(params.ID)) {
        let source = params.offset_i + u32(iw) * params.si0 + u32(ih) * params.si1 +
                     u32(id) * params.si2 + (n * params.IC + ic) * params.si3;
        value = input[source];
    }
    output[params.offset_o + index] = OUTPUT_TYPE(value);
}
